#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <future>
#include <iostream>
#include <random>
#include <string>
#include <vector>
#include <omp.h>
#include "../cblas.h"
#include "cpp_thread_safety_common.h"

struct DgemmVariant {
	const char *name;
	CBLAS_TRANSPOSE transA;
	CBLAS_TRANSPOSE transB;
};

static const DgemmVariant dgemmVariants[] = {
	{"NN", CblasNoTrans, CblasNoTrans},
	{"NT", CblasNoTrans, CblasTrans},
	{"TN", CblasTrans, CblasNoTrans},
	{"TT", CblasTrans, CblasTrans},
};

static const uint32_t numDgemmVariants = sizeof(dgemmVariants) / sizeof(dgemmVariants[0]);

void launch_cblas_dgemm_variant(const DgemmVariant variant, double* A, double* B, double* C, const blasint randomMatSize){
	cblas_dgemm(CblasColMajor, variant.transA, variant.transB, randomMatSize, randomMatSize, randomMatSize, 1.0, A, randomMatSize, B, randomMatSize, 0.1, C, randomMatSize);
}

int main(int argc, char* argv[]){
	blasint randomMatSize = 512;
	uint32_t numConcurrentThreads = 32;
	uint32_t numTestRounds = 8;
	uint32_t maxHwThreads = omp_get_max_threads();
	const double tolerance = 1.0E-10;

	if (maxHwThreads < numConcurrentThreads)
		numConcurrentThreads = maxHwThreads;

	if (argc > 4){
		std::cout<<"ERROR: too many arguments for mixed DGEMM thread safety tester"<<std::endl;
		abort();
	}

	if(argc == 4){
		std::vector<std::string> cliArgs;
		for (int i = 1; i < argc; i++){
			cliArgs.push_back(argv[i]);
			std::cout<<argv[i]<<std::endl;
		}
		randomMatSize = std::stoul(cliArgs[0]);
		numConcurrentThreads = std::stoul(cliArgs[1]);
		numTestRounds = std::stoul(cliArgs[2]);
	}

	FailIfThreadsAreZero(numConcurrentThreads);

	const size_t matrixElements = static_cast<size_t>(randomMatSize) * static_cast<size_t>(randomMatSize);

	std::uniform_real_distribution<double> rngdist{-1.0, 1.0};
	std::vector<std::vector<double>> matBlock(numConcurrentThreads * 3);
	std::vector<std::vector<double>> baseBlock(3);
	std::vector<std::vector<double>> referenceBlock(numDgemmVariants);
	std::vector<std::future<void>> futureBlock(numConcurrentThreads);

	std::cout<<"*----------------------------------*\n";
	std::cout<<"| Mixed DGEMM thread safety tester |\n";
	std::cout<<"*----------------------------------*\n";
	std::cout<<"Size of random matrices(N=M=K): "<<randomMatSize<<'\n';
	std::cout<<"Number of concurrent calls into OpenBLAS : "<<numConcurrentThreads<<'\n';
	std::cout<<"Number of testing rounds : "<<numTestRounds<<'\n';
	std::cout<<"This test will need "<<(static_cast<uint64_t>(matrixElements) * numConcurrentThreads * 3 * 8)/static_cast<double>(1024*1024)<<" MiB of RAM\n"<<std::endl;

	std::cout<<"Initializing random number generator..."<<std::flush;
	std::mt19937_64 PRNG = InitPRNG();
	std::cout<<"done\n";

	std::cout<<"Preparing to test mixed CBLAS DGEMM thread safety\n";
	std::cout<<"Allocating matrices..."<<std::flush;
	for(uint32_t i=0; i<numConcurrentThreads*3; i++){
		matBlock[i].resize(matrixElements);
	}
	for(uint32_t i=0; i<3; i++){
		baseBlock[i].resize(matrixElements);
	}
	std::cout<<"done\n";

	std::cout<<"Filling matrices with random numbers..."<<std::flush;
	for(uint32_t i=0; i<3; i++){
		for(size_t j=0; j<matrixElements; j++){
			baseBlock[i][j] = rngdist(PRNG);
		}
	}
	std::cout<<"done\n";

	std::cout<<"Computing reference results..."<<std::flush;
	for(uint32_t variant=0; variant<numDgemmVariants; variant++){
		referenceBlock[variant] = baseBlock[2];
		launch_cblas_dgemm_variant(dgemmVariants[variant], &baseBlock[0][0], &baseBlock[1][0], &referenceBlock[variant][0], randomMatSize);
	}
	std::cout<<"done\n";

	std::cout<<"Testing mixed CBLAS DGEMM thread safety\n";
	omp_set_num_threads(numConcurrentThreads);
	const DgemmVariant *variants = dgemmVariants;
	const uint32_t variantCount = numDgemmVariants;
	for(uint32_t R=0; R<numTestRounds; R++){
		std::cout<<"Mixed DGEMM round #"<<R<<std::endl;

		for(uint32_t i=0; i<numConcurrentThreads; i++){
			matBlock[i*3] = baseBlock[0];
			matBlock[i*3+1] = baseBlock[1];
			matBlock[i*3+2] = baseBlock[2];
		}

		std::cout<<"Launching "<<numConcurrentThreads<<" threads simultaneously using OpenMP..."<<std::flush;
		#pragma omp parallel for default(none) shared(futureBlock, matBlock, randomMatSize, numConcurrentThreads, variants, variantCount)
		for(uint32_t i=0; i<numConcurrentThreads; i++){
			const DgemmVariant variant = variants[i % variantCount];
			futureBlock[i] = std::async(std::launch::async, launch_cblas_dgemm_variant, variant, &matBlock[i*3][0], &matBlock[i*3+1][0], &matBlock[i*3+2][0], randomMatSize);
		}
		std::cout<<"done\n";

		std::cout<<"Waiting for threads to finish..."<<std::flush;
		for(uint32_t i=0; i<numConcurrentThreads; i++){
			futureBlock[i].get();
		}
		std::cout<<"done\n";

		std::cout<<"Comparing results from different DGEMM variants..."<<std::flush;
		for(uint32_t i=0; i<numConcurrentThreads; i++){
			const uint32_t variant = i % numDgemmVariants;
			for(size_t j=0; j<matrixElements; j++){
				if (std::abs(matBlock[i*3+2][j] - referenceBlock[variant][j]) > tolerance){
					std::cout<<"ERROR: "<<dgemmVariants[variant].name<<" returned a different result! Thread index : "<<i<<", matrix index : "<<j<<std::endl;
					std::cout<<"Mixed CBLAS DGEMM thread safety test FAILED!"<<std::endl;
					return -1;
				}
			}
		}
		std::cout<<"OK!\n"<<std::endl;
	}

	std::cout<<"Mixed CBLAS DGEMM thread safety test PASSED!\n"<<std::endl;
	return 0;
}
