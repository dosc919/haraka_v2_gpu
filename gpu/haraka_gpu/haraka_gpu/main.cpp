
#include "haraka_cuda.h"
#include "helper.h"
#include "constants.h"

#include <stdio.h>
#include <random>
#include <algorithm>
#include <functional>

using namespace std;

int main()
{
	const int arraySize = 5;
	const int a[arraySize] = { 1, 2, 3, 4, 5 };
	const int b[arraySize] = { 10, 20, 30, 40, 50 };
	int c[arraySize] = { 0 };

	// Add vectors in parallel.
	cudaError_t cudaStatus = addWithCuda(c, a, b, arraySize);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "addWithCuda failed!");
		return 1;
	}

	printf("{1,2,3,4,5} + {10,20,30,40,50} = {%d,%d,%d,%d,%d}\n",
		c[0], c[1], c[2], c[3], c[4]);

	// cudaDeviceReset must be called before exiting in order for profiling and
	// tracing tools such as Nsight and Visual Profiler to show complete traces.
	cudaStatus = cudaDeviceReset();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaDeviceReset failed!");
		return 1;
	}

	


	// First create an instance of an engine.
	random_device rnd_device;
	// Specify the engine and distribution.
	mt19937 mersenne_engine(rnd_device());
	uniform_int_distribution<unsigned int> dist(0, UCHAR_MAX);

	auto gen = bind(dist, mersenne_engine);
	vector<unsigned char> input(64);
	generate(begin(input), end(input), gen);

	printVector(INPUT_TEXT, input);

	vector<unsigned char> digest(32);
	cudaError_t error = harakaCuda(input, digest);

	printVector(OUTPUT_TEXT, digest);

	while (1){}
	return 0;
}