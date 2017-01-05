
#include "haraka_cuda.h"
#include "helper.h"
#include "constants.h"

#include <stdio.h>
#include <random>
#include <algorithm>
#include <functional>
#include <fstream>
#include <Windows.h>

using namespace std;

const uint32_t NUM_MESSAGES = 4194304;
const uint32_t NUM_MEASURMENTS = 10;

//#define TEST_PERFORMANCE_512
//#define TEST_FUNCTIONALITY_512

//#define TEST_PERFORMANCE_256
//#define TEST_FUNCTIONALITY_256

#define TEST_OTS


int cmp_f(const void *x, const void *y)
{
	float xx = *(float*)x, yy = *(float*)y;
	if (xx < yy) return -1;
	if (xx > yy) return  1;
	return 0;
}

void testPerformance512()
{
	float sum = 0;
	float times[NUM_MEASURMENTS];

	uint64_t t_begin;
	uint64_t t_end;
	uint64_t freq;

	// First create an instance of an engine.
	random_device rnd_device;
	// Specify the engine and distribution.
	mt19937 mersenne_engine(rnd_device());
	uniform_int_distribution<int> dist(CHAR_MIN, CHAR_MAX);

	auto gen = bind(dist, mersenne_engine);

	for (int j = 0; j < NUM_MEASURMENTS; ++j)
	{
		char* input;
		cudaMallocHost((void**)&input, MSG_SIZE_BYTE_512 * NUM_MESSAGES);

		char* hash;
		cudaMallocHost((void**)&hash, HASH_SIZE_BYTE * NUM_MESSAGES);

		generate(input, input + MSG_SIZE_BYTE_512 * NUM_MESSAGES - 1, gen);

		QueryPerformanceCounter((LARGE_INTEGER *)&t_begin);

		harakaCuda512(input, hash, NUM_MESSAGES);

		QueryPerformanceCounter((LARGE_INTEGER *)&t_end);
		QueryPerformanceFrequency((LARGE_INTEGER *)&freq);

		times[j] = ((t_end - t_begin) * 1000.0f / freq);

		if (j > 1)
			sum += times[j];

		cout << times[j] << endl;

		cudaFreeHost(input);
		cudaFreeHost(hash);
	}

	qsort(times, NUM_MEASURMENTS, sizeof(float), cmp_f);
	cout << "median: " << times[NUM_MEASURMENTS / 2] << " ms\n   mid: " << sum / (NUM_MEASURMENTS - 2) << " ms" << endl;
}

const int testFunctionality512()
{
	char* input = new char[MSG_SIZE_BYTE_512 * NUM_MESSAGES];
	char* hash = new char[HASH_SIZE_BYTE * NUM_MESSAGES];
	char* hash_ref = new char[HASH_SIZE_BYTE * NUM_MESSAGES];

	// First create an instance of an engine.
	random_device rnd_device;
	// Specify the engine and distribution.
	mt19937 mersenne_engine(rnd_device());
	uniform_int_distribution<int> dist(CHAR_MIN, CHAR_MAX);

	auto gen = bind(dist, mersenne_engine);
	generate(input, input + MSG_SIZE_BYTE_512 * NUM_MESSAGES - 1, gen);

	cudaError_t cuda_status = harakaCuda512(input, hash, NUM_MESSAGES);

	if (cuda_status != cudaSuccess)
		return ERROR_CUDA;

	for (int i = 0; i < NUM_MESSAGES; ++i)
		haraka512256(&hash_ref[HASH_SIZE_BYTE * i], &input[MSG_SIZE_BYTE_512 * i]);
	
	int status = memcmp(hash, hash_ref, HASH_SIZE_BYTE * NUM_MESSAGES) == 0;

	delete input;
	delete hash;
	delete hash_ref;

	return status;
}

void testPerformance256()
{
	float sum = 0;
	float times[NUM_MEASURMENTS];

	uint64_t t_begin;
	uint64_t t_end;
	uint64_t freq;

	// First create an instance of an engine.
	random_device rnd_device;
	// Specify the engine and distribution.
	mt19937 mersenne_engine(rnd_device());
	uniform_int_distribution<int> dist(CHAR_MIN, CHAR_MAX);

	auto gen = bind(dist, mersenne_engine);

	for (int j = 0; j < NUM_MEASURMENTS; ++j)
	{
		char* input;
		cudaMallocHost((void**)&input, MSG_SIZE_BYTE_256 * NUM_MESSAGES);

		char* hash;
		cudaMallocHost((void**)&hash, HASH_SIZE_BYTE * NUM_MESSAGES);

		generate(input, input + MSG_SIZE_BYTE_256 * NUM_MESSAGES - 1, gen);

		QueryPerformanceCounter((LARGE_INTEGER *)&t_begin);

		harakaCuda256(input, hash, NUM_MESSAGES);

		QueryPerformanceCounter((LARGE_INTEGER *)&t_end);
		QueryPerformanceFrequency((LARGE_INTEGER *)&freq);

		times[j] = ((t_end - t_begin) * 1000.0f / freq);

		if (j > 1)
			sum += times[j];

		cout << times[j] << endl;

		cudaFreeHost(input);
		cudaFreeHost(hash);
	}

	qsort(times, NUM_MEASURMENTS, sizeof(float), cmp_f);
	cout << "median: " << times[NUM_MEASURMENTS / 2] << " ms\n   mid: " << sum / (NUM_MEASURMENTS - 2) << " ms" << endl;
}

const int testFunctionality256()
{
	char* input = new char[MSG_SIZE_BYTE_256 * NUM_MESSAGES];
	char* hash = new char[HASH_SIZE_BYTE * NUM_MESSAGES];
	char* hash_ref = new char[HASH_SIZE_BYTE * NUM_MESSAGES];

	// First create an instance of an engine.
	random_device rnd_device;
	// Specify the engine and distribution.
	mt19937 mersenne_engine(rnd_device());
	uniform_int_distribution<int> dist(CHAR_MIN, CHAR_MAX);

	auto gen = bind(dist, mersenne_engine);
	generate(input, input + MSG_SIZE_BYTE_256 * NUM_MESSAGES - 1, gen);
	

	cudaError_t cuda_status = harakaCuda256(input, hash, NUM_MESSAGES);

	if (cuda_status != cudaSuccess)
		return ERROR_CUDA;

	for (int i = 0; i < NUM_MESSAGES; ++i)
		haraka256256(&hash_ref[HASH_SIZE_BYTE * i], &input[MSG_SIZE_BYTE_256 * i]);

	int status = memcmp(hash, hash_ref, HASH_SIZE_BYTE * NUM_MESSAGES) == 0;

	delete input;
	delete hash;
	delete hash_ref;

	return status;
}

int testOTS()
{
	char* input = new char[MSG_SIZE_BYTE_256 * NUM_MESSAGES];
	char* signatures = new char[HASH_SIZE_BYTE * T * NUM_MESSAGES];
	char* pub_keys = new char[HASH_SIZE_BYTE * T * NUM_MESSAGES];

	// First create an instance of an engine.
	random_device rnd_device;
	// Specify the engine and distribution.
	mt19937 mersenne_engine(rnd_device());
	uniform_int_distribution<int> dist(CHAR_MIN, CHAR_MAX);

	auto gen = bind(dist, mersenne_engine);
	generate(input, input + MSG_SIZE_BYTE_256 * NUM_MESSAGES - 1, gen);


	int status = harakaWinternitzCudaSign(input, signatures, pub_keys, NUM_MESSAGES);

	if (status)
	{
		delete input;
		delete signatures;
		delete pub_keys;

		return status;
	}

	status = harakaWinternitzCudaVerify(input, signatures, pub_keys, NUM_MESSAGES);
	
	delete input;
	delete signatures;
	delete pub_keys;

	return status;
}


int main()
{

#ifdef TEST_PERFORMANCE_512
	testPerformance512();
#endif

	int status;
#ifdef TEST_FUNCTIONALITY_512
	status = testFunctionality512();

	if (status == TEST_SUCCESS)
		cout << TEST_SUCCESS_STRING << endl;
	else if (status == ERROR_HASH)
		cout << ERROR_HASH_MISSMATCH << endl;
	else
		cout << ERROR_CUDA_STRING << endl;
#endif

#ifdef TEST_PERFORMANCE_256
	testPerformance256();
#endif

#ifdef TEST_FUNCTIONALITY_256
	status = testFunctionality256();

	if (status == TEST_SUCCESS)
		cout << TEST_SUCCESS_STRING << endl;
	else if (status == ERROR_HASH)
		cout << ERROR_HASH_MISSMATCH << endl;
	else
		cout << ERROR_CUDA_STRING << endl;
#endif

#ifdef TEST_OTS
	status = testOTS();

	if (status == SUCCESS)
		cout << OTS_SUCCESS << endl;
	else if (status == FAILED_TO_ACQUIRE_CRYPT_PROV)
		cout << FAILED_TO_ACQUIRE_CRYPT_PROV_STRING << endl;
	else if (status == FAILED_TO_GENERATE_CRYPT_RAND_BYTES)
		cout << FAILED_TO_GENERATE_CRYPT_RAND_BYTES_STRING << endl;
#endif

	// cudaDeviceReset must be called before exiting in order for profiling and
	// tracing tools such as Nsight and Visual Profiler to show complete traces.
	cudaError_t cudaStatus = cudaDeviceReset();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaDeviceReset failed!");
		while (1) {}
		return 1;
	}


	while (1) {}

	return 0;
}