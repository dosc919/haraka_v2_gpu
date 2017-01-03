
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
//#define TEST_PERFORMANCE_265
#define TEST_FUNCTIONALITY_256
//#define TEST_OTS


int cmp_f(const void *x, const void *y)
{
	float xx = *(float*)x, yy = *(float*)y;
	if (xx < yy) return -1;
	if (xx > yy) return  1;
	return 0;
}

void testOTS()
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

	vector<char> digest_ref(HASH_SIZE_BYTE * NUM_MESSAGES);

	for (int j = 0; j < NUM_MEASURMENTS; ++j)
	{
		char* input;
		cudaMallocHost((void**)&input, MSG_SIZE_BYTE_512 * NUM_MESSAGES);
		generate(input, input + MSG_SIZE_BYTE_512 * NUM_MESSAGES - 1, gen);

		char* digest;
		cudaMallocHost((void**)&digest, HASH_SIZE_BYTE * T * NUM_MESSAGES);

		QueryPerformanceCounter((LARGE_INTEGER *)&t_begin);

		harakaWinternitzCuda(input, digest, NUM_MESSAGES);

		QueryPerformanceCounter((LARGE_INTEGER *)&t_end);
		QueryPerformanceFrequency((LARGE_INTEGER *)&freq);

		for (int i = 0; i < NUM_MESSAGES; ++i)
			haraka512256(&digest_ref[HASH_SIZE_BYTE * i], &input[MSG_SIZE_BYTE_512 * i]);

		//---------------------------------------------------------------------------------
		uint32_t* b = new uint32_t[T * NUM_MESSAGES];
		uint64_t* c = new uint64_t[NUM_MESSAGES];
		for (int i = 2; i < T; ++i)
		{
			for (int j = 0; j < NUM_MESSAGES; ++j)
			{
				b[i + j * T] = reinterpret_cast<uint32_t*>(&digest_ref[0])[i - 2 + T1 * j];
				c[j] += UINT32_MAX - (b[i + j * T] + 1);
			}
		}

		for (int i = 2; i < 6; ++i)
		{
			b[i] = reinterpret_cast<uint32_t*>(&digest_ref[0])[i - 2];
			c[0] += UINT32_MAX - (b[i] + 1);
		}

		for (int i = 0; i < T2; ++i)
		{
			for(int j = 0; j < NUM_MESSAGES; ++j)
			b[i + j * T] = (c[j] >> (i * 32));
		}

		vector<uint64_t> v(2 * T * NUM_MESSAGES);
		vector<uint64_t> y(2 * 6 * NUM_MESSAGES);
		for (int i = 0; i < NUM_MESSAGES; ++i)
		{
			v[12 * i] = reinterpret_cast<uint64_t*>(digest)[24 * i] +(UINT32_MAX - b[0]);
			v[12 * i + 1] = reinterpret_cast<uint64_t*>(digest)[24 * i + 1] + (v[12 * i] < (UINT32_MAX - b[0]));
			v[12 * i + 2] = reinterpret_cast<uint64_t*>(digest)[24 * i + 2] + UINT32_MAX - b[1];
			v[12 * i + 3] = reinterpret_cast<uint64_t*>(digest)[24 * i + 3] + (v[12 * i + 2] < (UINT32_MAX - b[1]));
			v[12 * i + 4] = reinterpret_cast<uint64_t*>(digest)[24 * i + 4] + UINT32_MAX - b[2];
			v[12 * i + 5] = reinterpret_cast<uint64_t*>(digest)[24 * i + 5] + (v[12 * i + 4] < (UINT32_MAX - b[2]));
			v[12 * i + 6] = reinterpret_cast<uint64_t*>(digest)[24 * i + 6] + UINT32_MAX - b[3];
			v[12 * i + 7] = reinterpret_cast<uint64_t*>(digest)[24 * i + 7] + (v[12 * i + 6] < (UINT32_MAX - b[3]));
			v[12 * i + 8] = reinterpret_cast<uint64_t*>(digest)[24 * i + 8] + UINT32_MAX - b[4];
			v[12 * i + 9] = reinterpret_cast<uint64_t*>(digest)[24 * i + 9] + (v[12 * i + 8] < (UINT32_MAX - b[4]));
			v[12 * i + 10] = reinterpret_cast<uint64_t*>(digest)[24 * i + 10] + UINT32_MAX - b[5];
			v[12 * i + 11] = reinterpret_cast<uint64_t*>(digest)[24 * i + 11] + (v[12 * i + 10] < (UINT32_MAX - b[5]));


			y[12 * i] = reinterpret_cast<uint64_t*>(digest)[24 * i + 12];
			y[12 * i + 1] = reinterpret_cast<uint64_t*>(digest)[24 * i + 13];
			y[12 * i + 2] = reinterpret_cast<uint64_t*>(digest)[24 * i + 14];
			y[12 * i + 3] = reinterpret_cast<uint64_t*>(digest)[24 * i + 15];
			y[12 * i + 4] = reinterpret_cast<uint64_t*>(digest)[24 * i + 16];
			y[12 * i + 5] = reinterpret_cast<uint64_t*>(digest)[24 * i + 17];
			y[12 * i + 6] = reinterpret_cast<uint64_t*>(digest)[24 * i + 18];
			y[12 * i + 7] = reinterpret_cast<uint64_t*>(digest)[24 * i + 19];
			y[12 * i + 8] = reinterpret_cast<uint64_t*>(digest)[24 * i + 20];
			y[12 * i + 9] = reinterpret_cast<uint64_t*>(digest)[24 * i + 21];
			y[12 * i + 10] = reinterpret_cast<uint64_t*>(digest)[24 * i + 22];
			y[12 * i + 11] = reinterpret_cast<uint64_t*>(digest)[24 * i + 23];
		}

		delete b;
		delete c;
		//---------------------------------------------------------------------------------
		for (int i = 0; i < 12 * 2; ++i)
			cout << y[i] << " | " << v[i] << " | " << memcmp((void*)&v[i], (void*)&y[i], 8) << endl;
		cout << memcmp((void*)&v[0], (void*)&y[0], T * NUM_MESSAGES * 8 * 2) << endl;


		times[j] = ((t_end - t_begin) * 1000.0 / freq);

		if(j > 1)
			sum += times[j];

		cout << times[j] << endl;

		cudaFreeHost(input);
		cudaFreeHost(digest);
	}



	qsort(times, NUM_MEASURMENTS, sizeof(float), cmp_f);
	cout << "median: " << times[NUM_MEASURMENTS/2] << " ms\n   mid: " << sum / (NUM_MEASURMENTS-2) << " ms" << endl;
	while (1);
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

		times[j] = ((t_end - t_begin) * 1000.0 / freq);

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
	testOTS();
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