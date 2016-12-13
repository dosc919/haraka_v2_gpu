
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

//#define TEST_PERFORMANCE
//#define TEST_FUNCTIONALITY_RANDOM
//#define TEST_FUNCTIONALITY
#define TEST_OTS


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

	vector<char> digest_ref(DIGEST_SIZE_BYTE * NUM_MESSAGES);

	for (int j = 0; j < NUM_MEASURMENTS; ++j)
	{
		char* input;
		cudaMallocHost((void**)&input, INPUT_SIZE_BYTE * NUM_MESSAGES);
		generate(input, input + INPUT_SIZE_BYTE * NUM_MESSAGES - 1, gen);

		char* digest;
		cudaMallocHost((void**)&digest, INPUT_SIZE_BYTE * NUM_MESSAGES);

		QueryPerformanceCounter((LARGE_INTEGER *)&t_begin);

		harakaWinternitzCuda(input, digest, NUM_MESSAGES);

		QueryPerformanceCounter((LARGE_INTEGER *)&t_end);
		QueryPerformanceFrequency((LARGE_INTEGER *)&freq);

		times[j] = ((t_end - t_begin) * 1000.0 / freq);

		if(j > 1)
			sum += times[j];

		cudaFreeHost(input);
		cudaFreeHost(digest);
	}



	qsort(times, NUM_MEASURMENTS, sizeof(float), cmp_f);
	cout << "median: " << times[NUM_MEASURMENTS/2] << " ms\n   mid: " << sum / (NUM_MEASURMENTS-2) << " ms" << endl;
	while (1);
}

int testPerformance()
{
	// First create an instance of an engine.
	random_device rnd_device;
	// Specify the engine and distribution.
	mt19937 mersenne_engine(rnd_device());
	uniform_int_distribution<int> dist(CHAR_MIN, CHAR_MAX);

	auto gen = bind(dist, mersenne_engine);
	vector<char> input(INPUT_SIZE_BYTE * NUM_MESSAGES);
	generate(begin(input), end(input), gen);

	vector<char> digest(DIGEST_SIZE_BYTE * NUM_MESSAGES);
	cudaError_t cuda_status = harakaCuda(input, digest);

	//for (int i = 0; i < NUM_MESSAGES; ++i)
	//{
	//	printVector(INPUT_TEXT, vector<char>(&input[i * 64], &input[i * 64 + 63] + 1));
	//	printVector(OUTPUT_TEXT, vector<char>(&digest[i * 32], &digest[i * 32 + 31] + 1));
	//}

	return cuda_status;
}

const vector<string> testFunctionalityRandom()
{
	vector<string> errors;
	vector<char> input(INPUT_SIZE_BYTE * NUM_MESSAGES);
	vector<char> digest(DIGEST_SIZE_BYTE * NUM_MESSAGES);
	vector<char> digest_ref(DIGEST_SIZE_BYTE * NUM_MESSAGES);

	// First create an instance of an engine.
	random_device rnd_device;
	// Specify the engine and distribution.
	mt19937 mersenne_engine(rnd_device());
	uniform_int_distribution<int> dist(CHAR_MIN, CHAR_MAX);

	auto gen = bind(dist, mersenne_engine);
	generate(begin(input), end(input), gen);

	cudaError_t cuda_status = harakaCuda(input, digest);

	if (cuda_status != cudaSuccess)
		errors.push_back(ERROR_CUDA);

	for (int i = 0; i < NUM_MESSAGES; ++i)
		haraka512256(&digest_ref[DIGEST_SIZE_BYTE * i], &input[INPUT_SIZE_BYTE * i]);

	if (memcmp(&digest[0], &digest_ref[0], DIGEST_SIZE_BYTE * NUM_MESSAGES))
		errors.push_back(ERROR_DIGEST_MISSMATCH);
	
	return errors;
}

const vector<string> testFunctionality()
{
	vector<string> errors;
	ifstream file;
	for (auto file_name : TEST_FILES)
	{
		file.open(file_name, ios::in | ios::binary);
		if (!file.good())
		{
			errors.push_back(file_name + ERROR_FILE_OPEN);
			continue;
		}

		vector<char> input(INPUT_SIZE_BYTE);
		file.read(&input[0], INPUT_SIZE_BYTE);
		if (!file.good())
		{
			file.close();
			errors.push_back(file_name + ERROR_FILE_READ_INPUT);
			continue;
		}

		vector<char> digest_ref(DIGEST_SIZE_BYTE);
		file.read(&digest_ref[0], DIGEST_SIZE_BYTE);
		if (!file.good())
		{
			file.close();
			errors.push_back(file_name + ERROR_FILE_READ_DIGEST);
			continue;
		}

		printVector(INPUT_TEXT, input);

		vector<char> digest(DIGEST_SIZE_BYTE);
		cudaError_t cuda_status = harakaCuda(input, digest);

		printVector(OUTPUT_TEXT, digest);
		printVector(OUTPUT_REFERENCE_TEXT, digest_ref);

		if (cuda_status != cudaSuccess)
			errors.push_back(file_name + ERROR_CUDA);

		if (memcmp(&digest[0], &digest_ref[0], DIGEST_SIZE_BYTE))
			errors.push_back(file_name + ERROR_DIGEST_MISSMATCH);

		file.close();
	}

	return errors;
}

int main()
{

#ifdef TEST_PERFORMANCE
	testPerformance();
#endif

	vector<string> error_msgs;
#ifdef TEST_FUNCTIONALITY
	error_msgs = testFunctionality();

	if (error_msgs.empty())
	{
		cout << "All " << TEST_FILES.size() << " tests succeeded." << endl;
	}
	else
	{
		cout << endl << error_msgs.size() << " out of " << TEST_FILES.size() << " tests failed." << endl;
		for (auto msg : error_msgs)
			cout << msg << endl;
	}
#endif

#ifdef TEST_FUNCTIONALITY_RANDOM
	error_msgs = testFunctionalityRandom();

	if (error_msgs.empty())
	{
		cout << "All " << TEST_FILES.size() << " tests succeeded." << endl;
	}
	else
	{
		cout << endl << error_msgs.size() << " out of " << TEST_FILES.size() << " tests failed." << endl;
		for (auto msg : error_msgs)
			cout << msg << endl;
	}
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

#ifdef TEST_FUNCTIONALITY
	while (1) {}
#endif
#ifdef TEST_FUNCTIONALITY_RANDOM
	while (1) {}
#endif

	return 0;
}