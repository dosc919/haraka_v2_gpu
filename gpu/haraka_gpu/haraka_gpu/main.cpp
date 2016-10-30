
#include "haraka_cuda.h"
#include "helper.h"
#include "constants.h"

#include <stdio.h>
#include <random>
#include <algorithm>
#include <functional>
#include <fstream>

using namespace std;

#define TEST_PERFORMANCE
#define NUM_MESSAGES 4194304

//#define TEST_FUNCTIONALITY

int testPerformance()
{
	// First create an instance of an engine.
	random_device rnd_device;
	// Specify the engine and distribution.
	mt19937 mersenne_engine(rnd_device());
	uniform_int_distribution<int> dist(CHAR_MIN, CHAR_MAX);

	auto gen = bind(dist, mersenne_engine);
	vector<char> input(64 * NUM_MESSAGES);
	generate(begin(input), end(input), gen);

	vector<char> digest(32 * NUM_MESSAGES);
	cudaError_t cuda_status = harakaCuda(input, digest);

	//for (int i = 0; i < NUM_MESSAGES; ++i)
	//{
	//	printVector(INPUT_TEXT, vector<char>(&input[i * 64], &input[i * 64 + 63] + 1));
	//	printVector(OUTPUT_TEXT, vector<char>(&digest[i * 32], &digest[i * 32 + 31] + 1));
	//}

	return cuda_status != cudaSuccess;
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
		printVector(OUTPUT_TEXT + "ref", digest_ref);

		if (cuda_status != cudaSuccess)
			errors.push_back(file_name +  ERROR_CUDA);
		
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

#ifdef TEST_FUNCTIONALITY
	const vector<string> error_msgs = testFunctionality();

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

	// cudaDeviceReset must be called before exiting in order for profiling and
	// tracing tools such as Nsight and Visual Profiler to show complete traces.
	cudaError_t cudaStatus = cudaDeviceReset();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaDeviceReset failed!");
		while (1){}
		return 1;
	}

#ifdef TEST_FUNCTIONALITY
	while (1){}
#endif
	//ofstream test("test_1.bin", std::ios::out| ios::binary);
	//if (!test.good())
	//{
	//	cout << "open error" << endl;
	//	while (1){}
	//}

	//char test_bytes[INPUT_SIZE_BYTE + DIGEST_SIZE_BYTE] = {0x00, 0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07, 0x08, 0x09, 0x0a, 0x0b, 0x0c, 0x0d, 0x0e, 0x0f,
	//													   0x10, 0x11, 0x12, 0x13, 0x14, 0x15, 0x16, 0x17, 0x18, 0x19, 0x1a, 0x1b, 0x1c, 0x1d, 0x1e, 0x1f, 
	//													   0x20, 0x21, 0x22, 0x23, 0x24, 0x25, 0x26, 0x27, 0x28, 0x29, 0x2a, 0x2b, 0x2c, 0x2d, 0x2e, 0x2f, 
	//													   0x30, 0x31, 0x32, 0x33, 0x34, 0x35, 0x36, 0x37, 0x38, 0x39, 0x3a, 0x3b, 0x3c, 0x3d, 0x3e, 0x3f,
	//													   0xbe, 0x7f, 0x72, 0x3b, 0x4e, 0x80, 0xa9, 0x98, 0x13, 0xb2, 0x92, 0x28, 0x7f, 0x30, 0x6f, 0x62,
	//													   0x5a, 0x6d, 0x57, 0x33, 0x1c, 0xae, 0x5f, 0x34, 0xdd, 0x92, 0x77, 0xb0, 0x94, 0x5b, 0xe2, 0xaa};

	//test.write(test_bytes, INPUT_SIZE_BYTE + DIGEST_SIZE_BYTE);

	//if (test.bad() || test.fail())
	//	cout << "write error" << endl;

	//test.close();

	return 0;
}