
#include "haraka_cuda_wrapper.h"
#include "helper.h"
#include "constants.h"

#include <stdio.h>
#include <random>
#include <algorithm>
#include <functional>
#include <Windows.h>
#include <wincrypt.h>

#include <intrin.h>

using namespace std;

const uint32_t NUM_MESSAGES = 200000;//200000;//4194304; //128MB input for haraka256 and 256MB input for haraka 512
const uint32_t NUM_MEASURMENTS = 1;
const uint32_t TREE_DEPTH = 18;

//#define TEST_PERFORMANCE_512
//#define TEST_FUNCTIONALITY_512

//#define TEST_PERFORMANCE_256
//#define TEST_FUNCTIONALITY_256

//#define TEST_OTS
//#define TEST_OTS_EXTERNAL_PRIVATE_KEY

#define TEST_MERKLE_TREE


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

	random_device rnd_device;
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

	random_device rnd_device;
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

	random_device rnd_device;
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

	random_device rnd_device;
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
	float sum = 0;
	uint64_t times[NUM_MEASURMENTS];

	uint64_t t_begin;
	uint64_t t_end;
	uint64_t freq;

	random_device rnd_device;
	mt19937 mersenne_engine(rnd_device());
	uniform_int_distribution<int> dist(CHAR_MIN, CHAR_MAX);

	auto gen = bind(dist, mersenne_engine);

	int status;

	int registers[4];
	unsigned int aux[32];

	for (int j = 0; j < NUM_MEASURMENTS; ++j)
	{
		char* input;
		cudaMallocHost((void**)&input, MSG_SIZE_BYTE_256 * NUM_MESSAGES);

		char* signatures;
		cudaMallocHost((void**)&signatures, HASH_SIZE_BYTE * T * NUM_MESSAGES);

		char* pub_keys;
		cudaMallocHost((void**)&pub_keys, HASH_SIZE_BYTE * T * NUM_MESSAGES);

		generate(input, input + MSG_SIZE_BYTE_256 * NUM_MESSAGES - 1, gen);

		//QueryPerformanceCounter((LARGE_INTEGER *)&t_begin);
		__cpuid(registers, 0);
		t_begin = __rdtsc();

		status = harakaWinternitzCudaSign(input, signatures, pub_keys, NUM_MESSAGES);

		t_end = __rdtscp(aux);
		__cpuid(registers, 0);


		//QueryPerformanceCounter((LARGE_INTEGER *)&t_end);
		//QueryPerformanceFrequency((LARGE_INTEGER *)&freq);

		if (status)
		{
			cudaFreeHost(input);
			cudaFreeHost(signatures);
			cudaFreeHost(pub_keys);

			return status;
		}

		times[j] = ((t_end - t_begin));// *1000.0f / freq);

		if (j > 1)
			sum += times[j];

		cout << times[j] << endl;

		status = harakaWinternitzCudaVerify(input, signatures, pub_keys, NUM_MESSAGES);

		cudaFreeHost(input);
		cudaFreeHost(signatures);
		cudaFreeHost(pub_keys);

		if (status)
			return status;
	}

	qsort(times, NUM_MEASURMENTS, sizeof(uint64_t), cmp_f);
	cout << "median: " << times[NUM_MEASURMENTS / 2] << " ms\n   mid: " << sum / (NUM_MEASURMENTS - 2) << " ms" << endl;

	return status;
}

int testOTSExternalPrivateKey()
{
	float sum = 0;
	float times[NUM_MEASURMENTS];

	uint64_t t_begin;
	uint64_t t_end;
	uint64_t freq;

	random_device rnd_device;
	mt19937 mersenne_engine(rnd_device());
	uniform_int_distribution<int> dist(CHAR_MIN, CHAR_MAX);

	HCRYPTPROV crypt_prov;
	if (!CryptAcquireContext(&crypt_prov, NULL, NULL, PROV_RSA_FULL, 0))
		return FAILED_TO_ACQUIRE_CRYPT_PROV;

	auto gen = bind(dist, mersenne_engine);

	int status;

	for (int j = 0; j < NUM_MEASURMENTS; ++j)
	{
		char* input;
		cudaMallocHost((void**)&input, MSG_SIZE_BYTE_256 * NUM_MESSAGES);

		char* signatures;
		cudaMallocHost((void**)&signatures, HASH_SIZE_BYTE * T * NUM_MESSAGES);

		char* pub_keys;
		cudaMallocHost((void**)&pub_keys, HASH_SIZE_BYTE * T * NUM_MESSAGES);

		char* priv_keys;
		cudaMallocHost((void**)&priv_keys, HASH_SIZE_BYTE * T * NUM_MESSAGES);

		if (!CryptGenRandom(crypt_prov, T * HASH_SIZE_BYTE * NUM_MESSAGES, (BYTE*)(priv_keys)))
			return FAILED_TO_GENERATE_CRYPT_RAND_BYTES;

		generate(input, input + MSG_SIZE_BYTE_256 * NUM_MESSAGES - 1, gen);

		QueryPerformanceCounter((LARGE_INTEGER *)&t_begin);

		status = harakaWinternitzCudaSign(input, priv_keys, signatures, pub_keys, NUM_MESSAGES);

		QueryPerformanceCounter((LARGE_INTEGER *)&t_end);
		QueryPerformanceFrequency((LARGE_INTEGER *)&freq);

		if (status)
		{
			cudaFreeHost(input);
			cudaFreeHost(signatures);
			cudaFreeHost(pub_keys);
			cudaFreeHost(priv_keys);

			return status;
		}

		times[j] = ((t_end - t_begin) * 1000.0f / freq);

		if (j > 1)
			sum += times[j];

		cout << times[j] << endl;

		status = harakaWinternitzCudaVerify(input, signatures, pub_keys, NUM_MESSAGES);

		cudaFreeHost(input);
		cudaFreeHost(signatures);
		cudaFreeHost(pub_keys);
		cudaFreeHost(priv_keys);

		if (status)
			return status;
	}

	qsort(times, NUM_MEASURMENTS, sizeof(float), cmp_f);
	cout << "median: " << times[NUM_MEASURMENTS / 2] << " ms\n   mid: " << sum / (NUM_MEASURMENTS - 2) << " ms" << endl;

	return status;
}


void testMerkleTree()
{
	float sum = 0;
	float times[NUM_MEASURMENTS];

	uint64_t t_begin;
	uint64_t t_end;
	uint64_t freq;

	random_device rnd_device;
	mt19937 mersenne_engine(rnd_device());
	uniform_int_distribution<int> dist(CHAR_MIN, CHAR_MAX);
	auto gen = bind(dist, mersenne_engine);

	for (int j = 0; j < NUM_MEASURMENTS; ++j)
	{
		char* tree;
		cudaMallocHost((void**)&tree, ((1 << (TREE_DEPTH + 1)) - 1) * HASH_SIZE_BYTE);

		//generate pseudo values for the leaves of the tree
		generate(tree + ((1 << TREE_DEPTH) - 1) * HASH_SIZE_BYTE, tree + ((1 << (TREE_DEPTH + 1)) - 1) * HASH_SIZE_BYTE - 1, gen);
		tree[((1 << TREE_DEPTH) - 1) * HASH_SIZE_BYTE] = j;

		char* tree_cpu = new char[((1 << (TREE_DEPTH + 1)) - 1) * HASH_SIZE_BYTE];
		memcpy(&tree_cpu[((1 << TREE_DEPTH) - 1) * HASH_SIZE_BYTE], tree + ((1 << TREE_DEPTH) - 1) * HASH_SIZE_BYTE, (1 << TREE_DEPTH) * HASH_SIZE_BYTE);
		for (int level = 0; level < TREE_DEPTH; ++level)
		{
			uint64_t parents = (1 << TREE_DEPTH - level - 1);
			loop_hash_2n_n(&tree_cpu[(parents - 1) * HASH_SIZE_BYTE], &tree_cpu[(2 * parents - 1) * HASH_SIZE_BYTE], parents);
		}
		printf("done\n");

		QueryPerformanceCounter((LARGE_INTEGER *)&t_begin);

		harakaBuildMerkleTree(tree, TREE_DEPTH);

		QueryPerformanceCounter((LARGE_INTEGER *)&t_end);
		QueryPerformanceFrequency((LARGE_INTEGER *)&freq);
		printf("done\n");
		times[j] = ((t_end - t_begin) * 1000.0f / freq);

		if (j > 1)
			sum += times[j];

		cout << times[j] << endl;


		/*for (int i = 0; i <= TREE_DEPTH; ++i)
		{
			int num_nodes = 1 << i;
			for (int j = 0; j < num_nodes; ++j)
			{
				for (int k = 0; k < HASH_SIZE_BYTE; ++k)
					printf("%02x", (unsigned char)tree[((1 << i) - 1 + j) * HASH_SIZE_BYTE + k]);

				printf(" ");
			}
			printf("\n");
			for (int j = 0; j < num_nodes; ++j)
			{
				for (int k = 0; k < HASH_SIZE_BYTE; ++k)
					printf("%02x", (unsigned char)tree_cpu[((1 << i) - 1 + j) * HASH_SIZE_BYTE + k]);

				printf(" ");
			}
			printf("\n");
		}*/

		printf("equal: %s\n", memcmp(&tree[0], &tree_cpu[0], ((1 << (TREE_DEPTH + 1)) - 1) * HASH_SIZE_BYTE)?"false":"true");

		delete tree_cpu;
		cudaFreeHost(tree);
	}

	qsort(times, NUM_MEASURMENTS, sizeof(float), cmp_f);
	cout << "median: " << times[NUM_MEASURMENTS / 2] << " ms\n   mid: " << sum / (NUM_MEASURMENTS - 2) << " ms" << endl;
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

#ifdef TEST_OTS_EXTERNAL_PRIVATE_KEY
	status = testOTSExternalPrivateKey();

	if (status == SUCCESS)
		cout << OTS_SUCCESS << endl;
	else if (status == FAILED_TO_ACQUIRE_CRYPT_PROV)
		cout << FAILED_TO_ACQUIRE_CRYPT_PROV_STRING << endl;
	else if (status == FAILED_TO_GENERATE_CRYPT_RAND_BYTES)
		cout << FAILED_TO_GENERATE_CRYPT_RAND_BYTES_STRING << endl;
#endif

#ifdef TEST_MERKLE_TREE
	testMerkleTree();
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