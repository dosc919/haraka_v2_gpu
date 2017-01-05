
#ifndef HARAKA_CUDA_H
#define HARAKA_CUDA_H

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <vector>

using namespace std;

//Cuda defines
const uint32_t MAX_THREAD = 256;
const uint32_t NUM_STREAMS = 1;

//Haraka defines
const uint32_t AES_BLOCK_SIZE = 16;
const uint32_t STATE_THREAD_AES = 4;
const uint32_t HASH_SIZE_BYTE = 32;
const uint32_t MSG_SIZE_BYTE_256 = 32;
const uint32_t MSG_SIZE_BYTE_512 = 64;

//Winternitz OTS defines
const uint32_t HASH_SIZE_BIT = 256;
const uint32_t WINTERNITZ_PARAM = 8;
const uint32_t T1 = (HASH_SIZE_BIT + WINTERNITZ_PARAM - 1) / WINTERNITZ_PARAM;
const uint32_t T2 = (int(log2f(float(T1))) + WINTERNITZ_PARAM + WINTERNITZ_PARAM) / WINTERNITZ_PARAM;
const uint32_t T = T1 + T2;

const int32_t FAILED_TO_ACQUIRE_CRYPT_PROV = -1;
const int32_t FAILED_TO_GENERATE_CRYPT_RAND_BYTES = -2;
const int32_t SUCCESS = 0;

//functions
cudaError_t harakaCuda512(const char* msgs, char* hashes, const uint32_t num_msgs);

cudaError_t harakaCuda256(const char* msgs, char* hashes, const uint32_t num_msgs);

int harakaWinternitzCudaSign(const char* msgs, char* signatures, char* pub_keys, const uint32_t num_msgs);

int harakaWinternitzCudaVerify(const char* msgs, const char* signatures, const char* pub_keys, const uint32_t num_msgs);

#endif