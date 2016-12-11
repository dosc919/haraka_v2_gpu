
#ifndef HARAKA_CUDA_O_H
#define HARAKA_CUDA_O_H

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <vector>

using namespace std;

const unsigned int MAX_THREAD = 1024;
const unsigned int AES_BLOCK_SIZE = 16;
const unsigned int STATE_THREAD_AES = 4;

cudaError_t harakaCuda(const vector<char>& msg, vector<char>& digest);

#endif