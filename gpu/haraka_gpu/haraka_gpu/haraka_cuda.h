
#ifndef HARAKA_CUDA_H
#define HARAKA_CUDA_H

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <vector>

using namespace std;

cudaError_t addWithCuda(int *c, const int *a, const int *b, unsigned int size);

cudaError_t harakaCuda(const vector<unsigned char>& msg, vector<unsigned char>& digest);

#endif