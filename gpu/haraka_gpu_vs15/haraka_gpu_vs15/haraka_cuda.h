
#ifndef HARAKA_CUDA_H
#define HARAKA_CUDA_H

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <vector>

//-----------------------------------------------------------------------------
//functions Haraka

// Input
// msg........Massages to be hashed. Has to be num_msgs times 512-bit large.
// num_msgs...Number of Messages to hash.
// Output
// hash.......Hashes of all msg from the input. Has to be num_msgs times 256-bit large.
// Descritpion
// Creates the hashes of all messages from the input.
__global__ void haraka512Kernel(const uint64_t* msg, uint64_t* hash, const uint32_t num_msgs);


// Input
// msg........Massages to be hashed. Has to be num_msgs times 256-bit large.
// num_msgs...Number of Messages to hash.
// Output
// hash.......Hashes of all msg from the input. Has to be num_msgs times 256-bit large.
// Descritpion
// Creates the hashes of all messages from the input.
__global__ void haraka256Kernel(const uint64_t* msg, uint64_t* hash, const uint32_t num_msgs);


//-----------------------------------------------------------------------------
//functions Winternitz OTS

// Input
// msg........Massages from which the bit-strings b for the OTS are created.
//            Has to be num_msgs times 256-bit large.
// num_msgs...Number of Messages.
// Output
// b..........bit-strings of all msg from the input. Has to be num_msgs times 34-Byte large.
// Descritpion
// Creates the bit-strings b needed for the Winternitz OTS
__global__ void harakaOTSKernel(const uint64_t* msg, uint16_t* b, const uint32_t num_msgs);


// Input
// priv_key.....private keys (signature keys) for the signature creation.
//              Has to be num_chunks times 256-bit large.
// b............bit-strings which determine how often each chunk of the private key is hashed.
//              Has to be num_chunks times 8-bit large.
// num_chunks...number of chunks in which the private keys are divided.
// Output
// pub_key......public keys (verification keys) for the corresponding private keys (signature keys).
//              Has to be num_chunks times 256-bit large.
// signature....signatures created with the bit-strings, which depend on the messages.
//              Has to be num_chunks times 256-bit large.
// Descritpion
// Creates the public keys (verification keys) and the signature (which depends on the bit-strings b)
// from the private keys (signature keys).
__global__ void harakaOTSCreateSignatureKernel(const uint64_t* priv_key, uint64_t* pub_key, uint64_t* signature, const uint8_t* b, const uint32_t num_chunks);


// Input
// b..............bit-strings which determine how often each chunk of the signature is hashed.
//                Has to be num_chunks times 8-bit large.
// signature......signatures from which the verification is computed
//                Has to be num_chunks times 256-bit large.
// num_chunks.....number of chunks in which the signatures are divided.
// Output
// verification...verifications created with the bit-strings, which depend on the messages.
//                Has to be num_chunks times 256-bit large.
// Descritpion
// Creates the verifications for the signatures, which should be equal to the
// public keys (verification keys).
__global__ void harakaOTSCreateVerificationKernel(const uint64_t* signature, uint64_t* verification, uint8_t* b, const uint32_t num_chunks);

#endif