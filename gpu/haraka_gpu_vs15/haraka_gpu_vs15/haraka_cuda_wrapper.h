
#ifndef HARAKA_CUDA_WRAPPER
#define HARAKA_CUDA_WRAPPER

#include "haraka_cuda.h"

//Cuda defines
const uint32_t MAX_THREAD = 256;
const uint32_t NUM_STREAMS = 16;

//Haraka defines
const uint32_t AES_BLOCK_SIZE = 16;
const uint32_t HASH_SIZE_BYTE = 32;
const uint32_t MSG_SIZE_BYTE_256 = 32;
const uint32_t MSG_SIZE_BYTE_512 = 64;

//Winternitz OTS defines
const uint32_t HASH_SIZE_BIT = 256;
const uint32_t WINTERNITZ_PARAM = 8;
const uint32_t T1 = (HASH_SIZE_BIT + WINTERNITZ_PARAM - 1) / WINTERNITZ_PARAM;
const uint32_t T2 = (int(log2f(float(T1))) + WINTERNITZ_PARAM + WINTERNITZ_PARAM - 1) / WINTERNITZ_PARAM;
const uint32_t T = T1 + T2;

//Merkle Tree defines
const uint32_t DEPTH_PER_KERNEL = 6;

const int32_t FAILED_TO_ACQUIRE_CRYPT_PROV = -1;
const int32_t FAILED_TO_GENERATE_CRYPT_RAND_BYTES = -2;
const int32_t SUCCESS = 0;


//-----------------------------------------------------------------------------
//Haraka Wrapper functions

// Input
// msgs........Massages to be hashed. Has to be num_msgs times 512-bit large.
// num_msgs....Number of Messages to hash.
// Output
// hashes......Hashes of all msg from the input. Has to be num_msgs times 256-bit large.
// Descritpion
// Creates the hashes of all messages from the input.
cudaError_t harakaCuda512(const char* msgs, char* hashes, const uint32_t num_msgs);


// Input
// msgs........Massages to be hashed. Has to be num_msgs times 256-bit large.
// num_msgs....Number of Messages to hash.
// Output
// hashes......Hashes of all msg from the input. Has to be num_msgs times 256-bit large.
// Descritpion
// Creates the hashes of all messages from the input.
cudaError_t harakaCuda256(const char* msgs, char* hashes, const uint32_t num_msgs);


//-----------------------------------------------------------------------------
//Winternitz OTS Wrapper functions

//Works only for Windows, because of the secure random number generator

// Input
// msgs........messages for which the signature is created. Has to be num_msgs times 256-bit large.
// num_msgs....number of messages that should be signed.
// Output
// signature...signatures of the input messages. Has to be num_msgs times 34 times 256-bit large.
// pub_key.....public keys (verification keys) for the corresponding signature - message pairs.
//             Has to be num_msgs times 34 times 256-bit large.
// Descritpion
// Creates the signatures and the corresponding public keys (verification keys) for the
// input messages.
#ifdef _WIN32
int harakaWinternitzCudaSign(const char* msgs, char* signatures, char* pub_keys, const uint32_t num_msgs);
#endif


// Should work on all OS, because of external private key generation.

// Input
// msgs........messages for which the signature is created. Has to be num_msgs times 256-bit large.
// num_msgs....number of messages that should be signed.
// Output
// signature...signatures of the input messages. Has to be num_msgs times 34 times 256-bit large.
// pub_key.....public keys (verification keys) for the corresponding signature - message pairs.
//             Has to be num_msgs times 34 times 256-bit large.
// Descritpion
// Creates the signatures and the corresponding public keys (verification keys) for the
// input messages from the private keys (signature keys).
int harakaWinternitzCudaSign(const char* msgs, const char* priv_keys, char* signatures, char* pub_keys, const uint32_t num_msgs);


// Input
// msgs........messages for which the signature should be verified. Has to be num_msgs times 256-bit large.
// signature...signatures of the input messages. Has to be num_msgs times 34 times 256-bit large.
// pub_key.....public keys (verification keys) for the corresponding signature - message pairs.
//             Has to be num_msgs times 34 times 256-bit large.
// num_msgs....number of messages that should be verified.
// Return Value
// 0...If the verification was successfull
// 1...If the verification failed
// Descritpion
// Verifies if the signatures and the messages fit to the corresponding public keys (verification keys).
int harakaWinternitzCudaVerify(const char* msgs, const char* signatures, const char* pub_keys, const uint32_t num_msgs);


//-----------------------------------------------------------------------------
//Merkle Tree Wrapper functions

// Input
// depth...depth of the Merkle tree that should be created
// Output
// tree....Merkle tree of the given depth, the leaves should already be populatet with the hashes of the public keys (verification keys)
//         from tree[(1 << depth) - 1] to tree[(1 << (depth + 1)) - 2]
// Description
// Builds the Merkle Tree of the given depth.
int harakaBuildMerkleTree(char* tree, const uint32_t depth);

#endif