
#ifndef CONSTANTS_H
#define CONSTANTS_H

#include <string>

//Strings
const string INPUT_TEXT = "Input:";
const string OUTPUT_TEXT = "Hash:";
const string OUTPUT_REFERENCE_TEXT = "Hash Reference:";

const string TEST_SUCCESS_STRING = "test succeeded";
const string ERROR_CUDA_STRING = "error: cuda error occured during execution";
const string ERROR_HASH_MISSMATCH = "error: hash doesn't match with reference hash";

const string FAILED_TO_ACQUIRE_CRYPT_PROV_STRING = "error: failed to acquire a cryptographic provider";
const string FAILED_TO_GENERATE_CRYPT_RAND_BYTES_STRING = "error: failed to generate cryptographically random bytes";
const string OTS_SUCCESS = "signatures successfully created and verified";

//Error Codes
const int ERROR_CUDA = -1;
const int ERROR_HASH = 0;
const int TEST_SUCCESS = 1;

#endif
