
#ifndef CONSTANTS_H
#define CONSTANTS_H

#include <string>

//Strings
const string INPUT_TEXT = "Input:";
const string OUTPUT_TEXT = "Digest:";

const string ERROR_FILE_OPEN = ": file cannot be opened";
const string ERROR_FILE_READ_INPUT = ": input for hash function cannot be read";
const string ERROR_FILE_READ_DIGEST = ": reference digest output cannot be read";
const string ERROR_CUDA = ": cuda error occured during execution";
const string ERROR_DIGEST_MISSMATCH = ": digest doesn't match with reference digest";

const vector<string> TEST_FILES = {
	"test_1.tst",
	"blub"
};

//Sizes
const int DIGEST_SIZE_BYTE = 32;
const int INPUT_SIZE_BYTE = 64;

#endif