
#ifndef HELPER_H
#define HELPER_H

#include <iostream>
#include <string>
#include <vector>

#define ROUNDS (5)
#define AES_PER_ROUND (2)

using namespace std;

void printVector(const string& message, const char* vec_to_print, uint32_t size);

int haraka512256(char *hash, const char *msg);

int haraka256256(char *hash, const char *msg);

#endif