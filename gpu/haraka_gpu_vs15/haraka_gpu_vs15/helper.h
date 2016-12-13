
#ifndef HELPER_H
#define HELPER_H

#include <iostream>
#include <string>
#include <vector>

#define ROUNDS (5)
#define AES_PER_ROUND (2)

using namespace std;

void printVector(const string& message, const vector<char>& vec_to_print);

int haraka512256(char *hash, const char *msg);

#endif