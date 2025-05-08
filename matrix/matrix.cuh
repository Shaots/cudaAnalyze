#ifndef MATRIX_MATRIX_CUH
#define MATRIX_MATRIX_CUH

#include <iostream>
#include <cstdlib>

// size(a) = M * K
// size(b) = K * N
__global__ void simpleMultiply(const int* a, const int* b, int* c, int M, int K, int N);

__global__ void coalescedMultiply(const int* a, const int* b, int* c, int M, int K, int N);

__global__ void sharedMultiply(const int* a, const int* b, int* c, int M, int K, int N);

typedef void (*multiFunc)(const int* a, const int* b, int* c, int M, int K, int N);

void testMultiply(multiFunc func, int warm = 20, int times = 100);

void multiply(const int* a, const int* b, int* c, int M, int K, int N);

#endif