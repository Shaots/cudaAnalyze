#ifndef MATRIX_MATRIX_CUH
#define MATRIX_MATRIX_CUH

#include <iostream>
#include <cstdlib>

// size(a) = M * K
// size(b) = K * N
__global__ void simpleMultiply(const int* a, const int* b, int* c, int M, int K, int N);

// a * a^T
// size(a) = M * K
__global__ void simpleMultiplyAAT(const int* a, int* c, int M, int K);

__global__ void coalescedMultiply(const int* a, const int* b, int* c, int M, int K, int N);

__global__ void sharedMultiply(const int* a, const int* b, int* c, int M, int K, int N);

typedef void (*multiFunc)(const int* a, const int* b, int* c, int M, int K, int N);

typedef void (*multiAATFunc)(const int* a, int* c, int M, int K);

void testMultiply(multiFunc func, int warm = 20, int times = 100);

void testMultiply(multiAATFunc func, int warm = 20, int times = 100);

void multiply(const int* a, const int* b, int* c, int M, int K, int N);

void multiply(const int* a, int* c, int M, int K);
#endif