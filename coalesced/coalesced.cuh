#ifndef COALESCED_CUH
#define COALESCED_CUH


__global__ void noCoalescedAccess(float* odata, float* idata, int size);

__global__ void coalescedAccess(float* odata, float* idata, int size);

typedef void (*coalesced)(float*, float*, int);

void test(coalesced f, int warm = 100, int times = 1000);

#endif