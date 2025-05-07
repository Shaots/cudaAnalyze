#include <iostream>
#include "coalesced.cuh"

__global__ void noCoalescedAccess(float* odata, float* idata, int size) {
    int len = (size - 1) / blockDim.x + 1;
    for (int i = 0; i < len; ++i) {
        odata[threadIdx.x * len + i] = idata[threadIdx.x * len + i];
    }
}

__global__ void coalescedAccess(float* odata, float* idata, int size) {
    int len = (size - 1) / blockDim.x + 1;
    for (int i = 0; i < len; ++i) {
        odata[i * blockDim.x + threadIdx.x] = idata[i * blockDim.x + threadIdx.x];
    }
}

void test(coalesced f, int warm, int times){
    size_t len = 100000;
    float* arr = new float[len];
    for (size_t i = 0; i < len; ++i) {
        arr[i] = 5.5f;
    }
    float* d_arr_in;
    cudaMalloc((void**)&d_arr_in, len * sizeof(float));
    cudaMemcpy(d_arr_in, arr, len, cudaMemcpyHostToDevice);

    float* d_arr_out;
    cudaMalloc((void**)&d_arr_out, len * sizeof(float));

    dim3 threadDim = 32;

    cudaEvent_t start, stop;
    float       time;

    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    for (int i = 0; i < warm; ++i) {
        f<<<1, threadDim>>>(d_arr_out, d_arr_in, len);
    }

    cudaDeviceSynchronize();

    cudaEventRecord(start, 0);

    for (int i = 0; i < times; ++i) {
        f<<<1, threadDim>>>(d_arr_out, d_arr_in, len);
    }

    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);

    cudaEventElapsedTime(&time, start, stop);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    float* out = new float[len];
    cudaMemcpy(out, d_arr_out, len, cudaMemcpyDeviceToHost);

    std::cout << time / times << " ms" << std::endl;

    cudaFree(d_arr_in);
    cudaFree(d_arr_out);
    delete[] arr;
    delete[] out;
}