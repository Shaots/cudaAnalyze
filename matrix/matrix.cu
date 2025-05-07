#include "matrix.cuh"

__global__ void simpleMultiply(const int* a, const int* b, int* c, int M, int K, int N) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int sum = 0;
    if (col >= N || row >= M) return;
    for (int i = 0; i < K; ++i) {
        sum += a[row * K + i] * b[i * N + col];
    }
    c[row * N + col] = sum;
}

__global__ void coalescedMultiply(const int* a, const int* b, int* c, int M, int K, int N) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;

    if (col >= N) return;

    __shared__ int sh[32][32];
    int            sum = 0;
    int            numSubMatrix = (K - 1) / 32 + 1;
    for (int sub = 0; sub < numSubMatrix; ++sub) {
        if (row >= M || threadIdx.x + sub * 32 >= K)
            sh[threadIdx.y][threadIdx.x] = 0;
        else
            sh[threadIdx.y][threadIdx.x] = a[row * K + sub * 32 + threadIdx.x];
        __syncthreads();
        for (int i = 0; i < 32; ++i) {
            sum += sh[threadIdx.y][i] * b[(sub * 32 + i) * N + col];
        }
        __syncthreads();
    }
    if (row < M && col < N) c[row * N + col] = sum;
}

void testMultiply(multiFunc func, int warm, int times) {
    int M = 1000;
    int K = 2000;
    int N = 3000;

    int* a = new int[M * K];
    int* b = new int[K * N];
    int* c = new int[M * N];

    for (int i = 0; i < M * K; ++i) a[i] = 2;
    for (int i = 0; i < K * N; ++i) b[i] = 3;

    int* d_a;
    int* d_b;
    int* d_c;
    cudaMalloc((void**)&d_a, M * K * sizeof(int));
    cudaMalloc((void**)&d_b, K * N * sizeof(int));
    cudaMalloc((void**)&d_c, M * N * sizeof(int));

    cudaMemcpy(d_a, a, M * K * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, b, K * N * sizeof(int), cudaMemcpyHostToDevice);

    dim3 blockDim = {32, 32};
    dim3 gridDim = {(N - 1) / blockDim.x + 1, (M - 1) / blockDim.y + 1};

    cudaEvent_t start, stop;
    float       time;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    for (int i = 0; i < warm; ++i) {
        func<<<gridDim, blockDim>>>(d_a, d_b, d_c, M, K, N);
    }

    cudaDeviceSynchronize();

    cudaEventRecord(start, 0);

    for (int i = 0; i < times; ++i) {
        func<<<gridDim, blockDim>>>(d_a, d_b, d_c, M, K, N);
    }

    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);

    cudaEventElapsedTime(&time, start, stop);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    cudaMemcpy(c, d_c, M * N * sizeof(int), cudaMemcpyDeviceToHost);

    bool flag = true;
    for (int i = 0; i < M * N; ++i) {
        if (c[i] != 6 * 2000) {
            std::cout << i << " " << c[i] << " ";
            flag = false;
            break;
        }
    }
    std::cout << flag << std::endl;

    std::cout << time / times << " ms" << std::endl;

    delete[] a;
    delete[] b;
    delete[] c;
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);
}