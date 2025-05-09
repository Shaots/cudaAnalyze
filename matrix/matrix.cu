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

__global__ void simpleMultiplyAAT(const int* a, int* c, int M, int K) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int sum = 0;
    if (col >= M || row >= M) return;
    for (int i = 0; i < K; ++i) {
        // In a[col * K + i] there is strided access
        // For thread 0: a[0 * K + i]
        // For thread 1: a[1 * K + i]
        sum += a[row * K + i] * a[col * K + i];
    }
    c[row * M + col] = sum;
}

__global__ void coalescedMultiply(const int* a, const int* b, int* c, int M, int K, int N) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;

    // Here is no condition return !!!

    __shared__ int sh[32][32];
    int            sum = 0;
    int            numSubMatrix = (K >> 5) + 1;
    for (int sub = 0; sub < numSubMatrix; ++sub) {
        // We cannot write
        // if (row >= M || col >= N || threadIdx.x + sub * 32 >= K)
        if (row >= M || threadIdx.x + sub * 32 >= K)
            sh[threadIdx.y][threadIdx.x] = 0;
        else
            sh[threadIdx.y][threadIdx.x] = a[row * K + sub * 32 + threadIdx.x];
        __syncwarp();
        for (int i = 0; i < 32; ++i) {
            sum += sh[threadIdx.y][i] * b[(sub * 32 + i) * N + col];
        }
        __syncwarp();
    }
    if (row < M && col < N) c[row * N + col] = sum;
}

// For example:
// blockIdx.x = 1, threadIdx.x = 1
// blockIdx.y = 0, threadIdx.y = 0
// --> col = 5, row = 0, c05 = a00 * b05 + a01 * b15 + a02 * b25 + ... + a07 * b75 + a08 * b85
//                           = a00 * b50 + a01 * b51 + a02 * b52 + ... + a07 * b57 + a08 * b58
// a = [a00, a01, a02 | a03, a04, a05 | a06, a07, a08
//      a10, a11, a12 | a13, a14, a15 | a16, a17, a18
//      a20, a21, a22 | a23, a24, a25 | a26, a27, a28
//      --------------|---------------|---------------
//      a30, a31, a32 | a33, a34, a35 | a36, a37, a38
//      a40, a41, a42 | a43, a44, a45 | a46, a47, a48
//      a50, a51, a52 | a53, a54, a55 | a56, a57, a58]
//
// shA = [a00, a01, a02 | a03, a04, a05 | a06, a07, a08
//        a10, a11, a12 | a13, a14, a15 | a16, a17, a18
//        a20, a21, a22 | a23, a24, a25 | a26, a27, a28]
// sub = | <---- 0 ---> | <---- 1 ----> | <---- 2 ---->|
//
// First step: read one block : IN ONE BLOCK WE TRANSPOSED
// tr(a) =
//       [a30, a40, a50 | a33, a43, a53 | a36, a46, a56
//        a31, a41, a51 | a34, a44, a54 | a37, a38, a39
//        a32, a42, a52 | a35, a45, a55 | a38, a48, a58]
// sub = | <---- 0 ---> | <---- 1 ----> | <---- 2 ---->|
// Second step: multiply
// c = [a00, a01, a02] .* [a50, a51, a52] // sub == 0, shA[threadIdx.y][i] * tr[i][threadIdx.x]
//   + [a03, a04, a05] .* [a53, a54, a55] // sub == 1
//   + [a06, a07, a08] .* [a56, a57, a58] // sub == 2
__global__ void coalescedMultiplyAAT(const int* a, int* c, int M, int K) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;

    __shared__ int shA[32][32];
    __shared__ int transposed[32][32+1]; // To prevent blank conflict
    int            sum = 0;
    int            numSubMatrix = (K >> 5) + 1;
    for (int sub = 0; sub < numSubMatrix; ++sub) {
        if (row >= M || threadIdx.x + sub * 32 >= K)
            shA[threadIdx.y][threadIdx.x] = 0;
        else
            shA[threadIdx.y][threadIdx.x] = a[row * K + sub * 32 + threadIdx.x];

        // In one block, transpose sub matrix
        if (blockIdx.x * blockDim.x + threadIdx.y >= M || threadIdx.x + sub * 32 >= K)
            transposed[threadIdx.x][threadIdx.y] = 0;
        else
            // transposed[threadIdx.x][threadIdx.y] = a[threadIdx.y][threadIdx.x];
            transposed[threadIdx.x][threadIdx.y] = a[(blockIdx.x * blockDim.x + threadIdx.y) * K + sub * 32 + threadIdx.x];
            // we can't do transposed[threadIdx.y][threadIdx.x] = a[threadIdx.x][threadIdx.y]; --- this is strided access
        __syncthreads();
        for (int i = 0; i < 32; ++i) {
            sum += shA[threadIdx.y][i] * transposed[i][threadIdx.x];
        }
        __syncthreads();
    }
    if (row < M && col < M) c[row * M + col] = sum;
}

__global__ void sharedMultiply(const int* a, const int* b, int* c, int M, int K, int N) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;

    // Here is no condition return !!!

    __shared__ int shA[32][32];
    __shared__ int shB[32][32];
    int            sum = 0;
    int            numSubMatrix = (K >> 5) + 1;
    for (int sub = 0; sub < numSubMatrix; ++sub) {
        // We cannot write
        // if (row >= M || col >= N || threadIdx.x + sub * 32 >= K)
        if (row >= M || sub * 32 + threadIdx.x >= K)
            shA[threadIdx.y][threadIdx.x] = 0;
        else
            shA[threadIdx.y][threadIdx.x] = a[row * K + sub * 32 + threadIdx.x];

        if (col >= N || sub * 32 + threadIdx.y >= K)
            shB[threadIdx.y][threadIdx.x] = 0;
        else
            shB[threadIdx.y][threadIdx.x] = b[(sub * 32 + threadIdx.y) * N + col];
        __syncthreads();
        for (int i = 0; i < 32; ++i) {
            sum += shA[threadIdx.y][i] * shB[i][threadIdx.x];
        }
        __syncthreads();
    }
    if (row < M && col < N) c[row * N + col] = sum;
}

void testMultiply(multiFunc func, int warm, int times) {
    std::srand(std::time({})); // use current time as seed for random generator

    int M = 1000;
    int K = 2000;
    int N = 3000;

    int* a = new int[M * K];
    int* b = new int[K * N];
    int* c = new int[M * N];
    int* cpuMultiply = new int[M * N];

    for (int i = 0; i < M * K; ++i) a[i] = std::rand() % 20;
    for (int i = 0; i < K * N; ++i) b[i] = std::rand() % 20;

    multiply(a, b, cpuMultiply, M, K, N);

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
        if (c[i] != cpuMultiply[i]) {
            std::cout << "Error: " << i << " " << c[i] << " " << cpuMultiply[i];
            flag = false;
            break;
        }
    }
    std::cout << std::boolalpha << flag << std::endl;

    std::cout << time / times << " ms" << std::endl;

    delete[] a;
    delete[] b;
    delete[] c;
    delete[] cpuMultiply;
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);
}

void testMultiply(multiAATFunc func, int warm, int times) {
    std::srand(std::time({})); // use current time as seed for random generator

    int M = 1000;
    int K = 2000;

    int* a = new int[M * K];
    int* c = new int[M * M];
    int* cpuMultiply = new int[M * M];

    for (int i = 0; i < M * K; ++i) a[i] = std::rand() % 20;

    multiply(a, cpuMultiply, M, K);

    int* d_a;
    int* d_c;
    cudaMalloc((void**)&d_a, M * K * sizeof(int));
    cudaMalloc((void**)&d_c, M * M * sizeof(int));

    cudaMemcpy(d_a, a, M * K * sizeof(int), cudaMemcpyHostToDevice);

    dim3 blockDim = {32, 32};
    dim3 gridDim = {(M - 1) / blockDim.x + 1, (M - 1) / blockDim.y + 1};

    cudaEvent_t start, stop;
    float       time;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    for (int i = 0; i < warm; ++i) {
        func<<<gridDim, blockDim>>>(d_a, d_c, M, K);
    }

    cudaDeviceSynchronize();

    cudaEventRecord(start, 0);

    for (int i = 0; i < times; ++i) {
        func<<<gridDim, blockDim>>>(d_a, d_c, M, K);
    }

    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);

    cudaEventElapsedTime(&time, start, stop);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    cudaMemcpy(c, d_c, M * M * sizeof(int), cudaMemcpyDeviceToHost);

    bool flag = true;
    for (int i = 0; i < M * M; ++i) {
        if (c[i] != cpuMultiply[i]) {
            std::cout << "Error: " << i << " " << c[i] << " " << cpuMultiply[i] << " ";
            flag = false;
            break;
        }
    }
    std::cout << std::boolalpha << flag << std::endl;

    std::cout << time / times << " ms" << std::endl;

    delete[] a;
    delete[] c;
    delete[] cpuMultiply;
    cudaFree(d_a);
    cudaFree(d_c);
}

void multiply(const int* a, const int* b, int* c, int M, int K, int N) {
    int sum;
    for(int row = 0; row < M; ++row) {
        for (int col = 0; col < N; ++col) {
            sum = 0;
            for (int k = 0; k < K; ++k) {
                sum += a[row * K + k] * b[k * N + col];
            }
            c[row * N + col] = sum;
        }
    }
}

void multiply(const int* a, int* c, int M, int K) {
    int sum;
    for(int row = 0; row < M; ++row) {
        for (int col = 0; col < M; ++col) {
            sum = 0;
            for (int k = 0; k < K; ++k) {
                sum += a[row * K + k] * a[col * K + k];
            }
            c[row * M + col] = sum;
        }
    }
}