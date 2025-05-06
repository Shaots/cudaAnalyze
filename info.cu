#include "info.cuh"

void printInfoDevice() {
    cudaDeviceProp deviceProp;
    cudaGetDeviceProperties(&deviceProp, 0);
    std::cout << "Cuda Properties: " << std::endl;
    std::cout << "Name: " << deviceProp.name << std::endl;
    std::cout << "Global memory available " << deviceProp.totalGlobalMem / 1024 << " KB, "
              << deviceProp.totalGlobalMem / (1024 * 1024) << " MB" << std::endl;
    std::cout << "Total constant memory: " << deviceProp.totalConstMem / 1024 << " KB" << std::endl;
    std::cout << "Shared memory per block: " << deviceProp.sharedMemPerBlock / 1024 << " KB" << std::endl;
    std::cout << "Registers per block: " << deviceProp.regsPerBlock << std::endl;
    std::cout << "Warp size: " << deviceProp.warpSize << std::endl;
    std::cout << "Max threads per block: " << deviceProp.maxThreadsPerBlock << std::endl;
    std::cout << "Max threads dimensions: x = " << deviceProp.maxThreadsDim[0]
              << ", y = " << deviceProp.maxThreadsDim[1] << ", z = " << deviceProp.maxThreadsDim[2] << std::endl;
    std::cout << "Max grid size: x = " << deviceProp.maxGridSize[0] << ", y = " << deviceProp.maxGridSize[1]
              << ", z = " << deviceProp.maxGridSize[2] << std::endl;
    std::cout << "Compute capability: " << deviceProp.major << "." << deviceProp.minor << std::endl;
    std::cout << "Number of multiprocessors on device: " << deviceProp.multiProcessorCount << std::endl;
    std::cout << "Peak memory clock frequency: " << deviceProp.memoryClockRate / 1000 << " MHz" << std::endl;
    std::cout << std::endl;
}