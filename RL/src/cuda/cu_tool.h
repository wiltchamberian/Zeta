#pragma once
#include <cuda_runtime.h>
#include <iostream>

#define CU_CHECK(x) \
    do { \
        cudaError_t err = x; \
        if (err != cudaSuccess) { \
            std::cerr << "CUDA error: " << cudaGetErrorString(err) << std::endl; \
            std::abort(); \
        } \
    } while (0)

cudaError_t CudaInit();

void CudaGetDeviceProps();

cudaError_t CudaGetLastError();

cudaError_t CudaDeviceSynchronize();

