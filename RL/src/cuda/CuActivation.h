#pragma once
//#include <cudnn.h>


// ===== Linear =====
__host__ __device__ __forceinline__
float Linear(float x) {
    return x;
}

__host__ __device__ __forceinline__
float DLinear(float /*x*/) {
    return 1.0;
}

// ===== ReLU =====
__host__ __device__ __forceinline__
float RELU(float x) {
    return x > 0.0 ? x : 0.0;
}

__host__ __device__ __forceinline__
float DRELU(float x) {
    return x > 0.0 ? 1.0 : 0.0;
}

// ===== Leaky ReLU =====
__host__ __device__ __forceinline__
float LeakyRELU(float x, float a) {
    return x > 0.0 ? x : a * x;
}

__host__ __device__ __forceinline__
float DLeakyRELU(float x, float a) {
    return x > 0.0 ? 1.0 : a;
}
