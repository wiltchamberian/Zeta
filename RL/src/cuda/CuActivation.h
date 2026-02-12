#pragma once
//#include <cudnn.h>


// ===== Linear =====
__host__ __device__ __forceinline__
double Linear(double x) {
    return x;
}

__host__ __device__ __forceinline__
double DLinear(double /*x*/) {
    return 1.0;
}

// ===== ReLU =====
__host__ __device__ __forceinline__
double RELU(double x) {
    return x > 0.0 ? x : 0.0;
}

__host__ __device__ __forceinline__
double DRELU(double x) {
    return x > 0.0 ? 1.0 : 0.0;
}

// ===== Leaky ReLU =====
__host__ __device__ __forceinline__
double LeakyRELU(double x, double a) {
    return x > 0.0 ? x : a * x;
}

__host__ __device__ __forceinline__
double DLeakyRELU(double x, double a) {
    return x > 0.0 ? 1.0 : a;
}
