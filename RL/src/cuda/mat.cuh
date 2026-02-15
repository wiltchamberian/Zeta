#pragma once

#include "device_launch_parameters.h"

#define TILE_DIM 16

__global__ void mat_vec_mul_kernel(
    const float* W,
    const float* v,
    float* output,
    int in_dim,
    int out_dim
);

__global__ void mat_mul_kernel(
    const float* W,
    const float* V,
    float* output,
    int A, int B, int C
);

// mat_mul_kernel: C = A*B * B*C
__global__ void tiled_mat_mul_kernel(
    const float* W,   // [A * B], row-major
    const float* V,   // [B * C], row-major
    float* output,    // [A * C], row-major
    int A, int B, int C
);

__global__ void tiled_mat_mul_kernel_ex(float* A, float* B, float* C, int N1, int N2, int N3);

__global__ void mat_add_kernel(const float* a, const float* b, float* c, int N1, int N2);




