#pragma once

#include "device_launch_parameters.h"

#define TILE_DIM 16

__global__ void mat_vec_mul_kernel(
    const double* W,
    const double* v,
    double* output,
    int in_dim,
    int out_dim
);

__global__ void mat_mul_kernel(
    const double* W,
    const double* V,
    double* output,
    int A, int B, int C
);

// mat_mul_kernel: C = A*B * B*C
__global__ void tiled_mat_mul_kernel(
    const double* W,   // [A * B], row-major
    const double* V,   // [B * C], row-major
    double* output,    // [A * C], row-major
    int A, int B, int C
);

__global__ void tiled_mat_mul_kernel_ex(double* A, double* B, double* C, int N1, int N2, int N3);

__global__ void mat_add_kernel(const double* a, const double* b, double* c, int N1, int N2);




