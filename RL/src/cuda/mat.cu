// matmul.cuh
#pragma once

#include "mat.cuh"

//#include <cuda_runtime.h>
#include <device_functions.h>

#define TILE_WIDTH 16

#define CUDA_CHECK(err) {if (err != cudaSuccess){printf("%s in %s at line %d \n", cudaGetErrorString(err), __FILE__, __LINE__);exit(EXIT_FAILURE);}}

__device__ __forceinline__
double dot(const double* a, const double* b, int n) {
    double s = 0.0;
    for (int i = 0; i < n; ++i)
        s += a[i] * b[i];
    return s;
}

//W:[out_dim * in_dim]
//v:[in_dim]
//output:[out_dim]
__global__ void mat_vec_mul_kernel(const double* W, const double* v, double* output, int in_dim, int out_dim)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= out_dim) {
        return;
    }
    double s = 0.0;
    for (int j = 0; j < in_dim; ++j) {
        s += W[i * in_dim + j] * v[j];
    }
    output[i] = s;
}

//dispatch A* C threads in total
__global__
void mat_mul_kernel(
    const double* W,   // [A * B]
    const double* V,   // [B * C]
    double* output,    // [A * C]
    int A,
    int B,
    int C
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    if (i >= A || j >= C) {
        return;
    }
    double s = 0.0;
    for (int k = 0; k < B; ++k) {
        s += W[i * B + k] * V[k * C + j];
    }
    output[i * C + j] = s;
}

// A: N1 x N2
// x: N2
// y: N1
__global__ void tiled_mat_vec_kernel(const double* A, const double* x, double* y, int N1, int N2)
{
    // Block/thread indices
    int by = blockIdx.y;
    int bx = blockIdx.x;   // 只用 bx = 0 就够了
    int ty = threadIdx.y;
    int tx = threadIdx.x;

    // Global row index
    int row = TILE_WIDTH * by + ty;

    if (row >= N1) return;

    __shared__ double sh_A[TILE_WIDTH][TILE_WIDTH];  // 缓存 A 的 tile
    __shared__ double sh_x[TILE_WIDTH];             // 缓存向量 x 的 tile

    double sum = 0.0;

    int numTiles = (N2 + TILE_WIDTH - 1) / TILE_WIDTH;

    for (int t = 0; t < numTiles; t++)
    {
        int col = t * TILE_WIDTH + tx;

        // Load tile of A
        if (col < N2) {
            sh_A[ty][tx] = A[row * N2 + col];
        }
        else {
            sh_A[ty][tx] = 0.0;
        }

        // Load tile of x (只需要 tx 线程加载一行)
        if (ty == 0 && col < N2) {
            sh_x[tx] = x[col];
        }

        __syncthreads();

        // Dot product within the tile
        for (int k = 0; k < TILE_WIDTH; k++) {
            sum += sh_A[ty][k] * sh_x[k];
        }

        __syncthreads();
    }

    // Store result
    y[row] = sum;
}


//A: N1*N2
//B: N2*N3
//C: N1*N3
__global__ void tiled_mat_mul_kernel(double* A, double* B, double* C, int N1, int N2, int N3)
{
    //// Ensure that TILE_WIDTH = BLOCK_SIZE
    //assert(TILE_WIDTH == blockDim.x);
    //assert(TILE_WIDTH == blockDim.y);

    // Details regarding this thread
    int by = blockIdx.y;
    int bx = blockIdx.x;

    int ty = threadIdx.y;
    int tx = threadIdx.x;

    // Working on C[i,j]
    int i = TILE_WIDTH * by + ty;
    int j = TILE_WIDTH * bx + tx;

    if (i >= N1 || j >= N3) {
        return;
    }

    // Allocating shared memory
    __shared__ double sh_A[TILE_WIDTH][TILE_WIDTH];
    __shared__ double sh_B[TILE_WIDTH][TILE_WIDTH];

    // Parallel mat mul
    double value = 0;
    int phaseCount = (N2 + TILE_WIDTH - 1) / TILE_WIDTH;
    for (int phase = 0; phase < phaseCount; phase++)
    {
        // Load Tiles into shared memory
        if ((phase * TILE_WIDTH + tx) < N2) {
            sh_A[ty][tx] = A[(i)*N2 + phase * TILE_WIDTH + tx];
        }
        else {
            sh_A[ty][tx] = 0.0;
        }
            
        if ((phase * TILE_WIDTH + ty) < N2){
            sh_B[tx][ty] = B[(phase * TILE_WIDTH + ty) * N3 + j];
        }
        else {
            sh_B[tx][ty] = 0.0;
        }
            
        __syncthreads();

        // Dot product
        for (int k = 0; k < TILE_WIDTH; k++) {
            value += sh_A[ty][k] * sh_B[tx][k];
        }
            
        __syncthreads();
    }

    // Assigning calculated value
    C[i * N3 + j] = value;
    
        
}

//A: N1*N2
//B: N2*N3
//C: N1*N3
__global__ void tiled_mat_mul_kernel_ex(double* A, double* B, double* C, int N1, int N2, int N3)
{
    // Details regarding this thread
    int by = blockIdx.y;
    int bx = blockIdx.x;

    int ty = threadIdx.y;
    int tx = threadIdx.x;

    // Working on C[i,j]
    int i = TILE_WIDTH * by + ty;
    int j = TILE_WIDTH * bx + tx;


    // Allocating shared memory
    __shared__ double sh_A[TILE_WIDTH][TILE_WIDTH];
    __shared__ double sh_B[TILE_WIDTH][TILE_WIDTH];

    // Parallel mat mul
    double value = 0;
    int phaseCount = (N2 + TILE_WIDTH - 1) / TILE_WIDTH;
    for (int phase = 0; phase < phaseCount; phase++)
    {
        int A_y = i;
        int A_x = phase * TILE_WIDTH + tx;
        if (A_y < N1 && A_x < N2) {
            sh_A[ty][tx] = A[A_y * N2 + A_x];
        }
        else {
            sh_A[ty][tx] = 0;
        }
        int B_x = j;
        int B_y = phase * TILE_WIDTH + ty;
        if (B_y < N2 && B_x < N3) {
            sh_B[ty][tx] = B[B_y * N3 + B_x];
        }
        else {
            sh_B[ty][tx] = 0;
        }

        __syncthreads();

        // Dot product
        for (int k = 0; k < TILE_WIDTH; k++) {
            value += sh_A[ty][k] * sh_B[k][tx];
        }

        __syncthreads();
    }

    if (i >= N1 || j >= N3) {
        return;
    }

    // Assigning calculated value
    C[i * N3 + j] = value;


}

__global__ void mat_add_kernel(const double* a, const double* b, double* c, int N1, int N2)
{
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    int i = blockIdx.y * blockDim.y + threadIdx.y;
    if (i >= N1 || j >= N2) return;


    c[i * N2 + j] = a[i * N2 + j] + b[i * N2 + j];
}

