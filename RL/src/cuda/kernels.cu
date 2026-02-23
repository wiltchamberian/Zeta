#include "kernels.h"
#include <cuda_runtime.h>
#include <cuda.h>
#include <cfloat>

#define WARP_SIZE 32

// Warp-level reduction for max
__inline__ __device__
float warp_reduce_max(float val) {
    for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2)
        val = fmaxf(val, __shfl_xor_sync(0xFFFFFFFF, val, offset));
    return val;
}

// Warp-level reduction for sum
__inline__ __device__
float warp_reduce_sum(float val) {
    for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2)
        val += __shfl_xor_sync(0xFFFFFFFF, val, offset);
    return val;
}


__global__ void tanh_forward_kernel(
    const float* input,   
    float* output,        
    int total
) {
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i >= total) return;

    output[i] = tanhf(input[i]);
}

__global__ void tanh_backward_kernel(
    const float* dC_da,    //N * dim     dC/da
    const float* a,        //N * dim
    float* output,         //N * dim     dC/dz
    int total
) {
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i >= total) return;

    float s = a[i];
    s = 1 - s * s;
    output[i] = dC_da[i] * s;

    return;
}

//input:X, W
//outupt := sigma(X * W^T)
__global__ void linear_leaky_relu_forward_kernel(
    const float* input,      // batch x in_dim
    const float* weights,    // out_dim x in_dim
    const float* bias,       // out_dim
    float* output,           // batch x out_dim
    int batch, int in_dim, int out_dim,
    float alpha
) {
    int ty = threadIdx.y;
    int tx = threadIdx.x;

    int i = blockDim.y * blockIdx.y + ty;
    int j = blockDim.x * blockIdx.x + tx;

    __shared__ float sh_A[TILE_WIDTH][TILE_WIDTH];
    __shared__ float sh_B[TILE_WIDTH][TILE_WIDTH];

    float value = 0;
    int phaseCount = (in_dim + TILE_WIDTH - 1) / TILE_WIDTH;

    for (int phase = 0; phase < phaseCount; phase++)
    {
        int A_y = i;
        int A_x = phase * TILE_WIDTH + tx;
        sh_A[ty][tx] = (A_y < batch && A_x < in_dim) ? input[A_y * in_dim + A_x] : 0.0;

        int B_x = j;
        int B_y = phase * TILE_WIDTH + ty;
        sh_B[ty][tx] = (B_y < in_dim && B_x < out_dim) ? weights[B_x * in_dim + B_y] : 0.0;

        __syncthreads();

        for (int k = 0; k < TILE_WIDTH; k++) {
            value += sh_A[ty][k] * sh_B[k][tx];
        }

        __syncthreads();
    }

    if (i >= batch || j >= out_dim) return;

    value += bias[j];

    // Fused LeakyReLU
    output[i * out_dim + j] = value > 0.0 ? value : alpha * value;
}

__global__ void linear_tanh_forward_kernel(
    const float* input,      // batch x in_dim
    const float* weights,    // out_dim x in_dim
    const float* bias,       // out_dim
    float* output,           // batch x out_dim
    int batch, int in_dim, int out_dim
) {

    int ty = threadIdx.y;
    int tx = threadIdx.x;

    int i = blockDim.y * blockIdx.y + ty;
    int j = blockDim.x * blockIdx.x + tx;

    __shared__ float sh_A[TILE_WIDTH][TILE_WIDTH];
    __shared__ float sh_B[TILE_WIDTH][TILE_WIDTH];

    float value = 0;
    int phaseCount = (in_dim + TILE_WIDTH - 1) / TILE_WIDTH;

    for (int phase = 0; phase < phaseCount; phase++)
    {
        int A_y = i;
        int A_x = phase * TILE_WIDTH + tx;
        sh_A[ty][tx] = (A_y < batch && A_x < in_dim) ? input[A_y * in_dim + A_x] : 0.0;

        int B_x = j;
        int B_y = phase * TILE_WIDTH + ty;
        sh_B[ty][tx] = (B_y < in_dim && B_x < out_dim) ? weights[B_x * in_dim + B_y] : 0.0;

        __syncthreads();

        for (int k = 0; k < TILE_WIDTH; k++) {
            value += sh_A[ty][k] * sh_B[k][tx];
        }

        __syncthreads();
    }

    if (i >= batch || j >= out_dim) return;

    value += bias[j];

    output[i * out_dim + j] = tanhf(value);


}


__global__ void leaky_relu_forward_kernel(
    const float* input,
    float* output,
    int total_elements,
    float alpha
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx >= total_elements) return;

    float x = input[idx];
    output[idx] = x > 0.0 ? x : alpha * x;
}

__global__ void leaky_relu_backward_kernel(
    const float* dC_da,
    const float* a,
    float* dC_dz,
    int total_elements,
    float alpha
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total_elements) return;

    float der = a[idx] > 0 ? 1 : alpha;

    dC_dz[idx] = dC_da[idx] * der;
}



__global__ void mse_loss_kernel(
    const float* a,       // batch * dim 
    const float* y,       // batch * dim
    float* loss,         // 1
    int total
) {
    int j = blockDim.x * blockIdx.x + threadIdx.x;

    float square_sum = 0;

    int laneId = threadIdx.x & (WARP_SIZE - 1);
    int warpId = threadIdx.x / WARP_SIZE;
    int tid = threadIdx.x;
    int numWarps = blockDim.x / WARP_SIZE;

    for (int i = threadIdx.x; i < total;  i += blockDim.x) {
        float temp = a[i] - y[i];
        square_sum += temp * temp;
    }

    float wrap_sum = warp_reduce_sum(square_sum);

    __shared__ float shared_m[32]; //make sure block_dim /WRAP_SIZE <= 32
    if (laneId == 0) {
        shared_m[warpId] = wrap_sum;
    }
    __syncthreads();

    if (warpId == 0) {
        float sum = tid < numWarps ? shared_m[tid] : 0;
        sum = warp_reduce_sum(sum);

        if (j == 0) {
            loss[0] = sum / total;
        }
    }
    
    return;

}

__global__ void mse_loss_backward_kernel(
    const float* a,       // batch x out_dim, a^L
    const float* y,       // batch x out_dim
    float* delta,         // batch x out_dim Êä³ö ¦Ä^L
    int batch,
    int out_dim
) {
    int j = blockIdx.x * blockDim.x + threadIdx.x; // neural
    int i = blockIdx.y * blockDim.y + threadIdx.y; // sample

    if (i >= batch || j >= out_dim) return;

    int idx = i * out_dim + j;
    //here needs to divide by feature size!!!!
    delta[idx] = (a[idx] - y[idx]) * 2 / (batch * out_dim);
}

//(BP2) ¦Ä^(l-1) = (¦Ä^{l} ¡¤ W^{l}) ¡Ñ ¦Ò'(z^(l-1))
__global__ void linear_leaky_relu_backward_kernel(
    const float* delta, // batch * dim_delta ¦Ä^{l} 
    const float* W,     // dim_delta * dim_delta_prev W^{l} 
    const float* a_prev, // batch * dim_delta_prev a^(l-1) 
    float* delta_prev,   // batch x dim_delta_prev Êä³ö ¦Ä^(l-1) 
    bool add,
    int batch,
    int dim_delta_prev,
    int dim_delta,
    float alpha              // LeakyReLU alpha
) {
    int ty = threadIdx.y;
    int tx = threadIdx.x;

    int i = blockDim.y * blockIdx.y + ty;
    int j = blockDim.x * blockIdx.x + tx;

    __shared__ float sh_A[TILE_WIDTH][TILE_WIDTH];
    __shared__ float sh_B[TILE_WIDTH][TILE_WIDTH];

    float value = 0.0;
    int phaseCount = (dim_delta + TILE_WIDTH - 1) / TILE_WIDTH;

    for (int phase = 0; phase < phaseCount; phase++)
    {
        int A_y = i;
        int A_x = phase * TILE_WIDTH + tx;
        sh_A[ty][tx] = (A_y < batch && A_x < dim_delta) ? delta[A_y * dim_delta + A_x] : 0.0;

        int B_x = j;
        int B_y = phase * TILE_WIDTH + ty;
        sh_B[ty][tx] = (B_y < dim_delta && B_x < dim_delta_prev) ? W[B_y * dim_delta_prev + B_x] : 0.0;

        __syncthreads();

        for (int k = 0; k < TILE_WIDTH; k++) {
            value += sh_A[ty][k] * sh_B[k][tx];
        }

        __syncthreads();
    }

    if (i >= batch || j >= dim_delta_prev) return;

    // Fused LeakyReLU backward
    float der = a_prev[i * dim_delta_prev + j] > 0.0 ? 1.0 : alpha;

    if (add) {
        delta_prev[i * dim_delta_prev + j] += (value * der);
    }
    else {
        delta_prev[i * dim_delta_prev + j] = value * der;
    }
    
}

__global__ void linear_tanh_backward_kernel(
    const float* delta, // batch x dim_delta_next ¦Ä^{l}
    const float* W,     // dim_delta_next x dim_delta W^{l}
    const float* a_prev,          // batch x dim_delta a^(l-1)
    float* delta_prev,            // batch x dim_delta Êä³ö ¦Ä^(l-1)
    bool add,
    int batch,
    int dim_delta_prev,
    int dim_delta
) {
    int ty = threadIdx.y;
    int tx = threadIdx.x;

    int i = blockDim.y * blockIdx.y + ty;
    int j = blockDim.x * blockIdx.x + tx;

    __shared__ float sh_A[TILE_WIDTH][TILE_WIDTH];
    __shared__ float sh_B[TILE_WIDTH][TILE_WIDTH];

    float value = 0.0;
    int phaseCount = (dim_delta + TILE_WIDTH - 1) / TILE_WIDTH;

    for (int phase = 0; phase < phaseCount; phase++)
    {
        int A_y = i;
        int A_x = phase * TILE_WIDTH + tx;
        sh_A[ty][tx] = (A_y < batch && A_x < dim_delta) ? delta[A_y * dim_delta + A_x] : 0.0;

        int B_x = j;
        int B_y = phase * TILE_WIDTH + ty;
        sh_B[ty][tx] = (B_y < dim_delta && B_x < dim_delta_prev) ? W[B_y * dim_delta_prev + B_x] : 0.0;

        __syncthreads();

        for (int k = 0; k < TILE_WIDTH; k++) {
            value += sh_A[ty][k] * sh_B[k][tx];
        }

        __syncthreads();
    }

    if (i >= batch || j >= dim_delta_prev) return;

    float a = a_prev[i * dim_delta_prev + j];
    a = 1 - a * a;
    if (add) {
        delta_prev[i * dim_delta_prev + j] += (value * a );
    }
    else {
        delta_prev[i * dim_delta_prev + j] = value * a;
    }

}


//(BP4) ¦Ä^T * a
__global__ void compute_grad_w_kernel(
    const float* a_prev,   // batch x dim_delta_prev
    const float* delta,    // batch x dim_delta
    float* grad_w,         // dim_delta x dim_delta_prev
    int batch,
    int dim_delta_prev,
    int dim_delta
) {
    int ty = threadIdx.y;
    int tx = threadIdx.x;

    // Working on C[i,j]
    int i = blockDim.y * blockIdx.y + ty;
    int j = blockDim.x * blockIdx.x + tx;

    __shared__ float sh_A[TILE_WIDTH][TILE_WIDTH];
    __shared__ float sh_B[TILE_WIDTH][TILE_WIDTH];

    float value = 0;
    int phaseCount = (batch + TILE_WIDTH - 1) / TILE_WIDTH;

    for (int phase = 0; phase < phaseCount; phase++)
    {
        int A_y = i;
        int A_x = phase * TILE_WIDTH + tx;
        if (A_y < dim_delta && A_x < batch) {
            //¦Ä^ T[A_y,A_x] = ¦Ä[A_x, A_y]
            sh_A[ty][tx] = delta[A_x * dim_delta + A_y];
        }
        else {
            sh_A[ty][tx] = 0;
        }
        int B_x = j;
        int B_y = phase * TILE_WIDTH + ty;
        if (B_y < batch && B_x < dim_delta_prev) {
            sh_B[ty][tx] = a_prev[B_y * dim_delta_prev + B_x];
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

    if (i >= dim_delta || j >= dim_delta_prev) return;

    grad_w[i * dim_delta_prev + j] = value;
}

//BP3
__global__ void compute_grad_b_kernel(
    const float* delta,  // batch * dim_delta
    float* grad_b,       // outdim * delta_dim
    int batch,
    int dim_delta
) {
    int j = blockIdx.x * blockDim.x + threadIdx.x; // neuron
    if (j >= dim_delta) return;

    float sum = 0.0;
    for (int i = 0; i < batch; ++i) {
        sum += delta[i * dim_delta + j];
    }
    grad_b[j] = sum;
}

__global__ void apply_gradien_kernel(
    const float* grad_w, // K * CPQ
    const float* grad_b, // K
    float* w,
    float* b,
    int K,
    int CPQ,
    float learning_rate
) {
    int i = blockDim.y * blockIdx.y + threadIdx.y; //K
    int j = blockDim.x * blockIdx.x + threadIdx.x; //CPQ
    if (i >= K || j >= CPQ) return;

    w[i * CPQ + j] -= learning_rate * grad_w[i * CPQ + j];
    if (j == 0) {
        b[i] -= learning_rate * grad_b[i];
    }
}

/*****************gpt********************************/

struct SoftmaxState {
    float m;  // maximum value
    float d;  // denominator (sum of exponentials)
};

__device__ SoftmaxState reduceOp(SoftmaxState a, SoftmaxState b) {
    SoftmaxState res;
    res.m = fmaxf(a.m, b.m);
    float factor_a = (a.m == -INFINITY) ? 0.0f : __expf(a.m - res.m);
    float factor_b = (b.m == -INFINITY) ? 0.0f : __expf(b.m - res.m);
    res.d = a.d * factor_a + b.d * factor_b;
    return res;
}

__device__ SoftmaxState warpReduceSoftmax(SoftmaxState val) {
#pragma unroll
    for (int offset = 16; offset > 0; offset /= 2) {
        SoftmaxState other;
        other.m = __shfl_down_sync(0xffffffff, val.m, offset);
        other.d = __shfl_down_sync(0xffffffff, val.d, offset);

        val = reduceOp(val, other);
    }
    return val;
}

__global__ void cross_entropy_kernel(
    const float* p,
    const float* y,
    float* loss,
    int batch,
    int total) {

    int j = blockDim.x * blockIdx.x + threadIdx.x;

    float cross = 0;

    int laneId = threadIdx.x & (WARP_SIZE - 1);
    int warpId = threadIdx.x / WARP_SIZE;
    int tid = threadIdx.x;
    int row = blockIdx.x;
    int numWarps = blockDim.x / WARP_SIZE;

    for (int i = threadIdx.x; i < total;  i += blockDim.x) {
        //float p_safe = fmaxf(p[i], 1e-8f);
        cross += ( - logf(p[i]) * y[i]);
    }

    float wrap_sum = warp_reduce_sum(cross);

    __shared__ float shared_m[32]; //make sure block_dim /WRAP_SIZE <= 32
    if (laneId == 0) {
        shared_m[warpId] = wrap_sum;
    }
    __syncthreads();

    if (warpId == 0) {
        float sum = tid < numWarps ? shared_m[tid] : 0;
        sum = warp_reduce_sum(sum);

        if (j == 0) {
            loss[0] = sum / batch;
        }
    }

    return;


}


__global__ void softmax_forward_kernel(
    const float* input, 
    float* output, 
    int M) 
{
    int laneId = threadIdx.x & (WARP_SIZE - 1);
    int warpId = threadIdx.x / WARP_SIZE;
    int tid = threadIdx.x;
    int row = blockIdx.x;
    int numWarps = blockDim.x / WARP_SIZE;
    const float* row_input = input + row * M;
    float* row_output = output + row * M;

    // Each thread processes elements in a strided loop
    SoftmaxState localState = { -INFINITY, 0.0f };
    for (int i = threadIdx.x; i < M; i += blockDim.x) {
        float val = static_cast<float>(row_input[i]);
        float new_m = fmaxf(localState.m, val);
        float factor = __expf(localState.m - new_m);
        localState.d = localState.d * factor + __expf(val - new_m);
        localState.m = new_m;
    }
    // Warp-level reduction ¡ú shared memory ¡ú final reduction
    __shared__ float shared_m[32];  // Max 32 warps per block
    __shared__ float shared_d[32];

    // Each warp reduces its values
    localState = warpReduceSoftmax(localState);
    // First thread in each warp writes to shared memory
    // Try to imagine this in your head; multiple warps 
    // storing their final_state in their first position
    if (laneId == 0) {
        shared_m[warpId] = localState.m;
        shared_d[warpId] = localState.d;
    }
    __syncthreads();

    // First warp reduces all warp results
    if (warpId == 0) {
        SoftmaxState warpState = (tid < numWarps) ?
            SoftmaxState{ shared_m[tid], shared_d[tid] } :
            SoftmaxState{ -INFINITY, 0.0f };

        warpState = warpReduceSoftmax(warpState);

        if (tid == 0) {
            shared_m[0] = warpState.m;
            shared_d[0] = warpState.d;
        }
    }
    __syncthreads();

    // Final normalization pass
    for (int i = threadIdx.x; i < M; i += blockDim.x) {
        float val = row_input[i];
        row_output[i] = __expf(val - shared_m[0]) / shared_d[0];
    }
}

//dL/dz_m = p_m * \sum_j(y_j) - y_m
__global__ void softmax_backward_kernel(
    const float* ylabel,
    const float* ps,
    float* output,
    int N,
    int CHW
) {
    int i = blockIdx.y * blockDim.y + threadIdx.y;
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N || j >= CHW) return;

    //this formula only holds if sum y_label == 1....
    //output[i * CHW + j] = (ps[i * CHW + j] - ylabel[i * CHW + j]);

    //FIX ME ,TODO
    float sum = 0;
    for (int k = 0; k < CHW; ++k) {
        sum += ylabel[i * CHW + k];
    }
    output[i * CHW + j] = (ps[i * CHW + j] * sum - ylabel[i * CHW + j])/N;
}

__global__ void conv_forward_kernel(
    const float* input,      // N * C * H * W -> CRS * NPQ
    const float* weights,    // K * (CRS)
    const float* bias,       // K
    float* output,           // N * K * P * Q
    int batch, int C, int H, int W, int R, int S,
    int strideH, int strideW, int padH, int padW, int K, int P, int Q, float alpha //P,Q are not independent variables
) {
    int ty = threadIdx.y;
    int tx = threadIdx.x;

    // Working on C[i,j]
    int i = blockDim.y * blockIdx.y + ty;
    int j = blockDim.x * blockIdx.x + tx;

    __shared__ float sh_A[TILE_WIDTH][TILE_WIDTH];
    __shared__ float sh_B[TILE_WIDTH][TILE_WIDTH];

    int PQ = P * Q;
    int RS = R * S;

    int middle_dim = C * RS;
    int out_dim = batch * PQ;
    float value = 0;
    int phaseCount = (middle_dim + TILE_WIDTH - 1) / TILE_WIDTH;

    int HW = H * W;
    int CHW = C * HW;
    
    //non-trival index computation, TODO: use bit operation to optimize
    int batchID = j / PQ;
    int beta = j % PQ;
    int p = beta / Q;
    int q = beta % Q;

    for (int phase = 0; phase < phaseCount; phase++)
    {
        int A_y = i;
        int A_x = phase * TILE_WIDTH + tx;
        if (A_y < K && A_x < middle_dim) {
            sh_A[ty][tx] = weights[A_y * middle_dim + A_x];
        }
        else {
            sh_A[ty][tx] = 0;
        }

        int B_x = j;
        int B_y = phase * TILE_WIDTH + ty;
        if (B_y < middle_dim && B_x < out_dim) {

            //non-trival index computation, TODO: use bit operation to optimize
            int channelID = B_y / RS;
            int rs_index = B_y % RS;
            
            int r = rs_index / S;
            int s = rs_index % S;

            //input(batchID, channelID, h=p * strideH -padH + r, w = q * strideW -padW + s) 
            int inner_y = p * strideH -padH + r;
            int inner_x = q * strideW - padW + s;
            int index = batchID * CHW + channelID * HW + inner_y * W + inner_x;
            if (inner_y < 0 || inner_y >= H || inner_x < 0 || inner_x >= W) {
                sh_B[ty][tx] = 0;
            }
            else {
                sh_B[ty][tx] = input[index];
            }
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

    if (i >= K || j >= batch * PQ) {
        return;
    }

    value += bias[i];

    //i (K)  j (NPQ) KNPQ -> NKPQ
    int m = i * PQ + beta;
    // add activation function
    output[batchID * K * PQ + m ] = value > 0.0 ? value : alpha * value;

}

__global__ void conv_dgrad_kernel(
    const float* delta_next, // NKPQ ->  KRS * NHW ¦Ä^{l+1} 
    const float* W_next,     // KCRS  -> C*KRS W^R^{l+1}
    const float* a,          // NCHW -> C*NHW
    float* delta,            // NCHW -> C*NHW   output: ¦Ä^l
    bool add,
    int N,
    int C,
    int H, int W, int P, int Q, int R, int S, int strideH, int strideW, int padH, int padW, int K, int alpha
) {
    int ty = threadIdx.y;
    int tx = threadIdx.x;

    // Working on C[i,j]
    int i = blockDim.y * blockIdx.y + ty;
    int j = blockDim.x * blockIdx.x + tx;

    __shared__ float sh_A[TILE_WIDTH][TILE_WIDTH];
    __shared__ float sh_B[TILE_WIDTH][TILE_WIDTH];

    int c = i;

    int RS = R * S;
    int HW = H * W;
    int CRS = C * RS;
    int KHW = K * HW;
    int PQ = P * Q;
    int KPQ = K * PQ;

    int in_dim = K * RS;
    int out_dim = N * HW;
    float value = 0;
    int phaseCount = (in_dim + TILE_WIDTH - 1) / TILE_WIDTH;


    for (int phase = 0; phase < phaseCount; phase++)
    {
        int A_x = phase * TILE_WIDTH + tx;
        if (c < C && A_x < in_dim) {  
            int k = A_x / RS;
            int rs = A_x % (RS);
            sh_A[ty][tx] = W_next[k * CRS + c * RS + rs];
        }
        else {
            sh_A[ty][tx] = 0;
        }

        int B_x = j;
        int B_y = phase * TILE_WIDTH + ty;
        if (B_y < in_dim && B_x < out_dim) {
            int krs = B_y;
            int k = krs / RS;
            int rs = krs % RS;
            int r = rs / S;
            int s = rs % S;
            int n = B_x / HW;
            int hw = B_x % HW;
            int h = hw / W;
            int w = hw % W;
            int inner_y = (h- r + padH)/strideH;
            int inner_x = (w - s + padW) / strideW;
            bool ok = ((h - r + padH) == inner_y * strideH) && ((w - s + padW) == inner_x * strideW)
                && (inner_y >=0 && inner_y <P ) && (inner_x >=0 && inner_x < Q);
            if (ok) {
                sh_B[ty][tx] = delta_next[n * KPQ + k * PQ + inner_y * Q + inner_x];
            }
            else {
                sh_B[ty][tx] = 0;
            }
            
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

    if (i >= C || j >= N * HW) {
        return;
    }

    int n = j / HW;
    int hw = j % HW;

    if (add) {
        delta[n * C * HW + c * HW + hw] += ( a[n * C * HW + c * HW + hw] > 0 ? value : (value * alpha));
    }
    else {
        delta[n * C * HW + c * HW + hw] = a[n * C * HW + c * HW + hw] > 0 ? value : (value * alpha);
    }
    
}

__global__ void conv_bgrad_test(
    const float* delta,    //NCHW -> C * NHW 
    float* grad_b,
    int N, int C, int H, int W
) {
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i >= C) return;
    int c = i;

    int HW = H * W;
    int NHW = N * HW;
    float value = 0;
    for (int j = 0; j < NHW; ++j) {
        int n = j / HW;
        int hw = j % HW;
        value += delta[n * C * HW + c * HW + hw];
    }
    grad_b[c] = value;
}

__global__ void conv_bgrad_kernel( //[C * NHW] * [NHW  *1]
    const float* delta,    //NCHW -> C * NHW 
    float* grad_b,         //C
    int N, int C, int H, int W
) {
    int ty = threadIdx.y;
    int tx = threadIdx.x;

    // Working on C[i,j]
    int i = blockDim.y * blockIdx.y + ty;
    int j = blockDim.x * blockIdx.x + tx;

    __shared__ float sh_A[TILE_WIDTH][TILE_WIDTH];
    __shared__ float sh_B[TILE_WIDTH][TILE_WIDTH];

    int HW = H * W;
    int middle_dim = N * HW;

    float value = 0;
    int phaseCount = (middle_dim + TILE_WIDTH - 1) / TILE_WIDTH;

    for (int phase = 0; phase < phaseCount; phase++) {

        int A_y = i;
        int A_x = phase * TILE_WIDTH + tx;
        if (A_y < C && A_x < middle_dim) {
            int n = A_x / (HW);
            int hw = A_x % HW;
            sh_A[ty][tx] = delta[n * C * HW + i * HW + hw];
        }
        else {
            sh_A[ty][tx] = 0;
        }

        int B_x = j;
        int B_y = phase * TILE_WIDTH + ty;
        if (B_y < middle_dim && B_x < 1) {
            sh_B[ty][tx] = 1;
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
    if (i >= C || j >= 1){
        return;
    }
    grad_b[i] = value;
}

__global__ void conv_wgrad_kernel(
    const float* delta, //NKPQ -> K * NPQ
    const float* a_prev, //NCHW -> NPQ * CRS
    float* grad_w, //KCRS -> K * CRS
    int N, int K, int C, int H, int W, int P, int Q, int R, int S, int strideH, int strideW, int padH, int padW
) {
    int ty = threadIdx.y;
    int tx = threadIdx.x;

    // Working on C[i,j]
    int i = blockDim.y * blockIdx.y + ty;
    int j = blockDim.x * blockIdx.x + tx;

    __shared__ float sh_A[TILE_WIDTH][TILE_WIDTH];
    __shared__ float sh_B[TILE_WIDTH][TILE_WIDTH];

    int PQ = P * Q;
    int RS = R * S;
    int HW = H * W;
    int in_dim = N * PQ;
    int out_dim = C * RS;

    float value = 0;
    int phaseCount = (in_dim + TILE_WIDTH - 1) / TILE_WIDTH;

    for (int phase = 0; phase < phaseCount; phase++) {

        int A_y = i;
        int A_x = phase * TILE_WIDTH + tx;
        if (A_y < K && A_x < in_dim) {
            int k = i;
            int n = A_x / (PQ);
            int pq = A_x % PQ;
            sh_A[ty][tx] = delta[n * (K*PQ) + k * PQ + pq];
        }
        else {
            sh_A[ty][tx] = 0;
        }

        int B_x = j;
        int B_y = phase * TILE_WIDTH + ty;
        if (B_y < in_dim && B_x < out_dim) {
            int n = B_y / (PQ);
            int c = B_x / (RS);
            int rs = B_x % RS;
            int r = rs / S;
            int s = rs % S;
            int pq = B_y % PQ;
            int p = pq / Q;
            int q = pq % Q;
            int inner_y = r + p * strideH - padH;
            int inner_x = s + q * strideW - padW;
            bool ok = inner_y >= 0 && inner_y < H && inner_x >= 0 && inner_x < W;
            if (ok) {
                sh_B[ty][tx] = a_prev[n * C * HW + c * HW + inner_y * W + inner_x];
            }
            else {
                sh_B[ty][tx] = 0;
            }
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

    if (i >= K || j >= out_dim) return;

    grad_w[i * out_dim + j] = value;
}