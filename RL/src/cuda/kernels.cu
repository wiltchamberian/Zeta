#include "kernels.h"
#include <cuda_runtime.h>



//input:X, W
//outupt := sigma(X * W^T)

__global__ void linear_leaky_relu_forward_kernel(
    const double* input,      // batch x in_dim
    const double* weights,    // out_dim x in_dim
    const double* bias,       // out_dim
    double* output,           // batch x out_dim
    int batch, int in_dim, int out_dim,
    double alpha
) {
    int by = blockIdx.y;
    int bx = blockIdx.x;

    int ty = threadIdx.y;
    int tx = threadIdx.x;

    int i = blockDim.y * by + ty;
    int j = blockDim.x * bx + tx;

    __shared__ double sh_A[TILE_WIDTH][TILE_WIDTH];
    __shared__ double sh_B[TILE_WIDTH][TILE_WIDTH];

    double value = 0;
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


__global__ void leaky_relu_forward_kernel(
    const double* input,
    double* output,
    int total_elements,
    double alpha
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx >= total_elements) return;

    double x = input[idx];
    output[idx] = x > 0.0 ? x : alpha * x;
}


__global__ void mse_loss_kernel(
    const double* a,       // batch x out_dim, a^L
    const double* y,       // batch x out_dim
    double* delta,         // batch x out_dim Êä³ö ¦Ä^L
    int batch,
    int out_dim
) {
    int j = blockIdx.x * blockDim.x + threadIdx.x; // neuron
    int i = blockIdx.y * blockDim.y + threadIdx.y; // sample

    if (i >= batch || j >= out_dim) return;

    int idx = i * out_dim + j;
    delta[idx] = (a[idx] - y[idx]);
}

//(BP2) ¦Ä^l = (¦Ä^{l+1} ¡¤ W^{l+1}) ¡Ñ ¦Ò'(z^l)
__global__ void linear_leaky_relu_backward_kernel(
    const double* delta_next, // batch x dim_delta_next ¦Ä^{l+1}
    const double* W_next,     // dim_delta_next x dim_delta W^{l+1}
    const double* a,          // batch x dim_delta a^l
    double* delta,            // batch x dim_delta Êä³ö ¦Ä^l
    int batch,
    int dim_delta,
    int dim_delta_next,
    double alpha              // LeakyReLU alpha
) {
    int by = blockIdx.y;
    int bx = blockIdx.x;

    int ty = threadIdx.y;
    int tx = threadIdx.x;

    int i = blockDim.y * by + ty;
    int j = blockDim.x * bx + tx;

    __shared__ double sh_A[TILE_WIDTH][TILE_WIDTH];
    __shared__ double sh_B[TILE_WIDTH][TILE_WIDTH];

    double value = 0.0;
    int phaseCount = (dim_delta_next + TILE_WIDTH - 1) / TILE_WIDTH;

    for (int phase = 0; phase < phaseCount; phase++)
    {
        int A_y = i;
        int A_x = phase * TILE_WIDTH + tx;
        sh_A[ty][tx] = (A_y < batch && A_x < dim_delta_next) ? delta_next[A_y * dim_delta_next + A_x] : 0.0;

        int B_x = j;
        int B_y = phase * TILE_WIDTH + ty;
        sh_B[ty][tx] = (B_y < dim_delta_next && B_x < dim_delta) ? W_next[B_y * dim_delta + B_x] : 0.0;

        __syncthreads();

        for (int k = 0; k < TILE_WIDTH; k++) {
            value += sh_A[ty][k] * sh_B[k][tx];
        }

        __syncthreads();
    }

    if (i >= batch || j >= dim_delta) return;

    // Fused LeakyReLU backward
    double der = a[i * dim_delta + j] > 0.0 ? 1.0 : alpha;
    delta[i * dim_delta + j] = value * der;
}


//(BP4) ¦Ä^T * a
__global__ void compute_grad_w_kernel(
    const double* a_prev,   // batch x dim_delta_prev
    const double* delta,    // batch x dim_delta
    double* grad_w,         // dim_delta x dim_delta_prev
    int batch,
    int dim_delta_prev,
    int dim_delta
) {
    int by = blockIdx.y;
    int bx = blockIdx.x;

    int ty = threadIdx.y;
    int tx = threadIdx.x;

    // Working on C[i,j]
    int i = blockDim.y * by + ty;
    int j = blockDim.x * bx + tx;

    __shared__ double sh_A[TILE_WIDTH][TILE_WIDTH];
    __shared__ double sh_B[TILE_WIDTH][TILE_WIDTH];

    double value = 0;
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

    grad_w[i * dim_delta_prev + j] = value / batch;
}

//BP3
__global__ void compute_grad_b_kernel(
    const double* delta,  // batch x dim_delta
    double* grad_b,       // outdim_delta_dim
    int batch,
    int dim_delta
) {
    int j = blockIdx.x * blockDim.x + threadIdx.x; // neuron
    if (j >= dim_delta) return;

    double sum = 0.0;
    for (int i = 0; i < batch; ++i) {
        sum += delta[i * dim_delta + j];
    }
    grad_b[j] = sum / batch;
}

__global__ void apply_gradien_kernel(
    const double* grad_w, // dim_y x dim_x
    const double* grad_b, // dim_y
    double* w,
    double* b,
    int dim_y,
    int dim_x,
    double learning_rate
) {
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    int j = blockDim.y * blockIdx.y + threadIdx.y;
    if (i >= dim_y || j >= dim_x) return;

    w[i * dim_y + j] -= learning_rate * grad_w[i * dim_y + j];
    if (j == 0) {
        b[i] -= learning_rate * grad_b[i];
    }
}

__global__ void conv_forward_kernel(
    const double* input,      // N * C * H * W
    const double* weights,    // K * (CRS)
    const double* bias,       // K
    double* output,           // N * K * P * Q
    int batch, int C, int H, int W, int R, int S,
    int strideH, int strideW, int K, int P, int Q, float alpha //P,Q are not independent variables
) {
    int by = blockIdx.y;
    int bx = blockIdx.x;

    int ty = threadIdx.y;
    int tx = threadIdx.x;

    // Working on C[i,j]
    int i = blockDim.y * by + ty;
    int j = blockDim.x * bx + tx;

    __shared__ double sh_A[TILE_WIDTH][TILE_WIDTH];
    __shared__ double sh_B[TILE_WIDTH][TILE_WIDTH];

    int PQ = P * Q;
    int RS = R * S;

    int in_dim = C * RS;
    int out_dim = batch * PQ;
    double value = 0;
    int phaseCount = (in_dim + TILE_WIDTH - 1) / TILE_WIDTH;

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
        if (A_y < K && A_x < in_dim) {
            sh_A[ty][tx] = weights[A_y * in_dim + A_x];
        }
        else {
            sh_A[ty][tx] = 0;
        }

        int B_x = j;
        int B_y = phase * TILE_WIDTH + ty;
        if (B_y < in_dim && B_x < out_dim) {

            //non-trival index computation, TODO: use bit operation to optimize
            int channelID = B_y / RS;
            int alpha = B_y % RS;
            
            int r = alpha / S;
            int s = alpha % S;

            //input(batchID, channelID, h=p * strideH + r, w = q * strideW + s) 
            int index = batchID * CHW + channelID * HW + (p*strideH + r) * W + (q* strideW +s);
            sh_B[ty][tx] = input[index];
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
    const double* delta_next, // NKHW ->  KRS * NHW ¦Ä^{l+1} 
    const double* W_next,     // KCRS  -> C*KRS W^{l+1}
    const double* a,          // NCHW -> C*NHW
    double* delta,            // NCHW -> C*NHW   output: ¦Ä^l
    int N,
    int C,
    int H, int W, int R, int S, int strideH, int strideW, int K, int alpha
) {
    int by = blockIdx.y;
    int bx = blockIdx.x;

    int ty = threadIdx.y;
    int tx = threadIdx.x;

    // Working on C[i,j]
    int i = blockDim.y * by + ty;
    int j = blockDim.x * bx + tx;

    __shared__ double sh_A[TILE_WIDTH][TILE_WIDTH];
    __shared__ double sh_B[TILE_WIDTH][TILE_WIDTH];

    int c = i;

    int RS = R * S;
    int HW = H * W;
    int CRS = C * RS;
    int KHW = K * HW;

    int in_dim = K * RS;
    int out_dim = N * HW;
    double value = 0;
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
            if (h - strideH + 1 + r >= 0 && w - strideW + 1 + s >= 0) {
                sh_B[ty][tx] = delta_next[n * KHW + k * HW + (h - strideH + 1 + r) * W + (w - strideW + 1 + s)];
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
    delta[n * C * HW + c * HW + hw] = value * (a[n * C*HW+c*HW + hw]>0?1:alpha);

}

__global__ void conv_bgrad_kernel(
    const double* delta,    //NCHW -> C * NHW 
    double* grad_b,         //C
    int N, int C, int H, int W
) {
    int by = blockIdx.y;
    int bx = blockIdx.x;

    int ty = threadIdx.y;
    int tx = threadIdx.x;

    // Working on C[i,j]
    int i = blockDim.y * by + ty;
    int j = blockDim.x * bx + tx;

    __shared__ double sh_A[TILE_WIDTH][TILE_WIDTH];
    __shared__ double sh_B[TILE_WIDTH][TILE_WIDTH];

    int HW = H * W;
    int in_dim = N * HW;

    double value = 0;
    int phaseCount = (in_dim + TILE_WIDTH - 1) / TILE_WIDTH;

    int c = i;
    for (int phase = 0; phase < phaseCount; phase++) {

        int A_y = i;
        int A_x = phase * TILE_WIDTH + tx;
        if (A_y < C && A_x < in_dim) {
            int n = A_x / (HW);
            int hw = A_x % HW;
            sh_A[ty][tx] = delta[n * C * HW + c * HW + hw];
        }
        else {
            sh_A[ty][tx] = 0;
        }

        int B_x = j;
        int B_y = phase * TILE_WIDTH + ty;
        if (B_y < in_dim && B_x < 1) {
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
    if (c >= C){
        return;
    }
    grad_b[c] = value;
}

__global__ void conv_wgrad_kernel(
    const double* delta, //NKPQ -> K * NPQ
    const double* a_prev, //NCRS -> NPQ * CRS
    double* grad_w, //KCRS -> K * CRS
    int N, int K, int C, int P, int Q, int R, int S
) {
    int by = blockIdx.y;
    int bx = blockIdx.x;

    int ty = threadIdx.y;
    int tx = threadIdx.x;

    // Working on C[i,j]
    int i = blockDim.y * by + ty;
    int j = blockDim.x * bx + tx;

    __shared__ double sh_A[TILE_WIDTH][TILE_WIDTH];
    __shared__ double sh_B[TILE_WIDTH][TILE_WIDTH];

    int PQ = P * Q;
    int RS = R * S;
    int in_dim = N * PQ;
    int out_dim = C * RS;

    double value = 0;
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
            if (p + r < P && q + s < Q) {
                sh_B[ty][tx] = a_prev[n * C * PQ + c * PQ + (p + r) * Q + (q + s)];
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