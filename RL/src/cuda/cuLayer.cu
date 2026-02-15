#include "cuLayer.h"
#include "kernels.h"
#include <cudnn.h>


/**********************CuLinearLeakyReluLayer*****************************/
void CuLinearLeakyReluLayer::forward(const float* inputData) { /* kernel launch */

    int out_dim = dl.w_size / dl.in_dim;
    dim3 block(TILE_WIDTH, TILE_WIDTH);
    dim3 grid((out_dim + block.x - 1) / block.x, (inputShape.N + block.y - 1) / block.y);

    linear_leaky_relu_forward_kernel << <grid, block >> > (
        inputData,
        dl.weights,
        dl.bias,
        dl.activation,
        inputShape.N, dl.in_dim, out_dim,
        alpha
        );
}

void CuLinearLeakyReluLayer::backward(const float* delta_next, const float* w_next) {
    int dim_delta = dl.in_dim;
    int dim_delta_next = dl.b_size;
    dim3 block(TILE_WIDTH, TILE_WIDTH);
    dim3 grid((inputShape.N + block.x - 1) / block.x, (dim_delta + block.y - 1) / block.y);
    linear_leaky_relu_backward_kernel << <grid, block >> > (
        delta_next,        // δ^{l+1}
        w_next,
        dl.activation,       // z^l (或 a^{l-1} 用于 σ'(z))
        dl.delta,            // δ^l 输出
        inputShape.N,
        dim_delta,
        dim_delta_next,
        alpha
        );
}

void CuLinearLeakyReluLayer::wgrad(const float* prev_activation) {
    dim3 block(TILE_WIDTH, TILE_WIDTH);
    int N = inputShape.N;
    int dim_delta_prev = inputShape.Dim();
    int dim_delta = outputShape.Dim();
    dim3 grid((dim_delta_prev + block.x - 1) / block.x, (dim_delta + block.y - 1) / block.y);
    compute_grad_w_kernel << <grid, block >> > (
        prev_activation,
        dl.delta,
        dl.grad_w,
        N,
        dim_delta_prev,
        dim_delta
        );

}

void CuLinearLeakyReluLayer::bgrad() {
    dim3 block(TILE_WIDTH, TILE_WIDTH);
    int dim_delta = outputShape.Dim();
    dim3 grid((dim_delta + block.x - 1) / block.x);
    compute_grad_b_kernel << <grid, block.x >> > (
        dl.delta/*ws.deltas[l]*/,
        dl.grad_b/*deviceLayers[l].grad_b*/,
        inputShape.N,
        dim_delta
        );
}

TensorShape CuLinearLeakyReluLayer::InferOutputShape(TensorShape shape) {
    TensorShape result;
    result.N = shape.N;
    result.C = weights.shape[0];
    result.H = 1;
    result.W = 1;
    this->inputShape = shape;
    this->outputShape = result;
    return result;
}

size_t CuLinearLeakyReluLayer::GetWorkspaceSize() {
    return outputShape.NumElements() * 2 * sizeof(float);
}

void CuLinearLeakyReluLayer::BindWorkspace(void* ptr) {
    float* d = reinterpret_cast<float*>(ptr);
    dl.activation = d;
    dl.delta = d + outputShape.NumElements() * sizeof(float);
}

/******************************convolution layer*********************************/
CuConvolutionLayer::CuConvolutionLayer(int K, int C, int R, int S)
{
    weights = Tensor(K, C, R, S);
    b = Tensor(K);
}

TensorShape CuConvolutionLayer::InferOutputShape(TensorShape shape) {
    TensorShape result;
    result.N = shape.N;
    result.C = weights.shape[0];
    result.H = (shape.H + padH * 2 - weights.shape[2]) / strideH + 1;
    result.W = (shape.W + padW * 2 - weights.shape[3]) / strideW + 1;
    this->inputShape = shape;
    this->outputShape = result;
    return result;
}

size_t CuConvolutionLayer::GetWorkspaceSize() {
    size_t siz = 0;
    siz += outputShape.NumElements() * sizeof(float) * 2;
    return siz;
}

void CuConvolutionLayer::BindWorkspace(void* ptr) {
    float* data = reinterpret_cast<float*>(ptr);
    dl.activation = data;
    dl.delta = data + outputShape.NumElements() * sizeof(float);
}

void CuConvolutionLayer::forward(const float* input) {
    /*
    const float* input,      // N * C * H * W -> (CRS) * NPQ
    const float* weights,    // K * (CRS)
    const float* bias,       // K
    float* output,           // N * K * P * Q
    int batch, int C, int H, int W, int R, int S,
    int strideH, int strideW, int K, int P, int Q, float alpha
    */

    dim3 block(TILE_WIDTH, TILE_WIDTH);
    int C = inputShape.C;
    int K = weights.shape[0];
    int R = weights.shape[2];
    int S = weights.shape[3];
    int CRS = C * R * S;
    int NPQ = inputShape.N * outputShape.H * outputShape.W;
    dim3 grid((K + TILE_WIDTH - 1) / TILE_WIDTH, (NPQ + TILE_WIDTH - 1) / TILE_WIDTH);
    conv_forward_kernel <<<grid, block >> > (
        input, 
        dl.weights, 
        dl.bias, 
        dl.activation, 
        inputShape.N, inputShape.C, inputShape.H, inputShape.W, R, S,
        strideH, strideW, padH, padW,  K, outputShape.H, outputShape.W, alpha);
}

void CuConvolutionLayer::backward(const float* delta_next, const float* w_next) {
    /*
    const float* delta_next, // NKPQ ->  KRS * NHW δ^{l+1}
    const float* W_next,     // KCRS  -> C*KRS W^R^{l+1}
    const float* a,          // NCHW -> C*NHW
    float* delta,            // NCHW -> C*NHW   output: δ^l
    int N,
    int C,
    int H, int W, int P, int Q, int R, int S, int strideH, int strideW, int K, int alpha*/
    assert(next != nullptr);
    int N = next->outputShape.N;
    int K = next->outputShape.C;
    int P = next->outputShape.H;
    int Q = next->outputShape.W;

    int C = next->weights.shape[1];
    int R = next->weights.shape[2];
    int S = next->weights.shape[3];
    int H = outputShape.H;
    int W = outputShape.W;
    int NHW = N * H * W;

    dim3 block(TILE_WIDTH, TILE_WIDTH);
    dim3 grid((C + TILE_WIDTH - 1) / TILE_WIDTH, (NHW + TILE_WIDTH - 1) / TILE_WIDTH);
    conv_dgrad_kernel<<<grid,block>>>(delta_next,
        w_next,
        dl.activation,
        dl.delta,
        N,
        C,
        H, W, P, Q, R, S, strideH, strideW, padH, padW, K, alpha);
}

void CuConvolutionLayer::wgrad(const float* prev_activation) {
    /*
    const float* delta, //NKPQ -> K * NPQ
    const float* a_prev, //NCHW -> NPQ * CRS
    float* grad_w, //KCRS -> K * CRS
    int N, int K, int C, int H, int W, int P, int Q, int R, int S, int strideH, int strideW, int padH, int padW
    */
    int N = inputShape.N;
    int K = outputShape.C;
    int P = outputShape.H;
    int Q = outputShape.W;
    int C = inputShape.C;
    int H = inputShape.H;
    int W = inputShape.W;
    int R = weights.shape[2];
    int S = weights.shape[3];
    int CRS = C * R * S;
    dim3 block(TILE_WIDTH, TILE_WIDTH);
    dim3 grid((K + TILE_WIDTH - 1) / TILE_WIDTH, (CRS + TILE_WIDTH - 1) / TILE_WIDTH);
    conv_wgrad_kernel << <grid, block >> > (
        dl.delta, //NKPQ -> K * NPQ
        prev_activation, //NCRS -> NPQ * CRS
        dl.grad_w, //KCRS -> K * CRS
        N, K, C, H, W, P, Q, R, S, strideH, strideW, padH, padW
        );
}

void CuConvolutionLayer::bgrad() {
    /*
    conv_bgrad_kernel(       //NHW * 1 I
    const float* delta,    //NCHW -> C * NHW 
    float* grad_b,         //C
    int N, int C, int H, int W
    ) */
    int N = inputShape.N;
    int C = outputShape.C;
    int H = outputShape.H;
    int W = outputShape.W;

    dim3 block(TILE_WIDTH, TILE_WIDTH);
    dim3 grid((C + TILE_WIDTH - 1) / TILE_WIDTH, 1);
    conv_bgrad_kernel<<<grid, block>>>(
        dl.delta,    //NCHW -> C * NHW 
        dl.grad_b,         //C
        N, C, H, W
    );

    //dim3 block(TILE_WIDTH);
    //dim3 grid((C + TILE_WIDTH - 1) / TILE_WIDTH);
    //conv_bgrad_test << <grid, block >> > (
    //    dl.delta,
    //    dl.grad_b,
    //    N, C, H, W
    //    );
}

