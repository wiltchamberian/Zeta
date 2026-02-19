#include "cuLayer.h"
#include "kernels.h"
#include <cudnn.h>
#include "cu_tool.h"
#include "device_launch_parameters.h"

/**********************CuLinearLeakyReluLayer*****************************/
void CuLinearLeakyReluLayer::forward() { /* kernel launch */
    float* inputData = prevs.size() > 0 ? prevs[0]->GetActivation() : nullptr;
    if (inputData == nullptr) {
        assert(false);
        return;
    }
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

void CuLinearLeakyReluLayer::backwardEx() {
    dgrad();
    wgrad();
    bgrad();
}

void CuLinearLeakyReluLayer::dgrad() {
    int dim_delta_prev = prevs[0]->GetDeltaSize();
    int dim_delta = GetDeltaSize();;
    dim3 block(TILE_WIDTH, TILE_WIDTH);
    dim3 grid((inputShape.N + block.x - 1) / block.x, (dim_delta + block.y - 1) / block.y);
    linear_leaky_relu_backward_kernel << <grid, block >> > (
        dl.delta,        // δ^{l}
        dl.weights,
        prevs[0]->GetActivation(),       // z^l (或 a^{l-1} 用于 σ'(z))
        prevs[0]->GetDelta(),            // δ^(l-1) 输出
        inputShape.N,
        dim_delta_prev,
        dim_delta,
        alpha
        );
}

void CuLinearLeakyReluLayer::wgrad() {
    dim3 block(TILE_WIDTH, TILE_WIDTH);
    int N = inputShape.N;
    int dim_delta_prev = inputShape.Dim();
    int dim_delta = outputShape.Dim();
    dim3 grid((dim_delta_prev + block.x - 1) / block.x, (dim_delta + block.y - 1) / block.y);
    float* prev_activation = (prevs.size()>0) ? (prevs[0]->GetActivation()) : prevActivation;
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

size_t CuLinearLeakyReluLayer::GetDeviceSize() {
    size_t total = 0;
    size_t w = weights.numel();
    size_t b = this->b.numel();
    total += w;  // weights
    total += w;  // grad_w
    total += b;  // bias
    total += b;  // grad_b
    return total;
}

void CuLinearLeakyReluLayer::BindWorkspace(void* ptr) {
    float* d = reinterpret_cast<float*>(ptr);
    dl.activation = d;
    dl.delta = d + outputShape.NumElements() * sizeof(float);
}

void CuLinearLeakyReluLayer::BindDevice(void* ptr) {
    float* addr = reinterpret_cast<float*>(ptr);

    // -------- weights --------
    Tensor w = weights.contiguous();
    size_t w_size = w.numel();
    dl.weights = reinterpret_cast<float*>(addr);
    dl.w_size = w_size;
    dl.in_dim = w_size / w.shape[0];//layer->weights.shape[1];

    CUDA_CHECK(cudaMemcpy(
        dl.weights,
        w.data(),
        w_size * sizeof(float),
        cudaMemcpyHostToDevice
    ));
    addr += w_size * sizeof(float);

    // -------- bias --------
    Tensor b = b.contiguous();
    size_t b_size = b.numel();
    dl.bias = reinterpret_cast<float*>(addr);
    dl.b_size = b_size;

    CUDA_CHECK(cudaMemcpy(
        dl.bias,
        b.data(),
        b_size * sizeof(float),
        cudaMemcpyHostToDevice
    ));
    addr += b_size * sizeof(float);

    // -------- grad_w --------
    dl.grad_w = reinterpret_cast<float*>(addr);
    addr += w_size * sizeof(float);

    // -------- grad_b --------
    dl.grad_b = reinterpret_cast<float*>(addr);
    addr += b_size * sizeof(float);
}

float* CuLinearLeakyReluLayer::GetActivation() {
    return dl.activation;
}

size_t CuLinearLeakyReluLayer::GetActivationSize() {
    return dl.b_size;
}

float* CuLinearLeakyReluLayer::GetDelta() {
    return dl.delta;
}

size_t CuLinearLeakyReluLayer::GetDeltaSize() {
    return dl.b_size;
}

float* CuLinearLeakyReluLayer::GetPrevActivation() {
    if (prevs[0] != nullptr) {
        return prevs[0]->GetActivation();
    }
    else {
        return prevActivation;
    }
}

void CuSoftmaxCrossEntropyLayer::forward() {
    assert(!prevs.empty());
    constexpr int BLOCK = 1024;
    int M = prevs[0]->GetActivationSize();
    float* prev_activation = prevs[0]->GetActivation();
    softmax_forward_kernel<<<batchSize, BLOCK >> > (prev_activation, activation,M);
}

void CuSoftmaxCrossEntropyLayer::backwardEx() {
    assert(!prevs.empty());
    int M = prevs[0]->GetActivationSize();
    dim3 block(TILE_WIDTH, TILE_WIDTH);
    dim3 grid((M+TILE_WIDTH-1)/TILE_WIDTH, (batchSize + TILE_WIDTH - 1) / TILE_WIDTH);
    softmax_backward_kernel << <grid, block >> > (y, activation,prevs[0]->GetDelta(), batchSize, M);
}

/************************************************/
CuMseLayer::CuMseLayer(int C, int R, int S)
{
    label = Tensor(C, R, S);
}

void CuMseLayer::forward() {

}

void CuMseLayer::backwardEx() {
    assert(!prevs.empty());
    auto prev = prevs[0];

    int out_dim = label.numel() / label.shape[0];
    int batch = inputShape.N;

    dim3 block(TILE_WIDTH, TILE_WIDTH);
    dim3 grid((out_dim + block.x - 1) / block.x, (batch + block.y - 1) / block.y);
    
    mse_loss_kernel << <grid, block >> > (
        prev->GetActivation(),
        y,
        prev->GetDelta(),
        batch,
        out_dim
        );
}

TensorShape CuMseLayer::InferOutputShape(TensorShape shape) {
    TensorShape result;

    result.N = 1;
    result.H = 1;
    result.W = 1;
    result.C = 1;
    return result;
}

size_t CuMseLayer::GetWorkspaceSize() {
    //for softmax result p (a)
    //and crossEntropy p - label (dL/dp)
    return label.numel() * sizeof(float) * 2;
}

void CuMseLayer::BindWorkspace(void* pointer) {
    float* ptr = reinterpret_cast<float*>(pointer);
    p = ptr;
    ptr += label.numel();
    grad = ptr;
    return;
}

size_t CuMseLayer::GetDeviceSize() {
    //for label
    return label.numel() * sizeof(float);
}

void CuMseLayer::BindDevice(void* ptr) {
    y = reinterpret_cast<float*>(ptr);
}

size_t CuMseLayer::GetActivationSize() {
    return label.numel();
}

float* CuMseLayer::GetActivation() {
    return p;
}

float* CuMseLayer::GetDelta() {
    assert(false);
    return nullptr;
}

size_t CuMseLayer::GetDeltaSize() {
    assert(false);
    return 0;
}

/******************************convolution layer*********************************/
CuConvolutionLayer::CuConvolutionLayer(int K, int C, int R, int S)
:padH(1)
,padW(1)
,strideH(1)
,strideW(1)
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

void CuConvolutionLayer::BindDevice(void* ptr) {

    float* addr = reinterpret_cast<float*>(ptr);

    // -------- weights --------
    Tensor w = weights.contiguous();
    size_t w_size = w.numel();
    dl.weights = reinterpret_cast<float*>(addr);
    dl.w_size = w_size;
    dl.in_dim = w_size / w.shape[0];//layer->weights.shape[1];

    CUDA_CHECK(cudaMemcpy(
        dl.weights,
        w.data(),
        w_size * sizeof(float),
        cudaMemcpyHostToDevice
    ));
    addr += w_size * sizeof(float);

    // -------- bias --------
    Tensor b = b.contiguous();
    size_t b_size = b.numel();
    dl.bias = reinterpret_cast<float*>(addr);
    dl.b_size = b_size;

    CUDA_CHECK(cudaMemcpy(
        dl.bias,
        b.data(),
        b_size * sizeof(float),
        cudaMemcpyHostToDevice
    ));
    addr += b_size * sizeof(float);

    // -------- grad_w --------
    dl.grad_w = reinterpret_cast<float*>(addr);
    addr += w_size * sizeof(float);

    // -------- grad_b --------
    dl.grad_b = reinterpret_cast<float*>(addr);
    addr += b_size * sizeof(float);
    
}

size_t CuConvolutionLayer::GetDeviceSize() {
    size_t total = 0;
    size_t w = weights.numel();
    size_t b = this->b.numel();
    total += w;  // weights
    total += w;  // grad_w
    total += b;  // bias
    total += b;  // grad_b
    return total;
}

size_t CuConvolutionLayer::GetActivationSize() { 
    return dl.b_size;
}

float* CuConvolutionLayer::GetActivation() {
    return dl.activation;
}

size_t CuConvolutionLayer::GetDeltaSize() {
    return dl.b_size;
}

float* CuConvolutionLayer::GetDelta() {
    return dl.delta;
}

void CuConvolutionLayer::forward() {
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

    float* input_ptr = prevs.size()>0 ? prevs[0]->GetActivation() : prevActivation;

    dim3 grid((NPQ + TILE_WIDTH - 1) / TILE_WIDTH, ( K + TILE_WIDTH - 1) / TILE_WIDTH);
    conv_forward_kernel <<<grid, block >> > (
        input_ptr,
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
    
    //assert(nexts.size()>0);
    //auto next = nexts[0];
    //int N = next->outputShape.N;
    //int K = next->outputShape.C;
    //int P = next->outputShape.H;
    //int Q = next->outputShape.W;

    //int C = next->weights.shape[1];
    //int R = next->weights.shape[2];
    //int S = next->weights.shape[3];
    //int H = outputShape.H;
    //int W = outputShape.W;
    //int NHW = N * H * W;

    //dim3 block(TILE_WIDTH, TILE_WIDTH);
    //dim3 grid((NHW + TILE_WIDTH - 1) / TILE_WIDTH, (C + TILE_WIDTH - 1) / TILE_WIDTH);
    //conv_dgrad_kernel<<<grid,block>>>(delta_next,
    //    w_next,
    //    dl.activation,
    //    dl.delta,
    //    N,
    //    C,
    //    H, W, P, Q, R, S, strideH, strideW, padH, padW, K, alpha);//?
}

void CuConvolutionLayer::backwardEx() {
    dgrad();
    wgrad();
    bgrad();
}

void CuConvolutionLayer::dgrad() {
    /*
    const float* delta, // NKPQ ->  KRS * NHW δ^{l}
    const float* W,     // KCRS  -> C*KRS W^{l}
    const float* a_prev,          // NCHW -> C*NHW
    float* delta_prev,       // NCHW -> C*NHW   output: δ^(l-1)
    int N,
    int C,
    int H, int W, int P, int Q, int R, int S, int strideH, int strideW, int K, int alpha*/
    if (prevs.empty()) {
        return;
    }
    auto prev = prevs[0];
    
    int N = outputShape.N;
    int K = outputShape.C;
    int P = outputShape.H;
    int Q = outputShape.W;

    int C = weights.shape[1];
    int R = weights.shape[2];
    int S = weights.shape[3];
    int H = prevs[0]->outputShape.H;
    int W = prevs[0]->outputShape.W;
    int NHW = N * H * W;

    dim3 block(TILE_WIDTH, TILE_WIDTH);
    dim3 grid((NHW + TILE_WIDTH - 1) / TILE_WIDTH, (C + TILE_WIDTH - 1) / TILE_WIDTH);
    conv_dgrad_kernel << <grid, block >> > (dl.delta,
        dl.weights,
        prev->GetActivation(),
        prev->GetDelta(),
        N,
        C,
        H, W, P, Q, R, S, strideH, strideW, padH, padW, K, alpha);

    
}

void CuConvolutionLayer::wgrad() {
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

    float* prev_activation = prevs.size() > 0 ? prevs[0]->GetActivation() : prevActivation;
    
    dim3 block(TILE_WIDTH, TILE_WIDTH);
    dim3 grid((CRS + TILE_WIDTH - 1) / TILE_WIDTH, (K + TILE_WIDTH - 1) / TILE_WIDTH);
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
    dim3 grid(1 , (C + TILE_WIDTH - 1) / TILE_WIDTH);
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

