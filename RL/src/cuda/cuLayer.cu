#include "cuLayer.h"
#include "kernels.h"
#include <cudnn.h>
#include "cu_tool.h"
#include "device_launch_parameters.h"
#include "CuNN.h"

/**********************CuLinearLeakyReluLayer*****************************/
CuLinearLeakyReluLayer::CuLinearLeakyReluLayer(int input, int output)
    :in_dim(input)
    , out_dim(output) {
    weights = Tensor(output, input); //reverse order to level up computation performance
    b = Tensor(output);
}

void CuLinearLeakyReluLayer::forward() { /* kernel launch */
    float* inputData = prevs.size() > 0 ? prevs[0]->GetActivation() : prevActivation;
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
        false,
        inputShape.N,
        dim_delta,
        dim_delta_next,
        alpha
        );
}

void CuLinearLeakyReluLayer::backwardEx() {
    add = false;
    dgrad();
    wgrad();
    bgrad();
}

void CuLinearLeakyReluLayer::applyGradient() {
    int CPQ = weights.numel() / weights.shape[0];
    int K = weights.shape[0];
    int block_y = (K + TILE_WIDTH - 1) / TILE_WIDTH;
    int block_x = (CPQ + TILE_WIDTH - 1) / TILE_WIDTH;
    dim3 grid(block_x, block_y);
    dim3 block(TILE_WIDTH, TILE_WIDTH);
    apply_gradien_kernel << <grid, block >> > (dl.grad_w, dl.grad_b, dl.weights, dl.bias, K, CPQ, nn->learningRate);

}

void CuLinearLeakyReluLayer::dgrad() {
    if (prevs.empty()) {
        return;
    }
    int dim_delta_prev = inputShape.Dim();//prevs[0]->GetDeltaSize();
    int dim_delta = GetDeltaSize();
    float* prev_activation = prevs[0]->GetActivation();
    dim3 block(TILE_WIDTH, TILE_WIDTH);
    dim3 grid((dim_delta_prev + block.x - 1) / block.x, (inputShape.N + block.y - 1) / block.y);
    linear_leaky_relu_backward_kernel << <grid, block >> > (
        dl.delta,        // δ^{l}
        dl.weights,
        prev_activation,       // z^l (或 a^{l-1} 用于 σ'(z))
        prevs[0]->GetDelta(),            // δ^(l-1) 输出
        prevs[0]->add,
        inputShape.N,
        dim_delta_prev,
        dim_delta,
        prevs[0]->GetAlpha()   //must use prev's alpha, pay attention
        );
    prevs[0]->add = true;
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

void CuLinearLeakyReluLayer::InferOutputShape(TensorShape networkInput) {
    TensorShape shape = prevs.empty() ? networkInput : prevs[0]->outputShape;
    TensorShape result;
    result.N = shape.N;
    result.C = weights.shape[0];
    result.H = 1;
    result.W = 1;
    this->inputShape = shape;
    this->outputShape = result;
    return;
}

size_t CuLinearLeakyReluLayer::GetWorkspaceSize() {
    return outputShape.NumElements() * 2 ;
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
    dl.delta = d + outputShape.NumElements();
}

void CuLinearLeakyReluLayer::BindDevice(void* ptr) {
    char* addr = reinterpret_cast<char*>(ptr);

    // -------- weights --------
    Tensor w = weights.contiguous();
    size_t w_size = w.numel();
    dl.weights = reinterpret_cast<float*>(addr);
    dl.w_size = w_size;
    dl.in_dim = w_size / w.shape[0];//layer->weights.shape[1];

    if (w.data()) {
        CUDA_CHECK(cudaMemcpy(
            dl.weights,
            w.data(),
            w_size * sizeof(float),
            cudaMemcpyHostToDevice
        ));
        
    }
    addr += w_size * sizeof(float);

    // -------- bias --------
    Tensor bias = b.contiguous();
    size_t b_size = bias.numel();
    dl.bias = reinterpret_cast<float*>(addr);
    dl.b_size = b_size;

    if (bias.data()) {
        CUDA_CHECK(cudaMemcpy(
            dl.bias,
            bias.data(),
            b_size * sizeof(float),
            cudaMemcpyHostToDevice
        ));
    }
    
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

void CuLinearLeakyReluLayer::Print() {

    std::cout << "weights:\n";
    weights.print("W_");
    std::cout << "biases:\n";
    b.print("B_");
    std::cout << std::endl;
}

void CuLinearLeakyReluLayer::PrintGrad() {

    std::cout << "weights_grad:\n";
    weights_grad.print("W_");
    std::cout << "bias_grad:\n";
    bias_grad.print("B_");
}

void CuLinearLeakyReluLayer::FetchResultToCpu() {
    weights.zeros(dl.b_size, dl.in_dim);
    b.zeros(dl.b_size);
    CUDA_CHECK(cudaMemcpy(weights.data(), dl.weights, dl.w_size * sizeof(float), cudaMemcpyKind::cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(b.data(), dl.bias, dl.b_size * sizeof(float), cudaMemcpyKind::cudaMemcpyDeviceToHost));
}

void CuLinearLeakyReluLayer::FetchGradToCpu() {
    weights_grad.zeros(dl.b_size, dl.in_dim);
    bias_grad.zeros(dl.b_size);
    CUDA_CHECK(cudaMemcpy(weights_grad.data(), dl.grad_w, dl.w_size * sizeof(float), cudaMemcpyKind::cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(bias_grad.data(), dl.grad_b, dl.b_size * sizeof(float), cudaMemcpyKind::cudaMemcpyDeviceToHost));
}

/// <summary>
/// SoftmaxEntropyLayer
/// </summary>
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

void CuSoftmaxCrossEntropyLayer::applyGradient() {

}

void CuSoftmaxCrossEntropyLayer::InferOutputShape(TensorShape networkInput) {
    TensorShape shape = prevs.empty() ? networkInput : prevs[0]->outputShape;
    TensorShape result;
    result.N = 1;
    result.C = 1;
    result.H = 1;
    result.W = 1;

    this->inputShape = shape;
    this->outputShape = result;
    return;
}

/************************************************/
CuMseLayer::CuMseLayer(int C) {
    label = Tensor(C);
}

CuMseLayer::CuMseLayer(int C, int H) {
    label = Tensor(C, H);
}

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
        y_label,
        prev->GetDelta(),
        batch,
        out_dim
        );

    //test
    cudaDeviceSynchronize();
    std::vector<float> test(10);
    cudaMemcpy(test.data()+1, prev->GetActivation(), 4, cudaMemcpyKind::cudaMemcpyDeviceToHost);
    cudaMemcpy(test.data()+2, y_label, 4, cudaMemcpyKind::cudaMemcpyDeviceToHost);
    cudaMemcpy(test.data(), prev->GetDelta(), 4, cudaMemcpyKind::cudaMemcpyDeviceToHost);
}

void CuMseLayer::applyGradient() {

}

void CuMseLayer::InferOutputShape(TensorShape networkInput) {
    TensorShape shape = prevs.empty() ? networkInput : prevs[0]->outputShape;
    TensorShape result;
    result.N = 1;
    result.H = 1;
    result.W = 1;
    result.C = 1;
    this->inputShape = shape;
    this->outputShape = result;
    return;
}

size_t CuMseLayer::GetWorkspaceSize() {
    //for softmax result p (a)
    //and crossEntropy p - label (dL/dp)
    return label.numel() *  2;
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
    return label.numel();
}

void CuMseLayer::BindDevice(void* ptr) {
    y_label = reinterpret_cast<float*>(ptr);
    CUDA_CHECK(cudaMemcpy(y_label, label.data(), label.numel() * sizeof(float), cudaMemcpyHostToDevice));
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

void CuMseLayer::FetchPredYToCpu() {
    int d = inputShape.NumElements();
    predY.zeros(inputShape.N, inputShape.Dim());
    CUDA_CHECK(cudaMemcpy(predY.data(), prevs[0]->GetActivation(), d * sizeof(float), cudaMemcpyKind::cudaMemcpyDeviceToHost));
}

void CuMseLayer::PrintPredY() {
    predY.print("predY_");
}

void CuMseLayer::FetchResultToCpu() {
    
}

void CuMseLayer::Print() {


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

void CuConvolutionLayer::InferOutputShape(TensorShape networkInput) {
    TensorShape shape = prevs.empty() ? networkInput : prevs[0]->outputShape;
    TensorShape result;
    result.N = shape.N;
    result.C = weights.shape[0];
    result.H = (shape.H + padH * 2 - weights.shape[2]) / strideH + 1;
    result.W = (shape.W + padW * 2 - weights.shape[3]) / strideW + 1;
    this->inputShape = shape;
    this->outputShape = result;
    return ;
}

size_t CuConvolutionLayer::GetWorkspaceSize() {
    size_t siz = 0;
    siz += outputShape.NumElements() * 2;
    return siz;
}

void CuConvolutionLayer::BindWorkspace(void* ptr) {
    float* data = reinterpret_cast<float*>(ptr);
    dl.activation = data;
    dl.delta = data + outputShape.NumElements();
}

void CuConvolutionLayer::BindDevice(void* ptr) {

    char* addr = reinterpret_cast<char*>(ptr);

    // -------- weights --------
    Tensor w = weights.contiguous();
    size_t w_size = w.numel();
    dl.weights = reinterpret_cast<float*>(addr);
    dl.w_size = w_size;
    dl.in_dim = w_size / w.shape[0];//layer->weights.shape[1];

    if (w.data()) {
        CUDA_CHECK(cudaMemcpy(
            dl.weights,
            w.data(),
            w_size * sizeof(float),
            cudaMemcpyHostToDevice
        ));
    }
    
    addr += w_size * sizeof(float);

    // -------- bias --------
    Tensor b = b.contiguous();
    size_t b_size = b.numel();
    dl.bias = reinterpret_cast<float*>(addr);
    dl.b_size = b_size;

    if (b.data()) {
        CUDA_CHECK(cudaMemcpy(
            dl.bias,
            b.data(),
            b_size * sizeof(float),
            cudaMemcpyHostToDevice
        ));
    }
    
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
    add = false;
    dgrad();
    wgrad();
    bgrad();
}

void CuConvolutionLayer::applyGradient() {

    int CPQ = weights.numel() / weights.shape[0];
    int K = weights.shape[0];
    int block_y = (K + TILE_WIDTH - 1) / TILE_WIDTH;
    int block_x = (CPQ + TILE_WIDTH - 1) / TILE_WIDTH;
    dim3 grid(block_x, block_y);
    dim3 block(TILE_WIDTH, TILE_WIDTH);
    apply_gradien_kernel <<<grid, block>>>(dl.grad_w, dl.grad_b, dl.weights, dl.bias, K, CPQ, nn->learningRate);

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
    conv_dgrad_kernel << <grid, block >> > (
        dl.delta,
        dl.weights,
        prev->GetActivation(),
        prev->GetDelta(),
        prev->add,
        N,
        C,
        H, W, P, Q, R, S, strideH, strideW, padH, padW, K, prev->GetAlpha());
    prev->add = true;
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

void CuConvolutionLayer::Print() {
    std::cout << "weights:\n";
    weights.print("W_");
    std::cout << "biases:\n";
    b.print("B_");
    std::cout << std::endl;
}

void CuConvolutionLayer::PrintGrad() {
    std::cout << "weights_grad:\n";
    weights_grad.print("W_");
    std::cout << "bias_grad:\n";
    bias_grad.print("B_");
}

void CuConvolutionLayer::FetchResultToCpu() {
    weights.zeros(dl.b_size, dl.in_dim);
    b.zeros(dl.b_size);
    CUDA_CHECK(cudaMemcpy(weights.data(), dl.weights, dl.w_size * sizeof(float), cudaMemcpyKind::cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(b.data(), dl.bias, dl.b_size * sizeof(float), cudaMemcpyKind::cudaMemcpyDeviceToHost));
}

void CuConvolutionLayer::FetchGradToCpu() {
    weights_grad.zeros(dl.b_size, dl.in_dim);
    bias_grad.zeros(dl.b_size);
    CUDA_CHECK(cudaMemcpy(weights_grad.data(), dl.grad_w, dl.w_size * sizeof(float), cudaMemcpyKind::cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(bias_grad.data(), dl.grad_b, dl.b_size * sizeof(float), cudaMemcpyKind::cudaMemcpyDeviceToHost));
}
