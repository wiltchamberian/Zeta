#include "cuLayer.h"
#include "kernels.h"
#include <random>
#include <cudnn.h>
#include "cu_tool.h"
#include "device_launch_parameters.h"
#include "CuNN.h"

/**********************CuLinearLeakyReluLayer*****************************/
CuLinearLeakyReluLayer::CuLinearLeakyReluLayer() {
}

CuLinearLeakyReluLayer::CuLinearLeakyReluLayer(int input, int output)
    :in_dim(input)
    , out_dim(output) {
    weights = Tensor(output, input); //reverse order to level up computation performance
    b = Tensor(output);

    RandomParameters();
}

void CuLinearLeakyReluLayer::RandomParameters() {
    //He/Kaiming
    std::default_random_engine generator;
    int fan_in = weights.shape[0];   //input neural network
    float stddev = std::sqrt(2.0f / fan_in);
    std::normal_distribution<float> distribution(0.0f, stddev);
    for (int i = 0; i < weights.shape[0]; ++i) {
        for (int j = 0; j < weights.shape[1]; ++j) {
            weights(i,j) = distribution(generator);
        }
    }
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

    //float dd[1000];
    //cudaMemcpy(dd, dl.activation, out_dim * sizeof(float), cudaMemcpyDeviceToHost);
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
    if (nn->c != 0) {
        regular_grad();
    }
    
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
    int dim_delta = outputShape.Dim();
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

    float delta[16];
    cudaMemcpy(delta, dl.weights, 16 * sizeof(float), cudaMemcpyKind::cudaMemcpyDeviceToHost);
    //float prevAc[16];
    //cudaMemcpy(prevAc, prev_activation, dim_delta_prev * sizeof(float), cudaMemcpyKind::cudaMemcpyDeviceToHost);
    //float test[16];
    //cudaMemcpy(test, prevs[0]->GetDelta(), dim_delta_prev * sizeof(float), cudaMemcpyKind::cudaMemcpyDeviceToHost);
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

    //float dd[18];
    //cudaMemcpy(dd, dl.delta, dim_delta * sizeof(float) * 2, cudaMemcpyKind::cudaMemcpyDeviceToHost);
}

void CuLinearLeakyReluLayer::regular_grad()
{
    int total = weights.numel();
    dim3 block(TILE_WIDTH);
    dim3 grid((total + TILE_WIDTH - 1) / TILE_WIDTH);
    regular_kernel << <grid, block >> > (dl.weights, dl.grad_w, total, nn->c);
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
    //return dl.b_size;
    return outputShape.NumElements();
}

float* CuLinearLeakyReluLayer::GetDelta() {
    return dl.delta;
}

size_t CuLinearLeakyReluLayer::GetDeltaSize() {
    //return dl.b_size;
    return outputShape.NumElements();
}

float* CuLinearLeakyReluLayer::GetPrevActivation() {
    if (prevs[0] != nullptr) {
        return prevs[0]->GetActivation();
    }
    else {
        return prevActivation;
    }
}

CuLayer* CuLinearLeakyReluLayer::Clone() const {
    CuLinearLeakyReluLayer* abc = new CuLinearLeakyReluLayer();
    abc->in_dim = this->in_dim;
    abc->out_dim = this->out_dim;
    abc->weights = this->weights.Clone();
    abc->b = this->b.Clone();

    abc->alpha = this->alpha;
    return abc;
}

void CuLinearLeakyReluLayer::Print() {

    std::cout << "weights:\n";
    weights.print("W_");
    std::cout << "biases:\n";
    b.print("B_");
    std::cout << std::endl;
}

void CuLinearLeakyReluLayer::PrintGrad() {
    std::cout << "CuLinearLeakyReluLayer:\n";
    std::cout << "weights_grad:\n";
    weights_grad.print_torch_style();
    std::cout << "bias_grad:\n";
    bias_grad.print_torch_style();
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

void CuLinearLeakyReluLayer::FetchActivationToCpu() {
    ac.zeros(outputShape.N, outputShape.C, outputShape.H, outputShape.W);
    CUDA_CHECK(cudaMemcpy( ac.data(), dl.activation, outputShape.NumElements()*sizeof(float), cudaMemcpyKind::cudaMemcpyDeviceToHost));
}

void CuLinearLeakyReluLayer::PrintDelta() {
    Tensor delta(outputShape.N, outputShape.C, outputShape.H, outputShape.W);
    cudaMemcpy(delta.data(), dl.delta, outputShape.NumElements()*sizeof(float), cudaMemcpyKind::cudaMemcpyDeviceToHost);

}

void CuLinearLeakyReluLayer::Save(std::fstream fs) {
    FetchResultToCpu();

}

/// <summary>
/// CuLinearTanhLayer
/// </summary>

void CuLinearTanhLayer::forward() {
    float* inputData = prevs.size() > 0 ? prevs[0]->GetActivation() : prevActivation;
    if (inputData == nullptr) {
        assert(false);
        return;
    }
    int out_dim = dl.w_size / dl.in_dim;
    dim3 block(TILE_WIDTH, TILE_WIDTH);
    dim3 grid((out_dim + block.x - 1) / block.x, (inputShape.N + block.y - 1) / block.y);


    linear_tanh_forward_kernel << <grid, block >> > (
        inputData,
        dl.weights,
        dl.bias,
        dl.activation,
        inputShape.N, dl.in_dim, out_dim
        );

}

void CuLinearTanhLayer::backwardEx() {
    add = false;
    dgrad();
    wgrad();
    bgrad();
}

void CuLinearTanhLayer::dgrad() {
    if (prevs.empty()) {
        return;
    }
    int dim_delta_prev = inputShape.Dim();//prevs[0]->GetDeltaSize();
    int dim_delta = GetDeltaSize();
    float* prev_activation = prevs[0]->GetActivation();
    dim3 block(TILE_WIDTH, TILE_WIDTH);
    dim3 grid((dim_delta_prev + block.x - 1) / block.x, (inputShape.N + block.y - 1) / block.y);
    linear_tanh_backward_kernel << <grid, block >> > (
        dl.delta,        // δ^{l}
        dl.weights,
        prev_activation,       // z^l (或 a^{l-1} 用于 σ'(z))
        prevs[0]->GetDelta(),            // δ^(l-1) 输出
        prevs[0]->add,
        inputShape.N,
        dim_delta_prev,
        dim_delta
        );
    prevs[0]->add = true;

}

/// <summary>
/// SoftmaxEntropyLayer
/// </summary>
/// 

float CuSoftmaxCrossEntropyLayer::FetchLoss() {
    int total = inputShape.NumElements();
    int batch = nn->batchSize;
    dim3 block(1024);
    dim3 grid(1);

    //float dd[1000];
    //cudaMemcpy(dd, activation, total * sizeof(float), cudaMemcpyKind::cudaMemcpyDeviceToHost);
    //float ds[1000];
    //cudaMemcpy(ds, y, total * sizeof(float), cudaMemcpyKind::cudaMemcpyDeviceToHost);

    cross_entropy_kernel << <grid, block >> > (activation, y, loss, batch, total);
    float res = 0;
    CUDA_CHECK(cudaMemcpy(&res, loss, sizeof(float), cudaMemcpyKind::cudaMemcpyDeviceToHost));
    return res;
}

void CuSoftmaxCrossEntropyLayer::forward() {
    assert(!prevs.empty());
    constexpr int BLOCK = 1024;
    int M = inputShape.Dim();
    int batchSize = nn->batchSize;
    float* prev_activation = prevs[0]->GetActivation();

    //softmax_forward_kernel<<<batchSize, BLOCK >> > (prev_activation, activation,M);
    simple_softmax_forward_kernel << <batchSize, 1 >> > (prev_activation, activation, M);


    ////TEST, FIX ME, TODO
    //float dd[1000];
    //float dd1[1000];
    //CUDA_CHECK(cudaMemcpy(dd, prev_activation, inputShape.NumElements() * sizeof(float), cudaMemcpyKind::cudaMemcpyDeviceToHost));
    //CUDA_CHECK(cudaMemcpy(dd1, activation, inputShape.NumElements() * sizeof(float), cudaMemcpyKind::cudaMemcpyDeviceToHost));

}

void CuSoftmaxCrossEntropyLayer::backwardEx() {
    assert(!prevs.empty());
    //int M = prevs[0]->GetActivationSize();
    int M = inputShape.Dim();
    
   /* float kk[16];
    cudaMemcpy(kk, y, M * sizeof(float), cudaMemcpyKind::cudaMemcpyDeviceToHost);
    float tt[16];
    cudaMemcpy(tt, activation, M * sizeof(float), cudaMemcpyKind::cudaMemcpyDeviceToHost);*/

    
    int batchSize = nn->batchSize;
    dim3 block(TILE_WIDTH, TILE_WIDTH);
    dim3 grid((M+TILE_WIDTH-1)/TILE_WIDTH, (batchSize + TILE_WIDTH - 1) / TILE_WIDTH);
    softmax_backward_kernel << <grid, block >> > (y, activation,prevs[0]->GetDelta(), batchSize, M);

    //TEST TODO
    /*float dd[16];
    cudaMemcpy(dd, prevs[0]->GetDelta(), M * sizeof(float), cudaMemcpyKind::cudaMemcpyDeviceToHost);*/
    
}

void CuSoftmaxCrossEntropyLayer::applyGradient() {

}

void CuSoftmaxCrossEntropyLayer::BindLabelToDevice() {
    if (y != nullptr) {
        int num = inputShape.NumElements();
        CUDA_CHECK(cudaMemcpy(y, label.data(), num * sizeof(float), cudaMemcpyHostToDevice));
    }
}

void CuSoftmaxCrossEntropyLayer::BindWorkspace(void* ptr) {
    activation = reinterpret_cast<float*>(ptr);
    int num = inputShape.NumElements();
    y = reinterpret_cast<float*>(ptr) + num;
    if (label.data()) {
        CUDA_CHECK(cudaMemcpy(y, label.data(), num * sizeof(float), cudaMemcpyHostToDevice));
    }
    loss = reinterpret_cast<float*>(ptr) + num * 2;
    //float dd[18];
    //CUDA_CHECK(cudaMemcpy(dd, y, num * sizeof(float), cudaMemcpyDeviceToHost));
}

size_t CuSoftmaxCrossEntropyLayer::GetWorkspaceSize() {
    //for label-y and activation and loss
    return inputShape.NumElements() * 2 + 1;
}

void CuSoftmaxCrossEntropyLayer::BindDevice(void* ptr) {
    
}

size_t CuSoftmaxCrossEntropyLayer::GetDeviceSize() {
    return 0;
}

CuLayer* CuSoftmaxCrossEntropyLayer::Clone() const {
    CuSoftmaxCrossEntropyLayer* r = new CuSoftmaxCrossEntropyLayer();
    r->inputShape = this->inputShape;
    r->outputShape = this->outputShape;
    return r;
}

void CuSoftmaxCrossEntropyLayer::InferOutputShape(TensorShape networkInput) {
    TensorShape shape = prevs.empty() ? networkInput : prevs[0]->outputShape;
    TensorShape result;
    result.N = shape.N;
    result.C = shape.C;
    result.H = shape.H;
    result.W = shape.W;

    this->inputShape = shape;
    this->outputShape = result;
    return;
}

void CuSoftmaxCrossEntropyLayer::FetchPredYToCpu() {
  
}

void CuSoftmaxCrossEntropyLayer::PrintPredY() {

}

void CuSoftmaxCrossEntropyLayer::FetchActivationToCpu() {
    distribution.zeros(outputShape.N, outputShape.C * outputShape.H *  outputShape.W);
    CUDA_CHECK(cudaMemcpy(distribution.data(), activation, outputShape.NumElements() * sizeof(float), cudaMemcpyKind::cudaMemcpyDeviceToHost));
}

/************************************************/
CuMseLayer::CuMseLayer() {

}

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

float CuMseLayer::FetchLoss() {
    assert(!prevs.empty());
    auto prev = prevs[0];
    float* a = prev->GetActivation();
    int total = inputShape.NumElements();

    dim3 block(1024);
    dim3 grid(1);
    mse_loss_kernel << <grid, block >> > (a, y_label, loss, total);
    float res = 0;
    CUDA_CHECK(cudaMemcpy(&res, loss, sizeof(float), cudaMemcpyKind::cudaMemcpyDeviceToHost));
    return res;
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
   

    mse_loss_backward_kernel << <grid, block >> > (
        prev->GetActivation(),
        y_label,
        prev->GetDelta(),
        batch,
        out_dim
        );
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

void CuMseLayer::BindLabelToDevice() {
    if (y_label != nullptr) {
        int num = inputShape.NumElements();
        CUDA_CHECK(cudaMemcpy(y_label, label.data(), num * sizeof(float), cudaMemcpyHostToDevice));
    }
}

size_t CuMseLayer::GetWorkspaceSize() {
    //label-y
    return inputShape.NumElements() + 1;
}

void CuMseLayer::BindWorkspace(void* pointer) {
    float* ptr = reinterpret_cast<float*>(pointer);

    int num = inputShape.NumElements();
    y_label = reinterpret_cast<float*>(ptr);
    if (label.data()) {
        CUDA_CHECK(cudaMemcpy(y_label, label.data(), num * sizeof(float), cudaMemcpyHostToDevice));
    }

    ptr += num;
    loss = ptr;
    return;
}

size_t CuMseLayer::GetDeviceSize() {
    return 0;
}

void CuMseLayer::BindDevice(void* ptr) {
    
}

size_t CuMseLayer::GetActivationSize() {
    return label.numel();
}

float* CuMseLayer::GetActivation() {
    assert(false);
    return nullptr;
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

CuLayer* CuMseLayer::Clone() const {
    CuMseLayer* layer = new CuMseLayer();
    layer->inputShape = this->inputShape;
    layer->outputShape = this->outputShape;
    layer->label = this->label.Clone();
    return layer;
}

/******************************convolution layer*********************************/
CuConvolutionLayer::CuConvolutionLayer() {

}

CuConvolutionLayer::CuConvolutionLayer(int K, int C, int R, int S)
{
    weights = Tensor(K, C, R, S);
    b = Tensor(K);
    dl.w_size = K * C * R * S;
    dl.b_size = K;

    RandomParameters();
}

void CuConvolutionLayer::RandomParameters() {
    //He/Kaiming initialize
    int fan_in = weights.shape[2] * weights.shape[3] * weights.shape[1];
    std::default_random_engine generator;
    std::normal_distribution<float> distribution(0.0, sqrt(2.0 / fan_in));
    for (int i = 0; i < weights.shape[0]; ++i) {
        for (int j = 0; j < weights.shape[1]; ++j) {
            for (int k = 0; k < weights.shape[2]; ++k) {
                for (int l = 0; l < weights.shape[3]; ++l) {
                    weights(i, j, k, l) = distribution(generator);
                }
            }
        }
    }
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
    Tensor bias = b.contiguous();
    size_t b_size = bias.numel();
    dl.bias = reinterpret_cast<float*>(addr);
    dl.b_size = b_size;

    if (b.data()) {
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
    //return dl.b_size;
    return outputShape.NumElements();
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

CuLayer* CuConvolutionLayer::Clone() const{
    CuConvolutionLayer* layer = new CuConvolutionLayer();
    layer->inputShape = this->inputShape;
    layer->outputShape = this->outputShape;
    layer->weights = this->weights.Clone();
    layer->b = this->b.Clone();
    layer->weights_grad = this->weights_grad.Clone();
    layer->bias_grad = this->bias_grad.Clone();
    layer->ac = this->ac.Clone();
    layer->in_dim = this->in_dim;
    layer->out_dim = this->out_dim;
    layer->alpha = this->alpha;
    layer->padH = this->padH;
    layer->padW = this->padW;
    layer->strideH = this->strideH;
    layer->strideW = this->strideW;
    layer->dl.w_size = this->dl.w_size;
    layer->dl.b_size = this->dl.b_size;
    
    return layer;
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
    if (nn->c != 0) {
        regular_grad();
    }
    
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

void CuConvolutionLayer::regular_grad() 
{
    int total = weights.numel();
    dim3 block(TILE_WIDTH);
    dim3 grid((total + TILE_WIDTH - 1) / TILE_WIDTH);
    regular_kernel << <grid, block >> > (dl.weights, dl.grad_w, total, nn->c);
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

void CuConvolutionLayer::PrintDelta() {

    Tensor tensor(outputShape.N,outputShape.C,outputShape.H,outputShape.W);
    cudaMemcpy(tensor.data(), dl.delta, outputShape.NumElements() * sizeof(float), cudaMemcpyDeviceToHost);
}

void CuConvolutionLayer::Print() {
    std::cout << "weights:\n";
    //weights.print("W_");
    weights.print_torch_style();
    std::cout << "biases:\n";
    //b.print("B_");
    b.print_torch_style();
    std::cout << std::endl;
}

void CuConvolutionLayer::PrintGrad() {
    std::cout << "CuConvolutionLayer\n";
    std::cout << "weights_grad:\n";
    weights_grad.print_torch_style();
    std::cout << "bias_grad:\n";
    bias_grad.print_torch_style();
}

void CuConvolutionLayer::FetchResultToCpu() {
    weights.zeros(weights.shape);
    b.zeros(b.shape);
    CUDA_CHECK(cudaMemcpy(weights.data(), dl.weights, dl.w_size * sizeof(float), cudaMemcpyKind::cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(b.data(), dl.bias, dl.b_size * sizeof(float), cudaMemcpyKind::cudaMemcpyDeviceToHost));
}

void CuConvolutionLayer::FetchGradToCpu() {
    weights_grad.zeros(weights.shape);
    bias_grad.zeros(b.shape);
    CUDA_CHECK(cudaMemcpy(weights_grad.data(), dl.grad_w, dl.w_size * sizeof(float), cudaMemcpyKind::cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(bias_grad.data(), dl.grad_b, dl.b_size * sizeof(float), cudaMemcpyKind::cudaMemcpyDeviceToHost));
}

void CuConvolutionLayer::FetchActivationToCpu() {
    ac.zeros(outputShape.N, outputShape.C, outputShape.H, outputShape.W);
    CUDA_CHECK(cudaMemcpy(ac.data(), dl.activation, outputShape.NumElements()* sizeof(float), cudaMemcpyKind::cudaMemcpyDeviceToHost));
}
