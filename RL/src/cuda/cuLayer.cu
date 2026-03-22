#include "cuLayer.h"
#include "kernels.h"
#include <random>
#include <cudnn.h>
#include "cu_tool.h"
#include "device_launch_parameters.h"
#include "CuNN.h"
#include "cudnn_backend.h"
#include "TensorStream.h"

namespace zeta {

float* CuLayer::GetActivation() {
    return output->v;
}

size_t CuLayer::GetActivationSize() {
    return output->shape.NumElements();
}

float* CuLayer::GetDelta() {
    return output->delta;
}

size_t CuLayer::GetDeltaSize() {
    return output->shape.NumElements();
}

CuLayer* CuLayer::AddLayer(CuLayer* layer) {
    nn->Connect(this, layer);
    return layer;
}

CuLayer* CuLayer::Add(CuLayer* layer) {
    nn->Connect(this, layer);
    return layer;
}

/**********************Linear*****************************/
Linear::Linear() {
    layerType = LayerType::Fully;
}

Linear::Linear(int input, int output)
    : in_dim(input)
    , out_dim(output)
{
    layerType = LayerType::Fully;

    weights = Tensor(output, input); //reverse order to level up computation performance
    b = Tensor(output);

    RandomParameters();
}

void Linear::RandomParameters() {
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

void Linear::forward() { /* kernel launch */
    float* inputData = input->v;
    if (inputData == nullptr) {
        assert(false);
        return;
    }
    int out_dim = dl.w_size / dl.in_dim;
    dim3 block(TILE_WIDTH, TILE_WIDTH);
    dim3 grid((out_dim + block.x - 1) / block.x, (input->shape.N + block.y - 1) / block.y);

    
    linear_leaky_relu_forward_kernel << <grid, block >> > (
        inputData,
        dl.weights,
        dl.bias,
        output->v,
        input->shape.N, dl.in_dim, out_dim,
        alpha
        );

    //float dd[1000];
    //cudaMemcpy(dd, dl.activation, out_dim * sizeof(float), cudaMemcpyDeviceToHost);
}

void Linear::backwardEx() {
    add = false;
    dgrad();
    wgrad();
    bgrad();
    if (nn->c != 0) {
        regular_grad();
    }
}

void Linear::applyGradient() {
    if (nn->optimizerType == SGD) {
        int CPQ = weights.numel() / weights.shape[0];
        int K = weights.shape[0];
        int block_y = (K + TILE_WIDTH - 1) / TILE_WIDTH;
        int block_x = (CPQ + TILE_WIDTH - 1) / TILE_WIDTH;
        dim3 grid(block_x, block_y);
        dim3 block(TILE_WIDTH, TILE_WIDTH);
        apply_gradient_kernel << <grid, block >> > (dl.grad_w, dl.grad_b, dl.weights, dl.bias, K, CPQ, nn->learningRate);
    }
    else if (nn->optimizerType == Adam) {
        int total = in_dim * out_dim;
        dim3 block(TILE_WIDTH);
        dim3 grid((total + TILE_WIDTH - 1) / TILE_WIDTH);
        adam_gradient_kernel << <grid, block >> > (dl.grad_w, dl.w_m, dl.w_v, nn->learningRate,
            nn->beta1, nn->beta2, nn->beta1_t, nn->beta2_t, nn->epsilon, dl.weights, total);
        int biasCount = out_dim;
        dim3 grid1((biasCount + TILE_WIDTH - 1) / TILE_WIDTH);
        adam_gradient_kernel << <grid1, block >> > (dl.grad_b, dl.b_m, dl.b_v, nn->learningRate,
            nn->beta1, nn->beta2, nn->beta1_t, nn->beta2_t, nn->epsilon, dl.bias, biasCount);
    }
}

void Linear::dgrad() {
    if (prevs.empty()) {
        return;
    }
    int dim_delta_prev = input->shape.Dim();//prevs[0]->GetDeltaSize();
    int dim_delta = output->shape.Dim();
    float* prev_activation = prevs[0]->GetActivation();
    dim3 block(TILE_WIDTH, TILE_WIDTH);
    dim3 grid((dim_delta_prev + block.x - 1) / block.x, (input->shape.N + block.y - 1) / block.y);
    linear_leaky_relu_backward_kernel << <grid, block >> > (
        output->delta,        // δ^{l}
        dl.weights,
        prev_activation,       // z^l (或 a^{l-1} 用于 σ'(z))
        input->delta,            // δ^(l-1) 输出
        prevs[0]->add,
        input->shape.N,
        dim_delta_prev,
        dim_delta,
        prevs[0]->GetAlpha()   //must use prev's alpha, pay attention
        );
    prevs[0]->add = true;

    //float delta[16];
    //cudaMemcpy(delta, dl.weights, 16 * sizeof(float), cudaMemcpyKind::cudaMemcpyDeviceToHost);
    //float prevAc[16];
    //cudaMemcpy(prevAc, prev_activation, dim_delta_prev * sizeof(float), cudaMemcpyKind::cudaMemcpyDeviceToHost);
    //float test[16];
    //cudaMemcpy(test, prevs[0]->GetDelta(), dim_delta_prev * sizeof(float), cudaMemcpyKind::cudaMemcpyDeviceToHost);
}

void Linear::wgrad() {
    dim3 block(TILE_WIDTH, TILE_WIDTH);
    int N = input->shape.N;
    int dim_delta_prev = input->shape.Dim();
    int dim_delta = output->shape.Dim();
    dim3 grid((dim_delta_prev + block.x - 1) / block.x, (dim_delta + block.y - 1) / block.y);
    float* prev_activation = input->v;
    compute_grad_w_kernel << <grid, block >> > (
        prev_activation,
        output->delta,
        dl.grad_w,
        N,
        dim_delta_prev,
        dim_delta
        );

}

void Linear::bgrad() {
    dim3 block(TILE_WIDTH, TILE_WIDTH);
    int dim_delta = output->shape.Dim();
    dim3 grid((dim_delta + block.x - 1) / block.x);
    compute_grad_b_kernel << <grid, block.x >> > (
        output->delta/*ws.deltas[l]*/,
        dl.grad_b/*deviceLayers[l].grad_b*/,
        input->shape.N,
        dim_delta
        );

    //float dd[18];
    //cudaMemcpy(dd, dl.delta, dim_delta * sizeof(float) * 2, cudaMemcpyKind::cudaMemcpyDeviceToHost);
}

void Linear::regular_grad()
{
    int total = weights.numel();
    dim3 block(TILE_WIDTH);
    dim3 grid((total + TILE_WIDTH - 1) / TILE_WIDTH);
    regular_kernel << <grid, block >> > (dl.weights, dl.grad_w, total, nn->c);
}

void Linear::InferOutputShape(TensorShape networkInput) {
    TensorShape shape = prevs.empty() ? networkInput : prevs[0]->output->shape;
    TensorShape result;
    result.N = shape.N;
    result.C = weights.shape[0];
    result.H = 1;
    result.W = 1;
    //this->inputShape = shape;
    //this->outputShape = result;
    this->output->shape = result;
    return;
}

size_t Linear::GetWorkspaceSize() {
    return output->shape.NumElements() * 2 ;
}

size_t Linear::GetDeviceSize() {
    size_t total = 0;
    if (nn->optimizerType == SGD) {
        size_t w = weights.numel();
        size_t b = this->b.numel();
        total += w;  // weights
        total += w;  // grad_w
        total += b;  // bias
        total += b;  // grad_b
    }
    else if (nn->optimizerType == Adam) {
        size_t w = weights.numel();
        size_t b = this->b.numel();
        total += w * 4;  // weights
        total += b * 4;
    }

    return total;
}

void Linear::BindWorkspace(void* ptr) {
    float* d = reinterpret_cast<float*>(ptr);
    output->v = d;
    output->delta = d + output->shape.NumElements();
}

void Linear::BindDevice(void* ptr) {
    char* addr = reinterpret_cast<char*>(ptr);

    // -------- weights --------
    Tensor w = weights.contiguous();
    size_t w_size = w.numel();
    dl.weights = reinterpret_cast<float*>(addr);
    dl.w_size = w_size;
    dl.in_dim = w_size / w.shape[0];//layer->weights.shape[1];

    //if (w.data()) {
    //    CU_CHECK(cudaMemcpy(
    //        dl.weights,
    //        w.data(),
    //        w_size * sizeof(float),
    //        cudaMemcpyHostToDevice
    //    ));
    //    
    //}
    addr += w_size * sizeof(float);

    // -------- bias --------
    Tensor bias = b.contiguous();
    size_t b_size = bias.numel();
    dl.bias = reinterpret_cast<float*>(addr);
    dl.b_size = b_size;

    //if (bias.data()) {
    //    CU_CHECK(cudaMemcpy(
    //        dl.bias,
    //        bias.data(),
    //        b_size * sizeof(float),
    //        cudaMemcpyHostToDevice
    //    ));
    //}
    
    addr += b_size * sizeof(float);

    // -------- grad_w --------
    dl.grad_w = reinterpret_cast<float*>(addr);
    addr += w_size * sizeof(float);

    // -------- grad_b --------
    dl.grad_b = reinterpret_cast<float*>(addr);
    addr += b_size * sizeof(float);

    if (nn->optimizerType == SGD) {
        ;
    }
    else if (nn->optimizerType == Adam) {
        //CU_CHECK(cudaMemset(addr, 0, (w_size + b_size) * 2 * sizeof(float)));
        dl.w_m = reinterpret_cast<float*>(addr);
        addr += w_size * sizeof(float);
        dl.b_m = reinterpret_cast<float*>(addr);
        addr += b_size * sizeof(float);
        dl.w_v = reinterpret_cast<float*>(addr);
        addr += w_size * sizeof(float);
        dl.b_v = reinterpret_cast<float*>(addr);
        addr += b_size * sizeof(float);
    }
}

void Linear::HostToDevice() {
    Tensor w = weights.contiguous();
    if (w.data() && dl.weights) {
        CU_CHECK(cudaMemcpy(
            dl.weights,
            w.data(),
            dl.w_size * sizeof(float),
            cudaMemcpyHostToDevice
        ));
    }
    Tensor bias = b.contiguous();
    if (b.data() && dl.bias) {
        CU_CHECK(cudaMemcpy(
            dl.bias,
            bias.data(),
            dl.b_size * sizeof(float),
            cudaMemcpyHostToDevice
        ));
    }
}

float* Linear::GetDelta() {
    return output->delta;
}

size_t Linear::GetDeltaSize() {
    //return dl.b_size;
    return output->shape.NumElements();
}

float* Linear::GetPrevActivation() {
    return input->v;
}

CuLayer* Linear::Clone() const {
    Linear* abc = new Linear();
    abc->in_dim = this->in_dim;
    abc->out_dim = this->out_dim;
    abc->weights = this->weights.Clone();
    abc->b = this->b.Clone();

    abc->alpha = this->alpha;
    return abc;
}

void Linear::Print() {

    std::cout << "weights:\n";
    weights.print("W_");
    std::cout << "biases:\n";
    b.print("B_");
    std::cout << std::endl;
}

void Linear::PrintGrad() {
    std::cout << "Linear:\n";
    std::cout << "weights_grad:\n";
    weights_grad.print_torch_style();
    std::cout << "bias_grad:\n";
    bias_grad.print_torch_style();
}

void Linear::FetchResultToCpu() {
    weights.zeros(dl.b_size, dl.in_dim);
    b.zeros(dl.b_size);
    CU_CHECK(cudaMemcpy(weights.data(), dl.weights, dl.w_size * sizeof(float), cudaMemcpyKind::cudaMemcpyDeviceToHost));
    CU_CHECK(cudaMemcpy(b.data(), dl.bias, dl.b_size * sizeof(float), cudaMemcpyKind::cudaMemcpyDeviceToHost));
}

void Linear::FetchGradToCpu() {
    weights_grad.zeros(dl.b_size, dl.in_dim);
    bias_grad.zeros(dl.b_size);
    CU_CHECK(cudaMemcpy(weights_grad.data(), dl.grad_w, dl.w_size * sizeof(float), cudaMemcpyKind::cudaMemcpyDeviceToHost));
    CU_CHECK(cudaMemcpy(bias_grad.data(), dl.grad_b, dl.b_size * sizeof(float), cudaMemcpyKind::cudaMemcpyDeviceToHost));
}

void Linear::FetchActivationToCpu() {
    ac.zeros(output->shape.N, output->shape.C, output->shape.H, output->shape.W);
    CU_CHECK(cudaMemcpy( ac.data(), output->v, output->shape.NumElements()*sizeof(float), cudaMemcpyKind::cudaMemcpyDeviceToHost));
}

void Linear::PrintActivation() {
    FetchActivationToCpu();
    ac.print_torch_style("linear:");
}

void Linear::PrintDelta() {
    if (output->shape.H == 1 && output->shape.W == 1) {
        Tensor delta(output->shape.N, output->shape.C);
        cudaMemcpy(delta.data(), output->delta, output->shape.NumElements() * sizeof(float), cudaMemcpyKind::cudaMemcpyDeviceToHost);
        delta.print_torch_style("delta:");
    }
    else {
        Tensor delta(output->shape.N, output->shape.C, output->shape.H, output->shape.W);
        cudaMemcpy(delta.data(), output->delta, output->shape.NumElements() * sizeof(float), cudaMemcpyKind::cudaMemcpyDeviceToHost);
        delta.print_torch_style("delta:");
    }
    
}

void Linear::PrintBGrad() {
    Tensor delta(out_dim);
    cudaMemcpy(delta.data(), dl.grad_b, out_dim * sizeof(float), cudaMemcpyKind::cudaMemcpyDeviceToHost);
    delta.print_torch_style("dBias:");
}

void Linear::PrintWGrad() {
    Tensor delta(out_dim, in_dim);
    cudaMemcpy(delta.data(), dl.grad_w, out_dim * in_dim * sizeof(float), cudaMemcpyKind::cudaMemcpyDeviceToHost);
    delta.print_torch_style("dW:");
}

void Linear::Save(std::fstream fs) {
    FetchResultToCpu();

}

void Linear::Save(BinaryStream& stream) const {
    Tensor theWeights;
    theWeights.zeros(out_dim, in_dim);
    Tensor theBias;
    theBias.zeros(out_dim);
    CU_CHECK(cudaMemcpy(theWeights.data(), dl.weights, dl.w_size * sizeof(float), cudaMemcpyKind::cudaMemcpyDeviceToHost));
    CU_CHECK(cudaMemcpy(theBias.data(), dl.bias, out_dim * sizeof(float), cudaMemcpyKind::cudaMemcpyDeviceToHost));
    
    stream.write<int>((int)LayerType::Fully);
    stream.write<int>(in_dim);
    stream.write<int>(out_dim);
    TensorStream::Save(theWeights, stream);
    TensorStream::Save(theBias, stream);
}

void Linear::Load(BinaryStream& stream) {
    stream.read<int>();
    dl.in_dim = stream.read<int>();
    dl.b_size = stream.read<int>();
    out_dim = dl.b_size;
    in_dim = dl.in_dim;
    dl.w_size = dl.in_dim * dl.b_size;
    weights = TensorStream::Load(dl.in_dim, dl.b_size, stream);
    b = TensorStream::Load(dl.b_size, stream);
    HostToDevice();
}

/// <summary>
/// CuLinearTanhLayer
/// </summary>

void CuLinearTanhLayer::forward() {
    float* inputData = input->v;
    if (inputData == nullptr) {
        assert(false);
        return;
    }
    int out_dim = dl.w_size / dl.in_dim;
    dim3 block(TILE_WIDTH, TILE_WIDTH);
    dim3 grid((out_dim + block.x - 1) / block.x, (input->shape.N + block.y - 1) / block.y);


    linear_tanh_forward_kernel << <grid, block >> > (
        inputData,
        dl.weights,
        dl.bias,
        output->v,
        input->shape.N, dl.in_dim, out_dim
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
    int dim_delta_prev = input->shape.Dim();
    int dim_delta = GetDeltaSize();
    float* prev_activation = prevs[0]->GetActivation();
    dim3 block(TILE_WIDTH, TILE_WIDTH);
    dim3 grid((dim_delta_prev + block.x - 1) / block.x, (input->shape.N + block.y - 1) / block.y);
    linear_tanh_backward_kernel << <grid, block >> > (
        output->delta,        // δ^{l}
        dl.weights,
        prev_activation,       // z^l (或 a^{l-1} 用于 σ'(z))
        input->delta,            // δ^(l-1) 输出
        prevs[0]->add,
        input->shape.N,
        dim_delta_prev,
        dim_delta
        );
    prevs[0]->add = true;

}

/// <summary>
/// SoftmaxEntropyLayer
/// </summary>
/// 

Tensor CuSoftmaxCrossEntropyLayer::FetchLoss() {
    int total = input->shape.NumElements();
    int batch = nn->batchSize;
    dim3 block(1024);
    dim3 grid(1);

    //float dd[1000];
    //cudaMemcpy(dd, activation, total * sizeof(float), cudaMemcpyKind::cudaMemcpyDeviceToHost);
    //float ds[1000];
    //cudaMemcpy(ds, y, total * sizeof(float), cudaMemcpyKind::cudaMemcpyDeviceToHost);

    cross_entropy_kernel << <grid, block >> > (output->v, yLabel, loss, batch, total);
    //stable_cross_entropy_kernel<<<grid,block>>>()
    
    cudaDeviceSynchronize();
    Tensor res(1);
    CU_CHECK(cudaMemcpy(res.data(), loss, sizeof(float), cudaMemcpyKind::cudaMemcpyDeviceToHost));
    return res;
}

void CuSoftmaxCrossEntropyLayer::forward() {
    assert(!prevs.empty());
    constexpr int BLOCK = 1024;
    int M = input->shape.Dim();
    int batchSize = nn->batchSize;
    float* prev_activation = input->v;

    //softmax_forward_kernel<<<batchSize, BLOCK >> > (prev_activation, output->v,M);
    simple_softmax_forward_kernel << <batchSize, 1 >> > (prev_activation, output->v, M);


    ////TEST, FIX ME, TODO
    /*float dd[1000];
    float dd1[1000];
    CU_CHECK(cudaMemcpy(dd, prev_activation, input->shape.NumElements() * sizeof(float), cudaMemcpyKind::cudaMemcpyDeviceToHost));
    CU_CHECK(cudaMemcpy(dd1, output->v, input->shape.NumElements() * sizeof(float), cudaMemcpyKind::cudaMemcpyDeviceToHost));*/

}

void CuSoftmaxCrossEntropyLayer::backwardEx() {
    add = false;
    assert(!prevs.empty());
    int M = input->shape.Dim();
    
   /* float kk[16];
    cudaMemcpy(kk, y, M * sizeof(float), cudaMemcpyKind::cudaMemcpyDeviceToHost);
    float tt[16];
    cudaMemcpy(tt, activation, M * sizeof(float), cudaMemcpyKind::cudaMemcpyDeviceToHost);*/

    
    int batchSize = nn->batchSize;
    dim3 block(TILE_WIDTH, TILE_WIDTH);
    dim3 grid((M+TILE_WIDTH-1)/TILE_WIDTH, (batchSize + TILE_WIDTH - 1) / TILE_WIDTH);
    softmax_backward_kernel << <grid, block >> > (yLabel, output->v,input->delta, batchSize, M);

    //TEST TODO
    /*float dd[16];
    cudaMemcpy(dd, prevs[0]->GetDelta(), M * sizeof(float), cudaMemcpyKind::cudaMemcpyDeviceToHost);*/
    
}

void CuSoftmaxCrossEntropyLayer::applyGradient() {

}

void CuSoftmaxCrossEntropyLayer::BindLabelToDevice() {
    if (yLabel != nullptr) {
        int num = input->shape.NumElements();
        CU_CHECK(cudaMemcpy(yLabel, label.data(), num * sizeof(float), cudaMemcpyHostToDevice));
    }
}

void CuSoftmaxCrossEntropyLayer::BindWorkspace(void* ptr) {
    output->v = reinterpret_cast<float*>(ptr);
    int num = input->shape.NumElements();
    yLabel = reinterpret_cast<float*>(ptr) + num;
    if (label.data()) {
        CU_CHECK(cudaMemcpy(yLabel, label.data(), num * sizeof(float), cudaMemcpyHostToDevice));
    }
    loss = reinterpret_cast<float*>(ptr) + num * 2;
    //float dd[18];
    //CUDA_CHECK(cudaMemcpy(dd, y, num * sizeof(float), cudaMemcpyDeviceToHost));
}

size_t CuSoftmaxCrossEntropyLayer::GetWorkspaceSize() {
    //for label-y and activation and loss
    return input->shape.NumElements() * 2 + 1;
}

void CuSoftmaxCrossEntropyLayer::BindDevice(void* ptr) {
    
}

size_t CuSoftmaxCrossEntropyLayer::GetDeviceSize() {
    return 0;
}

CuLayer* CuSoftmaxCrossEntropyLayer::Clone() const {
    CuSoftmaxCrossEntropyLayer* r = new CuSoftmaxCrossEntropyLayer();
    //r->inputShape = this->inputShape;
    //r->outputShape = this->outputShape;
    r->output = output->Clone();
    return r;
}

void CuSoftmaxCrossEntropyLayer::Save(BinaryStream& stream) const {
    stream.write<int>((int)LayerType::Softmax);
}

void CuSoftmaxCrossEntropyLayer::Load(BinaryStream& stream) {
    stream.read<int>();
}

void CuSoftmaxCrossEntropyLayer::InferOutputShape(TensorShape networkInput) {
    TensorShape shape = prevs.empty() ? networkInput : input->shape;
    TensorShape result;
    result.N = shape.N;
    result.C = shape.C;
    result.H = shape.H;
    result.W = shape.W;

    //this->inputShape = shape;
    //this->outputShape = result;
    this->output->InitShape(result);
    return;
}

void CuSoftmaxCrossEntropyLayer::FetchPredYToCpu() {
  
}

void CuSoftmaxCrossEntropyLayer::PrintPredY() {

}

void CuSoftmaxCrossEntropyLayer::FetchActivationToCpu() {
    distribution.zeros(output->shape.N, output->shape.C * output->shape.H * output->shape.W);
    CU_CHECK(cudaMemcpy(distribution.data(), output->v, output->shape.NumElements() * sizeof(float), cudaMemcpyKind::cudaMemcpyDeviceToHost));
}

void CuSoftmaxCrossEntropyLayer::PrintActivation() {
    FetchActivationToCpu();
    distribution.print_torch_style("softmax:");
}

/************************************************/
CuMseLayer::CuMseLayer() {
    layerType = LayerType::Mse;
}

CuMseLayer::CuMseLayer(int C) {
    layerType = LayerType::Mse;
    label = Tensor(C);
}

CuMseLayer::CuMseLayer(int C, int H) {
    layerType = LayerType::Mse;
    label = Tensor(C, H);
}

CuMseLayer::CuMseLayer(int C, int R, int S)
{
    layerType = LayerType::Mse;
    label = Tensor(C, R, S);
}

Tensor CuMseLayer::FetchLoss() {
    assert(!prevs.empty());
    auto prev = prevs[0];
    float* a = prev->GetActivation();
    int total = input->shape.NumElements();

    dim3 block(1024);
    dim3 grid(1);
    mse_loss_kernel << <grid, block >> > (a, y_label, loss, total);
    Tensor res(1);
    CU_CHECK(cudaMemcpy(res.data(), loss, sizeof(float), cudaMemcpyKind::cudaMemcpyDeviceToHost));
    return res;
}

void CuMseLayer::forward() {
    
}

void CuMseLayer::backwardEx() {
    assert(!prevs.empty());
    auto prev = prevs[0];

    int out_dim = label.numel() / label.shape[0];
    int batch = input->shape.N;

    dim3 block(TILE_WIDTH, TILE_WIDTH);
    dim3 grid((out_dim + block.x - 1) / block.x, (batch + block.y - 1) / block.y);
   

    mse_loss_backward_kernel << <grid, block >> > (
        input->v,
        y_label,
        input->delta,
        batch,
        out_dim
        );
}

void CuMseLayer::applyGradient() {

}

void CuMseLayer::InferOutputShape(TensorShape networkInput) {
    TensorShape shape = prevs.empty() ? networkInput : input->shape;
    TensorShape result;
    result.N = 1;
    result.H = 1;
    result.W = 1;
    result.C = 1;
    //this->inputShape = shape;
    //this->outputShape = result;
    this->output->shape = result;
    return;
}

void CuMseLayer::BindLabelToDevice() {
    if (y_label != nullptr) {
        int num = input->shape.NumElements();
        CU_CHECK(cudaMemcpy(y_label, label.data(), num * sizeof(float), cudaMemcpyHostToDevice));
    }
}

size_t CuMseLayer::GetWorkspaceSize() {
    //label-y
    return input->shape.NumElements() + 1;
}

void CuMseLayer::BindWorkspace(void* pointer) {
    float* ptr = reinterpret_cast<float*>(pointer);

    int num = input->shape.NumElements();
    y_label = reinterpret_cast<float*>(ptr);
    if (label.data()) {
        CU_CHECK(cudaMemcpy(y_label, label.data(), num * sizeof(float), cudaMemcpyHostToDevice));
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

void CuMseLayer::FetchPredYToCpu() {
    int d = input->shape.NumElements();
    predY.zeros(input->shape.N, input->shape.Dim());
    CU_CHECK(cudaMemcpy(predY.data(), prevs[0]->GetActivation(), d * sizeof(float), cudaMemcpyKind::cudaMemcpyDeviceToHost));
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
    
    layer->output = output->Clone();

    layer->label = this->label.Clone();
    return layer;
}

/******************************convolution layer*********************************/
Conv2d::Conv2d() {
    layerType == LayerType::Conv;
}

Conv2d::Conv2d(int K, int C, int R, int S, Size2D padding, Size2D stride)
    :padH(padding.h)
    ,padW(padding.w)
    ,strideH(stride.h)
    ,strideW(stride.w)
{
    layerType == LayerType::Conv;

    weights = Tensor(K, C, R, S);
    b = Tensor(K);
    dl.w_size = K * C * R * S;
    dl.b_size = K;

    weightsShape.N = K;
    weightsShape.C = C;
    weightsShape.H = R;
    weightsShape.W = S;

    RandomParameters();
}

Conv2d::~Conv2d() {

}

void Conv2d::RandomParameters() {
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

void Conv2d::InferOutputShape(TensorShape networkInput) {
    TensorShape shape = prevs.empty() ? networkInput : input->shape;
    TensorShape result;
    result.N = shape.N;
    result.C = weights.shape[0];
    result.H = (shape.H + padH * 2 - weights.shape[2]) / strideH + 1;
    result.W = (shape.W + padW * 2 - weights.shape[3]) / strideW + 1;
    //this->inputShape = shape;
    //this->outputShape = result;
    this->output->InitShape(result);
    return ;

}

size_t Conv2d::GetWorkspaceSize() {
    size_t siz = 0;
    siz += output->shape.NumElements() * 2;
    return siz;
}

void Conv2d::BindWorkspace(void* ptr) {
    float* data = reinterpret_cast<float*>(ptr);
    output->v = data;
    output->delta = data + output->shape.NumElements();
}

void Conv2d::BindDevice(void* ptr) {

    char* addr = reinterpret_cast<char*>(ptr);

    // -------- weights --------
    Tensor w = weights.contiguous();
    size_t w_size = w.numel();
    dl.weights = reinterpret_cast<float*>(addr);
    dl.w_size = w_size;
    dl.in_dim = w_size / w.shape[0];//layer->weights.shape[1];

    //if (w.data()) {
    //    CU_CHECK(cudaMemcpy(
    //        dl.weights,
    //        w.data(),
    //        w_size * sizeof(float),
    //        cudaMemcpyHostToDevice
    //    ));
    //}
    
    addr += w_size * sizeof(float);

    // -------- bias --------
    Tensor bias = b.contiguous();
    size_t b_size = bias.numel();
    dl.bias = reinterpret_cast<float*>(addr);
    dl.b_size = b_size;

    //if (b.data()) {
    //    CU_CHECK(cudaMemcpy(
    //        dl.bias,
    //        bias.data(),
    //        b_size * sizeof(float),
    //        cudaMemcpyHostToDevice
    //    ));
    //}
    
    addr += b_size * sizeof(float);

    // -------- grad_w --------
    dl.grad_w = reinterpret_cast<float*>(addr);
    addr += w_size * sizeof(float);

    // -------- grad_b --------
    dl.grad_b = reinterpret_cast<float*>(addr);
    addr += b_size * sizeof(float);
    
    if (nn->optimizerType == SGD) {

    }
    else if (nn->optimizerType == Adam) {
        //CU_CHECK(cudaMemset(addr, 0, (w_size + b_size) * 2 * sizeof(float)));
        dl.w_m = reinterpret_cast<float*>(addr);
        addr += w_size * sizeof(float);
        dl.b_m = reinterpret_cast<float*>(addr);
        addr += b_size * sizeof(float);
        dl.w_v = reinterpret_cast<float*>(addr);
        addr += w_size * sizeof(float);
        dl.b_v = reinterpret_cast<float*>(addr);
        addr += b_size * sizeof(float);
    }
}

void Conv2d::HostToDevice() {
    Tensor w = weights.contiguous();
    if (w.data() && dl.weights) {
        CU_CHECK(cudaMemcpy(
            dl.weights,
            w.data(),
            dl.w_size * sizeof(float),
            cudaMemcpyHostToDevice
        ));
    }
    Tensor bias = b.contiguous();
    if (b.data() && dl.bias) {
        CU_CHECK(cudaMemcpy(
            dl.bias,
            bias.data(),
            dl.b_size * sizeof(float),
            cudaMemcpyHostToDevice
        ));
    }
}

size_t Conv2d::GetDeviceSize() {
    size_t total = 0;
    if (nn->optimizerType == SGD) {
        size_t w = weights.numel();
        size_t b = this->b.numel();
        total += w;  // weights
        total += w;  // grad_w
        total += b;  // bias
        total += b;  // grad_b
    }
    else if(nn->optimizerType == Adam){
        size_t w = weights.numel();
        size_t b = this->b.numel();
        total += w * 4;  // weights
        total += b * 4;
    }
    
    return total;
}

void Conv2d::forward() {
    /*
    const float* input,      // N * C * H * W -> (CRS) * NPQ
    const float* weights,    // K * (CRS)
    const float* bias,       // K
    float* output,           // N * K * P * Q
    int batch, int C, int H, int W, int R, int S,
    int strideH, int strideW, int K, int P, int Q, float alpha
    */

    dim3 block(TILE_WIDTH, TILE_WIDTH);
    int C = input->shape.C;
    int K = weights.shape[0];
    int R = weights.shape[2];
    int S = weights.shape[3];
    int CRS = C * R * S;
    int NPQ = input->shape.N * output->shape.H * output->shape.W;

    float* input_ptr = input->v;

    dim3 grid((NPQ + TILE_WIDTH - 1) / TILE_WIDTH, ( K + TILE_WIDTH - 1) / TILE_WIDTH);
    conv_forward_kernel <<<grid, block >> > (
        input_ptr,
        dl.weights, 
        dl.bias, 
        output->v, 
        input->shape.N, input->shape.C, input->shape.H, input->shape.W, R, S,
        strideH, strideW, padH, padW,  K, output->shape.H, output->shape.W, alpha);
}

CuLayer* Conv2d::Clone() const{
    Conv2d* layer = new Conv2d();
    //layer->inputShape = this->inputShape;
    //layer->outputShape = this->outputShape;
    layer->output = this->output->Clone();

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

void Conv2d::Save(BinaryStream& stream) const {
    Tensor theWeights;
    theWeights.zeros(weightsShape.N, weightsShape.C, weightsShape.H, weightsShape.W);
    Tensor theBias;
    theBias.zeros(weightsShape.N);
    CU_CHECK(cudaMemcpy(theWeights.data(), dl.weights, dl.w_size * sizeof(float), cudaMemcpyKind::cudaMemcpyDeviceToHost));
    CU_CHECK(cudaMemcpy(theBias.data(), dl.bias, dl.b_size * sizeof(float), cudaMemcpyKind::cudaMemcpyDeviceToHost));

    stream.write<int>((int)LayerType::Conv);
    stream.write<int>(padH);
    stream.write<int>(padW);
    stream.write<int>(strideH);
    stream.write<int>(strideW);

    stream.write<int>(weightsShape.N);
    stream.write<int>(weightsShape.C);
    stream.write<int>(weightsShape.H);
    stream.write<int>(weightsShape.W);
    TensorStream::Save(theWeights, stream);
    TensorStream::Save(theBias, stream);
}

void Conv2d::Load(BinaryStream& stream) {
    LayerType lt = (LayerType)stream.read<int>();
    if (lt != LayerType::Conv) {
        assert(false);
    }
    padH = stream.read<int>();
    padW = stream.read<int>();
    strideH = stream.read<int>();
    strideW = stream.read<int>();
    weightsShape.N = stream.read<int>();
    weightsShape.C = stream.read<int>();
    weightsShape.H = stream.read<int>();
    weightsShape.W = stream.read<int>();
    dl.w_size = weightsShape.NumElements();
    dl.b_size = weightsShape.N;
    weights = TensorStream::Load(weightsShape.N, weightsShape.C, weightsShape.H, weightsShape.W, stream);
    b = TensorStream::Load(weightsShape.N, stream);
    HostToDevice();
}

void Conv2d::backward(const float* delta_next, const float* w_next) {
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

void Conv2d::backwardEx() {
    add = false;
    dgrad();
    wgrad();
    bgrad();
    if (nn->c != 0) {
        regular_grad();
    }
    
}

void Conv2d::applyGradient() {
    if (nn->optimizerType == SGD) {
        int CPQ = weights.numel() / weights.shape[0];
        int K = weights.shape[0];
        int block_y = (K + TILE_WIDTH - 1) / TILE_WIDTH;
        int block_x = (CPQ + TILE_WIDTH - 1) / TILE_WIDTH;
        dim3 grid(block_x, block_y);
        dim3 block(TILE_WIDTH, TILE_WIDTH);
        apply_gradient_kernel << <grid, block >> > (dl.grad_w, dl.grad_b, dl.weights, dl.bias, K, CPQ, nn->learningRate);
    }
    else if (nn->optimizerType == Adam) {
        int total = in_dim * out_dim;
        dim3 block(TILE_WIDTH);
        dim3 grid((total + TILE_WIDTH - 1) / TILE_WIDTH);
        adam_gradient_kernel << <grid, block >> > (dl.grad_w, dl.w_m, dl.w_v, nn->learningRate,
            nn->beta1, nn->beta2, nn->beta1_t, nn->beta2_t, nn->epsilon, dl.weights, total);
        int biasCount = out_dim;
        dim3 grid1((biasCount + TILE_WIDTH - 1) / TILE_WIDTH);
        adam_gradient_kernel << <grid1, block >> > (dl.grad_b, dl.b_m, dl.b_v, nn->learningRate,
            nn->beta1, nn->beta2, nn->beta1_t, nn->beta2_t, nn->epsilon, dl.bias, biasCount);
    }
}

void Conv2d::dgrad() {
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
    
    int N = output->shape.N;
    int K = output->shape.C;
    int P = output->shape.H;
    int Q = output->shape.W;

    int C = weights.shape[1];
    int R = weights.shape[2];
    int S = weights.shape[3];
    int H = input->shape.H;
    int W = input->shape.W;
    int NHW = N * H * W;

    dim3 block(TILE_WIDTH, TILE_WIDTH);
    dim3 grid((NHW + TILE_WIDTH - 1) / TILE_WIDTH, (C + TILE_WIDTH - 1) / TILE_WIDTH);
    conv_dgrad_kernel << <grid, block >> > (
        output->delta,
        dl.weights,
        input->v,
        input->delta,
        prev->add,
        N,
        C,
        H, W, P, Q, R, S, strideH, strideW, padH, padW, K, prev->GetAlpha());
    prev->add = true;
}

void Conv2d::regular_grad()
{
    int total = weights.numel();
    dim3 block(TILE_WIDTH);
    dim3 grid((total + TILE_WIDTH - 1) / TILE_WIDTH);
    regular_kernel << <grid, block >> > (dl.weights, dl.grad_w, total, nn->c);
}

void Conv2d::wgrad() {
    /*
    const float* delta, //NKPQ -> K * NPQ
    const float* a_prev, //NCHW -> NPQ * CRS
    float* grad_w, //KCRS -> K * CRS
    int N, int K, int C, int H, int W, int P, int Q, int R, int S, int strideH, int strideW, int padH, int padW
    */
    int N = input->shape.N;
    int K = output->shape.C;
    int P = output->shape.H;
    int Q = output->shape.W;
    int C = input->shape.C;
    int H = input->shape.H;
    int W = input->shape.W;
    int R = weights.shape[2];
    int S = weights.shape[3];
    int CRS = C * R * S;

    float* prev_activation = input->v;
    
    dim3 block(TILE_WIDTH, TILE_WIDTH);
    dim3 grid((CRS + TILE_WIDTH - 1) / TILE_WIDTH, (K + TILE_WIDTH - 1) / TILE_WIDTH);
    conv_wgrad_kernel << <grid, block >> > (
        output->delta, //NKPQ -> K * NPQ
        prev_activation, //NCRS -> NPQ * CRS
        dl.grad_w, //KCRS -> K * CRS
        N, K, C, H, W, P, Q, R, S, strideH, strideW, padH, padW
        );
}

void Conv2d::bgrad() {
    /*
    conv_bgrad_kernel(       //NHW * 1 I
    const float* delta,    //NCHW -> C * NHW 
    float* grad_b,         //C
    int N, int C, int H, int W
    ) */
    int N = input->shape.N;
    int C = output->shape.C;
    int H = output->shape.H;
    int W = output->shape.W;

    dim3 block(TILE_WIDTH, TILE_WIDTH);
    dim3 grid(1 , (C + TILE_WIDTH - 1) / TILE_WIDTH);
    conv_bgrad_kernel<<<grid, block>>>(
        output->delta,    //NCHW -> C * NHW 
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

void Conv2d::PrintDelta() {

    Tensor tensor(output->shape.N, output->shape.C, output->shape.H, output->shape.W);
    cudaMemcpy(tensor.data(), output->delta, output->shape.NumElements() * sizeof(float), cudaMemcpyDeviceToHost);
    std::cout << "delta:" << std::endl;
    tensor.print_torch_style();
}

void Conv2d::Print() {
    std::cout << "weights:\n";
    //weights.print("W_");
    weights.print_torch_style();
    std::cout << "biases:\n";
    //b.print("B_");
    b.print_torch_style();
    std::cout << std::endl;
}

void Conv2d::PrintGrad() {
    std::cout << "CuConvolutionLayer\n";
    std::cout << "weights_grad:\n";
    weights_grad.print_torch_style();
    std::cout << "bias_grad:\n";
    bias_grad.print_torch_style();
}

void Conv2d::FetchResultToCpu() {
    weights.zeros(weights.shape);
    b.zeros(b.shape);
    CU_CHECK(cudaMemcpy(weights.data(), dl.weights, dl.w_size * sizeof(float), cudaMemcpyKind::cudaMemcpyDeviceToHost));
    CU_CHECK(cudaMemcpy(b.data(), dl.bias, dl.b_size * sizeof(float), cudaMemcpyKind::cudaMemcpyDeviceToHost));
}

void Conv2d::FetchGradToCpu() {
    weights_grad.zeros(weights.shape);
    bias_grad.zeros(b.shape);
    CU_CHECK(cudaMemcpy(weights_grad.data(), dl.grad_w, dl.w_size * sizeof(float), cudaMemcpyKind::cudaMemcpyDeviceToHost));
    CU_CHECK(cudaMemcpy(bias_grad.data(), dl.grad_b, dl.b_size * sizeof(float), cudaMemcpyKind::cudaMemcpyDeviceToHost));
}

void Conv2d::PrintBGrad() {
    bias_grad.zeros(b.shape);
    CU_CHECK(cudaMemcpy(bias_grad.data(), dl.grad_b, dl.b_size * sizeof(float), cudaMemcpyKind::cudaMemcpyDeviceToHost));
    bias_grad.print_torch_style("conv_Bgrad:");
}

void Conv2d::PrintWGrad() {
    weights_grad.zeros(weights.shape);
    CU_CHECK(cudaMemcpy(weights_grad.data(), dl.grad_w, dl.w_size * sizeof(float), cudaMemcpyKind::cudaMemcpyDeviceToHost));
    weights_grad.print_torch_style("conv_Wgrad:");
}

void Conv2d::FetchActivationToCpu() {
    ac.zeros(output->shape.N, output->shape.C, output->shape.H, output->shape.W);
    CU_CHECK(cudaMemcpy(ac.data(), output->v, output->shape.NumElements()* sizeof(float), cudaMemcpyKind::cudaMemcpyDeviceToHost));
}

void Conv2d::PrintActivation() {
    FetchActivationToCpu();
    ac.print_torch_style("conv2d:");
}

void CuLayer::test_cudnn_frontend() {
    //if (is_arch_supported_by_cudnn() == false) {
    //    SKIP("Architecture is not supported by current cudnn version");
    //}
    //namespace fe = cudnn_frontend;

    //// matmul problem size
    //int64_t const b = 16;
    //int64_t const m = 32;
    //int64_t const n = 64;
    //int64_t const k = 128;

    //// Initialize input tensors
    //Surface<half> A_gpu(b * m * k, false);
    //Surface<half> B_gpu(b * k * n, false);

    //// Make cudnn graph
    //fe::graph::Graph graph{};

    //// Create the two non-virtual input tensors A and B.
    //// There are read from global memory.
    //auto A_attributes = fe::graph::Tensor_attributes()
    //    .set_name("A")
    //    .set_dim({ b, m, k })
    //    .set_stride({ m * k, k, 1 })
    //    .set_data_type(fe::DataType_t::BFLOAT16);
    //auto A = graph.tensor(A_attributes);
    //auto B_attributes = fe::graph::Tensor_attributes()
    //    .set_name("B")
    //    .set_dim({ b, k, n })
    //    .set_stride({ k * n, n, 1 })
    //    .set_data_type(fe::DataType_t::BFLOAT16);
    //auto B = graph.tensor(B_attributes);

    //auto matmul_attributes = fe::graph::Matmul_attributes().set_compute_data_type(fe::DataType_t::FLOAT);
    //auto C = graph.matmul(A, B, matmul_attributes);
    //C->set_output(true).set_data_type(fe::DataType_t::FLOAT);

    //std::cout << graph << std::endl;
    //REQUIRE(graph.validate().is_good());

    //// Create a unique_ptr for the cuDNN handle
    //auto handle_ptr = create_cudnn_handle();
    //auto handle = *handle_ptr;

    //REQUIRE(graph.build_operation_graph(handle).is_good());
    //REQUIRE(graph.create_execution_plans({ fe::HeurMode_t::A }).is_good());

    //graph.deselect_engines({ "eng4_" });
    //REQUIRE(graph.check_support().is_good());

    //REQUIRE(graph.build_plans(fe::BuildPlanPolicy_t::ALL).is_good());

    //// Run cudnn graph
    //Surface<float> C_gpu(b * m * n, false);
    //int64_t workspace_size = 0;
    //REQUIRE(graph.get_workspace_size(workspace_size).is_good());
    //Surface<int8_t> workspace(workspace_size, false);

    //std::unordered_map<std::shared_ptr<fe::graph::Tensor_attributes>, void*> variant_pack = {
    //    {A, A_gpu.devPtr}, {B, B_gpu.devPtr}, {C, C_gpu.devPtr} };
    //REQUIRE(graph.execute(handle, variant_pack, workspace.devPtr).is_good());

}
}
