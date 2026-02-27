#include "reluLayer.h"
#include "kernels.h"

void CuReluLayer::forward() {
    float* inputData = prevs.size() > 0 ? prevs[0]->GetActivation() : prevActivation;
    if (inputData == nullptr) {
        assert(false);
        return;
    }
    int total = inputShape.NumElements();
    dim3 block(TILE_WIDTH);
    dim3 grid((total + TILE_WIDTH - 1) / TILE_WIDTH);
    leaky_relu_forward_kernel << <grid, block >> > (inputData, ys, total, alpha);

    //float d[100];
    //cudaMemcpy(d, ys, H * W, cudaMemcpyKind::cudaMemcpyDeviceToHost);
}

void CuReluLayer::backwardEx() {
    if (prevs.empty()) return;
    int total = inputShape.NumElements();
    float* output = prevs[0]->GetDelta();
    dim3 block(TILE_WIDTH);
    dim3 grid((total + TILE_WIDTH - 1) / TILE_WIDTH);
    leaky_relu_backward_kernel << <grid, block >> > (delta, ys, output, total, alpha);
}

void CuReluLayer::applyGradient() {

}

void CuReluLayer::InferOutputShape(TensorShape networkInput) {
    if (!prevs.empty()) {
        inputShape = prevs[0]->outputShape;
        outputShape = inputShape;
    }
    else {
        inputShape = networkInput;
        outputShape = inputShape;
    }
}

void CuReluLayer::BindWorkspace(void* ptr) {
    float* p = reinterpret_cast<float*>(ptr);
    ys = p;
    p += inputShape.NumElements();
    delta = p;
    return;
}


size_t CuReluLayer::GetWorkspaceSize() {
    return inputShape.NumElements() * 2;
}

void CuReluLayer::BindDevice(void* ptr) {
    return;
}

size_t CuReluLayer::GetDeviceSize() {
    return 0;
}


float* CuReluLayer::GetActivation() {
    return ys;
}

size_t CuReluLayer::GetActivationSize() {
    return inputShape.NumElements();
}

float* CuReluLayer::GetDelta() {
    return delta;
}

size_t CuReluLayer::GetDeltaSize() {
    return inputShape.NumElements();
}

CuLayer* CuReluLayer::Clone() const {
    auto res =  new CuReluLayer();
    res->inputShape = inputShape;
    res->outputShape = outputShape;
    res->alpha = this->alpha;
    return res;
}

void CuReluLayer::FetchResultToCpu() {

}

void CuReluLayer::FetchGradToCpu() {

}

void CuReluLayer::Print() {
}

void CuReluLayer::PrintGrad() {
}

