#include "tanhLayer.h"
#include "kernels.h"

void CuTanhLayer::forward() {
    float* inputData = prevs.size() > 0 ? prevs[0]->GetActivation() : prevActivation;
    if (inputData == nullptr) {
        assert(false);
        return;
    }
    int total = inputShape.NumElements();
    dim3 block(TILE_WIDTH);
    dim3 grid((total + TILE_WIDTH-1)/TILE_WIDTH);
    tanh_forward_kernel << <grid, block >> > (inputData, ys, total);
}

void CuTanhLayer::backwardEx() {
    if (prevs.empty()) return;
    int total = inputShape.NumElements();
    float* output = prevs[0]->GetDelta();
    dim3 block(TILE_WIDTH);
    dim3 grid((total + TILE_WIDTH - 1) / TILE_WIDTH);
    tanh_backward_kernel << <grid, block >> > (delta, ys, output, total);
}

void CuTanhLayer::applyGradient() {

}

void CuTanhLayer::InferOutputShape(TensorShape networkInput) {
    if (!prevs.empty()) {
        inputShape = prevs[0]->outputShape;
        outputShape = inputShape;
    }
    else {
        inputShape = networkInput;
        outputShape = inputShape;
    }
}

void CuTanhLayer::BindWorkspace(void* ptr) {
    float* p = reinterpret_cast<float*>(ptr);
    ys = p;
    p += inputShape.NumElements();
    delta = p;
    return;
}


size_t CuTanhLayer::GetWorkspaceSize() {
    return inputShape.NumElements() * 2;
}

void CuTanhLayer::BindDevice(void* ptr) {
    return;
}

size_t CuTanhLayer::GetDeviceSize() {
    return 0;
}


float* CuTanhLayer::GetActivation() {
    return ys;
}

size_t CuTanhLayer::GetActivationSize() {
    return inputShape.NumElements();
}

float* CuTanhLayer::GetDelta() {
    return delta;
}

size_t CuTanhLayer::GetDeltaSize() {
    return inputShape.NumElements();
}

void CuTanhLayer::FetchResultToCpu() {

}

void CuTanhLayer::FetchGradToCpu() {

}

void CuTanhLayer::Print() {
}

void CuTanhLayer::PrintGrad() {
}
