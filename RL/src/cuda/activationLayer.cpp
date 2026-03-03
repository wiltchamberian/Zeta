#include "activationLayer.h"
#include "kernels.h"

ActivationLayer::ActivationLayer() {
    layerType = LayerType::Activation;
}

void ActivationLayer::forward() {

}

void ActivationLayer::backwardEx() {

}

void ActivationLayer::applyGradient() {

}

void ActivationLayer::InferOutputShape(TensorShape networkInput) {
    if (!prevs.empty()) {
        inputShape = prevs[0]->outputShape;
        outputShape = inputShape;
    }
    else {
        inputShape = networkInput;
        outputShape = inputShape;
    }
}

void ActivationLayer::BindWorkspace(void* ptr) {
    float* p = reinterpret_cast<float*>(ptr);
    y = p;
    p += inputShape.NumElements();
    dy = p;
    return;
}


size_t ActivationLayer::GetWorkspaceSize() {
    return inputShape.NumElements() * 2;
}

void ActivationLayer::BindDevice(void* ptr) {
    return;
}

size_t ActivationLayer::GetDeviceSize() {
    return 0;
}


float* ActivationLayer::GetActivation() {
    return y;
}

size_t ActivationLayer::GetActivationSize() {
    return inputShape.NumElements();
}

float* ActivationLayer::GetDelta() {
    return dy;
}

size_t ActivationLayer::GetDeltaSize() {
    return inputShape.NumElements();
}

CuLayer* ActivationLayer::Clone() const {
    auto res = new ActivationLayer();
    res->inputShape = inputShape;
    res->outputShape = outputShape;
    res->alpha = this->alpha;
    return res;
}

void ActivationLayer::FetchResultToCpu() {

}

void ActivationLayer::FetchGradToCpu() {

}

void ActivationLayer::Print() {
}

void ActivationLayer::PrintGrad() {
}

