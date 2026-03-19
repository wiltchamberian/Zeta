#include "activationLayer.h"
#include "kernels.h"
#include "cu_tool.h"
#include <cuda_runtime.h>

namespace zeta {
    ActivationLayer::ActivationLayer() {
        layerType = LayerType::Activation;
    }

    ActivationLayer::ActivationLayer(LayerType lt) {
        layerType = lt;
    }

    void ActivationLayer::forward() {

    }

    void ActivationLayer::backwardEx() {

    }

    void ActivationLayer::applyGradient() {

    }

    void ActivationLayer::InferOutputShape(TensorShape networkInput) {
        output->shape = prevs.empty() ? networkInput : input->shape;
    }

    void ActivationLayer::BindWorkspace(void* ptr) {
        float* p = reinterpret_cast<float*>(ptr);
        output->v = p;
        p += input->shape.NumElements();
        output->delta = p;
        return;
    }


    size_t ActivationLayer::GetWorkspaceSize() {
        return input->shape.NumElements() * 2;
    }

    void ActivationLayer::BindDevice(void* ptr) {
        return;
    }

    void ActivationLayer::HostToDevice() {

    }

    size_t ActivationLayer::GetDeviceSize() {
        return 0;
    }

    float* ActivationLayer::GetDelta() {
        return output->delta;
    }

    size_t ActivationLayer::GetDeltaSize() {
        return input->shape.NumElements();
    }

    CuLayer* ActivationLayer::Clone() const {
        auto res = new ActivationLayer();
        res->layerType = layerType;
        res->alpha = this->alpha;
        return res;
    }

    void ActivationLayer::FetchResultToCpu() {

    }

    void ActivationLayer::FetchGradToCpu() {

    }

    Tensor ActivationLayer::FetchActivationToCpu() {
        int siz = output->shape.NumElements();
        Tensor tensor(output->shape.N, output->shape.C, output->shape.H, output->shape.W);
        CU_CHECK(cudaMemcpy(tensor.data(), output->v, siz * sizeof(float), cudaMemcpyKind::cudaMemcpyDeviceToHost));
        return tensor;
    }

    void ActivationLayer::Print() {
    }

    void ActivationLayer::PrintGrad() {
    }
}

