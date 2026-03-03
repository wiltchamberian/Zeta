#include "reluLayer.h"
#include "kernels.h"

CuReluLayer::CuReluLayer() {
    layerType = LayerType::Act_Relu;
}

void CuReluLayer::forward() {
    float* inputData = prevs.size() > 0 ? prevs[0]->GetActivation() : prevActivation;
    if (inputData == nullptr) {
        assert(false);
        return;
    }
    int total = inputShape.NumElements();
    dim3 block(TILE_WIDTH);
    dim3 grid((total + TILE_WIDTH - 1) / TILE_WIDTH);
    leaky_relu_forward_kernel << <grid, block >> > (inputData, y, total, alpha);

    //float d[100];
    //cudaMemcpy(d, ys, H * W, cudaMemcpyKind::cudaMemcpyDeviceToHost);
}

void CuReluLayer::backwardEx() {
    if (prevs.empty()) return;
    int total = inputShape.NumElements();
    float* output = prevs[0]->GetDelta();
    dim3 block(TILE_WIDTH);
    dim3 grid((total + TILE_WIDTH - 1) / TILE_WIDTH);
    leaky_relu_backward_kernel << <grid, block >> > (dy, y, output, total, alpha);
}

CuLayer* CuReluLayer::Clone() const {
    auto res =  new CuReluLayer();
    res->inputShape = inputShape;
    res->outputShape = outputShape;
    res->alpha = this->alpha;
    return res;
}