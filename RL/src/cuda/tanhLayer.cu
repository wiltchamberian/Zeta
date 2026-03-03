#include "tanhLayer.h"
#include "kernels.h"

CuTanhLayer::CuTanhLayer() {
    layerType = LayerType::Act_Tanh;
}

void CuTanhLayer::forward() {
    float* inputData = prevs.size() > 0 ? prevs[0]->GetActivation() : prevActivation;
    if (inputData == nullptr) {
        assert(false);
        return;
    }
    int total = inputShape.NumElements();
    dim3 block(TILE_WIDTH);
    dim3 grid((total + TILE_WIDTH-1)/TILE_WIDTH);
    tanh_forward_kernel << <grid, block >> > (inputData, y, total);
}

void CuTanhLayer::backwardEx() {
    if (prevs.empty()) return;
    int total = inputShape.NumElements();
    float* output = prevs[0]->GetDelta();
    dim3 block(TILE_WIDTH);
    dim3 grid((total + TILE_WIDTH - 1) / TILE_WIDTH);
    tanh_backward_kernel << <grid, block >> > (dy, y, output, total);
}

CuLayer* CuTanhLayer::Clone() const {
    CuTanhLayer* r = new CuTanhLayer();
    r->inputShape = inputShape;
    r->outputShape = outputShape;
    return r;
}

