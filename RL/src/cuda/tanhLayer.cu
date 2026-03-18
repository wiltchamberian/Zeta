#include "tanhLayer.h"
#include "kernels.h"

namespace zeta {
    CuTanhLayer::CuTanhLayer() {
        layerType = LayerType::Act_Tanh;
    }

    void CuTanhLayer::forward() {
        float* inputData = input->v;
        if (inputData == nullptr) {
            assert(false);
            return;
        }
        int total = input->shape.NumElements();
        dim3 block(TILE_WIDTH);
        dim3 grid((total + TILE_WIDTH - 1) / TILE_WIDTH);
        tanh_forward_kernel << <grid, block >> > (inputData, output->v, total);
    }

    void CuTanhLayer::backwardEx() {
        if (prevs.empty()) return;
        int total = input->shape.NumElements();
        dim3 block(TILE_WIDTH);
        dim3 grid((total + TILE_WIDTH - 1) / TILE_WIDTH);
        tanh_backward_kernel << <grid, block >> > (output->delta, output->v, input->delta, total);
    }

    CuLayer* CuTanhLayer::Clone() const {
        CuTanhLayer* r = new CuTanhLayer();
        return r;
    }

}

