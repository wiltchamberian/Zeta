#include "reluLayer.h"
#include "kernels.h"

namespace zeta {
    CuReluLayer::CuReluLayer() {
        layerType = LayerType::Act_Relu;
    }

    void CuReluLayer::forward() {
        float* inputData = input->v;
        if (inputData == nullptr) {
            assert(false);
            return;
        }
        int total = input->shape.NumElements();
        dim3 block(TILE_WIDTH);
        dim3 grid((total + TILE_WIDTH - 1) / TILE_WIDTH);
        leaky_relu_forward_kernel << <grid, block >> > (inputData, output->v, total, alpha);

        //float d[100];
        //cudaMemcpy(d, ys, H * W, cudaMemcpyKind::cudaMemcpyDeviceToHost);
    }

    void CuReluLayer::backwardEx() {
        if (prevs.empty()) return;
        int total = input->shape.NumElements();
        dim3 block(TILE_WIDTH);
        dim3 grid((total + TILE_WIDTH - 1) / TILE_WIDTH);
        leaky_relu_backward_kernel << <grid, block >> > (output->delta, output->v, input->delta, total, alpha);
    }

    CuLayer* CuReluLayer::Clone() const {
        auto res = new CuReluLayer();
        //res->inputShape = inputShape;
        //res->outputShape = outputShape;
        res->output = output->Clone();
        res->alpha = this->alpha;
        return res;
    }
}