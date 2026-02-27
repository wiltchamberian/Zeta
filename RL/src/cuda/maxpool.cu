#include "maxpool.h"
#include "kernels.h"
#include "CuNN.h"

void MaxPool2d::forward() {
    const float* in = prevs.empty()?prevActivation :prevs[0]->GetActivation();
    int N = nn->batchSize * outputShape.C;
    int H = outputShape.H;
    int W = outputShape.W;

    //float dd[500];
    //cudaMemcpy(dd, in, H * W * 4 * N * sizeof(float), cudaMemcpyKind::cudaMemcpyDeviceToHost);

    dim3 block(TILE_SMALL, TILE_SMALL, TILE_SMALL);
    dim3 grid((W+ TILE_SMALL -1)/ TILE_SMALL, (H + TILE_SMALL - 1) / TILE_SMALL, (N + TILE_SMALL - 1) / TILE_SMALL);
    max_pool_2d_forward_kernel << <grid, block >> > (in, activation, maxIndex, N, h, w, H, W );

    

    /*float d[500];
    cudaMemcpy(d, activation, H * W * N * sizeof(float), cudaMemcpyKind::cudaMemcpyDeviceToHost);
    int m[500];
    cudaMemcpy(m, maxIndex, H * W * N * sizeof(int), cudaMemcpyKind::cudaMemcpyDeviceToHost);*/
}

void MaxPool2d::backwardEx() {
    assert(!prevs.empty());
    int N = nn->batchSize * outputShape.C;
    int H = outputShape.H;
    int W = outputShape.W;
    dim3 block(TILE_SMALL, TILE_SMALL, TILE_SMALL);
    dim3 grid((W + TILE_SMALL - 1) / TILE_SMALL, (H + TILE_SMALL - 1) / TILE_SMALL, (N + TILE_SMALL - 1) / TILE_SMALL);
    cudaMemset(prevs[0]->GetDelta(), 0, inputShape.NumElements() * sizeof(float));
    max_pool_2d_backward_kernel << <grid, block >> > (dC_da, maxIndex, prevs[0]->GetDelta(), N, H, W);

    
}

void MaxPool2d::applyGradient() {
    return;
}

void MaxPool2d::InferOutputShape(TensorShape networkInput) {
    if (prevs.empty()) {
        inputShape = networkInput;
    }
    else {
        inputShape = prevs[0]->outputShape;
    }

    outputShape.N = inputShape.N;
    outputShape.C = inputShape.C;
    outputShape.H = inputShape.H / h;
    outputShape.W = inputShape.W / w;
}

size_t MaxPool2d::GetWorkspaceSize() {
    return outputShape.NumElements() * 3;
}

size_t MaxPool2d::GetDeviceSize() {
    return 0;
}

void MaxPool2d::BindWorkspace(void* ptr) {
    float* f = reinterpret_cast<float*>(ptr);
    activation = f;
    f += outputShape.NumElements();
    dC_da = f;
    f += outputShape.NumElements();
    maxIndex = reinterpret_cast<int*>(f);
    return;
}

void MaxPool2d::BindDevice(void* ptr) {
    return;
}

float* MaxPool2d::GetActivation() {
    return activation;
}

size_t MaxPool2d::GetActivationSize() {
    return outputShape.NumElements();
}

float* MaxPool2d::GetDelta() {
    return dC_da;
}

size_t MaxPool2d::GetDeltaSize() {
    return outputShape.NumElements();
}

CuLayer* MaxPool2d::Clone() const {
    MaxPool2d* r = new MaxPool2d();
    r->inputShape = inputShape;
    r->outputShape = outputShape;
    r->h = h;
    r->w = w;
    return r;
}

void MaxPool2d::FetchResultToCpu() {

}

void MaxPool2d::FetchGradToCpu() {

}

void MaxPool2d::Print() {}

void MaxPool2d::PrintGrad() {}

float MaxPool2d::GetAlpha() { return 1; }