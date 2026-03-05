#include "maxpool.h"
#include "kernels.h"
#include "CuNN.h"

MaxPool2d::MaxPool2d() {


}

MaxPool2d::MaxPool2d(int H, int W)
:h(H)
,w(W){

}

void MaxPool2d::forward() {
    const float* in = input->v;
    int N = nn->batchSize * output->shape.C;
    int H = output->shape.H;
    int W = output->shape.W;

    //float dd[500];
    //cudaMemcpy(dd, in, H * W * 4 * N * sizeof(float), cudaMemcpyKind::cudaMemcpyDeviceToHost);

    dim3 block(TILE_SMALL, TILE_SMALL, TILE_SMALL);
    dim3 grid((W+ TILE_SMALL -1)/ TILE_SMALL, (H + TILE_SMALL - 1) / TILE_SMALL, (N + TILE_SMALL - 1) / TILE_SMALL);
    max_pool_2d_forward_kernel << <grid, block >> > (in, output->v, maxIndex, N, h, w, H, W );

    

    /*float d[500];
    cudaMemcpy(d, activation, H * W * N * sizeof(float), cudaMemcpyKind::cudaMemcpyDeviceToHost);
    int m[500];
    cudaMemcpy(m, maxIndex, H * W * N * sizeof(int), cudaMemcpyKind::cudaMemcpyDeviceToHost);*/
}

void MaxPool2d::backwardEx() {
    assert(!prevs.empty());
    int N = nn->batchSize * output->shape.C;
    int H = output->shape.H;
    int W = output->shape.W;
    dim3 block(TILE_SMALL, TILE_SMALL, TILE_SMALL);
    dim3 grid((W + TILE_SMALL - 1) / TILE_SMALL, (H + TILE_SMALL - 1) / TILE_SMALL, (N + TILE_SMALL - 1) / TILE_SMALL);
    cudaMemset(input->delta, 0, input->shape.NumElements() * sizeof(float));
    max_pool_2d_backward_kernel << <grid, block >> > (output->delta, maxIndex, input->delta, N, H, W);

    
}

void MaxPool2d::applyGradient() {
    return;
}

void MaxPool2d::InferOutputShape(TensorShape networkInput) {
    //if (prevs.empty()) {
    //    inputShape = networkInput;
    //}
    //else {
    //    inputShape = prevs[0]->outputShape;
    //}

    //outputShape.N = inputShape.N;
    //outputShape.C = inputShape.C;
    //outputShape.H = inputShape.H / h;
    //outputShape.W = inputShape.W / w;

    TensorShape shape = prevs.empty() ? networkInput : input->shape;
    TensorShape result;
    result.N = shape.N;
    result.C = shape.C;
    result.H = shape.H / h;
    result.W = shape.W / w;
    output->shape = result;
}

size_t MaxPool2d::GetWorkspaceSize() {
    return output->shape.NumElements() * 3;
}

size_t MaxPool2d::GetDeviceSize() {
    return 0;
}

void MaxPool2d::BindWorkspace(void* ptr) {
    float* f = reinterpret_cast<float*>(ptr);
    output->v = f;
    f += output->shape.NumElements();
    output->delta = f;
    f += output->shape.NumElements();
    maxIndex = reinterpret_cast<int*>(f);
    return;
}

void MaxPool2d::BindDevice(void* ptr) {
    return;
}

CuLayer* MaxPool2d::Clone() const {
    MaxPool2d* r = new MaxPool2d();
    //r->inputShape = inputShape;
    //r->outputShape = outputShape;
    r->output = output->Clone();

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