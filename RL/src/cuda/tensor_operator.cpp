#include "tensor_operator.h"
#include "cu_tool.h"
#include <cuda_runtime.h>

int ToDevice(const Tensor& tensor, void** addr) {

    Tensor w = tensor.contiguous();
    int l = w.numel() * sizeof(double);
    CUDA_CHECK(cudaMalloc(addr, l));
    CUDA_CHECK(cudaMemcpy(*addr, w.data(), l, cudaMemcpyKind::cudaMemcpyHostToDevice));
    return l;
}

void ToTensor(Tensor& tensor, void* addr) {
    CUDA_CHECK(cudaMemcpy(tensor.data(), addr, tensor.numel()*sizeof(double), cudaMemcpyKind::cudaMemcpyDeviceToHost));
}

void FreeDevice(void* addr) {
    CUDA_CHECK(cudaFree(addr));
}

