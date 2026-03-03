#include "tensor_operator.h"
#include "cu_tool.h"
#include <cuda_runtime.h>

int ToDevice(const Tensor& tensor, void** addr) {

    Tensor w = tensor.contiguous();
    int l = w.numel() * sizeof(float);
    CU_CHECK(cudaMalloc(addr, l));
    CU_CHECK(cudaMemcpy(*addr, w.data(), l, cudaMemcpyKind::cudaMemcpyHostToDevice));
    return l;
}

void ToTensor(Tensor& tensor, void* addr) {
    CU_CHECK(cudaMemcpy(tensor.data(), addr, tensor.numel()*sizeof(float), cudaMemcpyKind::cudaMemcpyDeviceToHost));
}

void FreeDevice(void* addr) {
    CU_CHECK(cudaFree(addr));
}

