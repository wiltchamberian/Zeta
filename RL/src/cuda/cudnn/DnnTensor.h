#pragma once

#include "CuTensor.h"
#include <cudnn_ops.h>
#include <cublasLt.h>

class DnnTensorDescriptor {
public:
    ~DnnTensorDescriptor();
    void Create(CuTensor* tensor);
    void Destroy();
    cudnnTensorDescriptor_t cudnnDesc;
};

class BlasDescriptor {
public:
    ~BlasDescriptor();
    void Create(int batch, int dim);
    void Destroy();
    cublasLtMatrixLayoutOpaque_t* layout = nullptr;
};

class DnnTensor : public CuTensor {
public:
    ~DnnTensor();
    void Create() override;
    void Destroy();
    CuTensor* Clone() const override;

    static TensorShape getTensorShape(const Tensor& tensor);
};