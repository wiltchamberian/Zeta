#include "DnnTensor.h"
#include <cudnn_backend.h>

#include "DnnHelp.h"

DnnTensorDescriptor::~DnnTensorDescriptor() {
    Destroy();
}

void DnnTensorDescriptor::Create(CuTensor* tensor) {
    if (cudnnDesc == nullptr) {
        DNN_CHECK(cudnnCreateTensorDescriptor(&cudnnDesc));

        cudnnDataType_t dataType = CUDNN_DATA_FLOAT;
        int dims[4] = { tensor->shape.N, tensor->shape.C, tensor->shape.H, tensor->shape.W };
        int strides[4] = {};
        generateStrides(dims, strides, 4, CUDNN_TENSOR_NCHW);
        DNN_CHECK(cudnnSetTensorNdDescriptor(
            cudnnDesc,
            dataType,
            4,
            dims,
            strides
        ));
    }

}

void DnnTensorDescriptor::Destroy() {
    if (cudnnDesc != nullptr) {
        DNN_CHECK(cudnnDestroyTensorDescriptor(cudnnDesc));
    }
}

BlasDescriptor::~BlasDescriptor() {
    Destroy();
}

void BlasDescriptor::Create(int batch, int dim) {
    if (layout == nullptr) {
        // X, row-major
        BLAS_CHECK(cublasLtMatrixLayoutCreate(
            &layout,
            CUDA_R_32F,
            batch, dim, dim
        ));
        cublasLtOrder_t order = CUBLASLT_ORDER_ROW;
        BLAS_CHECK(cublasLtMatrixLayoutSetAttribute(
            layout,
            CUBLASLT_MATRIX_LAYOUT_ORDER,
            &order,
            sizeof(order)
        ));
    }

}

void BlasDescriptor::Destroy() {
    if (layout) {
        BLAS_CHECK(cublasLtMatrixLayoutDestroy(layout));
        layout = nullptr;
    }
}

DnnTensor::~DnnTensor() {
    if (desc) {
        delete desc;
    }
    if (blasDesc) {
        delete blasDesc;
    }
}

void DnnTensor::Create() {
    Destroy();
    desc = new DnnTensorDescriptor();
    desc->Create(this);
    blasDesc = new BlasDescriptor();
    blasDesc->Create(shape.N, shape.Dim());
}

void DnnTensor::Destroy() {
    if (desc) {
        delete desc;
        desc = nullptr;
    }
    if (blasDesc) {
        delete blasDesc;
        blasDesc = nullptr;
    }
}

TensorShape DnnTensor::getTensorShape(const Tensor& x) {
    TensorShape ts;
    int rk = x.rank();
    if (rk == 1) {
        ts.N = x.shape[0];
    }
    else if (rk == 2) {
        ts.N = x.shape[0];
        ts.C = x.shape[1];
    }
    else if (rk == 3) {
        ts.N = x.shape[0];
        ts.C = x.shape[1];
        ts.H = x.shape[2];
    }
    else if (rk == 4) {
        ts.N = x.shape[0];
        ts.C = x.shape[1];
        ts.H = x.shape[2];
        ts.W = x.shape[3];
    }
    else {
        assert(false);
    }
    return ts;
}