#pragma once

#include "cuLayer.h"
#include <cudnn.h>
#include <cudnn_backend.h>

class DNN;

class CudnnConv : public Conv2d {
public:
    CudnnConv(int K, int C, int R, int S);
    ~CudnnConv() override;
    void BindWorkspace(void* ptr) override;
    void forward() override;
    void backwardEx() override;
    cudnnDataType_t dataType = CUDNN_DATA_FLOAT;
    cudnnTensorFormat_t filterFormat = CUDNN_TENSOR_NCHW;
    cudnnConvolutionMode_t mode = CUDNN_CROSS_CORRELATION;
    cudnnConvolutionFwdAlgo_t algo = CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_PRECOMP_GEMM;
    cudnnConvolutionBwdDataAlgo_t algo_bwd = CUDNN_CONVOLUTION_BWD_DATA_ALGO_0;
    cudnnTensorDescriptor_t cudnnIdesc;
    cudnnFilterDescriptor_t cudnnFdesc;
    cudnnTensorDescriptor_t cudnnOdesc;
    cudnnTensorDescriptor_t cudnnBdesc;
    cudnnActivationDescriptor_t cudnnAdesc;
    cudnnConvolutionDescriptor_t cudnnConvDesc;

    int convDim = 2;
    //workspace, TODO(change to global later)
    void* workSpace = nullptr;
    size_t workSpaceSize = 0;

    //backward workspace
    void* workSpaceBwd = nullptr;
    size_t workSpaceSizeBwd = 0;

    DNN* dnn = nullptr;
};
