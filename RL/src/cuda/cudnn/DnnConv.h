#pragma once

#include "cuLayer.h"
#include <cudnn.h>
#include <cudnn_backend.h>

namespace zeta {
    class DNN;

    class DnnConv : public Conv2d {
    public:
        DnnConv() {}
        DnnConv(int K, int C, int R, int S, Size2D padding = { 0,0 }, Size2D stride = { 1,1 }, bool useBias = true);
        virtual ~DnnConv() override;
        void BindWorkspace(void* ptr) override;
        void forward() override;
        void backwardEx() override;
        void SetNN(CuNN* nn) override;
        void dgrad();
        void wgrad();
        void bgrad();
        CuLayer* Clone() const override;
        virtual void Load(BinaryStream& stream) override;

        void init(int K, int C, int R, int S);
        //protected:
        void workSpaceReAlloc(void** workSpace, size_t& siz, size_t oldSize);
        cudnnDataType_t dataType = CUDNN_DATA_FLOAT;
        cudnnTensorFormat_t filterFormat = CUDNN_TENSOR_NCHW;
        cudnnConvolutionMode_t mode = CUDNN_CROSS_CORRELATION;
        cudnnConvolutionFwdAlgo_t algo = CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_GEMM;
        cudnnConvolutionBwdDataAlgo_t algo_bwd = CUDNN_CONVOLUTION_BWD_DATA_ALGO_0;
        cudnnConvolutionBwdFilterAlgo_t algo_filter_bwd = CUDNN_CONVOLUTION_BWD_FILTER_ALGO_1;

        cudnnFilterDescriptor_t cudnnFdesc;
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

        //backward filter workspace
        void* workSpaceFilterBwd = nullptr;
        size_t workSpaceFilterSizeBwd = 0;

        DNN* dnn = nullptr;
    };
}
