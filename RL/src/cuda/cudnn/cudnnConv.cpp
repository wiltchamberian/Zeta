#include "cudnnConv.h"
#include <cudnn_backend.h>
#include <cudnn_graph.h>
#include "DNN.h"
#include "DnnHelp.h"
#include "cu_tool.h"

CudnnConv::CudnnConv(int K, int C, int R, int S)
{
    weights = Tensor(K, C, R, S);
    b = Tensor(K);
    dl.w_size = K * C * R * S;
    dl.b_size = K;


    DNN_CHECK(cudnnCreateTensorDescriptor(&cudnnIdesc));
    DNN_CHECK(cudnnCreateTensorDescriptor(&cudnnOdesc));

    DNN_CHECK(cudnnCreateFilterDescriptor(&cudnnFdesc));
    DNN_CHECK(cudnnCreateConvolutionDescriptor(&cudnnConvDesc));
    DNN_CHECK(cudnnCreateTensorDescriptor(&cudnnBdesc));
    DNN_CHECK(cudnnCreateActivationDescriptor(&cudnnAdesc));
    
    initImage(weights.data(), dl.w_size * sizeof(float));

    int filterdimA_padded[4] = { K,C,R,S };
    int strideA_padded[4];

    int biasDimA[4] = { 1, K, 1, 1 };
    int biasStrideA[4];
    generateStrides(biasDimA, biasStrideA, 4, filterFormat);
    
    int padA[2] = { padH, padW };
    int convStrideA[2] = { strideH, strideW };
    int dilationA[2] = { 1, 1 };
    //generateStrides(filterdimA_padded, strideA_padded, 4, filterFormat);

    DNN_CHECK(cudnnSetConvolutionNdDescriptor(cudnnConvDesc, convDim, padA, convStrideA, dilationA, mode, CUDNN_DATA_FLOAT));
    DNN_CHECK(cudnnSetFilterNdDescriptor(cudnnFdesc, dataType, filterFormat, convDim + 2, filterdimA_padded));
    DNN_CHECK(cudnnSetTensorNdDescriptor(
        cudnnBdesc,
        dataType,
        4,
        biasDimA,
        biasStrideA
    ));
    DNN_CHECK(cudnnSetActivationDescriptor(
        cudnnAdesc,
        CUDNN_ACTIVATION_IDENTITY,  // activation function
        CUDNN_PROPAGATE_NAN,
        0.0f                    //for Clipped ReLU or elu
    ));
}

CudnnConv::~CudnnConv() {
    if (workSpace) {
        cudaFree(workSpace);
    }
    if (workSpaceBwd) {
        cudaFree(workSpaceBwd);
    }
    DNN_CHECK(cudnnDestroyTensorDescriptor(cudnnIdesc));
    DNN_CHECK(cudnnDestroyTensorDescriptor(cudnnOdesc));

    DNN_CHECK(cudnnDestroyFilterDescriptor(cudnnFdesc));
    DNN_CHECK(cudnnDestroyConvolutionDescriptor(cudnnConvDesc));
    DNN_CHECK(cudnnDestroyTensorDescriptor(cudnnBdesc));
    DNN_CHECK(cudnnDestroyActivationDescriptor(cudnnAdesc));
}

void CudnnConv::BindWorkspace(void* ptr) {
    Conv2d::BindWorkspace(ptr);

    int dimA_padded[4] = {inputShape.N, inputShape.C, inputShape.H, inputShape.W};
    int strideA_padded[4] = {};
    generateStrides(dimA_padded, strideA_padded, 4, filterFormat);
    int outdimA_padded[4] = { outputShape.N, outputShape.C, outputShape.H, outputShape.W };
    int outstrideA_padded[4] = {};
    generateStrides(outdimA_padded, outstrideA_padded, 4, filterFormat);
    DNN_CHECK(cudnnSetTensorNdDescriptor(cudnnIdesc, dataType, convDim + 2, dimA_padded, strideA_padded));
    DNN_CHECK(cudnnSetTensorNdDescriptor(cudnnOdesc, dataType, convDim + 2, outdimA_padded, outstrideA_padded));

    size_t oldSize = workSpaceSize;
    DNN_CHECK(cudnnGetConvolutionForwardWorkspaceSize(
        dnn->handle_, cudnnIdesc, cudnnFdesc, cudnnConvDesc, cudnnOdesc, algo, &workSpaceSize));

    if (workSpace != nullptr) {
        if (workSpaceSize <= oldSize) {
            CU_CHECK(cudaMemset(workSpace, 0, oldSize));
        }
        else {
            CU_CHECK(cudaFree(workSpace));
            CU_CHECK(cudaMalloc(&workSpace, workSpaceSize));
        }
    }
    else {
        CU_CHECK(cudaMalloc(&workSpace, workSpaceSize));
    }

    size_t oldSizeBwd = workSpaceSizeBwd;
    DNN_CHECK(cudnnGetConvolutionBackwardDataWorkspaceSize(
        dnn->handle_, cudnnFdesc, cudnnOdesc, cudnnConvDesc, cudnnIdesc, algo_bwd, &workSpaceSizeBwd));
    if (workSpaceBwd != nullptr) {
        if (workSpaceSizeBwd <= oldSizeBwd) {
            CU_CHECK(cudaMemset(workSpaceBwd, 0, oldSizeBwd));
        }
        else {
            CU_CHECK(cudaFree(workSpaceBwd));
            CU_CHECK(cudaMalloc(&workSpaceBwd, workSpaceSizeBwd));
        }
    }
    else {
        CU_CHECK(cudaMalloc(&workSpaceBwd, workSpaceSizeBwd));
    }

}

void CudnnConv::forward() {
    float* devPtrI = prevs.size() > 0 ? prevs[0]->GetActivation() : prevActivation;
    float* devPtrF = dl.weights;
    float* devPtrO = dl.activation;

    float alpha = 1.0;
    float beta = 0.0;
    /*DNN_CHECK(cudnnConvolutionForward(dnn->handle_,
        (void*)(&alpha),
        cudnnIdesc,
        devPtrI,
        cudnnFdesc,
        devPtrF,
        cudnnConvDesc,
        algo,
        workSpace,
        workSpaceSize,
        (void*)(&beta),
        cudnnOdesc,
        devPtrO));*/

    float alpha1 = 1;
    float alpha2 = 0;
    DNN_CHECK(cudnnConvolutionBiasActivationForward(dnn->handle_,
        &alpha1,
        cudnnIdesc,
        devPtrI,
        cudnnFdesc,
        dl.weights,
        cudnnConvDesc,
        algo,
        workSpace,
        workSpaceSize,
        &alpha2,
        nullptr,
        nullptr,
        cudnnBdesc,
        dl.bias,
        cudnnAdesc,
        cudnnOdesc,
        devPtrO)
        );
}

void CudnnConv::backwardEx() {
    if (prevs.empty()) {
        return;
    }
    auto prev = prevs[0];

    float alpha1 = 1;
    float beta1 = prev->add? 1 : 0;
    DNN_CHECK(cudnnConvolutionBackwardData(dnn->handle_,
        (void*)(&alpha1),
        cudnnFdesc,
        dl.weights,
        cudnnOdesc,
        dl.delta,
        cudnnConvDesc,
        algo_bwd,
        workSpaceBwd,
        workSpaceSizeBwd,
        (void*)(&beta1),
        cudnnIdesc,
        prev->GetDelta()));


    //cudnnActivationBackward()
}