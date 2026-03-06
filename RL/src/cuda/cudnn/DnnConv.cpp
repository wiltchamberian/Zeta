#include "DnnConv.h"
#include <cudnn_backend.h>
#include <cudnn_graph.h>
#include "DNN.h"
#include "DnnHelp.h"
#include "cu_tool.h"
#include "kernels.h"
#include "DnnTensor.h"

DnnConv::DnnConv(int K, int C, int R, int S, Size2D padding, Size2D stride)
    :Conv2d(K,C,R,S,padding,stride)
{
    DNN_CHECK(cudnnCreateFilterDescriptor(&cudnnFdesc));
    DNN_CHECK(cudnnCreateConvolutionDescriptor(&cudnnConvDesc));
    DNN_CHECK(cudnnCreateTensorDescriptor(&cudnnBdesc));
    DNN_CHECK(cudnnCreateActivationDescriptor(&cudnnAdesc));
    
    //initImage(weights.data(), dl.w_size);

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

DnnConv::~DnnConv() {
    if (workSpace) {
        cudaFree(workSpace);
    }
    if (workSpaceBwd) {
        cudaFree(workSpaceBwd);
    }

    DNN_CHECK(cudnnDestroyFilterDescriptor(cudnnFdesc));
    DNN_CHECK(cudnnDestroyConvolutionDescriptor(cudnnConvDesc));
    DNN_CHECK(cudnnDestroyTensorDescriptor(cudnnBdesc));
    DNN_CHECK(cudnnDestroyActivationDescriptor(cudnnAdesc));
}

void DnnConv::BindWorkspace(void* ptr) {
    Conv2d::BindWorkspace(ptr);
    output->Create();

    size_t oldSize = workSpaceSize;
    auto status = cudnnGetConvolutionForwardWorkspaceSize(
        dnn->handle_, input->desc->cudnnDesc, cudnnFdesc, cudnnConvDesc, output->desc->cudnnDesc, algo, &workSpaceSize);
    if (status != CUDNN_STATUS_SUCCESS) {
        std::cout << "error!" << std::endl;
        assert(false);
    }
    workSpaceReAlloc(&workSpace, workSpaceSize, oldSize);
   

    size_t oldSizeBwd = workSpaceSizeBwd;
    DNN_CHECK(cudnnGetConvolutionBackwardDataWorkspaceSize(
        dnn->handle_, cudnnFdesc, output->desc->cudnnDesc, cudnnConvDesc, input->desc->cudnnDesc, algo_bwd, &workSpaceSizeBwd));
    workSpaceReAlloc(&workSpaceBwd, workSpaceSizeBwd, oldSizeBwd);
    
    size_t oldSizeFilterBwd = workSpaceFilterSizeBwd;
    DNN_CHECK(cudnnGetConvolutionBackwardFilterWorkspaceSize(
        dnn->handle_, input->desc->cudnnDesc, output->desc->cudnnDesc, cudnnConvDesc, cudnnFdesc, algo_filter_bwd, &workSpaceFilterSizeBwd));
    workSpaceReAlloc(&workSpaceFilterBwd, workSpaceFilterSizeBwd, oldSizeFilterBwd);


}

void DnnConv::forward() {
    float* devPtrI = input->v;
    float* devPtrF = dl.weights;
    float* devPtrO = output->v;

    float alpha = 1.0;
    float beta = 0.0;
    DNN_CHECK(cudnnConvolutionForward(dnn->handle_,
        (void*)(&alpha),
        input->desc->cudnnDesc,
        devPtrI,
        cudnnFdesc,
        devPtrF,
        cudnnConvDesc,
        algo,
        workSpace,
        workSpaceSize,
        (void*)(&beta),
        output->desc->cudnnDesc,
        devPtrO));

    int C = output->shape.C;
    int HW = output->shape.H * output->shape.W;
    int NCHW = output->shape.NumElements();
    dim3 block(TILE_WIDTH);
    dim3 grid((NCHW + TILE_WIDTH - 1) / TILE_WIDTH);
    tensor_add_bias_kernel << <grid, block >> > (devPtrO, dl.bias, HW,C,NCHW);
    /*float alpha1 = 1.0f;
    float beta1 = 1.0f;
    auto status = cudnnAddTensor(dnn->handle_, &alpha1, cudnnBdesc, dl.bias, &beta1, cudnnOdesc, devPtrO);
    if (status != CUDNN_STATUS_SUCCESS) {
        std::cout << "error!" << std::endl;
    }*/
    /*float alpha1 = 1.0f;
    float alpha2 = 0.0f;
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
        );*/
}

void DnnConv::backwardEx() {
    add = false;
    dgrad();
    wgrad();
    bgrad();
    if (nn->c != 0) {
        regular_grad();
    }
}

void DnnConv::dgrad() {
    if (prevs.empty()) {
        return;
    }

    auto prev = prevs[0];
    float alpha1 = 1;
    float beta1 = prev->add ? 1 : 0;
    DNN_CHECK(cudnnConvolutionBackwardData(dnn->handle_,
        (void*)(&alpha1),
        cudnnFdesc,
        dl.weights,
        output->desc->cudnnDesc,
        output->delta,
        cudnnConvDesc,
        algo_bwd,
        workSpaceBwd,
        workSpaceSizeBwd,
        (void*)(&beta1),
        input->desc->cudnnDesc,
        input->delta));
    prev->add = true;
}

void DnnConv::wgrad() {
    float* x = input->v;
    float alpha2 = 1.0f;
    float beta2 = 0.0f;
    auto status = cudnnConvolutionBackwardFilter(dnn->handle_,
        &alpha2,
        input->desc->cudnnDesc,
        x,
        output->desc->cudnnDesc, //should be the same as DyDesc,
        output->delta,
        cudnnConvDesc,
        algo_filter_bwd,
        workSpaceFilterBwd,
        workSpaceFilterSizeBwd,
        &beta2,
        cudnnFdesc,
        dl.grad_w
    );
    if (status != CUDNN_STATUS_SUCCESS) {
        assert(false);
    }
}

void DnnConv::bgrad() {
    float alpha2 = 1.0f;
    float beta2 = 0.0f;
    auto status = cudnnConvolutionBackwardBias(dnn->handle_,
        &alpha2,
        output->desc->cudnnDesc,
        output->delta,
        &beta2,
        cudnnBdesc,
        dl.grad_b);
    if (status != CUDNN_STATUS_SUCCESS) {
        assert(false);
    }
}

void DnnConv::workSpaceReAlloc(void** workSpace, size_t& workSpaceSize, size_t oldSize) {
    if (*workSpace != nullptr) {
        if (workSpaceSize <= oldSize) {
            CU_CHECK(cudaMemset(*workSpace, 0, oldSize));
        }
        else {
            CU_CHECK(cudaFree(*workSpace));
            CU_CHECK(cudaMalloc(workSpace, workSpaceSize));
        }
    }
    else {
        if (workSpaceSize > 0) {
            CU_CHECK(cudaMalloc(workSpace, workSpaceSize));
        }
    }
}