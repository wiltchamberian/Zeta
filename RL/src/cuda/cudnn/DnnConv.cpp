#include "DnnConv.h"
#include <cudnn_backend.h>
#include <cudnn_graph.h>
#include "DNN.h"
#include "DnnHelp.h"
#include "cu_tool.h"
#include "kernels.h"

DnnConv::DnnConv(int K, int C, int R, int S, Size2D padding, Size2D stride)
{
    weights = Tensor(K, C, R, S);
    b = Tensor(K);
    dl.w_size = K * C * R * S;
    dl.b_size = K;

    padH = padding.h;
    padW = padding.w;
    strideH = stride.h;
    strideW = stride.w;

    DNN_CHECK(cudnnCreateTensorDescriptor(&cudnnIdesc));
    DNN_CHECK(cudnnCreateTensorDescriptor(&cudnnOdesc));

    DNN_CHECK(cudnnCreateFilterDescriptor(&cudnnFdesc));
    DNN_CHECK(cudnnCreateConvolutionDescriptor(&cudnnConvDesc));
    DNN_CHECK(cudnnCreateTensorDescriptor(&cudnnBdesc));
    DNN_CHECK(cudnnCreateActivationDescriptor(&cudnnAdesc));
    
    initImage(weights.data(), dl.w_size);

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
    DNN_CHECK(cudnnDestroyTensorDescriptor(cudnnIdesc));
    DNN_CHECK(cudnnDestroyTensorDescriptor(cudnnOdesc));

    DNN_CHECK(cudnnDestroyFilterDescriptor(cudnnFdesc));
    DNN_CHECK(cudnnDestroyConvolutionDescriptor(cudnnConvDesc));
    DNN_CHECK(cudnnDestroyTensorDescriptor(cudnnBdesc));
    DNN_CHECK(cudnnDestroyActivationDescriptor(cudnnAdesc));
}

void DnnConv::BindWorkspace(void* ptr) {
    Conv2d::BindWorkspace(ptr);

    int dimA_padded[4] = {input->shape.N, input->shape.C, input->shape.H, input->shape.W};
    int strideA_padded[4] = {};
    generateStrides(dimA_padded, strideA_padded, 4, filterFormat);
    int outdimA_padded[4] = { output->shape.N, output->shape.C, output->shape.H, output->shape.W };
    int outstrideA_padded[4] = {};
    generateStrides(outdimA_padded, outstrideA_padded, 4, filterFormat);
    DNN_CHECK(cudnnSetTensorNdDescriptor(cudnnIdesc, dataType, convDim + 2, dimA_padded, strideA_padded));
    DNN_CHECK(cudnnSetTensorNdDescriptor(cudnnOdesc, dataType, convDim + 2, outdimA_padded, outstrideA_padded));

    size_t oldSize = workSpaceSize;
    auto status = cudnnGetConvolutionForwardWorkspaceSize(
        dnn->handle_, cudnnIdesc, cudnnFdesc, cudnnConvDesc, cudnnOdesc, algo, &workSpaceSize);
    if (status != CUDNN_STATUS_SUCCESS) {
        std::cout << "error!" << std::endl;
        assert(false);
    }
    workSpaceReAlloc(&workSpace, workSpaceSize, oldSize);
   

    size_t oldSizeBwd = workSpaceSizeBwd;
    DNN_CHECK(cudnnGetConvolutionBackwardDataWorkspaceSize(
        dnn->handle_, cudnnFdesc, cudnnOdesc, cudnnConvDesc, cudnnIdesc, algo_bwd, &workSpaceSizeBwd));
    workSpaceReAlloc(&workSpaceBwd, workSpaceSizeBwd, oldSizeBwd);
    
    size_t oldSizeFilterBwd = workSpaceFilterSizeBwd;
    DNN_CHECK(cudnnGetConvolutionBackwardFilterWorkspaceSize(
        dnn->handle_, cudnnIdesc, cudnnOdesc, cudnnConvDesc, cudnnFdesc, algo_filter_bwd, &workSpaceFilterSizeBwd));
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
        cudnnOdesc,
        output->delta,
        cudnnConvDesc,
        algo_bwd,
        workSpaceBwd,
        workSpaceSizeBwd,
        (void*)(&beta1),
        cudnnIdesc,
        input->delta));
    
}

void DnnConv::wgrad() {
    float* x = input->v;
    float alpha2 = 1.0f;
    float beta2 = 0.0f;
    auto status = cudnnConvolutionBackwardFilter(dnn->handle_,
        &alpha2,
        cudnnIdesc,
        x,
        cudnnOdesc, //should be the same as DyDesc,
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
        cudnnOdesc,
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