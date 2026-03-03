#include "dnnAct.h"
#include <cudnn_backend.h>
#include <cudnn.h>
#include "DNN.h"
#include "DnnHelp.h"

DnnActLayer::DnnActLayer()
:ActivationLayer(){
    Init(LayerType::Act_Identity);
}

DnnActLayer::DnnActLayer(LayerType lt) {
    Init(layerType);
}

void DnnActLayer::Init(LayerType lt) {
    layerType = lt;
    DNN_CHECK(cudnnCreateTensorDescriptor(&xDesc));
    DNN_CHECK(cudnnCreateTensorDescriptor(&yDesc));
    DNN_CHECK(cudnnCreateTensorDescriptor(&dyDesc));
    DNN_CHECK(cudnnCreateActivationDescriptor(&actDesc));
    cudnnActivationMode_t mode = CUDNN_ACTIVATION_IDENTITY;
    cudnnNanPropagation_t reluNanOpt = CUDNN_PROPAGATE_NAN;

    switch (lt) {
    case LayerType::Act_Relu:
        {
        mode = CUDNN_ACTIVATION_RELU;
        }
        break;
    case LayerType::Act_Tanh: {
        mode = CUDNN_ACTIVATION_TANH;
        }
        break;
    case LayerType::Act_Sigmoid: {
        mode = CUDNN_ACTIVATION_SIGMOID;
        }
        break;
    case LayerType::Act_ClippedRelu: {
        mode = CUDNN_ACTIVATION_CLIPPED_RELU;
        }
        break;
    case LayerType::Act_Elu: {
        mode = CUDNN_ACTIVATION_ELU;
        }
        break;
    case LayerType::Act_Identity: {
        mode = CUDNN_ACTIVATION_IDENTITY;
        }
        break;
    case LayerType::Act_SWISH: {
        mode = CUDNN_ACTIVATION_SWISH;
        }
        break;
    default:
        mode = CUDNN_ACTIVATION_IDENTITY;
        break;
    }

    DNN_CHECK(cudnnSetActivationDescriptor(
        actDesc,
        mode,          // 激活类型
        reluNanOpt,    // NaN 处理方式
        0           // 某些激活用到的参数
    ));
}

DnnActLayer::~DnnActLayer() {
    DNN_CHECK(cudnnDestroyTensorDescriptor(xDesc));
    DNN_CHECK(cudnnDestroyTensorDescriptor(yDesc));
    DNN_CHECK(cudnnDestroyTensorDescriptor(dyDesc));
    DNN_CHECK(cudnnDestroyActivationDescriptor(actDesc));
}

void DnnActLayer::forward() {
    float* x = prevs.size() > 0 ? prevs[0]->GetActivation() : prevActivation;
    if (x == nullptr) {
        assert(false);
        return;
    }

    float alpha1 = 1;
    float beta1 = 0;
    DNN_CHECK(cudnnActivationForward(dnn->handle_,
        actDesc,
        &alpha1,
        xDesc,
        x,
        &beta1,
        yDesc,
        y
    ));
}

void DnnActLayer::backwardEx() {
    if (prevs.empty()) return;
    float* x = prevs[0]->GetActivation();
    float* dx = prevs[0]->GetDelta();

    float alpha1 = 1;
    float beta1 = 0;
    DNN_CHECK(cudnnActivationBackward(dnn->handle_,
        actDesc,
        &alpha1,
        yDesc,
        y,
        dyDesc,
        dy,
        xDesc,
        x,
        &beta1,
        xDesc,
        dx
    ));
}

void DnnActLayer::BindWorkspace(void* ptr) {
    ActivationLayer::BindWorkspace(ptr);
    cudnnDataType_t dataType = CUDNN_DATA_FLOAT;

    int dimA[4] = {inputShape.N, inputShape.C, inputShape.H, inputShape.W};
    int strideA[4] = {};
    cudnnTensorFormat_t tensorFormat = CUDNN_TENSOR_NCHW;
    generateStrides(dimA, strideA, 4, tensorFormat);
    cudnnSetTensorNdDescriptor(xDesc, dataType, 4, dimA, strideA);

    int dimB[4] = { outputShape.N, outputShape.C, outputShape.H, outputShape.W };
    int strideB[4] = {};
    generateStrides(dimB, strideB, 4, tensorFormat);
    cudnnSetTensorNdDescriptor(yDesc, dataType, 4, dimB, strideB);
    cudnnSetTensorNdDescriptor(dyDesc, dataType, 4, dimB, strideB);
}