#pragma once
#include "activationLayer.h"
#include <cudnn_ops.h>

class DNN;

class DnnActLayer : public ActivationLayer {
public:
    DnnActLayer();
    DnnActLayer(LayerType lt);
    ~DnnActLayer();
    void Init(LayerType lt);
    void forward() override;
    void backwardEx() override;
    void BindWorkspace(void* ptr) override;
    cudnnActivationDescriptor_t actDesc = nullptr;
    cudnnTensorDescriptor_t xDesc = nullptr;
    cudnnTensorDescriptor_t yDesc = nullptr;
    cudnnTensorDescriptor_t dyDesc = nullptr;
    DNN* dnn = nullptr;
};
