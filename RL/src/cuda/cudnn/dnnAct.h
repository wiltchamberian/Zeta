#pragma once
#include "activationLayer.h"
#include <cudnn_ops.h>

namespace zeta {
    class DNN;

    class DnnAct : public ActivationLayer {
    public:
        DnnAct();
        DnnAct(LayerType lt);
        ~DnnAct();
        void Init(LayerType lt);
        void forward() override;
        void backwardEx() override;
        void BindWorkspace(void* ptr) override;
        void SetNN(CuNN* nn) override;
        CuLayer* Clone() const override;
        cudnnActivationDescriptor_t actDesc = nullptr;
        cudnnTensorDescriptor_t xDesc = nullptr;
        cudnnTensorDescriptor_t yDesc = nullptr;
        cudnnTensorDescriptor_t dyDesc = nullptr;
        DNN* dnn = nullptr;
    };
}
