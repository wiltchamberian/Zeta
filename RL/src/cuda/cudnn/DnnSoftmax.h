#pragma once

#include "cuLayer.h"
#include <cudnn.h>
#include <cudnn_backend.h>

class DNN;

class DnnSoftmax : public CuSoftmaxCrossEntropyLayer {
public:
    void forward() override;
    void backwardEx() override;
    void BindWorkspace(void* ptr) override;
    virtual void SetNN(CuNN* nn) override;
    virtual CuLayer* Clone() const override;
    cudnnSoftmaxAlgorithm_t algo = CUDNN_SOFTMAX_ACCURATE;
    cudnnSoftmaxMode_t mode = CUDNN_SOFTMAX_MODE_INSTANCE;
    DNN* dnn = nullptr;
};