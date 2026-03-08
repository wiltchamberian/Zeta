#pragma once

#include "cuLayer.h"
#include <cudnn.h>
#include <cudnn_backend.h>
#include "maxpool.h"

namespace zeta {
    class DNN;

    class DnnPooling : public MaxPool2d {
    public:
        DnnPooling();
        DnnPooling(int h, int w);
        ~DnnPooling();
        void forward() override;
        void backwardEx() override;
        void BindWorkspace(void* ptr) override;
        void SetNN(CuNN*) override;
        CuLayer* Clone() const override;
        void init(int h, int w);
        cudnnNanPropagation_t nanProg = CUDNN_PROPAGATE_NAN;
        cudnnPoolingMode_t mode = CUDNN_POOLING_MAX;
        cudnnPoolingDescriptor_t PDesc;
        DNN* dnn = nullptr;
    };
}