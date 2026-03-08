#pragma once

#include "CuNN.h"
#include <cudnn_backend.h>
#include <cublasLt.h>

namespace zeta {
    class DNN : public CuNN {
    public:
        DNN();
        ~DNN();
        template<typename T, typename... Args>
        T* CreateDnnLayer(Args&&... args)
        {
            layers.emplace_back(std::make_unique<T>(std::forward<Args>(args)...));
            T* res = dynamic_cast<T*>(layers.back().get());
            res->nn = this;
            res->dnn = this;
            return res;
        }
        virtual void InitInput(const Tensor& tensor) override;
        virtual void Connect(CuLayer* l1, CuLayer* l2) override;
        virtual CuNN* Clone() const override;
        cudnnHandle_t handle_ = nullptr;
        cublasLtContext* ltHandle = nullptr;
    };
}