#pragma once

#include "CuNN.h"
#include <cudnn_backend.h>

class DNN : public CuNN {
public:
    DNN();
    ~DNN();
    template<typename T, typename... Args>
    T* CreateDnnLayer(Args&&... args)
    {
        layers.emplace_back(std::make_unique<T>(std::forward<Args>(args)...));
        T* res = static_cast<T*>(layers.back().get());
        res->nn = this;
        res->dnn = this;
        return res;
    }
    cudnnHandle_t handle_ = nullptr;
};