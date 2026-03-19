#pragma once
#include "cuLayer.h"

namespace zeta {
    class MaxPool2d : public CuLayer {
    public:
        MaxPool2d();
        MaxPool2d(int h, int w);
        virtual void forward();
        virtual void backwardEx();
        virtual void applyGradient();

        virtual void InferOutputShape(TensorShape networkInput);
        virtual size_t GetWorkspaceSize();
        virtual size_t GetDeviceSize();
        virtual void BindWorkspace(void* ptr);
        virtual void BindDevice(void* ptr);
        virtual void HostToDevice() override;
        virtual CuLayer* Clone() const override;
        virtual void FetchResultToCpu();
        virtual void FetchGradToCpu();
        virtual void Print();
        virtual void PrintGrad();

        virtual float GetAlpha();

        //float* activation = nullptr;
        //float* dC_da = nullptr;
        int* maxIndex = nullptr;
        int h = 2;
        int w = 2;
    };
}

