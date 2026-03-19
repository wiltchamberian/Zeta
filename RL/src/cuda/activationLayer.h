#pragma once
#include "cuLayer.h"

namespace zeta {
    class ActivationLayer : public CuDefaultLayer {
    public:
        ActivationLayer();
        ActivationLayer(LayerType lt);
        virtual void forward();
        virtual void backwardEx();
        virtual void applyGradient();
        virtual void InferOutputShape(TensorShape networkInput);
        virtual size_t GetWorkspaceSize();
        virtual size_t GetDeviceSize();
        virtual void BindWorkspace(void* ptr);
        virtual void BindDevice(void* ptr);
        virtual void HostToDevice() override;
        virtual float* GetDelta();
        virtual size_t GetDeltaSize();
        virtual CuLayer* Clone() const override;
        virtual void FetchResultToCpu();
        virtual void FetchGradToCpu();
        virtual Tensor FetchActivationToCpu();
        virtual void Print();
        virtual void PrintGrad();

        //float* y = nullptr;
        //float* dy = nullptr;
        float alpha = 0;

    };
}