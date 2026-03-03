#pragma once
#include "cuLayer.h"

class ActivationLayer : public CuDefaultLayer {
public:
    ActivationLayer();
    virtual void forward();
    virtual void backwardEx();
    virtual void applyGradient();
    virtual void InferOutputShape(TensorShape networkInput);
    virtual size_t GetWorkspaceSize();
    virtual size_t GetDeviceSize();
    virtual void BindWorkspace(void* ptr);
    virtual void BindDevice(void* ptr);
    virtual float* GetActivation();
    virtual size_t GetActivationSize();
    virtual float* GetDelta();
    virtual size_t GetDeltaSize();
    virtual CuLayer* Clone() const override;
    virtual void FetchResultToCpu();
    virtual void FetchGradToCpu();
    virtual void Print();
    virtual void PrintGrad();

    float* y = nullptr;
    float* dy = nullptr;
    float alpha = 0;

};