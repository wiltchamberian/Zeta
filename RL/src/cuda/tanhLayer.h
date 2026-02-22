#pragma once
#include "cuLayer.h"


class CuTanhLayer : public CuDefaultLayer {
public:
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
    virtual void FetchResultToCpu();
    virtual void FetchGradToCpu();
    virtual void Print();
    virtual void PrintGrad();

    float* ys = nullptr;
    float* delta = nullptr;
};