#pragma once
#include "Define.h"

#ifdef ZETA_LIB_EXPORTS
#define DLL_API __declspec(dllexport)
#else
#define DLL_API __declspec(dllimport)
#endif

using DnnHandle = void*;
using TensorHandle = void*;

struct LayerHandle {
    void* h;
    LayerType type;
};

struct TsShape {
    int N;
    int B;
    int H;
    int W;
};

extern "C" {
    DLL_API DnnHandle CreateDnn();
    DLL_API void DestroyDnn(DnnHandle hd);
    DLL_API void ClearDnn(DnnHandle hd);
    DLL_API void SetLearningRate(DnnHandle handle, float lr);
    DLL_API LayerHandle CreateConv2d(DnnHandle handle, TsShape shape, Size2D padding, Size2D stride);
    DLL_API LayerHandle CreateConv2dNoBias(DnnHandle handle, TsShape shape, Size2D padding, Size2D stride);
    DLL_API LayerHandle CreateLinear(DnnHandle handle, int input, int output);
    DLL_API LayerHandle CreateMse(DnnHandle handle);
    DLL_API LayerHandle CreateAdd(DnnHandle handle);
    DLL_API LayerHandle CreateAct(DnnHandle handle, LayerType layerType);
    DLL_API LayerHandle CreateSoftmax(DnnHandle handle);
    DLL_API LayerHandle CreateOutput(DnnHandle handle);
    DLL_API void SetWeights(LayerHandle handle, float* weights, int n);
    DLL_API void AddEdge(LayerHandle from, LayerHandle to);
    DLL_API void AllocDeviceMemory(DnnHandle h);
    DLL_API void FetchWeights(LayerHandle h, float* d, int* n);
    DLL_API TensorHandle CreateTensor1d(int l);
    DLL_API TensorHandle CreateTensor2d(int n, int b);
    DLL_API TensorHandle CreateTensor3d(int n, int b, int h);
    DLL_API TensorHandle CreateTensor4d(int n, int b, int h, int w);
    DLL_API void TensorCopyFrom(TensorHandle h, float* from, int x);
    DLL_API void BindLabelToDevice(LayerHandle, float* tensor, int n);
    DLL_API void Backward(DnnHandle h);
    DLL_API void Foward(DnnHandle h, TensorHandle tensor);
    DLL_API void Step(DnnHandle h);
    DLL_API DnnHandle Clone(DnnHandle h);
    DLL_API float FetchLoss(LayerHandle h);
    //may cause gpu to cpu devlier
    DLL_API void Print(TensorHandle h);

}