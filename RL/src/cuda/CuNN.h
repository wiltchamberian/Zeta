#pragma once
#include "Layer.h"
#include "Activation.h"
#include "tensor.h"
#include "cuLayer.h"
#include <memory>

using Sample = std::vector<double>;


//struct Weight {
//    int offset;
//    int w;
//    int h;
//};
//
//struct Bias {
//    int offset;
//    int l;
//};



struct CuNNWorkspace {
    void Clear() {
        x = nullptr;
        y = nullptr;
        loss_vec = nullptr;
        loss = nullptr;
    }
    // ----------------- Forward -----------------
    double* x = nullptr;                           // 输入激活 X, size = batch * in_dim
    // ----------------- 输出 / Loss -----------------
    double* y = nullptr;           // batch 内标签 y, size = batch * output_dim
    double* loss_vec = nullptr;    // batch 内每个样本 loss, size = batch
    double* loss = nullptr;        // 总 loss, size = 1
};


class CuNN
{
public:
    CuNN(double lr = 1.0)
        :learningRate(lr)
        ,deviceMemorySize(0)
        ,deviceMemory(nullptr)
        ,deviceWorkspace(nullptr)
    {
    }

    ~CuNN() {
        ReleaseDeviceMemory();
    }

    //reset to the state of just created, only keep the learning rate
    void Clear();

    void SetLearningRate(double lr) {
        learningRate = lr;
    }

    void AddLayer(std::unique_ptr<CuLayer> layer);

    void Build(TensorShape shape);

    void AllocDeviceMemory();

    void AllocWorkSpaceIfNeeded();

    void Forward(const Tensor& x);
    Tensor ForwardAndFetch(const Tensor& x);

    void Backward(Tensor& x, Tensor& y);

    void Step();

    void FetchResultToCpu();

    double MseLoss(Tensor& xs, Tensor& ys);

    void Train(Tensor& xs, Tensor& ys, int maxEpochs, double tolerance);

    void Print();

    void PrintGrad();

    size_t GetBatchSize() const;

    void ReleaseDeviceMemory();

protected:

    std::vector<std::unique_ptr<CuLayer>> layers;
    double learningRate = 1.0;


    //device memory manager, all used memory use one buffer....
    //I feel it is crazy but this seems most efficent.
    //layout: x, layer1,layer2,...layer_n, y
    //layer_i:
    //W,b,dW,db, z,a,
    void* deviceMemory;
    int deviceMemorySize;

    void* deviceWorkspace;
    size_t workspaceSize = 0;
    CuNNWorkspace ws;

};