#pragma once
#include "Layer.h"
#include "Activation.h"
#include "tensor.h"
#include <memory>

using Sample = std::vector<double>;

struct Weight {
    int offset;
    int w;
    int h;
};

struct Bias {
    int offset;
    int l;
};

struct DeviceLayer {
    double* weights;
    double* bias;
    double* grad_w;
    double* grad_b;
    int w_size;
    int in_dim;
    int b_size;
};

struct CuNNWorkspace {
    // ----------------- Forward -----------------
    double* x;                           // 输入激活 X, size = batch * in_dim
    std::vector<double*> activations;    // 每层 forward activations 的起始指针

    // ----------------- Backward -----------------
    std::vector<double*> deltas;         // 每层 delta = dC/dA 的起始指针

    // ----------------- 输出 / Loss -----------------
    double* y;           // batch 内标签 y, size = batch * output_dim
    double* loss_vec;    // batch 内每个样本 loss, size = batch
    double* loss;        // 总 loss, size = 1
};


class CuNN
{
public:
    CuNN(double lr = 1.0)
        :learningRate(lr)
        ,deviceMemorySize(0)
        ,deviceMemory(nullptr)
        ,batchSize(0)
        ,deviceWorkspace(nullptr)
    {
    }

    void SetLearningRate(double lr) {
        learningRate = lr;
    }

    void AddLayer(Layer& layer) {
        layers.push_back(layer);
    }

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

protected:
    Tensor input;

protected:
    std::shared_ptr<Activation> activation;

    std::vector<Layer> layers;
    std::vector<Layer> gradLayers;
    double learningRate = 1.0;

    //temp
    Tensor a;
    Tensor z;

    //device memory manager, all used memory use one buffer....
    //I feel it is crazy but this seems most efficent.
    //layout: x, layer1,layer2,...layer_n, y
    //layer_i:
    //W,b,dW,db, z,a,
    void* deviceMemory;
    int deviceMemorySize;

    std::vector<DeviceLayer> deviceLayers;

    void* deviceWorkspace;
    size_t workspaceSize = 0;
    CuNNWorkspace ws;
    int batchSize;



};