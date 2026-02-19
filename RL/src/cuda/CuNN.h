#pragma once
#include "Layer.h"
#include "Activation.h"
#include "tensor.h"
#include "cuLayer.h"
#include <memory>
#include <functional>

using Sample = std::vector<float>;


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
    float* x = nullptr;                           // 输入激活 X, size = batch * in_dim
    // ----------------- 输出 / Loss -----------------
    float* y = nullptr;           // batch 内标签 y, size = batch * output_dim
    float* loss_vec = nullptr;    // batch 内每个样本 loss, size = batch
    float* loss = nullptr;        // 总 loss, size = 1
};

class CuHead {
public:
    std::vector<float> policy;
    float value;
};

class CuNN
{
public:
    CuNN(float lr = 1.0)
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

    void SetLearningRate(float lr) {
        learningRate = lr;
    }

    void SetInput(const Tensor& tensor);

    void SetLabel(const Tensor& tensor);

    void AddLayer(std::shared_ptr<CuLayer> layer);

    TensorShape Build(TensorShape shape);

    void AllocDeviceMemory();

    void AllocWorkSpaceIfNeeded();

    void Forward(const Tensor& x);
    Tensor ForwardAndFetchPredY(const Tensor& x);

    void Backward(const Tensor& y);

    void Step();

    void FetchGrad();

    void FetchResultToCpu();

    void SetHead(std::shared_ptr<CuLayer> l);
    void SetTail(std::shared_ptr<CuLayer> l);

    float MseLoss(Tensor& xs, Tensor& ys);

    void Train(Tensor& xs, Tensor& ys, int maxEpochs, float tolerance);

    void Print();

    void Travel(std::function<void(CuLayer*)> ff);

    void TravelBackward(std::function<void(CuLayer*)> ff);

    void PrintGrad();

    size_t GetBatchSize() const;

    void ReleaseDeviceMemory();

protected:
    //backup of input and label y
    Tensor input;
    Tensor label;

    std::shared_ptr<CuLayer> head = nullptr;
    std::shared_ptr<CuLayer> tail = nullptr;//to loss
    std::vector<std::shared_ptr<CuLayer>> layers;
    float learningRate = 1.0;


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