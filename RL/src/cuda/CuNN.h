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
    float learningRate = 1.0;

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

    template<typename T, typename... Args>
    T* CreateLayer(Args&&... args)
    {
        layers.emplace_back(std::make_unique<T>(std::forward<Args>(args)...));
        T* res =  static_cast<T*>(layers.back().get());
        res->nn = this;
        return res;
    }

    //reset to the state of just created, only keep the learning rate
    void Clear();

    void SetLearningRate(float lr) {
        learningRate = lr;
    }

    void SetInput(const Tensor& tensor);

    void SetLabel(const Tensor& tensor);

    //deprecated
    TensorShape Build(TensorShape shape);

    void AllocDeviceMemory();

    void AllocWorkSpaceIfNeeded();

    void Forward(const Tensor& x);

    void Backward();

    void Step();

    void FetchGrad();

    void FetchResultToCpu();

    void SetHead(CuLayer* l);
    void SetTail(CuLayer* l);

    float MseLoss(Tensor& xs, Tensor& ys);

    void Print();

    void Travel(std::function<bool(CuLayer*)> ff);

    void TravelBackward(std::function<void(CuLayer*)> ff);

    void PrintGrad();

    size_t GetBatchSize() const;

    void ReleaseDeviceMemory();

    int batchSize = 0;
protected:
    //backup of input and label y
    Tensor input;
    Tensor label;

    CuLayer* head = nullptr;
    CuLayer* tail = nullptr;//to loss
    std::vector<std::unique_ptr<CuLayer>> layers;
    


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