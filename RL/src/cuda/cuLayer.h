#pragma once
#include "tensor.h"
#include "Layer.h"
#include "CuTensor.h"
//#include <cudnn.h>
#include <fstream>
//#include <cudnn_frontend.h>



//used for padding and stride
struct Size2D {
    int h;
    int w;
};

struct DeviceLayer {
    float* weights = nullptr;
    float* bias = nullptr;
    float* grad_w = nullptr;
    float* grad_b = nullptr;
    int w_size = 0;
    int in_dim = 0;
    int b_size = 0;

    DeviceLayer PesudoClone() const {
        DeviceLayer res;
        res.w_size = this->w_size;
        res.in_dim = this->in_dim;
        res.b_size = this->b_size;
        return res;
    }
};

class CuNN;

enum class LayerType {
    Basic,
    Fully,
    Conv,
    Activation,
    Act_Relu,
    Act_Tanh,
    Act_Sigmoid,
    Act_ClippedRelu,
    Act_Elu,
    Act_Identity,
    Act_SWISH,
    Mse,
    Softmax
};



class CuLayer {
public:
    CuLayer(){

    }

    virtual void forward() = 0;
    virtual void backwardEx() = 0;
    virtual void applyGradient() = 0;

    virtual void InferOutputShape(TensorShape networkInput) = 0;
    virtual size_t GetWorkspaceSize() = 0;
    virtual size_t GetDeviceSize() = 0;
    virtual void BindWorkspace(void* ptr) = 0;
    virtual void BindDevice(void* ptr) = 0;
    float* GetActivation();
    size_t GetActivationSize();
    float* GetDelta();
    size_t GetDeltaSize();
    virtual void FetchResultToCpu() = 0;
    virtual void FetchGradToCpu() = 0;
    virtual CuLayer* Clone() const = 0;
    virtual void Print() {}
    virtual void PrintGrad() {}
    virtual void Save(std::fstream fs) {}
    virtual float GetAlpha() { return 1;  }

    CuLayer* AddLayer(CuLayer* layer);
    bool IsRoot() const {
        return prevs.empty();
    }
    bool IsTail() const {
        return nexts.empty();
    }
    //used for backward, flag whether this is the first visit in one backward
    //if it is the first visit, add== false, else make it true
    bool add = false;
    //TensorShape inputShape;
    //TensorShape outputShape;
    std::vector<CuLayer*> prevs;
    std::vector<CuLayer*> nexts;
    //used for operation fuse
    CuLayer* forward_next = nullptr; //forward fuse
    CuLayer* backward_prev = nullptr; //backward fuse

    CuTensor* input = nullptr;
    CuTensor* output = nullptr;

    CuNN* nn = nullptr;

    LayerType layerType = LayerType::Basic;

    //middle variable
    int visit_count = 0;
    //float* prevActivation = nullptr;

    //a hack, only used for clone but dont copy it while cloning
    CuLayer* ref = nullptr;

    void test_cudnn_frontend();
};

class CuDefaultLayer :public CuLayer {
public:
    using CuLayer::CuLayer;
    virtual void forward() {}
    virtual void backwardEx() {};
    virtual void InferOutputShape(TensorShape networkInput) {
    };
    virtual void applyGradient() {

    }
    virtual CuLayer* Clone() const {
        return nullptr;
    }
    virtual size_t GetWorkspaceSize() {
        return 0;
    };
    virtual size_t GetDeviceSize() {
        return 0;
    }
    virtual void BindWorkspace(void* ptr) {
        return ;
    }
    virtual void BindDevice(void* ptr) {
        return ;
    }

    virtual void FetchResultToCpu() {
        ;
    }
    virtual void FetchGradToCpu() {
        ;
    }
};

class CuAddLayer : public CuDefaultLayer {
public:
    using CuDefaultLayer::CuDefaultLayer;
    virtual CuLayer* Clone() const {
        CuAddLayer* add = new CuAddLayer();
        add->output = this->output->Clone();
        add->visit_count = 0;
        return add;
    }
};

class OutputLayer : public CuDefaultLayer {
public:
    using CuDefaultLayer::CuDefaultLayer;
    virtual CuLayer* Clone() const {
        OutputLayer* add = new OutputLayer();
        add->output = this->output->Clone();
        add->visit_count = 0;
        return add;
    }
};


class CuLinearLeakyReluLayer :public CuLayer {
public:
    CuLinearLeakyReluLayer();
    CuLinearLeakyReluLayer(int input, int output);
    void RandomParameters();
    void forward() override;
    void backwardEx();
    void applyGradient();
    void dgrad();
    void wgrad();
    void bgrad();
    void regular_grad();
    void InferOutputShape(TensorShape networkInput) override;
    size_t GetWorkspaceSize();
    size_t GetDeviceSize();
    void BindWorkspace(void* ptr);
    void BindDevice(void* ptr);
    float* GetDelta();
    size_t GetDeltaSize();
    float* GetPrevActivation();
    virtual void Print();
    virtual void PrintGrad();
    virtual void FetchResultToCpu();
    virtual void FetchGradToCpu();
    virtual CuLayer* Clone() const;
    void FetchActivationToCpu();
    void PrintDelta();
    virtual void Save(std::fstream fs);
    float GetAlpha() {
        return alpha;
    }
    Tensor& data() {
        return weights;
    }

    Tensor weights;
    Tensor b;

    Tensor weights_grad;
    Tensor bias_grad;

    int in_dim = 0;
    int out_dim = 0;

    DeviceLayer dl;
    float alpha = 1.0;

    Tensor ac;
};

class CuLinearTanhLayer : public CuLinearLeakyReluLayer {
public:
    using CuLinearLeakyReluLayer::CuLinearLeakyReluLayer;
    void forward();
    void backwardEx();
    void dgrad();
};

class CuSoftmaxCrossEntropyLayer : public CuDefaultLayer {
public:
    CuSoftmaxCrossEntropyLayer()  {
        layerType = LayerType::Softmax;
    }
    float FetchLoss();
    void forward();
    void backwardEx();
    void applyGradient();
    void BindLabelToDevice();
    void BindWorkspace(void* ptr);
    size_t GetWorkspaceSize();
    void BindDevice(void* ptr);
    size_t GetDeviceSize();
    CuLayer* Clone() const;

    void InferOutputShape(TensorShape networkInput);

    void FetchPredYToCpu();
    void PrintPredY();
    void FetchActivationToCpu();

    Tensor label;
    float* yLabel = nullptr;
    //int batchSize = 0;
    //softmax of input
    //float* activation = nullptr;
    float* loss = nullptr;

    //
    Tensor distribution;
};

class CuMseLayer :public CuDefaultLayer {
public:
    CuMseLayer();
    CuMseLayer(int C);
    CuMseLayer(int C, int H);
    CuMseLayer(int C, int H, int W);
    float FetchLoss();
    void forward();
    void backwardEx();
    void applyGradient();

    void InferOutputShape(TensorShape shape);
    void BindLabelToDevice();
    size_t GetWorkspaceSize();
    void BindWorkspace(void* ptr);
    size_t GetDeviceSize();
    void BindDevice(void* ptr);
    void FetchPredYToCpu();
    void PrintPredY();
    void FetchResultToCpu();
    void Print();
    CuLayer* Clone() const;
    Tensor label;
    float* y_label = nullptr;
    Tensor predY;

    //device loss
    float* loss = nullptr;
};

/********************convolution layer**************************/
class Conv2d : public CuLayer {
public:
    using CuLayer::CuLayer;
    Conv2d();
    Conv2d(int K, int C, int R, int S, Size2D padding = { 0,0 }, Size2D stride = { 1,1 });
    virtual ~Conv2d();
    void RandomParameters();
    void InferOutputShape(TensorShape shape) override;
    size_t GetWorkspaceSize();
    void BindWorkspace(void* ptr);
    void BindDevice(void* ptr);
    size_t GetDeviceSize();
    void forward();
    CuLayer* Clone() const;
    void backward(const float* delta_next, const float* w_next);
    void dgrad();
    void backwardEx();
    void applyGradient();

    void wgrad();
    void regular_grad();
    void bgrad();

    void PrintDelta();
    void Print();
    void PrintGrad();
    
    void FetchResultToCpu();
    void FetchGradToCpu();
    void FetchActivationToCpu();
    void FetchDeltaToCpu();

    float GetAlpha() {
        return alpha;
    }

    Tensor& data() {
        return weights;
    }

    Tensor weights;
    Tensor b;

    Tensor weights_grad;
    Tensor bias_grad;

    Tensor ac;

    int in_dim;
    int out_dim;

    DeviceLayer dl;

    float alpha = 1.0;

protected:
    int padH = 0;
    int padW = 0;
    int strideH = 1;
    int strideW = 1;

   
};

using Linear = CuLinearLeakyReluLayer;



