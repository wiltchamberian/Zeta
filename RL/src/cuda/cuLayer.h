#pragma once
#include "tensor.h"
#include "Layer.h"
#include <cudnn.h>

struct TensorShape {
    TensorShape() {}
    TensorShape(int n, int c, int h, int w) :N(n), C(c), H(h), W(w) {}
    int N = 1, C = 1, H = 1, W = 1;

    size_t NumElements() const {
        return (size_t)N * C * H * W;
    }

    size_t Dim() const {
        return C * H * W;
    }

    static TensorShape FromTensor(const Tensor& tensor) {
        if (tensor.rank() == 4) {
            return TensorShape(tensor.shape[0], tensor.shape[1], tensor.shape[2], tensor.shape[3]);
        }
        else if (tensor.rank() == 3) {
            return TensorShape(tensor.shape[0], tensor.shape[1], tensor.shape[2],1);
        }
        else if (tensor.rank() == 2) {
            return TensorShape(tensor.shape[0], tensor.shape[1], 1,1);
        }
        else if (tensor.rank() == 1) {
            return TensorShape(1, tensor.shape[0],1,1);
        }
        assert(false);
        return TensorShape();
    }
};


struct DeviceLayer {
    float* weights = nullptr;
    float* bias = nullptr;
    float* grad_w = nullptr;
    float* grad_b = nullptr;
    int w_size = 0;
    int in_dim = 0;
    int b_size = 0;

    float* activation = nullptr;
    float* delta = nullptr;
    float* delta1 = nullptr;
    
};

class CuNN;

class CuLayer {
public:
    CuLayer(){

    }

    virtual void forward() = 0;
    //virtual void backward(const float* delta_next, const float* w_next) = 0;
    //virtual void dgrad() = 0;
    //virtual void wgrad() = 0;
    //virtual void bgrad() = 0;
    virtual void backwardEx() = 0;
    virtual void applyGradient() = 0;

    //virtual Shape GetInputShape() = 0;
    //virtual Shape GetOutputShape() = 0;
    virtual void InferOutputShape(TensorShape networkInput) = 0;
    virtual size_t GetWorkspaceSize() = 0;
    virtual size_t GetDeviceSize() = 0;
    virtual void BindWorkspace(void* ptr) = 0;
    virtual void BindDevice(void* ptr) = 0;
    virtual float* GetActivation() = 0;
    virtual size_t GetActivationSize() = 0;
    virtual float* GetDelta() = 0;
    virtual size_t GetDeltaSize() = 0;
    virtual void FetchResultToCpu() = 0;
    virtual void FetchGradToCpu() = 0;
    virtual void Print() {}
    virtual void PrintGrad() {}
    
    virtual float GetAlpha() { return 1;  }

    void AddLayer(CuLayer* layer) {
        this->nexts.push_back(layer);
        layer->prevs.push_back(this);
    }
    bool IsRoot() const {
        return prevs.empty();
    }
    bool IsTail() const {
        return nexts.empty();
    }
    //used for backward, flag whether this is the first visit in one backward
    //if it is the first visit, add== false, else make it true
    bool add = false;
    TensorShape inputShape;
    TensorShape outputShape;
    std::vector<CuLayer*> prevs;
    std::vector<CuLayer*> nexts;

    CuNN* nn = nullptr;

    //middle variable
    int visit_count = 0;
    float* prevActivation = nullptr;
};

class CuDefaultLayer :public CuLayer {
public:
    using CuLayer::CuLayer;
    virtual void forward() {}
    virtual void backwardEx() {};
    virtual void InferOutputShape(TensorShape networkInput) {
        this->inputShape = prevs.empty() ? networkInput : prevs[0]->outputShape;
        outputShape.N = 1;
        outputShape.C = 1;
        outputShape.H = 1;
        outputShape.W = 1;
        return;
    };
    virtual void applyGradient() {

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
    virtual float* GetActivation() {
        return nullptr;
    }
    virtual size_t GetActivationSize() { return 0; }
    virtual float* GetDelta() { return nullptr; }
    virtual size_t GetDeltaSize() {
        return 0;
    }
    virtual void FetchResultToCpu() {
        ;
    }
    virtual void FetchGradToCpu() {
        ;
    }
};

class CuInputLayer : public CuDefaultLayer {
public:

};

class CuAddLayer : public CuDefaultLayer {
public:
    using CuDefaultLayer::CuDefaultLayer;
};

class CuLinearLeakyReluLayer :public CuLayer {
public:
    CuLinearLeakyReluLayer(int input, int output);
    void RandomParameters();
    void forward() override;

    void backward(const float* delta_next, const float* w_next);
    void backwardEx();
    void applyGradient();
    void dgrad();
    void wgrad();
    void bgrad();

    void InferOutputShape(TensorShape networkInput) override;
    size_t GetWorkspaceSize();
    size_t GetDeviceSize();
    void BindWorkspace(void* ptr);
    void BindDevice(void* ptr);
    float* GetActivation();
    size_t GetActivationSize();
    float* GetDelta();
    size_t GetDeltaSize();
    float* GetPrevActivation();
    virtual void Print();
    virtual void PrintGrad();
    virtual void FetchResultToCpu();
    virtual void FetchGradToCpu();
    void FetchActivationToCpu();
    void PrintDelta();
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

    int in_dim;
    int out_dim;

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
    CuSoftmaxCrossEntropyLayer() /*:batchSize(0)*/ {
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

    float* GetActivation() {
        return activation;
    }
    size_t GetActivationSize() {
        auto& shape = prevs[0]->outputShape;
        return shape.C * shape.H * shape.W;
    }
    float* GetDelta() {
        assert(false);
        return nullptr;
    }
    size_t GetDeltaSize() {
        assert(false);
        return 0;
    }

    void InferOutputShape(TensorShape networkInput);

    void FetchPredYToCpu();
    void PrintPredY();
    void FetchActivationToCpu();

    Tensor label;
    float* y = nullptr;
    //int batchSize = 0;
    //softmax of input
    float* activation = nullptr;
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
    size_t GetActivationSize();
    float* GetActivation();
    float* GetDelta();
    size_t GetDeltaSize();
    void FetchPredYToCpu();
    void PrintPredY();
    void FetchResultToCpu();
    void Print();

    Tensor label;
    float* y_label = nullptr;
    Tensor predY;

    //device loss
    float* loss = nullptr;
};

/********************convolution layer**************************/
class CuConvolutionLayer :public CuLayer {
public:
    using CuLayer::CuLayer;
    CuConvolutionLayer(int K, int C, int R, int S);
    void RandomParameters();
    void InferOutputShape(TensorShape shape) override;
    size_t GetWorkspaceSize();
    void BindWorkspace(void* ptr);
    void BindDevice(void* ptr);
    size_t GetDeviceSize();
    size_t GetActivationSize();
    float* GetActivation();
    size_t GetDeltaSize();
    float* GetDelta();
    void forward();

    void backward(const float* delta_next, const float* w_next);
    void dgrad();
    void backwardEx();
    void applyGradient();

    void wgrad();

    void bgrad();

    void PrintDelta();
    void Print();
    void PrintGrad();
    void FetchResultToCpu();
    void FetchGradToCpu();
    void FetchActivationToCpu();

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

    int padH = 0;
    int padW = 0;
    int strideH = 1;
    int strideW = 1;
};






