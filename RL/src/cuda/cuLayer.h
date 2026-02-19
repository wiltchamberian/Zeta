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

    //virtual Shape GetInputShape() = 0;
    //virtual Shape GetOutputShape() = 0;
    virtual TensorShape InferOutputShape(TensorShape shape) = 0;
    virtual size_t GetWorkspaceSize() = 0;
    virtual size_t GetDeviceSize() = 0;
    virtual void BindWorkspace(void* ptr) = 0;
    virtual void BindDevice(void* ptr) = 0;
    virtual float* GetActivation() = 0;
    virtual size_t GetActivationSize() = 0;
    virtual float* GetDelta() = 0;
    virtual size_t GetDeltaSize() = 0;

    void AddLayer(CuLayer* layer) {
        this->nexts.push_back(layer);
        layer->prevs.push_back(this);
    }
    TensorShape inputShape;
    TensorShape outputShape;
    std::vector<CuLayer*> prevs;
    std::vector<CuLayer*> nexts;

    

    //middle variable
    int visit_count = 0;
    float* prevActivation = nullptr;
};

class CuLinearLeakyReluLayer :public CuLayer {
public:
    using CuLayer::CuLayer;
    CuLinearLeakyReluLayer(int input, int output)
    :in_dim(input)
    ,out_dim(output){
        weights = Tensor(output, input); //reverse order to level up computation performance
        b = Tensor(output);
    }

    void forward() override;

    void backward(const float* delta_next, const float* w_next);
    void backwardEx();
    void dgrad();
    void wgrad();
    void bgrad();

    TensorShape InferOutputShape(TensorShape shape) override;
    size_t GetWorkspaceSize();
    size_t GetDeviceSize();
    void BindWorkspace(void* ptr);
    void BindDevice(void* ptr);
    float* GetActivation();
    size_t GetActivationSize();
    float* GetDelta();
    size_t GetDeltaSize();
    float* GetPrevActivation();

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
};

class CuSoftmaxCrossEntropyLayer : public CuLayer {
public:
    CuSoftmaxCrossEntropyLayer(int batchSize) :batchSize(batchSize) {

    }
    void forward();
    void backwardEx();

    size_t GetDeviceSize() {
        auto& shape = prevs[0]->outputShape;
        return shape.C * shape.H * shape.W;
    }
    void BindWorkspace(void* ptr) {
        activation = reinterpret_cast<float*>(ptr);
    }
    void BindDevice(void* ptr) {
        y = reinterpret_cast<float*>(ptr);
    }
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

    size_t GetWorkspaceSize() {
        auto& shape = prevs[0]->outputShape;
        return shape.C * shape.H * shape.W;
    }

    TensorShape InferOutputShape(TensorShape shape) {
        assert(false);
        return TensorShape(1, 1, 1, 1);
    }

    float* y = nullptr;
    int batchSize = 0;
    //softmax of input
    float* activation = nullptr;

};

class CuMseLayer :public CuLayer {
public:
    using CuLayer::CuLayer;
    CuMseLayer(int C, int H, int W);
    void forward();
    void backwardEx();

    TensorShape InferOutputShape(TensorShape shape);
    size_t GetWorkspaceSize();
    void BindWorkspace(void* ptr);
    size_t GetDeviceSize();
    void BindDevice(void* ptr);
    size_t GetActivationSize();
    float* GetActivation();
    float* GetDelta();
    size_t GetDeltaSize();

    Tensor label;
    float* y = nullptr;
    float* p = nullptr;
    float* grad = nullptr;
};

/********************convolution layer**************************/
class CuConvolutionLayer :public CuLayer {
public:
    using CuLayer::CuLayer;
    CuConvolutionLayer(int K, int C, int R, int S);

    TensorShape InferOutputShape(TensorShape shape) override;
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

    void wgrad();

    void bgrad();

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

    int padH = 0;
    int padW = 0;
    int strideH = 1;
    int strideW = 1;
};






