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
    
};

class CuLayer {
public:
    CuLayer()
        :in_dim(0)
        , out_dim(0) {

    }

    CuLayer(int input, int output)
        :in_dim(input)
        , out_dim(output)
    {
        weights = Tensor(output, input); //reverse order to level up computation performance

        b = Tensor(output);
    }

    virtual void forward(const float* input) = 0;
    virtual void backward(const float* delta_next, const float* w_next) = 0;
    virtual void wgrad(const float*) = 0;
    virtual void bgrad() = 0;

    //virtual Shape GetInputShape() = 0;
    //virtual Shape GetOutputShape() = 0;
    virtual TensorShape InferOutputShape(TensorShape shape) = 0;
    virtual size_t GetWorkspaceSize() = 0;
    virtual void BindWorkspace(void* ptr) = 0;

    Tensor& data() {
        return weights;
    }

    TensorShape inputShape;
    TensorShape outputShape;

    Tensor weights;
    Tensor b;

    Tensor weights_grad;
    Tensor bias_grad;

    int in_dim;
    int out_dim;

    DeviceLayer dl;
    
    CuLayer* next = nullptr;
    CuLayer* prev = nullptr;

    float alpha = 1.0;
};

class CuLinearLeakyReluLayer :public CuLayer {
public:
    using CuLayer::CuLayer;

    void forward(const float* input) override;

    void backward(const float* delta_next, const float* w_next);
    void wgrad(const float*);
    void bgrad();

    TensorShape InferOutputShape(TensorShape shape) override;
    size_t GetWorkspaceSize();
    void BindWorkspace(void* ptr);


};

/********************convolution layer**************************/
class CuConvolutionLayer :public CuLayer {
public:
    using CuLayer::CuLayer;
    CuConvolutionLayer(int K, int C, int R, int S);

    TensorShape InferOutputShape(TensorShape shape) override;
    size_t GetWorkspaceSize();
    void BindWorkspace(void* ptr);

    void forward(const float* input);

    void backward(const float* delta_next, const float* w_next);

    void wgrad(const float*);

    void bgrad();

    int padH = 0;
    int padW = 0;
    int strideH = 1;
    int strideW = 1;
};






