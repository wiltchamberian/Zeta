#pragma once
#include "tensor.h"


struct TensorShape {
    TensorShape() {}
    TensorShape(int n, int c, int h, int w) :N(n), C(c), H(h), W(w) {}
    int N = 1, C = 1, H = 1, W = 1;

    bool operator == (const TensorShape& ts) const {
        return (N == ts.N) && (C == ts.C) && (H == ts.H) && (W == ts.W);
    }
    bool operator != (const TensorShape& ts) const {
        return !((N == ts.N) && (C == ts.C) && (H == ts.H) && (W == ts.W));
    }
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
            return TensorShape(tensor.shape[0], tensor.shape[1], tensor.shape[2], 1);
        }
        else if (tensor.rank() == 2) {
            return TensorShape(tensor.shape[0], tensor.shape[1], 1, 1);
        }
        else if (tensor.rank() == 1) {
            return TensorShape(1, tensor.shape[0], 1, 1);
        }
        assert(false);
        return TensorShape();
    }
};

class DnnTensorDescriptor;
class BlasDescriptor;

//save the tensor data between layers
class CuTensor {
public:
    virtual ~CuTensor() {}
    TensorShape shape;

    void InitShape(const Tensor& tensor);
    void InitShape(const TensorShape& shape);
    float* v = nullptr;
    float* delta = nullptr; //dC/dv
    //MUST be called after InitShape
    virtual void Create() {}
    CuTensor* Clone() const {
        //TODO
        return nullptr;
    }

    DnnTensorDescriptor* desc = nullptr;
    BlasDescriptor* blasDesc = nullptr;
};