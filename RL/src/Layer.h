#pragma once
#include <vector>
#include "tensor.h"

//using Matrix = std::vector<std::vector<double>>;
using Vector = std::vector<double>;
using Tensor = TensorT<double>;

class Matrix {
public:
    // 默认构造
    Matrix()
        : width(0), height(0) {
    }

    // 构造指定大小
    Matrix(int width, int height)
        : width(width), height(height)
    {
        data_.resize(width * height, 0.0);
    }

    // 拷贝构造
    Matrix(const Matrix& m)
        : width(m.width), height(m.height), data_(m.data_) {
    }

    // 拷贝赋值
    Matrix& operator= (const Matrix& m) {
        if (this != &m) {
            width = m.width;
            height = m.height;
            data_ = m.data_;
        }
        return *this;
    }

    // 移动构造
    Matrix(Matrix&& m) noexcept
        : width(m.width), height(m.height), data_(std::move(m.data_))
    {
        m.width = 0;
        m.height = 0;
    }

    // 移动赋值
    Matrix& operator= (Matrix&& m) noexcept {
        if (this != &m) {
            width = m.width;
            height = m.height;
            data_ = std::move(m.data_);

            m.width = 0;
            m.height = 0;
        }
        return *this;
    }

    // 访问
    void Set(int i, int j, double v) {
        data_[i * width + j] = v;
    }

    double* operator[] (int k) {
        return data_.data() + k * width;
    }

    void resize(int width, int height, double val = 0.0) {
        data_.resize(width * height, val);
        this->width = width;
        this->height = height;
    }

    int size() const {
        return height;
    }

    const double* data() const {
        return data_.data();
    }

    int width;
    int height;
    std::vector<double> data_;
};

class Layer
{
public:
  Layer()
  :in_dim(0)
  ,out_dim(0){

  }

  Layer(int input, int output)
  :in_dim(input)
  ,out_dim(output)
  {
    //weights.height = output;
    //weights.width = input;
    //weights.data_.resize(input * output, 0);
    weights = Tensor(output, input); //reverse order to level up computation performance

    b = Tensor(output);
  }

  void ApplyGradient(Layer& other, double learningRate);


  friend Layer operator + (const Layer& l1, const Layer& l2);

  Layer& operator /= (int n);

  Tensor& data() {
      return weights;
  }

  Tensor weights;
  Tensor b;

  int in_dim;
  int out_dim;

};

//friend Layer operator + (const Layer& l1, const Layer& l2);





