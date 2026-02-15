#pragma once
#include <vector>
#include "tensor.h"


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

  virtual void forward(dim3 grid, dim3 block, const float* input,      // batch x in_dim
      const float* weights,    // out_dim x in_dim
      const float* bias,       // out_dim
      float* output,           // batch x out_dim
      int batch, int in_dim, int out_dim) {
  };

  virtual void backward(dim3 grid, dim3 block, const float* delta_next, // batch x dim_delta_next ¦Ä^{l+1}
      const float* W_next,     // dim_delta_next x  dim_delta W^{l+1}
      const float* a,          // batch x dim_delta a^l
      float* delta,            // batch x dim_delta Êä³ö ¦Ä^l
      int batch,
      int dim_delta,
      int dim_delta_next) {
  };

  void ApplyGradient(Layer& other, float learningRate);


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





