#pragma once
#include <vector>

using Matrix = std::vector<std::vector<double>>;
using Vector = std::vector<double>;

class Layer
{
public:
  Layer() {

  }

  Layer(int input, int output) {
    weights.resize(output);
    for (int i = 0; i < output; ++i) {
      weights[i].resize(input, 0);
    }
    b.resize(output, 0);
  }

  void ApplyGradient(Layer& other, double learningRate);


  friend Layer operator + (const Layer& l1, const Layer& l2);

  Layer& operator /= (int n);

  std::vector<std::vector<double>>& data() {
    return weights;
  }

  Matrix weights;
  Vector b;

};

//friend Layer operator + (const Layer& l1, const Layer& l2);





