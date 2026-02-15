#include "Layer.h"
#include <algorithm>

void Layer::ApplyGradient(Layer& other, float learningRate) {
  for (int i = 0; i < weights.shape[0]; ++i) {
    for (int j = 0; j < weights.shape[1]; ++j) {
      weights(i, j) = weights(i, j) - other.weights(i, j) * learningRate;
    }
    b(i) = b(i) - other.b(i) * learningRate;
  }
}


Layer operator + (const Layer& l1, const Layer& l2) {
  Layer layer(l1.weights.shape[1], l1.weights.shape[0]);
  for (int i = 0; i < l1.weights.shape[0]; ++i) {
    for (int j = 0; j < l1.weights.shape[1]; ++j) {
        layer.weights(i, j) = l1.weights(i, j) + l2.weights(i, j);
    }
    layer.b(i) = l1.b(i) + l2.b(i);
  }
  return layer;
}

Layer& Layer::operator /= (int n) {
  for (int i = 0; i < weights.shape[0]; ++i) {
    for (int j = 0; j < weights.shape[1]; ++j) {
      weights(i, j) = weights(i, j) / n;
    }
    b(i) = b(i) / n;
  }
  return *this;
}

