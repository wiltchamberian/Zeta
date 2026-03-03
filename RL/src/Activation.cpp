#include "Activation.h"

float LinearActivation::activate(const float x) {
  return x;
}

float LinearActivation::dActivate(const float x) {
  return 1.0;
}

float RELU::activate(const float x) {
  return x > 0 ? x : 0.0f;
}

float RELU::dActivate(const float x) {
  return x > 0 ? 1.0 : 0.0f;
}

float LeakyRELU::activate(const float x) {
  return x > 0 ? x : (a * x);
}

float LeakyRELU::dActivate(const float x) {
  return x > 0 ? 1.0 : a;
}
