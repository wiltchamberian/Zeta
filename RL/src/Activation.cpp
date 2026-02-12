#include "Activation.h"

double LinearActivation::activate(const double x) {
  return x;
}

double LinearActivation::dActivate(const double x) {
  return 1;
}

double RELU::activate(const double x) {
  return x > 0 ? x : 0.0;
}

double RELU::dActivate(const double x) {
  return x > 0 ? 1.0 : 0.0;
}

double LeakyRELU::activate(const double x) {
  return x > 0 ? x : (a * x);
}

double LeakyRELU::dActivate(const double x) {
  return x > 0 ? 1.0 : a;
}
