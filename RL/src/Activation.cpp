#include "Activation.h"

double LinearActivation::activate(double x) {
  return x;
}

double LinearActivation::dActivate(double x) {
  return 1;
}

double RELU::activate(double x) {
  return x > 0 ? x : 0.0;
}

double RELU::dActivate(double x) {
  return x > 0 ? 1.0 : 0.0;
}

double LeakyRELU::activate(double x) {
  return x > 0 ? x : (a * x);
}

double LeakyRELU::dActivate(double x) {
  return x > 0 ? 1.0 : a;
}
