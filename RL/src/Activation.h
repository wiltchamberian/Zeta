#pragma once

class Activation {
public:
  virtual double activate(double x) = 0;
  virtual double dActivate(double x) = 0;
};

class LinearActivation :public Activation {
public:
  double activate(double x);
  double dActivate(double x);
};

class RELU: public Activation
{
public:
  double activate(double x);
  double dActivate(double x);
};

class LeakyRELU : public Activation {
public:
  LeakyRELU(double slop = 0.01) :a(slop) {

  }
  double activate(double x);
  double dActivate(double x);
  void SetA(double x) {
    a = x;
  }
  double GetA(double x) {
    return a;
  }
protected:
  double a;
};
