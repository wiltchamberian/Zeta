#pragma once

class Activation {
public:
  virtual double activate(const double x) = 0;
  virtual double dActivate(const double x) = 0;
};

class LinearActivation :public Activation {
public:
  double activate(const double x);
  double dActivate(const double x);
};

class RELU: public Activation
{
public:
  double activate(const double x);
  double dActivate(const double x);
};

class LeakyRELU : public Activation {
public:
  LeakyRELU(double slop = 0.01) :a(slop) {

  }
  double activate(const double x);
  double dActivate(const double x);
  void SetA(double x) {
    a = x;
  }
  double GetA(double x) {
    return a;
  }
protected:
  double a;
};
