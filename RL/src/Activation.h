#pragma once

class Activation {
public:
  virtual float activate(const float x) = 0;
  virtual float dActivate(const float x) = 0;
};

class LinearActivation :public Activation {
public:
    float activate(const float x);
    float dActivate(const float x);
};

class RELU: public Activation
{
public:
    float activate(const float x);
    float dActivate(const float x);
};

class LeakyRELU : public Activation {
public:
  LeakyRELU(float slop = 0.01) :a(slop) {

  }
  float activate(const float x);
  float dActivate(const float x);
  void SetA(float x) {
    a = x;
  }
  float GetA(float x) {
    return a;
  }
protected:
    float a;
};
