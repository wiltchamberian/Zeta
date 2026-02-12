#pragma once
#include "Layer.h"
#include "Activation.h"
#include <memory>

using Sample = std::vector<double>;

/*
Formula of BP
delta = (dC/da) h \sigma'


*/

class NeuralNetwork
{
public:
  NeuralNetwork(double lr = 1.0)
    :learningRate(lr){
    activation = std::make_shared<RELU>();
  }

  void SetLearningRate(double lr) {
    learningRate = lr;
  }

  void AddLayer(Layer& layer) {
    layers.push_back(layer);
  }

  void SetActivation(std::shared_ptr<Activation> active) {
    activation = active;
  }

  std::vector<double> Forward(std::vector<double>& x);

  void Backward(std::vector<Sample>& x, std::vector<Sample>& y);

  void Step();

  double MseLoss(const std::vector<Sample>& xs, const std::vector<Sample>& ys);

  void Train(std::vector<Sample>& xs, std::vector<Sample>& ys, int maxEpochs, double tolerance);

  void Print();

  void PrintGrad();

protected:

  std::vector<double> dC_da(std::vector<double>& a, std::vector<double>& y);
  std::vector<double> dsigma_dz(std::vector<double>& z);
  
  Tensor dC_dw(Sample& a, const Tensor& delta);
  //d^L= dC/da^{L} * a'(z^L)
  Tensor BP1(std::vector<double>& input); 
  //d^l=((w^{l+1}^T d^{l+1}) * a'(z^l)   //* is hadamard product
  Tensor BP2(const Tensor& w, const Tensor& delta, std::vector<double>& z);
  //dC/dw^l_{j,k} = d^l_j a^{l-1}_k
  Tensor BP4(std::vector<double>& a, const Tensor& delta);



protected:
  std::vector<Layer> layers;
  std::vector<Layer> gradLayers;
  std::shared_ptr<Activation> activation;
  double learningRate = 1.0;

  //temp
  std::vector<std::vector<double>> a;
  std::vector<std::vector<double>> z;
};

