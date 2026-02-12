#include "NeuralNetwork.h"
#include <cmath>
#include <iostream>
#include <string>

std::vector<double> NeuralNetwork::dC_da(std::vector<double>& a, std::vector<double>& y) {
  std::vector<double> output(a.size(), 0);
  for (int i = 0; i < a.size(); ++i) {
    output[i] = a[i] - y[i];
  }
  return output;
}

std::vector<double> NeuralNetwork::dsigma_dz(std::vector<double>& z) {
  std::vector<double> b(z.size());
  for (int i = 0; i < z.size(); ++i) {
    b[i] = activation->dActivate(z[i]);
  }
  return b;
}

Tensor NeuralNetwork::dC_dw(Sample& a, const Tensor& delta) {
  Tensor w( delta.size(), a.size());
  for (int j = 0; j < delta.size(); ++j) {
    for (int k = 0; k < a.size(); ++k) {
      w(j,k) = a[k] * delta(j);
    }
  }
  return w;
}

Tensor NeuralNetwork::BP1(std::vector<double>& y) {
  Tensor output(y.size());

  auto d1 = dC_da(a.back(), y);
  auto d2 = dsigma_dz(z.back());
  for (int i = 0; i < d1.size(); ++i) {
    output(i) = d1[i] * d2[i];
  }
  return output;
}

Tensor NeuralNetwork::BP2(const Tensor& w, const Tensor& delta, std::vector<double>& z) {
  Tensor output(z.size());
  for (int i = 0; i < output.size(); ++i) {
    output(i) = 0;
    for (int j = 0; j < delta.size(); ++j) {
      output(i) += delta(j) * w(j, i);
    }
  }
  Sample s = dsigma_dz(z);
  for (int i = 0; i < s.size(); ++i) {
    output(i) = output(i) * s[i];
  }
  return output;
}

Tensor NeuralNetwork::BP4(std::vector<double>& a, const Tensor& delta) {
  return dC_dw(a, delta);
}

std::vector<double> NeuralNetwork::Forward(std::vector<double>& x) {
  a.resize(layers.size());
  for (int i = 0; i < layers.size(); ++i) {
    a[i].resize(layers[i].data().shape[0], 0);
  }
  z.resize(layers.size());
  for (int i = 0; i < layers.size(); ++i) {
    z[i].resize(layers[i].data().shape[0], 0);
  }
  
  for (int i = 0; i < layers.size(); ++i) {
    for (int j = 0; j < layers[i].data().shape[0];++j) {
      auto v = layers[i].data()[j];
      z[i][j] = 0.0;
      if (i == 0) {
        for (int k = 0; k < x.size(); ++k) {
          z[i][j] += v(k) * x[k];
        }
        z[i][j] += layers[i].b(j);
        a[i][j] = activation->activate(z[i][j]);
      }
      else {
        for (int k = 0; k < a[i - 1].size(); ++k) {
          z[i][j] += v(k) * a[i - 1][k];
        }
        z[i][j] += layers[i].b(j);
        a[i][j] = activation->activate(z[i][j]);
      }
    }
  }
  return a.back();
}

void NeuralNetwork::Backward(std::vector<Sample>& xs, std::vector<Sample>& ys) {

  std::vector<Layer> nLayers(layers.size());
  for (int i = 0; i < nLayers.size(); ++i) {
    nLayers[i].weights.zeros(layers[i].weights.shape[0], layers[i].weights.shape[1]);
    nLayers[i].b.zeros(layers[i].b.shape[0]);
  }

  gradLayers = nLayers;

  for (int i = 0; i < xs.size(); ++i) {
    Forward(xs[i]);

    std::vector<Layer> dlayers = nLayers;
    for (int l = layers.size() - 1; l >= 0; --l) {
      if (l == layers.size() - 1) {
          dlayers[l].b = BP1(ys[i]);
      }
      else {
          dlayers[l].b = BP2(layers[l + 1].data(), dlayers[l + 1].b, z[l]);
      }
      if (l >= 1) {
        dlayers[l].weights = BP4(a[l - 1], dlayers[l].b);
      }
      else {
        dlayers[l].weights = BP4(xs[i], dlayers[l].b);
      }
    }

    for (int l = 0; l < layers.size(); ++l) {
        gradLayers[l] = gradLayers[l] + dlayers[l];
    }
  }

  //update weights;
  for (int i = 0; i < gradLayers.size(); ++i) {
      gradLayers[i] /= xs.size();
  }

}

void NeuralNetwork::Step() {
    for (int i = 0; i < layers.size(); ++i) {
        layers[i].ApplyGradient(gradLayers[i], learningRate);
    }
}

double NeuralNetwork::MseLoss(const std::vector<Sample>& xs, const std::vector<Sample>& ys) {
    double totalLoss = 0.0;
    for (int i = 0; i < xs.size(); ++i) {
    const std::vector<double>& pred = xs[i];
        for (int j = 0; j < pred.size(); ++j) {
            double diff = pred[j] - ys[i][j];
            totalLoss += diff * diff;
        }
    }
    return totalLoss * 0.5 / xs.size();
}

void NeuralNetwork::Train(std::vector<Sample>& xs, std::vector<Sample>& ys,
  int maxEpochs, double tolerance) {
    for (int epoch = 0; epoch < maxEpochs; ++epoch) {

        double loss = MseLoss(xs, ys);
        std::cout << "Epoch " << epoch << " Loss: " << loss << std::endl;

        Backward(xs, ys);
    

        if (loss < tolerance) {
            std::cout << "Converged at epoch " << epoch << std::endl;
            break;
        }
    }
}

void NeuralNetwork::PrintGrad() {
    std::cout << "-----------PrintGrad---------------\n";
    for (int i = 0; i < gradLayers.size(); ++i) {
        std::cout << "gradlayer:" << i << std::endl;
        auto& data = gradLayers[i].data();
        for (int j = 0; j < data.shape[0]; ++j) {
            for (int k = 0; k < data.shape[1]; ++k) {
                std::cout << "W_" << j << "," << k << "=" << data(j, k) << " ";
            }
            std::cout << std::endl;
        }
        for (int j = 0; j < gradLayers[i].b.shape[0]; ++j) {
            std::cout << "B_" << j << "=" << gradLayers[i].b(j) << " ";
        }
        std::cout << std::endl << std::endl;
    }
}

void NeuralNetwork::Print() {
  for (int i = 0; i < layers.size(); ++i) {
    std::cout << "layer:" << i << std::endl;
    auto& data = layers[i].data();
    for (int j = 0; j < data.shape[0]; ++j) {
      for (int k = 0; k < data.shape[1]; ++k) {
          std::cout << "W_" << j << "," << k << "=" << data(j,k)<< " ";
      }
      std::cout << std::endl;
    }
    for (int j = 0; j < layers[i].b.shape[0]; ++j) {
      std::cout << "B_" << j << "=" << layers[i].b(j) << " ";
    }
    std::cout << std::endl << std::endl;
  }
}
