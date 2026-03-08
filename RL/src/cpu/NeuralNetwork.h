#pragma once
#include "Layer.h"
#include "Activation.h"
#include <memory>

namespace zeta {

    using Sample = std::vector<float>;

    /*
    Formula of BP
    delta = (dC/da) h \sigma'


    */

    class NeuralNetwork
    {
    public:
        NeuralNetwork(float lr = 1.0)
            :learningRate(lr) {
            activation = std::make_shared<RELU>();
        }

        void SetLearningRate(float lr) {
            learningRate = lr;
        }

        void AddLayer(Layer& layer) {
            layers.push_back(layer);
        }

        void SetActivation(std::shared_ptr<Activation> active) {
            activation = active;
        }

        std::vector<float> Forward(std::vector<float>& x);

        void Backward(std::vector<Sample>& x, std::vector<Sample>& y);

        void Step();

        float MseLoss(const std::vector<Sample>& xs, const std::vector<Sample>& ys);

        void Train(std::vector<Sample>& xs, std::vector<Sample>& ys, int maxEpochs, float tolerance);

        void Print();

        void PrintGrad();

    protected:

        std::vector<float> dC_da(std::vector<float>& a, std::vector<float>& y);
        std::vector<float> dsigma_dz(std::vector<float>& z);

        Tensor dC_dw(Sample& a, const Tensor& delta);
        //d^L= dC/da^{L} * a'(z^L)
        Tensor BP1(std::vector<float>& input);
        //d^l=((w^{l+1}^T d^{l+1}) * a'(z^l)   //* is hadamard product
        Tensor BP2(const Tensor& w, const Tensor& delta, std::vector<float>& z);
        //dC/dw^l_{j,k} = d^l_j a^{l-1}_k
        Tensor BP4(std::vector<float>& a, const Tensor& delta);



    protected:
        std::vector<Layer> layers;
        std::vector<Layer> gradLayers;
        std::shared_ptr<Activation> activation;
        float learningRate = 1.0;

        //temp
        std::vector<std::vector<float>> a;
        std::vector<std::vector<float>> z;
    };

}

