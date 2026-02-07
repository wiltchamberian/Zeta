#include <iostream>
#include "NeuralNetwork.h"

int main()
{
  NeuralNetwork network(0.1);
  network.SetActivation(std::make_shared<RELU>());
  Layer layer1(2, 2);
  layer1.weights[0][0] = 1;
  layer1.weights[0][1] = 1;
  layer1.weights[1][0] = 1;
  layer1.weights[1][1] = 1;
  layer1.b[0] = 0;
  layer1.b[1] = 0;
  Layer layer2(2, 1);
  layer2.weights[0][0] = 1;
  layer2.weights[0][1] = 1;
  layer2.b[0] = 0;
  network.AddLayer(layer1);
  network.AddLayer(layer2);

  Sample x = { 0.0, 1.0 };
  Sample y = { 1.0 };
  Sample x2 = { 3.0, 2.0 };
  Sample y2 = { 11.0 };

  std::vector<Sample> xs = { x,x2 };
  std::vector<Sample> ys = { y, y2 };
 
  network.Train(xs, ys, 100, 1e-4);
  network.Print();
}

