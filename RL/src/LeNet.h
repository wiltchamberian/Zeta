#pragma once
#include "DNN.h"
#include "reluLayer.h"
#include "maxpool.h"

class LeNet: public DNN{
public:
    void createNetwork();
    void createDnnNetwork();
    void Backward();
    CuSoftmaxCrossEntropyLayer* head = nullptr;

    Conv2d* c1 = nullptr;
    CuReluLayer* relu1 = nullptr;
    MaxPool2d* maxpool = nullptr;
    Conv2d* c2 = nullptr;
    CuReluLayer* relu2 = nullptr;
    CuReluLayer* relu3 = nullptr;
    Linear* fc = nullptr;
    Linear* fc2 = nullptr;


};



