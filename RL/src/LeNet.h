#pragma once
#include "CuNN.h"
#include "reluLayer.h"
#include "maxpool.h"

class LeNet: public CuNN{
public:
    void createNetwork();
    void Backward();
    CuSoftmaxCrossEntropyLayer* head = nullptr;

    Conv2d* c1 = nullptr;
    CuReluLayer* relu1 = nullptr;
    MaxPool2d* maxpool = nullptr;
    Conv2d* c2 = nullptr;
    CuReluLayer* relu2 = nullptr;
    Linear* fc = nullptr;
    Linear* fc2 = nullptr;


};

void test_lenet();


