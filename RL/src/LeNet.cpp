#include "LeNet.h"


void LeNet::createNetwork(){
    //input: N * C* H* W
    c1 = CreateLayer<Conv2d>(10, 1,5, 5);
    c1->padH = 0;
    c1->padW = 0;
    relu1 = CreateLayer< CuReluLayer>();
    maxpool = CreateLayer<MaxPool2d>();
    c2 = CreateLayer<Conv2d>(20,10,3,3);
    relu2 = CreateLayer<CuReluLayer>();
    fc = CreateLayer<Linear>(20 * 10 * 10, 200);
    fc2 = CreateLayer<Linear>(200, 10);
    auto softmax = CreateLayer<CuSoftmaxCrossEntropyLayer>();
    c1->AddLayer(relu1)->AddLayer(maxpool)->AddLayer(c2)
        ->AddLayer(relu2)->AddLayer(fc)->AddLayer(fc2)->AddLayer(softmax);

    head = softmax;

    AllocDeviceMemory();
}

void LeNet::Backward() {
    head->backwardEx();
    fc2->backwardEx();

    fc->backwardEx();
    relu2->backwardEx();

    c2->backwardEx();

    maxpool->backwardEx();

    relu1->backwardEx();
    c1->backwardEx();


}

void test_lenet() {


}

