#include "cnn_test.h"
#include "CuNN.h"
#include "cuLayer.h"
#include "TicTac.h"
#include <memory>

void test_cnn_linear() {
    CuNN network;
    network.SetLearningRate(0.1);

    // 第一层 2 -> 2
    CuLinearLeakyReluLayer* layer1 = network.CreateLayer<CuLinearLeakyReluLayer>(2, 2);
    layer1->weights(0, 0) = 0.1;
    layer1->weights(0, 1) = 0.2;
    layer1->weights(1, 0) = 0.3;
    layer1->weights(1, 1) = 0.4;
    layer1->b(0) = 0.5;
    layer1->b(1) = 0.6;
    layer1->alpha = 0;

    // 第二层 2 -> 1
    CuLinearLeakyReluLayer* layer2 = network.CreateLayer<CuLinearLeakyReluLayer>(2, 1);
    layer2->weights(0, 0) = 0.7;
    layer2->weights(0, 1) = 0.8;
    layer2->b(0) = 0.9;
    
    layer1->AddLayer(layer2);

    

    CuMseLayer* mse = network.CreateLayer<CuMseLayer>(1);
    mse->label(0) = 1.0;
    layer2->AddLayer(mse);

    network.Print();

    Tensor xs(1, 2);
    xs(0, 0) = -100.0;
    xs(0, 1) = 2.0;
    TensorShape sp;
    sp.N = 1;
    sp.C = 2;
    network.Build(sp);

    
    network.Forward(xs);

    mse->FetchPredYToCpu();
    mse->PrintPredY();


    network.Backward();
    network.FetchGrad();
    network.PrintGrad();


    network.Step();

    

    network.FetchResultToCpu();

    network.Print();

    
}

void test_cnn_conv() {
    
    /********************convolution********************/
    //test for tic-tac 

    CuNN network;
    network.SetLearningRate(0.1);
    network.Clear();
    std::cout << "start test convolution\n";

    int batchSize = 1;
    //input N * 2 * 3 * 3
   

    //layer
    auto c1 = network.CreateLayer<CuConvolutionLayer>(8, 2, 3, 3);
    c1->alpha = 0.1;
    network.SetHead(c1);

    auto c2 = network.CreateLayer<CuConvolutionLayer>(4, 8, 3, 3);
    c2->alpha = 0.1;
    c1->AddLayer(c2);

    //1d conv
    auto c3 = network.CreateLayer<CuConvolutionLayer>(1, 4, 3, 3);
    c3->alpha = 0.1;
    c2->AddLayer(c3);

    auto fully1 = network.CreateLayer<CuLinearLeakyReluLayer>(9, 9);
    auto cross = network.CreateLayer<CuSoftmaxCrossEntropyLayer>(batchSize);
    fully1->AddLayer(cross);

    auto fully2 = network.CreateLayer<CuLinearLeakyReluLayer>(9, 1);
    auto mse = network.CreateLayer<CuMseLayer>(1,1,1);
    fully2->AddLayer(mse);

    c3->AddLayer(fully1);
    c3->AddLayer(fully2);

    auto tail = network.CreateLayer<CuAddLayer>();
    cross->AddLayer(tail);


    //input
    int H = 4;
    int W = 4;
    Tensor convX(1, 2, H, W);
    TensorShape shape(1, 2, H, W);
    
    std::vector<float> d;
    for (int i = 0; i < 32; ++i) {
        d.push_back(i);
    }
    convX.setData(d);
    
    
    TensorShape outShape = network.Build(shape);

    //output
    //Tensor label(1, 2, outShape.H, outShape.W);
    Tensor predY = network.ForwardAndFetchPredY(convX);
    std::cout << "predY:\n";
    predY.print("y");
    network.Backward();

    network.FetchGrad();
    network.PrintGrad();

    network.Step();

    network.FetchResultToCpu();

    network.Print();
}

void tiktac() {


}

