#include "cnn_test.h"
#include "CuNN.h"
#include "cuLayer.h"
#include "TicTac.h"
#include <memory>

void test_cnn_linear() {
    CuNN network;
    network.SetLearningRate(0.1);

    // 第一层 2 -> 2
    std::unique_ptr<CuLinearLeakyReluLayer> layer1 = std::make_unique<CuLinearLeakyReluLayer>(2, 2);
    layer1->weights(0, 0) = 0.1;
    layer1->weights(0, 1) = 0.2;
    layer1->weights(1, 0) = 0.3;
    layer1->weights(1, 1) = 0.4;
    layer1->b(0) = 0.5;
    layer1->b(1) = 0.6;
    network.AddLayer(std::move(layer1));

    // 第二层 2 -> 1
    std::unique_ptr<CuLinearLeakyReluLayer> layer2 = std::make_unique<CuLinearLeakyReluLayer>(2, 1);
    layer2->weights(0, 0) = 0.7;
    layer2->weights(0, 1) = 0.8;
    layer2->b(0) = 0.9;
    network.AddLayer(std::move(layer2));

    network.Print();

    // --- 样本输入，与 PyTorch 一致 ---
    Tensor xs(1, 2);
    xs(0, 0) = -100.0;
    xs(0, 1) = 2.0;
    Tensor ys(1, 1);
    ys(0, 0) = 1.0;

    TensorShape sp;
    sp.N = 1;
    sp.C = 2;
    network.Build(sp);

    network.Backward(ys);
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
    auto c1 = std::make_shared<CuConvolutionLayer>(8, 2, 3, 3);
    c1->alpha = 0.1;
    network.SetHead(c1);

    auto c2 = std::make_shared<CuConvolutionLayer>(4, 8, 3, 3);
    c2->alpha = 0.1;
    c1->AddLayer(c2.get());

    //1d conv
    auto c3 = std::make_shared<CuConvolutionLayer>(1, 4, 3, 3);
    c3->alpha = 0.1;
    c2->AddLayer(c3.get());

    auto fully1 = std::make_shared<CuLinearLeakyReluLayer>(9, 9);
    auto cross = std::make_shared<CuSoftmaxCrossEntropyLayer>(batchSize);
    fully1->AddLayer(cross.get());

    auto fully2 = std::make_shared<CuLinearLeakyReluLayer>(9, 1);
    auto c4 = std::make_shared<CuMseLayer>(1,1,1);
    fully2->AddLayer(c4.get());

    c3->AddLayer(fully1.get());
    c3->AddLayer(fully2.get());

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
    Tensor label(1, 2, outShape.H, outShape.W);
    Tensor predY = network.ForwardAndFetchPredY(convX);
    std::cout << "predY:\n";
    predY.print("y");
    network.Backward(label);

    network.FetchGrad();
    network.PrintGrad();

    network.Step();

    network.FetchResultToCpu();

    network.Print();
}

void tiktac() {


}

