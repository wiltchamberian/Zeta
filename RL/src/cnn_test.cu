#include "cnn_test.h"
#include "CuNN.h"
#include "cuLayer.h"
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
    CuNN network;
    network.SetLearningRate(0.1);
    network.Clear();
    std::cout << "start test convolution\n";

    std::unique_ptr<CuConvolutionLayer> c1 = std::make_unique<CuConvolutionLayer>(2, 2, 2, 2);
    // out 0
    c1->weights.setData({ 1,0,0,1, 0,1,1,0,1,1,1,1,1,-1,-1,1 });
    network.AddLayer(std::move(c1));

    Tensor convX(1, 2, 4, 4);
    int t = 0;
    std::vector<float> d;
    for (int i = 0; i < 32; ++i) {
        d.push_back(i);
    }
    convX.setData(d);
    TensorShape shape(1, 2, 4, 4);
    Tensor convY(1, 2, 3, 3);

    network.Build(shape);

    Tensor predY = network.ForwardAndFetchPredY(convX);
    std::cout << "predY:\n";
    predY.print("y");
    network.Backward(convY);

    network.FetchGrad();
    network.PrintGrad();

    network.Step();

    network.FetchResultToCpu();

    network.Print();
}

