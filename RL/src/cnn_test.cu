#include "cnn_test.h"
#include "CuNN.h"


void test_cnn() {

    CuNN network;
    network.SetLearningRate(0.1);

    // 第一层 2 -> 2
    Layer layer1(2, 2);
    layer1.weights(0, 0) = 0.1;
    layer1.weights(0, 1) = 0.2;
    layer1.weights(1, 0) = 0.3;
    layer1.weights(1, 1) = 0.4;
    layer1.b(0) = 0.5;
    layer1.b(1) = 0.6;
    network.AddLayer(layer1);

    // 第二层 2 -> 1
    Layer layer2(2, 1);
    layer2.weights(0, 0) = 0.7;
    layer2.weights(0, 1) = 0.8;
    layer2.b(0) = 0.9;
    network.AddLayer(layer2);

    network.Print();

    // --- 样本输入，与 PyTorch 一致 ---
    Tensor xs(1, 2);
    xs(0, 0) = 1.0;
    xs(0, 1) = 2.0;
    Tensor ys(1, 1);
    ys(0, 0) = 1.0;

    network.AllocDeviceMemory();

    network.Backward(xs, ys);
    network.Step();

    network.FetchResultToCpu();

    network.Print();
}

