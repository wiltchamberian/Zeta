#include "cnn_test.h"
#include "CuNN.h"
#include "tanhLayer.h"
#include "reluLayer.h"
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

    

    CuMseLayer* mse = network.CreateLayer<CuMseLayer>();
    mse->label = Tensor(1);
    mse->label(0) = 1.0;
    layer2->AddLayer(mse);

    network.Print();

    Tensor xs(1, 2);
    xs(0, 0) = -100.0;
    xs(0, 1) = 2.0;

    network.AllocDeviceMemory();

    
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

    CuNN network;
    network.SetLearningRate(0.1);
    int batchSize = 1;

    //layer
    auto c1 = network.CreateLayer<CuConvolutionLayer>(8, 2, 3, 3);
    c1->alpha = 0.0;
    for (int i = 0; i < 8; ++i) {
        for (int j = 0; j < 2; ++j) {
            for (int k = 0; k < 3; ++k) {
                for (int t = 0; t < 3; ++t) {
                    c1->weights(i, j, k, t) = 0.1;
                }
            }
        }
    }

    auto c2 = network.CreateLayer<CuConvolutionLayer>(4, 8, 3, 3);
    c2->alpha = 0.0;
    for (int i = 0; i < 4; ++i) {
        for (int j = 0; j < 8; ++j) {
            for (int k = 0; k < 3; ++k) {
                for (int t = 0; t < 3; ++t) {
                    c2->weights(i, j, k, t) = 0.1;
                }
            }
        }
    }
    c1->AddLayer(c2);

    //1d conv
    auto c3 = network.CreateLayer<CuConvolutionLayer>(1, 4, 3, 3);
    c3->alpha = 0.0;
    for (int i = 0; i < 1; ++i) {
        for (int j = 0; j < 4; ++j) {
            for (int k = 0; k < 3; ++k) {
                for (int t = 0; t < 3; ++t) {
                    c3->weights(i, j, k, t) = 0.1;
                }
            }
        }
    }
    c2->AddLayer(c3);

    CuSoftmaxCrossEntropyLayer* mse = network.CreateLayer<CuSoftmaxCrossEntropyLayer>();
    mse->label = Tensor(1, 3, 3);
    for (int i = 0; i < 9; ++i) {
        mse->label(0, i/3, i%3) = 1.0 / 9.0;
    }
    c3->AddLayer(mse);

    int H = 3;
    int W = 3;
    Tensor convX(1, 2, H, W);

    for (int t = 0; t < W; ++t) {
        convX(0, 0, 0, t) = 1;
        convX(0, 0, 1, t) = 0;
        convX(0, 0, 2, t) = 0;
        convX(0, 1, 0, t) = 0;
        convX(0, 1, 1, t) = 0;
        convX(0, 1, 2, t) = 1;
    }

    network.AllocDeviceMemory();
    network.Forward(convX);

    c3->FetchActivationToCpu();
    c3->ac.print("ac:");
    mse->FetchActivationToCpu();
    mse->distribution.print("distribution:");

    network.Backward();

    network.FetchGrad();
    network.PrintGrad();

    network.Step();
    network.FetchResultToCpu();
    network.Print();
}

void test_cnn_tictac() {
    
    /********************convolution********************/
    //test for tic-tac 

    CuNN network;
    network.SetLearningRate(0.1);
    network.Clear();
    std::cout << "start test convolution\n";

    int batchSize = 2;
    //input N * 2 * 3 * 3
   

    //layer
    auto c1 = network.CreateLayer<CuConvolutionLayer>(8, 2, 3, 3);
    c1->alpha = 0.0;
    for (int i = 0; i < 8; ++i) {
        for (int j = 0; j < 2; ++j) {
            for (int k = 0; k < 3; ++k) {
                for (int t = 0; t < 3; ++t) {
                    c1->weights(i, j, k, t) = 0.1;
                }
            }
        }
    }

    auto c2 = network.CreateLayer<CuConvolutionLayer>(4, 8, 3, 3);
    c2->alpha = 0.0;
    for (int i = 0; i < 4; ++i) {
        for (int j = 0; j < 8; ++j) {
            for (int k = 0; k < 3; ++k) {
                for (int t = 0; t < 3; ++t) {
                    c2->weights(i, j, k, t) = 0.1;
                }
            }
        }
    }
    c1->AddLayer(c2);

    //1d conv
    auto c3 = network.CreateLayer<CuConvolutionLayer>(1, 4, 3, 3);
    c3->alpha = 0.0;
    for (int i = 0; i < 1; ++i) {
        for (int j = 0; j < 4; ++j) {
            for (int k = 0; k < 3; ++k) {
                for (int t = 0; t < 3; ++t) {
                    c3->weights(i, j, k, t) = 0.1;
                }
            }
        }
    }
    c2->AddLayer(c3);

    auto fully1 = network.CreateLayer<CuLinearLeakyReluLayer>(9, 9);
    for (int i = 0; i < 9; ++i) {
        for (int j = 0; j < 9; ++j) {
            fully1->weights(i, j) = 0.1;
        }
    }
    fully1->alpha = 1;

    auto relu = network.CreateLayer<CuReluLayer>();
    fully1->AddLayer(relu);

    auto cross = network.CreateLayer<CuSoftmaxCrossEntropyLayer>();
    cross->label = Tensor(batchSize, 9);
    for (int j = 0; j < batchSize; ++j) {
        for (int i = 0; i < 9; ++i) {
            cross->label(j, i) = 0.0; //1.0 / 9.0;
        }
    }
    cross->label(0, 0) = 1.0f;
    
    relu->AddLayer(cross);

    auto fully2 = network.CreateLayer<CuLinearLeakyReluLayer>(9, 1);
    fully2->alpha = 1;
    for (int i = 0; i < 9; ++i) {
        for (int j = 0; j < 1; ++j) {
            fully2->weights(j, i) = 0.1;
        }
    }

    auto tanh = network.CreateLayer<CuTanhLayer>();
    fully2->AddLayer(tanh);

    auto mse = network.CreateLayer<CuMseLayer>();
    mse->label = Tensor(batchSize, 1);
    for (int i = 0; i < batchSize; ++i) {
        mse->label(i, 0) = 0;
    }
    

    tanh->AddLayer(mse);

    c3->AddLayer(fully1);
    c3->AddLayer(fully2);

    auto tail = network.CreateLayer<CuAddLayer>();
    cross->AddLayer(tail);
    mse->AddLayer(tail);

    //input
    int H = 3;
    int W = 3;
    Tensor convX(batchSize, 2, H, W);
    TensorShape shape(batchSize, 2, H, W);
    for (int i = 0; i < batchSize; ++i) {
        for (int t = 0; t < W; ++t) {
            convX(i, 0, 0, t) = 1;
            convX(i, 0, 1, t) = 0;
            convX(i, 0, 2, t) = 0;
            convX(i, 1, 0, t) = 0;
            convX(i, 1, 1, t) = 0;
            convX(i, 1, 2, t) = 1;
        }
    }
    
  
    network.AllocDeviceMemory();

    //output
    network.Forward(convX);
    mse->FetchPredYToCpu();
    mse->predY.print("predY");
    fully1->FetchActivationToCpu();
    fully1->ac.print("distri");

    

    network.Backward();
    c3->PrintDelta();
    fully1->PrintDelta();
    fully2->PrintDelta();

    network.FetchGrad();
    network.PrintGrad();

    network.Step();

    network.FetchResultToCpu();

    network.Print();
}

void tiktac() {


}

