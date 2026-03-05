#include "cnn_test.h"
#include "CuNN.h"
#include "DNN.h"
#include "tanhLayer.h"
#include "reluLayer.h"
#include "maxpool.h"
#include "DnnPooling.h"
#include "TicTac.h"
#include "DnnConv.h"
#include "DnnSoftmax.h"
#include "DnnAct.h"
#include "DnnLinear.h"
#include <cublasLt.h>
#include <memory>

void test_cnn_linear() {
    CuNN network;
    network.SetLearningRate(0.1);

    // 第一层 2 -> 2
    CuLinearLeakyReluLayer* layer1 = network.CreateLayer<CuLinearLeakyReluLayer>(2, 2);
    layer1->weights(0, 0) = 0.1f;
    layer1->weights(0, 1) = 0.2f;
    layer1->weights(1, 0) = 0.3f;
    layer1->weights(1, 1) = 0.4f;
    layer1->b(0) = 0.5f;
    layer1->b(1) = 0.6f;
    layer1->alpha = 0;

    // 第二层 2 -> 1
    CuLinearLeakyReluLayer* layer2 = network.CreateLayer<CuLinearLeakyReluLayer>(2, 1);
    layer2->weights(0, 0) = 0.7f;
    layer2->weights(0, 1) = 0.8f;
    layer2->b(0) = 0.9f;
    
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
    network.SetLearningRate(0.1f);

    //layer
    auto c1 = network.CreateLayer<Conv2d>(8, 2, 3, 3, Size2D{ 1,1 });
    for (int i = 0; i < 8; ++i) {
        for (int j = 0; j < 2; ++j) {
            for (int k = 0; k < 3; ++k) {
                for (int t = 0; t < 3; ++t) {
                    c1->weights(i, j, k, t) = 0.1f;
                }
            }
        }
    }
    auto relu1 = network.CreateLayer<CuReluLayer>();

    auto c2 = network.CreateLayer<Conv2d>(4, 8, 3, 3, Size2D{ 1,1 });
    auto relu2 = network.CreateLayer<CuReluLayer>();

    for (int i = 0; i < 4; ++i) {
        for (int j = 0; j < 8; ++j) {
            for (int k = 0; k < 3; ++k) {
                for (int t = 0; t < 3; ++t) {
                    c2->weights(i, j, k, t) = 0.1f;
                }
            }
        }
    }

    //1d conv
    auto c3 = network.CreateLayer<Conv2d>(1, 4, 3, 3, Size2D{ 1,1 });
    for (int i = 0; i < 1; ++i) {
        for (int j = 0; j < 4; ++j) {
            for (int k = 0; k < 3; ++k) {
                for (int t = 0; t < 3; ++t) {
                    c3->weights(i, j, k, t) = 0.1f;
                }
            }
        }
    }

    CuSoftmaxCrossEntropyLayer* cross = network.CreateLayer<CuSoftmaxCrossEntropyLayer>();
    cross->label = Tensor(1, 3, 3);
    for (int i = 0; i < 9; ++i) {
        cross->label(0, i/3, i%3) = 1.0f / 9.0f;
    }

    auto outputLayer = network.CreateLayer<OutputLayer>();
    c1->AddLayer(relu1)->AddLayer(c2)->AddLayer(relu2)->AddLayer(c3)->AddLayer(cross)->AddLayer(outputLayer);

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
    c3->ac.print_torch_style("ac:");
    cross->FetchActivationToCpu();
    cross->distribution.print_torch_style("distribution:");
    

    network.Backward();
    c3->PrintDelta();
    c3->FetchGradToCpu();
    c3->weights_grad.print_torch_style();
    c3->bias_grad.print_torch_style();
    auto act = relu2->FetchActivationToCpu();
    act.print_torch_style();

    network.FetchGrad();
    network.PrintGrad();
    

    network.Step();
    network.FetchResultToCpu();
    network.Print();
}

void test_dnn_conv() {

    DNN network;
    network.SetLearningRate(0.1f);

    //layer
    auto c1 = network.CreateDnnLayer<DnnConv>(8, 2, 3, 3, Size2D{ 1,1 });
    for (int i = 0; i < 8; ++i) {
        for (int j = 0; j < 2; ++j) {
            for (int k = 0; k < 3; ++k) {
                for (int t = 0; t < 3; ++t) {
                    c1->weights(i, j, k, t) = 0.1f;
                }
            }
        }
    }
    auto relu1 = network.CreateDnnLayer<DnnActLayer>(LayerType::Act_Relu);

    auto maxpool = network.CreateDnnLayer<DnnPooling>(2, 2);

    auto c2 = network.CreateDnnLayer<DnnConv>(4, 8, 3, 3, Size2D{ 1,1 });
    auto relu2 = network.CreateDnnLayer<DnnActLayer>(LayerType::Act_Relu);

    for (int i = 0; i < 4; ++i) {
        for (int j = 0; j < 8; ++j) {
            for (int k = 0; k < 3; ++k) {
                for (int t = 0; t < 3; ++t) {
                    c2->weights(i, j, k, t) = 0.1f;
                }
            }
        }
    }

    //1d conv
    auto c3 = network.CreateDnnLayer<DnnConv>(1, 4, 3, 3, Size2D{ 1,1 });
    for (int i = 0; i < 1; ++i) {
        for (int j = 0; j < 4; ++j) {
            for (int k = 0; k < 3; ++k) {
                for (int t = 0; t < 3; ++t) {
                    c3->weights(i, j, k, t) = 0.1f;
                }
            }
        }
    }

    //fully linear 
    DnnLinear* linear = network.CreateDnnLayer<DnnLinear>(9, 9);
    for (int i = 0; i < 9; ++i) {
        for (int j = 0; j < 9; ++j) {
            linear->weights(i, j) = 0.1f;
        }
    }

    DnnSoftmax* cross = network.CreateDnnLayer<DnnSoftmax>();
    cross->label = Tensor(1, 3, 3);
    for (int i = 0; i < 9; ++i) {
        cross->label(0, i / 3, i % 3) = 1.0f / 9.0f;
    }

    auto outputLayer = network.CreateLayer<OutputLayer>();

    c1->AddLayer(relu1)->AddLayer(maxpool)->AddLayer(c2)->AddLayer(relu2)
        ->AddLayer(c3)->AddLayer(cross)->AddLayer(outputLayer);

    int H = 6;
    int W = 6;
    Tensor convX(1, 2, H, W);

    for (int t = 0; t < W; ++t) {
        convX(0, 0, 0, t) = 1;
        convX(0, 0, 1, t) = 0;
        convX(0, 0, 2, t) = 0;
        convX(0, 0, 3, t) = 1;
        convX(0, 0, 4, t) = 0;
        convX(0, 0, 5, t) = 0;

        convX(0, 1, 0, t) = 0;
        convX(0, 1, 1, t) = 0;
        convX(0, 1, 2, t) = 1;
        convX(0, 1, 3, t) = 0;
        convX(0, 1, 4, t) = 0;
        convX(0, 1, 5, t) = 1;
    }

    network.AllocDeviceMemory();
    network.Forward(convX);

    c3->FetchActivationToCpu();
    c3->ac.print_torch_style("ac:");
    cross->FetchActivationToCpu();
    cross->distribution.print_torch_style("distribution:");


    network.Backward();
    c3->PrintDelta();
    c3->FetchGradToCpu();
    c3->weights_grad.print_torch_style();
    c3->bias_grad.print_torch_style();
    auto act = relu2->FetchActivationToCpu();
    act.print_torch_style();

    network.FetchGrad();
    network.PrintGrad();


    network.Step();
    network.FetchResultToCpu();
    network.Print();
}

void test_dnn_linear() {

    DNN network;
    network.SetLearningRate(0.1);

    // 第一层 2 -> 2
    DnnLinear* layer1 = network.CreateDnnLayer<DnnLinear>(2, 2);
    layer1->weights(0, 0) = 0.1f;
    layer1->weights(0, 1) = 0.2f;
    layer1->weights(1, 0) = 0.3f;
    layer1->weights(1, 1) = 0.4f;
    layer1->b(0) = 0.5f;
    layer1->b(1) = 0.6f;

    auto act = network.CreateDnnLayer<DnnActLayer>(LayerType::Act_Sigmoid);

    // 第二层 2 -> 1
    DnnLinear* layer2 = network.CreateDnnLayer<DnnLinear>(2, 1);
    layer2->weights(0, 0) = 0.7f;
    layer2->weights(0, 1) = 0.8f;
    layer2->b(0) = 0.9f;

    CuMseLayer* mse = network.CreateLayer<CuMseLayer>();
    mse->label = Tensor(1);
    mse->label(0) = 1.0;

    OutputLayer* output = network.CreateLayer<OutputLayer>();

    layer1->AddLayer(act)->AddLayer(layer2)->AddLayer(mse)->AddLayer(output);

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

void test_cnn_tictac() {
    
    /********************convolution********************/
    //test for tic-tac 

    CuNN network;
    network.SetLearningRate(0.1f);
    network.Clear();
    std::cout << "start test convolution\n";

    int batchSize = 4;
    //input N * 2 * 3 * 3
   

    //layer
    auto c1 = network.CreateLayer<Conv2d>(8, 2, 3, 3, Size2D{ 1,1 });
    for (int i = 0; i < 8; ++i) {
        for (int j = 0; j < 2; ++j) {
            for (int k = 0; k < 3; ++k) {
                for (int t = 0; t < 3; ++t) {
                    c1->weights(i, j, k, t) = 0.1f;
                }
            }
        }
    }
    auto relu_for_conv = network.CreateLayer<CuReluLayer>();
    c1->AddLayer(relu_for_conv);
    auto maxpool = network.CreateLayer<MaxPool2d>();
    maxpool->w = 2;
    maxpool->h = 2;

    relu_for_conv->AddLayer(maxpool);

    auto c2 = network.CreateLayer<Conv2d>(4, 8, 3, 3, Size2D{ 1,1 });
    c2->alpha = 0.0;
    for (int i = 0; i < 4; ++i) {
        for (int j = 0; j < 8; ++j) {
            for (int k = 0; k < 3; ++k) {
                for (int t = 0; t < 3; ++t) {
                    c2->weights(i, j, k, t) = 0.1f;
                }
            }
        }
    }
    maxpool->AddLayer(c2);

    //1d conv
    auto c3 = network.CreateLayer<Conv2d>(1, 4, 3, 3, Size2D{ 1,1 });
    c3->alpha = 0.0;
    for (int i = 0; i < 1; ++i) {
        for (int j = 0; j < 4; ++j) {
            for (int k = 0; k < 3; ++k) {
                for (int t = 0; t < 3; ++t) {
                    c3->weights(i, j, k, t) = 0.1f;
                }
            }
        }
    }
    c2->AddLayer(c3);

    auto fully1 = network.CreateLayer<CuLinearLeakyReluLayer>(9, 9);
    for (int i = 0; i < 9; ++i) {
        for (int j = 0; j < 9; ++j) {
            fully1->weights(i, j) = 0.1f;
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
            fully2->weights(j, i) = 0.1f;
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
    int H = 6;
    int W = 6;
    Tensor convX(batchSize, 2, H, W);
    TensorShape shape(batchSize, 2, H, W);
    for (int i = 0; i < batchSize; ++i) {
        for (int t = 0; t < W; ++t) {
            convX(i, 0, 0, t) = 1;
            convX(i, 0, 1, t) = 0;
            convX(i, 0, 2, t) = 0;
            convX(i, 0, 3, t) = 1;
            convX(i, 0, 4, t) = 0;
            convX(i, 0, 5, t) = 0;
            convX(i, 1, 0, t) = 0;
            convX(i, 1, 1, t) = 0;
            convX(i, 1, 2, t) = 1;
            convX(i, 1, 3, t) = 0;
            convX(i, 1, 4, t) = 0;
            convX(i, 1, 5, t) = 1;
        }
    }
    
  
    network.AllocDeviceMemory();

    int iterNum = 10;
    for (int i = 0; i < iterNum; ++i) {
        network.Forward(convX);

        mse->BindLabelToDevice();
        cross->BindLabelToDevice();

        network.Backward();

        float mseLoss = mse->FetchLoss();
        float crossLoss = cross->FetchLoss();
        float loss = mseLoss + crossLoss;
        std::cout << "loss:" << loss << std::endl;
        std::cout << "mseLoss:" << mseLoss << "\n";
        std::cout << "crossLoss:" << crossLoss << std::endl;
        std::cout << std::endl;

        network.Step();

        //network.FetchResultToCpu();
        //network.Print();
    }
   

    mse->FetchPredYToCpu();
    mse->predY.print("predY");
    fully1->FetchActivationToCpu();
    fully1->ac.print("distri");

    
    

    
    c3->PrintDelta();
    fully1->PrintDelta();
    fully2->PrintDelta();

    network.FetchGrad();
    network.PrintGrad();
    
}


#define CHECK_CUDA(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            std::cerr << "CUDA error: " << cudaGetErrorString(err) << std::endl; \
            exit(EXIT_FAILURE); \
        } \
    } while (0)

#define CHECK_CUBLAS(call) \
    do { \
        cublasStatus_t status = call; \
        if (status != CUBLAS_STATUS_SUCCESS) { \
            std::cerr << "cuBLASLt error\n"; \
            exit(EXIT_FAILURE); \
        } \
    } while (0)

int test_blaslt() {
    // ==========================================
    // Dimensions
    // ==========================================
    const int batch = 2;// 32;
    const int in_features = 2;// 128;
    const int out_features = 2;// 64;

    Tensor w(2, 2);
    Tensor x(2, 2);
    Tensor y(2, 2);
    x(0, 0) = 1; x(0, 1) = 2; x(1, 0) = 3; x(1, 1) = 4;
    w(0, 0) = 2; w(0, 1) = 1; w(1, 0) = 3; w(1, 1) = 0;
    Tensor b(2);
    b(0) = 1; b(1) = 2;

    const int m = batch;
    const int n = out_features;
    const int k = in_features;

    // ==========================================
    // Allocate device memory
    // ==========================================
    float* d_X; //batch * in_features
    float* d_W; //out_features * in_featurs
    float* d_Y; //batch * out_features
    float* d_b; //out_features

    CHECK_CUDA(cudaMalloc((void**)(&d_X), m * k * sizeof(float)));
    CHECK_CUDA(cudaMalloc((void**)(&d_W), n * k * sizeof(float)));
    CHECK_CUDA(cudaMalloc((void**)(&d_Y), m * n * sizeof(float)));
    //CHECK_CUDA(cudaMalloc((void**)(&d_b), n * sizeof(float)));

    CHECK_CUDA(cudaMemcpy((void*)d_X, (void*)x.data(), m * k * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy((void*)d_W, (void*)w.data(), n * k * sizeof(float), cudaMemcpyHostToDevice));
    //CHECK_CUDA(cudaMemcpy((void*)d_b, (void*)b.data(), n * sizeof(float), cudaMemcpyHostToDevice));
    // ==========================================
    // Create cuBLASLt handle
    // ==========================================
    cublasLtHandle_t ltHandle;
    CHECK_CUBLAS(cublasLtCreate(&ltHandle));

    // ==========================================
    // Matmul descriptor
    // ==========================================
    cublasLtMatmulDesc_t operationDesc;
    CHECK_CUBLAS(
        cublasLtMatmulDescCreate(
            &operationDesc,
            CUBLAS_COMPUTE_32F,
            CUDA_R_32F
        )
    );

    cublasLtEpilogue_t epi = CUBLASLT_EPILOGUE_BIAS;

    //CHECK_CUBLAS(cublasLtMatmulDescSetAttribute(
    //    operationDesc,
    //    CUBLASLT_MATMUL_DESC_EPILOGUE,
    //    &epi,
    //    sizeof(epi)
    //));

    //CHECK_CUBLAS(cublasLtMatmulDescSetAttribute(
    //    operationDesc,
    //    CUBLASLT_MATMUL_DESC_BIAS_POINTER,
    //    &d_b,
    //    sizeof(d_b)
    //));

    // We compute Y = X * W^T
    cublasOperation_t transA = CUBLAS_OP_N;
    cublasOperation_t transB = CUBLAS_OP_T;

    CHECK_CUBLAS(cublasLtMatmulDescSetAttribute(
        operationDesc,
        CUBLASLT_MATMUL_DESC_TRANSA,
        &transA,
        sizeof(transA)));

    CHECK_CUBLAS(cublasLtMatmulDescSetAttribute(
        operationDesc,
        CUBLASLT_MATMUL_DESC_TRANSB,
        &transB,
        sizeof(transB)));

    // ==========================================
    // Layouts (ROW-MAJOR)
    // ==========================================

    cublasLtMatrixLayout_t layoutA, layoutB, layoutC;

    cublasLtOrder_t order = CUBLASLT_ORDER_ROW;

    // A: X (m x k), row-major → ld = k
    CHECK_CUBLAS(
        cublasLtMatrixLayoutCreate(
            &layoutA,
            CUDA_R_32F,
            m, k, k
        )
    );

    CHECK_CUBLAS(
        cublasLtMatrixLayoutSetAttribute(
            layoutA,
            CUBLASLT_MATRIX_LAYOUT_ORDER,
            &order,
            sizeof(order)
        )
    );

    // B: W (n x k), row-major → ld = k
    CHECK_CUBLAS(
        cublasLtMatrixLayoutCreate(
            &layoutB,
            CUDA_R_32F,
            n, k, k
        )
    );

    CHECK_CUBLAS(
        cublasLtMatrixLayoutSetAttribute(
            layoutB,
            CUBLASLT_MATRIX_LAYOUT_ORDER,
            &order,
            sizeof(order)
        )
    );

    // C: Y (m x n), row-major → ld = n
    CHECK_CUBLAS(
        cublasLtMatrixLayoutCreate(
            &layoutC,
            CUDA_R_32F,
            m, n, n
        )
    );

    CHECK_CUBLAS(
        cublasLtMatrixLayoutSetAttribute(
            layoutC,
            CUBLASLT_MATRIX_LAYOUT_ORDER,
            &order,
            sizeof(order)
        )
    );

    // ==========================================
    // Execute matmul
    // ==========================================

    float alpha = 1.0f;
    float beta = 0.0f;

    CHECK_CUBLAS(
        cublasLtMatmul(
            ltHandle,
            operationDesc,
            &alpha,
            d_X, layoutA,
            d_W, layoutB,
            &beta,
            d_Y, layoutC,
            d_Y, layoutC,
            nullptr,        // algo
            nullptr,        // workspace
            0,              // workspace size
            0               // stream
        )
    );

    CHECK_CUDA(cudaDeviceSynchronize());
    CHECK_CUDA(cudaMemcpy(y.data(), d_Y, m * n * sizeof(float), cudaMemcpyDeviceToHost));

    y.print_torch_style();
    std::cout << "Row-major Matmul Done!" << std::endl;

    // ==========================================
    // Cleanup
    // ==========================================

    cublasLtMatrixLayoutDestroy(layoutA);
    cublasLtMatrixLayoutDestroy(layoutB);
    cublasLtMatrixLayoutDestroy(layoutC);
    cublasLtMatmulDescDestroy(operationDesc);
    cublasLtDestroy(ltHandle);

    cudaFree(d_X);
    cudaFree(d_W);
    cudaFree(d_Y);

    return 0;

}
