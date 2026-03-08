#include "LeNet.h"
#include "DnnConv.h"
#include "DNN.h"
#include "dnnAct.h"
#include "DnnPooling.h"
#include "DnnLinear.h"
#include "DnnSoftmax.h"
#include <cublasLt.h>

void LeNet::createNetwork(){
    //input: N * C* H* W
    c1 = CreateLayer<Conv2d>(10, 1,5, 5);
    relu1 = CreateLayer< CuReluLayer>();
    maxpool = CreateLayer<MaxPool2d>();
    c2 = CreateLayer<Conv2d>(20,10,3,3);
    relu2 = CreateLayer<CuReluLayer>();
    fc = CreateLayer<Linear>(20 * 10 * 10, 200);
    relu3 = CreateLayer<CuReluLayer>();
    fc2 = CreateLayer<Linear>(200, 10);
    auto softmax = CreateLayer<CuSoftmaxCrossEntropyLayer>();
    c1->AddLayer(relu1)->AddLayer(maxpool)->AddLayer(c2)
        ->AddLayer(relu2)->AddLayer(fc)->AddLayer(relu3)->AddLayer(fc2)->AddLayer(softmax);

    head = softmax;

    AllocDeviceMemory();
}

void LeNet::createDnnNetwork() {
    //input: N * C* H* W
    c1 = CreateDnnLayer<DnnConv>(10, 1, 5, 5);
    //c1->weights.constants(0.1f);

    auto relu1 = CreateDnnLayer<DnnAct>(LayerType::Act_Relu);
    maxpool = CreateDnnLayer<DnnPooling>(2, 2);
    c2 = CreateDnnLayer<DnnConv>(20, 10, 3, 3);
    //c2->weights.constants(0.1f);

    auto relu2 = CreateDnnLayer<DnnAct>(LayerType::Act_Relu);
    fc = CreateDnnLayer<DnnLinear>(20 * 10 * 10, 200);
    auto relu3 = CreateDnnLayer<DnnAct>(LayerType::Act_Relu);
    fc2 = CreateDnnLayer<DnnLinear>(200, 10);
    //fc->weights.constants(0.1f);
    //fc2->weights.constants(0.1f);

    auto softmax = CreateDnnLayer<DnnSoftmax>();
    auto output = CreateLayer<OutputLayer>();
    c1->AddLayer(relu1)->AddLayer(maxpool)->AddLayer(c2)
        ->AddLayer(relu2)->AddLayer(fc)->AddLayer(relu3)->AddLayer(fc2)
        ->AddLayer(softmax)->AddLayer(output);

    head = softmax;

    AllocDeviceMemory();
}

void LeNet::Backward() {
    
    DNN::Backward();

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

int testBlasLt() {
    // ==========================================
    // Dimensions
    // ==========================================
    const int batch = 2;// 32;
    const int in_features = 2;// 128;
    const int out_features = 2;// 64;

    Tensor w(2, 2);
    Tensor x(2, 2);
    Tensor y(2, 2);
    x(0, 0) = 1; x(0, 1) = 2; x(1, 0) =3; x(1, 1) = 4;
    w(0, 0) = 1; w(0, 1) = 2; w(1, 0) = 3; w(1, 1) = 4;


    const int m = batch;
    const int n = out_features;
    const int k = in_features;

    // ==========================================
    // Allocate device memory
    // ==========================================
    float* d_X; //batch * in_features
    float* d_W; //out_features * in_featurs
    float* d_Y; //batch * out_features

    CHECK_CUDA(cudaMalloc((void**)(&d_X), m * k * sizeof(float)));
    CHECK_CUDA(cudaMalloc((void**)(&d_W), n * k * sizeof(float)));
    CHECK_CUDA(cudaMalloc((void**)(&d_Y), m * n * sizeof(float)));

    CHECK_CUDA(cudaMemcpy((void*)d_X, (void*)x.data(), m * k * sizeof(float),cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy((void*)d_W, (void*)w.data(), n * k * sizeof(float), cudaMemcpyHostToDevice));
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

    // A: X (m x k), row-major ¡ú ld = k
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

    // B: W (n x k), row-major ¡ú ld = k
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

    // C: Y (m x n), row-major ¡ú ld = n
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
    CHECK_CUDA(cudaMemcpy(y.data(), d_Y, m* n * sizeof(float), cudaMemcpyDeviceToHost));

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
