#include "LeNet.h"
#include "DnnConv.h"
#include "DNN.h"
#include "dnnAct.h"
#include <cublasLt.h>

void LeNet::createNetwork(){
    //input: N * C* H* W
    c1 = CreateLayer<Conv2d>(10, 1,5, 5);
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

void LeNet::createDnnNetwork() {
    //input: N * C* H* W
    c1 = CreateDnnLayer<DnnConv>(10, 1, 5, 5);
    auto relu1 = CreateDnnLayer<DnnActLayer>(LayerType::Act_Relu);
    maxpool = CreateLayer<MaxPool2d>();
    c2 = CreateDnnLayer<DnnConv>(20, 10, 3, 3);
    auto relu2 = CreateDnnLayer<DnnActLayer>(LayerType::Act_Relu);
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

int test() {

    // ========================
    // Matrix dimensions
    // X: [batch, in_features]
    // W: [out_features, in_features]
    // Y: [batch, out_features]
    // ========================

    const int batch = 32;
    const int in_features = 128;
    const int out_features = 64;

    const int m = batch;
    const int n = out_features;
    const int k = in_features;

    float* d_X, * d_W, * d_Y;

    CHECK_CUDA(cudaMalloc((void**)(&d_X), m * k * sizeof(float)));
    CHECK_CUDA(cudaMalloc((void**)(&d_W), n * k * sizeof(float)));
    CHECK_CUDA(cudaMalloc((void**)(&d_Y), m * n * sizeof(float)));

    // ========================
    // Create cuBLASLt handle
    // ========================
    cublasLtHandle_t ltHandle;
    CHECK_CUBLAS(cublasLtCreate(&ltHandle));

    // ========================
    // Operation descriptor
    // ========================
    cublasLtMatmulDesc_t operationDesc;
    CHECK_CUBLAS(
        cublasLtMatmulDescCreate(
            &operationDesc,
            CUBLAS_COMPUTE_32F,
            CUDA_R_32F
        )
    );

    // We want Y = X * W^T
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

    // ========================
    // Matrix layout descriptors
    // ========================

    cublasLtMatrixLayout_t layoutA, layoutB, layoutC;

    // A: X (m x k)
    CHECK_CUBLAS(
        cublasLtMatrixLayoutCreate(
            &layoutA,
            CUDA_R_32F,
            m, k, k
        )
    );

    // B: W (n x k) but transposed in op
    CHECK_CUBLAS(
        cublasLtMatrixLayoutCreate(
            &layoutB,
            CUDA_R_32F,
            n, k, k
        )
    );

    // C: Y (m x n)
    CHECK_CUBLAS(
        cublasLtMatrixLayoutCreate(
            &layoutC,
            CUDA_R_32F,
            m, n, n
        )
    );

    float alpha = 1.0f;
    float beta = 0.0f;

    // ========================
    // Run matmul
    // ========================

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
            nullptr,
            nullptr,
            0,
            0
        )
    );

    CHECK_CUDA(cudaDeviceSynchronize());

    std::cout << "Matmul done!" << std::endl;

    // ========================
    // Cleanup
    // ========================

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
