#pragma once

#include "cuLayer.h"
#include <cudnn.h>
#include <cudnn_backend.h>
#include <cublasLt.h>

class DNN;

class DnnLinear : public CuLinearLeakyReluLayer {
public:
    DnnLinear() {}
    DnnLinear(int dimIn, int dimOut);
    ~DnnLinear();
    void forward() override;
    void backwardEx() override;
    void dgrad();
    void wgrad();
    void bgrad();
    void BindWorkspace(void* ptr) override;
    void SetNN(CuNN* nn) override;
    CuLayer* Clone() const override;
    void init();
    void selectAlgo(cublasLtMatmulDesc_t opDesc,
        cublasLtMatrixLayout_t A,
        cublasLtMatrixLayout_t B,
        cublasLtMatrixLayout_t C,
        cublasLtMatmulAlgo_t* algo, void** workspace, size_t* workspaceSize);
    cublasLtMatmulDesc_t createAndSetDesc(cublasOperation_t transA, cublasOperation_t transB);
    cublasLtOrder_t order = CUBLASLT_ORDER_ROW;
    cublasOperation_t transX = CUBLAS_OP_N;
    cublasOperation_t transW = CUBLAS_OP_T;
    cublasLtMatmulDesc_t opDesc = nullptr; //forward
    cublasLtMatmulDesc_t opDesc_dw = nullptr;
    cublasLtMatmulDesc_t opDesc_delta = nullptr;
    cublasLtMatmulDesc_t opDesc_db = nullptr;
    cublasLtMatrixLayout_t layoutW = nullptr;

    cublasLtMatmulAlgo_t algo_fwd;
    cublasLtMatmulAlgo_t algo_dx;
    cublasLtMatmulAlgo_t algo_dw;

    size_t workspaceSize_fwd = 0;
    size_t workspaceSize_dx = 0;
    size_t workspaceSize_dw = 0;

    void* workspace_fwd = nullptr;
    void* workspace_dx = nullptr;
    void* workspace_dw = nullptr;

    DNN* dnn = nullptr;

};