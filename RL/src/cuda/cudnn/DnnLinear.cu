#include "DnnLinear.h"
#include <cublasLt.h>
#include "DnnHelp.h"
#include "DnnTensor.h"
#include "cu_tool.h"
#include "DNN.h"
#include "kernels.h"


DnnLinear::DnnLinear(int dimIn, int dimOut)
    :CuLinearLeakyReluLayer(dimIn, dimOut){
    init();
}

void DnnLinear::init() {
    //forward opdesc
    opDesc = createAndSetDesc(transX, transW);

    //opDesc_delta
    opDesc_delta = createAndSetDesc(CUBLAS_OP_N, CUBLAS_OP_N);

    opDesc_dw = createAndSetDesc(CUBLAS_OP_T, CUBLAS_OP_N);

    //W 
    BLAS_CHECK(cublasLtMatrixLayoutCreate(
        &layoutW,
        CUDA_R_32F,
        out_dim, in_dim, in_dim
    ));

    BLAS_CHECK(cublasLtMatrixLayoutSetAttribute(
        layoutW,
        CUBLASLT_MATRIX_LAYOUT_ORDER,
        &order,
        sizeof(order)
    ));
}

DnnLinear::~DnnLinear() {
    if (layoutW) {
        BLAS_CHECK(cublasLtMatrixLayoutDestroy(layoutW));
    }
    if (opDesc) {
        BLAS_CHECK(cublasLtMatmulDescDestroy(opDesc));
    }
    if (opDesc_delta)
        BLAS_CHECK(cublasLtMatmulDescDestroy(opDesc_delta));

    if (opDesc_dw)
        BLAS_CHECK(cublasLtMatmulDescDestroy(opDesc_dw));

    if (workspace_fwd) {
        CU_CHECK(cudaFree(workspace_fwd));
    }
    if (workspace_dx) {
        CU_CHECK(cudaFree(workspace_dx));
    }
    if (workspace_dw) {
        CU_CHECK(cudaFree(workspace_dw));
    }
}

void DnnLinear::forward() {

    float alpha = 1.0f;
    float beta = 0.0f;

    BLAS_CHECK(cublasLtMatmul(
        dnn->ltHandle,
        opDesc,
        &alpha,
        input->v, input->blasDesc->layout,
        dl.weights, layoutW,
        &beta,
        output->v, output->blasDesc->layout,
        output->v, output->blasDesc->layout,
        &algo_fwd,        // algo
        workspace_fwd,        // workspace
        workspaceSize_fwd,              // workspace size
        0               // stream
    ));

    int C = output->shape.C;
    int HW = output->shape.H * output->shape.W;
    int NCHW = output->shape.NumElements();
    dim3 block(TILE_WIDTH);
    dim3 grid((NCHW + TILE_WIDTH - 1) / TILE_WIDTH);
    tensor_add_bias_kernel<<<grid, block >>> (output->v, dl.bias, HW, C, NCHW);
   
}

void DnnLinear::backwardEx() {
    add = false;

    float alpha = 1.0f;
    float beta = 0.0f;

    /*
        1️⃣ dX = dY * W

        [B,out] * [out,in] = [B,in]
    */
    if (!prevs.empty()) {
        cublasStatus_t status = cublasLtMatmul(
            dnn->ltHandle,
            opDesc_delta,
            &alpha,
            output->delta, output->blasDesc->layout,   // dY
            dl.weights, layoutW,                       // W
            &beta,
            input->delta, input->blasDesc->layout,     // dX
            input->delta, input->blasDesc->layout,
            &algo_dx,
            workspace_dx,
            workspaceSize_dx,
            0
        );
        if (status != CUBLAS_STATUS_SUCCESS) {
            assert(false);
        }
        prevs[0]->add = true;
    }
    


    /*
        2️⃣ dW = dY^T * X

        [out,B] * [B,in] = [out,in]
    */

    auto status = cublasLtMatmul(
        dnn->ltHandle,
        opDesc_dw,
        &alpha,
        output->delta, output->blasDesc->layout,   // dY
        input->v, input->blasDesc->layout,         // X
        &beta,
        dl.grad_w, layoutW,                        // dW
        dl.grad_w, layoutW,
        &algo_dw,
        workspace_dw,
        workspaceSize_dw,
        0
    );
    if (status != CUBLAS_STATUS_SUCCESS) {
        assert(false);
    }
    
    CuLinearLeakyReluLayer::bgrad();
    
    if (nn->c != 0) {
        regular_grad();
    }
}

void DnnLinear::BindWorkspace(void* ptr) {
    CuLinearLeakyReluLayer::BindWorkspace(ptr);
    output->Create();

    selectAlgo(opDesc,
        input->blasDesc->layout,
        layoutW,
        output->blasDesc->layout,
        &algo_fwd, &workspace_fwd, & workspaceSize_fwd);

    selectAlgo(opDesc_delta,
        output->blasDesc->layout,
        layoutW,
        input->blasDesc->layout,
        &algo_dx, &workspace_dx, &workspaceSize_dx);

    selectAlgo(opDesc_dw,
        output->blasDesc->layout,
        input->blasDesc->layout,
        layoutW,
        &algo_dw,&workspace_dw,&workspaceSize_dw);

}


void DnnLinear::selectAlgo(
    cublasLtMatmulDesc_t opDesc,
    cublasLtMatrixLayout_t A,
    cublasLtMatrixLayout_t B,
    cublasLtMatrixLayout_t C,
    cublasLtMatmulAlgo_t* algo, void** workspace, size_t* workspaceSize)
{
    const int requestAlgoCount = 20;

    cublasLtMatmulHeuristicResult_t heuristics[requestAlgoCount];
    int returnedResults = 0;

    cublasLtMatmulPreference_t preference;

    BLAS_CHECK(cublasLtMatmulPreferenceCreate(&preference));

    // 不设置 workspace 限制
    BLAS_CHECK(cublasLtMatmulAlgoGetHeuristic(
        dnn->ltHandle,
        opDesc,
        A,
        B,
        C,
        C,
        preference,
        requestAlgoCount,
        heuristics,
        &returnedResults));

    if (returnedResults == 0)
        throw std::runtime_error("No cublasLt algorithm found");

    auto heristic = heuristics[0];

    if (*workspaceSize < heristic.workspaceSize) {
        if (*workspace) {
            cudaFree(*workspace);
        }
        *algo = heristic.algo;
        cudaMalloc(workspace, heristic.workspaceSize);
        *workspaceSize = heristic.workspaceSize;
    }
    else {
        *algo = heristic.algo;
    }

    cublasLtMatmulPreferenceDestroy(preference);
}

cublasLtMatmulDesc_t DnnLinear::createAndSetDesc(cublasOperation_t transA, cublasOperation_t transB) {
    cublasLtMatmulDesc_t desc;
    BLAS_CHECK(cublasLtMatmulDescCreate(
        &desc,
        CUBLAS_COMPUTE_32F,
        CUDA_R_32F
    ));
    BLAS_CHECK(cublasLtMatmulDescSetAttribute(
        desc,
        CUBLASLT_MATMUL_DESC_TRANSA,
        &transA,
        sizeof(transA)));

    BLAS_CHECK(cublasLtMatmulDescSetAttribute(
        desc,
        CUBLASLT_MATMUL_DESC_TRANSB,
        &transB,
        sizeof(transB)));
    return desc;
}