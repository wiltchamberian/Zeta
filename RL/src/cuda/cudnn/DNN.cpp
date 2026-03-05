#include "DNN.h"
#include "DnnHelp.h"
#include "DnnTensor.h"
#include <cudnn_backend.h>
#include <cublasLt.h>

DNN::DNN(): CuNN() {

    BLAS_CHECK(cublasLtCreate(&ltHandle));
    DNN_CHECK(cudnnCreate(&handle_));
}

DNN::~DNN() {
    if (handle_) {
        DNN_CHECK(cudnnDestroy(handle_));
    }
    if (ltHandle) {
        BLAS_CHECK(cublasLtDestroy(ltHandle));
    }
}

void DNN::InitInput(const Tensor& tensor) {
    if (input == nullptr) {
        auto dnnTensor = std::make_unique<DnnTensor>();
        dnnTensor->InitShape(tensor);
        dnnTensor->Create();
        input = std::move(dnnTensor);
    }
    else {
        TensorShape ts = getTensorShape(tensor);
        if (input->shape != ts) {
            input->InitShape(ts);
            input->Create();
        }
        else {

        }
    }
}

void DNN::Connect(CuLayer* l1, CuLayer* l2) {
    l1->nexts.push_back(l2);
    l2->prevs.push_back(l1);
    if (l1->output == nullptr) {
        DnnTensor* tensor = this->CreateTensor<DnnTensor>();
        l1->output = tensor;
        l2->input = tensor;
    }
    else {
        ;
    }
    return;

}