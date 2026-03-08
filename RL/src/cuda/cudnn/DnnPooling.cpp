#include "DnnPooling.h"

#include <cudnn_backend.h>
#include "DNN.h"
#include "DnnHelp.h"
#include "cu_tool.h"
#include "DnnTensor.h"

DnnPooling::DnnPooling() {
    DNN_CHECK(cudnnCreatePoolingDescriptor(&PDesc));
   
    DNN_CHECK(cudnnSetPooling2dDescriptor(PDesc,
        mode,
        nanProg,
        h,w,
        0,0,
        h,w));
}

DnnPooling::DnnPooling(int H, int W)
    :MaxPool2d(H,W)
{
    init(h, w);
}

DnnPooling::~DnnPooling() {
    DNN_CHECK(cudnnDestroyPoolingDescriptor(PDesc));
}

void DnnPooling::forward() {

    float alpha = 1.0f;
    float beta = 0.0f;
    
    DNN_CHECK(cudnnPoolingForward(dnn->handle_,
        PDesc,
        &alpha,
        input->desc->cudnnDesc,
        input->v,
        &beta,
        output->desc->cudnnDesc,
        output->v
    ));
}

void DnnPooling::backwardEx() {
    add = false;
    float alpha = 1.0f;
    float beta = 0.0f;
    DNN_CHECK(cudnnPoolingBackward(dnn->handle_,
        PDesc,
        &alpha,
        output->desc->cudnnDesc,
        output->v,
        output->desc->cudnnDesc,
        output->delta,
        input->desc->cudnnDesc,
        input->v,
        &beta,
        input->desc->cudnnDesc,
        input->delta));
}

void DnnPooling::BindWorkspace(void* ptr) {
    MaxPool2d::BindWorkspace(ptr);
    output->Create();
}

void DnnPooling::SetNN(CuNN* nn) {
    this->nn = nn;
    this->dnn = dynamic_cast<DNN*>(nn);
}

CuLayer* DnnPooling::Clone() const {
    DnnPooling* pooling = new DnnPooling(this->h,this->w);
    return pooling;
}

void DnnPooling::init(int h, int w) {
    DNN_CHECK(cudnnCreatePoolingDescriptor(&PDesc));
    DNN_CHECK(cudnnSetPooling2dDescriptor(PDesc,
        mode,
        nanProg,
        h, w,
        0, 0,
        h, w));
}