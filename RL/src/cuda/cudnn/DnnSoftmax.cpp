#include "DnnSoftmax.h"
#include <cudnn_backend.h>
//#include <cudnn_graph.h>
#include "DNN.h"
#include "DnnHelp.h"
#include "cu_tool.h"
#include "DnnTensor.h"

void DnnSoftmax::forward() {

    float alpha = 1.0f;
    float beta = 0.0f;
    DNN_CHECK(cudnnSoftmaxForward(dnn->handle_,
        algo,
        mode,
        &alpha,
        input->desc->cudnnDesc,
        input->v,
        &beta,
        output->desc->cudnnDesc,
        output->v
    ));
}

void DnnSoftmax::backwardEx() {
    CuSoftmaxCrossEntropyLayer::backwardEx();
}

void DnnSoftmax::BindWorkspace(void* ptr) {
    CuSoftmaxCrossEntropyLayer::BindWorkspace(ptr);
    output->Create();
}

void DnnSoftmax::SetNN(CuNN* nn) {
    this->nn = nn;
    this->dnn = dynamic_cast<DNN*>(nn);
}

CuLayer* DnnSoftmax::Clone() const {
    DnnSoftmax* a = new DnnSoftmax();
    return a;
}