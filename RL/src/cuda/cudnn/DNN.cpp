#include "DNN.h"
#include "DnnHelp.h"

DNN::DNN(): CuNN() {

    DNN_CHECK(cudnnCreate(&handle_));
}

DNN::~DNN() {
    if (handle_) {
        DNN_CHECK(cudnnDestroy(handle_));
    }
}