#pragma once

#include "CuNN.h"
#include <cudnn_backend.h>

class DNN : public CuNN {
public:

    cudnnHandle_t handle_;
};