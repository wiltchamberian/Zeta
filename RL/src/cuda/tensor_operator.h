#pragma once

#include "tensor.h"

namespace zeta {
    int ToDevice(const Tensor& tensor, void** addr);

    void ToTensor(Tensor& tensor, void* addr);
}