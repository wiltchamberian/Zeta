#pragma once

#include "tensor.h"

int ToDevice(const Tensor& tensor, void** addr);

void ToTensor(Tensor& tensor, void* addr);
