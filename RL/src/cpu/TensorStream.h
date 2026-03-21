#pragma once
#include "tensor.h"
#include "binary.h"

namespace zeta {

    namespace TensorStream {
        void Save(const Tensor& tensor, BinaryStream& stream);
        Tensor Load(int in_dim, int out_dim , BinaryStream& stream);
        Tensor Load(int dim, BinaryStream& stream);
        Tensor Load(int N, int C, int H, int W, BinaryStream& stream);
    }
    
}