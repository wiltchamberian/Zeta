#include "TensorStream.h"

namespace zeta {
    namespace TensorStream {
        void Save(const Tensor& tensor, BinaryStream& stream) {
            if (tensor.is_continuous()) {
                auto d = tensor.start();
                int siz = tensor.ElementCount();
                stream.writeBytes(d, siz * sizeof(Tensor::ElementType));
                
            }
            else {
                assert(false);
                //TODO: not support yet..
            }

        }

        Tensor Load(int in_dim, int out_dim, BinaryStream& stream) {
            Tensor tensor;
            tensor.zeros(out_dim, in_dim);
            stream.readBytes(tensor.data(), tensor.ElementCount() * sizeof(Tensor::ElementType));
            return tensor;
        }

        Tensor Load(int dim, BinaryStream& stream) {
            Tensor tensor;
            tensor.zeros(dim);
            stream.readBytes(tensor.data(), tensor.ElementCount() * sizeof(Tensor::ElementType));
            return tensor;
        }

        Tensor Load(int N, int C, int H, int W, BinaryStream& stream) {
            Tensor tensor;
            tensor.zeros(N,C,H,W);
            stream.readBytes(tensor.data(), tensor.ElementCount() * sizeof(Tensor::ElementType));
            return tensor;
        }
    }
    
}
