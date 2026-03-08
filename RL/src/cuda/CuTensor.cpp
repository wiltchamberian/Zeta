#include "CuTensor.h"

namespace zeta {
    void CuTensor::InitShape(const Tensor& x) {
        TensorShape ts;
        int rk = x.rank();
        if (rk == 1) {
            ts.N = x.shape[0];
        }
        else if (rk == 2) {
            ts.N = x.shape[0];
            ts.C = x.shape[1];
        }
        else if (rk == 3) {
            ts.N = x.shape[0];
            ts.C = x.shape[1];
            ts.H = x.shape[2];
        }
        else if (rk == 4) {
            ts.N = x.shape[0];
            ts.C = x.shape[1];
            ts.H = x.shape[2];
            ts.W = x.shape[3];
        }
        else {
            assert(false);
        }
        this->shape = ts;
    }

    void CuTensor::InitShape(const TensorShape& sp) {
        this->shape = sp;
    }
}