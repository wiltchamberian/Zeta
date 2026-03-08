#pragma once
#include "activationLayer.h"

namespace zeta {
    class CuTanhLayer : public ActivationLayer {
    public:
        CuTanhLayer();
        virtual void forward();
        virtual void backwardEx();
        virtual CuLayer* Clone() const override;
    };

}