#pragma once
#include "activationLayer.h"

class CuReluLayer : public ActivationLayer {
public:
    CuReluLayer();
    virtual void forward();
    virtual void backwardEx();
    virtual CuLayer* Clone() const override;
};




