#pragma once

#include "CuNN.h"
#include <cudnn_backend.h>
#include <cublasLt.h>

namespace zeta {
    class DNN : public CuNN {
    public:
        DNN();
        virtual ~DNN();
        template<typename T, typename... Args>
        T* CreateDnnLayer(Args&&... args)
        {
            layers.emplace_back(std::make_unique<T>(std::forward<Args>(args)...));
            T* res = dynamic_cast<T*>(layers.back().get());
            res->nn = this;
            res->dnn = this;
            return res;
        }
        virtual CuLayer* CreateLayerBy(LayerType tp) override;
        virtual void InitInput(const Tensor& tensor) override;
        virtual void Connect(CuLayer* l1, CuLayer* l2) override;
        virtual CuNN* Clone() const override;
        virtual void Save(const std::string& path) const override;
        virtual void Save(BinaryStream& stream) const override;
        virtual void Load(BinaryStream& stream) override;
        cudnnHandle_t handle_ = nullptr;
        cublasLtContext* ltHandle = nullptr;

        
    };
}