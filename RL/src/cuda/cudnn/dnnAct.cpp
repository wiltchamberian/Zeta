#include "DnnAct.h"
#include <cudnn_backend.h>
#include <cudnn.h>
#include "DNN.h"
#include "DnnHelp.h"

namespace zeta {
    DnnAct::DnnAct()
        :ActivationLayer() {
        Init(LayerType::Act_Identity);
    }

    DnnAct::DnnAct(LayerType lt)
        :ActivationLayer(lt) {
        Init(lt);
    }

    void DnnAct::Init(LayerType lt) {
        layerType = lt;
        DNN_CHECK(cudnnCreateTensorDescriptor(&xDesc));
        DNN_CHECK(cudnnCreateTensorDescriptor(&yDesc));
        DNN_CHECK(cudnnCreateTensorDescriptor(&dyDesc));
        DNN_CHECK(cudnnCreateActivationDescriptor(&actDesc));
        cudnnActivationMode_t mode = CUDNN_ACTIVATION_IDENTITY;
        cudnnNanPropagation_t reluNanOpt = CUDNN_PROPAGATE_NAN;

        switch (lt) {
        case LayerType::Act_Relu:
        {
            mode = CUDNN_ACTIVATION_RELU;
        }
        break;
        case LayerType::Act_Tanh: {
            mode = CUDNN_ACTIVATION_TANH;
        }
                                break;
        case LayerType::Act_Sigmoid: {
            mode = CUDNN_ACTIVATION_SIGMOID;
        }
                                   break;
        case LayerType::Act_ClippedRelu: {
            mode = CUDNN_ACTIVATION_CLIPPED_RELU;
        }
                                       break;
        case LayerType::Act_Elu: {
            mode = CUDNN_ACTIVATION_ELU;
        }
                               break;
        case LayerType::Act_Identity: {
            mode = CUDNN_ACTIVATION_IDENTITY;
        }
                                    break;
        case LayerType::Act_SWISH: {
            mode = CUDNN_ACTIVATION_SWISH;
        }
                                 break;
        default:
            mode = CUDNN_ACTIVATION_IDENTITY;
            break;
        }

        DNN_CHECK(cudnnSetActivationDescriptor(
            actDesc,
            mode,          // 激活类型
            reluNanOpt,    // NaN 处理方式
            0           // 某些激活用到的参数
        ));
    }

    DnnAct::~DnnAct() {
        DNN_CHECK(cudnnDestroyTensorDescriptor(xDesc));
        DNN_CHECK(cudnnDestroyTensorDescriptor(yDesc));
        DNN_CHECK(cudnnDestroyTensorDescriptor(dyDesc));
        DNN_CHECK(cudnnDestroyActivationDescriptor(actDesc));
    }

    void DnnAct::forward() {
        float* x = input->v;
        if (x == nullptr) {
            assert(false);
            return;
        }

        float alpha1 = 1;
        float beta1 = 0;
        DNN_CHECK(cudnnActivationForward(dnn->handle_,
            actDesc,
            &alpha1,
            xDesc,
            x,
            &beta1,
            yDesc,
            output->v
        ));
    }

    void DnnAct::backwardEx() {
        add = false;
        if (prevs.empty()) return;
        float* x = input->v;
        float* dx = input->delta;

        float alpha1 = 1;
        float beta1 = prevs[0]->add ? 1 : 0;
        auto status = cudnnActivationBackward(dnn->handle_,
            actDesc,
            &alpha1,
            yDesc,
            output->v,
            dyDesc,
            output->delta,
            xDesc,
            x,
            &beta1,
            xDesc,
            dx
        );
        prevs[0]->add = true;
        if (status != CUDNN_STATUS_SUCCESS) {
            assert(false);
        }
    }

    void DnnAct::BindWorkspace(void* ptr) {
        ActivationLayer::BindWorkspace(ptr);
        output->Create();

        cudnnDataType_t dataType = CUDNN_DATA_FLOAT;

        int dimA[4] = { input->shape.N, input->shape.C, input->shape.H, input->shape.W };
        int strideA[4] = {};
        cudnnTensorFormat_t tensorFormat = CUDNN_TENSOR_NCHW;
        generateStrides(dimA, strideA, 4, tensorFormat);
        cudnnSetTensorNdDescriptor(xDesc, dataType, 4, dimA, strideA);

        int dimB[4] = { output->shape.N, output->shape.C, output->shape.H, output->shape.W };
        int strideB[4] = {};
        generateStrides(dimB, strideB, 4, tensorFormat);
        cudnnSetTensorNdDescriptor(yDesc, dataType, 4, dimB, strideB);
        cudnnSetTensorNdDescriptor(dyDesc, dataType, 4, dimB, strideB);
    }

    void DnnAct::SetNN(CuNN* nn) {
        this->nn = nn;
        this->dnn = dynamic_cast<DNN*>(nn);
    }

    CuLayer* DnnAct::Clone() const {
        auto res = new DnnAct(layerType);
        res->layerType = layerType;
        res->alpha = this->alpha;
        return res;
    }
}