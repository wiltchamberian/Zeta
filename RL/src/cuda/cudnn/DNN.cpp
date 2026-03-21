#include "DNN.h"
#include "DnnHelp.h"
#include "DnnTensor.h"
#include <cudnn_backend.h>
#include <cublasLt.h>
#include "cu_tool.h"

namespace zeta {
    DNN::DNN() : CuNN() {
        CU_CHECK(cudaStreamCreate(&stream));
        
        BLAS_CHECK(cublasLtCreate(&ltHandle));
        DNN_CHECK(cudnnCreate(&handle_));

        DNN_CHECK(cudnnSetStream(handle_, stream));
        //cublasSetStream(ltHandle,stream);
    }

    DNN::~DNN() {
        if (handle_) {
            DNN_CHECK(cudnnDestroy(handle_));
        }
        if (ltHandle) {
            BLAS_CHECK(cublasLtDestroy(ltHandle));
        }
    }

    void DNN::InitInput(const Tensor& tensor) {
        if (input == nullptr) {
            auto dnnTensor = std::make_unique<DnnTensor>();
            dnnTensor->InitShape(tensor);
            dnnTensor->Create();
            input = std::move(dnnTensor);
        }
        else {
            TensorShape ts = getTensorShape(tensor);
            if (input->shape != ts) {
                input->InitShape(ts);
                input->Create();
            }
            else {

            }
        }
    }

    void DNN::Connect(CuLayer* l1, CuLayer* l2) {
        l1->nexts.push_back(l2);
        l2->prevs.push_back(l1);
        if (l1->output == nullptr) {
            DnnTensor* tensor = this->CreateTensor<DnnTensor>();
            l1->output = tensor;
            l2->input = tensor;
        }
        else {
            l2->input = l1->output;
        }
        return;

    }

    CuNN* DNN::Clone() const {
        DNN* nn = new DNN();
        nn->c = this->c;
        nn->learningRate = this->learningRate;
        nn->optimizerType = this->optimizerType;
        for (int i = 0; i < tensors.size(); ++i) {
            CuTensor* tensor = tensors[i]->Clone();
            tensors[i]->ref = tensor;
            nn->tensors.push_back(std::unique_ptr<CuTensor>(tensor));
        }
        for (int i = 0; i < layers.size(); ++i) {
            CuLayer* newLayer = layers[i]->Clone();
            newLayer->SetNN(nn);
            layers[i]->ref = newLayer;
            nn->layers.push_back(std::unique_ptr<CuLayer>(newLayer));
        }
        for (int i = 0; i < layers.size(); ++i) {
            for (auto& l : layers[i]->prevs) {
                layers[i]->ref->prevs.push_back(l->ref);
            }
            for (auto& l : layers[i]->nexts) {
                layers[i]->ref->nexts.push_back(l->ref);
            }
            if (layers[i]->input) {
                layers[i]->ref->input = layers[i]->input->ref;
            }
            if (layers[i]->output) {
                layers[i]->ref->output = layers[i]->output->ref;
            }
            //layers[i]->ref = nullptr;
        }
        nn->CopyAndBindDeviceMemory(deviceMemory, deviceMemorySize);
        return nn;
    }

    void DNN::Save(const std::string& path) const {
        
    }
}