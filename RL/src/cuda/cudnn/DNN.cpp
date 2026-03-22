#include "DNN.h"
#include "DnnHelp.h"
#include "DnnTensor.h"
#include <cudnn_backend.h>
#include <unordered_map>
#include <cublasLt.h>
#include "cu_tool.h"
#include "DnnSoftmax.h"
#include "DnnPooling.h"
#include "DnnLinear.h"
#include "DnnConv.h"
#include "DnnAct.h"

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

    CuLayer* DNN::CreateLayerBy(LayerType tp) {
        CuLayer* layer = nullptr;

        switch (tp) {
        case LayerType::Basic:
            assert(false);
            break;

        case LayerType::Fully:
            layer = CreateDnnLayer<DnnLinear>();
            break;

        case LayerType::Conv:
            layer = CreateDnnLayer<DnnConv>();
            break;

        // ===== Activation 系列 =====
        case LayerType::Activation:
            layer = CreateDnnLayer<DnnAct>(LayerType::Activation);
            break;

        case LayerType::Act_Relu:
            layer = CreateDnnLayer<DnnAct>(LayerType::Act_Relu);
            break;

        case LayerType::Act_Tanh:
            layer = CreateDnnLayer<DnnAct>(LayerType::Act_Tanh);
            break;

        case LayerType::Act_Sigmoid:
            layer = CreateDnnLayer<DnnAct>(LayerType::Act_Sigmoid);  // 如果你没有专门类
            break;

        case LayerType::Act_ClippedRelu:
            layer = CreateDnnLayer<DnnAct>(LayerType::Act_ClippedRelu);
            break;

        case LayerType::Act_Elu:
            layer = CreateDnnLayer<DnnAct>(LayerType::Act_Elu);
            break;

        case LayerType::Act_Identity:
            layer = CreateDnnLayer<DnnAct>(LayerType::Act_Identity);
            break;

        case LayerType::Act_SWISH:
            layer = CreateDnnLayer<DnnAct>(LayerType::Act_SWISH);
            break;

        // ===== Loss =====
        case LayerType::Mse:
            layer = CuNN::CreateLayer<CuMseLayer>();
            break;

        case LayerType::Softmax:
            layer = CreateDnnLayer<DnnSoftmax>();
            break;

            // ===== 结构层 =====
        case LayerType::Add:
            layer = CuNN::CreateLayer<CuAddLayer>();
            break;

        case LayerType::Output:
            layer = CuNN::CreateLayer<OutputLayer>();
            break;

        case LayerType::MaxPooling:
            layer = CreateDnnLayer<DnnPooling>();
            break;

        default:
            throw std::runtime_error("Unknown LayerType");
        }

        return layer;

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

    void DNN::Save(BinaryStream& stream) const {
        stream.write<int>((int)NNType::DNN);
        stream.write<int>(tensors.size());
        std::unordered_map<CuTensor*, size_t> tensorIndex;
        for (int i = 0; i < tensors.size(); ++i) {
            tensorIndex.insert(std::make_pair(tensors[i].get(), i));
            tensors[i]->Save(stream);
        }
        tensorIndex.insert(std::make_pair(nullptr, -1));
        stream.write<int>(layers.size());

        std::unordered_map<CuLayer*, size_t> indexMap;
        indexMap.insert(std::make_pair(nullptr, -1));
        for (int i = 0; i < layers.size(); ++i) {
            indexMap.insert(std::make_pair(layers[i].get(), i));
            layers[i]->Save(stream);
        }
        for (int i = 0; i < layers.size(); ++i) {
            stream.write<int>(layers[i]->prevs.size());
            for (int k = 0; k < layers[i]->prevs.size(); ++k) {
                stream.write<int>(indexMap[layers[i]->prevs[k]]);
            }
            stream.write<int>(layers[i]->nexts.size());
            for (int k = 0; k < layers[i]->nexts.size(); ++k) {
                stream.write<int>(indexMap[layers[i]->nexts[k]]);
            }
            stream.write<int>(tensorIndex[layers[i]->input]);
            stream.write<int>(tensorIndex[layers[i]->output]);
        }
    }

    void DNN::Load(BinaryStream& stream) {
        NNType nnType = (NNType)stream.peek<int>();
        if (nnType != NNType::DNN) {
            std::cout << "NN Type Not Match\n";
            return;
        }
        stream.read<int>();

        tensors.clear();
        layers.clear();
        // =========================
        // 1. 读取 Tensor
        // =========================
        int tensorCount = stream.read<int>();
        for (int i = 0; i < tensorCount; ++i) {
            std::unique_ptr<DnnTensor> tensor = std::make_unique<DnnTensor>();
            tensor->Load(stream);
            tensors.push_back(std::move(tensor));
        }

        // =========================
        // 2. 读取 Layer
        // =========================
        int layerCount = stream.read<int>();
        for (int i = 0; i < layerCount; ++i) {
            LayerType tp = (LayerType)stream.peek<int>();
            CuLayer* layer = CreateLayerBy(tp);
            layer->Load(stream);
        }

        // =========================
        // 3. 重建连接关系
        // =========================
        for (int i = 0; i < layerCount; ++i) {

            // prevs
            int prevCount = stream.read<int>();
            layers[i]->prevs.resize(prevCount);
            for (int k = 0; k < prevCount; ++k) {
                int idx = stream.read<int>();
                layers[i]->prevs[k] = (idx == -1) ? nullptr : layers[idx].get();
            }

            // nexts
            int nextCount = stream.read<int>();
            layers[i]->nexts.resize(nextCount);
            for (int k = 0; k < nextCount; ++k) {
                int idx = stream.read<int>();
                layers[i]->nexts[k] = (idx == -1) ? nullptr : layers[idx].get();
            }

            // input tensor
            int inIdx = stream.read<int>();
            layers[i]->input = (inIdx == -1) ? nullptr : tensors[inIdx].get();

            // output tensor
            int outIdx = stream.read<int>();
            layers[i]->output = (outIdx == -1) ? nullptr : tensors[outIdx].get();
        }

        AllocDeviceMemory();
    }
}