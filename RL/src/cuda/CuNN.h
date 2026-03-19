#pragma once
#include "Layer.h"
#include "Activation.h"
#include "tensor.h"
#include "cuLayer.h"
#include <memory>
#include <functional>

namespace zeta {
    using Sample = std::vector<float>;


    enum OptimizerType {
        SGD,
        Adam,
    };

    struct CuNNWorkspace {
        void Clear() {
            x = nullptr;
            y = nullptr;
            loss_vec = nullptr;
            loss = nullptr;
        }
        // ----------------- Forward -----------------
        float* x = nullptr;                           // 输入激活 X, size = batch * in_dim
        // ----------------- 输出 / Loss -----------------
        float* y = nullptr;           // batch 内标签 y, size = batch * output_dim
        float* loss_vec = nullptr;    // batch 内每个样本 loss, size = batch
        float* loss = nullptr;        // 总 loss, size = 1
    };

    class CuHead {
    public:
        std::vector<float> policy;
        float value;
    };

    class CuNN
    {
    public:
        float learningRate = 1.0;
        OptimizerType optimizerType = SGD;
        float beta1 = 0.9;
        float beta2 = 0.999f;
        float beta1_t = 1.0f;
        float beta2_t = 1.0f;
        float epsilon = 1e-8;
        int t = 0;

        CuNN(float lr = 1.0);

        virtual ~CuNN();

        void ResetOptimizer();

        template<typename T, typename... Args>
        T* CreateLayer(Args&&... args)
        {
            layers.emplace_back(std::make_unique<T>(std::forward<Args>(args)...));
            T* res = dynamic_cast<T*>(layers.back().get());
            res->nn = this;
            return res;
        }

        template<typename T, typename... Args>
        T* CreateTensor(Args&&... args)
        {
            tensors.emplace_back(std::make_unique<T>(std::forward<Args>(args)...));
            T* res = static_cast<T*>(tensors.back().get());
            return res;
        }

        void SetOptimizer(OptimizerType opt);

        virtual void InitInput(const Tensor& tensor);

        virtual void Connect(CuLayer* l1, CuLayer* l2);

        //reset to the state of just created, only keep the learning rate
        void Clear();

        void SetLearningRate(float lr) {
            learningRate = lr;
        }

        void SetLabel(const Tensor& tensor);

        //deprecated
        TensorShape Build(TensorShape shape);

        void AllocDeviceMemory();

        void CopyAndBindDeviceMemory(void* deviceMemory, size_t siz);

        void AllocWorkSpaceIfNeeded();

        void Forward(const Tensor& x);

        void Backward();

        void Step();

        void FetchGrad();

        void FetchResultToCpu();

        void SetHead(CuLayer* l);
        void SetTail(CuLayer* l);

        float MseLoss(Tensor& xs, Tensor& ys);

        void Print();

        CuLayer* findRootLayer() const;

        void Travel(std::function<bool(CuLayer*)> ff);

        void TravelBackward(std::function<void(CuLayer*)> ff);

        void PrintGrad();

        size_t GetBatchSize() const;

        void Save(const std::string& path) const;

        void ReleaseDeviceMemory();

        void ErrorCheck() const;

        virtual CuNN* Clone() const;

        void CleanRefs(); //use with clone

        int batchSize = 0;
        float c = 0.0f; //regularization parameter

        TensorShape getTensorShape(const Tensor& tensor) const;
        void* GetDeviceMemory() { return deviceMemory; }
        void* GetWorkspace() { return deviceWorkspace; }
        int GetDeiviceSize() { return deviceMemorySize; }
        int GetWorkspaceSize() { return workspaceSize; }
    protected:
        //backup of input and label y
        //Tensor input;
        Tensor label;

        std::unique_ptr<CuTensor> input;

        CuLayer* head = nullptr;
        CuLayer* tail = nullptr;//to loss
        std::vector<std::unique_ptr<CuLayer>> layers;
        std::vector<std::unique_ptr<CuTensor>> tensors;


        //device memory manager, all used memory use one buffer....
        //I feel it is crazy but this seems most efficent.
        //layout: x, layer1,layer2,...layer_n, y
        //layer_i:
        //W,b,dW,db, z,a,
        void* deviceMemory;
        int deviceMemorySize;

        void* deviceWorkspace;
        size_t workspaceSize = 0;
        CuNNWorkspace ws;


    };
}