// dllmain.cpp : 定义 DLL 应用程序的入口点。
#include "zetalib.h"
#include "DnnHead.h"

using namespace zeta;


DnnHandle CreateDnn() {
    DNN* dnn = new DNN();
    return dnn;
}

void DestroyDnn(DnnHandle hd) {
    DNN* dnn = (DNN*)hd;
    if (dnn != nullptr) {
        delete dnn;
    }
    return;
}

void ClearDnn(DnnHandle hd) {
    DNN* dnn = (DNN*)hd;
    if (dnn != nullptr) {
        dnn->Clear();
    }
    return;
}

void SetLearningRate(DnnHandle handle, float lr) {
    DNN* dnn = (DNN*)handle;
    dnn->SetLearningRate(lr);
}

LayerHandle CreateConv2d(DnnHandle handle, TsShape shape, Size2D padding, Size2D stride) {
    DNN* dnn = (DNN*)handle;
    LayerHandle hd;
    hd.h = dnn->CreateDnnLayer<DnnConv>(shape.N, shape.B, shape.H, shape.W, padding, stride, true);
    hd.type = LayerType::Conv;
    return hd;
}

LayerHandle CreateConv2dNoBias(DnnHandle handle, TsShape shape, Size2D padding, Size2D stride) {
    DNN* dnn = (DNN*)handle;
    LayerHandle hd;
    hd.h = dnn->CreateDnnLayer<DnnConv>(shape.N, shape.B, shape.H, shape.W, padding, stride, false);
    hd.type = LayerType::Conv;
    return hd;
}

LayerHandle CreateLinear(DnnHandle handle, int input, int output) {
    DNN* dnn = (DNN*)handle;
    LayerHandle hd;
    hd.h = dnn->CreateDnnLayer<DnnLinear>(input, output);
    hd.type = LayerType::Fully;
    return hd;
}

LayerHandle CreateMse(DnnHandle handle) {
    DNN* dnn = (DNN*)handle;
    LayerHandle hd;
    hd.h = dnn->CreateLayer<CuMseLayer>();
    hd.type = LayerType::Mse;
    return hd;
}

LayerHandle CreateAdd(DnnHandle handle) {
    DNN* dnn = (DNN*)handle;
    LayerHandle h;
    h.h = dnn->CreateLayer<CuAddLayer>();
    h.type = LayerType::Add;
    return h;
}

LayerHandle CreateAct(DnnHandle handle, LayerType layerType) {
    DNN* dnn = (DNN*)handle;
    LayerHandle h;
    h.h = dnn->CreateDnnLayer<DnnAct>(layerType);
    h.type = layerType;
    return h;
}

LayerHandle CreateSoftmax(DnnHandle handle) {
    DNN* dnn = (DNN*)handle;
    LayerHandle hd;
    hd.h = dnn->CreateDnnLayer<DnnSoftmax>();
    hd.type = LayerType::Softmax;
    return hd;
}

LayerHandle CreateOutput(DnnHandle handle) {
    DNN* dnn = (DNN*)handle;
    LayerHandle hd;
    hd.h = dnn->CreateLayer<OutputLayer>();
    hd.type = LayerType::Output;
    return hd;
}

void SetWeights(LayerHandle handle, float* weights, int n) {

    if (handle.type == LayerType::Fully) {
        DnnLinear* layer = (DnnLinear*)(handle.h);
        layer->weights.setData(weights, n);
    }
    else if (handle.type == LayerType::Conv) {
        DnnConv* layer = (DnnConv*)(handle.h);
        layer->weights.setData(weights, n);
    }

}

void AddEdge(LayerHandle from, LayerHandle to) {
    CuLayer* c1 = (CuLayer*)from.h;
    CuLayer* c2 = (CuLayer*)to.h;
    c1->Add(c2);
}

void AllocDeviceMemory(DnnHandle h) {
    DNN* dnn = (DNN*)h;
    dnn->AllocDeviceMemory();
}

void FetchWeights(LayerHandle h, float* d, int* n) {
    if (d == nullptr) {
        if (h.type == LayerType::Fully) {
            DnnLinear* l = (DnnLinear*)(h.h);
            *n = l->in_dim* l->out_dim;
            return;
        }
        else if (h.type == LayerType::Conv) {
            DnnConv* l = (DnnConv*)(h.h);
            *n = l->in_dim * l->out_dim;
            return;
        }
    }
    else {
        if (h.type == LayerType::Fully) {
            DnnLinear* l = (DnnLinear*)(h.h);
            l->FetchResultToCpu();
            std::copy(l->weights.data(), l->weights.data() + l->weights.numel(), d);
            return;
        }
        else if (h.type == LayerType::Conv) {
            DnnConv* l = (DnnConv*)(h.h);
            l->FetchResultToCpu();
            std::copy(l->weights.data(), l->weights.data() + l->weights.numel(), d);
            return;
        }
    }

}

TensorHandle CreateTensor1d(int l) {
    Tensor* t = new Tensor(l);
    return t;
}
TensorHandle CreateTensor2d(int n, int b) {
    Tensor* t = new Tensor(n, b);
    return t;
}
TensorHandle CreateTensor3d(int n, int b, int h) {
    Tensor* t = new Tensor(n, b, h);
    return t;
}

TensorHandle CreateTensor4d(int n, int b, int h, int w) {
    Tensor* t = new Tensor(n, b, h, w);
    return t;
}

void TensorCopyFrom(TensorHandle h, float* from, int x) {
    Tensor* t = (Tensor*)h;
    float* d = t->data();
    memcpy(d, from, x * sizeof(float));
}

void BindLabelToDevice(LayerHandle h, float* data, int n) {
    if (h.type == LayerType::Softmax) {
        DnnSoftmax* l = (DnnSoftmax*)(h.h);
        if (l->label.shape.size() == 0 && l->input) {
            l->label = Tensor(l->input->shape.N, l->input->shape.C * l->input->shape.H * l->input->shape.W);
        }
        l->label.setData(data, n);
        l->BindLabelToDevice();
    }
    else if (h.type == LayerType::Mse) {
        CuMseLayer* l = (CuMseLayer*)(h.h);
        if (l->label.shape.size() == 0 && l->input) {
            l->label = Tensor(l->input->shape.N, l->input->shape.C * l->input->shape.H * l->input->shape.W);
        }
        l->label.setData(data, n);
        l->BindLabelToDevice();
    }
    return;
}

void Backward(DnnHandle h) {
    DNN* dnn = (DNN*)h;
    dnn->Backward();
}

void Foward(DnnHandle h, TensorHandle tensor) {
    DNN* dnn = (DNN*)h;
    dnn->Forward(*((Tensor*)tensor));
}

void Step(DnnHandle h) {
    DNN* dnn = (DNN*)h;
    dnn->Step();
}

DnnHandle Clone(DnnHandle h) {
    return ((DNN*)(h))->Clone();
}

float FetchLoss(LayerHandle h) {
    if (h.type == LayerType::Mse) {
        CuMseLayer* layer = (CuMseLayer*)(h.h);
        Tensor loss = layer->FetchLoss();
        return loss(0);
    }
    else if (h.type == LayerType::Softmax) {
        DnnSoftmax* layer = (DnnSoftmax*)(h.h);
        Tensor loss = layer->FetchLoss();
        return loss(0);
    }
    assert(false);
    return 0;
}

//may cause gpu to cpu devlier
void Print(TensorHandle h) {
    Tensor* tensor = (Tensor*)h;
    tensor->print_torch_style();
}



