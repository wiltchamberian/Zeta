#include "CuNN.h"
#include <cmath>
#include <cassert>
#include <iostream>
#include <string>
#include "device_launch_parameters.h"
//#include <cudnn.h>
#include <cuda_runtime.h>
#include "cu_tool.h"

#include "kernels.h"

void CuNN::SetInput(const Tensor& tensor) {
    input = tensor;
}

void CuNN::SetLabel(const Tensor& tensor) {
    label = tensor;
}

TensorShape CuNN::Build(TensorShape shape) {
    Travel([shape](CuLayer* l)->bool {
        l->InferOutputShape(shape);
        return false;
        });
    AllocDeviceMemory();
    return shape;
}

void CuNN::Clear() {
    ReleaseDeviceMemory();
    layers.clear();
}

void CuNN::AllocDeviceMemory() {
    // 1️⃣ 计算总大小：weights + grad_w + bias + grad_b
    size_t total = 0;
    Travel([&total](CuLayer* l)->bool {
        total += l->GetDeviceSize();
        return false;
        });

    if (deviceMemory != nullptr) {
        CU_CHECK(cudaFree(deviceMemory));
        deviceMemorySize = 0;
    }

    // 2️⃣ 一次性分配
    deviceMemorySize = total * sizeof(float);
    CU_CHECK(cudaMalloc(&deviceMemory, deviceMemorySize));
    CU_CHECK(cudaMemset(deviceMemory, 0, deviceMemorySize));

    char* addr = static_cast<char*>(deviceMemory);

    // 3️⃣ 遍历每层，分配指针并拷贝 host 数据
    Travel([&addr](CuLayer* layer)->bool {
        layer->BindDevice(addr);
        addr += layer->GetDeviceSize() * sizeof(float);
        return false;
     });

}

void CuNN::CopyAndBindDeviceMemory(void* memory, size_t siz){
    if (deviceMemory != nullptr) {
        CU_CHECK(cudaFree(deviceMemory));
        deviceMemorySize = 0;
    }

    // 2️⃣ 一次性分配
    deviceMemorySize = siz;
    CU_CHECK(cudaMalloc(&deviceMemory, deviceMemorySize));
    CU_CHECK(cudaMemcpy(deviceMemory, memory, deviceMemorySize, cudaMemcpyDeviceToDevice));

    char* addr = static_cast<char*>(deviceMemory);

    // 3️⃣ 遍历每层，分配指针并拷贝 host 数据
    Travel([&addr](CuLayer* layer)->bool {
        layer->BindDevice(addr);
        addr += layer->GetDeviceSize() * sizeof(float);
        return false;
        });

}

void CuNN::AllocWorkSpaceIfNeeded() {
    // 1️⃣ 计算总大小(单位：元素数)
    size_t total = 0;

    if (layers.size() == 0) {
        assert(false);
    }

    int batchSize = layers[0]->inputShape.N;

    // 输入 x
    int in_dim = layers[0]->inputShape.Dim();
    total += layers[0]->inputShape.NumElements();

    Travel([&total](CuLayer* rover)->bool {
        total += rover->GetWorkspaceSize();
        return false;
        });

    size_t bytes = total * sizeof(float);

    // 2️⃣ 如果已有 workspace 内存够用就直接返回
    if (deviceWorkspace && workspaceSize >= bytes) {
        CU_CHECK(cudaMemset(deviceWorkspace, 0, workspaceSize));
        //return;
    }
    else {
        // 3️⃣ 如果需要，释放旧内存并重新分配
        if (deviceWorkspace) {
            CU_CHECK(cudaFree(deviceWorkspace));
            deviceWorkspace = nullptr;
        }
        CU_CHECK(cudaMalloc(&deviceWorkspace, bytes));
        CU_CHECK(cudaMemset(deviceWorkspace, 0, bytes));
        workspaceSize = bytes;
    }


    // 4️⃣ 按顺序划分各个 buffer
    char* addr = static_cast<char*>(deviceWorkspace);

    // 输入 x
    ws.x = reinterpret_cast<float*>(addr);
    addr += batchSize * in_dim * sizeof(float);


    Travel([&addr, this](CuLayer* layer)->bool {
        if (layer->IsRoot()) {
            layer->prevActivation = ws.x;
        }
        layer->BindWorkspace(addr);
        addr += layer->GetWorkspaceSize() * sizeof(float);
        return false;
    });

}


void CuNN::Forward(const Tensor& x) {
    //record shape of x, check legal
    input = x;
    batchSize = x.shape[0];

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
    
    Travel([ts](CuLayer* l)->bool {
        l->InferOutputShape(ts);
        return false;
        });

    AllocWorkSpaceIfNeeded();

    // 1️⃣ 拷贝输入 x 到 workspace
    CU_CHECK(cudaMemcpy(ws.x, x.data(), x.numel() * sizeof(float), cudaMemcpyHostToDevice));

    Travel([](CuLayer* rover)->bool {
        rover->forward();
        return false;
        });
}

void CuNN::Travel(std::function<bool(CuLayer*)> func) {
    CuLayer* rover = nullptr;
    for (int i = 0; i < layers.size(); ++i) {
        if (layers[i]->IsRoot()) {
            rover = layers[i].get();
            break;
        }
    }
    std::vector<CuLayer*> stack;
    while (rover != nullptr) {
        rover->visit_count += 1;
        if (rover->visit_count < rover->prevs.size()) {
            if (!stack.empty()) {
                rover = stack.back();
                stack.pop_back();
            }
            else {
                rover = nullptr;
            }
        }
        else {
            rover->visit_count = 0;
            bool isReturn = func(rover);
            if (isReturn) {
                return;
            }
            if (rover->nexts.size() > 1) {
                for (int k = 1; k < rover->nexts.size(); ++k) {
                    stack.push_back(rover->nexts[k]);
                }
                rover = rover->nexts[0];
            }
            else if (!rover->nexts.empty()) {
                rover = rover->nexts[0];
            }
            else {
                if (!stack.empty()) {
                    rover = stack.back();
                    stack.pop_back();
                }
                else {
                    rover = nullptr;
                }
            }
        }
    }
}

void CuNN::TravelBackward(std::function<void(CuLayer*)> func) {
    CuLayer* rover = nullptr;
    for (int i = layers.size() - 1; i >= 0; --i) {
        if (layers[i]->IsTail()) {
            rover = layers[i].get();
            break;
        }
    }
    std::vector<CuLayer*> stack;
    while (rover != nullptr) {
        rover->visit_count += 1;
        if (rover->visit_count < rover->nexts.size()) {
            if (!stack.empty()) {
                rover = stack.back();
                stack.pop_back();
            }
            else {
                rover = nullptr;
            }
        }
        else {
            rover->visit_count = 0;
            func(rover);
            if (rover->prevs.size() > 1) {
                for (int k = 1; k < rover->prevs.size(); ++k) {
                    stack.push_back(rover->prevs[k]);
                }
                rover = rover->prevs[0];
            }
            else if (!rover->prevs.empty()) {
                rover = rover->prevs[0];
            }
            else {
                if (!stack.empty()) {
                    rover = stack.back();
                    stack.pop_back();
                }
                else {
                    rover = nullptr;
                }
            }
        }
    }

}

void CuNN::Backward() {
    TravelBackward([](CuLayer* layer) {
        layer->backwardEx();
        });

}


void CuNN::Step() {
    Travel([](CuLayer* l)->bool {
        l->applyGradient();
        return false;
        });
}

void CuNN::FetchGrad() {
    Travel([](CuLayer* l)->bool {
        l->FetchGradToCpu();
        return false;
        });
}


void CuNN::FetchResultToCpu() {
    Travel([](CuLayer* l)->bool {
        l->FetchResultToCpu();
        return false;
        });
}

void CuNN::SetHead(CuLayer* l) {
    head = l;
}

void CuNN::SetTail(CuLayer* l) {
    tail = l;
}

float CuNN::MseLoss(Tensor& xs, Tensor& ys) {
    float totalLoss = 0.0;
    for (int i = 0; i < xs.size(); ++i) {
        Tensor pred = xs[i];
        for (int j = 0; j < pred.size(); ++j) {
            float diff = pred(j) - ys(i,j);
            totalLoss += diff * diff;
        }
    }
    return totalLoss / xs.shape[0];
}

size_t CuNN::GetBatchSize() const {
    if (layers.size() > 0) {
        return layers[0]->inputShape.N;
    }
    return 0;
}

void CuNN::Save(const std::string& path) const {

}

void CuNN::ReleaseDeviceMemory() {
    CU_CHECK(cudaFree(deviceMemory));
    deviceMemory = nullptr;
    deviceMemorySize = 0;
    CU_CHECK(cudaFree(deviceWorkspace));
    deviceWorkspace = nullptr;
    workspaceSize = 0;
    ws.Clear();
}

void CuNN::ErrorCheck() const {
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("Kernel launch error: %s\n", cudaGetErrorString(err));
        assert(false);
    }
}

CuNN* CuNN::Clone() const {
    CuNN* nn = new CuNN();
    nn->c = this->c;
    for (int i = 0; i < layers.size(); ++i) {
        CuLayer* newLayer = layers[i]->Clone();
        newLayer->nn = nn;
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
        //layers[i]->ref = nullptr;
    }
    nn->CopyAndBindDeviceMemory(deviceMemory, deviceMemorySize);
    return nn;
}

void CuNN::CleanRefs() {
    for (int i = 0; i < layers.size();++i) {
        layers[i]->ref = nullptr;
    }
}

void CuNN::Print() {
    Travel([](CuLayer* l)->bool {
        l->Print();
        return false;
        });
    std::cout << std::endl;
}

void CuNN::PrintGrad() {
    Travel([](CuLayer* l)->bool {
        l->PrintGrad();
        return false;
        });
    std::cout << std::endl;
}
