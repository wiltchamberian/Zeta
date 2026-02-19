#include "CuNN.h"
#include <cmath>
#include <cassert>
#include <iostream>
#include <string>
#include "device_launch_parameters.h"
//#include <cudnn.h>
#include <cuda_runtime.h>
#include "cu_tool.h"

#include "CuActivation.h"
#include "kernels.h"

void CuNN::SetInput(const Tensor& tensor) {
    input = tensor;
}

void CuNN::SetLabel(const Tensor& tensor) {
    label = tensor;
}

void CuNN::AddLayer(std::shared_ptr<CuLayer> layer) {
    //if (layers.empty()) {
    //    layer->prev = nullptr;
    //    layer->next = nullptr;
    //}
    //else {
    //    layer->prev = layers.back().get();
    //    layer->next = nullptr;
    //    layers.back()->next = layer.get();
    //}
    layers.push_back(layer);
}

TensorShape CuNN::Build(TensorShape shape) {
    for (int i = 0; i < layers.size(); ++i) {
        shape = layers[i]->InferOutputShape(shape);
    }
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
    Travel([&total](CuLayer* l) {
        l->GetDeviceSize();
        });

    if (deviceMemory != nullptr) {
        CUDA_CHECK(cudaFree(deviceMemory));
        deviceMemorySize = 0;
    }

    // 2️⃣ 一次性分配
    deviceMemorySize = total * sizeof(float);
    CUDA_CHECK(cudaMalloc(&deviceMemory, deviceMemorySize));
    CUDA_CHECK(cudaMemset(deviceMemory, 0, deviceMemorySize));

    char* addr = static_cast<char*>(deviceMemory);

    // 3️⃣ 遍历每层，分配指针并拷贝 host 数据
    Travel([&addr](CuLayer* layer) {
        layer->BindDevice(addr);
        addr += layer->GetDeviceSize();
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
    total += layers[0]->inputShape.NumElements() * sizeof(float);

    Travel([&total](CuLayer* rover) {
        total += rover->GetWorkspaceSize();
        });

    // 输出层 label 和 loss
    int out_dim = layers.back()->outputShape.Dim();
    total += batchSize * out_dim * sizeof(float); // 标签 y
    total += batchSize * sizeof(float);           // 每个样本 loss
    total += sizeof(float);                   // 总 loss

    size_t bytes = total;

    // 2️⃣ 如果已有 workspace 内存够用就直接返回
    if (deviceWorkspace && workspaceSize >= bytes) return;

    // 3️⃣ 如果需要，释放旧内存并重新分配
    if (deviceWorkspace) {
        CUDA_CHECK(cudaFree(deviceWorkspace));
        deviceWorkspace = nullptr;
    }

    CUDA_CHECK(cudaMalloc(&deviceWorkspace, bytes));
    workspaceSize = bytes;

    // 4️⃣ 按顺序划分各个 buffer
    char* addr = static_cast<char*>(deviceWorkspace);

    // 输入 x
    ws.x = reinterpret_cast<float*>(addr);
    addr += batchSize * in_dim * sizeof(float);

    head->prevActivation = ws.x;
    Travel([&addr](CuLayer* layer) {
        layer->BindWorkspace(addr);
        addr += layer->GetWorkspaceSize();
        });

    // 标签 y
    ws.y = reinterpret_cast<float*>(addr);
    addr += batchSize * out_dim * sizeof(float);

    // 每个样本 loss
    ws.loss_vec = reinterpret_cast<float*>(addr);
    addr += batchSize * sizeof(float);

    // 总 loss
    ws.loss = reinterpret_cast<float*>(addr);
}


void CuNN::Forward(const Tensor& x) {
    //record shape of x, check legal
    input = x;

    AllocWorkSpaceIfNeeded();

    // 1️⃣ 拷贝输入 x 到 workspace
    CUDA_CHECK(cudaMemcpy(ws.x, x.data(), x.numel() * sizeof(float), cudaMemcpyHostToDevice));

    Travel([](CuLayer* rover) {
        rover->forward();
        });
}

void CuNN::Travel(std::function<void(CuLayer*)> func) {
    CuLayer* rover = head.get();
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
            func(rover);
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
    CuLayer* rover = tail.get();
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

Tensor CuNN::ForwardAndFetchPredY(const Tensor& x) {
    Forward(x);
    // 3️⃣ 返回最后一层 activations (host 端 Tensor)
    TensorShape sp = layers.back()->outputShape;
    Shape shape({sp.N, sp.C, sp.H, sp.W});

    Tensor output(shape);
    //CUDA_CHECK(cudaMemcpy((void*)output.data(), layers.back()->dl.activation, output.numel() * sizeof(float), cudaMemcpyDeviceToHost));

    return output;

}


void CuNN::Backward(const Tensor& ys) {
    label = ys;

    // 1️⃣ 确保 workspace 足够大
    AllocWorkSpaceIfNeeded();

    // 2️⃣ 拷贝标签 ys 到 GPU, xs will be copied in Forward
    CUDA_CHECK(cudaMemcpy(ws.y, ys.data(), ys.numel() * sizeof(float), cudaMemcpyHostToDevice));

    TravelBackward([](CuLayer* layer) {
        layer->backwardEx();
        });

}


void CuNN::Step() {
    //for (auto& layer : layers) {
    //    int CPQ = layer->weights.numel() / layer->weights.shape[0];
    //    int K = layer->weights.shape[0];
    //    int block_y = (K + TILE_WIDTH - 1) / TILE_WIDTH;
    //    int block_x = (CPQ + TILE_WIDTH - 1) / TILE_WIDTH;
    //    dim3 grid(block_x, block_y);
    //    dim3 block(TILE_WIDTH, TILE_WIDTH);
    //    apply_gradien_kernel <<<grid, block>>>(layer->dl.grad_w, layer->dl.grad_b, layer->dl.weights, layer->dl.bias, K, CPQ, learningRate);
    //}
}

void CuNN::FetchGrad() {
    //for (int i = 0; i < layers.size(); ++i) {
    //    layers[i]->weights_grad.zeros(layers[i]->dl.b_size, layers[i]->dl.in_dim);
    //    layers[i]->bias_grad.zeros(layers[i]->dl.b_size);
    //    CUDA_CHECK(cudaMemcpy(layers[i]->weights_grad.data(), layers[i]->dl.grad_w, layers[i]->dl.w_size * sizeof(float), cudaMemcpyKind::cudaMemcpyDeviceToHost));
    //    CUDA_CHECK(cudaMemcpy(layers[i]->bias_grad.data(), layers[i]->dl.grad_b, layers[i]->dl.b_size * sizeof(float), cudaMemcpyKind::cudaMemcpyDeviceToHost));
    //}

}


void CuNN::FetchResultToCpu() {
    //for (int i = 0; i < layers.size(); ++i) {
    //    layers[i]->weights.zeros(layers[i]->dl.b_size, layers[i]->dl.in_dim);
    //    layers[i]->b.zeros(layers[i]->dl.b_size);
    //    CUDA_CHECK(cudaMemcpy(layers[i]->weights.data(), layers[i]->dl.weights, layers[i]->dl.w_size * sizeof(float), cudaMemcpyKind::cudaMemcpyDeviceToHost));
    //    CUDA_CHECK(cudaMemcpy(layers[i]->b.data(), layers[i]->dl.bias, layers[i]->dl.b_size * sizeof(float), cudaMemcpyKind::cudaMemcpyDeviceToHost));
    //}
}

void CuNN::SetHead(std::shared_ptr<CuLayer> l) {
    head = l;
}

void CuNN::SetTail(std::shared_ptr<CuLayer> l) {
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

void CuNN::Train(Tensor& xs, Tensor& ys, int maxEpochs, float tolerance) {
    for (int epoch = 0; epoch < maxEpochs; ++epoch) {

        float loss = MseLoss(xs, ys);
        std::cout << "Epoch " << epoch << " Loss: " << loss << std::endl;

        Forward(xs);
        Backward(ys);


        if (loss < tolerance) {
            std::cout << "Converged at epoch " << epoch << std::endl;
            break;
        }
    }
}

size_t CuNN::GetBatchSize() const {
    if (layers.size() > 0) {
        return layers[0]->inputShape.N;
    }
    return 0;
}

void CuNN::ReleaseDeviceMemory() {
    CUDA_CHECK(cudaFree(deviceMemory));
    deviceMemory = nullptr;
    deviceMemorySize = 0;
    CUDA_CHECK(cudaFree(deviceWorkspace));
    deviceWorkspace = nullptr;
    workspaceSize = 0;
    ws.Clear();
}

void CuNN::Print() {
    //for (int i = 0; i < layers.size(); ++i) {
    //    std::cout << "layer:" << i << std::endl;
    //    auto& data = layers[i]->data();
    //    std::cout << "weights:\n";
    //    data.print("W_");
    //    std::cout << "biases:\n";
    //    layers[i]->b.print("B_");
    //    std::cout << std::endl << std::endl;
    //}
}

void CuNN::PrintGrad() {
    //for (int i = 0; i < layers.size(); ++i) {
    //    std::cout << "weights_grad:\n";
    //    layers[i]->weights_grad.print("W_");
    //    std::cout << "bias_grad:\n";
    //    layers[i]->bias_grad.print("B_");
    //}
}
