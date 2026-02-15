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

void CuNN::AddLayer(std::unique_ptr<CuLayer> layer) {
    if (layers.empty()) {
        layer->prev = nullptr;
        layer->next = nullptr;
    }
    else {
        layer->prev = layers.back().get();
        layer->next = nullptr;
        layers.back()->next = layer.get();
    }
    layers.push_back(std::move(layer));
}

void CuNN::Build(TensorShape shape) {
    
    for (int i = 0; i < layers.size(); ++i) {
        shape = layers[i]->InferOutputShape(shape);
    }
    AllocDeviceMemory();
}

void CuNN::Clear() {
    ReleaseDeviceMemory();
    layers.clear();
}

void CuNN::AllocDeviceMemory() {
    // 1️⃣ 计算总大小：weights + grad_w + bias + grad_b
    size_t total = 0;
    for (auto& l : layers) {
        size_t w = l->weights.numel();
        size_t b = l->b.numel();
        total += w;  // weights
        total += w;  // grad_w
        total += b;  // bias
        total += b;  // grad_b
    }

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
    for (int i = 0; i < layers.size(); ++i) {
        auto& layer = layers[i];

        // -------- weights --------
        Tensor w = layer->weights.contiguous();
        size_t w_size = w.numel();
        layer->dl.weights = reinterpret_cast<float*>(addr);
        layer->dl.w_size = w_size;
        layer->dl.in_dim = w_size / w.shape[0];//layer->weights.shape[1];

        CUDA_CHECK(cudaMemcpy(
            layer->dl.weights,
            w.data(),
            w_size * sizeof(float),
            cudaMemcpyHostToDevice
        ));
        addr += w_size * sizeof(float);

        // -------- bias --------
        Tensor b = layer->b.contiguous();
        size_t b_size = b.numel();
        layer->dl.bias = reinterpret_cast<float*>(addr);
        layer->dl.b_size = b_size;

        CUDA_CHECK(cudaMemcpy(
            layer->dl.bias,
            b.data(),
            b_size * sizeof(float),
            cudaMemcpyHostToDevice
        ));
        addr += b_size * sizeof(float);

        // -------- grad_w --------
        layer->dl.grad_w = reinterpret_cast<float*>(addr);
        addr += w_size * sizeof(float);

        // -------- grad_b --------
        layer->dl.grad_b = reinterpret_cast<float*>(addr);
        addr += b_size * sizeof(float);
    }
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

    // Forward activations 和 Backward deltas
    for (auto& l : layers) {
        total += l->GetWorkspaceSize();
    }

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

    
    for (auto& l : layers) {
        l->BindWorkspace(addr);
        addr += l->GetWorkspaceSize();
    }

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

    // 2️⃣ 遍历每层
    for (int l = 0; l < layers.size(); ++l) {
        float* input_ptr = (l == 0) ? ws.x : layers[l - 1]->dl.activation;  //ws.activations[l - 1];
        //A = sigma(X * W^T)
        layers[l]->forward(input_ptr);
    }

    
}

Tensor CuNN::ForwardAndFetchPredY(const Tensor& x) {
    Forward(x);
    // 3️⃣ 返回最后一层 activations (host 端 Tensor)
    TensorShape sp = layers.back()->outputShape;
    Shape shape({sp.N, sp.C, sp.H, sp.W});

    Tensor output(shape);
    CUDA_CHECK(cudaMemcpy((void*)output.data(), layers.back()->dl.activation, output.numel() * sizeof(float), cudaMemcpyDeviceToHost));

    return output;

}


void CuNN::Backward(const Tensor& ys) {
    label = ys;

    // 1️⃣ 确保 workspace 足够大
    AllocWorkSpaceIfNeeded();

    // 2️⃣ 拷贝标签 ys 到 GPU, xs will be copied in Forward
    CUDA_CHECK(cudaMemcpy(ws.y, ys.data(), ys.numel() * sizeof(float), cudaMemcpyHostToDevice));

    // 3️⃣ Forward pass: batch 全部激活
    //Forward(xs); // ws.activations 会被填充

    // 4️⃣ Backward pass
    // 最后一层 delta
    int L = layers.size() - 1;
    int out_dim = ys.numel()/ys.shape[0];
    int batch = GetBatchSize();

    dim3 block(TILE_WIDTH, TILE_WIDTH);
    dim3 grid((out_dim + block.x - 1) / block.x, (batch + block.y - 1) / block.y);

    //(BP1) δ^L = (a^L - y) ⊙ σ'(z^L)
    mse_loss_kernel << <grid, block >> > (
        layers[L]->dl.activation,    //ws.activations[L], // a^L
        ws.y,              // y
        layers[L]->dl.delta,      // δ^L 输出
        batch,
        out_dim,
        layers[L]->alpha
        );

    //(BP2) δ^l = (δ^{l+1} · W^{l+1}) ⊙ σ'(z^l)
    for (int l = L - 1; l >= 0; --l) {
        layers[l]->backward(layers[l+1]->dl.delta, layers[l+1]->dl.weights);
    }

    //Compute grad_w and grad_b for each layer
    for (int l = 0; l < layers.size(); ++l) {
        //BP4
        layers[l]->wgrad((l == 0 ? ws.x : layers[l - 1]->dl.activation/*ws.activations[l - 1]*/));
        //BP3
        layers[l]->bgrad();

    }
}


void CuNN::Step() {
    for (auto& layer : layers) {
        int CPQ = layer->weights.numel() / layer->weights.shape[0];
        int K = layer->weights.shape[0];
        int block_y = (K + TILE_WIDTH - 1) / TILE_WIDTH;
        int block_x = (CPQ + TILE_WIDTH - 1) / TILE_WIDTH;
        dim3 grid(block_x, block_y);
        dim3 block(TILE_WIDTH, TILE_WIDTH);
        apply_gradien_kernel <<<grid, block>>>(layer->dl.grad_w, layer->dl.grad_b, layer->dl.weights, layer->dl.bias, K, CPQ, learningRate);
    }
}

void CuNN::FetchGrad() {
    for (int i = 0; i < layers.size(); ++i) {
        layers[i]->weights_grad.zeros(layers[i]->dl.b_size, layers[i]->dl.in_dim);
        layers[i]->bias_grad.zeros(layers[i]->dl.b_size);
        CUDA_CHECK(cudaMemcpy(layers[i]->weights_grad.data(), layers[i]->dl.grad_w, layers[i]->dl.w_size * sizeof(float), cudaMemcpyKind::cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(layers[i]->bias_grad.data(), layers[i]->dl.grad_b, layers[i]->dl.b_size * sizeof(float), cudaMemcpyKind::cudaMemcpyDeviceToHost));
    }

}


void CuNN::FetchResultToCpu() {
    for (int i = 0; i < layers.size(); ++i) {
        layers[i]->weights.zeros(layers[i]->dl.b_size, layers[i]->dl.in_dim);
        layers[i]->b.zeros(layers[i]->dl.b_size);
        CUDA_CHECK(cudaMemcpy(layers[i]->weights.data(), layers[i]->dl.weights, layers[i]->dl.w_size * sizeof(float), cudaMemcpyKind::cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(layers[i]->b.data(), layers[i]->dl.bias, layers[i]->dl.b_size * sizeof(float), cudaMemcpyKind::cudaMemcpyDeviceToHost));
    }
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
    for (int i = 0; i < layers.size(); ++i) {
        std::cout << "layer:" << i << std::endl;
        auto& data = layers[i]->data();
        std::cout << "weights:\n";
        data.print("W_");
        std::cout << "biases:\n";
        layers[i]->b.print("B_");
        std::cout << std::endl << std::endl;
    }
}

void CuNN::PrintGrad() {
    for (int i = 0; i < layers.size(); ++i) {
        std::cout << "weights_grad:\n";
        layers[i]->weights_grad.print("W_");
        std::cout << "bias_grad:\n";
        layers[i]->bias_grad.print("B_");
    }
}
