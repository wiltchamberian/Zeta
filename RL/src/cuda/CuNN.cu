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

    // 2️⃣ 一次性分配
    CUDA_CHECK(cudaMalloc(&deviceMemory, total * sizeof(double)));

    char* addr = static_cast<char*>(deviceMemory);

    // 3️⃣ 遍历每层，分配指针并拷贝 host 数据
    for (int i = 0; i < layers.size(); ++i) {
        auto& layer = layers[i];

        // -------- weights --------
        Tensor w = layer->weights.contiguous();
        size_t w_size = w.numel();
        layer->dl.weights = reinterpret_cast<double*>(addr);
        layer->dl.w_size = w_size;
        layer->dl.in_dim = layer->weights.shape[1];

        CUDA_CHECK(cudaMemcpy(
            layer->dl.weights,
            w.data(),
            w_size * sizeof(double),
            cudaMemcpyHostToDevice
        ));
        addr += w_size * sizeof(double);

        // -------- bias --------
        Tensor b = layer->b.contiguous();
        size_t b_size = b.numel();
        layer->dl.bias = reinterpret_cast<double*>(addr);
        layer->dl.b_size = b_size;

        CUDA_CHECK(cudaMemcpy(
            layer->dl.bias,
            b.data(),
            b_size * sizeof(double),
            cudaMemcpyHostToDevice
        ));
        addr += b_size * sizeof(double);

        // -------- grad_w --------
        layer->dl.grad_w = reinterpret_cast<double*>(addr);
        CUDA_CHECK(cudaMemset(layer->dl.grad_w, 0, w_size * sizeof(double)));
        addr += w_size * sizeof(double);

      
        // -------- grad_b --------
        layer->dl.grad_b = reinterpret_cast<double*>(addr);
        CUDA_CHECK(cudaMemset(layer->dl.grad_b, 0, b_size * sizeof(double)));
        addr += b_size * sizeof(double);
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
    total += layers[0]->inputShape.NumElements() * sizeof(double);

    // Forward activations 和 Backward deltas
    for (auto& l : layers) {
        total += l->GetWorkspaceSize();
    }

    // 输出层 label 和 loss
    int out_dim = layers.back()->outputShape.Dim();
    total += batchSize * out_dim * sizeof(double); // 标签 y
    total += batchSize * sizeof(double);           // 每个样本 loss
    total += sizeof(double);                   // 总 loss

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
    ws.x = reinterpret_cast<double*>(addr);
    addr += batchSize * in_dim * sizeof(double);

    
    for (auto& l : layers) {
        l->BindWorkspace(addr);
        addr += l->GetWorkspaceSize();
    }

    // 标签 y
    ws.y = reinterpret_cast<double*>(addr);
    addr += batchSize * out_dim * sizeof(double);

    // 每个样本 loss
    ws.loss_vec = reinterpret_cast<double*>(addr);
    addr += batchSize * sizeof(double);

    // 总 loss
    ws.loss = reinterpret_cast<double*>(addr);
}


void CuNN::Forward(const Tensor& x) {
    AllocWorkSpaceIfNeeded();

    // 1️⃣ 拷贝输入 x 到 workspace
    CUDA_CHECK(cudaMemcpy(ws.x, x.data(), x.numel() * sizeof(double), cudaMemcpyHostToDevice));

    // 2️⃣ 遍历每层
    for (int l = 0; l < layers.size(); ++l) {
        double* input_ptr = (l == 0) ? ws.x : layers[l - 1]->dl.activation;  //ws.activations[l - 1];
        //A = sigma(X * W^T)
        layers[l]->forward(input_ptr);
    }

    
}

Tensor CuNN::ForwardAndFetch(const Tensor& x) {
    Forward(x);
    // 3️⃣ 返回最后一层 activations (host 端 Tensor)
    Tensor output(layers.back()->b.numel(), GetBatchSize());
    CUDA_CHECK(cudaMemcpy((void*)output.data(), layers.back()->dl.activation, output.numel() * sizeof(double), cudaMemcpyDeviceToHost));

    return output;

}


void CuNN::Backward(Tensor& xs, Tensor& ys) {
    // 1️⃣ 确保 workspace 足够大
    AllocWorkSpaceIfNeeded();

    // 2️⃣ 拷贝标签 ys 到 GPU, xs will be copied in Forward
    CUDA_CHECK(cudaMemcpy(ws.y, ys.data(), ys.numel() * sizeof(double), cudaMemcpyHostToDevice));

    // 3️⃣ Forward pass: batch 全部激活
    Forward(xs); // ws.activations 会被填充

    // 4️⃣ Backward pass
    // 最后一层 delta
    int L = layers.size() - 1;
    int out_dim = layers[L]->b.size();
    int batch = GetBatchSize();

    dim3 block(TILE_WIDTH, TILE_WIDTH);
    dim3 grid((batch + block.x - 1) / block.x, (out_dim + block.y - 1) / block.y);

    //(BP1) δ^L = (a^L - y) ⊙ σ'(z^L)
    mse_loss_kernel << <grid, block >> > (
        layers[L]->dl.activation,    //ws.activations[L], // a^L
        ws.y,              // y
        layers[L]->dl.delta,      // δ^L 输出
        batch,
        out_dim
        );

    //CUDA_CHECK(cudaDeviceSynchronize());

    //(BP2) δ^l = (δ^{l+1} · W^{l+1}) ⊙ σ'(z^l)
    for (int l = L - 1; l >= 0; --l) {
        layers[l]->backward(layers[l+1]->dl.delta, layers[l+1]->dl.weights);
    }

    //Compute grad_w and grad_b for each layer
    for (int l = 0; l < layers.size(); ++l) {
        int dim_delta_prev = (l == 0) ? xs.shape[1] : layers[l - 1]->b.size();
        int dim_delta = layers[l]->b.size();

        //BP4
        layers[l]->wgrad((l == 0 ? ws.x : layers[l - 1]->dl.activation/*ws.activations[l - 1]*/));

        //BP3
        layers[l]->bgrad();

    }
}


void CuNN::Step() {
    for (auto& layer : layers) {
        int block_y = (layer->dl.b_size + TILE_WIDTH - 1) / TILE_WIDTH;
        int block_x = (layer->dl.in_dim + TILE_WIDTH - 1) / TILE_WIDTH;
        dim3 grid(block_x, block_y);
        dim3 block(TILE_WIDTH, TILE_WIDTH);
        apply_gradien_kernel <<<grid, block>>>(layer->dl.grad_w, layer->dl.grad_b, layer->dl.weights, layer->dl.bias, layer->dl.b_size, layer->dl.in_dim, learningRate);
    }
}

void CuNN::FetchResultToCpu() {
    for (int i = 0; i < layers.size(); ++i) {
        layers[i]->weights.zeros(layers[i]->dl.b_size, layers[i]->dl.in_dim);
        layers[i]->b.zeros(layers[i]->dl.b_size);
        CUDA_CHECK(cudaMemcpy(layers[i]->weights.data(), layers[i]->dl.weights, layers[i]->dl.w_size * sizeof(double), cudaMemcpyKind::cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(layers[i]->b.data(), layers[i]->dl.bias, layers[i]->dl.b_size * sizeof(double), cudaMemcpyKind::cudaMemcpyDeviceToHost));
    }
}

double CuNN::MseLoss(Tensor& xs, Tensor& ys) {
    double totalLoss = 0.0;
    for (int i = 0; i < xs.size(); ++i) {
        Tensor pred = xs[i];
        for (int j = 0; j < pred.size(); ++j) {
            double diff = pred(j) - ys(i,j);
            totalLoss += diff * diff;
        }
    }
    return totalLoss / xs.shape[0];
}

void CuNN::Train(Tensor& xs, Tensor& ys, int maxEpochs, double tolerance) {
    for (int epoch = 0; epoch < maxEpochs; ++epoch) {

        double loss = MseLoss(xs, ys);
        std::cout << "Epoch " << epoch << " Loss: " << loss << std::endl;

        Backward(xs, ys);


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
        for (int j = 0; j < data.shape[0]; ++j) {
            for (int k = 0; k < data.shape[1]; ++k) {
                std::cout << "W_" << k << "," << j << "=" << data(j,k) << " ";
            }
            std::cout << std::endl;
        }
        for (int j = 0; j < layers[i]->b.numel(); ++j) {
            std::cout << "B_" << j << "=" << layers[i]->b(j) << " ";
        }
        std::cout << std::endl << std::endl;
    }
}
