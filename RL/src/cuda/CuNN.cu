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

#define TILE_WIDTH 16


//input:X, W
//outupt := sigma(X * W^T)
__global__ void ForwardKernel(
    const double* input,      // batch x in_dim
    const double* weights,    // out_dim x in_dim
    const double* bias,       // out_dim
    double* output,           // batch x out_dim
    int batch, int in_dim, int out_dim
) {
    int by = blockIdx.y;
    int bx = blockIdx.x;

    int ty = threadIdx.y;
    int tx = threadIdx.x;

    // Working on C[i,j]
    int i = blockDim.y * by + ty;
    int j = blockDim.x * bx + tx;



    __shared__ double sh_A[TILE_WIDTH][TILE_WIDTH];
    __shared__ double sh_B[TILE_WIDTH][TILE_WIDTH];

    double value = 0;
    int phaseCount = (in_dim + TILE_WIDTH - 1) / TILE_WIDTH;

    for (int phase = 0; phase < phaseCount; phase++)
    {
        int A_y = i;
        int A_x = phase * TILE_WIDTH + tx;
        if (A_y < batch && A_x < in_dim) {
            sh_A[ty][tx] = input[A_y * in_dim + A_x];
        }
        else {
            sh_A[ty][tx] = 0;
        }

        int B_x = j;
        int B_y = phase * TILE_WIDTH + ty;
        if (B_y < in_dim && B_x < out_dim) {
            //weights^T[B_y,B_x] = weights[B_x,B_y]
            sh_B[tx][ty] = weights[B_x * in_dim + B_y];
        }
        else {
            sh_B[tx][ty] = 0;
        }
        __syncthreads();

        // Dot product
        for (int k = 0; k < TILE_WIDTH; k++) {
            value += sh_A[ty][k] * sh_B[tx][k];
        }

        __syncthreads();
    }

    if (i >= batch || j >= out_dim) {
        return;
    }

    value += bias[j];

    // add activation function
    output[i * out_dim + j] = value;

}

//for relu, leakyRelu, linear, we dont need z to compute da/dz, instead we could a to compute.
__device__ double device_dActivate(double x) {
    return 1.0;
}

__global__ void ComputeDeltaLastLayerKernel(
    const double* a,       // batch x out_dim, a^L
    const double* y,       // batch x out_dim
    double* delta,         // batch x out_dim 输出 δ^L
    int batch,
    int out_dim
) {
    int j = blockIdx.x * blockDim.x + threadIdx.x; // neuron
    int i = blockIdx.y * blockDim.y + threadIdx.y; // sample

    if (i >= batch || j >= out_dim) return;

    int idx = i * out_dim + j;
    delta[idx] = (a[idx] - y[idx]) * device_dActivate(a[idx]);
}

//(BP2) δ^l = (δ^{l+1} · W^{l+1}) ⊙ σ'(z^l)
__global__ void ComputeDeltaHiddenLayerKernel(
    const double* delta_next, // batch x dim_delta_next δ^{l+1}
    const double* W_next,     // dim_delta_next x  dim_delta W^{l+1}
    const double* a,          // batch x dim_delta a^l
    double* delta,            // batch x dim_delta 输出 δ^l
    int batch,
    int dim_delta,
    int dim_delta_next
) {
    int by = blockIdx.y;
    int bx = blockIdx.x;

    int ty = threadIdx.y;
    int tx = threadIdx.x;

    // Working on C[i,j]
    int i = blockDim.y * by + ty;
    int j = blockDim.x * bx + tx;

    __shared__ double sh_A[TILE_WIDTH][TILE_WIDTH];
    __shared__ double sh_B[TILE_WIDTH][TILE_WIDTH];

    double value = 0;
    int phaseCount = (dim_delta_next + TILE_WIDTH - 1) / TILE_WIDTH;

    for (int phase = 0; phase < phaseCount; phase++)
    {
        int A_y = i;
        int A_x = phase * TILE_WIDTH + tx;
        if (A_y < batch && A_x < dim_delta_next) {
            sh_A[ty][tx] = delta_next[A_y * dim_delta_next + A_x];
        }
        else {
            sh_A[ty][tx] = 0;
        }
        int B_x = j;
        int B_y = phase * TILE_WIDTH + ty;
        if (B_y < dim_delta_next && B_x < dim_delta) {
            sh_B[tx][ty] = W_next[B_y * dim_delta + B_x];
        }
        else {
            sh_B[tx][ty] = 0;
        }

        __syncthreads();

        // Dot product
        for (int k = 0; k < TILE_WIDTH; k++) {
            value += sh_A[ty][k] * sh_B[tx][k];
        }

        __syncthreads();
    }

    if (i >= batch || j >= dim_delta) return;

    delta[i * dim_delta + j] = value * device_dActivate(a[i * dim_delta + j]);
}

// δ^T * a
__global__ void ComputeGradWKernel(
    const double* a_prev,   // batch x dim_delta_prev
    const double* delta,    // batch x dim_delta
    double* grad_w,         // dim_delta x dim_delta_prev
    int batch,
    int dim_delta_prev,
    int dim_delta
) {
    int by = blockIdx.y;
    int bx = blockIdx.x;

    int ty = threadIdx.y;
    int tx = threadIdx.x;

    // Working on C[i,j]
    int i = blockDim.y * by + ty;
    int j = blockDim.x * bx + tx;

    __shared__ double sh_A[TILE_WIDTH][TILE_WIDTH];
    __shared__ double sh_B[TILE_WIDTH][TILE_WIDTH];

    double value = 0;
    int phaseCount = (batch + TILE_WIDTH - 1) / TILE_WIDTH;

    for (int phase = 0; phase < phaseCount; phase++)
    {
        int A_y = i;
        int A_x = phase * TILE_WIDTH + tx;
        if (A_y < dim_delta && A_x < batch) {
            //δ^ T[A_y,A_x] = δ[A_x, A_y]
            sh_A[ty][tx] = delta[A_x * dim_delta + A_y];
        }
        else {
            sh_A[ty][tx] = 0;
        }
        int B_x = j;
        int B_y = phase * TILE_WIDTH + ty;
        if (B_y < batch && B_x < dim_delta_prev) {
            sh_B[tx][ty] = a_prev[B_y * dim_delta_prev + B_x];
        }
        else {
            sh_B[tx][ty] = 0;
        }

        __syncthreads();

        // Dot product
        for (int k = 0; k < TILE_WIDTH; k++) {
            value += sh_A[ty][k] * sh_B[tx][k];
        }

        __syncthreads();
    }

    if (i >= dim_delta || j >= dim_delta_prev) return;

    grad_w[i * dim_delta_prev + j] = value / batch;
}

//
__global__ void ComputeGradBKernel(
    const double* delta,  // batch x dim_delta
    double* grad_b,       // outdim_delta_dim
    int batch,
    int dim_delta
) {
    int j = blockIdx.x * blockDim.x + threadIdx.x; // neuron
    if (j >= dim_delta) return;

    double sum = 0.0;
    for (int i = 0; i < batch; ++i) {
        sum += delta[i * dim_delta + j];
    }
    grad_b[j] = sum / batch;
}

__global__ void ApplyGradientKernel(
    const double* grad_w, // dim_y x dim_x
    const double* grad_b, // dim_y
    double* w,
    double* b,
    int dim_y,
    int dim_x,
    double learning_rate
) {
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    int j = blockDim.y * blockIdx.y + threadIdx.y;
    if (i >= dim_y || j >= dim_x) return;

    w[i * dim_y + j] -= learning_rate * grad_w[i * dim_y + j];
    if (j == 0) {
        b[i] -= learning_rate * grad_b[i];
    }
}

void CuNN::AllocDeviceMemory() {
    // 1️⃣ 计算总大小：weights + grad_w + bias + grad_b
    size_t total = 0;
    for (auto& l : layers) {
        size_t w = l.weights.numel();
        size_t b = l.b.numel();
        total += w;  // weights
        total += w;  // grad_w
        total += b;  // bias
        total += b;  // grad_b
    }

    // 2️⃣ 一次性分配
    CUDA_CHECK(cudaMalloc(&deviceMemory, total * sizeof(double)));

    char* addr = static_cast<char*>(deviceMemory);
    deviceLayers.resize(layers.size());

    // 3️⃣ 遍历每层，分配指针并拷贝 host 数据
    for (int i = 0; i < layers.size(); ++i) {
        auto& l = layers[i];

        // -------- weights --------
        Tensor w = l.weights.contiguous();
        size_t w_size = w.numel();
        deviceLayers[i].weights = reinterpret_cast<double*>(addr);
        deviceLayers[i].w_size = w_size;
        deviceLayers[i].in_dim = l.weights.shape[1];

        CUDA_CHECK(cudaMemcpy(
            deviceLayers[i].weights,
            w.data(),
            w_size * sizeof(double),
            cudaMemcpyHostToDevice
        ));
        addr += w_size * sizeof(double);

        // -------- bias --------
        Tensor b = l.b.contiguous();
        size_t b_size = b.numel();
        deviceLayers[i].bias = reinterpret_cast<double*>(addr);
        deviceLayers[i].b_size = b_size;

        CUDA_CHECK(cudaMemcpy(
            deviceLayers[i].bias,
            b.data(),
            b_size * sizeof(double),
            cudaMemcpyHostToDevice
        ));
        addr += b_size * sizeof(double);

        // -------- grad_w --------
        deviceLayers[i].grad_w = reinterpret_cast<double*>(addr);
        CUDA_CHECK(cudaMemset(deviceLayers[i].grad_w, 0, w_size * sizeof(double)));
        addr += w_size * sizeof(double);

      
        // -------- grad_b --------
        deviceLayers[i].grad_b = reinterpret_cast<double*>(addr);
        CUDA_CHECK(cudaMemset(deviceLayers[i].grad_b, 0, b_size * sizeof(double)));
        addr += b_size * sizeof(double);
    }
}

void CuNN::AllocWorkSpaceIfNeeded() {
    // 1️⃣ 计算总大小(单位：元素数)
    size_t total = 0;

    // 输入 x
    int in_dim = layers.front().weights.shape[1];
    total += batchSize * in_dim;

    // Forward activations 和 Backward deltas
    for (auto& l : layers) {
        int out_dim = l.weights.shape[0];
        total += batchSize * out_dim;  // forward activations
        total += batchSize * out_dim;  // backward deltas
    }

    // 输出层 label 和 loss
    int out_dim = layers.back().weights.shape[0];
    total += batchSize * out_dim; // 标签 y
    total += batchSize;           // 每个样本 loss
    total += 1;                   // 总 loss

    size_t bytes = total * sizeof(double);

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

    // forward activations 每层
    ws.activations.clear();
    for (auto& l : layers) {
        int out_dim = l.weights.shape[0];
        ws.activations.push_back(reinterpret_cast<double*>(addr));
        addr += batchSize * out_dim * sizeof(double);
    }

    // backward deltas 每层
    ws.deltas.clear();
    for (auto& l : layers) {
        int out_dim = l.weights.shape[0];
        ws.deltas.push_back(reinterpret_cast<double*>(addr));
        addr += batchSize * out_dim * sizeof(double);
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
    batchSize = x.shape[0];
    AllocWorkSpaceIfNeeded();

    // 1️⃣ 拷贝输入 x 到 workspace
    CUDA_CHECK(cudaMemcpy(ws.x, x.data(), x.numel() * sizeof(double), cudaMemcpyHostToDevice));

    // 2️⃣ 遍历每层
    for (int l = 0; l < layers.size(); ++l) {
        double* input_ptr = (l == 0) ? ws.x : ws.activations[l - 1];
        double* output_ptr = ws.activations[l];

        int in_dim = deviceLayers[l].in_dim; // W.shape = in_dim x out_dim
        int out_dim = deviceLayers[l].w_size/ in_dim;

        dim3 block(TILE_WIDTH, TILE_WIDTH);
        dim3 grid((out_dim + block.x - 1) / block.x, (batchSize + block.y - 1) / block.y);

        //A = sigma(X * W^T)
        ForwardKernel << <grid, block >> > (
            input_ptr,
            deviceLayers[l].weights,
            deviceLayers[l].bias,
            output_ptr,
            batchSize, in_dim, out_dim
            );
    }

    
}

Tensor CuNN::ForwardAndFetch(const Tensor& x) {
    Forward(x);
    // 3️⃣ 返回最后一层 activations (host 端 Tensor)
    Tensor output(layers.back().b.numel(), batchSize);
    CUDA_CHECK(cudaMemcpy((void*)output.data(), ws.activations.back(), output.numel() * sizeof(double), cudaMemcpyDeviceToHost));

    return output;

}


void CuNN::Backward(Tensor& xs, Tensor& ys) {
    batchSize = xs.shape[0];
    // 1️⃣ 确保 workspace 足够大
    AllocWorkSpaceIfNeeded();

    // 2️⃣ 拷贝标签 ys 到 GPU, xs will be copied in Forward
    CUDA_CHECK(cudaMemcpy(ws.y, ys.data(), ys.numel() * sizeof(double), cudaMemcpyHostToDevice));

    // 3️⃣ Forward pass: batch 全部激活
    Forward(xs); // ws.activations 会被填充

    // 4️⃣ Backward pass
    // 最后一层 delta
    int L = layers.size() - 1;
    int out_dim = layers[L].b.size();
    int batch = batchSize;

    dim3 block(TILE_WIDTH, TILE_WIDTH);
    dim3 grid((batch + block.x - 1) / block.x, (out_dim + block.y - 1) / block.y);

    //(BP1) δ^L = (a^L - y) ⊙ σ'(z^L)
    ComputeDeltaLastLayerKernel << <grid, block >> > (
        ws.activations[L], // a^L
        ws.y,              // y
        ws.deltas[L],      // δ^L 输出
        batch,
        out_dim
        );

    //CUDA_CHECK(cudaDeviceSynchronize());

    //(BP2) δ^l = (δ^{l+1} · W^{l+1}) ⊙ σ'(z^l)
    for (int l = L - 1; l >= 0; --l) {
        int dim_delta = deviceLayers[l].b_size;
        int dim_delta_next = deviceLayers[l + 1].b_size;
        dim3 grid_delta((batch + block.x - 1) / block.x, (dim_delta + block.y - 1) / block.y);

        ComputeDeltaHiddenLayerKernel << <grid_delta, block >> > (
            ws.deltas[l + 1],        // δ^{l+1}
            deviceLayers[l + 1].weights,
            ws.activations[l],       // z^l (或 a^{l-1} 用于 σ'(z))
            ws.deltas[l],            // δ^l 输出
            batch,
            dim_delta,
            dim_delta_next
            );
        //CUDA_CHECK(cudaDeviceSynchronize());
    }

    //Compute grad_w and grad_b for each layer
    for (int l = 0; l < layers.size(); ++l) {
        int dim_delta_prev = (l == 0) ? xs.shape[1] : layers[l - 1].b.size();
        int dim_delta = layers[l].b.size();

        dim3 grid_grad((dim_delta_prev + block.x - 1) / block.x, (dim_delta + block.y - 1) / block.y);
        ComputeGradWKernel << <grid_grad, block >> > (
            (l == 0 ? ws.x : ws.activations[l - 1]),
            ws.deltas[l],
            deviceLayers[l].grad_w,
            batch,
            dim_delta_prev,
            dim_delta
            );

        dim3 grid_grad_b((dim_delta + block.x - 1) / block.x);
        ComputeGradBKernel << <grid_grad_b, block.x >> > (
            ws.deltas[l],
            deviceLayers[l].grad_b,
            batch,
            dim_delta
            );

        //CUDA_CHECK(cudaDeviceSynchronize());
    }
}


void CuNN::Step() {
    for (auto& layer : deviceLayers) {
        int block_y = (layer.b_size + TILE_WIDTH - 1) / TILE_WIDTH;
        int block_x = (layer.in_dim + TILE_WIDTH - 1) / TILE_WIDTH;
        dim3 grid(block_x, block_y);
        dim3 block(TILE_WIDTH, TILE_WIDTH);
        ApplyGradientKernel <<<grid, block>>>(layer.grad_w, layer.grad_b, layer.weights, layer.bias, layer.b_size, layer.in_dim, learningRate);
    }
}

void CuNN::FetchResultToCpu() {
    for (int i = 0; i < deviceLayers.size(); ++i) {
        auto& dlayer = deviceLayers[i];
        layers[i].weights.zeros(dlayer.b_size, dlayer.in_dim);
        layers[i].b.zeros(dlayer.b_size);
        CUDA_CHECK(cudaMemcpy(layers[i].weights.data(), dlayer.weights, dlayer.w_size * sizeof(double), cudaMemcpyKind::cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(layers[i].b.data(), dlayer.bias, dlayer.b_size * sizeof(double), cudaMemcpyKind::cudaMemcpyDeviceToHost));
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

void CuNN::PrintGrad() {
    std::cout << "-----------PrintGrad---------------\n";
    for (int i = 0; i < gradLayers.size(); ++i) {
        std::cout << "gradlayer:" << i << std::endl;
        auto& data = gradLayers[i].data();
        for (int j = 0; j < data.shape[0]; ++j) {
            for (int k = 0; k < data.shape[1]; ++k) {
                std::cout << "W_" << k << "," << j << "=" << data(j,k) << " ";
            }
            std::cout << std::endl;
        }
        for (int j = 0; j < gradLayers[i].b.numel(); ++j) {
            std::cout << "B_" << j << "=" << gradLayers[i].b(j) << " ";
        }
        std::cout << std::endl << std::endl;
    }
}

void CuNN::Print() {
    for (int i = 0; i < layers.size(); ++i) {
        std::cout << "layer:" << i << std::endl;
        auto& data = layers[i].data();
        for (int j = 0; j < data.shape[0]; ++j) {
            for (int k = 0; k < data.shape[1]; ++k) {
                std::cout << "W_" << k << "," << j << "=" << data(j,k) << " ";
            }
            std::cout << std::endl;
        }
        for (int j = 0; j < layers[i].b.numel(); ++j) {
            std::cout << "B_" << j << "=" << layers[i].b(j) << " ";
        }
        std::cout << std::endl << std::endl;
    }
}
