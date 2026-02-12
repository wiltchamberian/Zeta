#include "test.h"
#include "tensor.h"
#include "tensor_operator.h"
#include "NeuralNetwork.h"
#include "mat.cuh"
#include "cu_tool.h"
#include <cuda_runtime.h>

void test_tensor_slice_permute() {
    using T = double;

    // -----------------------------
    // 基础张量初始化
    // -----------------------------
    TensorT<T> t1(2, 3, 4); // rank3: 2x3x4

    // 填充一些可预测数据
    int val = 1;
    for (int i = 0; i < 2; ++i)
        for (int j = 0; j < 3; ++j)
            for (int k = 0; k < 4; ++k)
                t1(i, j, k) = val++;

    // -----------------------------
    // 1️⃣ 测试 slice 单维
    // -----------------------------
    {
        auto t2 = t1.slice(Slice{ 0,1 }, Slice{ 1, TensorT<T>::end }, Slice{ 0, TensorT<T>::end });
        assert(t2.shape[0] == 1 && t2.shape[1] == 2 && t2.shape[2] == 4);
        assert(t2.numel() == 1 * 2 * 4);

        // 检查数据值
        for (int i = 0; i < t2.shape[0]; ++i)
            for (int j = 0; j < t2.shape[1]; ++j)
                for (int k = 0; k < t2.shape[2]; ++k)
                    assert(t2(i, j, k) == t1(i, j + 1, k));
    }

    // -----------------------------
    // 2️⃣ 测试 slice 多维
    // -----------------------------
    {
        auto t3 = t1.slice(Slice{ 0,2 }, Slice{ 0,2 }, Slice{ 1,4,2 }); // step测试
        assert(t3.shape[0] == 2 && t3.shape[1] == 2 && t3.shape[2] == 2);
        assert(t3.numel() == 2 * 2 * 2);

        // 检查数据值
        for (int i = 0; i < t3.shape[0]; ++i)
            for (int j = 0; j < t3.shape[1]; ++j)
                for (int k = 0; k < t3.shape[2]; ++k)
                    assert(t3(i, j, k) == t1(i, j, k * 2 + 1));
    }

    // -----------------------------
    // 3️⃣ 测试 permute
    // -----------------------------
    {
        auto t4 = t1.permute({ 2,0,1 }); // 原 shape: 2x3x4 -> 4x2x3
        assert(t4.shape[0] == 4 && t4.shape[1] == 2 && t4.shape[2] == 3);
        assert(t4.numel() == 24);

        // 检查 stride正确性
        for (int i = 0; i < 2; i++)
            for (int j = 0; j < 3; j++)
                for (int k = 0; k < 4; k++) {
                    assert(t1(i, j, k) == t4(k, i, j));
                }
    }

    // -----------------------------
    // 4️⃣ slice + permute 交替使用
    // -----------------------------
    {
        auto t5 = t1.slice(Slice{ 0,2 }, Slice{ 1,3 }, Slice{ 0,4,2 }).permute({ 2,0,1 });
        // slice shape: 2x2x2 -> permute {2,0,1} -> 2x2x2
        assert(t5.shape[0] == 2 && t5.shape[1] == 2 && t5.shape[2] == 2);
        assert(t5.numel() == 8);

        // 检查数据值
        for (int i = 0; i < t5.shape[0]; ++i)
            for (int j = 0; j < t5.shape[1]; ++j)
                for (int k = 0; k < t5.shape[2]; ++k)
                    assert(t5(i, j, k) == t1(j, k + 1, i * 2));
    }

    // -----------------------------
    // 5️⃣ 多层 slice
    // -----------------------------
    {
        auto t6 = t1.slice(Slice{ 1,2 }, Slice{ 0,3 }, Slice{ 1,3 });
        assert(t6.shape[0] == 1 && t6.shape[1] == 3 && t6.shape[2] == 2);
        assert(t6.numel() == 1 * 3 * 2);
    }

    // -----------------------------
    // 6️⃣ slice 全部维度
    // -----------------------------
    {
        auto t7 = t1.slice(Slice{}, Slice{}, Slice{});
        assert(t7.shape[0] == 2 && t7.shape[1] == 3 && t7.shape[2] == 4);
        assert(t7.numel() == 24);
    }

    // -----------------------------
    // 7️⃣ 连续 slice 测试
    // -----------------------------
    {
        // t1: shape 2x3x4
        // 第一次 slice
        auto t8_1 = t1.slice(Slice{ 0,2 }, Slice{ 0,3 }, Slice{ 0,4 }); // shape: 2x3x4
        // 第二次 slice 在 t8_1 上
        auto t8_2 = t8_1.slice(Slice{ 0,1 }, Slice{ 1,3 }, Slice{ 1,4 }); // shape: 1x2x3

        // 检查 shape
        assert(t8_2.shape[0] == 1 && t8_2.shape[1] == 2 && t8_2.shape[2] == 3);

        // 检查元素总数
        assert(t8_2.numel() == 1 * 2 * 3);

        // 检查数据值正确性
        for (int i = 0; i < t8_2.shape[0]; ++i)
            for (int j = 0; j < t8_2.shape[1]; ++j)
                for (int k = 0; k < t8_2.shape[2]; ++k)
                    assert(t8_2(i, j, k) == t1(i, j + 1, k + 1));
    }

    TensorT<T> A1(2, 2);
    TensorT<T> B1(2, 2);  

    // 填充数据
    for (int i = 0; i < 2; i++)
        for (int j = 0; j < 2; j++)
            A1(i, j) = i + j;

    for (int i = 0; i < 2; i++)
        for (int j = 0; j < 2; j++)
            B1(i, j) = i + j;

    // --------------------
    // 2️⃣ 普通 matmul
    // --------------------
    TensorT<T> C1 = A1.matmul(B1);
    assert((C1.shape[0] == 2) && (C1.shape[1] == 2) && C1.numel() == 4);
    std::cout << C1(0, 0) <<"," << C1(0, 1)<<"," << C1(1, 0) <<"," << C1(1, 1) << std::endl;



    TensorT<int> A(3, 4); // 3x4
    TensorT<int> B(4, 5); // 4x5

    // 填充数据：A(i,j) = i*10 + j, B(i,j) = i*10 + j
    for (int i = 0; i < 3; i++)
        for (int j = 0; j < 4; j++)
            A(i, j) = i * 10 + j;

    for (int i = 0; i < 4; i++)
        for (int j = 0; j < 5; j++)
            B(i, j) = i * 10 + j;

    // --------------------
    // 2️⃣ 矩阵乘法
    // --------------------
    TensorT<int> C = A.matmul(B); // shape 3x5
    assert(C.shape[0] == 3 && C.shape[1] == 5 && C.numel() == 15);

    // --------------------
    // 3️⃣ 打印输出
    // --------------------
    std::cout << "Result of A.matmul(B):\n";
    for (int i = 0; i < C.shape[0]; i++) {
        for (int j = 0; j < C.shape[1]; j++) {
            std::cout << C(i, j) << "\t";
        }
        std::cout << "\n";
    }

    // --------------------
    // 4️⃣ 验证部分值手工计算
    // C(0,0) = A row0 dot B col0 = 0*0 + 1*10 + 2*20 + 3*30 = 140
    // C(2,4) = A row2 dot B col4 = 20*4 + 21*14 + 22*24 + 23*34 = ?
    // --------------------
    int expected_00 = 0 * 0 + 1 * 10 + 2 * 20 + 3 * 30; // 0 + 10 + 40 + 90 = 140
    int expected_24 = 20 * 4 + 21 * 14 + 22 * 24 + 23 * 34; // 80 + 294 + 528 + 782 = 1684

    assert(C(0, 0) == expected_00);
    assert(C(2, 4) == expected_24);
    

    std::cout << "All Tensor slice/permutation tests passed!\n";



    /*-----------------------final test--------------------------*/
    NeuralNetwork network(0.1);
    std::shared_ptr<Activation> activation = std::make_shared<LinearActivation>();
    network.SetActivation(activation);

    // 第一层 2 -> 2
    Layer layer1(2, 2);
    layer1.weights(0, 0) = 0.1;
    layer1.weights(0, 1) = 0.2;
    layer1.weights(1, 0) = 0.3;
    layer1.weights(1, 1) = 0.4;
    layer1.b(0) = 0.5;
    layer1.b(1) = 0.6;
    network.AddLayer(layer1);

    // 第二层 2 -> 1
    Layer layer2(2, 1);
    layer2.weights(0, 0) = 0.7;
    layer2.weights(0, 1) = 0.8;
    layer2.b(0) = 0.9;
    network.AddLayer(layer2);

    network.Print();

    // --- 样本输入，与 PyTorch 一致 ---
    Sample x = { 1.0, 2.0 };
    Sample y = { 1.0 };


    std::vector<Sample> xs = { x };
    std::vector<Sample> ys = { y };

    auto y_pred = network.Forward(x);
    double loss = network.MseLoss(std::vector<Sample>{y_pred}, ys);

    network.Backward(xs, ys);

    network.PrintGrad();

    network.Step();
    network.Print();

    std::cout << "Final tests passed!\n";
}

void test_mat_mul() {
 

    Tensor a1(2, 2);
    Tensor a2(2, 2);
    a1(0, 0) = 1;
    a1(0, 1) = 2;
    a1(1, 0) = 3;
    a1(1, 1) = 4;
    a2 = a1;
    Tensor c(2, 2);
    double* addr1;
    int l1 = ToDevice(a1, (void**) &addr1);
    double* addr2;
    int l2 = ToDevice(a2, (void**)&addr2);
    double* addr3;
    int l3 = ToDevice(c, (void**)&addr3);
    dim3 grid(1, 1);
    dim3 block(TILE_DIM, TILE_DIM);
    tiled_mat_mul_kernel_ex<<< grid,block >>>(addr1, addr2, addr3, 2, 2, 2);
    //mat_add_kernel << <grid, block >> > (addr1, addr2, addr3, 2, 2);

    CudaGetLastError();
    //CudaDeviceSynchronize();

    ToTensor(c, addr3);

}