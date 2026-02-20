#include <cuda_runtime.h>
#include "device_launch_parameters.h"
#include "NeuralNetwork.h"

//#include <cudnn.h>
#include "test.h"
#include "cu_tool.h"
#include "cnn_test.h"
#include "TicTac.h"

int main()
{
    CudaInit();

    CudaGetDeviceProps();

    //test_cnn_linear();
    test_cnn_conv();

    std::unique_ptr<TicTacNNProxy> proxy = std::make_unique<TicTacNNProxy>();
    proxy->createNetwork();
    TicTacMcts mcts;
    mcts.proxy = proxy.get();
    mcts.train();

    return 0;
}