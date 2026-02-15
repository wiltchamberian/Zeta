#include <cuda_runtime.h>
#include "device_launch_parameters.h"
#include "NeuralNetwork.h"

//#include <cudnn.h>
#include "test.h"
#include "cu_tool.h"
#include "cnn_test.h"

int main()
{
    CudaInit();

    CudaGetDeviceProps();

    //test_cnn_linear();
    test_cnn_conv();


    return 0;
}