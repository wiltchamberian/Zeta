#include "DnnHelp.h"
#include <stdio.h>



#define SWITCH_CHAR '-'
#define THRESHOLD 2.0e-2

namespace zeta {
    // Generate uniform numbers [0,1)
    void initImage(float* image, int imageSize) {
        static unsigned seed = 123456789;
        for (int index = 0; index < imageSize; index++) {
            seed = (1103515245 * seed + 12345) & 0xffffffff;
            image[index] = float(seed) * 2.3283064e-10;  // 2^-32
        }
    }

    // Currently set to generate uniform integers [-2, 2] to avoid int8 overflow
    void initImage(int8_t* image, int imageSize) {
        static unsigned seed = 123456789;
        for (int index = 0; index < imageSize; index++) {
            seed = (1103515245 * seed + 12345) & 0xffffffff;
            // Takes floats from [0, 1), scales and casts to ints from [0, 4], then
            // subtracts from 2
            image[index] = 2 - (int8_t)(5 * float(seed) * 2.3283064e-10);  // 2^-32
        }
    }

    void initImagePadded(int8_t* image, int dimA[], int dimPadded[], int stridePadded[], cudnnDataType_t dataType) {
        static unsigned seed = 123456789;
        int resizeFactor = (dataType == CUDNN_DATA_INT8x4) ? 4 : 32;
        int totalSize = dimPadded[0] * dimPadded[1] * dimPadded[2] * dimPadded[3];

        // #pragma omp parallel for
        for (int i = 0; i < totalSize; i++) {
            int n = (i / stridePadded[0]) % dimPadded[0];
            int c1 = (i / (stridePadded[1] * resizeFactor)) % (dimPadded[1] / resizeFactor);
            int c2 = i % resizeFactor;
            int c = c1 * resizeFactor + c2;
            if (n < dimA[0] && c < dimA[1]) {
                seed = (1103515245 * seed + 12345) & 0xffffffff;
                image[i] = 2 - (int8_t)(5 * float(seed) * 2.3283064e-10);  // 2^-32
            }
            else {
                image[i] = 0;
            }
        }
    }

    int checkCudaError(cudaError_t code, const char* expr, const char* file, int line) {
        if (code) {
            printf("CUDA error at %s:%d, code=%d (%s) in '%s'", file, line, (int)code, cudaGetErrorString(code), expr);
            return 1;
        }
        return 0;
    }

    int checkCudnnError(cudnnStatus_t code, const char* expr, const char* file, int line) {
        if (code) {
            printf("CUDNN error at %s:%d, code=%d (%s) in '%s'\n", file, line, (int)code, cudnnGetErrorString(code), expr);
            return 1;
        }
        return 0;
    }

    void generateStrides(const int* dimA, int* strideA, int nbDims, cudnnTensorFormat_t filterFormat) {
        // For INT8x4 and INT8x32 we still compute standard strides here to input
        // into the cuDNN functions. We will manually scale by resizeFactor in the cpu
        // ref.
        if (filterFormat == CUDNN_TENSOR_NCHW || filterFormat == CUDNN_TENSOR_NCHW_VECT_C) {
            strideA[nbDims - 1] = 1;
            for (int d = nbDims - 2; d >= 0; d--) {
                strideA[d] = strideA[d + 1] * dimA[d + 1];
            }
        }
        else {
            // Here we assume that the format is CUDNN_TENSOR_NHWC
            strideA[1] = 1;
            strideA[nbDims - 1] = strideA[1] * dimA[1];
            for (int d = nbDims - 2; d >= 2; d--) {
                strideA[d] = strideA[d + 1] * dimA[d + 1];
            }
            strideA[0] = strideA[2] * dimA[2];
        }
    }
}