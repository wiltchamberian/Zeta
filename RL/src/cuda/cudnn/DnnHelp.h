#pragma once
#include <cassert>
#include "fp16_dev.h"
#include "fp16_emu.h"

// Generate uniform numbers [0,1)
void initImage(float* image, int imageSize);

void initImage(half1* image, int imageSize);

// Currently set to generate uniform integers [-2, 2] to avoid int8 overflow
void initImage(int8_t* image, int imageSize);


void initImagePadded(int8_t* image, int dimA[], int dimPadded[], int stridePadded[], cudnnDataType_t dataType);

int checkCudaError(cudaError_t code, const char* expr, const char* file, int line);

int checkCudnnError(cudnnStatus_t code, const char* expr, const char* file, int line);

void generateStrides(const int* dimA, int* strideA, int nbDims, cudnnTensorFormat_t filterFormat);

#define checkCudaErr(...)                                                        \
    do {                                                                         \
        int err = checkCudaError(__VA_ARGS__, #__VA_ARGS__, __FILE__, __LINE__); \
        if (err) {                                                               \
            numErrors++;                                                         \
            goto clean;                                                          \
        }                                                                        \
    } while (0)

#define checkCudnnErr(...)                                                        \
    do {                                                                          \
        int err = checkCudnnError(__VA_ARGS__, #__VA_ARGS__, __FILE__, __LINE__); \
        if (err) {                                                                \
            numErrors++;                                                          \
            goto clean;                                                           \
        }                                                                         \
    } while (0)

#define DNN_CHECK(x) \
    do { \
        cudnnStatus_t status = (x); \
        if (status != CUDNN_STATUS_SUCCESS) { \
            int err = checkCudnnError(status, "", __FILE__, __LINE__); \
            assert(false);\
        } \
    } while (0)
