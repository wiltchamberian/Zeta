#ifndef __MY_KERNELS__
#define __MY_KERNELS__


#include "device_launch_parameters.h"

#define TILE_WIDTH 32
#define TILE_SMALL 8

namespace zeta {
    __global__ void tanh_forward_kernel(
        const float* input,   // N * in_dim
        float* output,        // N * in_dim
        int total
    );

    __global__ void tanh_backward_kernel(
        const float* dC_da,    //N * dim     dC/da
        const float* a,        //N * dim
        float* output,         //N * dim     dC/dz
        int N
    );

    __global__ void leaky_relu_forward_kernel(
        const float* input,
        float* output,
        int total_elements,
        float alpha
    );

    __global__ void leaky_relu_backward_kernel(
        const float* dC_da,
        const float* a,
        float* dC_dz,
        int total_elements,
        float alpha
    );

    __global__ void mse_loss_kernel(
        const float* a,       // batch * dim 
        const float* y,       // batch * dim
        float* loss,         // 1
        int total
    );

    __global__ void max_pool_2d_forward_kernel(
        const float* in,
        float* out,
        int* maxIndex,
        int N,
        int h,
        int w,
        int H,
        int W
    );

    __global__ void max_pool_2d_backward_kernel(
        const float* dC_da,
        const int* maxIndex,
        float* delta,
        int N,
        int H,//d_da
        int W
    );

    __global__ void linear_forward_kernel(
        const float* input,      // batch x in_dim
        const float* weights,    // out_dim x in_dim
        const float* bias,       // out_dim
        float* output,           // batch x out_dim
        int batch, int in_dim, int out_dim
    );

    __global__ void linear_leaky_relu_forward_kernel(
        const float* input,      // batch x in_dim
        const float* weights,    // out_dim x in_dim
        const float* bias,       // out_dim
        float* output,           // batch x out_dim
        int batch, int in_dim, int out_dim,
        float alpha
    );

    __global__ void linear_tanh_forward_kernel(
        const float* input,      // batch x in_dim
        const float* weights,    // out_dim x in_dim
        const float* bias,       // out_dim
        float* output,           // batch x out_dim
        int batch, int in_dim, int out_dim
    );

    __global__ void mse_loss_backward_kernel(
        const float* a,       // batch x out_dim, a^L
        const float* y,       // batch x out_dim
        float* delta,         // batch x out_dim 输出 δ^L
        int batch,
        int out_dim
    );

    __global__ void linear_backward_kernel(
        const float* delta_next, // batch x dim_delta_next δ^{l+1}
        const float* W_next,     // dim_delta_next x  dim_delta W^{l+1}
        const float* a,          // batch x dim_delta a^l
        float* delta,            // batch x dim_delta 输出 δ^l
        int batch,
        int dim_delta,
        int dim_delta_next
    );

    __global__ void linear_leaky_relu_backward_kernel(
        const float* delta, // batch x dim_delta_next δ^{l}
        const float* W,     // dim_delta_next x dim_delta W^{l}
        const float* a_prev,          // batch x dim_delta a^(l-1)
        float* delta_prev,            // batch x dim_delta 输出 δ^(l-1)
        bool add,
        int batch,
        int dim_delta_prev,
        int dim_delta,
        float alpha              // LeakyReLU alpha
    );

    __global__ void linear_tanh_backward_kernel(
        const float* delta, // batch x dim_delta_next δ^{l}
        const float* W,     // dim_delta_next x dim_delta W^{l}
        const float* a_prev,          // batch x dim_delta a^(l-1)
        float* delta_prev,            // batch x dim_delta 输出 δ^(l-1)
        bool add,
        int batch,
        int dim_delta_prev,
        int dim_delta
    );

    __global__ void compute_grad_w_kernel(
        const float* a_prev,   // batch x dim_delta_prev
        const float* delta,    // batch x dim_delta
        float* grad_w,         // dim_delta x dim_delta_prev
        int batch,
        int dim_delta_prev,
        int dim_delta
    );

    __global__ void compute_grad_b_kernel(
        const float* delta,  // batch x dim_delta
        float* grad_b,       // outdim_delta_dim
        int batch,
        int dim_delta
    );

    //c  = alpha * a + beta * b
    __global__ void matrix_add_kernel(
        const float* a,
        const float* b,
        float* c,
        float alpha,
        float beta,
        int total
    );

    //adam ubiased second raw moment
    __global__ void adam_second_moment_kernel(
        const float* v_old,
        const float* g,
        const float beta2,
        float* v,
        int total
    );

    __global__ void adam_gradient_kernel(
        const float* g,
        float* m,
        float* v,
        const float alpha,
        const float beta1,
        const float beta2,
        const float beta1_t,
        const float beta2_t,
        const float epsilon,
        float* theta,
        int total
    );

    __global__ void apply_weights_kernel(
        const float* grad_w, // K * CPQ
        float* w,
        int K,
        int CPQ,
        float learning_rate
    );

    __global__ void apply_gradient_kernel(
        const float* grad_w, // dim_y x dim_x
        const float* grad_b, // dim_y
        float* w,
        float* b,
        int dim_y,
        int dim_x,
        float learning_rate
    );

    __global__ void cross_entropy_kernel(
        const float* p,
        const float* y,
        float* loss,
        int batch,
        int total);

    __global__ void stable_cross_entropy_kernel(
        const float* logits,
        const float* labels,
        float* Loss,
        int batch,
        int classes);

    __global__ void simple_softmax_forward_kernel(
        const float* input,
        float* output,
        int CHW
    );

    __global__ void softmax_forward_kernel(
        const float* input, // N * C * H * W
        float* output, //N * C * H * W
        int CHW
    );

    __global__ void softmax_backward_kernel(
        const float* ylabel,
        const float* ps,
        float* output,
        int N,
        int CHW
    );

    // A = A + B (B is broad casted)
    __global__ void tensor_add_bias_kernel(
        float* A, //NCHW
        const float* B,  //1C11
        int HW,
        int C,
        int NCHW
    );

    __global__ void conv_forward_kernel(
        const float* input,      // N * C * H * W
        const float* weights,    // K * (CRS)
        const float* bias,       // K
        float* output,           // N * K * P * Q
        int batch, int C, int H, int W, int R, int S,
        int strideH, int strideW, int padH, int padW, int K, int P, int Q, float alpha //P,Q are not independent variables
    );

    //by default use leaky relu, since leaky relu include relu
    __global__ void conv_dgrad_kernel(
        const float* delta_next, // NKPQ ->  KRS * NHW δ^{l+1} 
        const float* W_next,     // KCRS  -> C*KRS W^R^{l+1}
        const float* a,          // NCHW -> C*NHW
        float* delta,            // NCHW -> C*NHW   output: δ^l
        bool add,
        int N,
        int C,
        int H, int W, int P, int Q, int R, int S, int strideH, int strideW, int padH, int padW, int K, int alpha
    );

    __global__ void conv_bgrad_test(
        const float* delta,    //NCHW -> C * NHW 
        float* grad_b,
        int N, int C, int H, int W
    );

    __global__ void conv_bgrad_kernel(
        const float* delta,    //NCHW -> 
        float* grad_b,         //C
        int N, int C, int H, int W
    );

    __global__ void conv_wgrad_kernel(
        const float* delta, //NKPQ -> K * NPQ
        const float* a_prev, //NCHW -> NPQ * CRS
        float* grad_w, //KCRS -> K * CRS
        int N, int K, int C, int H, int W, int P, int Q, int R, int S, int strideH, int strideW, int padH, int padW
    );

    __global__ void regular_kernel(
        const float* weights, //NCHW
        float* grad_w,        //NCHW
        int NCHW,
        float c
    );

    //normalize sum
    __global__ void normalize_sum(
        float* inputs,
        float* sum,
        int total
    );
}


#endif
