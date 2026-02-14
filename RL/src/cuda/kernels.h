#ifndef __MY_KERNELS__
#define __MY_KERNELS__


#include "device_launch_parameters.h"

#define TILE_WIDTH 16

__global__ void linear_forward_kernel(
    const double* input,      // batch x in_dim
    const double* weights,    // out_dim x in_dim
    const double* bias,       // out_dim
    double* output,           // batch x out_dim
    int batch, int in_dim, int out_dim
);

__global__ void linear_relu_forward_kernel(
    const double* input,      // batch x in_dim
    const double* weights,    // out_dim x in_dim
    const double* bias,       // out_dim
    double* output,           // batch x out_dim
    int batch, int in_dim, int out_dim
);

__global__ void linear_leaky_relu_forward_kernel(
    const double* input,      // batch x in_dim
    const double* weights,    // out_dim x in_dim
    const double* bias,       // out_dim
    double* output,           // batch x out_dim
    int batch, int in_dim, int out_dim,
    double alpha
);

__global__ void relu_forward_kernel(
    const double* input,   // batch x dim
    double* output,        // batch x dim
    int total_elements     // batch * dim
);

__global__ void leaky_relu_forward_kernel(
    const double* input,
    double* output,
    int total_elements,
    double alpha
);

__global__ void mse_loss_kernel(
    const double* a,       // batch x out_dim, a^L
    const double* y,       // batch x out_dim
    double* delta,         // batch x out_dim 输出 δ^L
    int batch,
    int out_dim
);

__global__ void linear_backward_kernel(
    const double* delta_next, // batch x dim_delta_next δ^{l+1}
    const double* W_next,     // dim_delta_next x  dim_delta W^{l+1}
    const double* a,          // batch x dim_delta a^l
    double* delta,            // batch x dim_delta 输出 δ^l
    int batch,
    int dim_delta,
    int dim_delta_next
);

__global__ void linear_relu_backward_kernel(
    const double* delta_next, // batch x dim_delta_next δ^{l+1}
    const double* W_next,     // dim_delta_next x dim_delta W^{l+1}
    const double* a,          // batch x dim_delta a^l
    double* delta,            // batch x dim_delta 输出 δ^l
    int batch,
    int dim_delta,
    int dim_delta_next
);

__global__ void linear_leaky_relu_backward_kernel(
    const double* delta_next, // batch x dim_delta_next δ^{l+1}
    const double* W_next,     // dim_delta_next x dim_delta W^{l+1}
    const double* a,          // batch x dim_delta a^l
    double* delta,            // batch x dim_delta 输出 δ^l
    int batch,
    int dim_delta,
    int dim_delta_next,
    double alpha              // LeakyReLU alpha
);

__global__ void compute_grad_w_kernel(
    const double* a_prev,   // batch x dim_delta_prev
    const double* delta,    // batch x dim_delta
    double* grad_w,         // dim_delta x dim_delta_prev
    int batch,
    int dim_delta_prev,
    int dim_delta
);

__global__ void compute_grad_b_kernel(
    const double* delta,  // batch x dim_delta
    double* grad_b,       // outdim_delta_dim
    int batch,
    int dim_delta
);

__global__ void apply_gradien_kernel(
    const double* grad_w, // dim_y x dim_x
    const double* grad_b, // dim_y
    double* w,
    double* b,
    int dim_y,
    int dim_x,
    double learning_rate
);

__global__ void conv_forward_kernel(
    const double* input,      // N * C * H * W
    const double* weights,    // K * (CRS)
    const double* bias,       // K
    double* output,           // N * K * P * Q
    int batch, int C, int H, int W, int R, int S,
    int strideH, int strideW, int K, int P, int Q, float alpha //P,Q are not independent variables
);

//by default use leaky relu, since leaky relu include relu
__global__ void conv_dgrad_kernel(
    const double* delta_next, // NKHW ->  KRS * NHW δ^{l+1} 
    const double* W_next,     // KCRS  -> C*KRS W^{l+1}
    const double* a,          // NCHW -> C*NHW
    double* delta,            // NCHW -> C*NHW   output: δ^l
    int N,
    int C,
    int H, int W, int R, int S, int strideH, int strideW, int K, int alpha
);

__global__ void conv_bgrad_kernel(
    const double* delta,    //NCHW -> 
    double* grad_b,         //C
    int N, int C, int H, int W
);

__global__ void conv_wgrad_kernel(
    const double* delta,
    const double* a_prev,
    double* grad_w,
    int N, int K, int C, int P,int Q, int R, int S
);



#endif
