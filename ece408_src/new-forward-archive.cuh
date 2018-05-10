
#ifndef MXNET_OPERATOR_NEW_FORWARD_CUH_
#define MXNET_OPERATOR_NEW_FORWARD_CUH_

#define TILE_WIDTH 32

#include <mxnet/base.h>

namespace mxnet
{
namespace op
{


__constant__ float device_k[2400];


// forward kernel no constant memory
__global__ void forward_kernel(float *y, const float *x, const float *k,
                               const int B, const int M, const int C,
                               const int H, const int W, const int K)
{

    /*
    Modify this function to implement the forward pass described in Chapter 16.
    We have added an additional dimension to the tensors to support an entire mini-batch
    The goal here is to be correct AND fast.
    We have some nice #defs for you below to simplify indexing. Feel free to use them, or create your own.
    */

    const int H_out = H - K + 1;
    const int W_out = W - K + 1;
    int n, m, h0, w0, h_base, w_base, h, w;

    int W_grid = ceil(1.0 * W_out / TILE_WIDTH);

    n = blockIdx.x;
    m = blockIdx.y;
    h0 = threadIdx.y;
    w0 = threadIdx.x;
    h_base = (blockIdx.z / W_grid) * TILE_WIDTH;
    w_base = (blockIdx.z % W_grid) * TILE_WIDTH;
    h = h_base + h0;
    w = w_base + w0;

    float acc = 0;
    float cur_x;
    float cur_k;

    if((n < B) && (m < M) && (h < H_out) && (w < W_out)) {
       for(int c = 0; c < C; c++) {
           for(int p = 0; p < K; p++) {
               for(int q = 0; q < K; q++) {
                   cur_x = x[(n) * (C * H * W) + (c) * (H * W) + (h + p) * (W) + (w + q)];
                   cur_k = k[(m) * (C * K * K) + (c) * (K * K) + (p) * (K) + q];
                   acc += cur_x * cur_k;
               }
           }
       }
       y[(n) * (M * H_out * W_out) + (m) * (H_out * W_out) + (h) * (W_out) + w] = acc;
   }
}


// forward kernel with constant memory
__global__ void forward_kernel_const(float *y, const float *x,
                                     const int B, const int M, const int C,
                                     const int H, const int W, const int K)
{

    /*
    Modify this function to implement the forward pass described in Chapter 16.
    We have added an additional dimension to the tensors to support an entire mini-batch
    The goal here is to be correct AND fast.
    We have some nice #defs for you below to simplify indexing. Feel free to use them, or create your own.
    */

    const int H_out = H - K + 1;
    const int W_out = W - K + 1;

    // ==================================================================

    int n, m, h0, w0, h_base, w_base, h, w;

    int W_grid = ceil(1.0 * W_out / TILE_WIDTH);

    n = blockIdx.x;
    m = blockIdx.y;
    h0 = threadIdx.y;
    w0 = threadIdx.x;
    h_base = (blockIdx.z / W_grid) * TILE_WIDTH;
    w_base = (blockIdx.z % W_grid) * TILE_WIDTH;
    h = h_base + h0;
    w = w_base + w0;

    float acc = 0;
    float cur_x;
    float cur_k;

    if((n < B) && (m < M) && (h < H_out) && (w < W_out)) {
       for(int c = 0; c < C; c++) {
           for(int p = 0; p < K; p++) {
               for(int q = 0; q < K; q++) {
                   cur_x = x[(n) * (C * H * W) + (c) * (H * W) + (h + p) * (W) + (w + q)];
                   cur_k = device_k[(m) * (C * K * K) + (c) * (K * K) + (p) * (K) + q];
                   acc += cur_x * cur_k;
               }
           }
       }
   y[(n) * (M * H_out * W_out) + (m) * (H_out * W_out) + (h) * (W_out) + w] = acc;
   }
}


// kernel with shared memory
__global__ void forward_kernel_share(float *y, const float *x, const float *k,
                                     const int B, const int M, const int C,
                                     const int H, const int W, const int K)
{

    /*
    Modify this function to implement the forward pass described in Chapter 16.
    We have added an additional dimension to the tensors to support an entire mini-batch
    The goal here is to be correct AND fast.
    We have some nice #defs for you below to simplify indexing. Feel free to use them, or create your own.
    */

    const int H_out = H - K + 1;
    const int W_out = W - K + 1;

    // ==================================================================

    int n, m, h0, w0, h_base, w_base, h, w;

    int W_grid = ceil(1.0 * W_out / TILE_WIDTH);
    int H_grid = ceil(1.0 * H_out / TILE_WIDTH);

    int X_tile_width = TILE_WIDTH + K - 1;

    extern __shared__ float shmem[];
    float* X_shared = &shmem[0];
    float* W_shared = &shmem[X_tile_width * X_tile_width];

    n = blockIdx.x;
    m = blockIdx.y;
    h0 = threadIdx.y;
    w0 = threadIdx.x;
    h_base = (blockIdx.z / W_grid) * TILE_WIDTH;
    w_base = (blockIdx.z % W_grid) * TILE_WIDTH;
    h = h_base + h0;
    w = w_base + w0;

    float acc = 0;

    for (int c = 0; c < C; ++c) {

        // load kernel weights into shared memroy
        for (int i = h0; i < K; i += TILE_WIDTH) {
            for (int j = w0; j < K; j += TILE_WIDTH) {
                if (m < M) {
                    W_shared[i * K + j] = k[m * (C * K * K) + c * (K * K) + i * (K) + j];
                }
            }
        }
        __syncthreads();

        // load input elements into shared memory
        for (int i = h; i < h_base + X_tile_width; i += TILE_WIDTH) {
            for (int j = w; j < w_base + X_tile_width; j += TILE_WIDTH) {
                // cur_row = i - mask_w;
                // cur_col = j - mask_w;
                // X_shared[i - h_base, j - w_base], x4d(n, c, i, j)
                if ((n < B) && (i < H) && (j < W)) {
                    X_shared[(i - h_base) * X_tile_width + (j - w_base)] = \
                        x[n * (C * H * W) + c * (H * W) + i * (W) + j];
                } else {
                    X_shared[(i - h_base) * X_tile_width + (j - w_base)] = 0.0f;
                }
            }
        }
        __syncthreads();

        // convolution
        for (int p = 0; p < K; ++p) {
            for (int q = 0; q < K; ++q) {
                if ((h0 + p < X_tile_width) && (w0 + q < X_tile_width)) {
                    // X_shared[h + p, w + q], W_shared[p, q]
                    acc = acc + X_shared[(h0 + p) * X_tile_width + (w0 + q)] * W_shared[p * K + q];
                }
            }
        }
        __syncthreads();
    }

    // y4d[n, m, h, w]
    if ((n < B) && (m < M) && (h < H_out) && (w < W_out)) {
        y[n * (M * H_out * W_out) + m * (H_out * W_out) + h * (W_out) + w] = acc;
    }
}


// kernel with shared memory and constant kernel
__global__ void forward_kernel_share_constMem(float *y, const float *x,
                                              const int B, const int M, const int C,
                                              const int H, const int W, const int K)
{

    /*
    Modify this function to implement the forward pass described in Chapter 16.
    We have added an additional dimension to the tensors to support an entire mini-batch
    The goal here is to be correct AND fast.
    We have some nice #defs for you below to simplify indexing. Feel free to use them, or create your own.
    */

    const int H_out = H - K + 1;
    const int W_out = W - K + 1;

    // ==================================================================

    int n, m, h0, w0, h_base, w_base, h, w;

    int W_grid = ceil(1.0 * W_out / TILE_WIDTH);
    int H_grid = ceil(1.0 * H_out / TILE_WIDTH);

    int X_tile_width = TILE_WIDTH + K - 1;

    extern __shared__ float shmem[];
    float* X_shared = &shmem[0];

    n = blockIdx.x;
    m = blockIdx.y;
    h0 = threadIdx.y;
    w0 = threadIdx.x;
    h_base = (blockIdx.z / W_grid) * TILE_WIDTH;
    w_base = (blockIdx.z % W_grid) * TILE_WIDTH;
    h = h_base + h0;
    w = w_base + w0;

    float acc = 0;
    float cur_k;

    for (int c = 0; c < C; ++c) {

        // load input elements into shared memory
        for (int i = h0; i < X_tile_width; i += TILE_WIDTH) {
            for (int j = w0; j < X_tile_width; j += TILE_WIDTH) {
                // cur_row = i - mask_w;
                // cur_col = j - mask_w;
                // X_shared[i - h_base, j - w_base], x4d(n, c, i, j)
                if ((n < B) && (i + h_base < H) && (j + w_base < W)) {
                    X_shared[i * X_tile_width + j] = \
                        x[n * (C * H * W) + c * (H * W) + (i + h_base) * (W) + (j + w_base)];
                } else {
                    X_shared[i * X_tile_width + j] = 0.0;
                }
            }
        }
        __syncthreads();

        // convolution
        for (int p = 0; p < K; ++p) {
            for (int q = 0; q < K; ++q) {
                cur_k = device_k[m * (C * K * K) + c * (K * K) + p * K + q];
                // X_shared[h + p, w + q], W_shared[p, q]
                acc += X_shared[(h0 + p) * X_tile_width + (w0 + q)] * cur_k;
            }
        }
        __syncthreads();
    }

    // y4d[n, m, h, w]
    if ((n < B) && (m < M) && (h < H_out) && (w < W_out)) {
        y[n * (M * H_out * W_out) + m * (H_out * W_out) + h * (W_out) + w] = acc;
    }
}


// forward kernel with constant memory
__global__ void forward_shareInput_constKernel(float *y, float *x,
                                               const int B, const int M, const int C,
                                               const int H, const int W, const int K)
{

    /*
    Modify this function to implement the forward pass described in Chapter 16.
    We have added an additional dimension to the tensors to support an entire mini-batch
    The goal here is to be correct AND fast.
    We have some nice #defs for you below to simplify indexing. Feel free to use them, or create your own.
    */

    const int H_out = H - K + 1;
    const int W_out = W - K + 1;

    // ==================================================================

    int n, m, h0, w0, h_base, w_base, h, w;

    int W_grid = ceil(1.0 * W_out / TILE_WIDTH);

    n = blockIdx.x;
    m = blockIdx.y;
    h0 = threadIdx.y;
    w0 = threadIdx.x;
    h_base = (blockIdx.z / W_grid) * TILE_WIDTH;
    w_base = (blockIdx.z % W_grid) * TILE_WIDTH;
    h = h_base + h0;
    w = w_base + w0;

    __shared__ float share_x[7800];
    int cur_i = 0;
    int x_cur_i;
    int cur_n = n * (C * H * W);
    int X_tile_width = TILE_WIDTH + K - 1;
    for (int c = 0; c < C; c++) {
        int cur_c = c * (H * W);
        int cur_share_c = c * (X_tile_width * X_tile_width);
        for (int i = h0; i < X_tile_width; i += TILE_WIDTH) {
            for (int j = w0; j < X_tile_width; j += TILE_WIDTH) {
                cur_i = cur_share_c + i * X_tile_width + j;
                if ((i + h_base < H) && (j + w_base < W)) {
                    x_cur_i = cur_c + (i + h_base) * W + (j + w_base);
                    share_x[cur_i] = x[cur_n + x_cur_i];
                } else {
                    share_x[cur_i] = 0.0;
                }
            }
        }
    }
    __syncthreads();

    float acc = 0;
    float cur_x;
    float cur_k;

    if((n < B) && (m < M) && (h < H_out) && (w < W_out)) {
       int cur_m = m * C * K * K;

       for(int c = 0; c < C; c++) {
           int cur_c = c * K * K;
           int cur_share_c = c * X_tile_width * X_tile_width;

           for(int p = 0; p < K; p++) {
               for(int q = 0; q < K; q++) {
                   cur_x = share_x[cur_share_c + (h0 + p) * (X_tile_width) + (w0 + q)];
                   cur_k = device_k[cur_m + cur_c + (p) * (K) + q];
                   acc += cur_x * cur_k;
               }
           }
       }
       y[(n) * (M * H_out * W_out) + (m) * (H_out * W_out) + (h) * (W_out) + w] = acc;
   }
}




/*
   This function is called by new-inl.h
   Any code you write should be executed by this function.
   For ECE408, we only expect the float version of the operator to be called, so here we specialize with only floats.
*/
template <>
void forward<gpu, float>(mshadow::Tensor<gpu, 4, float> &y,
                         const mshadow::Tensor<gpu, 4, float> &x,
                         const mshadow::Tensor<gpu, 4, float> &w)
{

    // Use mxnet's CHECK_EQ to do assertions.
    // Remove this assertion when you do your implementation!
    // CHECK_EQ(0, 1) << "Remove this line and replace with your implementation";

    // Extract the tensor dimensions into B,M,C,H,W,K
    // ...
    const int B = x.shape_[0];
    const int M = y.shape_[1];
    const int C = x.shape_[1];
    const int H = x.shape_[2];
    const int W = x.shape_[3];
    const int K = w.shape_[3];

    std::cout << "B: " << B << " M: " << M << " C: " << C << " H: " << H << " W: " << W << " K: " << K << std::endl;

    // define constant memory for device kernel
    std::cout << "Kernel size: " << M * C * K * K << std::endl;
    float kernelSize = M * C * K * K * sizeof(float);
    cudaMemcpyToSymbol(device_k, w.dptr_, kernelSize);

    // Set the kernel dimensions
    // dim3 gridDim(0);
    // dim3 blockDim(0);
    int W_out = W - K + 1;
    int H_out = H - K + 1;
    int W_grid = ceil(1.0 * W_out / TILE_WIDTH);
    int H_grid = ceil(1.0 * H_out / TILE_WIDTH);
    dim3 blockDim(TILE_WIDTH, TILE_WIDTH, 1);
    dim3 gridDim(B, M, W_grid * H_grid);

    // 1. no share memory
    // forward_kernel<<<gridDim, blockDim>>>(y.dptr_, x.dptr_, w.dptr_, B, M, C, H, W, K);

    // 2. no share memory, constant kernel
    // forward_kernel_const<<<gridDim, blockDim>>>(y.dptr_, x.dptr_, B, M, C, H, W, K);

    // ===================================================================================

    // share memory
    // int X_tile_width = TILE_WIDTH + K - 1;

    // 3. share memory
    // int shared_size = (X_tile_width * X_tile_width + K * K) * sizeof(float);
    // forward_kernel_share<<<gridDim, blockDim, shared_size>>>(y.dptr_, x.dptr_, w.dptr_, B, M, C, H, W, K);

    // 4. share memory as well as constant kernel
    // int shared_size = (X_tile_width * X_tile_width) * sizeof(float);
    // forward_kernel_share_constMem<<<gridDim, blockDim, shared_size>>>(y.dptr_, x.dptr_, B, M, C, H, W, K);

    // 5. load x into shared memory
    forward_shareInput_constKernel<<<gridDim, blockDim>>>(y.dptr_, x.dptr_, B, M, C, H, W, K);

    /*
    forward_kernel_share_constMem<<<gridDim, blockDim, shared_size>>>(y.dptr_, shareX, B, M, C, H, W, K);
    */

    // Use MSHADOW_CUDA_CALL to check for CUDA runtime errors.
    MSHADOW_CUDA_CALL(cudaDeviceSynchronize());

}


/*
    This tells mxnet how to do an op when it's not a float.
    This is not used in the ECE408 project
*/
template <typename gpu, typename DType>
void forward(mshadow::Tensor<gpu, 4, DType> &y, const mshadow::Tensor<gpu, 4, DType> &x, const mshadow::Tensor<gpu, 4, DType> &w)
{
    CHECK_EQ(0,1) << "Remove this line and replace it with your implementation.";
}
}
}

#endif
