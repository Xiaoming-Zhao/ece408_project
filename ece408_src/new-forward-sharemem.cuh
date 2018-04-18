
#ifndef MXNET_OPERATOR_NEW_FORWARD_CUH_
#define MXNET_OPERATOR_NEW_FORWARD_CUH_

#define TILE_WIDTH 16

#include <mxnet/base.h>

namespace mxnet
{
namespace op
{


__global__ void forward_kernel(float *y, const float *x,
                               const float *k, const int B, const int M, const int C,
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
    // (void)H_out; // silence declared but never referenced warning. remove this line when you start working
    // (void)W_out; // silence declared but never referenced warning. remove this line when you start working

// An example use of these macros:
// float a = y4d(0,0,0,0)
// y4d(0,0,0,0) = a
// #define y4d(i3, i2, i1, i0) y[(i3) * (M * H_out * W_out) + (i2) * (H_out * W_out) + (i1) * (W_out) + i0]
// #define x4d(i3, i2, i1, i0) x[(i3) * (C * H * W) + (i2) * (H * W) + (i1) * (W) + i0]
// #define k4d(i3, i2, i1, i0) k[(i3) * (C * K * K) + (i2) * (K * K) + (i1) * (K) + i0]

    // ==================================================================

    int n, m, h0, w0, h_base, w_base, h, w;

    int W_grid = ceil(1.0 * W_out / TILE_WIDTH);
    int H_grid = ceil(1.0 * H_out / TILE_WIDTH);

    int X_tile_width = TILE_WIDTH + K - 1;
    int mask_r = floor(K * 1.0 / 2);

    extern __shared__ float shmem[];
    float* X_shared = &shmem[0];
    float* W_shared = &shmem[X_tile_width * X_tile_width];

    n = blockIdx.x;
    m = blockIdx.y;
    h0 = threadIdx.y;
    w0 = threadIdx.x;
    h_base = floor(1.0 * blockIdx.z / W_grid) * TILE_WIDTH;
    w_base = (blockIdx.z % W_grid) * TILE_WIDTH;
    h = h_base + h0;
    w = w_base + w0;

    float acc = 0;

    for (int c = 0; c < C; ++c) {

        // load kernel weights into shared memroy
        if ((h0 < K) && (w0 < K)) {
            // W_shared[h0, w0], k4d(m, c, h0, w0);
            W_shared[h0 * K + w0] = k[m * (C * K * K) + c * (K * K) + h0 * (K) + w0];
        }
        __syncthreads();

        // load input elements into shared memory
        for (int i = h; i < h_base + X_tile_width; i += TILE_WIDTH) {
            for (int j = w; j < w_base + X_tile_width; j += TILE_WIDTH) {
                // X_shared[i - h_base, j - w_base], x4d(n, c, i, j)
                if ((n < B) && (m < M) && (i < H) && (j < W)) {
                    X_shared[(i - h_base) * X_tile_width + j - w_base] = x[n * (C * H * W) + c * (H * W) + i * (W) + j];
                } else {
                    X_shared[(i - h_base) * X_tile_width + j - w_base] = 0.0f;
                }
            }
        }
        __syncthreads();

        // convolution
        for (int p = 0; p < K; ++p) {
            for (int q = 0; q < K; ++q) {
                if ((h + p) < X_tile_width && (w + q) < X_tile_width) {
                    // X_shared[h + p, w + q], W_shared[p, q]
                    acc = acc + X_shared[(h + p) * X_tile_width + (w + q)] * W_shared[p * K + q];
                }
            }
        }
        __syncthreads();
    }

    // y4d[n, m, h, w]
    if ((n < B) && (m < M) && (h < H_out) && (w < W_out)) {
        y[n * (M * H_out * W_out) + m * (H_out * W_out) + h * (W_out) + w] = acc;
    }

// #undef y4d
// #undef x4d
// #undef k4d
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

    // Set the kernel dimensions
    // dim3 gridDim(0);
    // dim3 blockDim(0);
    int W_out = W - K + 1;
    int H_out = H - K + 1;
    int W_grid = ceil(W_out / TILE_WIDTH);
    int H_grid = ceil(H_out / TILE_WIDTH);
    dim3 blockDim(TILE_WIDTH, TILE_WIDTH, 1);
    dim3 gridDim(B, M, W_grid * H_grid);

    // Call the kernel
    // forward_kernel<<<gridDim, blockDim, 0, s>>>(y.dptr_, x.dptr_, w.dptr_,
    //                                             B, M, C, H, W, K);
    int X_tile_width = TILE_WIDTH + K - 1;
    int shared_size = (X_tile_width * X_tile_width + K * K) * sizeof(float);
    forward_kernel<<<gridDim, blockDim, shared_size>>>(y.dptr_, x.dptr_, w.dptr_,
                                                       B, M, C, H, W, K);

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
