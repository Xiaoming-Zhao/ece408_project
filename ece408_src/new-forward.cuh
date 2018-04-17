
#ifndef MXNET_OPERATOR_NEW_FORWARD_CUH_
#define MXNET_OPERATOR_NEW_FORWARD_CUH_

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
#define y4d(i3, i2, i1, i0) y[(i3) * (M * H_out * W_out) + (i2) * (H_out * W_out) + (i1) * (W_out) + i0]
#define x4d(i3, i2, i1, i0) x[(i3) * (C * H * W) + (i2) * (H * W) + (i1) * (W) + i0]
#define k4d(i3, i2, i1, i0) k[(i3) * (C * K * K) + (i2) * (K * K) + (i1) * (K) + i0]

    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int tz = threadIdx.z;
    int x_o = blockIdx.x * TILE_WIDTH + tx;
    int y_o = blockIdx.y * TILE_WIDTH + ty;
    int z_o = blockIdx.z * TILE_WIDTH + tz;
    int x_i = x_o - MASK_RADIUS;
    int y_i = y_o - MASK_RADIUS;
    int z_i = z_o - MASK_RADIUS;

    // Load kernel data into shared memory
    __shared__ float N_ds[C][K][K];

    if ((z_i >= 0) && (z_i < z_size) &&
    	(y_i >= 0) && (y_i < y_size) &&
    	(x_i >= 0) && (x_i < x_size)) {
    	N_ds[tz][ty][tx] = input[z_i * (y_size * x_size) + y_i * x_size + x_i];
    } else {
    	N_ds[tz][ty][tx] = 0.0f;
    }

    __syncthreads();

    // Compute convolution
    float output_val = 0.0f;

    if (tz < TILE_WIDTH && ty < TILE_WIDTH && tx < TILE_WIDTH) {
    	for (int i = 0; i < MASK_WIDTH; ++i) {
    		for (int j = 0; j < MASK_WIDTH; ++j) {
    			for (int u = 0; u < MASK_WIDTH; ++u) {
    				output_val += deviceKernel[i][j][u] * N_ds[tz + i][ty + j][tx + u];
    			}
    		}
    	}

      if (z_o < z_size && y_o < y_size && x_o < x_size) {
      	output[z_o * (y_size * x_size) + y_o * x_size + x_o] = output_val;
      }
    }

    // ==================================================================

    for (int b = 0; b < B; ++b) {
        for (int m = 0; m < M; ++m) {

        }
    }

#undef y4d
#undef x4d
#undef k4d
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

    // Set the kernel dimensions
    // dim3 gridDim(0);
    // dim3 blockDim(0);

    // Call the kernel
    // forward_kernel<<<gridDim, blockDim, 0, s>>>(y.dptr_,x.dptr_,w.dptr_, B,M,C,H,W,K);

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
