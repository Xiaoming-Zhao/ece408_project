
#ifndef MXNET_OPERATOR_NEW_FORWARD_CUH_
#define MXNET_OPERATOR_NEW_FORWARD_CUH_

#define TILE_WIDTH 16
#define CUDA_MAX_NUM_THREADS 1024

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

// #undef y4d
// #undef x4d
// #undef k4d
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

// #undef y4d
// #undef x4d
// #undef k4d
}


// unroll matrix kernel
__global__ void unroll_kernel(int b, int C, int H, int W, int K, float *X, float *X_unroll)
{
    int c, s, h_out, w_out, h_unroll, w_unroll, h_base, p, q;
    int t = blockIdx.x * CUDA_MAX_NUM_THREADS + threadIdx.x;
    int H_out = H - K + 1;
    int W_out = W - K + 1;
    int W_unroll = H_out * W_out;

    if (t < C * W_unroll) {
        c = int(t / W_unroll);
        s = t % W_unroll;
        h_out = s / W_out;
        w_out = s % W_out;
        w_unroll = h_out * W_out + w_out;
        h_base = c * K * K;

        for (p = 0; p < K; p++) {
            for (q = 0; q < K; q++) {
                h_unroll = h_base + p * K + q;
                X_unroll[h_unroll * W_unroll + w_unroll] = \
                    X[b * (C * H * W) + c * (H * W) + (h_out + p) * W + (w_out + q)];
            }
        }
    }
}


// No tile: Compute C = A * B
__global__ void matrixMultiply(float *A, float *B, float *C,
                               int numARows, int numAColumns,
                               int numBRows, int numBColumns,
                               int b) {
  //@@ Insert code to implement matrix multiplication here
  // The row index for A and C
  int Row = blockIdx.y * blockDim.y + threadIdx.y;
  // The column index for B and C
  int Col = blockIdx.x * blockDim.x + threadIdx.x;

  int numCRows = numARows;
  int numCColumns = numBColumns;

  if((Row < numCRows) && (Col < numCColumns)) {
  	float pValue = 0.0;

  	for(int k = 0; k < numAColumns; k++) {
  		pValue += A[Row * numAColumns + k] * B[k * numBColumns + Col];
  	}

  	C[b * (numCRows * numCColumns) + Row * numCColumns + Col] = pValue;
  }
}


// Tile memory: Compute C = A * B
__global__ void matrixMultiplyShared(float *A, float *B, float *C,
                                     int numARows, int numAColumns,
                                     int numBRows, int numBColumns,
                                     int maxNum, int b)
{
    //@@ Insert code to implement matrix multiplication here
    //@@ You have to use shared memory for this MP
    __shared__ float Ads[TILE_WIDTH][TILE_WIDTH];
    __shared__ float Bds[TILE_WIDTH][TILE_WIDTH];

    int bx = blockIdx.x;
    int by = blockIdx.y;
    int tx = threadIdx.x;
    int ty = threadIdx.y;

    int numCRows = numARows;
    int numCColumns = numBColumns;

    // define Row and Column
    int Row = by * TILE_WIDTH + ty;
    int Col = bx * TILE_WIDTH + tx;

    float pValue = 0;

    // Loop over the A and B tiles
    for (int ph = 0; ph < ceil(maxNum / (float) TILE_WIDTH); ph++) {
        // Check boundary for A
        if ((Row < numARows) && (ph * TILE_WIDTH + tx < numAColumns)) {
            Ads[ty][tx] = A[Row * numAColumns + ph * TILE_WIDTH + tx];
        } else {
            Ads[ty][tx] = 0.0;
        }

        // Check boundary for B
        if ((ph * TILE_WIDTH + ty < numBRows) && (Col < numBColumns)) {
            Bds[ty][tx] = B[(ph * TILE_WIDTH + ty) * numBColumns + Col];
        } else {
            Bds[ty][tx] = 0.0;
        }

        __syncthreads();
        for (int k = 0; k < TILE_WIDTH; ++k) {
            pValue += Ads[ty][k] * Bds[k][tx];
        }
        __syncthreads();
    }

    if ((Row < numCRows) && (Col < numCColumns)) {
        C[b * (numCRows * numCColumns) + Row * numCColumns + Col] = pValue;
    }
}


// Tile memory: Compute C = A * B
__global__ void matrixMultiplySharedConst(float *B, float *C,
                                          int numARows, int numAColumns,
                                          int numBRows, int numBColumns,
                                          int maxNum, int b)
{
    //@@ Insert code to implement matrix multiplication here
    //@@ You have to use shared memory for this MP
    __shared__ float Ads[TILE_WIDTH][TILE_WIDTH];
    __shared__ float Bds[TILE_WIDTH][TILE_WIDTH];

    int bx = blockIdx.x;
    int by = blockIdx.y;
    int tx = threadIdx.x;
    int ty = threadIdx.y;

    int numCRows = numARows;
    int numCColumns = numBColumns;

    // define Row and Column
    int Row = by * TILE_WIDTH + ty;
    int Col = bx * TILE_WIDTH + tx;

    float pValue = 0;

    // Loop over the A and B tiles
    for (int ph = 0; ph < ceil(maxNum / (float) TILE_WIDTH); ph++) {
        // Check boundary for A
        if ((Row < numARows) && (ph * TILE_WIDTH + tx < numAColumns)) {
            Ads[ty][tx] = device_k[Row * numAColumns + ph * TILE_WIDTH + tx];
        } else {
            Ads[ty][tx] = 0.0;
        }

        // Check boundary for B
        if ((ph * TILE_WIDTH + ty < numBRows) && (Col < numBColumns)) {
            Bds[ty][tx] = B[(ph * TILE_WIDTH + ty) * numBColumns + Col];
        } else {
            Bds[ty][tx] = 0.0;
        }

        __syncthreads();
        for (int k = 0; k < TILE_WIDTH; ++k) {
            pValue += Ads[ty][k] * Bds[k][tx];
        }
        __syncthreads();
    }

    if ((Row < numCRows) && (Col < numCColumns)) {
        C[b * (numCRows * numCColumns) + Row * numCColumns + Col] = pValue;
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

    // Set the kernel dimensions
    // dim3 gridDim(0);
    // dim3 blockDim(0);
    int W_out = W - K + 1;
    int H_out = H - K + 1;

    //@@ Define constant memory for device kernel here
    std::cout << "Kernel size: " << M * C * K * K << std::endl;
    float kernelSize = M * C * K * K * sizeof(float);
    cudaMemcpyToSymbol(device_k, w.dptr_, kernelSize);

    // 1. no share memory
    int W_grid = ceil(1.0 * W_out / TILE_WIDTH);
    int H_grid = ceil(1.0 * H_out / TILE_WIDTH);
    // dim3 blockDim(TILE_WIDTH, TILE_WIDTH, 1);
    // dim3 gridDim(B, M, W_grid * H_grid);
    // forward_kernel<<<gridDim, blockDim>>>(y.dptr_, x.dptr_, w.dptr_, B, M, C, H, W, K);

    // 2. constant memory kernel
    // forward_kernel_const<<<gridDim, blockDim>>>(y.dptr_, x.dptr_, B, M, C, H, W, K);

    // 3. unroll
    // grid and block settings for unroll kernel
    int num_blocks_unroll = ceil((1.0 * C * H_out * W_out) / CUDA_MAX_NUM_THREADS);

    // parameters for unrolling
    int H_unroll = C * K * K;
    int W_unroll = H_out * W_out;
    float* x_unroll;
    cudaMalloc((void **) &x_unroll, W_unroll * H_unroll * sizeof(float));

    int maxNum;
    if (M > W_unroll) {
        maxNum = M;
    } else {
        maxNum = W_unroll;
    }
    if (maxNum < H_unroll) {
        maxNum = H_unroll;
    }

    // grid and block settings for tiled matrix multiplication
    // settings for launching grid
    dim3 blockDim(TILE_WIDTH, TILE_WIDTH);
    dim3 gridDim(ceil(1.0 * W_unroll / TILE_WIDTH), ceil(1.0 * M / TILE_WIDTH));
    for (int b = 0; b < B; b++) {
        // unroll each sample to expanded matrix
        unroll_kernel<<<num_blocks_unroll, CUDA_MAX_NUM_THREADS>>>(b, C, H, W, K, x.dptr_, x_unroll);

        // tiled matrix multiplication
        // matrixMultiply<<<gridDim, blockDim>>>(w.dptr_, x_unroll, y.dptr_, M, H_unroll, H_unroll, W_unroll, b);
        // matrixMultiplyShared<<<gridDim, blockDim>>>(w.dptr_, x_unroll, y.dptr_, M, H_unroll, H_unroll, W_unroll, maxNum, b);
        matrixMultiplySharedConst<<<gridDim, blockDim>>>(x_unroll, y.dptr_, M, H_unroll, H_unroll, W_unroll, maxNum, b);
    }


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
