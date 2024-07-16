#include <mma.h>
#include <algorithm>
#include <cassert>
#include <cstdio>
#include <cstdlib>
#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <mma.h>
#include "cuda_fp16.h"
using namespace nvcuda;

template <const int BM, const int BN, const int BK>
__global__ void matmul_kernel(float* A, float* B, float* C, int M, int N, int K) {
    // Leading dimensions
    const int lda = K;
    const int ldb = N;
    const int ldc = N;

    // Shared memory for tiles
    extern __shared__ float shared_mem[];

    // Define the tiles
    float* As = shared_mem;
    float* Bs = shared_mem + BM * BK;

    // Compute block index
    int blockRow = blockIdx.y;
    int blockCol = blockIdx.x;

    // Compute thread index
    int threadRow = threadIdx.y;
    int threadCol = threadIdx.x;

    // Define fragment variables
    wmma::fragment<wmma::matrix_a, BM, BK, BK, half, wmma::row_major> a_frag;
    wmma::fragment<wmma::matrix_b, BK, BN, BK, half, wmma::row_major> b_frag;
    wmma::fragment<wmma::accumulator, BM, BN, BK, float> c_frag;

    // Initialize output to zero
    wmma::fill_fragment(c_frag, 0.0f);

    for (uint bkIdx = 0; bkIdx < K; bkIdx += BK) {
        // Populate the SMEM caches
        for (uint loadOffset = 0; loadOffset < BM; loadOffset += strideA) {
            As[(innerRowA + loadOffset) * BK + innerColA] =
                __float2half(A[(innerRowA + loadOffset) * K + innerColA]);
        }
        for (uint loadOffset = 0; loadOffset < BK; loadOffset += strideB) {
            Bs[(innerRowB + loadOffset) * BN + innerColB] =
                __float2half(B[(innerRowB + loadOffset) * N + innerColB]);
        }
        __syncthreads();

        // Load the tiles into fragments
        wmma::load_matrix_sync(a_frag, As, BK);
        wmma::load_matrix_sync(b_frag, Bs, BN);

        // Perform the matrix multiplication
        wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);

        // Advance blocktile
        A += BK;  // move BK columns to right
        B += BK * N;  // move BK rows down
        __syncthreads();
    }

    // Store the results
    for (uint resIdxM = 0; resIdxM < TM; ++resIdxM) {
        for (uint resIdxN = 0; resIdxN < TN; ++resIdxN) {
            C[(blockRow * BM + resIdxM) * N + blockCol * BN + resIdxN] =
                c_frag.x[resIdxM * TN + resIdxN];
        }
    }
}