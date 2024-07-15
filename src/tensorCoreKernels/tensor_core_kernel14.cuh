#pragma once

#include <algorithm>
#include <cassert>
#include <cstdio>
#include <cstdlib>
#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <mma.h>
#include "cuda_fp16.h"

#define CEIL_DIV(M, N) (((M) + (N)-1) / (N))
#define WMMA_M 16
#define WMMA_N 16
#define WMMA_K 16
#define WARP_SIZE 32

using namespace nvcuda;

template <const int BM, const int BN, const int BK>
__global__ void __launch_bounds__((BM * BN) / (WMMA_M * WMMA_N), 1)
    sgemmTensorCores(int M, int N, int K, float alpha, const __half *A,
                     const __half *B, float beta, float *C) {
    const uint cRow = blockIdx.y;
    const uint cCol = blockIdx.x;

    const uint totalResultsBlocktile = BM * BN;
    const uint numThreadsBlocktile = totalResultsBlocktile / (WMMA_M * WMMA_N);
    assert(numThreadsBlocktile == blockDim.x);

    const int threadCol = threadIdx.x % (BN / WMMA_N);
    const int threadRow = threadIdx.x / (BN / WMMA_N);

    __shared__ __half As[BM * BK];
    __shared__ __half Bs[BK * BN];

    const __half *A_tile = A + cRow * BM * K;
    const __half *B_tile = B + cCol * BN;
    float *C_tile = C + cRow * BM * N + cCol * BN;

    const uint innerRowA = threadIdx.x / BK;
    const uint innerColA = threadIdx.x % BK;

    const uint strideA = numThreadsBlocktile / BK;
    const uint innerRowB = threadIdx.x / BN;
    const uint innerColB = threadIdx.x % BN;
    const uint strideB = numThreadsBlocktile / BN;

    wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> acc;
    wmma::fill_fragment(acc, 0.0f);

    for (uint bkIdx = 0; bkIdx < K; bkIdx += BK) {
        for (uint loadOffset = 0; loadOffset < BM; loadOffset += strideA) {
            if ((innerRowA + loadOffset) < BM && innerColA < BK) {
                As[(innerRowA + loadOffset) * BK + innerColA] =
                    A_tile[(innerRowA + loadOffset) * K + innerColA];
            }
        }
        for (uint loadOffset = 0; loadOffset < BK; loadOffset += strideB) {
            if ((innerRowB + loadOffset) < BK && innerColB < BN) {
                Bs[(innerRowB + loadOffset) * BN + innerColB] =
                    B_tile[(innerRowB + loadOffset) * N + innerColB];
            }
        }

        __syncthreads();

        wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, half, wmma::row_major> a_frag;
        wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, half, wmma::col_major> b_frag;

        wmma::load_matrix_sync(a_frag, As, BK);
        wmma::load_matrix_sync(b_frag, Bs, BN);

        wmma::mma_sync(acc, a_frag, b_frag, acc);
        __syncthreads();

        A_tile += BK;
        B_tile += BK * N;
    }

    for (uint resIdxM = 0; resIdxM < WMMA_M; ++resIdxM) {
        for (uint resIdxN = 0; resIdxN < WMMA_N; ++resIdxN) {
            uint row = threadRow * WMMA_M + resIdxM;
            uint col = threadCol * WMMA_N + resIdxN;
            if (row < BM && col < BN && (cRow * BM + row) < M && (cCol * BN + col) < N) {
                C_tile[row * N + col] =
                    alpha * acc.x[resIdxM * WMMA_N + resIdxN] +
                    beta * C_tile[row * N + col];
            }
        }
    }
}
