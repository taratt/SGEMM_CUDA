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

    // Determine block index and thread index
    const uint cRow = blockIdx.y;
    const uint cCol = blockIdx.x;
    const uint totalResultsBlocktile = BM * BN;
    const uint numThreadsBlocktile = totalResultsBlocktile / (WMMA_M * WMMA_N);

    assert(numThreadsBlocktile == blockDim.x);

    // Shared memory for sub-matrices
    extern __shared__ __half shared_mem[];
    __half *As = shared_mem;
    __half *Bs = shared_mem + BM * BK;

    const __half *A_tile = A + cRow * BM * K;
    const __half *B_tile = B + cCol * BN;
    float *C_tile = C + cRow * BM * N + cCol * BN;

    // Determine the row and column for loading A and B into shared memory
    const uint rowSharedLoaderA = threadIdx.x / BK;
    const uint colSharedLoaderA = threadIdx.x % BK;
    const uint rowSharedLoaderB = threadIdx.x / BN;
    const uint colSharedLoaderB = threadIdx.x % BN;

    const uint strideA = numThreadsBlocktile / BK;
    const uint strideB = numThreadsBlocktile / BN;


    //initialize the warp-level fragments
    wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, __half, wmma::row_major> a_frag;
    wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, __half, wmma::row_major> b_frag;
    wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> acc;
    wmma::fill_fragment(acc, 0.0f);

    for (uint bkIdx = 0; bkIdx < K; bkIdx += BK) {
        for (uint loadOffset = 0; loadOffset < BM; loadOffset += strideA) {
            if ((rowSharedLoaderA + loadOffset) < BM && colSharedLoaderA < BK) {
                As[(rowSharedLoaderA + loadOffset) * BK + colSharedLoaderA] =
                    A_tile[(rowSharedLoaderA + loadOffset) * K + colSharedLoaderA];
            }
        }
        for (uint loadOffset = 0; loadOffset < BK; loadOffset += strideB) {
            if ((rowSharedLoaderB + loadOffset) < BK && colSharedLoaderB < BN) {
                Bs[(rowSharedLoaderB + loadOffset) * BN + colSharedLoaderB] =
                    B_tile[(rowSharedLoaderB + loadOffset) * N + colSharedLoaderB];
            }
        }

        __syncthreads();

        A_tile += BK;
        B_tile += BK * N;

            // Declare matrix A and B fragments

            //
            // wmma::load_matrix_sync(a_frag, As, BK);
            // wmma::load_matrix_sync(b_frag, Bs, BN);
            //
            // wmma::mma_sync(acc, a_frag, b_frag, acc);
            // __syncthreads();

        for (int i = 0; i < BK; i += WMMA_K) {
            load_matrix_sync(a_frag, As + i, BK);
            load_matrix_sync(b_frag, Bs + i * BN, BN);

            mma_sync(acc, a_frag, b_frag, acc);
        }

        // Synchronize to make sure the multiplication is done before loading new tiles
        __syncthreads();


    }

    const int threadCol = threadIdx.x % (BN / WMMA_N);
    const int threadRow = threadIdx.x / (BN / WMMA_N);

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
