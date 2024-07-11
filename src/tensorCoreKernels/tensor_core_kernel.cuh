#pragma once

#include <algorithm>
#include <cassert>
#include <cstdio>
#include <cstdlib>
#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <mma.h>

using namespace nvcuda::wmma;
#define CEIL_DIV(M, N) (((M) + (N)-1) / (N))

template <const int BM, const int BN, const int BK, const int TM, const int TN>
__global__ void __launch_bounds__((BM * BN) / (TM * TN), 1)
    sgemmTensorCores(int M, int N, int K, float alpha, const float *A,
                     const float *B, float beta, float *C) {
    const uint cRow = blockIdx.y;
    const uint cCol = blockIdx.x;

    const uint totalResultsBlocktile = BM * BN;
    const uint numThreadsBlocktile = totalResultsBlocktile / (TM * TN);

    assert(numThreadsBlocktile == blockDim.x);

    const int threadCol = threadIdx.x % (BN / TN);
    const int threadRow = threadIdx.x / (BN / TN);

    __shared__ float As[BM * BK];
    __shared__ float Bs[BK * BN];

    A += cRow * BM * K;
    B += cCol * BN;
    C += cRow * BM * N + cCol * BN;

    const uint innerRowA = threadIdx.x / BK;
    const uint innerColA = threadIdx.x % BK;
    const uint strideA = numThreadsBlocktile / BK;
    const uint innerRowB = threadIdx.x / BN;
    const uint innerColB = threadIdx.x % BN;
    const uint strideB = numThreadsBlocktile / BN;

    fragment<matrix_a, BM, BK, BM, float, row_major> aFrag;
    fragment<matrix_b, BK, BN, BN, float, col_major> bFrag;
    fragment<accumulator, BM, BN, BM, float> accFrag;

    fill_fragment(accFrag, 0.0f);

    for (uint bkIdx = 0; bkIdx < K; bkIdx += BK) {
        for (uint loadOffset = 0; loadOffset < BM; loadOffset += strideA) {
            As[(innerRowA + loadOffset) * BK + innerColA] =
                A[(innerRowA + loadOffset) * K + innerColA];
        }
        for (uint loadOffset = 0; loadOffset < BK; loadOffset += strideB) {
            Bs[(innerRowB + loadOffset) * BN + innerColB] =
                B[(innerRowB + loadOffset) * N + innerColB];
        }
        __syncthreads();

        load_matrix_sync(aFrag, As, BK);
        load_matrix_sync(bFrag, Bs, BK);

        mma_sync(accFrag, aFrag, bFrag, accFrag);
        __syncthreads();

        A += BK;
        B += BK * N;
    }

    for (int i = 0; i < accFrag.num_elements; i++) {
        int row = i / TM;
        int col = i % TN;
        C[(threadRow * TM + row) * N + threadCol * TN + col] =
            alpha * accFrag.x[i] + beta * C[(threadRow * TM + row) * N + threadCol * TN + col];
    }
}