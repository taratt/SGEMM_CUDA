#pragma once

#include <algorithm>
#include <cassert>
#include <cstdio>
#include <cstdlib>
#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <runner.cuh>
#include "cuda_fp16.h"

#define CEIL_DIV(M, N) (((M) + (N)-1) / (N))
#define MMA_M 16
#define MMA_N 8
#define MMA_K 16
#define WARP_SIZE 32

template <const int BM, const int BN, const int BK>
__global__ void runSgemmPtxMma(int M, int N, int K, float alpha, __half *A,
                               __half *B, float beta, float *C) {
    // Determine block index and thread index
    const uint cRow = blockIdx.y;
    const uint cCol = blockIdx.x;
    const uint totalResultsBlocktile = BM * BN;
    const uint numThreadsBlocktile = 16 * totalResultsBlocktile / (MMA_M * 16);
    const uint numWarpBlocktile = numThreadsBlocktile / WARP_SIZE;

    assert(numThreadsBlocktile == blockDim.x);

    // Shared memory for sub-matrices
    __shared__ __half As[BM * BK];
    __shared__ __half Bs[BK * BN];

    A += cRow * BM * K;
    B += cCol * BN;
    C += cRow * BM * N + cCol * BN;

    const uint numAsElements = BM * BK;
    const uint numBsElements = BK * BN;

    // Determine the row and column for loading A and B into shared memory
    const uint rowSharedLoaderA = threadIdx.x / (BK / 8);
    const uint colSharedLoaderA = threadIdx.x % (BK / 8);

    const uint rowSharedLoaderB = (threadIdx.x - numAsElements / 8) / (BN / 8);
    const uint colSharedLoaderB = (threadIdx.x - numAsElements / 8) % (BN / 8);

    const int threadCol = threadIdx.x % (BN / MMA_N);
    const int threadRow = threadIdx.x / (BN / MMA_N);

    // The warp a thread is located in
    const int threadWarp = threadIdx.x / WARP_SIZE;
    int numWarpSpanBN = numWarpBlocktile / (BM / MMA_M);
    int numColSpanBN = (BN / MMA_N) / numWarpSpanBN;
    int warpRow = threadWarp / numWarpSpanBN;
    int warpCol = threadWarp % numWarpSpanBN;

    uint32_t ARegisters[4];
    uint32_t BRegisters[2];
    int lane = (threadIdx.x % 32);

    // Initialize registers for PTX-level MMA operations
    float acc0[MMA_M * MMA_N] = {0};
    float acc1[MMA_M * MMA_N] = {0};
    float acc2[MMA_M * MMA_N] = {0};
    float acc3[MMA_M * MMA_N] = {0};

    // for (uint bkIdx = 0; bkIdx < K; bkIdx += BK) {
    //     if (threadIdx.x < numAsElements / 8) {
    //         reinterpret_cast<float4 *>(&As[rowSharedLoaderA * BK + colSharedLoaderA * 8])[0] =
    //             reinterpret_cast<float4 *>(&A[rowSharedLoaderA * K + colSharedLoaderA * 8])[0];
    //     } else if (threadIdx.x >= numAsElements / 8 && threadIdx.x < (numAsElements + numBsElements) / 8) {
    //         reinterpret_cast<float4 *>(&Bs[rowSharedLoaderB * BN + colSharedLoaderB * 8])[0] =
    //             reinterpret_cast<float4 *>(&B[rowSharedLoaderB * N + colSharedLoaderB * 8])[0];
    //     }
    //
    //     __syncthreads();
    //
    //     if (threadIdx.x < numAsElements / 8) {
    //         A += BK;
    //     } else if (threadIdx.x >= numAsElements / 8 && threadIdx.x < (numAsElements + numBsElements) / 8) {
    //         B += BK * N;
    //     }
    //
    //     for (int i = 0; i < BK; i += MMA_K) {
    //
    //         asm volatile("ldmatrix.sync.aligned.x4.m8n8.shared.b16 {%0, %1, %2, %3}, [%4];\n"
    //                  : "=r"(ARegisters[0]), "=r"(ARegisters[1]), "=r"(ARegisters[2]), "=r"(ARegisters[3])
    //                  : "r"(__cvta_generic_to_shared(&As[lane*8])));
    //
    //         asm volatile("ldmatrix.sync.aligned.x2.m8n8.trans.shared.b16 {%0, %1}, [%4];\n"
    //                  : "=r"(BRegisters[0]), "=r"(BRegisters[1])
    //                  : "r"(__cvta_generic_to_shared(&Bs[lane*8])));
    //
    //         // // PTX inline assembly for MMA, using explicit casts to short
    //         // asm volatile(
    //         //     "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
    //         //     "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9, %10, %11}, {%12, %13, %14, %15};\n"
    //         //     : "=f"(acc0[0]), "=f"(acc0[1]), "=f"(acc0[2]), "=f"(acc0[3])  // Output registers
    //         //     : "h"(__half_as_short(A_frag[0])), "h"(__half_as_short(A_frag[1])),
    //         //       "h"(__half_as_short(A_frag[2])), "h"(__half_as_short(A_frag[3])),  // Input A
    //         //       "h"(__half_as_short(B_frag[0])), "h"(__half_as_short(B_frag[1])),
    //         //       "h"(__half_as_short(B_frag[2])), "h"(__half_as_short(B_frag[3])),  // Input B
    //         //       "f"(acc0[0]), "f"(acc0[1]), "f"(acc0[2]), "f"(acc0[3])    // Accumulators
    //         // );
    //         //
    //         //
    //         // asm volatile(
    //         //     "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
    //         //     "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9, %10, %11}, {%12, %13, %14, %15};\n"
    //         //     : "=f"(acc1[0]), "=f"(acc1[1]), "=f"(acc1[2]), "=f"(acc1[3])  // Output registers
    //         //     : "h"(__half_as_short(A_frag[0])), "h"(__half_as_short(A_frag[1])),
    //         //       "h"(__half_as_short(A_frag[2])), "h"(__half_as_short(A_frag[3])),  // Input A
    //         //       "h"(__half_as_short(B_frag[0])), "h"(__half_as_short(B_frag[1])),
    //         //       "h"(__half_as_short(B_frag[2])), "h"(__half_as_short(B_frag[3])),  // Input B
    //         //       "f"(acc1[0]), "f"(acc1[1]), "f"(acc1[2]), "f"(acc1[3])    // Accumulators
    //         // );
    //         //
    //         //
    //         //
    //         // asm volatile(
    //         //     "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
    //         //     "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9, %10, %11}, {%12, %13, %14, %15};\n"
    //         //     : "=f"(acc0[0]), "=f"(acc0[1]), "=f"(acc0[2]), "=f"(acc0[3])  // Output registers
    //         //     : "h"(__half_as_short(A_frag[0])), "h"(__half_as_short(A_frag[1])),
    //         //       "h"(__half_as_short(A_frag[2])), "h"(__half_as_short(A_frag[3])),  // Input A
    //         //       "h"(__half_as_short(B_frag[0])), "h"(__half_as_short(B_frag[1])),
    //         //       "h"(__half_as_short(B_frag[2])), "h"(__half_as_short(B_frag[3])),  // Input B
    //         //       "f"(acc0[0]), "f"(acc0[1]), "f"(acc0[2]), "f"(acc0[3])    // Accumulators
    //         // );
    //         //
    //         //
    //         // asm volatile(
    //         //     "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
    //         //     "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9, %10, %11}, {%12, %13, %14, %15};\n"
    //         //     : "=f"(acc1[0]), "=f"(acc1[1]), "=f"(acc1[2]), "=f"(acc1[3])  // Output registers
    //         //     : "h"(__half_as_short(A_frag[0])), "h"(__half_as_short(A_frag[1])),
    //         //       "h"(__half_as_short(A_frag[2])), "h"(__half_as_short(A_frag[3])),  // Input A
    //         //       "h"(__half_as_short(B_frag[0])), "h"(__half_as_short(B_frag[1])),
    //         //       "h"(__half_as_short(B_frag[2])), "h"(__half_as_short(B_frag[3])),  // Input B
    //         //       "f"(acc1[0]), "f"(acc1[1]), "f"(acc1[2]), "f"(acc1[3])    // Accumulators
    //         // );
    //     }
    //
    //     __syncthreads();
    // }
    //
    // // Store results
    // for (int i = 0; i < MMA_M; ++i) {
    //     for (int j = 0; j < MMA_N; ++j) {
    //         C[(warpRow * MMA_M + i) * N + (warpCol * numColSpanBN + j)] = acc0[i * MMA_N + j];
    //         C[(warpRow * MMA_M + i) * N + (warpCol * numColSpanBN + j) + 1] = acc1[i * MMA_N + j];
    //     }
    // }
}
