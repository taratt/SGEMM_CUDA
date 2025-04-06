#pragma once

#include "cuda_fp16.h"
#include <algorithm>
#include <cassert>
#include <cstdio>
#include <cstdlib>
#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <runner.cuh>

#define CEIL_DIV(M, N) (((M) + (N) - 1) / (N))
#define MMA_M 16
#define MMA_N 8
#define MMA_K 16
#define WARP_SIZE 32

template <const int BM, const int BN, const int BK>
__global__ void runSgemmIntPtxMma(int M, int N, int K, float alpha, int8_t *A,
                                  int8_t *B, float beta, int32_t *C) {
  // Determine block index and thread index
  const uint cRow = blockIdx.y;
  const uint cCol = blockIdx.x;
  const uint totalResultsBlocktile = BM * BN;
  const uint numThreadsBlocktile = 16 * totalResultsBlocktile / (MMA_M * 16);
  const uint numWarpBlocktile = numThreadsBlocktile / WARP_SIZE;

  assert(numThreadsBlocktile == blockDim.x);

  // Shared memory for sub-matrices
  __shared__ int8_t As[BM * BK];
  __shared__ int8_t Bs[BK * BN];

  A += cRow * BM * K;
  B += cCol * BN;
  C += cRow * BM * N + cCol * BN;

  const uint numAsElements = BM * BK;
  const uint numBsElements = BK * BN;

  // Determine the row and column for loading A and B into shared memory
  const uint rowSharedLoaderA = threadIdx.x / (BK / 16);
  const uint colSharedLoaderA = threadIdx.x % (BK / 16);

  const uint rowSharedLoaderB = (threadIdx.x - numAsElements / 16) / (BN / 16);
  const uint colSharedLoaderB = (threadIdx.x - numAsElements / 16) % (BN / 16);

  const int threadCol = threadIdx.x % (BN / MMA_N);
  const int threadRow = threadIdx.x / (BN / MMA_N);

  // The warp a thread is located in
  const int threadWarp = threadIdx.x / WARP_SIZE;
  int numWarpSpanBN = numWarpBlocktile / (BM / MMA_M);
  int numColSpanBN = (BN / MMA_N) / numWarpSpanBN;
  int warpRow = threadWarp / numWarpSpanBN;
  int warpCol = threadWarp % numWarpSpanBN;

  uint32_t ARegisters[2];
  uint32_t BRegisters[4];
  int lane = (threadIdx.x % WARP_SIZE);

  // Initialize registers for PTX-level MMA operations
  int32_t acc0[4] = {0, 0, 0, 0};
  int32_t acc1[4] = {0, 0, 0, 0};
  int32_t acc2[4] = {0, 0, 0, 0};
  int32_t acc3[4] = {0, 0, 0, 0};

  for (uint bkIdx = 0; bkIdx < K; bkIdx += BK) {
    if (threadIdx.x < numAsElements / 16) {
      reinterpret_cast<int4 *>(
          &As[rowSharedLoaderA * BK + colSharedLoaderA * 16])[0] =
          reinterpret_cast<int4 *>(
              &A[rowSharedLoaderA * K + colSharedLoaderA * 16])[0];
    } else if (threadIdx.x >= numAsElements / 16 &&
               threadIdx.x < (numAsElements + numBsElements) / 16) {
      reinterpret_cast<int4 *>(
          &Bs[rowSharedLoaderB * BN + colSharedLoaderB * 16])[0] =
          reinterpret_cast<int4 *>(
              &B[rowSharedLoaderB * N + colSharedLoaderB * 16])[0];
    }

    __syncthreads();

    if (threadIdx.x < numAsElements / 16) {
      A += BK;
    } else if (threadIdx.x >= numAsElements / 16 &&
               threadIdx.x < (numAsElements + numBsElements) / 16) {
      B += BK * N;
    }

    for (int i = 0; i < BK; i += MMA_K) {
      asm volatile("ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0, %1}, [%2];\n"
                   : "=r"(ARegisters[0]), "=r"(ARegisters[1])
                   : "r"(static_cast<uint32_t>(__cvta_generic_to_shared(
                       &(As[(warpRow * MMA_M) * BK + i])))));

      asm volatile(
          "ldmatrix.sync.aligned.m8n8.x2.trans.shared.b16 {%0, %1}, [%2];\n"
          : "=r"(BRegisters[0]), "=r"(BRegisters[1])
          : "r"(static_cast<uint32_t>(__cvta_generic_to_shared(
              &(Bs[i * BN + warpCol * numColSpanBN * MMA_N])))));

      // PTX inline assembly for MMA, using explicit casts to short
      asm volatile("mma.sync.aligned.m16n8k16.row.col.s32.s8.s8.s32 "
                   "{%0, %1, %2, %3}, {%4, %5}, {%6}, {%7, %8, %9, %10};\n"
                   : "=r"(acc0[0]), "=r"(acc0[1]), "=r"(acc0[2]),
                     "=r"(acc0[3]) // Output registers
                   : "r"(ARegisters[0]), "r"(ARegisters[1]), "r"(BRegisters[0]),
                     "r"(acc0[0]), "r"(acc0[1]), "r"(acc0[2]),
                     "r"(acc0[3]) // Accumulators
      );

      asm volatile("mma.sync.aligned.m16n8k16.row.col.s32.s8.s8.s32 "
                   "{%0, %1, %2, %3}, {%4, %5}, {%6}, {%7, %8, %9, %10};\n"
                   : "=r"(acc1[0]), "=r"(acc1[1]), "=r"(acc1[2]),
                     "=r"(acc1[3]) // Output registers
                   : "r"(ARegisters[0]), "r"(ARegisters[1]), "r"(BRegisters[1]),
                     "r"(acc1[0]), "r"(acc1[1]), "r"(acc1[2]),
                     "r"(acc1[3]) // Accumulators
      );

      asm volatile(
          "ldmatrix.sync.aligned.m8n8.x2.trans.shared.b16 {%0, %1}, [%2];\n"
          : "=r"(BRegisters[2]), "=r"(BRegisters[3])
          : "r"(static_cast<uint32_t>(__cvta_generic_to_shared(
              &(Bs[i * BN + (warpCol * numColSpanBN + 2) * MMA_N])))));

      asm volatile("mma.sync.aligned.m16n8k16.row.col.s32.s8.s8.s32 "
                   "{%0, %1, %2, %3}, {%4, %5}, {%6}, {%7, %8, %9, %10};\n"
                   : "=r"(acc2[0]), "=r"(acc2[1]), "=r"(acc2[2]),
                     "=r"(acc2[3]) // Output registers
                   : "r"(ARegisters[0]), "r"(ARegisters[1]), "r"(BRegisters[2]),
                     "r"(acc2[0]), "r"(acc2[1]), "r"(acc2[2]),
                     "r"(acc2[3]) // Accumulators
      );

      asm volatile("mma.sync.aligned.m16n8k16.row.col.s32.s8.s8.s32 "
                   "{%0, %1, %2, %3}, {%4, %5}, {%6}, {%7, %8, %9, %10};\n"
                   : "=r"(acc3[0]), "=r"(acc3[1]), "=r"(acc3[2]),
                     "=r"(acc3[3]) // Output registers
                   : "r"(ARegisters[0]), "r"(ARegisters[1]), "r"(BRegisters[3]),
                     "r"(acc3[0]), "r"(acc3[1]), "r"(acc3[2]),
                     "r"(acc3[3]) // Accumulators
      );
    }

    __syncthreads();
  }

  C[(warpRow * MMA_M) * N + warpCol * numColSpanBN * MMA_N + (lane / 4) * N +
    (lane % 4) * 2] = acc0[0];
  C[(warpRow * MMA_M) * N + warpCol * numColSpanBN * MMA_N + (lane / 4) * N +
    (lane % 4) * 2 + 1] = acc0[1];
  C[(warpRow * MMA_M) * N + warpCol * numColSpanBN * MMA_N +
    (lane / 4 + 8) * N + (lane % 4) * 2] = acc0[2];
  C[(warpRow * MMA_M) * N + warpCol * numColSpanBN * MMA_N +
    (lane / 4 + 8) * N + (lane % 4) * 2 + 1] = acc0[3];
  C[(warpRow * MMA_M) * N + (warpCol * numColSpanBN + 1) * MMA_N +
    (lane / 4) * N + (lane % 4) * 2] = acc1[0];
  C[(warpRow * MMA_M) * N + (warpCol * numColSpanBN + 1) * MMA_N +
    (lane / 4) * N + (lane % 4) * 2 + 1] = acc1[1];
  C[(warpRow * MMA_M) * N + (warpCol * numColSpanBN + 1) * MMA_N +
    (lane / 4 + 8) * N + (lane % 4) * 2] = acc1[2];
  C[(warpRow * MMA_M) * N + (warpCol * numColSpanBN + 1) * MMA_N +
    (lane / 4 + 8) * N + (lane % 4) * 2 + 1] = acc1[3];
  C[(warpRow * MMA_M) * N + (warpCol * numColSpanBN + 2) * MMA_N +
    (lane / 4) * N + (lane % 4) * 2] = acc2[0];
  C[(warpRow * MMA_M) * N + (warpCol * numColSpanBN + 2) * MMA_N +
    (lane / 4) * N + (lane % 4) * 2 + 1] = acc2[1];
  C[(warpRow * MMA_M) * N + (warpCol * numColSpanBN + 2) * MMA_N +
    (lane / 4 + 8) * N + (lane % 4) * 2] = acc2[2];
  C[(warpRow * MMA_M) * N + (warpCol * numColSpanBN + 2) * MMA_N +
    (lane / 4 + 8) * N + (lane % 4) * 2 + 1] = acc2[3];
  C[(warpRow * MMA_M) * N + (warpCol * numColSpanBN + 3) * MMA_N +
    (lane / 4) * N + (lane % 4) * 2] = acc3[0];
  C[(warpRow * MMA_M) * N + (warpCol * numColSpanBN + 3) * MMA_N +
    (lane / 4) * N + (lane % 4) * 2 + 1] = acc3[1];
  C[(warpRow * MMA_M) * N + (warpCol * numColSpanBN + 3) * MMA_N +
    (lane / 4 + 8) * N + (lane % 4) * 2] = acc3[2];
  C[(warpRow * MMA_M) * N + (warpCol * numColSpanBN + 3) * MMA_N +
    (lane / 4 + 8) * N + (lane % 4) * 2 + 1] = acc3[3];
}
