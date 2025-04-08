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

__device__ inline void store_mma_results_16x8_vectorized(int32_t *C_tile_ptr,
                                                         const int32_t *acc,
                                                         int N, int lane) {
  // Each thread handles a 2x2 block within the 16x8 tile
  int row_offset = (lane / 4);
  int col_offset = (lane % 4) * 2; // Base column for the pair

  // Calculate pointers to the start of the pair for the top and bottom rows
  int32_t *C_pair0_ptr = C_tile_ptr + row_offset * N + col_offset;
  int32_t *C_pair1_ptr = C_tile_ptr + (row_offset + 8) * N + col_offset;

  // Store using int2. Assumes N is even and C_tile_ptr alignment allows this.
  reinterpret_cast<int2 *>(C_pair0_ptr)[0] = make_int2(acc[0], acc[1]);
  reinterpret_cast<int2 *>(C_pair1_ptr)[0] = make_int2(acc[2], acc[3]);
}

template <const int BM, const int BN, const int BK>
__global__ void runSgemmIntPtxMma(int M, int N, int K, int32_t alpha, int8_t *A,
                                  int8_t *B, int32_t beta, int32_t *C) {
  // Determine block index and thread index
  const uint cRow = blockIdx.y;
  const uint cCol = blockIdx.x;
  const uint totalResultsBlocktile = BM * BN;
  const uint numThreadsBlocktile = totalResultsBlocktile / (MMA_M);
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
  int lane = (threadIdx.x % WARP_SIZE);
  int numWarpSpanBN = numWarpBlocktile / (BM / MMA_M);
  int numColSpanBN = (BN / MMA_N) / numWarpSpanBN;
  int warpRow = threadWarp / numWarpSpanBN;
  int warpCol = threadWarp % numWarpSpanBN;

  uint32_t ARegisters[2];
  uint32_t BRegisters[4];

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
                       &(As[(warpRow * MMA_M) * BK + i + (lane % 16) * BK +
                            (lane / 16) * 8])))));

      asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0, %1, "
                   "%2, %3}, [%4];\n"
                   : "=r"(BRegisters[0]), "=r"(BRegisters[1]),
                     "=r"(BRegisters[2]), "=r"(BRegisters[3])
                   : "r"(static_cast<uint32_t>(__cvta_generic_to_shared(
                       &(Bs[i * BN + warpCol * numColSpanBN * MMA_N +
                            (lane % 16) * BN])))));

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

  int32_t *C_warp_base = C + (warpRow * MMA_M) * N + (warpCol * 4 * MMA_N);
  store_mma_results_16x8_vectorized(C_warp_base + 0 * MMA_N, acc0, N,
                                    lane); // Tile 0
  store_mma_results_16x8_vectorized(C_warp_base + 1 * MMA_N, acc1, N,
                                    lane); // Tile 1
  store_mma_results_16x8_vectorized(C_warp_base + 2 * MMA_N, acc2, N,
                                    lane); // Tile 2
  store_mma_results_16x8_vectorized(C_warp_base + 3 * MMA_N, acc3, N,
                                    lane); // Tile 3
}
