#pragma once

#include <algorithm>
#include <cassert>
#include <cstdio>
#include <cstdlib>
#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <mma.h>
#include <runner.cuh>

#include "cuda_fp16.h"

#define CEIL_DIV(M, N) (((M) + (N)-1) / (N))
#define WMMA_M 16
#define WMMA_N 16
#define WMMA_K 16
#define WARP_SIZE 32

using namespace nvcuda;

template <const int BM, const int BN, const int BK>
__global__ void __launch_bounds__((2*BM * BN) / (WMMA_M * WMMA_N), 1)
    sgemmTensorCores2(int M, int N, int K, float alpha, const __half *A,
                     const __half *B, float beta, float *C) {
    // Determine block index and thread index
    const uint cRow = blockIdx.y;
    const uint cCol = blockIdx.x;
    const uint totalResultsBlocktile = BM * BN;
    const uint numThreadsBlocktile = totalResultsBlocktile / (WMMA_M * WMMA_N);
    const uint numWarpBlocktile = numThreadsBlocktile / WARPSIZE;

    assert(numThreadsBlocktile == blockDim.x);

    // Shared memory for sub-matrices
    __shared__ __half As[BM * BK];
    __shared__ __half Bs[BK * BN];
    // extern __shared__ __half shared_mem[];
    // __half *As = shared_mem;
    // __half *Bs = shared_mem + BM * BK;


    A += cRow * BM * K;
    B += cCol * BN;
    C += cRow * BM * N + cCol * BN;

    // Determine the row and column for loading A and B into shared memory
    const uint rowSharedLoaderA = threadIdx.x / BK;
    const uint colSharedLoaderA = threadIdx.x % BK;
    const uint rowSharedLoaderB = threadIdx.x / BN;
    const uint colSharedLoaderB = threadIdx.x % BN;

    const uint strideA = numThreadsBlocktile / BK;
    uint strideB = numThreadsBlocktile /BN;

    const int threadCol = threadIdx.x % (BN / WMMA_N);
    const int threadRow = threadIdx.x / (BN / WMMA_N);

    //The warp a thread is located in
    const int threadWarp = threadIdx.x / WARPSIZE;

    int numRowsWarp = BM/numWarpBlocktile;
    int numColsWarp = BN/numWarpBlocktile;

    //initialize the warp-level fragments
    wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, __half, wmma::row_major> a_frag;
    wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, __half, wmma::row_major> b_frag;
    //wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> acc;
    //wmma::fill_fragment(acc, 0.0f);
     wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> acc[16];
     for (int i =0; i<16; i++)
         wmma::fill_fragment(acc[i], 0.0f);
    __half threadResults[WMMA_M * WMMA_N] = {0.0};

    for (uint bkIdx = 0; bkIdx < K; bkIdx += BK) {
        if (threadIdx.x < numThreadsBlocktile) {
            for (uint loadOffset = 0; loadOffset < BM; loadOffset += strideA) {
                // if ((rowSharedLoaderA + loadOffset) < BM && colSharedLoaderA < BK) {
                // if( loadOffset==240 &&  bkIdx==0)
                //     printf("%d: %d %d\n", threadIdx.x,rowSharedLoaderA + loadOffset, colSharedLoaderA);
                As[(rowSharedLoaderA + loadOffset) * BK + colSharedLoaderA] =
                    A[(rowSharedLoaderA + loadOffset) * K + colSharedLoaderA];

                // }
            }
            for (uint loadOffset = 0; loadOffset < BK; loadOffset += strideB) {
                // if ((rowSharedLoaderB + loadOffset) < BK && colSharedLoaderB < BN) {
                Bs[(rowSharedLoaderB + loadOffset) * BN + colSharedLoaderB] =
                    B[(rowSharedLoaderB + loadOffset) * N + colSharedLoaderB];
                // if(Bs[(rowSharedLoaderB + loadOffset) * BN + colSharedLoaderB] !=  B[(rowSharedLoaderB + loadOffset) * N + colSharedLoaderB])
                //     printf("faulty %d \n", threadIdx.x);
                // }
            }

            __syncthreads();


            A += BK;
            B += BK * N;
        }
        // for (int warplocind = 0; warplocind < numRowsWarp; warplocind += WMMA_M) {
        // wmma::fill_fragment(acc[threadWarp+ (warplocind/WMMA_M)], 0.0f);
        for (int warpcol = 0; warpcol < BN; warpcol += WMMA_N) {
          //  wmma::fill_fragment(acc[warpcol/WMMA_N], 0.0f);
            for (int i = 0; i < BK; i += WMMA_K) {
                wmma::load_matrix_sync(a_frag, As + (threadWarp * WMMA_N) * BK + i, BK);
                wmma::load_matrix_sync(b_frag, Bs + i * BN + warpcol, BN);

                wmma::mma_sync(acc[warpcol/WMMA_N], a_frag, b_frag, acc[warpcol/WMMA_N]);
            }

            //if (threadIdx.x <WMMA_M*WMMA_N)
                    // for (int i = 0; i < acc[threadWarp].num_elements; ++i) {
                    //     C[(threadWarp + i/WMMA_N) * N + warpcol+i%WMMA_N] += acc[threadWarp].x[i];
                    // }
            //  float *c_tile = C + (threadWarp * WMMA_N) * N + warpcol;
            // wmma::store_matrix_sync(c_tile, acc[warpcol/WMMA_N], N, wmma::mem_row_major);
            // for (int i = 0; i < WMMA_M; ++i) {
            //     for (int j = 0; j < WMMA_N; ++j) {
            //         c_tile[i * N + j] = acc.x[warpOffset + i * WMMA_N + j];
            //     }
            // }

        }



        // Synchronize to make sure the multiplication is done before loading new tiles
        __syncthreads();
    }
    for (int warpcol = 0; warpcol < BN; warpcol += WMMA_N) {
        float *c_tile = C + (threadWarp * WMMA_N) * N + warpcol;
        wmma::store_matrix_sync(c_tile, acc[warpcol/WMMA_N], N, wmma::mem_row_major);
    }
            // }

            //     for (int i = 0; i < BK; i += WMMA_K) {
            //         wmma::load_matrix_sync(a_frag, As + i * BM, BK);
            //         wmma::load_matrix_sync(b_frag, Bs + i * BN, BN);
            //
            //         wmma::mma_sync(acc, a_frag, b_frag, acc);
            //     }
            //
            //     __syncthreads();
            // }
            //
            // wmma::store_matrix_sync(C + (threadRow * WMMA_M) * N + threadCol * WMMA_N, acc, N, wmma::mem_row_major);



            // for (uint resIdxM = 0; resIdxM < WMMA_M; ++resIdxM) {
            //     for (uint resIdxN = 0; resIdxN < WMMA_N; ++resIdxN) {
            //         uint row = threadRow * WMMA_M + resIdxM;
            //         uint col = threadCol * WMMA_N + resIdxN;
            //         if (row < BM && col < BN && (cRow * BM + row) < M && (cCol * BN + col) < N) {
            //             C_tile[row * N + col] =
            //                 alpha * acc.x[resIdxM * WMMA_N + resIdxN] +
            //                 beta * C_tile[row * N + col];
            //         }
            //     }
            // }

        // for (int i =0; i<16; i++)
        //     wmma::fill_fragment(acc[i], 0.0f);

}