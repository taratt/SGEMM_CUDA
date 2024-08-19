#pragma once

#include <algorithm>
#include <cassert>
#include <cstdio>
#include <cstdlib>
#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <mma.h>
#include <runner.cuh>
#include <cooperative_groups.h>
#include <cooperative_groups/memcpy_async.h>
#include <cuda/std/type_traits>
#include <cuda/barrier>
#include <cuda/pipeline>
#include "cuda_fp16.h"

#define CEIL_DIV(M, N) (((M) + (N)-1) / (N))
#define WMMA_M 16
#define WMMA_N 16
#define WMMA_K 16
#define WARP_SIZE 32

using namespace nvcuda;

template <const int BM, const int BN, const int BK>
__global__ void runSgemmDoubleBufferingTensorCoreStride3(int M, int N, int K, float alpha, __half *A,
                      __half *B, float beta, float *C) {
    // Determine block index and thread index
    const uint cRow = blockIdx.y;
    const uint cCol = blockIdx.x;
    const uint totalResultsBlocktile = BM * BN;
    const uint numThreadsBlocktile = 16*totalResultsBlocktile / (WMMA_M * WMMA_N);
    const uint numWarpBlocktile = numThreadsBlocktile / WARPSIZE;

    assert(numThreadsBlocktile == blockDim.x);

    // Shared memory for sub-matrices
    __shared__ __half As[3][BM * BK];
    __shared__ __half Bs[3][BK * BN];

    A += cRow * BM * K;
    B += cCol * BN;
    C += cRow * BM * N + cCol * BN;

    const uint numAsElements = BM * BK;
    const uint numBsElements = BK * BN;

    // Determine the row and column for loading A and B into shared memory
    const uint rowSharedLoaderA = threadIdx.x / (BK / 8);
    const uint colSharedLoaderA = threadIdx.x % (BK / 8);

    const uint rowSharedLoaderB = (threadIdx.x - numAsElements/8) / (BN / 8);
    const uint colSharedLoaderB = (threadIdx.x - numAsElements/8) % (BN / 8);


    const int threadCol = threadIdx.x % (BN / WMMA_N);
    const int threadRow = threadIdx.x / (BN / WMMA_N);

    //The warp a thread is located in
    const int threadWarp = threadIdx.x / WARPSIZE;
    int numWarpSpanBN = numWarpBlocktile / (BM/WMMA_M);
    int numColSpanBN = (BN / WMMA_N) / numWarpSpanBN;
    int warpRow = threadWarp / numWarpSpanBN;
    int warpCol = threadWarp % numWarpSpanBN;


    //initialize the warp-level fragments
    wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, __half, wmma::row_major> a_frag;
    wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, __half, wmma::row_major> b_frag;
    wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> accs[2];

#pragma unroll
    for (int i =0; i<3; i++)
        wmma::fill_fragment(accs[i], 0.0f);


    cuda::pipeline<cuda::thread_scope_thread> pipe = cuda::make_pipeline();

    for (int stage = 0; stage < 3; ++stage) {
        pipe.producer_acquire();
        if (threadIdx.x < numAsElements/8) {
            cuda::memcpy_async(
                reinterpret_cast<float4 *>(&As[stage][rowSharedLoaderA * BK + colSharedLoaderA * 8]),
                reinterpret_cast<float4 *>(&A[rowSharedLoaderA * K + colSharedLoaderA * 8]),
                sizeof(float4),
                pipe
            );


            // if (threadIdx.x==1 && blockIdx.x ==0 && blockIdx.y==0 && bkIdx==0)
            //     for (uint i = 0; i < 8; i+= 1)
            //         printf("%f  ",__half2float(As[rowSharedLoaderA * BN + colSharedLoaderA * 8+i]));
        }
        else if (threadIdx.x >= numAsElements/8 && threadIdx.x < (numAsElements + numBsElements)/8) {

            cuda::memcpy_async(
                reinterpret_cast<float4 *>(&Bs[stage][rowSharedLoaderB * BN + colSharedLoaderB * 8]),
                reinterpret_cast<float4 *>(&B[rowSharedLoaderB * N + colSharedLoaderB * 8]),
                sizeof(float4),
                pipe
            );

        }
        pipe.producer_commit();
        if (threadIdx.x < numAsElements/8)
            A += BK;
        else if (threadIdx.x >= numAsElements/8 && threadIdx.x < (numAsElements + numBsElements)/8)
            B += BK * N;
    }

    int stage = 0;
    for (uint bkIdx = 0; bkIdx < K; bkIdx += BK) {
        cuda::pipeline_consumer_wait_prior<2>(pipe);

        __syncthreads();
        // if (threadIdx.x==0 && blockIdx.x ==0 && blockIdx.y==0 && bkIdx==0) {
        //     for (int i = 0 ; i <BM*BK ; i++) {
        //         if(i%BK==0)
        //             printf("\n");
        //         printf("%f  ", __half2float(As[i]));
        //
        //     }
        // }


        for (int i = 0; i < BK; i += WMMA_K) {
            wmma::load_matrix_sync(a_frag, As[stage] + (warpRow * WMMA_M) * BK + i , BK);
            wmma::load_matrix_sync(b_frag, Bs[stage] + i* BN + warpCol * numColSpanBN * WMMA_N , BN);
            wmma::mma_sync(accs[0], a_frag, b_frag, accs[0]);

            wmma::load_matrix_sync(b_frag, Bs[stage] + i* BN + (warpCol * numColSpanBN +1 )* WMMA_N  , BN);
            wmma::mma_sync(accs[1], a_frag, b_frag, accs[1]);
        }

        __syncthreads();

        if (bkIdx < K - BK){
            pipe.consumer_release();

            pipe.producer_acquire();

            if (threadIdx.x < numAsElements/8) {
                cuda::memcpy_async(
                    reinterpret_cast<float4 *>(&As[stage][rowSharedLoaderA * BK + colSharedLoaderA * 8]),
                    reinterpret_cast<float4 *>(&A[rowSharedLoaderA * K + colSharedLoaderA * 8]),
                    sizeof(float4),
                    pipe
                );


                // if (threadIdx.x==1 && blockIdx.x ==0 && blockIdx.y==0 && bkIdx==0)
                //     for (uint i = 0; i < 8; i+= 1)
                //         printf("%f  ",__half2float(As[rowSharedLoaderA * BN + colSharedLoaderA * 8+i]));
            }
            else if (threadIdx.x >= numAsElements/8 && threadIdx.x < (numAsElements + numBsElements)/8) {

                cuda::memcpy_async(
                    reinterpret_cast<float4 *>(&Bs[stage][rowSharedLoaderB * BN + colSharedLoaderB * 8]),
                    reinterpret_cast<float4 *>(&B[rowSharedLoaderB * N + colSharedLoaderB * 8]),
                    sizeof(float4),
                    pipe
                );

            }

            pipe.producer_commit();
            if (threadIdx.x < numAsElements/8)
                A += BK;
            else if (threadIdx.x >= numAsElements/8 && threadIdx.x < (numAsElements + numBsElements)/8)
                B += BK * N;

            stage = (stage + 1) % 3;
        }
    }
        wmma::store_matrix_sync(C + (warpRow * WMMA_M) * N + warpCol * numColSpanBN * WMMA_N, accs[0], N, wmma::mem_row_major);
        wmma::store_matrix_sync(C + (warpRow * WMMA_M) * N + (warpCol * numColSpanBN + 1) * WMMA_N, accs[1], N, wmma::mem_row_major);

}