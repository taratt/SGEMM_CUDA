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
__global__ void runSgemmResolvingStallsTensorCore(int M, int N, int K, float alpha, __half * __restrict__ A,
                      __half * __restrict__ B, float beta, float * __restrict__ C) {
    // Determine block index and thread index
    const uint cRow = blockIdx.y;
    const uint cCol = blockIdx.x;
    const uint totalResultsBlocktile = BM * BN;
    const uint numThreadsBlocktile = 2*totalResultsBlocktile / (WMMA_M * WMMA_N);
    const uint numWarpBlocktile = (numThreadsBlocktile / WARPSIZE);

    assert(numThreadsBlocktile == blockDim.x);


    // Shared memory for sub-matrices
    extern __shared__ __half As[];
    extern __shared__ __half Bs[];

    A += cRow * BM * K;
    B += cCol * BN;
    C += cRow * BM * N + cCol * BN;

    const uint numAsElements = BM * BK;
    const uint numBsElements = BK * BN;

    const uint numLoadElements = numAsElements / (numThreadsBlocktile / 2) ;

    // Determine the row and column for loading A and B into shared memory
    const uint rowSharedLoaderA = threadIdx.x / (BK / numLoadElements);
    //const uint colSharedLoaderA = threadIdx.x % (BK / numLoadElements);

    const uint rowSharedLoaderB = (threadIdx.x - numAsElements/numLoadElements) / (BN / numLoadElements);
  //  const uint colSharedLoaderB = (threadIdx.x - numAsElements/numLoadElements) % (BN / numLoadElements);


    const int threadCol = threadIdx.x % (BN / WMMA_N);
    const int threadRow = threadIdx.x / (BN / WMMA_N);

    //The warp a thread is located in
    const int threadWarp = threadIdx.x / WARPSIZE;
    // int numWarpSpanBN = numWarpBlocktile / (BM/WMMA_M);
    // int numColSpanBN = (BN / WMMA_N) / numWarpSpanBN;
    // int warpRow = threadWarp / numWarpSpanBN;
    // int warpCol = threadWarp % numWarpSpanBN;

    int numWarpSpanBN = 1;
    int numColSpanBN = 8;
    int numRowSpanAN = 2;
    int warpRow = (threadWarp / numWarpSpanBN);
    int warpCol = 0;

    // if (threadIdx.x==1 && blockIdx.x ==0 && blockIdx.y==0)
    //     printf("He %d \n", numColSpanBN);

    //initialize the warp-level fragments
    wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, __half, wmma::row_major> a_frags[2];
    wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, __half, wmma::row_major> b_frag;
    wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> accs[16];

#pragma unroll
    for (int i =0; i<16; i++)
        wmma::fill_fragment(accs[i], 0.0f);

    cuda::pipeline<cuda::thread_scope_thread> pipe = cuda::make_pipeline();

    for (int stage = 0; stage < 2; ++stage) {
        pipe.producer_acquire();
        if (threadIdx.x < 64) {
            cuda::memcpy_async(
                reinterpret_cast<float4 *>(&As[stage*numAsElements + rowSharedLoaderA * BK]),
                reinterpret_cast<float4 *>(&A[rowSharedLoaderA * K]),
                8*sizeof(float4),
                pipe
            );
            cuda::memcpy_async(
                reinterpret_cast<float4 *>(&As[stage*numAsElements + (rowSharedLoaderA + 1) * BK]),
                reinterpret_cast<float4 *>(&A[(rowSharedLoaderA + 1) * K]),
                8*sizeof(float4),
                pipe
            );


            // if (threadIdx.x==1 && blockIdx.x ==0 && blockIdx.y==0 && bkIdx==0)
            //     for (uint i = 0; i < 8; i+= 1)
            //         printf("%f  ",__half2float(As[rowSharedLoaderA * BN + colSharedLoaderA * 8+i]));
        }
        else if (threadIdx.x >= numAsElements/numLoadElements && threadIdx.x < (numAsElements + numBsElements)/numLoadElements) {

            cuda::memcpy_async(
                reinterpret_cast<float4 *>(&Bs[stage*numBsElements + rowSharedLoaderB * BN]),
                reinterpret_cast<float4 *>(&B[rowSharedLoaderB * N]),
                16*sizeof(float4),
                pipe
            );

        }
        pipe.producer_commit();
        if (threadIdx.x < numAsElements/numLoadElements)
            A += BK;
        else if (threadIdx.x >= numAsElements/numLoadElements && threadIdx.x < (numAsElements + numBsElements)/numLoadElements)
            B += BK * N;
    }

    int stage = 0;
    for (uint bkIdx = 0; bkIdx < K; bkIdx += BK) {
        cuda::pipeline_consumer_wait_prior<1>(pipe);
        __syncthreads();
        if (threadIdx.x==1 && blockIdx.x ==0 && blockIdx.y==0 && bkIdx==0)
            for (uint x = 0; x < 128*64; x+= 1) {
                if (x%64==0)
                    printf("\n");
                printf("%f  ",__half2float(As[x]));

            }
        if (threadIdx.x < 128) {
        #pragma unroll
            for (int i = 0; i < BK; i += WMMA_K) {
                wmma::load_matrix_sync(a_frags[0], As+stage*numAsElements + (warpRow * WMMA_M) * BK + i , BK);
                wmma::load_matrix_sync(a_frags[1], As+stage*numAsElements + ((warpRow + 4) * WMMA_M) * BK + i , BK);

                wmma::load_matrix_sync(b_frag, Bs+stage*numBsElements + i* BN + warpCol * numColSpanBN * WMMA_N , BN);
                wmma::mma_sync(accs[0], a_frags[0], b_frag, accs[0]);
                wmma::mma_sync(accs[8], a_frags[1], b_frag, accs[8]);

                wmma::load_matrix_sync(b_frag, Bs+stage*numBsElements+ i* BN + (warpCol * numColSpanBN +1 )* WMMA_N  , BN);
                wmma::mma_sync(accs[1], a_frags[0], b_frag, accs[1]);
                wmma::mma_sync(accs[9], a_frags[1], b_frag, accs[9]);

                wmma::load_matrix_sync(b_frag, Bs+stage*numBsElements+ i* BN + (warpCol * numColSpanBN +2 )* WMMA_N  , BN);
                wmma::mma_sync(accs[2], a_frags[0], b_frag, accs[2]);
                wmma::mma_sync(accs[10], a_frags[1], b_frag, accs[10]);

                wmma::load_matrix_sync(b_frag, Bs+stage*numBsElements+ i* BN + (warpCol * numColSpanBN +3 )* WMMA_N  , BN);
                wmma::mma_sync(accs[3], a_frags[0], b_frag, accs[3]);
                wmma::mma_sync(accs[11], a_frags[1], b_frag, accs[11]);

                wmma::load_matrix_sync(b_frag, Bs+stage*numBsElements+ i* BN + (warpCol * numColSpanBN +4 )* WMMA_N  , BN);
                wmma::mma_sync(accs[4], a_frags[0], b_frag, accs[4]);
                wmma::mma_sync(accs[12], a_frags[1], b_frag, accs[12]);

                wmma::load_matrix_sync(b_frag, Bs+stage*numBsElements+ i* BN + (warpCol * numColSpanBN +5 )* WMMA_N  , BN);
                wmma::mma_sync(accs[5], a_frags[0], b_frag, accs[5]);
                wmma::mma_sync(accs[13], a_frags[1], b_frag, accs[13]);

                wmma::load_matrix_sync(b_frag, Bs+stage*numBsElements+ i* BN + (warpCol * numColSpanBN +6 )* WMMA_N  , BN);
                wmma::mma_sync(accs[6], a_frags[0], b_frag, accs[6]);
                wmma::mma_sync(accs[14], a_frags[1], b_frag, accs[14]);

                wmma::load_matrix_sync(b_frag, Bs+stage*numBsElements+ i* BN + (warpCol * numColSpanBN +7 )* WMMA_N  , BN);
                wmma::mma_sync(accs[7], a_frags[0], b_frag, accs[7]);
                wmma::mma_sync(accs[15], a_frags[1], b_frag, accs[15]);

            }

        }

        __syncthreads();

        if (bkIdx < K - BK){
            pipe.consumer_release();

            pipe.producer_acquire();

            if (threadIdx.x < 64) {
                cuda::memcpy_async(
                reinterpret_cast<float4 *>(&As[stage*numAsElements + rowSharedLoaderA * BK]),
                reinterpret_cast<float4 *>(&A[rowSharedLoaderA * K]),
                8*sizeof(float4),
                pipe
            );
                cuda::memcpy_async(
                    reinterpret_cast<float4 *>(&As[stage*numAsElements + (rowSharedLoaderA + 1) * BK]),
                    reinterpret_cast<float4 *>(&A[(rowSharedLoaderA + 1) * K]),
                    8*sizeof(float4),
                    pipe
                );


                // if (threadIdx.x==1 && blockIdx.x ==0 && blockIdx.y==0 && bkIdx==0)
                //     for (uint i = 0; i < 8; i+= 1)
                //         printf("%f  ",__half2float(As[rowSharedLoaderA * BN + colSharedLoaderA * 8+i]));
            }
            else if (threadIdx.x >= numAsElements/numLoadElements && threadIdx.x < (numAsElements + numBsElements)/numLoadElements) {

                cuda::memcpy_async(
                    reinterpret_cast<float4 *>(&Bs[stage*numBsElements + rowSharedLoaderB * BN]),
                    reinterpret_cast<float4 *>(&B[rowSharedLoaderB * N]),
                    16*sizeof(float4),
                    pipe
                );

            }

            pipe.producer_commit();
            if (threadIdx.x < numAsElements/numLoadElements)
                A += BK;
            else if (threadIdx.x >= numAsElements/numLoadElements && threadIdx.x < (numAsElements + numBsElements)/numLoadElements)
                B += BK * N;

            stage = (stage + 1) % 2;
        }
    }
    if (threadIdx.x < 128) {
        wmma::store_matrix_sync(C + (warpRow * WMMA_M) * N + warpCol * numColSpanBN * WMMA_N, accs[0], N, wmma::mem_row_major);
        wmma::store_matrix_sync(C + (warpRow * WMMA_M) * N + (warpCol * numColSpanBN + 1) * WMMA_N, accs[1], N, wmma::mem_row_major);
        wmma::store_matrix_sync(C + (warpRow * WMMA_M) * N + (warpCol * numColSpanBN + 2) * WMMA_N, accs[2], N, wmma::mem_row_major);
        wmma::store_matrix_sync(C + (warpRow * WMMA_M) * N + (warpCol * numColSpanBN + 3) * WMMA_N, accs[3], N, wmma::mem_row_major);
        wmma::store_matrix_sync(C + (warpRow * WMMA_M) * N + (warpCol * numColSpanBN + 4) * WMMA_N, accs[4], N, wmma::mem_row_major);
        wmma::store_matrix_sync(C + (warpRow * WMMA_M) * N + (warpCol * numColSpanBN + 5) * WMMA_N, accs[5], N, wmma::mem_row_major);
        wmma::store_matrix_sync(C + (warpRow * WMMA_M) * N + (warpCol * numColSpanBN + 6) * WMMA_N, accs[6], N, wmma::mem_row_major);
        wmma::store_matrix_sync(C + (warpRow * WMMA_M) * N + (warpCol * numColSpanBN + 7) * WMMA_N, accs[7], N, wmma::mem_row_major);

        wmma::store_matrix_sync(C + ((warpRow + 4) * WMMA_M) * N + (warpCol * numColSpanBN + 8) * WMMA_N, accs[8], N, wmma::mem_row_major);
        wmma::store_matrix_sync(C + ((warpRow + 4) * WMMA_M) * N + (warpCol * numColSpanBN + 9) * WMMA_N, accs[9], N, wmma::mem_row_major);
        wmma::store_matrix_sync(C + ((warpRow + 4) * WMMA_M) * N + (warpCol * numColSpanBN + 10) * WMMA_N, accs[10], N, wmma::mem_row_major);
        wmma::store_matrix_sync(C + ((warpRow + 4) * WMMA_M) * N + (warpCol * numColSpanBN + 11) * WMMA_N, accs[11], N, wmma::mem_row_major);
        wmma::store_matrix_sync(C + ((warpRow + 4) * WMMA_M) * N + (warpCol * numColSpanBN + 12) * WMMA_N, accs[12], N, wmma::mem_row_major);
        wmma::store_matrix_sync(C + ((warpRow + 4) * WMMA_M) * N + (warpCol * numColSpanBN + 13) * WMMA_N, accs[13], N, wmma::mem_row_major);
        wmma::store_matrix_sync(C + ((warpRow + 4) * WMMA_M) * N + (warpCol * numColSpanBN + 14) * WMMA_N, accs[14], N, wmma::mem_row_major);
        wmma::store_matrix_sync(C + ((warpRow + 4) * WMMA_M) * N + (warpCol * numColSpanBN + 15) * WMMA_N, accs[15], N, wmma::mem_row_major);


    }
}