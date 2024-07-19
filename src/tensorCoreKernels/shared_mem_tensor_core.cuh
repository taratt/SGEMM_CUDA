#include <cstdio>
#include <cstdlib>
#include <mma.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>

#define WMMA_M 16
#define WMMA_N 16
#define WMMA_K 16

using namespace nvcuda;

template <const int BLOCKSIZE>
__global__ void smemTensorCores(const __half *A, const __half *B, float *C, int M, int N, int K) {
    // Leading dimensions of A and B matrices
    int lda = K;
    int ldb = N;
    int ldc = N;

    __shared__ __half As[BLOCKSIZE * BLOCKSIZE];
    __shared__ __half Bs[BLOCKSIZE * BLOCKSIZE];

    // Fragment declaration
    wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, __half, wmma::row_major> a_frag;
    wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, __half, wmma::row_major> b_frag;
    wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> acc_frag;

    wmma::fill_fragment(acc_frag, 0.0f);

    // Calculate the row and column of the C matrix to be computed by this thread block which is also a warp
    int row = blockIdx.y * WMMA_M;
    int col = blockIdx.x * WMMA_N;

    // Loop over the K dimension to calculate partial results
    for (int i = 0; i < K; i += WMMA_K) {

        wmma::load_matrix_sync(a_frag, A + row * lda + i, lda);
        wmma::load_matrix_sync(b_frag, B + i * ldb + col, ldb);

        // Perform the matrix multiplication
        wmma::mma_sync(acc_frag, a_frag, b_frag, acc_frag);
    }

    // Store the result
    wmma::store_matrix_sync(C + row * ldc + col, acc_frag, ldc, wmma::mem_row_major);
}