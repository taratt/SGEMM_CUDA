#include <cstdio>
#include <cstdlib>
#include <ctime>
#include <fstream>
#include <iostream>
#include <vector>
#include <runner.cuh>
#include <cublas_v2.h>

#define cudaCheck(err) (cudaCheck(err, __FILE__, __LINE__))

const std::string errLogFile = "matrixValidationFailure.txt";

// =========================
// Template Specializations
// =========================

template <typename T>
struct MatrixInitializer {
    static void init_A(T* matrix, size_t size);
    static void init_B(T* matrix, size_t size);
    static void init_C(float* matrix, size_t size);
};

template <>
struct MatrixInitializer<float> {
    static void init_A(float* matrix, size_t size) { randomize_matrix(matrix, size); }
    static void init_B(float* matrix, size_t size) { randomize_matrix(matrix, size); }
    static void init_C(float* matrix, size_t size) { randomize_matrix(matrix, size); }
};

template <>
struct MatrixInitializer<__half> {
    static void init_A(__half* matrix, size_t size) { initialize_incremental_hf(matrix, size); }
    static void init_B(__half* matrix, size_t size) { initialize_one_hf(matrix, size); }
    static void init_C(float* matrix, size_t size) { initialize_one_float(matrix, size); }
};

// ======================
// CUDA Helper Functions
// ======================

struct CudaEventPair {
    cudaEvent_t beg, end;
    CudaEventPair() {
        cudaEventCreate(&beg);
        cudaEventCreate(&end);
    }
    ~CudaEventPair() {
        cudaEventDestroy(beg);
        cudaEventDestroy(end);
    }
};

// ======================
// Core Benchmark Logic
// ======================

template <typename T>
void dispatch_kernel(int kernel_num, long m, long n, long k, float alpha, T* dA, T* dB, float beta, float* dC, cublasHandle_t handle) {
    if constexpr (std::is_same_v<T, float>) {
        run_kernel(kernel_num, m, n, k, alpha, dA, dB, beta, dC, handle);
    } else {
        run_tensor_core_kernel(kernel_num, m, n, k, alpha, dA, dB, beta, dC, handle);
    }
}

template <typename T>
void handle_validation_error(T* A, T* B, float* C, float* C_ref, long m, long n) {
    std::cerr << "Validation failed against cuBLAS reference\n";
    if (m <= 4096) {
        std::ofstream fs(errLogFile);
        fs << "A:\n"; print_matrix_hf(A, m, n, fs);
        fs << "B:\n"; print_matrix_hf(B, m, n, fs);
        fs << "C:\n"; print_matrix(C, m, n, fs);
        fs << "Should:\n"; print_matrix(C_ref, m, n, fs);
    }
}

template <typename T>
void benchmark_gemm(int kernel_num, cublasHandle_t handle, const std::vector<int>& sizes, int deviceIdx) {
    CudaEventPair events;
    const long max_size = 8192;
    const int repeat_times = 50;

    // Host allocations
    T* A = static_cast<T*>(malloc(sizeof(T) * max_size * max_size));
    T* B = static_cast<T*>(malloc(sizeof(T) * max_size * max_size));
    float* C = static_cast<float*>(malloc(sizeof(float) * max_size * max_size));
    float* C_ref = static_cast<float*>(malloc(sizeof(float) * max_size * max_size));

    // Matrix initialization
    MatrixInitializer<T>::init_A(A, max_size * max_size);
    MatrixInitializer<T>::init_B(B, max_size * max_size);
    MatrixInitializer<T>::init_C(C, max_size * max_size);
    memcpy(C_ref, C, sizeof(float) * max_size * max_size);

    // Device allocations
    T *dA, *dB;
    float *dC, *dC_ref;
    cudaCheck(cudaMalloc(&dA, sizeof(T) * max_size * max_size));
    cudaCheck(cudaMalloc(&dB, sizeof(T) * max_size * max_size));
    cudaCheck(cudaMalloc(&dC, sizeof(float) * max_size * max_size));
    cudaCheck(cudaMalloc(&dC_ref, sizeof(float) * max_size * max_size));

    // Host-to-device transfers
    cudaCheck(cudaMemcpy(dA, A, sizeof(T) * max_size * max_size, cudaMemcpyHostToDevice));
    cudaCheck(cudaMemcpy(dB, B, sizeof(T) * max_size * max_size, cudaMemcpyHostToDevice));
    cudaCheck(cudaMemcpy(dC, C, sizeof(float) * max_size * max_size, cudaMemcpyHostToDevice));
    cudaCheck(cudaMemcpy(dC_ref, C_ref, sizeof(float) * max_size * max_size, cudaMemcpyHostToDevice));

    for (int size : sizes) {
        const long m = size, n = size, k = size;
        const float alpha = 1.0f, beta = 0.0f;

        if (kernel_num != 0) {
            // Reference run (cuBLAS)
            dispatch_kernel<T>(0, m, n, k, alpha, dA, dB, beta, dC_ref, handle);
            // Test kernel run
            dispatch_kernel<T>(kernel_num, m, n, k, alpha, dA, dB, beta, dC, handle);
            
            cudaCheck(cudaDeviceSynchronize());
            cudaCheck(cudaMemcpy(C, dC, sizeof(float) * m * n, cudaMemcpyDeviceToHost));
            cudaCheck(cudaMemcpy(C_ref, dC_ref, sizeof(float) * m * n, cudaMemcpyDeviceToHost));

            if (!verify_matrix(C_ref, C, m * n)) {
                handle_validation_error(A, B, C, C_ref, m, n);
                exit(EXIT_FAILURE);
            }
        }

        // Warmup
        for (int j = 0; j < 100; ++j)
            dispatch_kernel<T>(kernel_num, m, n, k, alpha, dA, dB, beta, dC, handle);

        // Timing
        cudaEventRecord(events.beg);
        for (int j = 0; j < repeat_times; ++j)
            dispatch_kernel<T>(kernel_num, m, n, k, alpha, dA, dB, beta, dC, handle);
        cudaEventRecord(events.end);
        cudaEventSynchronize(events.end);

        float elapsed_ms;
        cudaEventElapsedTime(&elapsed_ms, events.beg, events.end);
        const double elapsed = elapsed_ms / 1e3;
        const double gflops = (2.0 * m * n * k * repeat_times) / (elapsed * 1e9);

        printf("Average time: %.6fs, Performance: %.1f GFLOPS, Size: %ld\n",
               elapsed / repeat_times, gflops, m);
    }

    // Cleanup
    free(A); free(B); free(C); free(C_ref);
    cudaFree(dA); cudaFree(dB); cudaFree(dC); cudaFree(dC_ref);
}

// ==============
// Main Function
// ==============

int main(int argc, char** argv) {
    if (argc != 2) {
        std::cerr << "Usage: " << argv[0] << " <kernel 0-25>\n";
        return EXIT_FAILURE;
    }

    const int kernel_num = std::stoi(argv[1]);
    if (kernel_num < 0 || kernel_num > 25) {
        std::cerr << "Invalid kernel number\n";
        return EXIT_FAILURE;
    }

    int deviceIdx = 0;
    if (const char* env = getenv("DEVICE")) deviceIdx = atoi(env);
    cudaCheck(cudaSetDevice(deviceIdx));

    cublasHandle_t handle;
    cublasCreate(&handle);
    cublasSetMathMode(handle, CUBLAS_TENSOR_OP_MATH);

    const std::vector<int> sizes = {128, 256, 512, 1024, 2048, 4096};
    
    if (kernel_num == 0 || kernel_num >= 13)
        benchmark_gemm<__half>(kernel_num, handle, sizes, deviceIdx);
    else
        benchmark_gemm<float>(kernel_num, handle, sizes, deviceIdx);

    cublasDestroy(handle);
    return EXIT_SUCCESS;
}
