#pragma once
#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <fstream>
#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include <time.h>
#include <unistd.h>

void cudaCheck(cudaError_t error, const char *file,
               int line); // CUDA error check
void CudaDeviceInfo();    // print CUDA information

void range_init_matrix(float *mat, int N);
void randomize_matrix(float *mat, int N);
void randomize_matrix_hf(__half *mat, int N);
void initialize_one_hf(__half *mat, int N);
void initialize_incremental_float(float *mat, int N);
void initialize_identity_hf(__half *mat, int N);
void initialize_incremental_hf(__half *mat, int N);
void initialize_one_float(float *mat, int N);
void zero_init_matrix(float *mat, int N);
void copy_matrix(const float *src, float *dest, int N);
void print_matrix(const float *A, int M, int N, std::ofstream &fs);
void print_matrix_hf(const __half *A, int M, int N, std::ofstream &fs);
bool verify_matrix(float *mat1, float *mat2, int N);
bool verify_matrix_hf(__half *mat1, __half *mat2, int N);
float get_current_sec();                        // Get the current moment
float cpu_elapsed_time(float &beg, float &end); // Calculate time difference
void float_array_to_half(__half * half_mat, float * float_mat, int size);
void half_array_to_float(__half * half_mat, float * float_mat, int size);
void run_kernel(int kernel_num, int m, int n, int k, float alpha, float *A,
                float *B, float beta, float *C, cublasHandle_t handle);

void run_tensor_core_kernel(int kernel_num, int M, int N, int K, float alpha, __half *A,
                __half *B, float beta, float *C, cublasHandle_t handle);