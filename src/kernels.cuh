#pragma once

#include "kernels/10_kernel_warptiling.cuh"
#include "kernels/11_kernel_double_buffering.cuh"
#include "kernels/12_kernel_double_buffering.cuh"
#include "kernels/1_naive.cuh"
#include "kernels/2_kernel_global_mem_coalesce.cuh"
#include "kernels/3_kernel_shared_mem_blocking.cuh"
#include "kernels/4_kernel_1D_blocktiling.cuh"
#include "kernels/5_kernel_2D_blocktiling.cuh"
#include "kernels/6_kernel_vectorize.cuh"
#include "kernels/7_kernel_resolve_bank_conflicts.cuh"
#include "kernels/8_kernel_bank_extra_col.cuh"
#include "kernels/9_kernel_autotuned.cuh"
#include "kernels/9_kernel_autotuned.cuh"
#include "tensorCoreKernels/tensor_core_kernel.cuh"
#include "tensorCoreKernels/simple_tensor_core.cuh"
#include "tensorCoreKernels/tensor_core_kernel2.cuh"
#include "tensorCoreKernels/simple_tensor_core_multiwarp.cuh"
#include "tensorCoreKernels/vectorized_tensor_core_kernel.cuh"
#include "tensorCoreKernels/vectorized_tensor_core_kernel2.cuh"
#include "tensorCoreKernels/tensor_core_double_buffering.cuh"
#include "tensorCoreKernels/tensor_core_double_buffering_stride3.cuh"
#include "tensorCoreKernels/tensor_core_resolve_bank_conflicts.cuh"
#include "tensorCoreKernels/tensor_core_resolving_stalls.cuh"
#include "tensorCoreKernels/tensor_core_resolving_stalls2.cuh"
#include "tensorCoreKernels/tensor_core_mma.cuh"
