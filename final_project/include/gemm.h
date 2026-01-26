/**
 * High-Performance GEMM 头文件
 */

#ifndef GEMM_H
#define GEMM_H

#include <cuda_runtime.h>
#include <cublas_v2.h>

// 矩阵大小（可配置）
#ifndef MATRIX_SIZE
#define MATRIX_SIZE 4096
#endif

// Tile 大小
#define TILE_SIZE_V2 16
#define TILE_SIZE_V3 32
#define TILE_SIZE_V5 64

// CUDA 错误检查
#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            fprintf(stderr, "CUDA error at %s:%d: %s\n", \
                    __FILE__, __LINE__, cudaGetErrorString(err)); \
            exit(EXIT_FAILURE); \
        } \
    } while(0)

#define CUBLAS_CHECK(call) \
    do { \
        cublasStatus_t status = call; \
        if (status != CUBLAS_STATUS_SUCCESS) { \
            fprintf(stderr, "cuBLAS error at %s:%d\n", \
                    __FILE__, __LINE__); \
            exit(EXIT_FAILURE); \
        } \
    } while(0)

// ============================================
// Kernel 声明
// ============================================

// V1: Naive
__global__ void gemm_v1_naive(
    const float* A, const float* B, float* C,
    int M, int K, int N
);

// V2: Shared Memory Tiling
__global__ void gemm_v2_tiled(
    const float* A, const float* B, float* C,
    int M, int K, int N
);

// V3: Memory Coalescing 优化
__global__ void gemm_v3_coalescing(
    const float* A, const float* B, float* C,
    int M, int K, int N
);

// V4: 向量化访存 (float4)
__global__ void gemm_v4_vectorized(
    const float* A, const float* B, float* C,
    int M, int K, int N
);

// V5: 寄存器分块
__global__ void gemm_v5_register(
    const float* A, const float* B, float* C,
    int M, int K, int N
);

// V6: Double Buffering
__global__ void gemm_v6_double_buffer(
    const float* A, const float* B, float* C,
    int M, int K, int N
);

// ============================================
// 工具函数
// ============================================

// 初始化矩阵
void init_matrix(float* mat, int rows, int cols);

// 验证结果
bool verify_result(const float* C, const float* C_ref, int M, int N, float eps = 1e-3);

// 计算 GFLOPS
double compute_gflops(int M, int K, int N, float time_ms);

// 打印性能结果
void print_result(const char* name, float time_ms, double gflops, double speedup);

#endif // GEMM_H
