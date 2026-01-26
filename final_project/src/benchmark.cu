/**
 * GEMM Benchmark
 * 
 * 对比所有版本的性能
 */

#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include "../include/gemm.h"

// 导入各版本 Kernel
#include "v1_naive.cu"
#include "v2_tiled.cu"
// TODO: 后续添加 v3-v6

// ============================================
// 工具函数
// ============================================

void init_matrix(float* mat, int rows, int cols) {
    for (int i = 0; i < rows * cols; i++) {
        mat[i] = (float)(rand() % 10) / 10.0f;
    }
}

bool verify_result(const float* C, const float* C_ref, int M, int N, float eps) {
    int errors = 0;
    for (int i = 0; i < M * N; i++) {
        if (fabs(C[i] - C_ref[i]) > eps) {
            errors++;
            if (errors <= 5) {
                printf("Mismatch at %d: got %f, expected %f\n", i, C[i], C_ref[i]);
            }
        }
    }
    return errors == 0;
}

double compute_gflops(int M, int K, int N, float time_ms) {
    double flops = 2.0 * M * N * K;
    return flops / (time_ms / 1000.0) / 1e9;
}

void print_result(const char* name, float time_ms, double gflops, double speedup) {
    printf("%-20s %8.3f ms  %8.2f GFLOPS  %6.2fx\n", name, time_ms, gflops, speedup);
}

// ============================================
// Benchmark 函数
// ============================================

float benchmark_kernel(
    void (*kernel)(const float*, const float*, float*, int, int, int),
    dim3 grid, dim3 block,
    float* d_A, float* d_B, float* d_C,
    int M, int K, int N,
    int num_runs = 10
) {
    // Warmup
    kernel<<<grid, block>>>(d_A, d_B, d_C, M, K, N);
    cudaDeviceSynchronize();
    
    // 计时
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    
    cudaEventRecord(start);
    for (int i = 0; i < num_runs; i++) {
        kernel<<<grid, block>>>(d_A, d_B, d_C, M, K, N);
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    
    float time_ms;
    cudaEventElapsedTime(&time_ms, start, stop);
    
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    
    return time_ms / num_runs;
}

float benchmark_cublas(
    cublasHandle_t handle,
    float* d_A, float* d_B, float* d_C,
    int M, int K, int N,
    int num_runs = 10
) {
    float alpha = 1.0f, beta = 0.0f;
    
    // Warmup
    cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N,
                N, M, K, &alpha, d_B, N, d_A, K, &beta, d_C, N);
    cudaDeviceSynchronize();
    
    // 计时
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    
    cudaEventRecord(start);
    for (int i = 0; i < num_runs; i++) {
        cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N,
                    N, M, K, &alpha, d_B, N, d_A, K, &beta, d_C, N);
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    
    float time_ms;
    cudaEventElapsedTime(&time_ms, start, stop);
    
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    
    return time_ms / num_runs;
}

// ============================================
// Main
// ============================================

int main() {
    const int M = MATRIX_SIZE;
    const int K = MATRIX_SIZE;
    const int N = MATRIX_SIZE;
    
    printf("╔════════════════════════════════════════════════════════════╗\n");
    printf("║           High-Performance GEMM Benchmark                  ║\n");
    printf("╠════════════════════════════════════════════════════════════╣\n");
    printf("║  Matrix Size: %4d x %4d x %4d                            ║\n", M, K, N);
    printf("╚════════════════════════════════════════════════════════════╝\n\n");
    
    size_t size_A = M * K * sizeof(float);
    size_t size_B = K * N * sizeof(float);
    size_t size_C = M * N * sizeof(float);
    
    // 分配 Host 内存
    float *h_A = (float*)malloc(size_A);
    float *h_B = (float*)malloc(size_B);
    float *h_C = (float*)malloc(size_C);
    float *h_C_ref = (float*)malloc(size_C);
    
    // 初始化
    srand(42);
    init_matrix(h_A, M, K);
    init_matrix(h_B, K, N);
    
    // 分配 Device 内存
    float *d_A, *d_B, *d_C;
    CUDA_CHECK(cudaMalloc(&d_A, size_A));
    CUDA_CHECK(cudaMalloc(&d_B, size_B));
    CUDA_CHECK(cudaMalloc(&d_C, size_C));
    
    CUDA_CHECK(cudaMemcpy(d_A, h_A, size_A, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_B, h_B, size_B, cudaMemcpyHostToDevice));
    
    // cuBLAS
    cublasHandle_t handle;
    CUBLAS_CHECK(cublasCreate(&handle));
    
    printf("Running benchmarks...\n\n");
    printf("%-20s %11s  %14s  %8s\n", "Version", "Time", "Performance", "Speedup");
    printf("────────────────────────────────────────────────────────────\n");
    
    float baseline_time = 0;
    
    // -----------------------------------------
    // V1: Naive
    // -----------------------------------------
    {
        dim3 block(16, 16);
        dim3 grid((N + 15) / 16, (M + 15) / 16);
        
        float time = benchmark_kernel(gemm_v1_naive, grid, block, d_A, d_B, d_C, M, K, N);
        double gflops = compute_gflops(M, K, N, time);
        baseline_time = time;
        
        print_result("V1 Naive", time, gflops, 1.0);
        
        // 保存参考结果
        CUDA_CHECK(cudaMemcpy(h_C_ref, d_C, size_C, cudaMemcpyDeviceToHost));
    }
    
    // -----------------------------------------
    // V2: Tiled
    // -----------------------------------------
    {
        dim3 block(TILE_SIZE, TILE_SIZE);
        dim3 grid((N + TILE_SIZE - 1) / TILE_SIZE, (M + TILE_SIZE - 1) / TILE_SIZE);
        
        float time = benchmark_kernel(gemm_v2_tiled, grid, block, d_A, d_B, d_C, M, K, N);
        double gflops = compute_gflops(M, K, N, time);
        double speedup = baseline_time / time;
        
        print_result("V2 Tiled", time, gflops, speedup);
        
        // 验证
        CUDA_CHECK(cudaMemcpy(h_C, d_C, size_C, cudaMemcpyDeviceToHost));
        if (!verify_result(h_C, h_C_ref, M, N)) {
            printf("  ⚠️  V2 result mismatch!\n");
        }
    }
    
    // TODO: V3-V6
    printf("V3 Coalescing        (TODO)\n");
    printf("V4 Vectorized        (TODO)\n");
    printf("V5 Register          (TODO)\n");
    printf("V6 DoubleBuffer      (TODO)\n");
    
    // -----------------------------------------
    // cuBLAS (参考)
    // -----------------------------------------
    {
        float time = benchmark_cublas(handle, d_A, d_B, d_C, M, K, N);
        double gflops = compute_gflops(M, K, N, time);
        double speedup = baseline_time / time;
        
        print_result("cuBLAS (reference)", time, gflops, speedup);
    }
    
    printf("────────────────────────────────────────────────────────────\n");
    
    // 清理
    free(h_A); free(h_B); free(h_C); free(h_C_ref);
    cudaFree(d_A); cudaFree(d_B); cudaFree(d_C);
    cublasDestroy(handle);
    
    printf("\nDone!\n");
    return 0;
}
