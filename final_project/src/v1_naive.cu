/**
 * V1: Naive GEMM Implementation
 * 
 * 每个线程计算 C 的一个元素
 * 
 * 问题：
 * - 大量重复读取 Global Memory
 * - 没有利用数据局部性
 * - Memory Bound
 */

#include <stdio.h>
#include <cuda_runtime.h>
#include "../include/gemm.h"

// ============================================
// V1 Kernel: Naive Implementation
// ============================================
__global__ void gemm_v1_naive(
    const float* __restrict__ A,
    const float* __restrict__ B,
    float* __restrict__ C,
    int M, int K, int N
) {
    // 计算当前线程负责的 C 元素位置
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    
    // 边界检查
    if (row < M && col < N) {
        float sum = 0.0f;
        
        // 计算 C[row][col] = sum(A[row][k] * B[k][col])
        for (int k = 0; k < K; k++) {
            sum += A[row * K + k] * B[k * N + col];
        }
        
        C[row * N + col] = sum;
    }
}

// ============================================
// 测试代码
// ============================================
#ifdef TEST_MAIN

int main() {
    const int M = 1024;
    const int K = 1024;
    const int N = 1024;
    
    printf("=== V1 Naive GEMM Test ===\n");
    printf("Matrix: %d x %d x %d\n\n", M, K, N);
    
    size_t size_A = M * K * sizeof(float);
    size_t size_B = K * N * sizeof(float);
    size_t size_C = M * N * sizeof(float);
    
    // Host 内存
    float *h_A = (float*)malloc(size_A);
    float *h_B = (float*)malloc(size_B);
    float *h_C = (float*)malloc(size_C);
    
    // 初始化
    for (int i = 0; i < M * K; i++) h_A[i] = (float)(rand() % 10) / 10.0f;
    for (int i = 0; i < K * N; i++) h_B[i] = (float)(rand() % 10) / 10.0f;
    
    // Device 内存
    float *d_A, *d_B, *d_C;
    CUDA_CHECK(cudaMalloc(&d_A, size_A));
    CUDA_CHECK(cudaMalloc(&d_B, size_B));
    CUDA_CHECK(cudaMalloc(&d_C, size_C));
    
    CUDA_CHECK(cudaMemcpy(d_A, h_A, size_A, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_B, h_B, size_B, cudaMemcpyHostToDevice));
    
    // 配置
    dim3 block(16, 16);
    dim3 grid((N + 15) / 16, (M + 15) / 16);
    
    // Warmup
    gemm_v1_naive<<<grid, block>>>(d_A, d_B, d_C, M, K, N);
    cudaDeviceSynchronize();
    
    // 计时
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    
    cudaEventRecord(start);
    for (int i = 0; i < 10; i++) {
        gemm_v1_naive<<<grid, block>>>(d_A, d_B, d_C, M, K, N);
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    
    float time_ms;
    cudaEventElapsedTime(&time_ms, start, stop);
    time_ms /= 10;  // 平均
    
    double gflops = 2.0 * M * N * K / (time_ms / 1000.0) / 1e9;
    printf("Time: %.3f ms\n", time_ms);
    printf("Performance: %.2f GFLOPS\n", gflops);
    
    // 清理
    free(h_A); free(h_B); free(h_C);
    cudaFree(d_A); cudaFree(d_B); cudaFree(d_C);
    
    return 0;
}

#endif

// ============================================
// 分析
// ============================================
// 
// 性能瓶颈：Memory Bound
// 
// 每个线程：
// - 读取 A 的一行：K 次 Global Memory 读取
// - 读取 B 的一列：K 次 Global Memory 读取
// - 共 2K 次 Global Memory 访问
// 
// 总访问量：M * N * 2K
// 计算量：M * N * 2K（乘加）
// 计算/访存比 = 1（太低！）
// 
// 优化方向：
// 1. 使用 Shared Memory 复用数据 → V2
// 2. 优化内存访问模式 → V3
