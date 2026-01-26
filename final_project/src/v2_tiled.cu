/**
 * V2: Tiled GEMM with Shared Memory
 * 
 * 使用 Shared Memory 分块，减少 Global Memory 访问
 * 
 * 优化点：
 * - 将数据块加载到 Shared Memory
 * - 在 Shared Memory 中复用数据
 * - 减少 Global Memory 访问 TILE_SIZE 倍
 */

#include <stdio.h>
#include <cuda_runtime.h>
#include "../include/gemm.h"

#define TILE_SIZE 16

// ============================================
// V2 Kernel: Shared Memory Tiling
// ============================================
__global__ void gemm_v2_tiled(
    const float* __restrict__ A,
    const float* __restrict__ B,
    float* __restrict__ C,
    int M, int K, int N
) {
    // Shared Memory
    __shared__ float As[TILE_SIZE][TILE_SIZE];
    __shared__ float Bs[TILE_SIZE][TILE_SIZE];
    
    // 线程位置
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    
    // C 元素位置
    int row = blockIdx.y * TILE_SIZE + ty;
    int col = blockIdx.x * TILE_SIZE + tx;
    
    float sum = 0.0f;
    
    // 遍历所有 Tile
    int numTiles = (K + TILE_SIZE - 1) / TILE_SIZE;
    
    for (int t = 0; t < numTiles; t++) {
        // 加载 A 的 Tile
        int aCol = t * TILE_SIZE + tx;
        if (row < M && aCol < K) {
            As[ty][tx] = A[row * K + aCol];
        } else {
            As[ty][tx] = 0.0f;
        }
        
        // 加载 B 的 Tile
        int bRow = t * TILE_SIZE + ty;
        if (bRow < K && col < N) {
            Bs[ty][tx] = B[bRow * N + col];
        } else {
            Bs[ty][tx] = 0.0f;
        }
        
        // 同步：确保 Tile 加载完成
        __syncthreads();
        
        // 计算部分结果
        #pragma unroll
        for (int k = 0; k < TILE_SIZE; k++) {
            sum += As[ty][k] * Bs[k][tx];
        }
        
        // 同步：确保计算完成
        __syncthreads();
    }
    
    // 写回结果
    if (row < M && col < N) {
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
    
    printf("=== V2 Tiled GEMM Test ===\n");
    printf("Matrix: %d x %d x %d\n", M, K, N);
    printf("Tile Size: %d x %d\n\n", TILE_SIZE, TILE_SIZE);
    
    size_t size_A = M * K * sizeof(float);
    size_t size_B = K * N * sizeof(float);
    size_t size_C = M * N * sizeof(float);
    
    float *h_A = (float*)malloc(size_A);
    float *h_B = (float*)malloc(size_B);
    float *h_C = (float*)malloc(size_C);
    
    for (int i = 0; i < M * K; i++) h_A[i] = (float)(rand() % 10) / 10.0f;
    for (int i = 0; i < K * N; i++) h_B[i] = (float)(rand() % 10) / 10.0f;
    
    float *d_A, *d_B, *d_C;
    CUDA_CHECK(cudaMalloc(&d_A, size_A));
    CUDA_CHECK(cudaMalloc(&d_B, size_B));
    CUDA_CHECK(cudaMalloc(&d_C, size_C));
    
    CUDA_CHECK(cudaMemcpy(d_A, h_A, size_A, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_B, h_B, size_B, cudaMemcpyHostToDevice));
    
    dim3 block(TILE_SIZE, TILE_SIZE);
    dim3 grid((N + TILE_SIZE - 1) / TILE_SIZE, (M + TILE_SIZE - 1) / TILE_SIZE);
    
    // Warmup
    gemm_v2_tiled<<<grid, block>>>(d_A, d_B, d_C, M, K, N);
    cudaDeviceSynchronize();
    
    // 计时
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    
    cudaEventRecord(start);
    for (int i = 0; i < 10; i++) {
        gemm_v2_tiled<<<grid, block>>>(d_A, d_B, d_C, M, K, N);
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    
    float time_ms;
    cudaEventElapsedTime(&time_ms, start, stop);
    time_ms /= 10;
    
    double gflops = 2.0 * M * N * K / (time_ms / 1000.0) / 1e9;
    printf("Time: %.3f ms\n", time_ms);
    printf("Performance: %.2f GFLOPS\n", gflops);
    
    free(h_A); free(h_B); free(h_C);
    cudaFree(d_A); cudaFree(d_B); cudaFree(d_C);
    
    return 0;
}

#endif

// ============================================
// 分析
// ============================================
// 
// 优化效果：
// - Global Memory 访问减少 TILE_SIZE 倍
// - 数据在 Shared Memory 中被复用
// 
// 剩余问题：
// - 可能存在 Bank Conflict
// - 内存访问模式可以进一步优化
// - 每个线程只计算一个元素，寄存器利用不充分
// 
// 下一步优化：
// - V3: 优化内存访问模式
// - V4: 使用 float4 向量化
// - V5: 寄存器分块
