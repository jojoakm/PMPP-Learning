/**
 * Chapter 5 作业：Tiled 矩阵乘法
 * 
 * 目标：使用 Shared Memory 优化矩阵乘法
 * 
 * 编译：nvcc matmul_tiled.cu -o matmul_tiled
 * 运行：./matmul_tiled
 */

#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

#define M 1024
#define K 1024
#define N 1024

#define TILE_SIZE 16

#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            fprintf(stderr, "CUDA error at %s:%d: %s\n", \
                    __FILE__, __LINE__, cudaGetErrorString(err)); \
            exit(EXIT_FAILURE); \
        } \
    } while(0)

// ===========================================
// Kernel: Tiled 矩阵乘法
// ===========================================
__global__ void matmul_tiled(float *A, float *B, float *C,
                              int m, int k, int n) {
    // Shared Memory 用于存储 Tile
    __shared__ float As[TILE_SIZE][TILE_SIZE];
    __shared__ float Bs[TILE_SIZE][TILE_SIZE];
    
    // 计算当前线程负责的 C 元素位置
    int row = blockIdx.y * TILE_SIZE + threadIdx.y;
    int col = blockIdx.x * TILE_SIZE + threadIdx.x;
    
    float sum = 0.0f;
    
    // 计算需要多少个 Tile
    int numTiles = (k + TILE_SIZE - 1) / TILE_SIZE;
    
    // 遍历所有 Tile
    for (int t = 0; t < numTiles; t++) {
        // -----------------------------------------
        // 1. 加载 A 的 Tile 到 Shared Memory
        // -----------------------------------------
        int aCol = t * TILE_SIZE + threadIdx.x;
        if (row < m && aCol < k) {
            As[threadIdx.y][threadIdx.x] = A[row * k + aCol];
        } else {
            As[threadIdx.y][threadIdx.x] = 0.0f;
        }
        
        // -----------------------------------------
        // 2. 加载 B 的 Tile 到 Shared Memory
        // -----------------------------------------
        int bRow = t * TILE_SIZE + threadIdx.y;
        if (bRow < k && col < n) {
            Bs[threadIdx.y][threadIdx.x] = B[bRow * n + col];
        } else {
            Bs[threadIdx.y][threadIdx.x] = 0.0f;
        }
        
        // -----------------------------------------
        // 3. 同步：确保 Tile 加载完成
        // -----------------------------------------
        __syncthreads();
        
        // -----------------------------------------
        // 4. 计算部分结果
        // -----------------------------------------
        for (int i = 0; i < TILE_SIZE; i++) {
            sum += As[threadIdx.y][i] * Bs[i][threadIdx.x];
        }
        
        // -----------------------------------------
        // 5. 同步：确保计算完成后再加载下一个 Tile
        // -----------------------------------------
        __syncthreads();
    }
    
    // -----------------------------------------
    // 6. 写回结果
    // -----------------------------------------
    if (row < m && col < n) {
        C[row * n + col] = sum;
    }
}

// ===========================================
// Naive 版本（用于对比）
// ===========================================
__global__ void matmul_naive(float *A, float *B, float *C,
                              int m, int k, int n) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (row < m && col < n) {
        float sum = 0.0f;
        for (int i = 0; i < k; i++) {
            sum += A[row * k + i] * B[i * n + col];
        }
        C[row * n + col] = sum;
    }
}

// ===========================================
// 主函数
// ===========================================
int main() {
    printf("=== 矩阵乘法：Naive vs Tiled ===\n");
    printf("矩阵大小: %d x %d x %d\n", M, K, N);
    printf("Tile 大小: %d x %d\n\n", TILE_SIZE, TILE_SIZE);
    
    size_t size_A = M * K * sizeof(float);
    size_t size_B = K * N * sizeof(float);
    size_t size_C = M * N * sizeof(float);
    
    // 分配内存
    float *h_A = (float*)malloc(size_A);
    float *h_B = (float*)malloc(size_B);
    float *h_C_naive = (float*)malloc(size_C);
    float *h_C_tiled = (float*)malloc(size_C);
    
    // 初始化
    for (int i = 0; i < M * K; i++) h_A[i] = (float)(rand() % 10) / 10.0f;
    for (int i = 0; i < K * N; i++) h_B[i] = (float)(rand() % 10) / 10.0f;
    
    // 分配 Device 内存
    float *d_A, *d_B, *d_C;
    CUDA_CHECK(cudaMalloc(&d_A, size_A));
    CUDA_CHECK(cudaMalloc(&d_B, size_B));
    CUDA_CHECK(cudaMalloc(&d_C, size_C));
    
    CUDA_CHECK(cudaMemcpy(d_A, h_A, size_A, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_B, h_B, size_B, cudaMemcpyHostToDevice));
    
    dim3 block(TILE_SIZE, TILE_SIZE);
    dim3 grid((N + TILE_SIZE - 1) / TILE_SIZE,
              (M + TILE_SIZE - 1) / TILE_SIZE);
    
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    
    // -----------------------------------------
    // 测试 Naive 版本
    // -----------------------------------------
    cudaEventRecord(start);
    matmul_naive<<<grid, block>>>(d_A, d_B, d_C, M, K, N);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    
    float naive_time;
    cudaEventElapsedTime(&naive_time, start, stop);
    CUDA_CHECK(cudaMemcpy(h_C_naive, d_C, size_C, cudaMemcpyDeviceToHost));
    
    double naive_gflops = 2.0 * M * N * K / (naive_time / 1000.0) / 1e9;
    printf("Naive:  %.3f ms, %.2f GFLOPS\n", naive_time, naive_gflops);
    
    // -----------------------------------------
    // 测试 Tiled 版本
    // -----------------------------------------
    cudaEventRecord(start);
    matmul_tiled<<<grid, block>>>(d_A, d_B, d_C, M, K, N);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    
    float tiled_time;
    cudaEventElapsedTime(&tiled_time, start, stop);
    CUDA_CHECK(cudaMemcpy(h_C_tiled, d_C, size_C, cudaMemcpyDeviceToHost));
    
    double tiled_gflops = 2.0 * M * N * K / (tiled_time / 1000.0) / 1e9;
    printf("Tiled:  %.3f ms, %.2f GFLOPS\n", tiled_time, tiled_gflops);
    
    // -----------------------------------------
    // 性能对比
    // -----------------------------------------
    printf("\n加速比: %.2fx\n", naive_time / tiled_time);
    
    // -----------------------------------------
    // 验证正确性
    // -----------------------------------------
    int errors = 0;
    for (int i = 0; i < M * N; i++) {
        if (fabs(h_C_naive[i] - h_C_tiled[i]) > 1e-3) {
            errors++;
        }
    }
    if (errors == 0) {
        printf("✅ 结果验证通过！\n");
    } else {
        printf("❌ 结果不一致！%d 个错误\n", errors);
    }
    
    // 释放内存
    free(h_A); free(h_B); free(h_C_naive); free(h_C_tiled);
    cudaFree(d_A); cudaFree(d_B); cudaFree(d_C);
    cudaEventDestroy(start); cudaEventDestroy(stop);
    
    return 0;
}

// ===========================================
// 为什么 Tiled 更快？
// ===========================================
//
// Naive 版本：
// - 每个线程读取 2K 次 Global Memory
// - 总共 M*N 个线程，读取 2*M*N*K 次
//
// Tiled 版本：
// - 每个 Tile 只从 Global Memory 读取一次
// - 在 Shared Memory 中被复用 TILE_SIZE 次
// - Global Memory 读取减少到 2*M*N*K / TILE_SIZE
//
// 加速比理论值 ≈ TILE_SIZE = 16
// 实际加速比通常 3-5x（受其他因素影响）
