/**
 * Chapter 4 作业：矩阵乘法（Naive 版本）
 * 
 * 目标：理解 GPU 内存层次，实现基础矩阵乘法
 * 
 * 编译：nvcc matmul_naive.cu -o matmul_naive
 * 运行：./matmul_naive
 */

#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

// 矩阵大小
#define M 1024  // A: M x K
#define K 1024  // B: K x N
#define N 1024  // C: M x N

#define BLOCK_SIZE 16  // 16x16 threads per block

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

// ===========================================
// Kernel: Naive 矩阵乘法
// ===========================================
// 
// 每个线程计算 C 的一个元素
// C[row][col] = Σ A[row][k] * B[k][col]
//
__global__ void matmul_naive(float *A, float *B, float *C, 
                              int m, int k, int n) {
    // 计算当前线程负责的位置
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    
    // 边界检查
    if (row < m && col < n) {
        float sum = 0.0f;
        
        // 计算点积
        for (int i = 0; i < k; i++) {
            // A[row][i] * B[i][col]
            // A 是 row-major: A[row][i] = A[row * K + i]
            // B 是 row-major: B[i][col] = B[i * N + col]
            sum += A[row * k + i] * B[i * n + col];
        }
        
        C[row * n + col] = sum;
    }
}

// ===========================================
// CPU 版本（用于验证）
// ===========================================
void matmul_cpu(float *A, float *B, float *C, int m, int k, int n) {
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j++) {
            float sum = 0.0f;
            for (int l = 0; l < k; l++) {
                sum += A[i * k + l] * B[l * n + j];
            }
            C[i * n + j] = sum;
        }
    }
}

// ===========================================
// 主函数
// ===========================================
int main() {
    printf("=== 矩阵乘法 Naive ===\n");
    printf("A: %d x %d\n", M, K);
    printf("B: %d x %d\n", K, N);
    printf("C: %d x %d\n\n", M, N);
    
    // 内存大小
    size_t size_A = M * K * sizeof(float);
    size_t size_B = K * N * sizeof(float);
    size_t size_C = M * N * sizeof(float);
    
    // -----------------------------------------
    // 1. 分配 Host 内存
    // -----------------------------------------
    float *h_A = (float*)malloc(size_A);
    float *h_B = (float*)malloc(size_B);
    float *h_C = (float*)malloc(size_C);
    float *h_C_ref = (float*)malloc(size_C);  // CPU 参考结果
    
    // -----------------------------------------
    // 2. 初始化数据
    // -----------------------------------------
    for (int i = 0; i < M * K; i++) h_A[i] = (float)(rand() % 10) / 10.0f;
    for (int i = 0; i < K * N; i++) h_B[i] = (float)(rand() % 10) / 10.0f;
    
    // -----------------------------------------
    // 3. 分配 Device 内存
    // -----------------------------------------
    float *d_A, *d_B, *d_C;
    CUDA_CHECK(cudaMalloc(&d_A, size_A));
    CUDA_CHECK(cudaMalloc(&d_B, size_B));
    CUDA_CHECK(cudaMalloc(&d_C, size_C));
    
    // -----------------------------------------
    // 4. 拷贝数据到 Device
    // -----------------------------------------
    CUDA_CHECK(cudaMemcpy(d_A, h_A, size_A, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_B, h_B, size_B, cudaMemcpyHostToDevice));
    
    // -----------------------------------------
    // 5. 配置 Grid 和 Block
    // -----------------------------------------
    dim3 block(BLOCK_SIZE, BLOCK_SIZE);  // 16x16 = 256 threads
    dim3 grid((N + BLOCK_SIZE - 1) / BLOCK_SIZE, 
              (M + BLOCK_SIZE - 1) / BLOCK_SIZE);
    
    printf("Block: %d x %d\n", block.x, block.y);
    printf("Grid: %d x %d\n\n", grid.x, grid.y);
    
    // -----------------------------------------
    // 6. 计时并执行 Kernel
    // -----------------------------------------
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    
    cudaEventRecord(start);
    matmul_naive<<<grid, block>>>(d_A, d_B, d_C, M, K, N);
    cudaEventRecord(stop);
    
    CUDA_CHECK(cudaGetLastError());
    cudaEventSynchronize(stop);
    
    float gpu_time;
    cudaEventElapsedTime(&gpu_time, start, stop);
    printf("GPU 时间: %.3f ms\n", gpu_time);
    
    // -----------------------------------------
    // 7. 拷贝结果回 Host
    // -----------------------------------------
    CUDA_CHECK(cudaMemcpy(h_C, d_C, size_C, cudaMemcpyDeviceToHost));
    
    // -----------------------------------------
    // 8. CPU 计算（用于验证，可选）
    // -----------------------------------------
    #ifdef VERIFY
    printf("CPU 计算中（用于验证）...\n");
    matmul_cpu(h_A, h_B, h_C_ref, M, K, N);
    
    // 验证
    int errors = 0;
    for (int i = 0; i < M * N; i++) {
        if (fabs(h_C[i] - h_C_ref[i]) > 1e-3) {
            errors++;
        }
    }
    if (errors == 0) {
        printf("✅ 验证通过！\n");
    } else {
        printf("❌ 验证失败！%d 个错误\n", errors);
    }
    #endif
    
    // -----------------------------------------
    // 9. 计算性能指标
    // -----------------------------------------
    // 矩阵乘法的计算量：2 * M * N * K（乘法+加法）
    double flops = 2.0 * M * N * K;
    double gflops = flops / (gpu_time / 1000.0) / 1e9;
    printf("性能: %.2f GFLOPS\n", gflops);
    
    // -----------------------------------------
    // 10. 释放内存
    // -----------------------------------------
    free(h_A); free(h_B); free(h_C); free(h_C_ref);
    cudaFree(d_A); cudaFree(d_B); cudaFree(d_C);
    cudaEventDestroy(start); cudaEventDestroy(stop);
    
    printf("\n完成！\n");
    return 0;
}

// ===========================================
// 分析：为什么 Naive 版本慢？
// ===========================================
//
// 每个线程计算 C[row][col] 需要：
// - 读取 A[row][0..K-1]：K 次 Global Memory 读取
// - 读取 B[0..K-1][col]：K 次 Global Memory 读取
// - 总共 2K 次 Global Memory 访问
//
// 问题 1：重复读取
// - 同一行的线程都需要读取 A 的同一行
// - 同一列的线程都需要读取 B 的同一列
// - 大量数据被重复读取！
//
// 问题 2：内存带宽瓶颈
// - Global Memory 带宽有限
// - 计算量/内存访问量 = 2MNK / 2MNK = 1
// - 这是 memory-bound 的！
//
// 解决方案：使用 Shared Memory Tiling（下一章）
// - 把数据块加载到 Shared Memory
// - 在 Shared Memory 中重用数据
// - 减少 Global Memory 访问次数
