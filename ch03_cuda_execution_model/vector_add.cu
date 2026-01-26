/**
 * Chapter 3 作业：向量加法
 * 
 * 目标：理解 Grid/Block/Thread 层次结构
 * 
 * 编译：nvcc vector_add.cu -o vector_add
 * 运行：./vector_add
 */

#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

#define N 10000       // 向量长度
#define BLOCK_SIZE 256  // 每个 Block 的线程数

// ===========================================
// CUDA 错误检查宏
// ===========================================
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
// Kernel: 向量加法
// ===========================================
__global__ void vectorAdd(float *A, float *B, float *C, int n) {
    // TODO: 实现向量加法
    // 
    // 步骤：
    // 1. 计算全局线程 ID
    //    int i = blockIdx.x * blockDim.x + threadIdx.x;
    // 
    // 2. 边界检查
    //    if (i < n) { ... }
    // 
    // 3. 计算
    //    C[i] = A[i] + B[i];
    
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (i < n) {
        C[i] = A[i] + B[i];
    }
}

// ===========================================
// 主函数
// ===========================================
int main() {
    printf("=== 向量加法 (N = %d) ===\n\n", N);
    
    // 计算所需内存大小
    size_t size = N * sizeof(float);
    
    // -----------------------------------------
    // 1. 分配 Host 内存
    // -----------------------------------------
    float *h_A = (float*)malloc(size);
    float *h_B = (float*)malloc(size);
    float *h_C = (float*)malloc(size);
    
    // -----------------------------------------
    // 2. 初始化数据
    // -----------------------------------------
    for (int i = 0; i < N; i++) {
        h_A[i] = (float)i;
        h_B[i] = (float)(i * 2);
    }
    
    // -----------------------------------------
    // 3. 分配 Device 内存
    // -----------------------------------------
    float *d_A, *d_B, *d_C;
    CUDA_CHECK(cudaMalloc(&d_A, size));
    CUDA_CHECK(cudaMalloc(&d_B, size));
    CUDA_CHECK(cudaMalloc(&d_C, size));
    
    // -----------------------------------------
    // 4. 拷贝数据到 Device
    // -----------------------------------------
    CUDA_CHECK(cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice));
    
    // -----------------------------------------
    // 5. 计算 Grid 和 Block 大小
    // -----------------------------------------
    int numBlocks = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;  // 向上取整
    printf("Grid size: %d blocks\n", numBlocks);
    printf("Block size: %d threads\n", BLOCK_SIZE);
    printf("Total threads: %d\n\n", numBlocks * BLOCK_SIZE);
    
    // -----------------------------------------
    // 6. 启动 Kernel
    // -----------------------------------------
    printf("启动 Kernel...\n");
    vectorAdd<<<numBlocks, BLOCK_SIZE>>>(d_A, d_B, d_C, N);
    CUDA_CHECK(cudaGetLastError());  // 检查 Kernel 启动错误
    
    // -----------------------------------------
    // 7. 等待 GPU 完成
    // -----------------------------------------
    CUDA_CHECK(cudaDeviceSynchronize());
    
    // -----------------------------------------
    // 8. 拷贝结果回 Host
    // -----------------------------------------
    CUDA_CHECK(cudaMemcpy(h_C, d_C, size, cudaMemcpyDeviceToHost));
    
    // -----------------------------------------
    // 9. 验证结果
    // -----------------------------------------
    printf("验证结果...\n");
    int errors = 0;
    for (int i = 0; i < N; i++) {
        float expected = h_A[i] + h_B[i];
        if (fabs(h_C[i] - expected) > 1e-5) {
            if (errors < 10) {  // 只打印前10个错误
                printf("错误: C[%d] = %f, 期望 %f\n", i, h_C[i], expected);
            }
            errors++;
        }
    }
    
    if (errors == 0) {
        printf("✅ 验证通过！所有 %d 个元素正确\n", N);
    } else {
        printf("❌ 验证失败！%d 个错误\n", errors);
    }
    
    // -----------------------------------------
    // 10. 释放内存
    // -----------------------------------------
    free(h_A);
    free(h_B);
    free(h_C);
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    
    printf("\n完成！\n");
    return 0;
}

// ===========================================
// 思考题
// ===========================================
// 
// 1. 如果 N = 1000，BLOCK_SIZE = 256，需要多少个 Block？
//    答：(1000 + 255) / 256 = 4 个 Block
//
// 2. 最后一个 Block 有多少个线程是"浪费"的？
//    答：4 * 256 - 1000 = 24 个线程
//
// 3. 为什么需要边界检查 if (i < n)？
//    答：因为总线程数可能大于 N，多余的线程不应该访问数组
//
// 4. 如果去掉边界检查会发生什么？
//    答：可能访问非法内存，导致程序崩溃或结果错误
