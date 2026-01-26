/**
 * Chapter 2 作业：Hello CUDA
 * 
 * 目标：理解 CUDA 程序的基本结构
 * 
 * 编译：nvcc hello_cuda.cu -o hello_cuda
 * 运行：./hello_cuda
 */

#include <stdio.h>

// ===========================================
// Kernel 函数（在 GPU 上执行）
// ===========================================

// __global__ 表示这是一个 Kernel 函数
// - 从 Host（CPU）调用
// - 在 Device（GPU）上执行
__global__ void helloKernel() {
    // threadIdx.x 是当前线程在 block 内的索引
    int tid = threadIdx.x;
    printf("Hello from GPU! Thread %d\n", tid);
}

// ===========================================
// 主函数（在 CPU 上执行）
// ===========================================
int main() {
    printf("=== Hello CUDA ===\n\n");
    
    // 启动 Kernel
    // <<<numBlocks, threadsPerBlock>>>
    // <<<1, 8>>> = 1 个 block，每个 block 8 个线程
    helloKernel<<<1, 8>>>();
    
    // 等待 GPU 完成
    // GPU 是异步执行的，需要同步等待
    cudaDeviceSynchronize();
    
    printf("\nHello from CPU!\n");
    
    return 0;
}

// ===========================================
// 练习：修改代码，尝试以下变化
// ===========================================
// 
// 1. 把 <<<1, 8>>> 改成 <<<2, 4>>>
//    观察输出有什么变化？
//    提示：现在有 2 个 block，每个 4 个线程
//
// 2. 在 Kernel 中打印 blockIdx.x
//    printf("Block %d, Thread %d\n", blockIdx.x, threadIdx.x);
//
// 3. 计算全局线程 ID
//    int globalId = blockIdx.x * blockDim.x + threadIdx.x;
//    blockDim.x 是每个 block 的线程数
