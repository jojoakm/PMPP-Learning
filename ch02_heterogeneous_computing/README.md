# Chapter 2: Heterogeneous Data Parallel Computing

## 📖 本章内容

- 异构计算概念
- 数据并行 vs 任务并行
- CUDA 程序结构
- 第一个 CUDA 程序

---

## 🎯 学习目标

读完本章，你应该能：

- [ ] 理解什么是异构计算
- [ ] 区分数据并行和任务并行
- [ ] 理解 Host（CPU）和 Device（GPU）的关系
- [ ] 写出第一个 CUDA 程序

---

## 📝 核心概念

### 异构计算

```
┌─────────────────────────────────────────────────┐
│                  你的程序                        │
├─────────────────────────────────────────────────┤
│  CPU (Host)              GPU (Device)           │
│  ├── 串行代码            ├── 并行计算            │
│  ├── 复杂逻辑            ├── Kernel 函数         │
│  └── 内存管理            └── 大量线程            │
└─────────────────────────────────────────────────┘
```

### 数据并行 vs 任务并行

| 类型 | 说明 | 例子 |
|------|------|------|
| 数据并行 | 同一操作应用到不同数据 | 向量加法、图像处理 |
| 任务并行 | 不同操作同时执行 | 流水线 |

GPU 擅长**数据并行**！

### CUDA 程序结构

```cpp
// 1. Host 代码（CPU）
int main() {
    // 分配 GPU 内存
    cudaMalloc(...);
    
    // 拷贝数据到 GPU
    cudaMemcpy(..., cudaMemcpyHostToDevice);
    
    // 启动 Kernel
    myKernel<<<blocks, threads>>>(...);
    
    // 拷贝结果回 CPU
    cudaMemcpy(..., cudaMemcpyDeviceToHost);
    
    // 释放内存
    cudaFree(...);
}

// 2. Device 代码（GPU）
__global__ void myKernel(...) {
    // 每个线程执行这里的代码
}
```

---

## ✅ 本章作业：Hello CUDA

### 作业目标
写一个简单的 CUDA 程序，让 GPU 打印 "Hello from GPU!"

### 代码模板

文件：`hello_cuda.cu`

```cuda
#include <stdio.h>

// TODO: 写一个 Kernel，打印线程 ID
__global__ void helloKernel() {
    // 提示：使用 threadIdx.x 获取线程 ID
    printf("Hello from GPU! Thread %d\n", threadIdx.x);
}

int main() {
    // TODO: 启动 Kernel，使用 8 个线程
    // 提示：<<<1, 8>>> 表示 1 个 block，8 个 threads
    
    // TODO: 等待 GPU 完成
    // 提示：cudaDeviceSynchronize();
    
    printf("Hello from CPU!\n");
    return 0;
}
```

### 编译运行

```bash
nvcc hello_cuda.cu -o hello_cuda
./hello_cuda
```

### 预期输出

```
Hello from GPU! Thread 0
Hello from GPU! Thread 1
Hello from GPU! Thread 2
...
Hello from GPU! Thread 7
Hello from CPU!
```

---

## 🎯 检查点

完成后，确保你能回答：

- [ ] `__global__` 是什么意思？
- [ ] `<<<1, 8>>>` 中的两个数字分别代表什么？
- [ ] `threadIdx.x` 是什么？
- [ ] 为什么需要 `cudaDeviceSynchronize()`？

---

**完成后继续 Chapter 3！**
