# Chapter 3: CUDA Execution Model ⭐重点章节

## 📖 本章内容

- Grid, Block, Thread 层次结构
- 线程索引计算
- 多维 Grid 和 Block
- 向量加法实现

---

## 🎯 学习目标

读完本章，你应该能：

- [ ] 画出 Grid → Block → Thread 的层次图
- [ ] 计算任意线程的全局索引
- [ ] 理解 blockDim, gridDim, blockIdx, threadIdx
- [ ] 实现向量加法

---

## 📝 核心概念

### 线程层次结构

```
                    Grid（网格）
                        │
         ┌──────────────┼──────────────┐
         │              │              │
      Block 0        Block 1        Block 2
         │              │              │
    ┌────┼────┐    ┌────┼────┐    ┌────┼────┐
    │    │    │    │    │    │    │    │    │
   T0   T1   T2   T0   T1   T2   T0   T1   T2
   
   每个 Block 内的线程可以协作（共享内存、同步）
   不同 Block 之间相互独立
```

### 关键变量

| 变量 | 含义 | 类型 |
|------|------|------|
| `gridDim.x` | Grid 中 Block 的数量 | dim3 |
| `blockDim.x` | Block 中 Thread 的数量 | dim3 |
| `blockIdx.x` | 当前 Block 的索引 | uint3 |
| `threadIdx.x` | 当前 Thread 在 Block 内的索引 | uint3 |

### 全局线程 ID 计算

```cpp
// 1D 情况
int globalId = blockIdx.x * blockDim.x + threadIdx.x;

// 2D 情况
int row = blockIdx.y * blockDim.y + threadIdx.y;
int col = blockIdx.x * blockDim.x + threadIdx.x;
int globalId = row * width + col;
```

### 图解

```
Grid: <<<4, 8>>>  (4 blocks, 每个 block 8 threads)

Block 0        Block 1        Block 2        Block 3
[0,1,2,3,4,5,6,7] [8,9,10,11,12,13,14,15] [16,17,18,19,20,21,22,23] [24,25,26,27,28,29,30,31]
 └── threadIdx    └── globalId = blockIdx * blockDim + threadIdx
```

---

## ✅ 本章作业：向量加法

### 作业目标

实现 C = A + B，其中 A, B, C 都是长度为 N 的向量

### 要求

1. 在 GPU 上并行计算
2. 每个线程计算一个元素
3. 处理数组长度不是线程数整数倍的情况

### 代码框架

文件：`vector_add.cu`

```cuda
#include <stdio.h>
#include <cuda_runtime.h>

#define N 10000  // 向量长度
#define BLOCK_SIZE 256

// TODO: 实现 Kernel
__global__ void vectorAdd(float *A, float *B, float *C, int n) {
    // 1. 计算全局线程 ID
    // 2. 边界检查（id < n）
    // 3. C[id] = A[id] + B[id]
}

int main() {
    // 1. 分配 Host 内存
    // 2. 初始化数据
    // 3. 分配 Device 内存
    // 4. 拷贝数据到 Device
    // 5. 启动 Kernel
    // 6. 拷贝结果回 Host
    // 7. 验证结果
    // 8. 释放内存
}
```

### 检查点

- [ ] Kernel 正确计算了全局 ID
- [ ] 处理了边界情况（N 不是 BLOCK_SIZE 的整数倍）
- [ ] 结果验证通过

---

## 🎯 进阶练习

### 练习1：2D Grid

修改向量加法，使用 2D Grid：

```cpp
dim3 block(16, 16);  // 16x16 = 256 threads per block
dim3 grid((N + 15) / 16, 1);
```

### 练习2：性能测量

使用 CUDA Event 测量执行时间：

```cpp
cudaEvent_t start, stop;
cudaEventCreate(&start);
cudaEventCreate(&stop);

cudaEventRecord(start);
vectorAdd<<<grid, block>>>(...);
cudaEventRecord(stop);

cudaEventSynchronize(stop);
float ms;
cudaEventElapsedTime(&ms, start, stop);
printf("Kernel 执行时间: %.3f ms\n", ms);
```

---

**完成后继续 Chapter 4！**
