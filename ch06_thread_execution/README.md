# Chapter 6: Thread Execution Efficiency ⭐重点章节

## 📖 本章内容

- Warp 执行模型
- Warp Divergence
- Occupancy（占用率）
- 性能优化策略

---

## 🎯 学习目标

读完本章，你应该能：

- [ ] 理解 Warp 是什么
- [ ] 识别和避免 Warp Divergence
- [ ] 理解 Occupancy 的含义
- [ ] 使用 Nsight 分析性能

---

## 📝 核心概念

### Warp

**Warp = 32 个线程，是 GPU 调度的基本单位**

```
Block (256 threads)
├── Warp 0: Thread 0-31
├── Warp 1: Thread 32-63
├── Warp 2: Thread 64-95
├── ...
└── Warp 7: Thread 224-255

同一个 Warp 内的线程：
- 执行相同的指令
- 在同一时刻
- 但操作不同的数据（SIMT）
```

### Warp Divergence（分支分歧）

当 Warp 内的线程走不同的分支时，会导致性能下降。

```cpp
// 有 Divergence（坏）
if (threadIdx.x < 16) {
    // 前 16 个线程执行这里
    doA();
} else {
    // 后 16 个线程执行这里
    doB();
}
// GPU 必须先执行 doA()（后 16 个线程等待）
// 再执行 doB()（前 16 个线程等待）
// 效率降低 50%！

// 无 Divergence（好）
if (threadIdx.x / 32 == 0) {
    // 整个 Warp 0 执行这里
    doA();
} else {
    // 整个 Warp 1 执行这里
    doB();
}
// 每个 Warp 内部没有分歧
```

### Occupancy（占用率）

Occupancy = 活跃 Warp 数 / SM 最大 Warp 数

影响因素：
- 每个线程使用的 Register 数量
- 每个 Block 使用的 Shared Memory 大小
- Block 大小

```
高 Occupancy（好）：
- 更多 Warp 可以隐藏内存延迟
- GPU 利用率高

低 Occupancy（可能有问题）：
- Warp 等待内存时，没有其他 Warp 可以执行
- GPU 利用率低
```

---

## ✅ 本章作业：优化 Reduce（归约）

### 作业目标

实现并行求和：计算数组所有元素的和

### 版本演进

1. **Naive 版本**：有严重的 Warp Divergence
2. **优化版本**：消除 Divergence
3. **进一步优化**：减少空闲线程

### 代码框架

文件：`reduce.cu`

```cuda
// Naive 版本（有 Divergence）
__global__ void reduce_naive(float *input, float *output, int n) {
    __shared__ float sdata[256];
    int tid = threadIdx.x;
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    
    sdata[tid] = (i < n) ? input[i] : 0;
    __syncthreads();
    
    // 有 Divergence！
    for (int s = 1; s < blockDim.x; s *= 2) {
        if (tid % (2 * s) == 0) {  // 分支条件导致 Divergence
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }
    
    if (tid == 0) output[blockIdx.x] = sdata[0];
}

// 优化版本（无 Divergence）
__global__ void reduce_optimized(float *input, float *output, int n) {
    // TODO: 消除 Divergence
}
```

---

## 🎯 检查点

- [ ] 能解释什么是 Warp Divergence
- [ ] 能识别代码中的 Divergence
- [ ] 理解 Occupancy 的影响因素
- [ ] 优化版 Reduce 比 Naive 快 2x+

---

**完成后继续 Chapter 9（跳过 7、8）！**
