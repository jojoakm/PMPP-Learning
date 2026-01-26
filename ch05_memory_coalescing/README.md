# Chapter 5: Memory Coalescing & Shared Memory ⭐重点章节

## 📖 本章内容

- Memory Coalescing（内存合并）
- Shared Memory 使用
- Bank Conflict
- Tiled 矩阵乘法

---

## 🎯 学习目标

读完本章，你应该能：

- [ ] 理解什么是 Memory Coalescing
- [ ] 正确使用 Shared Memory
- [ ] 理解并避免 Bank Conflict
- [ ] 实现 Tiled 矩阵乘法
- [ ] 对比 Naive 和 Tiled 的性能差异

---

## 📝 核心概念

### Memory Coalescing（内存合并）

当同一个 Warp（32个线程）访问**连续的内存地址**时，GPU 可以合并成一次内存事务。

```
好的访问模式（Coalesced）：
Thread 0 → addr[0]
Thread 1 → addr[1]
Thread 2 → addr[2]
...
Thread 31 → addr[31]
→ 合并成 1 次内存事务

坏的访问模式（Non-coalesced）：
Thread 0 → addr[0]
Thread 1 → addr[32]
Thread 2 → addr[64]
...
→ 32 次内存事务！慢 32 倍！
```

### Shared Memory

```cpp
__shared__ float tile[16][16];  // 声明 Shared Memory

// 使用
tile[threadIdx.y][threadIdx.x] = data[...];
__syncthreads();  // 同步！确保所有线程都写完了
result = tile[...][...];
```

### Bank Conflict

Shared Memory 分成 32 个 bank，每个 bank 每周期只能服务一个请求。

```
Bank 0:  addr 0, 32, 64, ...
Bank 1:  addr 1, 33, 65, ...
Bank 2:  addr 2, 34, 66, ...
...
Bank 31: addr 31, 63, 95, ...

无冲突：每个线程访问不同 bank
有冲突：多个线程访问同一 bank → 串行化
```

### Tiled Matrix Multiplication

核心思想：把矩阵分成小块（Tile），加载到 Shared Memory，在 Shared Memory 中复用数据。

```
┌─────────────────┐     ┌─────────────────┐
│        A        │     │        B        │
│   ┌────┐        │     │   ┌────┐        │
│   │Tile│ ────────────→│   │Tile│        │
│   └────┘        │     │   └────┘        │
│                 │     │        │        │
└─────────────────┘     └────────│────────┘
                                 ↓
                        ┌────────────────┐
                        │  Shared Memory │
                        │  ┌────┐ ┌────┐ │
                        │  │ As │ │ Bs │ │
                        │  └────┘ └────┘ │
                        └────────────────┘
                                 │
                                 ↓
                              计算
```

---

## ✅ 本章作业：Tiled 矩阵乘法

### 作业目标

使用 Shared Memory 优化矩阵乘法

### 算法步骤

1. 每个 Block 负责计算 C 的一个 Tile
2. 循环遍历 A 和 B 的 Tile：
   - 加载 A 的一个 Tile 到 Shared Memory
   - 加载 B 的一个 Tile 到 Shared Memory
   - `__syncthreads()` 同步
   - 计算部分结果
   - `__syncthreads()` 同步
3. 写回结果到 Global Memory

### 代码框架

文件：`matmul_tiled.cu`

```cuda
#define TILE_SIZE 16

__global__ void matmul_tiled(float *A, float *B, float *C,
                              int M, int K, int N) {
    // Shared Memory
    __shared__ float As[TILE_SIZE][TILE_SIZE];
    __shared__ float Bs[TILE_SIZE][TILE_SIZE];
    
    int row = blockIdx.y * TILE_SIZE + threadIdx.y;
    int col = blockIdx.x * TILE_SIZE + threadIdx.x;
    
    float sum = 0.0f;
    
    // 遍历所有 Tile
    for (int t = 0; t < (K + TILE_SIZE - 1) / TILE_SIZE; t++) {
        // 1. 加载 A 的 Tile
        // 2. 加载 B 的 Tile
        // 3. __syncthreads()
        // 4. 计算
        // 5. __syncthreads()
    }
    
    // 写回结果
    if (row < M && col < N) {
        C[row * N + col] = sum;
    }
}
```

### 性能对比

| 版本 | 预期性能 |
|------|---------|
| Naive | ~100 GFLOPS |
| Tiled | ~500 GFLOPS |
| cuBLAS | ~10000 GFLOPS |

---

## 🎯 检查点

- [ ] 理解 Memory Coalescing 的重要性
- [ ] 能正确使用 `__shared__` 和 `__syncthreads()`
- [ ] 理解 Bank Conflict 并知道如何避免
- [ ] Tiled 版本比 Naive 快 3-5 倍

---

## 🚀 进阶练习

### 练习1：避免 Bank Conflict

修改 Shared Memory 声明，添加 padding：

```cpp
// 有 Bank Conflict
__shared__ float As[16][16];

// 无 Bank Conflict（添加 1 列 padding）
__shared__ float As[16][17];
```

### 练习2：更大的 Tile

尝试 TILE_SIZE = 32，观察性能变化。

---

**完成后继续 Chapter 6！**
