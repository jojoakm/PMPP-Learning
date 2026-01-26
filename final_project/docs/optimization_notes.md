# GEMM 优化笔记

> 记录每个版本的优化思路和学习心得

---

## V1: Naive Implementation

### 实现思路
- 每个线程计算 C 的一个元素
- 直接从 Global Memory 读取 A 和 B

### 性能分析
- [ ] 记录性能：_____ GFLOPS
- [ ] Memory Throughput：_____
- [ ] Compute Throughput：_____

### 学到的知识
```
（在这里记录你的理解）
```

---

## V2: Shared Memory Tiling

### 实现思路
- 将矩阵分成 TILE_SIZE x TILE_SIZE 的小块
- 每个 Block 负责计算 C 的一个 Tile
- 使用 Shared Memory 存储 A 和 B 的 Tile

### 优化效果
- [ ] 性能提升：_____ x
- [ ] Global Memory 访问减少：_____ 倍

### 遇到的问题
```
（在这里记录遇到的 bug 和解决方法）
```

### 学到的知识
```
（在这里记录你的理解）
```

---

## V3: Memory Coalescing

### 实现思路
- 优化 Global Memory 访问模式
- 确保 Warp 内线程访问连续地址

### 关键代码修改
```cuda
// 记录关键修改
```

### 性能提升
- [ ] 提升：_____ x

---

## V4: Vectorized Load/Store

### 实现思路
- 使用 float4 一次读取 4 个元素
- 提高内存带宽利用率

### 关键代码
```cuda
// float4 使用示例
float4 a = *reinterpret_cast<float4*>(&A[idx]);
```

### 性能提升
- [ ] 提升：_____ x

---

## V5: Register Tiling

### 实现思路
- 每个线程计算多个输出元素（如 4x4）
- 增加数据复用
- 减少 Shared Memory 压力

### 参数选择
- Thread Tile: _____ x _____
- Block Tile: _____ x _____

### 性能提升
- [ ] 提升：_____ x

---

## V6: Double Buffering

### 实现思路
- 计算当前 Tile 的同时，预加载下一个 Tile
- 隐藏内存延迟

### 实现要点
- 使用两组 Shared Memory buffer
- 交替使用

### 最终性能
- [ ] 性能：_____ GFLOPS
- [ ] 达到 cuBLAS 的 _____% 

---

## 总结

### 各版本性能对比

| 版本 | GFLOPS | vs V1 | vs cuBLAS |
|------|--------|-------|-----------|
| V1 | | 1.0x | |
| V2 | | | |
| V3 | | | |
| V4 | | | |
| V5 | | | |
| V6 | | | |
| cuBLAS | | | 100% |

### 最重要的优化技术

1. 
2. 
3. 

### 下一步学习方向

- [ ] 学习 CUTLASS 源码
- [ ] 了解 Tensor Core 优化
- [ ] 学习 Flash Attention 实现
