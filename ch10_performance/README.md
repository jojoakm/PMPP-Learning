# Chapter 10: Performance Considerations ⭐重点章节

## 📖 本章内容

- 性能分析方法论
- Nsight Compute 使用
- 常见性能瓶颈
- 优化策略总结

---

## 🎯 学习目标

读完本章，你应该能：

- [ ] 使用 Nsight Compute 分析 Kernel
- [ ] 识别性能瓶颈（Memory Bound vs Compute Bound）
- [ ] 应用合适的优化策略
- [ ] 写出高效的 CUDA 代码

---

## 📝 核心概念

### 性能瓶颈类型

| 类型 | 特征 | 优化方向 |
|------|------|---------|
| Memory Bound | 内存带宽跑满，计算单元空闲 | 减少内存访问，提高数据复用 |
| Compute Bound | 计算单元跑满，内存带宽有余 | 减少计算量，使用更快的指令 |
| Latency Bound | 等待时间长，利用率低 | 增加并行度，隐藏延迟 |

### Nsight Compute 使用

```bash
# 运行 Profiling
ncu --set full -o profile ./my_program

# 查看报告
ncu-ui profile.ncu-rep
```

### 关键指标

| 指标 | 含义 | 目标 |
|------|------|------|
| Occupancy | 活跃 Warp 比例 | > 50% |
| Memory Throughput | 内存带宽利用率 | 接近理论值 |
| Compute Throughput | 计算单元利用率 | 接近理论值 |
| Warp Stall | Warp 等待原因 | 尽量减少 |

---

## ✅ 本章作业：Profile 并优化矩阵乘法

### 作业目标

1. 使用 Nsight Compute 分析 Naive 矩阵乘法
2. 识别瓶颈
3. 应用优化
4. 对比优化前后的指标

### 步骤

```bash
# 1. 编译（加调试信息）
nvcc -g -G matmul_naive.cu -o matmul_naive

# 2. Profile
ncu --set full ./matmul_naive

# 3. 分析报告，找出瓶颈

# 4. 优化代码

# 5. 再次 Profile，对比
```

### 需要回答的问题

1. Naive 版本的瓶颈是什么？（Memory/Compute/Latency）
2. Memory Throughput 是多少？理论值是多少？
3. 优化后提升了多少？

---

## 🎯 检查点

- [ ] 能使用 Nsight Compute 生成报告
- [ ] 能解读关键性能指标
- [ ] 能识别性能瓶颈类型
- [ ] 知道对应的优化策略

---

## 📚 优化策略总结

### Memory Bound 优化

1. 使用 Shared Memory 减少 Global Memory 访问
2. Memory Coalescing
3. 避免 Bank Conflict
4. 使用更宽的数据类型（float4）

### Compute Bound 优化

1. 使用更快的数学函数（__sinf vs sinf）
2. 减少分支
3. 指令级并行（ILP）

### Latency Bound 优化

1. 增加 Occupancy
2. 增加每个线程的工作量
3. 使用异步操作

---

**恭喜！完成了 PMPP 核心章节！**

接下来可以：
1. 回顾跳过的章节（Ch7 卷积、Ch8 模板、Ch11 前缀和）
2. 开始 MiniDriveWorld 项目
3. 学习 PyTorch CUDA Extension
