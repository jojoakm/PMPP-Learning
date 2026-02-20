# 🚀 Final Project: High-Performance GEMM

> 从零实现高性能矩阵乘法，性能达到 cuBLAS 的 80%+

**这是一个可以写进简历的硬核 CUDA 项目！**

---

## 📖 项目简介

GEMM (General Matrix Multiply) 是深度学习的核心算子。本项目从 Naive 实现开始，逐步优化到接近 cuBLAS 的性能。

---

## 🎯 项目目标

| 版本 | 优化技术 | 预期性能 |
|------|---------|---------|
| V1 Naive | 基础实现 | ~100 GFLOPS |
| V2 Tiled | Shared Memory | ~500 GFLOPS |
| V3 Coalescing | 内存合并优化 | ~1000 GFLOPS |
| V4 Vectorized | float4 向量化 | ~2000 GFLOPS |
| V5 Register Tiling | 寄存器分块 | ~4000 GFLOPS |
| V6 Double Buffering | 隐藏延迟 | ~6000 GFLOPS |
| cuBLAS | NVIDIA 官方 | ~8000 GFLOPS |

**目标：达到 cuBLAS 80% 的性能！**

---

## 📁 文件结构

```
final_project/
├── README.md
├── Makefile
├── include/
│   └── gemm.h              # 头文件
├── src/
│   ├── v1_naive.cu         # V1: Naive 实现
│   ├── v2_tiled.cu         # V2: Shared Memory Tiling
│   ├── v3_coalescing.cu    # V3: 内存合并优化
│   ├── v4_vectorized.cu    # V4: float4 向量化
│   ├── v5_register.cu      # V5: 寄存器分块
│   ├── v6_double_buffer.cu # V6: Double Buffering
│   └── benchmark.cu        # 性能测试
├── tests/
│   └── test_correctness.cu # 正确性测试
└── docs/
    ├── optimization_notes.md  # 优化笔记
    └── performance_report.md  # 性能报告
```

---

## 🔧 优化技术详解

### V1: Naive Implementation
- 每个线程计算 C 的一个元素
- 问题：大量重复读取 Global Memory

### V2: Shared Memory Tiling
- 将矩阵分块加载到 Shared Memory
- 减少 Global Memory 访问

### V3: Memory Coalescing
- 优化内存访问模式
- 确保 Warp 内线程访问连续地址

### V4: Vectorized Load/Store
- 使用 float4 一次读取 4 个元素
- 提高内存带宽利用率

### V5: Register Tiling
- 每个线程计算多个输出元素
- 增加数据复用，减少 Shared Memory 压力

### V6: Double Buffering
- 计算和数据加载重叠
- 隐藏内存延迟

---

## 📊 性能测试

```bash
# 编译
make

# 指定架构编译（RTX 5070 Ti 建议）
make CUDA_ARCH=89

# 运行 benchmark
./benchmark

# 输出示例
Matrix Size: 4096 x 4096
V1 Naive:         120.5 GFLOPS
V2 Tiled:         523.7 GFLOPS (4.3x vs V1)
V3 Coalescing:   1024.3 GFLOPS (8.5x vs V1)
V4 Vectorized:   2156.8 GFLOPS (17.9x vs V1)
V5 Register:     4312.5 GFLOPS (35.8x vs V1)
V6 DoubleBuffer: 6245.2 GFLOPS (51.8x vs V1)
cuBLAS:          7823.4 GFLOPS
Achieved: 79.8% of cuBLAS
```

---

## 📝 简历写法

```
项目：High-Performance GEMM on CUDA

• 从零实现 GPU 矩阵乘法，通过 6 个版本迭代优化，最终性能达到 cuBLAS 的 80%
• 应用 Shared Memory Tiling、Memory Coalescing、向量化访存、寄存器分块等优化技术
• 使用 Nsight Compute 进行性能分析，识别并解决 Memory Bound 瓶颈
• 在 RTX 5070 Ti 上实现 6000+ GFLOPS，相比 Naive 版本提升 50 倍

技术栈：CUDA, C++, Nsight Compute
```

---

## 🎯 学习路线

### Week 1: V1 + V2
- [ ] 实现 Naive 版本
- [ ] 实现 Tiled 版本
- [ ] 对比性能差异

### Week 2: V3 + V4
- [ ] 优化内存访问模式
- [ ] 实现向量化读写
- [ ] 使用 Nsight 分析

### Week 3: V5 + V6
- [ ] 实现寄存器分块
- [ ] 实现 Double Buffering
- [ ] 最终性能调优

### Week 4: 文档 + 总结
- [ ] 撰写优化笔记
- [ ] 生成性能报告
- [ ] 整理代码，准备开源

---

## 📚 参考资料

- [CUTLASS](https://github.com/NVIDIA/cutlass) - NVIDIA 官方高性能 GEMM 库
- [How to Optimize GEMM](https://github.com/flame/how-to-optimize-gemm)
- [CUDA C++ Best Practices Guide](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/)

---

**这个项目完成后，你就是 CUDA 高手了！** 💪
