cu# 📚 PMPP 学习笔记与练习

> Programming Massively Parallel Processors (4th Edition)
> 
> 这是我学习 PMPP 的笔记和代码练习

---

## 📁 目录结构

```
PMPP_Learning/
├── ch01_introduction/         # 第1章：导论
├── ch02_heterogeneous_computing/  # 第2章：异构计算
├── ch03_cuda_execution_model/ # 第3章：CUDA 执行模型 ⭐重点
├── ch04_memory_architecture/  # 第4章：内存架构 ⭐重点
├── ch05_memory_coalescing/    # 第5章：内存合并 ⭐重点
├── ch06_thread_execution/     # 第6章：线程执行 ⭐重点
├── ch07_convolution/          # 第7章：卷积
├── ch08_stencil/              # 第8章：模板计算
├── ch09_reduction/            # 第9章：归约 ⭐重点
├── ch10_performance/          # 第10章：性能优化 ⭐重点
├── ch11_prefix_sum/           # 第11章：前缀和
├── ch12_histogram/            # 第12章：直方图
└── ch13_floating_point/       # 第13章：浮点数
```

---

## 📅 学习计划

| 周 | 章节 | 重点 | 练习 |
|----|------|------|------|
| Week 1 | Ch1-2 | 理解 GPU vs CPU | 无代码 |
| Week 1 | Ch3 | Grid/Block/Thread | 向量加法 |
| Week 2 | Ch4 | 内存层次 | 矩阵乘法 Naive |
| Week 2 | Ch5 | Shared Memory | 矩阵乘法 Tiled |
| Week 3 | Ch6 | Warp/Divergence | 优化矩阵乘法 |
| Week 3 | Ch9 | Reduction | 并行求和 |
| Week 4 | Ch10 | 性能分析 | Nsight Profiling |

---

## 🎯 学习目标

完成后你应该能够：

- [ ] 理解 GPU 架构（SM, Warp, Thread）
- [ ] 写基本的 CUDA Kernel
- [ ] 使用 Shared Memory 优化
- [ ] 避免 Bank Conflict
- [ ] 避免 Warp Divergence
- [ ] 使用 Nsight 分析性能
- [ ] 实现常见并行算法（Reduce, Scan）

---

## 🔧 环境配置

```bash
# 检查 CUDA
nvcc --version

# 检查 GPU
nvidia-smi

# 编译 CUDA 程序
nvcc -o output input.cu

# 编译并运行
nvcc -o test test.cu && ./test
```

---

## 📖 参考资源

- [CUDA C++ Programming Guide](https://docs.nvidia.com/cuda/cuda-c-programming-guide/)
- [CUDA Best Practices Guide](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/)
- [Nsight Compute Documentation](https://docs.nvidia.com/nsight-compute/)

---

**开始学习吧！** 🚀
