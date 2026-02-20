# NCU 命令速查（基于本机 `ncu 2022.4.1`）

> 先掌握 20% 命令，覆盖 80% 场景。

---

## 0) 先确认版本

```bash
ncu --version
```

建议先创建输出目录：

```bash
mkdir -p reports
```

---

## 1) 最常用：默认集合（快速看全局）

```bash
ncu --set default -o reports/default ./benchmark
```

适用：

- 初次分析某个 kernel
- 快速判断大方向（算力 vs 内存）

---

## 2) 指定 kernel（避免报告太杂）

```bash
ncu --set default \
    --kernel-name-base demangled \
    --kernel-name regex:gemm_v2_tiled \
    -o reports/v2_default \
    ./benchmark
```

说明：

- `--kernel-name-base demangled`：用可读函数名匹配
- `--kernel-name regex:...`：只抓你关心的 kernel

---

## 3) 采样某次调用（过滤 warmup 干扰）

```bash
ncu --set default \
    --launch-skip 5 \
    --launch-count 1 \
    --kernel-name-base demangled \
    --kernel-name regex:gemm_v1_naive \
    -o reports/v1_steady \
    ./benchmark
```

适用：

- benchmark 内部有多次重复调用
- 只想采“稳定阶段”的一次结果

---

## 4) 需要更细节时再上 `detailed` / `full`

```bash
ncu --set detailed -o reports/v2_detailed ./benchmark
```

```bash
ncu --set full -o reports/v2_full ./benchmark
```

建议：

- 日常优先 `default`
- 只有在定位不到根因时才用 `detailed/full`（更慢）

---

## 5) 离线查看报告（命令行）

```bash
ncu --import reports/v2_default.ncu-rep --page details
```

也可输出 raw：

```bash
ncu --import reports/v2_default.ncu-rep --page raw
```

---

## 6) 先看哪些 section（够你面试）

- `LaunchStats`：配置与资源占用
- `Occupancy`：并发潜力受限因素
- `SpeedOfLight`：总体吞吐利用情况

补充（需要时）：

- `MemoryWorkloadAnalysis`
- `WarpStateStats`

---

## 7) 一套实战判断流程

1. 看总耗时（是否真的变快）  
2. 看 `SpeedOfLight`（算力/内存哪个更吃紧）  
3. 看 `Occupancy`（是否被寄存器/shared memory 限制）  
4. 看 `LaunchStats`（block 配置是否合理）  
5. 回到代码验证假设（coalescing / tiling / 向量化）

---

## 8) 常见坑

- 直接上 `full`，导致 profiling 太慢、迭代效率低
- 同时改太多地方，无法确认哪项优化生效
- 只看一个指标，不结合 kernel 时间
- 忽略 warmup，导致数据波动大
