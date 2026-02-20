# NCU 练手项目（按你的仓库设计）

> 每个项目都要求：**结论必须有指标证据**。

---

## Project 1：Ch05 的 Naive vs Tiled 对比（必做）

目标：

- 建立你的第一条性能证据链

建议对象：

- `ch05_memory_coalescing/gemm_old.cu`
- `ch05_memory_coalescing/gemm_tile.cu`

你要回答：

1. 哪个版本更快？快多少？  
2. 主要瓶颈分别是什么？  
3. Tiled 为什么改善（从指标解释）？

验收标准：

- 有对比表（Time + 3 个关键指标）
- 有 3~5 句结论，不是“感觉变快”

---

## Project 2：final_project 的 V1 vs V2（必做）

目标：

- 把 Chapter 练习迁移到项目级 benchmark 场景

操作建议：

```bash
cd /home/gjj/projects/PMPP_Learning/final_project
make
mkdir -p reports
ncu --set default -o reports/v1v2_default ./benchmark
```

你要回答：

1. V2 相比 V1 的核心收益是访存还是计算？  
2. 如果收益不明显，是哪类限制导致？

验收标准：

- 能给出“优化前后变化最大的 2 个指标”

---

## Project 3：你实现 V3 后的假设验证（强烈推荐）

目标：

- 练“先假设、后验证”的性能工程思维

做法：

1. 在写代码前，先写下预期：  
   - 哪个指标会改善？  
   - 哪个指标可能恶化？  
2. 实现 V3（coalescing 优化）  
3. 用 NCU 验证是否符合预期

验收标准：

- 有“预期 vs 实际”对照
- 能解释偏差原因

---

## Project 4：V1~V6 最终报告（投递前必做）

目标：

- 形成你的实习投递核心材料

建议报告结构：

1. 版本迭代图（V1→V6）  
2. 每次优化动作  
3. 关键指标变化  
4. 最终性能（vs V1 / vs cuBLAS）  
5. 复盘：最有效的 2~3 个优化点

验收标准：

- 一页报告可读完
- 面试时可 3 分钟讲清楚

---

## 进阶加分（有精力再做）

### 加分 A：Kernel 粒度稳定性

- 固定输入规模，连续多次 profile
- 观察指标方差，理解波动来源

### 加分 B：配置扫描

- 比较不同 block size（如 `16x16` vs `32x8`）
- 用数据选择参数，而不是经验拍板

### 加分 C：简历表达优化

- 把“指标变化”翻译成业务语言：
  - 吞吐提升
  - 延迟降低
  - 资源利用率提升
