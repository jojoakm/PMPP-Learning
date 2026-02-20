# NCU 学习目录（面向 AI Infra 春招实习）

这个目录是给你当前 `PMPP_Learning` 仓库定制的 Nsight Compute（`ncu`）学习路线。

你的主线目标不是“学会点按钮”，而是：

1. 能用数据说明某个 CUDA Kernel 慢在哪里  
2. 能根据指标提出并验证优化方案  
3. 能把优化过程整理成面试可讲的工程故事

---

## 你需要学到什么程度（建议目标）

针对 AI Infra 春招实习，建议达到 **L2（可独立分析）→ L3（可面试表达）**：

- **L1 入门会用**：会跑 `ncu`，会看 `default` 报告
- **L2 能定位瓶颈**：能区分 Memory Bound / Compute Bound / Latency Bound
- **L3 能闭环优化**：指标 → 修改代码 → 再 profile → 量化收益

结论：**你不用成为性能工具专家，但必须能独立完成一次优化闭环。**

---

## 推荐学习顺序（配合你当前仓库）

1. `skill_target.md`：先看目标标准，避免学偏  
2. `command_cheatsheet.md`：掌握最常用命令  
3. `roadmap.md`：按 2 周节奏推进  
4. `hands_on_projects.md`：直接做练手项目（和你 V1-V6 强绑定）

---

## 目录说明

```text
ncu_learning/
├── README.md
├── skill_target.md        # 学习深度标准（L1~L3）
├── roadmap.md             # 2周学习路线（每日输出）
├── command_cheatsheet.md  # NCU 高频命令与解读框架
└── hands_on_projects.md   # 练手项目与验收标准
```

---

## 你这台机器上的 NCU 版本

当前检测到：

- `ncu 2022.4.1.0`

本目录命令都按这个版本的常用参数编写（`--set`、`--kernel-name`、`--launch-skip`、`--import`）。

---

## 一句话执行建议

你现在的最优路径是：  
**先完成 PMPP 前六章 + 做完 GEMM V1~V6 + 用 NCU 对每个版本做可量化对比，再投递。**

