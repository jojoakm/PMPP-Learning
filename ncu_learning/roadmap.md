# NCU 两周学习路线（贴合你的 PMPP + GEMM 计划）

> 节奏假设：每天 1~2 小时  
> 核心原则：每次 profile 都要有结论，不做“纯收集”

---

## Week 1：把工具用熟 + 建立分析框架

### Day 1：工具与最小闭环

- 跑通一次 `ncu --set default`
- 学会导出报告并查看 `details`
- 记录一个 kernel 的基础指标

产出：

- `baseline` 报告 1 份
- 3 行总结：瓶颈猜测 + 证据 + 下一步动作

---

### Day 2：读懂三大核心 section

- `LaunchStats`：block/grid、寄存器、shared memory
- `Occupancy`：潜在并发受什么限制
- `SpeedOfLight`：算力和内存利用总体水平

产出：

- 1 张“指标 -> 含义 -> 行动”对照表

---

### Day 3-4：在 Ch05 上做 V1 vs V2 对比

建议对象：

- `ch05_memory_coalescing/gemm_old.cu`
- `ch05_memory_coalescing/gemm_tile.cu`

重点：

- 用同一输入规模对比 profile
- 观察 memory 相关吞吐与整体耗时变化

产出：

- 一份 V1/V2 对比表（至少含时间和 3 个关键指标）

---

### Day 5-7：进入 final_project 的正式分析

- 编译并运行 `final_project/benchmark`
- 先把 `V1`、`V2` 的 profile 基线做扎实
- 用统一命名保存报告（方便后续版本扩展）

产出：

- `final_project` 下 V1/V2 的基线报告与结论

---

## Week 2：围绕 V3~V6 形成“优化证据链”

### Day 8-10：V3/V4（内存访问主线）

- 每加一个优化，先写“预期影响指标”
- 只改一个主要变量，避免结论混淆
- profile 前后都保留

产出：

- V3、V4 各 1 份“改动说明 + 指标变化 + 结论”

---

### Day 11-12：V5/V6（并发与隐藏延迟主线）

- 关注 occupancy、stall 变化趋势
- 重点解释为什么吞吐上升（或没上升）

产出：

- V5、V6 版本证据记录

---

### Day 13-14：收口成可投递材料

- 汇总 V1~V6：Time / GFLOPS / 核心指标
- 写 1 页性能优化报告
- 打磨简历描述与 3 分钟讲解脚本

产出：

- 一份最终对比表
- 一份性能报告（面试材料）

---

## 每次 profiling 固定模板（强制执行）

1. **问题**：当前怀疑的瓶颈是什么？  
2. **证据**：哪个 section / 指标支持这个判断？  
3. **动作**：下一步改什么？为什么？  
4. **验证**：改完后指标与耗时如何变化？

只要这 4 步持续做，你的 NCU 学习就会非常快。

