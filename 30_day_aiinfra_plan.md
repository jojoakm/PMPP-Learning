# 30天 AI Infra 实习冲刺计划（PMPP 路线版）

> 适用你当前状态：PMPP 快到 Ch6，前五章已实践  
> 核心目标：**一个月内达到“可投递 + 可面试讲清”**

---

## 总目标（30天结束时）

- [ ] PMPP 前六章完成并有自己的学习笔记
- [ ] `final_project` 完成 `V1~V6`（至少保证到 `V4`）
- [ ] 每个版本有基础性能数据 + NCU 结论
- [ ] 独立项目 `cuda_cpp_interview_lab` 看懂并手写重写一遍（V1）
- [ ] 完成简历项目描述 + 面试3分钟讲稿
- [ ] 开始并持续投递

---

## 每日时间建议（最低配置）

- 工作日：`2.5h ~ 4h`
- 周末：`5h ~ 8h`
- 每天固定 3 段：
  1. 理论/阅读（30~60min）
  2. 编码实现（90~180min）
  3. 记录复盘（20~30min）

---

## 第1周（Day 1-7）：收尾 Ch6 + 固化 V1/V2 + 建立 NCU 基线

### Day 1
- [ ] 完成 Ch6 阅读收尾
- [ ] 在 `ch06_thread_execution/learningnote.md` 写一页总结（warp/divergence/occupancy）
- [ ] 跑通 `final_project` 的 `V1/V2`

### Day 2
- [ ] 用相同输入规模跑 `V1/V2`
- [ ] 记录 `time/GFLOPS/speedup`
- [ ] 形成首版对比表（哪怕粗略）

### Day 3
- [ ] 跑 `ncu --set default` 分析 `V1`
- [ ] 记录 `LaunchStats/Occupancy/SpeedOfLight` 三块结论
- [ ] 写 3 句：瓶颈是什么、证据是什么、下一步做什么

### Day 4
- [ ] 对 `V2` 做同样 NCU 分析
- [ ] 和 `V1` 做对比结论
- [ ] 在 `final_project/docs/optimization_notes.md` 填写 V1/V2 部分

### Day 5
- [ ] 设计 `V3` 实现方案（coalescing）
- [ ] 写下“优化前预期指标变化”
- [ ] 完成 `V3` 第一版代码

### Day 6
- [ ] 验证 `V3` 正确性
- [ ] 跑性能并与 `V2` 比较
- [ ] 做 V3 的 NCU 基础分析

### Day 7（周复盘）
- [ ] 输出 Week1 小结（1页）
- [ ] 明确 Week2 目标风险（哪里最可能卡住）
- [ ] 投递准备：整理项目仓库结构和说明

---

## 第2周（Day 8-14）：完成 V4/V5（保底 V4）

### Day 8
- [ ] 设计 `V4`（float4 向量化）方案
- [ ] 列出边界条件（对齐、dim 可整除等）

### Day 9
- [ ] 完成 `V4` 初版
- [ ] 做 correctness 检查
- [ ] 跑基础性能

### Day 10
- [ ] 做 `V4` NCU 分析
- [ ] 对比 `V3`，写出“收益来源”

### Day 11
- [ ] 设计 `V5`（register tiling）参数
- [ ] 完成 `V5` 初版

### Day 12
- [ ] 修 bug + correctness
- [ ] 跑性能，若没提升先不深究极限调参

### Day 13
- [ ] 做 `V5` NCU 分析
- [ ] 记录 occupancy / 吞吐变化

### Day 14（周复盘）
- [ ] 如果 V5 卡住：保证 V4 稳定并写清原因
- [ ] 更新对比表：V1~当前版本
- [ ] 开始第一批投递（边做边投）

---

## 第3周（Day 15-21）：冲 V6 + 打磨证据链

### Day 15
- [ ] 设计 `V6`（double buffering）数据流
- [ ] 画简图（tile 加载与计算重叠）

### Day 16
- [ ] 完成 `V6` 初版
- [ ] correctness 检查

### Day 17
- [ ] 跑 `V6` 性能
- [ ] 做 `V6` NCU 分析

### Day 18
- [ ] 汇总 `V1~V6` 数据表（时间、GFLOPS、关键指标）
- [ ] 写“每版一句话瓶颈变化”

### Day 19
- [ ] 打磨 `final_project/docs/optimization_notes.md`
- [ ] 产出 1 页性能报告（可直接面试展示）

### Day 20
- [ ] 准备简历项目 bullet（量化）
- [ ] 准备 3 分钟项目讲稿

### Day 21（周复盘）
- [ ] 自测一次项目讲解（录音/录屏）
- [ ] 修正表达不清处
- [ ] 加投一批岗位

---

## 第4周（Day 22-30）：重写独立项目 + 面试冲刺

### Day 22
- [ ] 阅读 `../cuda_cpp_interview_lab/README_REWRITE.md`
- [ ] 新建 `rewrite_lab/` 骨架目录

### Day 23
- [ ] 手写 `rewrite_lab/include/pipeline.h`
- [ ] 手写 `rewrite_lab/src/pipeline_cpu.cpp`

### Day 24
- [ ] 手写 `rewrite_lab/src/main.cu`（先接 CPU）
- [ ] 保证可编译可运行

### Day 25
- [ ] 手写 `rewrite_lab/include/cuda_utils.h`
- [ ] 手写 `rewrite_lab/src/pipeline_gpu.cu`（V1）

### Day 26
- [ ] 接入 CPU/GPU 对比与判定
- [ ] 记录重写中踩坑清单

### Day 27
- [ ] 做一次 NCU 分析（独立项目）
- [ ] 输出 3 条可讲的优化思路（即使还没实现）

### Day 28
- [ ] 复习 C++ 高频点：RAII/move/template/STL
- [ ] 复习 CUDA 高频点：memory/warp/occupancy/coalescing

### Day 29
- [ ] 模拟面试：项目深挖 + 基础题
- [ ] 查漏补缺

### Day 30（收官）
- [ ] 终版简历 + 项目链接 + 讲稿定稿
- [ ] 制定接下来 2 周“投递 + 面试”节奏

---

## 保底策略（防止进度失控）

如果时间不够，按优先级砍任务：

1. 必保：`V1~V4 + NCU + 项目讲稿`
2. 次保：`V5`（可简化）
3. 可选：`V6` 深度调优、额外加分项

原则：**先保证可投递，再追求极致优化。**

---

## 每日复盘模板（建议复制到日志）

```text
今天完成：
遇到问题：
我是怎么定位的：
明天最重要的一件事：
```

