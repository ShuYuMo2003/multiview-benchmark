# 项目目的与初衷（Project Charter）

状态：持续维护。本文件用于记录项目的根本目标和边界；模型与训练方案可以
修改，但不应在没有明确讨论的情况下悄然改变这里定义的 benchmark 问题。

最后更新：2026-09-01 UTC

## 1. 项目初衷

机器人视频生成模型可以分别生成清晰、流畅的 head view 和 wrist view，
但这些视角未必在同一时刻表达同一个机器人与环境状态。现有视频质量指标
主要评价单视角画质、时间稳定性或生成分布，不能直接发现以下错误：

- head view 中的手臂位置与 wrist view 隐含的末端位置不一致；
- 两个视角中的夹爪开合、接触或抓取状态不一致；
- wrist view 中的物体、局部几何或交互阶段与 head view 不相容；
- 每个视角单独看都合理，但组合起来不可能同时发生。

本项目希望建立一个专门的 **Frame-level Cross-View State Consistency
Metric**，衡量生成的 head view 与 wrist view 是否描述了同一个物理状态，
并提供可以定位错误来源的诊断结果。

## 2. 核心评价问题

给定生成视频同一 nominal timestep 的 head frame 和一个 wrist frame，模型
需要回答：

> 在当前机器人本体和场景中，这两个画面所表达的末端执行器、夹爪以及局部
> 视觉状态是否彼此兼容？如果不兼容，差异主要是什么？

这里评价的是**状态一致性**，不是时间戳相等：候选 wrist frame 可以来自
任意时刻。只要两个画面表达相同的机器人和视觉状态，残差就应接近零；本
指标不以预测 temporal offset 为目标。

当前项目聚焦单一已知机器人本体（WAM setting）。head camera 视野较广，
通常能够观察 wrist camera/末端所在区域，因此任务包含从 head view 推断
wrist 局部状态的能力。它很困难，但正是希望 benchmark 测量的能力之一。

## 3. 预期输出

对每个 `(head, wrist)` pair，指标应保留以下可解释输出：

- EEF translation gap；
- EEF rotation gap；
- arm joint/configuration gap（数据支持时）；
- gripper mismatch；
- wrist-local visual/DINO feature gap；
- learned compatibility energy；
- validity/observability confidence（有可靠监督后）。

同一个共享模型分别运行 head–left-wrist 和 head–right-wrist 两次。benchmark
应分别报告左右 wrist 的子指标，再提供一个经过验证和校准的整体分数。统一
energy 可以成为 headline score 的候选，但不能取代诊断子指标。

## 4. 明确不做什么

- 不把单视角清晰度、美学质量或一般视频流畅度当作本指标的主要目标；
- 不要求 head view 与 wrist view 具有相同像素内容或相同相机视角；
- 不把 nominal frame index/time offset 本身作为监督目标；
- 不仅用一个不透明总分替代所有物理和视觉诊断；
- 不默认 action 等于真实 state，除非数据集明确给出绝对状态语义；
- 不把 mock data 或示例 MP4 probe 的结果当作正式 benchmark 结论。

## 5. 正式数据与监督原则

正式训练数据采用 LeRobot 格式的同步多视角真实机器人轨迹，并使用
observation 中的 EEF、joint 和 gripper state。物理状态必须统一到明确的
robot-base frame，所有字段的单位、旋转表示和 mask 都要版本化记录。

同轨迹 pair 可以构造明确的物理残差：

```text
translation = p_t - p_s
rotation = Log(R_s^T R_t)
joint = q_t - q_s
gripper = g_t - g_s
visual = z(W_t) - z(W_s)
```

跨轨迹 pair 的物理坐标通常不可直接比较，因此默认不施加 EEF、joint 和
gripper residual 监督。视觉监督或 compatibility 监督是否用于跨轨迹 pair，
必须通过消融验证，防止模型仅学习 episode/domain identity。

## 6. 当前技术假设（允许修改）

以下是当前实现方向，而不是不可改变的项目定义：

- frozen/partially trainable foundation visual encoder；
- DINOv2 patch tokens 或 VGGT joint tokens；
- 小型 pairwise cross-attention/readout；
- compatibility energy 主头与 physical/visual residual 辅助头并存；
- 冻结 DINO teacher feature 经过 train-only PCA 或 learned projection 降维；
- 离线、可复现的 positive/hard-negative/state-equivalent pair plan；
- 先用 DINOv2 验证，再扩展到 DINOv2-G、VGGT 等更大模型；
- 先使用 DDP；完整微调超大模型时按显存需求增加 FSDP。

这些选择应由真实 LeRobot 数据上的消融、校准和泛化结果决定，而不是因为
当前代码已经实现就永久保留。

## 7. Benchmark 成功标准

一个可发布的指标至少应满足：

1. 对同步真实 pair 给出低 mismatch，对受控错配给出更高 mismatch；
2. 对位移、旋转、夹爪和局部视觉扰动具有可解释且大体单调的响应；
3. 能识别“两个视角各自合理但组合不一致”的生成失败；
4. 与人工 cross-view consistency 判断具有稳定相关性；
5. 在未见 episode、任务、背景和生成模型上仍能泛化；
6. 不主要依赖时间差、episode identity、背景纹理或左右相机等 shortcut；
7. 总分经过 held-out calibration，且同时发布左右 wrist 和各类子指标；
8. 在不可观测或高度歧义的样本上能够表达低 confidence，而非制造确定残差。

## 8. 最终交付目标

- 一个可复现训练的跨视角一致性模型和 checkpoint；
- 一套固定数据 split、pair plan、扰动集和标注协议；
- 清晰定义的总体分数、子指标、归一化和校准方式；
- 对 DINOv2/VGGT、energy/residual、visual target 和采样策略的消融；
- 可在生成的 head/left-wrist/right-wrist 视频上直接运行的评估工具；
- 完整的训练、评估、版本和实验 memory。

## 9. 如何修改本文件

- 项目目的或 benchmark 问题改变：修改第 1–4 节，并在
  `DESIGN_DECISIONS.md` 记录原因；
- 数据假设改变：修改第 5 节和 `DATA_CONTRACT.md`；
- 只更换 backbone、head、loss 或训练规模：修改第 6 节，不需要重写初衷；
- 成功标准或发布协议改变：修改第 7–8 节和
  `EVALUATION_PROTOCOL.md`；
- 所有实际实验只追加到 `EXPERIMENT_LOG.md`，不要用实验结果覆盖历史记录。
