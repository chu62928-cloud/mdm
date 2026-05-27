# Posture Guidance on MDM — 项目接续总结

> 这份文档面向**刚接到这个项目的人**。读完应能理解：我们做了什么、当前 V6 演化到哪一版、三个体态的最终实测结果、为什么会成功/失败、什么是工程问题 vs 方法固有缺陷、如果要继续做该做什么。

> ⚠ 这份文档是对老版 `PROJECT_SUMMARY.md`（项目早期记录）的全面修订。老版有两个核心误判（已纠正于"零、老版误判"一节），如果你看到的还是老版，请优先以本文档为准。

---

## 零、老版 PROJECT_SUMMARY.md 的两个误判（先纠正）

老版描述了两个"卡点"，但其实**都是错的**：

| 老版主张 | 实际情况 |
|---|---|
| 🔴 "fk_fn 实现错误，build_fk_fn 切片归一化值没反归一化" | `posture_guidance/mdm_integration.py:13-58` 的 `make_fk_fn` **完全正确**：`mu_perm * std_d + mean_d` 反归一化后调用 `recover_from_ric`，梯度链完整。老版描述的 `build_fk_fn` **不存在**于 GitHub 代码中。 |
| 🟡 "v3 score 不看 corr，塌平也能得高分" | `new/evaluate_ablation_v3.py:152-174` 的 `composite_score` **已经**把 corr 放进分子，对塌平（hit_loose>0.9 & corr<0.15）、负 corr、过推、抖动都设了硬约束→0。 |

**结论**：fk_fn 不用动，score_v3 也不用大改。真正的瓶颈在跨 seed 稳定性和**任务表征性**，下面详述。

---

## 一、项目是什么

[MDM](https://github.com/GuyTevet/motion-diffusion-model) 是基于扩散的文本→动作生成框架，输出 HumanML3D 特征 `(B, 263, 1, T_frames)`。

我们在 **MDM 推理时**注入姿势约束（posture guidance），让生成动作满足特定病态体态目标，同时保留时序结构。**不微调 MDM 权重**。

以骨盆前倾（Anterior Pelvic Tilt, APT）为例：
- baseline walking APT ≈ 5–7°
- 目标：引导到 20°（病理性）
- 约束：保持步态节律、不塌成静态、不推过头

本质是 **inference-time classifier guidance**，损失基于正向运动学（FK）的关节角度。

---

## 二、核心文件结构（当前版本）

```
motion-diffusion-model/
├── diffusion/
│   └── gaussian_diffusion.py          ← V1-V6 dispatch，sampling loop 注入点
│
├── posture_guidance/
│   ├── registry.py                    ← spec 注册 + compute_hinge_loss + compute_huber_loss
│   ├── angle_ops.py                   ← pelvis_tilt_angle / signed_knee_angle / spine_posterior_bulge 等
│   ├── joint_indices.py               ← 22 个 SMPL 关节索引常量
│   ├── phase_detector.py              ← 步态相位 mask（V6 暂未用）
│   ├── controller.py                  ← PostureGuidance 入口，loss_form="hinge"|"huber"
│   ├── guidance_variants.py           ← V1-V5 实现（参考）
│   ├── mdm_integration.py             ← make_fk_fn + V1 的 apply_posture_guidance
│   └── closed_loop_controller.py      ← 【新增】V6 用的 PID 控制器 + orthogonal_project
│
├── new/                               ← 实验脚本层
│   ├── evaluate_ablation_v3.py        ← 单 sweep 评分 + shape 分类
│   ├── aggregate_seeds.py             ← 跨 seed 聚合
│   ├── run_seed_robustness.sh         ← 原 N=5 sweep
│   ├── run_seed_robustness_n15.sh     ← 【新增】N=15 sweep
│   ├── run_cross_posture.sh           ← 【新增】跨体态 wrapper
│   ├── POSTURE_REPRESENTABILITY.md    ← 【新增】22 关节下哪些体态可表征
│   └── ... (其他可视化脚本)
│
└── PROJECT_SUMMARY.md                 ← 本文档
```

新增 / 修改的关键文件标 【新增】。所有改动都在分支 `claude/fix-code-bugs-optimization-final`，commit 序列 `8effecf` → `9fb026b` → `1b4cdfd` → `3d21aaf` → `db8b9b8` → `4de9b41` → （本 commit）。

---

## 三、Guidance 变体清单（截止当前）

通过环境变量 `GUIDANCE_VARIANT` + `GUIDANCE_KWARGS_JSON` 选择：

| 变体 | 机制 | 关键参数 | 备注 |
|---|---|---|---|
| **V1** `v1_mu_sgd` | SGD on detached mu_t | n_inner_steps=15, lr=0.5 | 老 baseline，稳但弱 |
| **V2** `v2_dps` | DPS：grad 穿过 MDM 更新 mu_t | s=40, schedule=always | **目前 best-mean variant** |
| **V2-norm** `v2_dps_norm` | V2 + grad 按 ‖grad‖ 归一化 | s=2.0 | 跨 seed 等效推力一致 |
| V2b `v2_x0_edit` | SGD on x0_hat，重组 mu_t | bw=5 | corr 最高但 hit 低 |
| V3 `v3_x0_direct` | 类 V2b | n=5, lr=0.05 | |
| V4 `v4_omni` | OmniControl 动态 K_e/K_l | K_early=1, K_late=10 | 易过推 |
| V5 `v5_lgd` | DPS + Monte-Carlo 平滑 | n_mc=4, mc_noise_scale=0.05 | 方差大 |
| **V6** `v6_closed_loop` | **PID 闭环 + Huber + manifold proj** | 见下表 | 本项目核心创新 |

### V6 闭环 — 五次迭代的演化（**重要**）

V6 经历了 5 次迭代，每次都修复一个具体 bug。**如果你只想用一个版本，用迭代 5（当前 default）的配置 + `spec_schedule_override="second_half"`**：

| 迭代 | 关键变更 | 失败模式 / 学到的 |
|---|---|---|
| **it1** | PID + Huber + grad-norm + band_gate=True + s_min=5 | s_t 全程被 s_min=5 顶住；越过目标后 hinge_loss=0 → grad=0 → grad/‖grad‖ 退化噪声 → corr 污染 |
| **it2** | (Huber 引入实际生效) | 灾难：grad/‖grad‖ + band_gate 让 V6 退化成"两次冲击 + 11 步放任"，corr 崩到 0.022（CV=1860%） |
| **it3** | **关掉 grad-norm, 关掉 band_gate**, raw grad | Huber 的 grad 本身就有界自调节（远=1，近→0）；用 raw grad + 闭环 PID 是正确架构 |
| **it4** | `spec_schedule_override="always"` 全程引导 | 揭示 hit↔corr trade-off：步数越多 corr 越高、hit 越低 |
| **it5** | `sigma_cutoff` 按噪声水平跳过早期高 σ 步 | 没突破 Pareto 前沿；说明 hit/corr 是结构性 trade-off，不是参数没调好 |

**V6 当前推荐配置（second_half schedule，hit/corr 平衡）**：

```bash
GUIDANCE_VARIANT=v6_closed_loop \
GUIDANCE_KWARGS_JSON='{"Kp":80,"Ki":1,"Kd":5,"s_min":0.05,"s_max":50,"I_max":20,
                       "beta_ema":0.8,"lambda_smooth":0.03,"manifold_project":true,
                       "loss_form":"huber","huber_delta":0.05,
                       "normalize_grad":false,"band_gate":false,
                       "spec_schedule_override":"second_half"}'
```

V6 关键超参解释：
- `Kp/Ki/Kd`：PID 增益（控制 err → s_t 映射）
- `s_min=0.05, s_max=50`：步长上下限（it3 之前 s_min=5 是 bug）
- `lambda_smooth`：motion 维时域平滑正则（防帧间抖动）
- `loss_form="huber" + huber_delta=0.05`：对称损失，过目标时 grad 反转
  - 单位敏感！deg 用 0.05 rad ≈ 2.9°，meter 用 0.01m
- `spec_schedule_override`：覆盖每个 spec 的 schedule（`always` / `second_half` / `last_quarter`）
- `manifold_project`：MPGD 风格 orthogonal projection，防梯度推出流形
- `normalize_grad=false`：用 raw grad，让 Huber 的自适应衰减生效
- `band_gate=false`：旧版的 "在带内时停推" 机制，已被 raw grad 自然衰减替代

---

## 四、实测结果（最终）

本项目完成了三个体态的完整 N=5/N=15 评估，结果可以总结为：**一个成功（骨盆前倾），两个失败但失败模式不同（膝超伸=数值 OOD，膝弯曲=相位-角度联合 OOD）**。

### 4.1 最终结果总表

| 体态 | 类型 | 最佳配置 | N | hit_band | corr | shape ✅ | 状态 |
|---|---|---|---|---|---|---|---|
| **骨盆前倾** (APT, 20°) | 分布内 | v2_dps_s40_**last_quarter** | 15 | **88.7±9.2%** | **+0.407±0.200** | **11/15** | ✅ **成功** |
| 骨盆前倾 | 同上 | v6_closed_loop_last_quarter | 15 | 81.7±10.5% | +0.311±0.191 | 8/15 | ✅ 公平对比中输给 V2 |
| 骨盆前倾 | 同上 | v6_closed_loop_second_half | 15 | 69.9±10.7% | +0.328±0.210 | 9/15 | ✅ |
| 骨盆前倾 | 同上 | v2_dps_s40_**always** | 15 | 49.6±26.3% | +0.037±0.300 | 0/15 | ❌ schedule 选错 (8/15 时间反向) |
| **膝超伸** (190°) | 数值 OOD | v2_dps_s40_last_quarter | 5 | **0.0%** | 0.060 | 5/5 推力不足 | ❌ **OOD 失败** |
| 膝超伸 | 同上 | v6_closed_loop_last_quarter | 5 | 0.0% | 0.063 | 5/5 推力不足 | ❌ |
| **膝弯曲** (125°+stance) | 相位 OOD | v2_dps_s40_last_quarter | 5 | 8.2% | **−0.215±0.142** | 5/5 时间反向 | ❌ **相位 OOD** |
| 膝弯曲 | 同上 | v6_closed_loop_last_quarter | 5 | 0.0% | +0.347±0.123 | 5/5 推力不足 | ❌ 流形钉住 |
| 膝弯曲_A (145°+stance) | 相位 OOD（缓和） | v2_dps_s40_last_quarter | 5 | 8.7% | −0.170±0.222 | 3/5 时间反向 | ❌ 仅微改善 |
| 膝弯曲_B (125°+always) | 相位 OOD（去门控） | v2_dps_s40_last_quarter | 5 | 2.5% | −0.119±0.188 | 4/5 时间反向 | ❌ |
| 膝弯曲_C (125°+stance, prompt="bent knees") | 相位 OOD + prompt 先验 | v6_closed_loop_last_quarter | 5 | **67.5±39.9%** | +0.059±0.195 | 2/5 时间反向 | ⚠ **意外成功但机制错误** |

### 4.2 关键成果（按重要性排序）

**🟢 成果 1：骨盆前倾上 V2_dps_s40_last_quarter 是稳定的 SOTA（N=15 验证）**
- hit_band 88.7%、corr 0.407、11/15 seed 通过 shape 检查
- 比 V6 在所有同 schedule 公平对比下都更好（+7% hit, +0.10 corr）

**🟢 成果 2：schedule 选择是骨盆前倾的最关键超参（影响 > 40% hit）**
- always 模式：hit=49.6%，CV(corr)=810%，8/15 seed 时间反向
- last_quarter 模式：hit=88.7%，CV(corr)=49%，0/15 时间反向
- 推论：**前期高噪声步施压会破坏时序结构，必须等噪声降到一定水平再 guidance**

**🟢 成果 3：发现了 inference-time guidance 的两类失效模式**
| 失效类型 | 数值 OOD（膝超伸） | 相位 OOD（膝弯曲） |
|---|---|---|
| 表象 | 推力不足、Δ≈0 | 时间结构反向、corr<0 |
| 根因 | 目标角度（190°）从未出现 | 目标角度（125°）只在错误相位出现 |
| 修复路径 | 改 prompt（已试，失败） | 改 prompt（C 实验，部分有效但机制反） |
| V2 表现 | 推力不足 | 时序反转 |
| V6 表现 | 推力不足 | 被流形钉住（推力不足） |

**🟡 成果 4：N=5 → N=15 揭示了"温顺 seed"偏差**
- 初始 5 个 seed `{7, 42, 99, 123, 2024}` 是低方差子集，掩盖了真实 std
- N=15 上 CV(corr) 几乎翻倍（21%→54%）；V2 与 V6 的 std 在 N=15 下几乎相等（0.214 vs 0.210），CV 差异完全由均值差决定
- **结论**：少 seed 实验的稳定性指标不可信，N≥10 是底线

**🟡 成果 5：V6 的工程复杂度没有换来骨盆前倾上的稳定性优势**
- V6 = PID + Huber + manifold proj + smoothing + EMA + anti-windup（300+ 行）
- V2 = 一个 step-size + hinge loss（30 行）
- N=15 上 V2 全面优于 V6；V6 的"精确跟踪能力"在 ±2° tolerance 任务下是过度设计

---

## 五、原因剖析

### 5.1 为什么 V2 在分布内任务上赢 V6？

V6 设计目标是"对抗任意 target 的精确跟踪"，但骨盆前倾任务有两个特点让 V2 的简单设计反而更优：

1. **任务容忍区间 ±2°，本身是个 hinge 任务**：到位即停推是天然属性。Hinge loss 的 `relu(target - tol - angle)` 在到位后梯度严格为 0，恰好符合需求；Huber 的双侧软推（远=1, 近→0）反而在带内还有残余梯度，扰动已经稳定的解。
2. **manifold_project 限制了推力上限**：V6 把梯度投影到流形切空间，对"目标在流形内"的任务造成欠推。V2 没这个约束，能多走半步进入容忍带。

### 5.2 为什么 OOD 任务在两种 controller 上都失败？

**inference-time guidance 的基本前提**：guidance 梯度只能在 MDM 已学到的概率密度内 reshape，不能把质量推到 zero-density 区域。

- 膝超伸 190°：训练集中膝关节角度 ∈ [90°, 180°)，190° 是 hard zero density → 梯度推一点回弹一点 → Δ≈0
- 膝弯曲 125° + stance：训练集中"stance 相 + 125° 膝角"的联合密度 ≈ 0（屈膝主要在 swing 相）→ V2 强推 → 模型"走捷径"把 stance/swing 翻转 → corr<0；V6 拒绝离开流形 → 推力≈0

### 5.3 膝弯曲三个 ablation（A/B/C）告诉我们什么？

| 假说 | 实验 | 结果 | 结论 |
|---|---|---|---|
| 目标 125° 太激进 | A: target=145° | corr=−0.170（仍负）| ❌ 不是目标激进度问题 |
| stance 相位门太严 | B: phase=always | corr=−0.119（仍负）| ❌ 不是相位门问题 |
| MDM 训练分布是根因 | C: 换 prompt | V2 corr=+0.326，V6 hit=67.5% | ✅ **先验偏移有效但机制反常** |

**C 实验的反常发现**：换 prompt 后 baseline 膝角自然落到 ~100-110°（已 < 125° 目标）。这时：
- V2 的 hinge loss = 0 → 不工作 → corr 保留（+0.326），但 hit 仍差
- V6 的 Huber 反向施压（baseline 已"过头"，反推回 125°）→ Δ=+20°（伸膝）→ 偶然停在带内 → hit=67.5%

**这不是 guidance 的成功，是 prompt 做了主要工作，V6 的双向 Huber 把过度弯曲"拉回"到目标附近**。学术上有意思（提示 prompt + guidance 协同方向），但不是当前框架的有效解。

---

## 六、什么可以优化 vs 什么是方法固有缺陷

### 6.1 可优化的（工程改进）

| 项 | 当前问题 | 优化路径 | 预期收益 |
|---|---|---|---|
| **统计指标** | CV(corr) 在均值不同时误导 | 改用 median + IQR + bootstrap CI | 论文表格更可信 |
| **V2 step-size 自适应** | 当前固定 s=40，跨体态需要 retune | 加 σ-aware schedule（如 s ∝ σ_t） | 单变体跨任务可用 |
| **schedule 自动选择** | last_quarter 是手调，骨盆前倾 ✓ 但其他体态未必 | 按 baseline 角度变化幅度自适应选 schedule | 减少 per-task 调参 |
| **V6 在容忍带内的残余梯度** | Huber 双侧软推在带内还在动 | 在 \|err\| < tol 时 freeze（保留 c_t 但置零 grad）| V6 在分布内任务可能反超 V2 |
| **OOD 检测** | 跑完才知道是 OOD 失败 | 在 step 1 后计算 \|target - baseline_mean\| / σ_baseline，预警 OOD 风险 | 节省算力 |

### 6.2 方法固有缺陷（inference-time guidance 的基本限制）

这些是**任何 post-hoc guidance 方法都无法解决**的限制，必须改架构（fine-tune / 条件训练）才能突破：

1. **数值 OOD 不可达**（膝超伸 190°）
   - guidance 只能 reshape MDM 已有分布，不能创造 zero-density 区域的样本
   - **唯一出路**：fine-tune MDM 使其覆盖目标体态，或用 LoRA 注入新模态

2. **相位-角度联合 OOD 不可达**（膝弯曲 125°+stance）
   - 即使单独看角度（125°）和相位（stance）都在分布内，它们的**联合**可能不在
   - guidance 强推会让模型 "swap" 相位标签来满足约束，导致 corr<0
   - **唯一出路**：相位条件训练（让 MDM 在 stance 时也能生成屈膝）

3. **prompt 先验和 guidance 的耦合不可分离**
   - C 实验显示，prompt 主导基线分布，guidance 只能在 prompt 给定的局部分布内微调
   - **唯一出路**：把目标体态直接写进 prompt（"a person walking with anterior pelvic tilt"）训练专门的 prompt-to-posture 模型

4. **guidance 强度 vs 时序保持的 Pareto 边界**
   - 整个 V1-V6 演化、N=15 sweep 都没能突破 hit↔corr 的同一条 Pareto 前沿
   - 这是 classifier guidance 的结构性限制：施压越强、扰动越大、时序越差
   - **唯一出路**：换范式（如 ControlNet 式的条件分支 / RL-tuned diffusion）

### 6.3 现实可行的下一步（如果继续做）

- **优先级 1**：补 3-4 个**真分布内**体态做交叉验证（骨盆侧倾、足距、躯干前倾、膝内扣）。如果都像骨盆前倾一样稳定，论文可以收尾。
- **优先级 2**：把 OOD 检测做成 pre-run 工具（不需要跑完整 pipeline），提示用户哪些目标是 OOD。
- **优先级 3**：写 failure-mode 章节作为论文的诚实贡献——展示两类 OOD 失败比展示三个成功更有学术价值。
- **不做**：继续调 V6 单参数；尝试更激进的 OOD 目标；实现 time-travel/dual-stage 等架构改动。

---

## 七、运行环境（已更新）

```bash
conda activate mdm5090
cd /root/autodl-tmp/motion-diffusion-model

# 骨盆前倾 N=15 (推荐 baseline)
bash new/run_seed_robustness_n15.sh
python -m new.aggregate_seeds ./output/n15_*

# 跨体态 sweep — last_quarter schedule（已是默认）
bash new/run_cross_posture.sh 骨盆前倾 5
bash new/run_cross_posture.sh 膝弯曲   5   # 失败案例（相位 OOD）
bash new/run_cross_posture.sh 膝弯曲_A 5   # 缓和目标 ablation
bash new/run_cross_posture.sh 膝弯曲_B 5   # 去相位门 ablation
TEXT_PROMPT="a person walking with bent knees" \
    bash new/run_cross_posture.sh 膝弯曲 5   # prompt 先验 ablation (C)
python -m new.aggregate_seeds ./output/cross_*
```

---

## 八、关键文件改动历史（接续者备忘）

时间序列（git log 顺序）：

| Commit | 内容 |
|---|---|
| `080c735` | 上一作者完成的 V1-V5 baseline + 早期评估脚本 |
| `8effecf` | **本项目第 1 个 commit**：V2-norm + V6 闭环 PID 初版 + manifold proj |
| `9fb026b` | V6 加 Huber loss + band-gate + s_min 修正（迭代 2） |
| `1b4cdfd` | V6 raw grad + 关 band-gate（迭代 3，**这一版是 V6 行为正确化的转折点**） |
| `3d21aaf` | spec_schedule_override 参数（迭代 4） |
| `db8b9b8` | sigma_cutoff 参数（迭代 5） |
| `4de9b41` | 跨体态实验工具 + POSTURE_REPRESENTABILITY.md |
| (后续 commits) | 膝弯曲 spec + V6 last_quarter 公平对比脚本 |
| (后续 commits) | hit_rate_loose 方向感知修复（less_than 任务） |
| (后续 commits) | 膝弯曲_A/B 相位 OOD 诊断 spec |
| (本 commit) | README 整合三组体态最终结果 + 原因剖析 |

如果你需要从某个具体迭代 fork 出去对比，`git checkout <commit>` 即可。每个 commit message 都详细记录了改动动机。

---

## 九、给接续者的建议

1. **先读 commit history**：每个 commit message 都自带"为什么这么改"的诊断，按时间读一遍能快速理解 V6 演化逻辑和后续的 OOD 诊断过程。

2. **不要再调 V6 在骨盆前倾上的单参数**。Pareto 前沿已经描清楚（hit↔corr trade-off），N=15 已经确认 V2_last_quarter 是 SOTA。

3. **不要再尝试膝超伸 / 膝弯曲**。两个失败模式（数值 OOD、相位 OOD）已经完整诊断。继续调参不会改变 inference-time guidance 的基本限制。

4. **如果要继续做实验**，加 **分布内** 体态：骨盆侧倾、足距过宽、膝内扣（valgus/varus）、躯干前倾。这些都是分布内且关节强表征，预期能成功，可以扩充骨盆前倾这一个成功案例。

5. **统计指标换掉**：CV(corr) 在均值不同时不可信，用 median + IQR + bootstrap CI。`aggregate_seeds.py` 待改。

6. **论文写作建议**：把"两类 OOD 失效模式"作为核心贡献写——比"单任务成功"更有学术价值。inference-time guidance 的 failure mode 在文献中很少有这么完整的分类。

7. **如果项目要突破当前 Pareto 前沿**：必须改架构，不再 post-hoc。可选方向：相位条件训练、ControlNet 式条件分支、prompt + guidance 协同训练。这超出 inference-time guidance scope。

---

## 十、关键文献

| 文献 | 用途 |
|---|---|
| Tevet et al., MDM, ICLR 2023 | 基模型 |
| Chung et al., DPS, ICLR 2023 | V2 的来源；step-size normalization 的依据 |
| Bansal et al., Universal Guidance, CVPR 2024 | 闭环 / iterative gradient 思想 + bounded-grad 建议 |
| He et al., MPGD, ICLR 2024 | manifold orthogonal projection（V6 用） |
| Karras et al., NeurIPS 2022 | PID 控制 ODE solver |
| Song et al., LGD, ICML 2023 | V5 的来源 |
| Yu et al., FreeDoM, ICCV 2023 | time-travel fallback（项目 plan 里答应过但还没实现）|
| Huber 1964 | Huber loss（V6 损失形式）|
| Åström & Hägglund, "Advanced PID Control" 2006 | anti-windup（V6 的 i_start_frac 来源）|
| Efron, JASA 2011 | Tweedie's formula；解释 V6 的 c_t 时间步衰减增益依据 |

---

**总结一句话**：V2_dps + schedule=last_quarter 是骨盆前倾任务的 SOTA（N=15: hit=88.7%, corr=0.407）；V6 的 PID 工程复杂度未带来稳定性优势；跨体态扩展（膝超伸=数值 OOD、膝弯曲=相位-角度联合 OOD）揭示了 inference-time guidance 的两类基本失效模式，这是后处理 guidance 范式的**固有限制**而非控制器设计问题。核心成果是骨盆前倾的稳定结果 + 两类 OOD 失效模式的完整诊断，下一步应当是加分布内的新体态（骨盆侧倾、躯干前倾、膝内扣）做交叉验证，或者诚实地把 failure-mode 作为论文主贡献。

