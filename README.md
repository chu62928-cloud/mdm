# Posture Guidance on MDM — 项目接续总结

> 这份文档面向**刚接到这个项目的人**。读完应能理解：我们做了什么、当前演化到哪一版、各体态的实测结果、为什么会成功/失败、什么是工程问题 vs 方法固有缺陷、如果要继续做该做什么。

> ⚠ **阅读顺序提示（2026-06 更新）**：本文档按时间分层。§零–§十 是主体（APT/躯干前倾成功 + 两类 OOD 失效）；§十一 是 Route B（IK+SDEdit 撞 182° 墙）；**§十二 是最新一轮（距离度量 + 四度量裁决 + 膝超伸真伪判定），它修正了 §五.2 和 §十一 中关于"膝超伸"的部分结论**。如果你关心膝超伸，先读 §十二，再回头看 §十一。

> ⚠ 老版 `PROJECT_SUMMARY.md` 有两个核心误判（已纠正于"零、老版误判"一节），如果你看到的还是老版，请优先以本文档为准。

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
│   ├── angle_ops.py                   ← pelvis_tilt_angle / signed_knee_angle /
│   │                                     signed_knee_distance_sagittal【新增】/
│   │                                     signed_knee_angle_sagittal【新增,裁判角】 等
│   ├── joint_indices.py               ← 22 个 SMPL 关节索引常量
│   ├── phase_detector.py              ← 步态相位 mask（get_stance_mask）
│   ├── controller.py                  ← PostureGuidance 入口，loss_form="hinge"|"huber"
│   ├── guidance_variants.py           ← V1-V5 实现（参考）
│   ├── mdm_integration.py             ← make_fk_fn + V1 的 apply_posture_guidance
│   └── closed_loop_controller.py      ← V6 用的 PID 控制器 + orthogonal_project
│
├── new/                               ← 实验脚本层
│   ├── evaluate_ablation_v3.py        ← 单 sweep 评分 + shape 分类
│   ├── aggregate_seeds.py             ← 跨 seed 聚合
│   ├── run_cross_posture.sh           ← 跨体态 wrapper
│   ├── analyze_knee_angles.py         ← 训练集三点角分布
│   ├── analyze_knee_distance.py       ←【新增】训练集 signed_distance 分布
│   ├── analyze_overext_phase.py       ←【新增】超伸帧的 gait 相位归属
│   ├── analyze_dynamics.py            ←【新增】foot skating / 位移 / trunk lean
│   ├── POSTURE_REPRESENTABILITY.md    ← 22 关节下哪些体态可表征
│   └── ...
│
├── OOD_TAXONOMY.md                    ← OOD 四分类（膝超伸 = Class 1，见 §十二修正）
├── PLANA_RESULTS.md                   ← Plan A 距离度量结果（部分结论已被 §十二 修正）
├── ADJUDICATION_REPORT.md             ← 四度量裁决（Class 改分提议未采纳，见 §十二）
└── PROJECT_SUMMARY.md / README.md     ← 本文档
```

所有改动都在分支 `claude/fix-code-bugs-optimization-final`。

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
  - **单位敏感！deg 用 0.05 rad ≈ 2.9°，meter 用 0.01m**（见 §十二距离度量的单位适配）
- `spec_schedule_override`：覆盖每个 spec 的 schedule（`always` / `second_half` / `last_quarter`）
- `manifold_project`：MPGD 风格 orthogonal projection，防梯度推出流形
- `normalize_grad=false`：用 raw grad，让 Huber 的自适应衰减生效
- `band_gate=false`：旧版的 "在带内时停推" 机制，已被 raw grad 自然衰减替代

---

## 四、实测结果（角度度量，最终）

本项目完成了四个体态的完整 N=5/N=15 评估：**两个成功（骨盆前倾、躯干前倾），两个失败但失败模式不同（膝超伸=数值/几何 OOD，膝弯曲=相位-角度联合 OOD）**。

> ⚠ 膝超伸"失败"的结论在 §十二 被**部分修正**：用距离度量 + 步态保持的 last_quarter，guidance 实际**能**生成真超伸（渲染确认），只是停在边界外一小段。下表的 hit=0% 是用 acos 三点角（tpa）测的，而 tpa 在 >180° 区完全失明（§十二）。

### 4.1 最终结果总表

| 体态 | 类型 | 最佳配置 | N | hit_band | corr | shape ✅ | 状态 |
|---|---|---|---|---|---|---|---|
| **骨盆前倾** (APT, 20°) | 分布内 | v2_dps_s40_**last_quarter** | 15 | **88.7±9.2%** | **+0.407±0.200** | **11/15** | ✅ **成功** |
| 骨盆前倾 | 同上 | v6_closed_loop_last_quarter | 15 | 81.7±10.5% | +0.311±0.191 | 8/15 | ✅ V2 优 |
| **躯干前倾** (trunk lean, 15°) | 分布内 | v2_dps_s40_last_quarter | 15 | **88.9±10.3%** | **+0.295±0.254** | 8/15 | ✅ **成功（高 hit）** |
| 躯干前倾 | 同上 | v6_closed_loop_last_quarter | 15 | 37.2±17.1% | **+0.476±0.242** | 10/15 | ✅ **成功（高 corr）** |
| **膝超伸** (190°) | 几何 OOD | v2_dps_s40_last_quarter | 5 | **0.0%(tpa)** | 0.060 | — | ⚠ 见 §十二修正 |
| **膝弯曲** (125°+stance) | 相位 OOD | v2_dps_s40_last_quarter | 5 | 8.2% | **−0.215±0.142** | 5/5 时间反向 | ❌ **相位 OOD** |
| 膝弯曲_C (prompt="bent knees") | 相位 OOD + prompt | v6_closed_loop_last_quarter | 5 | **67.5±39.9%** | +0.059±0.195 | 2/5 时间反向 | ⚠ prompt 主导 |

### 4.2 关键成果（按重要性排序）

**🟢 成果 1：骨盆前倾上 V2_dps_s40_last_quarter 是稳定的 SOTA（N=15 验证）** — hit_band 88.7%、corr 0.407、11/15 seed 通过 shape 检查。

**🟢 成果 2：躯干前倾揭示 corr↔hit Pareto 前沿** — V2_s40 高 hit（88.9%）/ V6 default 高 corr（0.476），三轮消融证明两者在同一条 Pareto 曲线上。

**🟢 成果 3：schedule 是关键超参（影响 > 40% hit）** — always: hit=49.6%, 8/15 时间反向；last_quarter: hit=88.7%, 0/15 时间反向。

**🟢 成果 4：发现 inference-time guidance 的两类失效模式**
| 失效类型 | 数值/几何 OOD（膝超伸） | 相位 OOD（膝弯曲） |
|---|---|---|
| 表象 | 推力不足、Δ≈0（tpa 视角）| 时间结构反向、corr<0 |
| 根因 | 目标在训练集零密度（但见 §十二）| 目标角度只在错误相位出现 |

**🟡 成果 5：N=5 → N=15 揭示"温顺 seed"偏差** — CV(corr) 21%→54%，**N≥10 是底线**。

### 4.3 躯干前倾三轮消融：corr 是推力幅度 Δ 的函数

三轮系统消融（manifold/smooth、Huber/c_t、V2 步长扫描）证明：**corr 完全由推力幅度 Δ 决定，与 V2/V6 架构无关**。但 V6 的真实架构优势：**相同 Δ 下 hit 效率是 V2 的 2.1 倍**（V2_s20: hit/Δ=1.16%/°；V6_noBoth: hit/Δ=2.44%/°）——PID 误差反馈把推力精准集中在"还没到位"的样本上。

---

## 五、原因剖析

### 5.1 为什么 V2 在分布内任务上赢 V6？
1. 任务容忍区间 ±2°，本身是 hinge 任务，到位即停推是天然属性；Huber 双侧软推在带内还有残余梯度，扰动已稳定的解。
2. manifold_project 限制推力上限，对"目标在流形内"的任务造成欠推。

### 5.2 为什么 OOD 任务在两种 controller 上都失败？

**inference-time guidance 的基本前提**：guidance 梯度只能在 MDM 已学到的概率密度内 reshape，不能把质量推到 zero-density 区域。

1. **膝超伸 190°**：训练集膝角 ∈ [90°, 180°)，190° 是 hard zero density → 梯度推一点回弹一点。
   - ⚠ **§十二修正**：用**距离度量**（无 acos 奇点）+ 步态保持的 last_quarter，guidance 实际把膝推到了**直膝边界外约 10°**（sag 190°，渲染确认是真反弓超伸）。所以更准确的说法不是"完全做不到"，而是"**能越过边界外推一小段（~190°），但存在硬上限，推不到任意远（如临床深超伸或 IK 目标 190°+）**"。这与本节"零密度不可创造"的理论**不直接矛盾**——最可能是"边界外推"而非"零密度造密度"，但 A/B 二选一尚未由多 target 曲线裁定（§十二待办）。
2. **膝弯曲 125° + stance**：训练集"stance + 125°"联合密度 ≈ 0 → V2 强推使模型 swap 相位（corr<0）；V6 拒绝离开流形（推力≈0）。

### 5.3 膝弯曲三个 ablation（A/B/C）
- A（target=145°）corr 仍负 → 不是目标激进度问题
- B（phase=always）corr 仍负 → 不是相位门问题
- C（换 prompt）→ baseline 自然落到 100-110°，是 **prompt 做主要工作**，非 guidance 成功

---

## 六、V2 vs V6 选择指南 + 可优化 vs 固有缺陷

### 6.0 V2 vs V6 使用边界

| 场景 | 推荐 | 理由 |
|------|------|------|
| 已知任务、优先 hit | **V2_s40** | 骨盆前倾 SOTA (hit=88.7%) |
| 已知任务、优先 corr | **V2_s=10~15** | corr 可达 0.57+ |
| 平衡 hit+corr | **V6 default** | 自动找平衡点 |
| 未知体态、不想 retune | **V6** | PID 自适应 |

**核心结论**：corr/hit 在同一条 Pareto 曲线上，由 Δ 控制；V6 真实优势是相同 Δ 下 hit 效率 2.1×。两者都没突破 hit↔corr Pareto 前沿。

### 6.1 可优化的（工程改进）
- 统计指标改 median + IQR + bootstrap CI（CV 在均值不同时误导）
- V2 step-size σ-aware schedule；schedule 自动选择
- V6 在容忍带内 freeze grad
- OOD 预检测（step 1 后算 |target - baseline_mean|/σ）

### 6.2 方法固有缺陷（须改架构才能突破）
1. **数值/几何 OOD 大幅外推不可达**（膝超伸推不到 190°+，见 §十二）→ 出路：fine-tune / LoRA
2. **相位-角度联合 OOD 不可达**（膝弯曲 125°+stance）→ 出路：相位条件训练
3. **prompt 先验和 guidance 耦合不可分离** → 出路：prompt-to-posture 训练
4. **guidance 强度 vs 时序保持的 Pareto 边界** → 出路：换范式（ControlNet / RL-tuned）

---

## 七–十（运行环境、改动历史、接续建议、文献）

> 此处保留原文（运行命令、git log、文献表）。关键运行命令：

```bash
conda activate mdm5090
cd /root/autodl-tmp/motion-diffusion-model
source /etc/network_turbo   # ★ 必须：服务器需代理访问 HuggingFace BERT

bash new/run_cross_posture.sh 骨盆前倾 5
python -m new.aggregate_seeds ./output/cross_*
```

**关键文献**：MDM (Tevet ICLR2023)、DPS (Chung ICLR2023)、Universal Guidance (Bansal CVPR2024)、MPGD (He ICLR2024)、Karras NeurIPS2022 (PID)、LGD (Song ICML2023)、FreeDoM (Yu ICCV2023)、Huber 1964、Åström&Hägglund 2006、Efron JASA2011。新增应引：**「What does guidance do?」NeurIPS2024 (arXiv:2409.13074)**、**「Applying Guidance in a Limited Interval」NeurIPS2024 (arXiv:2404.07724)**、**LoRA-MDM「Dance Like a Chicken」CGF2025 (arXiv:2503.19557)**、**GAITGen NeurIPS2025 (arXiv:2503.22397)**。

---

## 十一、OOD 病态体态生成 — Route B：IK 注入 + SDEdit（2026-06）

> 这是一条**根本不同的方法**——不靠 guidance 注入密度，而是用 IK 强制构造 190° 超伸序列，再用 SDEdit（加噪-去噪）让 MDM 恢复自然性。结论：撞上可量化的硬上限。

### 11.1–11.4 要点
- 目标：生成膝超伸 190° 行走。`analyze_knee_angles.py` 确认 HumanML3D 全集**三点角** Max=179.95°（训练集不存在膝向后弯几何）。
- 五条路线（A 组合扩散 / B IK+SDEdit / C score 抑制 / D DOODL / E LoRA），逐条决策见原表。
- 模型身份核实：`humanml_trans_dec_512_bert-50steps`，trans_dec+BERT，50 步，非自回归 DiP。

### 11.5–11.6 Route B 结果：撞上 182° 硬上限
- 最强信号"推力越大结果越低"：Kp=160 给 179.3° < Kp=80 的 181.7°，CV 随推力单调上升（1.2→1.9→2.5）。这是 OOD 拔河的数学本质——MDM 回拉力随偏离流形距离超线性增长。
- **结论：182° 是该模型在膝超伸方向的推理时硬上限。**

> ⚠ **§十二的关键修正**：这个 182° 是用 `signed_knee_angle`（ska）测的。§十二证明 ska 在**超伸区可信**（渲染确认），所以这个 182° 墙**对 IK+SDEdit 路线仍然成立**。但 ska 在**深屈膝区会绕回误报**（sag=273° 实为蹲），因此凡涉及屈膝/蹲的分布统计不能用 ska。Route B 的 182° 结论本身不受影响（它在超伸区）。

### 11.7 接下来方向
1. Route E：LoRA 少样本微调（注入缺失结构，对标 Dance Like a Chicken）
2. 把 182° 上限作为论文贡献
3. 验证 Route B 在条件 OOD 上的有效性
- ✗ 不做：继续在膝超伸上调推理参数（11 配置已证无效）

---

## 十二、距离度量 + 四度量裁决 + 膝超伸真伪判定（2026-06，本轮，**最新**）

> 这一节是对 §四.2 和 §十一中"膝超伸"结论的修正。核心：换一个无奇点的度量后，guidance **确实**能生成真超伸（渲染确认），且暴露出旧的 acos 度量在超伸评估中系统性失明。但也发现新度量在深屈膝区会绕回误报，因此一连串"修复"中有的成立、有的必须撤回。

### 12.1 动机：质疑 acos 测量天花板

`signed_knee_angle`（ska）用 acos 三点角 + z-offset sigmoid，acos 在 180° 梯度饱和/NaN，造成"测量 artifact 天花板"（~182°）。问题：膝超伸失败中，多少是测量 artifact、多少是模型先验，无法分辨。

**Plan A**：用 `signed_knee_distance_sagittal`（膝到髋-踝连线的有符号矢状面**距离**）替换 acos 夹角。梯度在 dist=0（直膝）处连续，无奇点；正值=膝在前（正常），负值=膝在后（超伸）；target=-0.05m（约等效 6° 超伸）。

### 12.2 Plan A 距离度量实测（N=5）

| 配置 | dist guided | hit(<-0.05m) | 步态 | 判读 |
|---|---|---|---|---|
| V2 always (50 步) | -0.090m | 92.1% | ❌ **破坏**（43% 倒退、位移减半 1.43m vs 2.60m、corr=-0.201） | **假阳性** |
| V2 last_quarter (~12 步) | -0.036m | 11.9% | ✅ 保持（93% 前进） | 真推动 |
| V6 last_quarter | +0.063m | 0% | ✅ 最优 | 流形投影把正交推力归零 |

**结论**：V2 always 的 92% 是暴力破坏步态后偶然达标，弃用。**V2 last_quarter 才是有效结果——步态保持下膝被推到连线后方。**

### 12.3 度量分家危机与四度量裁决

**问题**：实际 guidance/渲染用 `ska`，评估列用 `three_point_angle`（tpa）。两者在 >180° 反向变化（tpa 是 acos，≤180°，超伸时反而下降）。用 ska 报"突破 182°"是**循环论证**（用被质疑的尺子证明超过它自己的刻度）。

**解决**：引入第三方裁判 `signed_knee_angle_sagittal`（sag）——矢状面内 acos 底角 + cross-product 符号，值域 (0°, 360°)，**既不用 z-offset 启发式（不像 ska），也不用 acos 硬截断（不像 tpa）**。（注：atan2 版有 180° discontinuity，已弃用，见 ADJUDICATION_ISSUES.md #1。）

**裁决结果**（5 seeds，guided）：

| 度量 | Baseline | Guided | Δ |
|---|---|---|---|
| `sag`（真几何角）| 163° | **190°** | **+27°** |
| `ska`（z-offset）| 160° | 194° | +34° |
| `tpa`（acos≤180°）| 158° | 161° | +3° |
| `sds`（距离）| +0.056m | -0.040m | -0.096m |

ska>180° 的帧中 **98.4% 在 sag 下也 >180°** → ska 的 192° 被独立裁判确认，tpa 的 +3° 是 acos 上限造成的系统性低估（差 60°+）。

### 12.4 渲染判定（决定性，绕开所有度量）

肉眼看侧视图，绕开一切度量构造：

- **训练集 sag=273° 帧（Image 1）**：实为**蹲/深屈膝**，膝在髋-踝连线**前方**。273° 是荒谬值 → **sag 在深屈膝区会绕回误报**。
- **v2_last_quarter guided 帧（Image 2）**：baseline 173.2° vs guided 194.7°，小人膝盖**明显向后顶呈反弓** → **肉眼确认真超伸**。

**核心结论：sag 度量分区可信——超伸区可信，深屈膝/蹲区不可信（>180° 绕回）。**

### 12.5 三个确定结论

1. **guided 膝超伸是真的**（Image 2 肉眼确认）。正面结果成立。
2. **膝超伸维持 Class 1**（几何支撑集外）。裁决报告曾提议改 Class 2，依据是"训练集 2680 帧 sag>190°"——但 §12.4 证明那 2680 帧是**蹲的误报**（它们 tpa 中值仅 145°=明显屈膝）。**Class 1→2 改分必须撤回；OOD_TAXONOMY.md 维持 Class 1（当前文件已是 Class 1，无需改）。** Phase 0 用 tpa 判"训练集无超伸几何"反而是**对的**。
3. **训练集零密度 + guidance 却生成超伸**，看似与"做不到几何 OOD"主线冲突，两种解释：
   - **解释 A（强烈倾向）**：边界外推一小段（~190°，比直膝超 10°）然后撞硬墙。与膝 190° 撞 182° 墙自洽（182° 也是 sag/ska 在超伸区测的，可信）。
   - **解释 B（需强证据）**：零密度区凭空造密度。与全部既有结论矛盾。
   - **未决**——由 Phase D 多 target 曲线裁定（§12.7）。**曲线出来前不要写"突破零密度"。**

### 12.6 步态质量与刻画（Phase G gate 数据）

- **G1 相位**：75% 超伸帧在 stance，97.8% stance 帧超伸。但散点图显示 stance 红点**全程恒定钉在 192-194°**（像给膝角加了偏置，**非相位选择性**）；swing 帧被撕扯下掉到 160-180°——**这是"迈腿不自然"的根因**（摆动相膝角在超伸先验和自然屈曲间对抗）。
- **G2 facing**：guided fz_mean≈0.99，fz_neg≈0（5 seeds）→ facing 在 guided 序列上一致，三度量共享的几何地基稳固。
- **G4 动力学**：位移大体保持（非 always 那种坍塌）；足部轻微穿地（foot_min_y≈-0.0004m，可忽略）；trunk lean 比 baseline 多 2-4°。步态可信。
- **临床合理性**：guided 190° = 比直膝超伸 10°，落在临床 genu recurvatum 范围（5-15°）内，**幅度合理**。真正存疑的是**全程恒定**（真 genu recurvatum 是承重相选择性过伸，不是全程）。

### 12.7 待办（按优先级）

| 优先级 | 任务 | 说明 |
|---|---|---|
| **立刻** | 撤回 Class 1→2 改分（已确认维持 Class 1）；记录 sag 在深屈膝区不可信、Class 判定需 sag+tpa+渲染**三者交叉验证**的教训 | 防止下一个人被 ADJUDICATION_REPORT 的改分提议误导 |
| **立刻** | 给 sag 加值域保护：`sag>210°` 或 `(tpa<160° & sag>180°)` 的帧标记为"度量失效/深屈膝" | 防止训练集分布统计被蹲帧污染 |
| **关键** | **Phase D 多 target 曲线**（target=-0.03/-0.05/-0.08/-0.12m），看达成超伸角**饱和**（解释 A，撞墙）还是**单调上升到 210°+**（解释 B，造密度）| 裁定 A vs B；曲线出来前不写"突破零密度" |
| 刻画 | 确认 guided 超伸是"全程恒定 192°"（偏置，非生理）还是"stance 选择性"（genu recurvatum）——把 guided sag 沿 gait cycle 画出来 | 决定临床合理性 |
| 论文最稳 | 把"tpa 在训练集超伸帧上比 sag/真几何角低 60°+"做成**独立方法学贡献**（为何 acos 三点角不能评估膝超伸）| **不依赖 guidance 真伪**，是最硬的产出 |
| 扩展 | N≥10（当前 N=5 是温顺 seed 偏差，用 median+IQR+bootstrap CI）；双膝对称；schedule pareto（hit vs 步态质量）| gate/AvsB 都过后再做 |

### 12.8 产物清单

| 文件 | 用途 |
|---|---|
| `posture_guidance/angle_ops.py` | 新增 `signed_knee_distance_sagittal`（guidance 用）+ `signed_knee_angle_sagittal`（裁判用）|
| `new/analyze_knee_distance.py` | 训练集 signed_distance 分布 |
| `new/analyze_overext_phase.py` | 超伸帧 gait 相位归属（G1）|
| `new/analyze_dynamics.py` | foot skating / 位移 / trunk lean（G4）|
| `PLANA_RESULTS.md` | Plan A 结果（注意：基于 tpa 的"膝变直 4°"结论已被 §12 修正为 sag +27°）|
| `ADJUDICATION_REPORT.md` | 四度量裁决（注意：§3.4 的 Class 1→2 改分提议**未采纳**，维持 Class 1）|
| `new_results/cross_metric_table.csv` | 5 seeds × 120 帧 × 4 度量 |
| `new_results/tpa_distribution_overlay.png` | 论文反面教材图：为何不能用 tpa 评估超伸 |
| `output/kneedist_v2_last_quarter/` | N=5 V2 结果（5 comparison.npy）|
| `new_results/kneedist_v2_last_quarter_seed42.mp4` | guided 超伸渲染（肉眼确认真反弓）|

### 12.9 关键教训（工程 + 方法学）

1. **几何度量的符号必须在真实数据上验证**——`facing = cross(up, lateral)` 的符号靠 comparison.npy 实测才定对（理论推导给反了，见 PLANA_ISSUES_LOG #1）。
2. **度量必须分区验证可信域，不能全局假定**——sag 在超伸区可信、深屈膝区绕回失效。一个在 A 区验证过的度量，不能直接用在 B 区。
3. **渲染是绕开所有度量构造的终极裁判**——3D 旋转视角会折叠矢状面信息（看不出超伸/屈膝），必须用矢状面侧视图或直接渲染人形。一帧侧视图解决了几轮的度量争议。
4. **系统中的度量必须独立于被评估的方法**——否则构成循环论证（用 ska 证明突破 ska 的天花板）。需要从未参与 guidance 的第三方裁判。
5. **高估比低估对论文更危险**——本轮一度从悲观（tpa 视角）跳到乐观（ska 视角），裁判出来前应默认偏保守。
6. **Class 判定需多度量 + 渲染交叉验证**——单一度量（无论 tpa 还是 sag）都可能在某个区域系统性出错。

---

## 十三、关节角 + 肌肉激活组合引导（整合，2026-06）

把"关节角约束（Module 1，本仓库）"与"肌肉激活约束（Module 2，`motion2muscle/`
的冻结代理 + 四分量 posture loss）"合并进**同一条 MDM 采样循环**，保持接口不变，
支持 **joint / muscle / both** 三种模式。

### 13.1 设计
- **统一组合 loss**（`posture_guidance/combined_loss.py` 的 `CombinedGuidance`）：
  `total = w_joint · L_joint − w_muscle · L_muscle`
  关节项 hinge/huber（到位即 0，最小化）；肌肉项 `MuscleGuidance.loss`（>0=越病态，
  取负号 → 等价梯度上升）。三模式由 `mode` 切换。
- **注入点**：统一走 **v2_dps**（README 实测 APT 最佳：`s=40 + last_quarter`），并保留
  **v6**。两个 variant 内部把关节 loss 换成 `combined.motion_loss(x0_hat, …)`，肌肉项的
  梯度天然穿过冻结代理（且在 v2/v6 里穿过 MDM）回到 `x_t`。其余 variant 行为不变。
- **两阶段采样**（肌肉 reference，HANDOFF §4.2）：muscle/both 模式下先无引导采样一次
  得正常样本 → `build_reference` 冻结 → 再带组合引导采样。`sample/generate.py` 编排。
- **归一化握手**：默认 `same_normalization=True`（代理与 MDM 同一套 HumanML3D Mean/Std）；
  否则用 `--proxy_mean_path/--proxy_std_path` 显式换算（见 HANDOFF §6.1）。

### 13.2 配置（env 优先，CLI 兜底）
```bash
GUIDANCE_MODE=both        # joint | muscle | both
JOINT_WEIGHT=1.0  MUSCLE_WEIGHT=1.0
GUIDANCE_VARIANT=v2_dps
GUIDANCE_KWARGS_JSON='{"s":40,"schedule":"last_quarter","base_weight":20}'
```
CLI：`--guidance_mode --joint_weight --muscle_weight --posture_instructions
--muscle_ckpt --muscle_posture --muscle_assets_dir --muscle_same_norm/--muscle_diff_norm`。

### 13.3 跑 APT 三模式对照
```bash
MODEL_PATH=<MDM_ckpt.pt> MUSCLE_CKPT=<net_best_loss.pth> bash new/run_apt_integrated.sh
```
纯链路自检（不需要权重/ckpt）：`python new/sanity_muscle_integration.py`（已验证全过）。

### 13.4 运行前置（由肌肉队补齐到 `motion2muscle/`）
1. 代理**模型类**定义（transformer，来自 motion2muscle-main）；2. `net_best_*.pth` 权重；
3. 代理 `Mean/Std`（或确认与 MDM 同一套 → `same_normalization=True`）。
缺 1/2 时 **joint 模式照常可跑**；`sanity_muscle_integration.py` 的纯 loss 自检也不需要权重。

---

## 总结一句话（更新）

两个分布内体态成功（骨盆前倾 N=15: hit=88.7%/corr=0.407；躯干前倾 N=15: hit=88.9%/corr=0.295）；三轮消融揭示 corr↔hit 由推力幅度 Δ 决定，V6 真实优势是相同 Δ 下 hit 效率 2.1×。**膝超伸的结论经本轮修正**：用无奇点的距离度量 + 步态保持的 last_quarter，guidance **能生成真超伸**（渲染确认 190°，比直膝超 10°），推翻了"完全做不到"的旧判断；但它停在边界外一小段，**最可能是"边界外推撞硬墙"而非"零密度造密度"**（A/B 待多 target 曲线裁定）。同时本轮暴露：旧的 acos 三点角（tpa）在 >180° 完全失明（比真几何角低 60°+），这是独立于 guidance 真伪的方法学贡献；而新的 sag 度量在深屈膝区会绕回误报，故膝超伸**维持 Class 1**、Class 改分提议撤回。**给接续者最重要的一句**：膝超伸的真伪只能靠 sag+tpa+渲染三者交叉验证；在 Phase D 多 target 曲线出来之前，不要写"突破零密度"——你自己膝 190° 撞 182° 墙的旧证据强烈指向"有硬上限"。