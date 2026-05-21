# Posture Guidance on MDM — 项目接续总结

> 这份文档面向**刚接到这个项目的人**。读完应能理解：我们做了什么、当前 V6 演化到哪一版、实测结果说明什么、未解决的根本问题是什么、接下来该跑什么实验。

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

## 四、实测结果（N=5 + N=15）

### 骨盆前倾 N=5（早期）

| variant | Δ | hit_band | corr | CV(corr) | shape ✅ |
|---|---|---|---|---|---|
| v2_dps_s40_always | +10.7±3.1° | 83.0±9.7% | 0.461±0.098 | 21.2% | 4/5 |
| v2b_x0_edit_bw5 | +12.9±2.7° | 56.2±15.6% | 0.514±0.140 | 27.2% | 4/5 |
| v6_closed_loop_test (it1) | +11.0±2.9° | 78.2±3.2% | 0.236±0.132 | 56.0% | 3/5 |
| v6_closed_loop (it3) | +9.6±2.8° | 83.7±6.7% | 0.351±0.119 | 34.0% | 3/5 |
| v6_closed_loop_second_half (it4) | +9.1±2.8° | 63.0±8.6% | 0.427±0.126 | 29.5% | **4/5** |
| v6_closed_loop_always (it4) | +8.9±2.8° | 54.3±9.9% | 0.454±0.149 | 32.9% | 3/5 |
| v6_closed_loop_sigma_cutoff_0.18 (it5) | +9.2±2.8° | 67.5±11.2% | 0.397±0.137 | 34.5% | 4/5 |

### 骨盆前倾 N=15（关键诊断！）

| variant | Δ | hit_band | corr | std(corr) | CV(corr) | shape ✅ |
|---|---|---|---|---|---|---|
| v2_dps_s40_always | +10.0±2.1° | **87.8±9.4%** | **0.396±0.214** | 0.214 | 54.0% | 11/15 |
| v6_closed_loop_second_half | +8.8±1.9° | 69.9±10.7% | 0.328±0.210 | 0.210 | 64.1% | 9/15 |

**关键洞察**：
1. **N=5 → N=15 std 翻倍**：原 5 个 seed `{7, 42, 99, 123, 2024}` 是"温顺"样本，N=15 揭示真实方差更大。CV(corr) 从 21%/30% 飙到 54%/64%。
2. **绝对 std 几乎一致（0.214 vs 0.210）**：V6 和 V2 的"分布形状"没有结构性差异。CV 差异完全来自不同的均值。
3. **V2 在骨盆前倾上每项指标都赢 V6**：hit_band, corr mean, shape pass rate 都更高。**V6 的 PID 复杂度没换来稳定性优势**。

### 跨体态实验（膝超伸，5 seed）

| variant | Δ | hit_band | corr | shape |
|---|---|---|---|---|
| v2_dps_s40_always | +0.57±2.21° | **0.0%** | 0.060 | 5/5 ⚠ 推力不足 |
| v6_closed_loop_second_half | +0.60±2.31° | **0.0%** | 0.063 | 5/5 ⚠ 推力不足 |

**两个 variant 失败模式完全一致**——这是个关键信号：**问题不在 controller，而在任务本身 OOD**。

### 关键发现：膝超伸是 OOD 任务

自然走路时膝盖角度在 130-180° 之间，**永远 < 180°**。膝超伸目标 190° 意味着膝盖向后反折——这不在 MDM 的训练分布里。

文本 prompt "a person is walking forward" + MDM 训练分布形成强先验，guidance 梯度根本推不动。**任何 inference-time guidance 方法在 OOD 任务上都会失败**——不是 V6 弱，是 fundamental limit。

---

## 五、当前真正的瓶颈

回顾整个项目，"提升 V6 稳定性"这个目标遇到三层障碍：

### 障碍 1：CV(corr) 是误导性指标
- N=5 的 21-30% 是 sample bias
- 真实 std 在 V2 和 V6 上几乎一致（0.21）
- 用 CV 比较两个均值不同的分布会得出错误结论
- **建议换指标**：median + IQR, 或 corr>0.3 的 seed 占比

### 障碍 2：V6 在单体态上没有可量化的优势
- V2 的简单"固定 s + hinge + 自动停推"反而打过 V6 的 PID
- Hinge 的"到位即停推"性质恰好契合本任务（≤2° tolerance 容忍区间）
- V6 的"精确跟踪"能力是过度设计

### 障碍 3：跨体态选项有限（**新发现，最关键**）

`new/POSTURE_REPRESENTABILITY.md` 写了 22 关节下哪些体态可表征。但还要叠加一层：**MDM 训练分布是否覆盖这个体态**。

| 体态 | 关节表征 | walking 分布内 | 适合跨体态实验 |
|---|---|---|---|
| 骨盆前倾 | ✓ | ✓ | ✅ 已做 |
| 膝超伸 (190°) | ✓ | ❌ | ❌ OOD，已失败 |
| 驼背 | △ 弱 | ❌ | ❌ 几乎确定同样失败 |
| 头前伸 | △ 弱 | △ 看 prompt | △ 风险 |
| **膝弯曲 (125°)** | ✓ | ✓（慢走） | ✅ **强推荐**，未做 |
| **躯干前倾** (10°) | ✓ | ✓（快走） | ✅ **强推荐**，未做 |
| **骨盆侧倾** (5°) | ✓ | ✓（病理步态） | ✅ 未做 |
| 骨盆后倾 (−5°) | ✓ | ✓ | ✅ 未做 |

**当前 registry.py 只注册了骨盆前倾 + 膝超伸 + 驼背**。要做有效的跨体态实验，需要先加注册分布内的体态。

---

## 六、下一步执行优先级

### 优先 1（强推荐）：加分布内体态做真正的跨体态实验

加 **膝弯曲** 和 **躯干前倾** 两个 spec 到 `registry.py`，然后跑 `run_cross_posture.sh`。

每个 spec ≈ 30 行：
- 膝弯曲 可以直接复用 `signed_knee_angle`，改 target=125, direction="less_than"
- 躯干前倾 需要新写 `trunk_forward_lean_angle()` in `angle_ops.py`（pelvis-spine1-spine3 sagittal angle）

也要在 `evaluate_ablation_v3.py:29` 的 `ANGLE_TARGETS` dict 加对应行。

### 优先 2（可并行）：OOD 假说快速验证（10 分钟）

把膝超伸的 text prompt 换掉看是否能推动：

```bash
TEXT_PROMPT="a person stands still with stiff straight legs" \
POSTURE=膝超伸 SEED=42 \
GUIDANCE_VARIANT=v2_dps \
GUIDANCE_KWARGS_JSON='{"s":40.0,"schedule":"always"}' \
bash new/run_posture_pipeline.sh
```

如果 hit_band 跳到 30%+ → OOD 假说确认，知道 prompt 怎么挑就能跑更多体态。

### 优先 3：bootstrap CI + median 评估（脚本改动）

`new/aggregate_seeds.py` 加 `scipy.stats.bootstrap` 计算 95% CI，加 median + IQR 输出。这样 N=15 数据有更可信的统计描述（CV 现在的报错信号不可靠）。

### 优先 4（论文 ablation 扩展）：加更多分布内体态

骨盆侧倾、足距过宽、膝内扣（valgus/varus，frontal plane）——这些都是分布内 + 关节强表征。`POSTURE_REPRESENTABILITY.md` 有完整清单。

### 不推荐做的事

- ❌ 继续在骨盆前倾上调 V6 单超参（5 轮迭代已经探到 Pareto 前沿，单参数无突破空间）
- ❌ 跑驼背 / 膝超伸 / 头前伸（OOD 风险大，预期失败）
- ❌ 实现 plan 里答应的 time-travel（FreeDoM）——优先级低，先看跨体态结果
- ❌ 实现 dual-stage hybrid（Huber 早 + hinge 晚）——架构改动大，先用更简单方案

---

## 七、运行环境

```bash
conda activate mdm5090
cd /root/autodl-tmp/motion-diffusion-model

# 单 seed（调试）
GUIDANCE_VARIANT=v6_closed_loop \
GUIDANCE_KWARGS_JSON='{"Kp":80,"Ki":1,"Kd":5,"s_min":0.05,"s_max":50,
                       "I_max":20,"beta_ema":0.8,"lambda_smooth":0.03,
                       "manifold_project":true,
                       "loss_form":"huber","huber_delta":0.05,
                       "normalize_grad":false,"band_gate":false,
                       "spec_schedule_override":"second_half"}' \
SEED=42 POSTURE=骨盆前倾 \
bash new/run_posture_pipeline.sh

# 多 seed sweep (N=5 兼容老脚本)
bash new/run_seed_robustness.sh

# 多 seed sweep (N=15，新)
bash new/run_seed_robustness_n15.sh
python -m new.aggregate_seeds ./output/n15_*

# 跨体态 sweep (自动适配 deg/meter)
bash new/run_cross_posture.sh 膝弯曲 5    # ←【需要先加 spec！】
bash new/run_cross_posture.sh 躯干前倾 5   # ←【需要先加 spec！】
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
| (本 commit) | 本 PROJECT_SUMMARY.md |

如果你需要从某个具体迭代 fork 出去对比，`git checkout <commit>` 即可。每个 commit message 都详细记录了改动动机。

---

## 九、给接续者的建议

1. **先读 commit history**：每个 commit message 都自带"为什么这么改"的诊断，按时间读一遍能快速理解 V6 演化逻辑。

2. **不要再调 V6 在骨盆前倾上的单参数**。Pareto 前沿已经描清楚（hit↔corr trade-off），多花时间无产出。

3. **首要任务是加分布内的新体态**（膝弯曲 + 躯干前倾），然后做真正的跨体态实验。这才是验证 V6 价值的关键测试。

4. **统计指标换掉**：CV(corr) 在均值不同时不可信，用 median + IQR + bootstrap CI。

5. **如果跨体态实验 V6 也无优势**：诚实接受 negative result，论文可以写"post-hoc PID guidance 对 in-distribution 单任务 inference-time guidance 收益有限，主要瓶颈在 MDM 训练分布覆盖"。

6. **如果 V6 在跨体态确有优势**：那 V6 的卖点是"无需 per-task 重调 s_max"，V2 在不同体态需要不同 s。

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

**总结一句话**：V6 的 PID 闭环架构在工程上完整实现且经过 5 次迭代调试，但实验证明它在骨盆前倾上的稳定性优势是 N=5 sample bias 的假象；真正能区分 V6 vs V2 价值的跨体态实验受限于 MDM 训练分布覆盖，需要先加分布内的新体态（膝弯曲、躯干前倾）才能继续。

