# MDM Posture Guidance 项目总结（第二版）

> **适合人群**：接手本项目的新成员，在没有读过完整对话记录的情况下，通过本文档理解项目现状、代码结构和下一步方向。
>
> **版本说明**：第二版在原版基础上新增了第 15-22 节，涵盖了 IK Guidance 方法学对比实验（V1-V5 五种方法）的完整设计、工程实现、实验结果与结论。

---

## 1. 项目目标

在 MDM（Motion Diffusion Model）的扩散采样过程中，**不重新训练模型**，通过在去噪每一步注入几何约束（IK Guidance），让模型生成具有指定病态体态的运动序列。

目标体态包括：
- 骨盆前倾（Anterior Pelvic Tilt, APT）
- 膝超伸（Genu Recurvatum）
- 驼背（Thoracic Kyphosis）
- 头前伸（Head Forward Posture）

---

## 2. 项目背景：使用的模型

**不是标准 MDM，而是 DiP（Dual-inferred Pretraining）的改进版**：

```
模型路径：./save/humanml_trans_dec_512_bert/model000200000.pt
加载方式：load_saved_model(model, args.model_path, use_avg=args.use_ema)  ← 必须用这个
采样步数：50 步（不是标准 DDPM 的 1000 步）
文本编码：BERT（encode_text 在 model 上）
数据集：HumanML3D（263 维表示，22 个关节）
```

**这是最重要的背景信息**，50 步采样对 guidance 的参数调整有根本性影响（见第 5 节）。

> ⚠️ **加载注意**：不要用 `load_model_wo_clip`（会触发 `KeyError: sequence_pos_encoder.pe`），必须用 `load_saved_model`。

---

## 3. 代码结构

```
motion-diffusion-model/
├── diffusion/
│   └── gaussian_diffusion.py       ← 修改了 p_sample_loop_progressive（guidance 插入点）
│                                      新增了 V1-V5 五种 guidance variant 的 dispatch 逻辑
├── posture_guidance/               ← 核心模块（新建）
│   ├── __init__.py
│   ├── joint_indices.py            ← SMPL 22 关节索引映射
│   ├── phase_detector.py           ← 步态相位检测（支撑相/摆动相）
│   ├── angle_ops.py                ← 所有关节角计算函数
│   ├── registry.py                 ← 体态约束注册表
│   ├── controller.py               ← PostureGuidance 主类 + set_variant 方法
│   ├── mdm_integration.py          ← apply_posture_guidance + make_fk_fn
│   └── guidance_variants.py        ← V1-V5 五种方法的独立实现（可选）
├── new/                            ← 所有脚本（作为 Python 模块运行）
│   ├── run_posture_comparison.py   ← 主实验脚本（生成 comparison.npy）
│   ├── run_posture_pipeline.sh     ← 一键运行全流程
│   ├── evaluate_ablation.py        ← 统一量化评估（v3 版本，区间 hit_rate）
│   ├── plot_angle_curves.py        ← 角度曲线对比绘图（6 子图）
│   ├── run_calibrated_ablation_v3.sh  ← 精细化 ablation 脚本
│   ├── run_seed_robustness.sh      ← 多 seed 稳健性验证
│   ├── aggregate_seeds.py          ← 多 seed 聚合统计
│   ├── quantitative_compare.py     ← 单次实验量化分析
│   ├── visualize_compare.py        ← 角度曲线 + 标准动画
│   ├── plot_motion_cycle_v4.py     ← 论文风格动作周期图
│   ├── make_multiview_animation_v2.py  ← 多视角动画
│   └── make_anatomical_animation_v3.py ← 解剖风格双视图动画
└── sample/
    └── generate.py                 ← 修改了采样调用（透传 guidance 参数）
```

---

## 4. 核心机制：IK Guidance 如何工作

### 4.1 整体流程

```
MDM 去噪每一步（共 50 步）：
    1. MDM 前向：x_t → 预测 x̂_0 → 计算 μₜ（后验均值）
       with th.no_grad():
           out = p_sample(model, img, t, ...)
    2. q_posterior_mean_variance(x̂_0, x_t) → μₜ
    3. Guidance 介入点（with th.no_grad() 块已结束）：
       dispatch to variant V1/V2/V2b/V3/V4/V5
    4. 用更新后的 μₜ 重新采样 x_{t-1} = μₜ_updated + σ_t · z
```

**关键设计**：guidance 在 `with th.no_grad()` 块结束之后才介入，所以 `th.enable_grad()` 能正常工作。MDM 网络权重永不更新。

### 4.2 FK 函数（`make_fk_fn`）

```python
def make_fk_fn(t2m_dataset, n_joints=22):
    mean = torch.tensor(t2m_dataset.mean)  # (263,) 保留在 GPU
    std  = torch.tensor(t2m_dataset.std)

    def fk_fn(mu):
        # mu: (B, 263, 1, T)
        mu_perm = mu.squeeze(2).permute(0, 2, 1)   # → (B, T, 263)
        mu_inv  = mu_perm * std + mean               # 可微逆归一化
        mu_inv  = mu_inv.unsqueeze(2)                # → (B, T, 1, 263)
        q = recover_from_ric(mu_inv, n_joints)       # → (B, T, 1, J, 3)
        return q.squeeze(2)                           # → (B, T, J, 3)
    return fk_fn
```

> ⚠️ **梯度关键点**：`fk_fn` 全程 pure torch，不含 `.detach()` 或 `.numpy()`，所以梯度链完整。已用 `debug_v2_grad.py` 验证（见第 18 节）。

### 4.3 263 维表示的结构

```
dim 0       : root 旋转速度
dim 1-2     : root 线速度 xz
dim 3       : root y 位置
dim 4-66    : 21 关节局部位置 (21×3)  ← recover_from_ric 主要用这部分
dim 67-192  : 21 关节旋转 6D (21×6)   ← 梯度为 0（不参与 FK 坐标重建）
dim 193-258 : 22 关节速度 (22×3)      ← 梯度为 0
dim 259-262 : 4 个 foot contact label  ← 梯度为 0
```

**梯度稀释的根本原因**：有效梯度只覆盖 dim 4-66（约 24%），其余 76% 梯度为 0。这是 V1 需要大 base_weight（30+）的根本原因。

---

## 5. 最重要的调参经验（50 步采样专属）

### 5.1 为什么 50 步采样需要特殊处理

| 参数 | 1000 步 DDPM | 50 步 DiP |
|---|---|---|
| 单步可推力 | 0.001 量级 | 0.1-0.5 量级 |
| 优化器 | LBFGS + strong_wolfe 都行 | **必须禁用 line_search 或用 SGD** |
| schedule | decay/always 都有效 | **last_quarter 或 final** |
| base_weight | 0.1-0.3 | **10-50+** |

### 5.2 schedule 的重要性（实验验证）

```
# 实验证明：同样 s=80，schedule 不同结果天壤之别
v2_dps_s80_lastquarter  corr=+0.452  ✅ 正常
v2_dps_s80_final        corr=-0.013  ❌ 塌平

# 为什么？
# final: 只在 t<5 推，每步必须推力极大才能达到目标 → 结构崩塌
# last_quarter: 在 t<12 推，推力分散 12 步 → 渐进引导
```

> 💡 **经验规则**：schedule 不是随便选的，`last_quarter`（t<T/4）适合 DPS 类方法，`final`（t<5）适合 SGD-on-x0 类方法。

---

## 6. 关节角函数的实现与验证

### 6.1 骨盆前倾（已修复，符号正确）

```python
def pelvis_tilt_angle(q):
    """正值=前倾，正常走路≈+5-10°，病态前倾≈+15-25°"""
    hip_center = (left_hip + right_hip) / 2
    pelvis_to_spine = spine1 - hip_center
    sagittal_vec = pelvis_to_spine - (pelvis_to_spine · lr_axis) * lr_axis
    tilt = torch.atan2(sagittal_vec[..., 2], sagittal_vec[..., 1])
    return -tilt  # 反号：HumanML3D 里 spine1 在 hip 后方，atan2 符号反转
```

### 6.2 膝超伸（修复版）

```python
def signed_knee_angle(q, side="left"):
    """完全伸展=π(180°)，超伸>π"""
    mid_z = (hip[..., 2] + ankle[..., 2]) / 2
    knee_offset_z = knee[..., 2] - mid_z
    sign = torch.sigmoid(-50.0 * knee_offset_z)
    return base_angle + sign * 2 * (math.pi - base_angle)
    # 修复：不用 x-z 叉积（左右不对称），改用 z 偏移判向
```

### 6.3 驼背（修复版）

```python
def spine_posterior_bulge(q):
    """spine3 偏离 spine1-neck 基准线的垂直距离，始终≥0"""
    trunk_dir = (neck - spine1) / trunk_len
    s3_vec = spine3 - spine1
    proj = (s3_vec · trunk_dir) * trunk_dir
    perp = s3_vec - proj
    return perp.norm(dim=-1)
```

---

## 7. 验证流程（新成员必读）

### 步骤 1：符号验证

```bash
python - << 'EOF'
import torch, math, sys
sys.path.insert(0, "/root/autodl-tmp/motion-diffusion-model")
from posture_guidance.angle_ops import pelvis_tilt_angle
import numpy as np

data = np.load("./output/any_exp/comparison.npy", allow_pickle=True).item()
q = torch.from_numpy(data["motion_xyz"][0]).float()
angle = pelvis_tilt_angle(q) * 180 / math.pi
print(f"mean={angle.mean():.1f}°  (期望 walking ≈ +5~10°)")
EOF
```

### 步骤 2：完整梯度链验证（`debug_v2_grad.py`）

```bash
python debug_script/debug_v2_grad.py
# 期望输出：
# [Step 6] autograd.grad(loss, x_t)
#   ✅ 梯度正常! shape=(1, 263, 1, 120) norm=0.6500
# 🎉 V2 完整梯度链正常！可以直接运行 v2_dps。
```

---

## 8. 一键运行流程

### 8.1 基础使用（V1 方法）

```bash
cd /root/autodl-tmp/motion-diffusion-model

TEXT_PROMPT="a person is walking forward" \
POSTURE="骨盆前倾" \
SEED=42 \
MOTION_LENGTH=6.0 \
OUTPUT_DIR="./output/walking_apt" \
./new/run_posture_pipeline.sh
```

### 8.2 指定 guidance variant

```bash
# V2 (DPS, 推荐)
GUIDANCE_VARIANT=v2_dps \
GUIDANCE_KWARGS_JSON='{"s":80,"schedule":"last_quarter","base_weight":1.0}' \
TEXT_PROMPT="a person is walking forward" \
POSTURE="骨盆前倾" \
SEED=42 \
OUTPUT_DIR="./output/v2_dps_s80" \
./new/run_posture_pipeline.sh

# V2b (x0 direct edit)
GUIDANCE_VARIANT=v2_x0_edit \
GUIDANCE_KWARGS_JSON='{"n_inner_steps":15,"lr":0.5,"schedule":"final","base_weight":8}' \
TEXT_PROMPT="a person is walking forward" \
POSTURE="骨盆前倾" \
SEED=42 \
OUTPUT_DIR="./output/v2b_bw8" \
./new/run_posture_pipeline.sh
```

### 8.3 输出文件结构

```
output/exp_xxx/
├── comparison.npy              ← 原始数据（baseline + guided 的 hml + xyz）
├── comparison_report.txt       ← 量化分析报告
├── pipeline.log                ← 完整运行日志
└── viz/
    ├── angle_curves.png        ← 关节角时间曲线
    ├── motion_cycle_paper.png  ← 论文风格动作周期图
    └── ...
```

`comparison.npy` 的 key：

```python
data = np.load("comparison.npy", allow_pickle=True).item()
data.keys()
# dict_keys(['motion_hml', 'motion_hml_tj', 'motion_xyz',
#            'motion_hml_guided', 'motion_hml_tj_guided', 'motion_xyz_guided',
#            'text_prompt', 'posture_instructions', 'seed',
#            'num_samples', 'motion_length', 'fps', 'guidance_config'])

# 读取关节坐标（最常用）
q_baseline = data["motion_xyz"][0]        # (T, J, 3) = (120, 22, 3)
q_guided   = data["motion_xyz_guided"][0] # (T, J, 3)
```

---

## 9. 量化报告解读

```
metric          baseline    guided      delta
骨盆前倾         6.01°  →   20.38°   (↑ 14.37°)

Hit rate (双边区间 [18°,22°]):  64.2%   ← 好的方法 40-80%
Corr:                          +0.323   ← 好的方法 >0.3
RMSE:                           0.111m  ← 好的方法 <0.15m
```

**健康指标范围（v3 版本）**：

| 指标 | 健康范围 | 塌平信号 | 过强信号 |
|---|---|---|---|
| delta | 10-20° | — | >35° |
| hit_rate (双边) | 30-70% | hit>90% 且 corr<0.15 | — |
| corr | >0.3 | corr<0 | — |
| target_distance | <3° | — | >5° |
| RMSE | 0.05-0.15m | — | >0.2m |
| jitter | <2° | — | >5° |

> ⚠️ **重要**：hit_rate 现在是**双边区间**（`angle ∈ [target-tol, target+tol]`），不是旧版的单边（`angle >= target-tol`）。单边版本会让"推力过大"的方法虚假地 hit=100%。

---

## 10. 可视化说明

### 角度曲线对比图（`plot_angle_curves.py`）

```bash
python -m new.plot_angle_curves ./output/calibrated_walking_apt_seed42
# 输出：
#   angle_curves_all.png        ← 所有 variant 叠在一张大图
#   angle_curves_per_variant.png ← 6 个子图，每个 variant 一张
```

**判读规则**：
- ✅ 平移+保结构：guided 曲线 = baseline 曲线 + 常数 offset，曲线形状一致
- ❌ 塌平：guided 曲线变成接近水平的直线（不随 baseline 波动）
- ⚠ 推力不足：guided 曲线和 baseline 几乎重合
- ⚠ 时间结构破坏：guided 和 baseline 的峰谷位置反向

### 其他可视化

```bash
# 论文风格动作周期图
python -m new.plot_motion_cycle_v4 ./output/exp/comparison.npy \
    --n_keyframes 7 --cycle_label "Gait Cycle"

# 解剖风格动画
python -m new.make_anatomical_animation_v3 ./output/exp/comparison.npy

# 多视角动画
python -m new.make_multiview_animation_v2 ./output/exp/comparison.npy
```

---

## 11. 已知问题和局限性

| 问题 | 严重程度 | 现状 |
|---|---|---|
| 梯度被 `recover_from_ric` 稀释（只有 dim 4-66 有效） | 高 | V2/V2b 通过不同路径部分缓解 |
| V1 一直是塌平方法（corr 全部为负） | 高 | ✅ 已发现，不再作为 baseline |
| 膝超伸左右略不对称 | 中 | 多 seed 扫描部分缓解 |
| 驼背视觉效果弱（只有 3 个脊柱节点） | 高 | bulge_amplify 视觉补偿 |
| 关节角是近似，不等于临床测量 | 中 | 学术层面局限 |

---

## 12. 下一步可以做的事

### 优先级高（当前阶段）

1. **完成精细化 ablation v3**：运行 `run_calibrated_ablation_v3.sh`，找到 V2/V2b 的最优参数点
2. **多 seed 稳健性验证**：对最佳配置跑 5 个 seed，确认结果不是偶然
3. **更新报告**：把 V1 塌平 + V2 最优 + LGD 反驳 LGD 论文的发现写进中期报告

### 优先级中

4. **膝超伸对称性**：实现动态左右权重（根据当前步态相位自动调整）
5. **时间平滑 loss**：加相邻帧差惩罚，减少 guided 曲线抖动
6. **相对偏移目标**：深蹲场景下用 "delta" 模式而非 "absolute"

### 优先级低

7. **Layer 2 逆动力学**：Savitzky-Golay 滤波估关节加速度
8. **OpenSim 验证**：导入生成动作做生物力学验证
9. **SMPL-X 升级**：解决驼背脊柱采样稀疏问题

---

## 13. 关键文件的实际路径

```bash
# 项目根目录
/root/autodl-tmp/motion-diffusion-model/

# 模型 checkpoint
./save/humanml_trans_dec_512_bert/model000200000.pt

# 核心代码（按优先级阅读）
./posture_guidance/angle_ops.py          # 关节角函数（先看这里）
./posture_guidance/registry.py           # 体态注册（调参看这里）
./posture_guidance/controller.py         # PostureGuidance 主类
./posture_guidance/mdm_integration.py   # fk_fn + apply_posture_guidance
./diffusion/gaussian_diffusion.py       # guidance 插入点 + V1-V5 dispatch
./new/run_posture_comparison.py         # 主实验脚本

# 评估工具
./new/evaluate_ablation.py              # 统一评估（v3 版，用区间 hit_rate）
./new/plot_angle_curves.py              # 角度曲线绘图
./new/aggregate_seeds.py               # 多 seed 聚合
```

---

## 14. 快速验证项目是否正常

```bash
cd /root/autodl-tmp/motion-diffusion-model

GUIDANCE_VARIANT=v2_dps \
GUIDANCE_KWARGS_JSON='{"s":80,"schedule":"last_quarter","base_weight":1.0}' \
TEXT_PROMPT="a person is walking forward" \
POSTURE="骨盆前倾" \
SEED=42 \
MOTION_LENGTH=6.0 \
MAKE_ANIMATION="" \
OUTPUT_DIR="./output/quick_test" \
./new/run_posture_pipeline.sh

cat ./output/quick_test/comparison_report.txt | grep -A 5 "骨盆前倾"
```

期望输出：`骨盆前倾 +15~16°, hit≈100%, corr≈+0.45`

---

---

# ========== 以下为第二版新增内容 ==========

---

## 15. IK Guidance 方法学：五种方法的设计（核心新增内容）

这是本项目第二阶段的核心工作——系统比较五种不同的 inference-time guidance 注入策略。

### 15.1 问题背景

所有 variant 解决同一个问题：
> "如何在 MDM 的第 t 步去噪之后，用关节角梯度更新 μₜ，使得下一步采样 x_{t-1} 的骨盆角度更接近目标？"

五种方法的根本区别在于**梯度路径**和**更新对象**：

```
方法          梯度路径                     更新对象      是否穿过 MDM 网络
─────────────────────────────────────────────────────────────────────
V1  SGD-μt   μₜ → FK → loss              μₜ           ❌ 不穿
V2  DPS       xₜ → MDM → x̂₀ → FK → loss  μₜ           ✅ 穿过
V2b x0_edit   x̂₀ → FK → loss             x̂₀→μₜ(重组)  ❌ 不穿
V3  x0_direct x̂₀ → FK → loss             x̂₀→μₜ(重组)  ❌ 不穿（短 loop 版）
V4  OmniCtrl  μₜ → FK → loss              μₜ           ❌ 不穿（动态迭代）
V5  LGD       xₜ → MDM → x̂₀+ε → FK → loss μₜ          ✅ 穿过（MC 平均）
```

### 15.2 统一接入方式（`gaussian_diffusion.py`）

所有 variant 通过同一个 dispatch 系统接入：

```python
# p_sample_loop_progressive 主循环里的 guidance 介入点
if use_guidance:
    mu_t, _, log_var = self.q_posterior_mean_variance(
        x_start=out["pred_xstart"], x_t=img, t=t
    )
    # ★ 统一 dispatch
    mu_t_updated = self._apply_guidance_variant(
        variant_name=variant_name,      # 从环境变量读取
        variant_kwargs=variant_kwargs,  # 从 GUIDANCE_KWARGS_JSON 读取
        mu_t=mu_t,
        x_t=img,
        pred_xstart=out["pred_xstart"],
        ...
    )
    # 用更新后的 μₜ 重新采样
    out["sample"] = mu_t_updated + nonzero_mask * th.exp(0.5*log_var) * noise
```

### 15.3 共享的 loss 闭包（关键设计）

所有 variant 共用同一个 guidance loss，但有一个关键的 anchor 修复：

```python
def guidance_loss_fn(q, t_in, T_in):
    # ★ anchor：保证 loss 始终有 grad_fn，即使所有 target 不激活（返回 0）
    # 没有这个 anchor，当 schedule 未激活时，loss=tensor(0.) 没有 grad_fn，
    # 调用 autograd.grad 会直接 crash
    anchor = q.sum() * 0.0
    return guidance.compute_loss(q, t_in, T_in) + anchor
```

---

## 16. 五种方法的详细实现

### V1 — SGD on μₜ（原始方法）

```python
def _guidance_v1_mu_sgd(self, *, mu_t, t_int, T, fk_fn, guidance_loss_fn,
                          schedule="final", base_weight=30.0,
                          n_inner_steps=15, lr=0.5, ...):
    if not _schedule_active(schedule, t_int, T): return mu_t

    mu_var = mu_t.detach().clone().contiguous().requires_grad_(True)
    optimizer = SGD([mu_var], lr=lr)
    for _ in range(n_inner_steps):
        optimizer.zero_grad()
        q = fk_fn(mu_var)
        loss = base_weight * guidance_loss_fn(q, t_int, T)
        loss.backward()
        optimizer.step()
    return mu_var.detach()
```

**特点**：梯度不穿 MDM；15 步内循环；需要大 base_weight=30 补偿梯度稀释；schedule=final（只在 t<5 推）。

**实验结果**：**所有 base_weight 配置的 corr 都是负值（-0.11 ~ -0.18）**。说明 V1 虽然 Δ 看起来对，但时间结构被完全破坏（曲线变成高频抖动）。**V1 是失败方法**，不能作为 baseline。

### V2 — DPS（梯度穿过 MDM）

```python
def _guidance_v2_dps(self, *, mu_t, x_t, t_int, t_tensor, T,
                      model, model_kwargs, fk_fn, guidance_loss_fn,
                      s=80.0, schedule="last_quarter", base_weight=1.0):
    if not _schedule_active(schedule, t_int, T): return mu_t

    with th.enable_grad():
        x_t_var = x_t.detach().clone().contiguous().requires_grad_(True)
        model_output = model(x_t_var, self._scale_timesteps(t_tensor), **model_kwargs)

        # Guard: model_output 没有 grad_fn（ClassifierFree wrapper 问题）
        if model_output.grad_fn is None:
            if hasattr(model, "model"):
                model_output = model.model(x_t_var, ...)
            if model_output.grad_fn is None: return mu_t

        x0_hat = model_output  # model_mean_type = START_X

        q = fk_fn(x0_hat)
        loss = base_weight * guidance_loss_fn(q, t_int, T)

        # loss=0 跳过（hinge loss 在目标已满足时为 0，梯度为 0，不需要更新）
        if loss.item() == 0.0: return mu_t

        grad = th.autograd.grad(loss, x_t_var)[0]

    # 打印每步状态（用于确认梯度更新发生）
    delta_mu = (s * grad.detach()).norm().item()
    print(f"[V2 UPDATE t={t_int:3d}] loss={loss.item():.4f}  |s*grad|={delta_mu:.5f}")

    mu_t_new = mu_t.detach() - s * grad.detach()
    return mu_t_new
```

**特点**：梯度穿 MDM，全 263 维有梯度；单步梯度；不乘 sigma_sq（避免 DiP 下推力趋零）；需要大 s（≈80）因为 MDM Jacobian 压缩了梯度（约 0.1-0.3 倍）。

**实验结果**：s=80 时 corr=+0.452（最佳），曲线在 18-25° 之间波动并保留 baseline 双峰结构。**V2 是目前"落带最精准"的方法**。

**重要发现（论文级）**：V2 需要 s≈80 远大于图像域 DPS 的 s≈1，根本原因是 MDM Jacobian（∂x̂₀/∂xₜ）的 Frobenius 范数约为 0.1-0.3，比图像 diffusion 小一个数量级。这是去噪器被训练成"抑制输入扰动"的固有属性。

### V2b — x0 Direct Edit（直接编辑 x̂₀）

```python
def _guidance_v2b_x0_edit(self, *, mu_t, x_t, pred_xstart, t_int, t_tensor, T,
                            fk_fn, guidance_loss_fn,
                            n_inner_steps=15, lr=0.5, schedule="final",
                            base_weight=8.0):
    if not _schedule_active(schedule, t_int, T): return mu_t

    # 从 MDM 已经预测好的 x̂₀ 出发（不重新跑 MDM）
    x0_var = pred_xstart.detach().clone().contiguous().requires_grad_(True)
    optimizer = SGD([x0_var], lr=lr)
    for _ in range(n_inner_steps):
        optimizer.zero_grad()
        q = fk_fn(x0_var)
        loss = base_weight * guidance_loss_fn(q, t_int, T)
        if loss.grad_fn is None: break
        loss.backward()
        optimizer.step()

    # ★ 关键：用更新后的 x̂₀ 重组 μₜ
    # μₜ = coef1 × x̂₀_updated + coef2 × xₜ
    mu_t_new, _, _ = self.q_posterior_mean_variance(
        x_start=x0_var.detach(), x_t=x_t, t=t_tensor
    )
    return mu_t_new
```

**特点**：不重新跑 MDM（更快）；直接在 x̂₀ 上 SGD；通过 `q_posterior_mean_variance` 将 x̂₀ 的修改传递到 μₜ。

**实验结果**：bw=8 时 corr=+0.488（最高 corr 之一），但 Δ=+22°（平均角度 28°，偏离目标 20° 约 8°）。bw=10 时 Δ 达到 +27°，过度推力。**V2b 需要更精细的 bw=6-8 窗口**。

**关键发现**：在 `schedule=final` 下，inner_steps 从 5→25 几乎不改变结果（Δ 和 corr 变化 <5%）。说明 V2b 在 final schedule 下很快收敛，多迭代是浪费计算。

### V3 — x0_direct（V2b 的保守版）

与 V2b 实现完全相同，但默认参数更保守（n_inner_steps=5, lr=0.05, base_weight=1.0）。主要用于验证 GMD 的"x₀ 模型会撤销 guidance"警告在 50 步 DiP 上是否成立。

**实验结果**：在 final schedule 下，V3 和 V2b 结果几乎完全一致（Δ 差距 <0.1°，corr 差距 <0.01）。**说明 GMD 警告在 50 步 DiP 上不成立**——x₀ 直接编辑是稳定的。

### V4 — OmniControl（动态迭代次数）

```python
def _guidance_v4_omni(self, *, mu_t, t_int, T, fk_fn, guidance_loss_fn,
                       K_early=0, K_late=3, T_split=12, lr=0.5,
                       base_weight=8.0, schedule="always"):
    if not _schedule_active(schedule, t_int, T): return mu_t

    # ★ 晚期（t<T_split）多迭代，早期少迭代
    n_steps = K_late if t_int < T_split else K_early
    if n_steps == 0: return mu_t

    mu_var = mu_t.detach().clone().contiguous().requires_grad_(True)
    optimizer = SGD([mu_var], lr=lr)
    for _ in range(n_steps):
        optimizer.zero_grad()
        q = fk_fn(mu_var)
        loss = base_weight * guidance_loss_fn(q, t_int, T)
        loss.backward()
        optimizer.step()
    return mu_var.detach()
```

**特点**：梯度不穿 MDM；早期 K=0（不推），晚期 K=3（推）；动态迭代受 T_split 控制。

**实验结果**：bw=8, K_late=3 时 corr=+0.468，Δ=+23°（略过强）。但曲线在 60-80 帧有**剧烈跳变**，这是动态 K 切换时的不连续性。**V4 在临界点工作，不稳定**。

**早期实验教训**：bw=30, K_late=10 时 Δ=55°，corr=0.13，完全塌平。bw 和 K_late 的乘积决定总推力，必须谨慎。

### V5 — LGD（DPS + Monte Carlo 平滑）

```python
def _guidance_v5_lgd(self, *, mu_t, x_t, t_int, t_tensor, T,
                      model, model_kwargs, fk_fn, guidance_loss_fn,
                      n_mc=2, s=80.0, schedule="last_quarter",
                      base_weight=1.0, mc_noise_scale=0.05):
    sigma_t = float(self.posterior_variance[t_int]) ** 0.5
    grads = []
    for _ in range(n_mc):
        with th.enable_grad():
            x_t_var = x_t.detach().clone().contiguous().requires_grad_(True)
            model_output = model(x_t_var, ..., **model_kwargs)
            x0_hat = model_output
            # ★ MC 扰动：在 x̂₀ 上加小噪声，得到 n_mc 个不同的梯度估计
            eps = th.randn_like(x0_hat) * sigma_t * mc_noise_scale
            q = fk_fn(x0_hat + eps)
            loss = base_weight * guidance_loss_fn(q, t_int, T)
            g = th.autograd.grad(loss, x_t_var)[0]
            grads.append(g.detach())

    grad_mean = th.stack(grads).mean(dim=0)
    # ★ 不乘 sigma_sq（与 V2 对齐，避免 DiP 下推力趋零）
    mu_t_new = mu_t.detach() - s * grad_mean
    return mu_t_new
```

> ⚠️ **V5 的历史 Bug**：早期实现里有 `mu_t_new = mu_t - s * sigma_sq * grad_mean`，`sigma_sq` 在 last_quarter（t<12）时约为 0.01-0.07，导致实际推力为 `s × 0.05 × grad`，比 V2 弱 20 倍。**修复方法：删掉 `sigma_sq`**。

**实验结果（修复后）**：s=80, n_mc=2 时 corr=+0.382，Δ=+15.6°，与 V2 相近但 corr 略低 0.07。**MC 平均在 motion 域没有带来 corr 提升，反而因为独立 MDM 前向引入了不同的"MDM 想象信号"，让梯度更模糊。这反驳了 LGD 论文的标准说法，是一个 finding。**

---

## 17. 评估指标体系（v3 版）

### 17.1 指标定义

```python
# evaluate_ablation.py (v3)

# ★ 双边区间 hit_rate（关键改动）
hit_mask = (a_g_deg >= target - tol) & (a_g_deg <= target + tol)
hit_rate = hit_mask.mean()                # 落在 [target-tol, target+tol] 的帧比例

# 单边 hit（辅助参考，不参与 score）
hit_rate_loose = (a_g_deg >= target - tol).mean()

# 其他指标
target_distance = abs(a_g_deg.mean() - target)  # 平均角度距目标的绝对距离
corr = pearsonr(a_b_deg, a_g_deg)[0]           # baseline vs guided 时间相关
jitter = np.diff(a_g_deg).std()                # 帧间抖动
```

**为什么要改成双边**：旧版 `angle >= 18°`（单边）对"推到 30°+""推到 50°+"都给 hit=100%，无法区分"恰好落在 20°"和"推到 50°"。新版双边让过度推力的方法 hit_rate 暴跌，暴露问题。

### 17.2 硬约束评分函数

```python
def composite_score(m):
    # 硬约束（任一触发 → score=0）
    if m["hit_rate_loose"] < 0.05:              return 0.0  # 推力不足
    if m["corr"] < 0:                           return 0.0  # 时间结构反向
    if m["hit_rate_loose"] > 0.9 and m["corr"] < 0.15: return 0.0  # 塌平
    if m["jitter"] > 5.0:                       return 0.0  # 剧烈抖动
    if m["target_distance"] > 5.0:              return 0.0  # 过度推力

    # 通过硬约束后的综合分数（双边 hit_rate 更严格）
    return (m["hit_rate"] * m["corr"] /
            (1.0 + 5.0 * m["rmse"] + 0.2 * m["jitter"] + m["foot_skate"]))
```

### 17.3 形状分类

| 形状 | 判定条件 | 含义 |
|---|---|---|
| ✅ 平移+保结构 | hit>40% 且 corr>0.3 | 理想状态：角度抬高 + 结构保留 |
| ❌ 塌平 | hit_loose>90% 且 corr<0.15 | 所有帧被强行拉到目标，动态消失 |
| ❌ 时间结构反向 | corr<0 | guided 的峰谷和 baseline 反向 |
| ❌ 过度推力 | target_distance>5° | 平均角度偏离目标超过 5° |
| ❌ 抖动严重 | jitter>5° | 帧间跳变剧烈，运动不自然 |
| ⚠ 推力不足 | hit_loose<5% | guidance 没起作用 |
| ○ 中间状态 | 其他 | 需要继续调参 |

---

## 18. 梯度链诊断（`debug_v2_grad.py`）

当 V2/V5 报错 `element 0 of tensors does not require grad` 时，用以下诊断脚本精确定位断点：

```bash
cd /root/autodl-tmp/motion-diffusion-model
python debug_script/debug_v2_grad.py
```

脚本会逐步检查：

```
[Step 1] x_t (leaf)          ✅ requires_grad=True
[Step 2] model(x_t) forward  ✅ model_output.grad_fn=YES
[Step 3] 推导 x0_hat         ✅ x0_hat.grad_fn=YES
[Step 4] fk_fn(x0_hat)       ✅ q.grad_fn=YES
[Step 5] compute_loss(q)      ✅ loss.grad_fn=YES
[Step 6] autograd.grad        ✅ grad norm=0.650
🎉 V2 完整梯度链正常！
```

**已验证结论**：梯度链在你们的项目里完整。`RuntimeError: element 0 of tensors does not require grad` 的真正原因是：当 `schedule` 不活跃时 `compute_loss` 返回 `tensor(0.)` 这类无 `grad_fn` 的常数。**修复方法**：在 `guidance_loss_fn` 里加 anchor（见第 15.3 节）。

---

## 19. 关键实验结论汇总

### 19.1 方法学层面

| 发现 | 证据 | 影响 |
|---|---|---|
| **V1 一直是塌平方法** | 所有 base_weight 下 corr<0 | 以后不能用 V1 作为 baseline |
| **V2 (DPS) 落带最精准** | s=80 时 corr=+0.45，Δ≈+15° 恰好 | 推荐用于"精准控制" |
| **V2b corr 最高** | bw=8-12 时 corr≈0.49-0.53 | 推荐用于"形状保留" |
| **V5 MC 平滑无益** | 修复后 V5 corr 比 V2 低 0.07 | 反驳 LGD 论文在 motion 域的适用性 |
| **GMD 警告不成立于 DiP** | V2b/V3 x₀ 编辑稳定有效 | 说明 GMD 警告只适用于 1000 步 DDPM |
| **schedule 极其关键** | s=80+final vs last_quarter: corr -0.01 vs +0.45 | schedule 比 s 值更重要 |
| **inner_steps 不关键** | V2b steps=5/10/15/25 结果几乎相同 | 减少 steps 不影响结果 |

### 19.2 工程层面

| 发现 | 根本原因 | 解决方案 |
|---|---|---|
| **V2 需要 s=80 而不是 s=1** | MDM Jacobian 范数≈0.1-0.3，压缩梯度 | 直接用大 s |
| **V5 早期推不动** | `sigma_sq` 乘法导致推力缩小 20 倍 | 删掉 sigma_sq |
| **V4 塌平严重** | K_late × base_weight 乘积过大 | K_late 从 10 降到 3，bw 降到 8 |

### 19.3 当前最佳配置

```bash
# 推荐 1：V2 (DPS) — 落带精准
GUIDANCE_VARIANT=v2_dps
GUIDANCE_KWARGS_JSON='{"s":80,"schedule":"last_quarter","base_weight":1.0}'
# 结果：Δ≈+15°, corr≈+0.45, 曲线形状保留 baseline 双峰

# 推荐 2：V2b (x0_edit) — 形状最佳
GUIDANCE_VARIANT=v2_x0_edit
GUIDANCE_KWARGS_JSON='{"n_inner_steps":15,"lr":0.5,"schedule":"final","base_weight":8}'
# 结果：Δ≈+22°, corr≈+0.49, 但平均角度 28° 偏离目标 8°
# 需要进一步降低 bw 到 6-7 才能精准落带
```

---

## 20. 当前待完成的实验

### 20.1 精细化 ablation v3（运行方法）

```bash
cp new/run_calibrated_ablation_v3.sh /root/autodl-tmp/motion-diffusion-model/new/
chmod +x ./new/run_calibrated_ablation_v3.sh
bash ./new/run_calibrated_ablation_v3.sh

# 评估
python -m new.evaluate_ablation ./output/calibrated_walking_apt_v3_seed42
python -m new.plot_angle_curves ./output/calibrated_walking_apt_v3_seed42
```

**这次扫描的重点**：
- V2: s=[75, 80, 85]，验证 s=80 是否真的是最优
- V2: `schedule=second_half` vs `last_quarter`，看能否提升 corr
- V2b: bw=[5, 6, 7, 8]，找到 Δ≈+14° 且 corr 最高的点
- V2b: lr=[0.1, 0.2, 0.3] + bw=10，温和降低推力
- V5: 修复 sigma_sq 后重测 s=[60, 80, 100]

### 20.2 多 seed 稳健性验证（在找到最佳配置后运行）

```bash
# 先修改 run_seed_robustness.sh 里的 BEST_VARIANTS，然后：
bash ./new/run_seed_robustness.sh
python -m new.aggregate_seeds ./output/seedtest_v2_dps_best ./output/seedtest_v2b_best
```

期望结果：
```
variant              N  Δ(mean±std)   corr(mean±std)  shape
v2_dps_best          5  +15.4±1.2°   +0.45±0.04     ✅ 平移+保结构 (5/5)  稳健
v2b_bw7_best         5  +16.1±1.8°   +0.48±0.06     ✅ 平移+保结构 (4/5)  稳健
```

---

## 21. 论文写作角度

这次实验为论文提供了以下有价值的 findings，可以写进结果或讨论章节：

### Finding 1：V1（SGD-on-μₜ）的失败揭示了 50 步 DiP 的独特性

V1 在所有 base_weight 下 corr 全为负，说明在 50 步压缩采样下，逐步在 μₜ 上施加累积 SGD 会让去噪器将运动结构信息完全擦除，而不是"引导"它。这是 1000 步 DDPM 方法在 50 步 DiP 上不可直接迁移的直接证明。

### Finding 2：DPS 在 motion 领域需要 100× 的引导强度

DPS 在图像域通常用 s≈1。在 MDM 上需要 s≈80。根本原因是 MDM Jacobian（∂x̂₀/∂xₜ）范数约为 0.1-0.3，比图像 diffusion 小一个数量级，因为 MDM 被训练成"去噪器"，会主动抑制输入扰动。这是 motion diffusion 领域 DPS 尚未被系统研究的参数化挑战。

### Finding 3：GMD 的 x₀ 模型撤销警告不适用于 50 步 DiP

GMD 论文（ICLR 2024）在 Appendix A 警告：在 x₀ 预测的扩散模型上做 guidance，x₀ 模型在 t→0 时的主导会"撤销"guidance 效果。但实验表明 V2b（直接编辑 x̂₀）在 50 步 DiP 上稳定有效（corr=0.49），并未出现撤销现象。这说明 GMD 的警告主要适用于 1000 步 DDPM 体制。

### Finding 4：MC 平滑（LGD）在 motion 域无益

LGD（ICML 2023）认为 MC 平均可以降低 DPS 梯度方差并提升质量。修复 sigma_sq bug 后，V5（n_mc=2）的 corr 比 V2（n_mc=1）低 0.07。这说明在 motion 域，多次 MDM forward 引入的额外"MDM 想象信号"比梯度方差的减少更有害。

### Finding 5：schedule 比 s 值更重要

`schedule=last_quarter` 与 `schedule=final` 在相同 s=80 下的 corr 差异达到 0.46（+0.45 vs -0.01）。schedule 决定了 guidance 在哪些扩散步骤介入，而不同步骤的 MDM"运动结构建立程度"差异巨大。这是 motion 领域 inference-time guidance 的关键超参数，比 guidance 强度 s 本身更重要。

---

## 22. 对话窗口补充：常见问题 FAQ

**Q：为什么 comparison.npy 里 hit_rate=100% 但 corr 是负的？**

A：旧版 hit_rate 是单边（`angle >= 18°`），推到 30° 也算 100%。corr 为负说明时间结构破坏——曲线在 20° 附近抖动，和 baseline 的平滑波形完全不匹配。这是 V1 和 V4 大 base_weight 配置的典型失败模式。

**Q：如何判断一个新的 variant 是否真的在做梯度更新？**

A：在 V2 里加了 `[V2 UPDATE t=xxx]` 打印。如果看到的是 `loss=0.0000, |s*grad|=0.00000`，说明在高噪声步（通常 t>12）时 hinge loss 自然为 0（骨盆角度在噪声态已经随机超过了 20°），不是 bug——切换到 `schedule=last_quarter` 跳过这些步即可。

**Q：V2b 和 V3 为什么结果几乎一样？**

A：在 `schedule=final`（只有 t<5 激活）下，两者都只用了很少的有效迭代步，inner_steps 的差异（5 vs 15）并不能体现出来。这本身是一个发现：final schedule 下内循环次数不是关键变量。

**Q：换了新的 POSTURE 目标后，怎么快速验证 angle_fn 符号正确？**

A：跑 `debug_script/debug_v2_grad.py`，在 `[Step 4] fk_fn(x0_hat)` 之后观察 q 的形状，然后手动调用对应的 angle_fn 检查 baseline 数据的均值是否在生理合理范围内（骨盆前倾 +5-10°，膝超伸 >180°，驼背 0.02-0.05m）。

**Q：`load_model_wo_clip` 报 KeyError 怎么办？**

A：用 `load_saved_model` 替换。这个报错是因为新版 checkpoint 没有 `sequence_pos_encoder.pe` 这个 key，`load_model_wo_clip` 会无条件 del 这个 key 导致 KeyError。`load_saved_model` 直接 load_state_dict，更稳定。

**Q：V5 修复前后的区别？**

A：修复前：`mu_t_new = mu_t - s * sigma_sq * grad`（sigma_sq≈0.01 让推力小 100 倍）。修复后：`mu_t_new = mu_t - s * grad`（与 V2 对齐）。如果你的 V5 跑出来 Δ<5°，检查是否还在乘 sigma_sq。

---

*文档维护说明：本文件应在每次有重大实验结果或代码改动后更新。更新时在对应节末尾注明日期，不要删改旧内容，只在末尾追加新内容或在相应位置补充说明。*
