# 肌肉引导 MDM —— 交接文档

本文档面向负责 MDM 生成端的同学。目标：让 MDM 生成的骨架经过一个**冻结的 motion2muscle 代理模型**预测肌肉激活，再用一套**有方向性的体态损失（posture loss）**把生成结果推向"病态姿势"，同时产出对应的肌肉激活。

---

## 1. 整体数据流

```
文本/条件 ──► MDM ──► 生成的运动 x0 (B, T, 263, HumanML3D 表示)
                          │
                          ▼  反归一化(MDM 统计) → 重归一化(proxy 统计)
              冻结的 motion2muscle 代理 (transformer)
                          │
                          ▼
              肌肉激活 a (B, T, 402)，范围 [0,1]
                          │
                          ▼  按功能肌群 mean-pool（muscle_rollup）
              功能肌群激活
                          │
                          ▼  与 reference 体态比较（posture_loss_torch）
              posture loss（标量，可微）
                          │
                          ▼  对 x0 求梯度，梯度**上升**（最大化 loss）
              把 x0 往病态方向推
```

一句话：**代理模型把"骨架→肌肉"这件事变成一个可微函数，posture loss 在肌肉空间里定义"什么叫病态"，两者串起来就成了 MDM 的一个引导项。**

---

## 2. 交接清单（我这侧交付的东西）

| 文件 | 作用 |
|------|------|
| `net_best_*.pth`（在 `output_vqfinal/transformer_baseline_full/`） | 训好并冻结的 motion2muscle transformer 权重。**见 §6.4 选哪个 checkpoint** |
| `muscle_rollup.py` | 402 条肌束 → 功能肌群的映射（mean-pool）。loss 的"词典" |
| `posture_loss.py` | 体态损失的 **numpy 原版**（含 POSTURE_PRIORS 病态模板表、参考构建、合成扰动）。当作"真值/文档"用 |
| `posture_loss_torch.py` | 体态损失的 **torch 可微版**（引导时实际调用的就是它）。已逐项核对，与 numpy 版数值一致（误差 ~1e-8） |
| `muscle_guidance.py` | 引导封装层 `MuscleGuidance` + MDM 采样循环 hook 的伪代码。**你主要对接这一个类** |
| `mint_cols.json`（**需要我补交**，见 §6.3） | 402 列肌束的**有序**列名。loss 全靠它把输出下标对应到肌肉 |

代理模型的代码（transformer 定义）来自 `motion2muscle-main`，加载权重时需要用它原本的模型类。

---

## 3. 代理模型的输入/输出契约（最关键，决定适配器怎么写）

根据 Muscles in Time 论文，这个 transformer 的契约是：

- **输入**：用 Guo 等人（HumanML3D，CVPR'22）的预处理得到的 **263 维**逐帧描述子，形状 `x ∈ R^{T×263}`。
- **训练片段**：1.4 秒、20 fps，即 **T = 28 帧**。
- **输出**：`f(x) = y ∈ R^{T×402}`，每帧 402 条肌束激活，**范围 [0,1]**。前 80 条是下肢模型（Lai/Arnold），后 322 条是胸腰段模型（Bruno）。
- **结构**：16 层标准 transformer（论文里效果最好的版本）。

**为什么这对接入是天大的好消息**：MDM 在 HumanML3D 上生成的运动，用的**恰好就是同一个 263 维 HumanML3D 表示**。所以 MDM 输出 → 代理输入，维度天然对齐，不需要重训任何骨架编码器。真正要小心的不是"维度对不对"，而是 §6 里的几个**归一化/帧率握手**。

> MDM 的另一个有利特性：它直接预测 `x0`（干净信号本身），而不是噪声。也就是说采样的**每一步都能拿到一个干净运动估计 `x0_hat`**，可以直接喂给代理——这正是 §5 引导方案的前提。

---

## 4. Loss 是怎么设计的

### 4.1 表驱动 + 有方向性

`POSTURE_PRIORS` 是一张**病态模板表**，目前内置 4 种：前骨盆倾斜 `anterior_pelvic_tilt`、后骨盆倾斜 `posterior_pelvic_tilt`、头前伸 `forward_head_posture`、Trendelenburg 步态。加新病态**只改表、不改代码**。

每种病态由**四类机制**描述（对应文献里姿势代偿的四种模式），全部相对 `reference_acts`（"正常"体态的每肌群平均激活）来判定：

1. **antagonist（拮抗失衡）**：主动肌相对其拮抗肌异常占优。两种判定：拮抗肌基线够大时用**比值**模式，基线接近 0 时退化为**绝对超出**模式。
2. **chain（代偿链）**：原动肌被抑制 **且** 替代肌过度激活——两个门控相乘（必须同时成立才触发）。
3. **synergy（协同失衡）**：协同肌群里主导肌占总激活的份额下降。
4. **stabilizer（稳定肌抑制）**：深层稳定肌跌破 `ref × (1 − 阈值比例)`。

**方向性是核心**（不是对称的）：

- `loss = 0` → 激活与 reference 一致（正常）；
- `loss > 0` → 激活朝命名病态的方向偏离。

所以做引导时——**要生成病态，就最大化这个 loss（梯度上升）**。它本来就是为 MDM 引导设计的。

### 4.2 reference 机制（你的设计：第一波生成当作 ref）

reference 是一个 `{肌群名: 标量平均激活}` 的字典，用一段**正常样本**经 `build_reference_from_activations` 算出来。在我们的流程里，**这段正常样本就是 MDM 第一波（无引导）生成的形态**：

- **阶段一**：MDM 正常采样 → 正常运动 `x0_ref` → 过代理 → 算出 `reference_acts`，**冻结**（之后是常量，不带梯度）。
- **阶段二**：MDM 再生成，开启引导。每步把 `x0_hat` 过代理得到激活，和阶段一的 `reference_acts` 比，按病态方向被推动。

> 一个内置的自检（§7）：拿阶段一的参考样本自己去打分，loss 应当 ≈ 0。我已在 torch 版上验证：参考样本上 loss≈7e-4，合成 APT 扰动上 loss≈1.14，方向正确。

### 4.3 权重与归一化

默认分量权重 `{antagonist:1.0, chain:1.5, synergy:0.5, stabilizer:0.5}`，并按各分量"触发的规则条数"归一（`normalize_by_count=True`），使不同病态之间可比。这些都能在调用时覆盖。

---

## 5. 怎么接入 MDM（核心步骤）

你主要对接 `MuscleGuidance`（见 `muscle_guidance.py`）。它把"代理 + 归一化握手 + roll-up + 可微 loss"全部封好，对外只暴露三件事：`build_reference()`、`loss()`、`guidance_grad()`。

### 5.1 构造

```python
from muscle_guidance import MuscleGuidance
guide = MuscleGuidance(
    proxy            = load_frozen_proxy(ckpt_path),  # 你的 transformer 类 + net_best_*.pth
    mint_cols        = json.load(open("mint_cols.json")),
    posture_name     = "anterior_pelvic_tilt",
    mdm_mean=..., mdm_std=...,      # MDM 训练用的 Mean/Std（见 §6.1）
    proxy_mean=..., proxy_std=...,  # 代理训练用的 Mean/Std
    # same_normalization=True,     # 若确认两边用同一套统计，可置 True 跳过换算
    device="cuda",
)
```

代理在内部已被设成 `eval()` 且参数 `requires_grad_(False)`——**只冻结参数，不冻结输入路径**，这样梯度能从 loss 经代理回传到运动 `x0`。

### 5.2 两阶段采样

```python
# 阶段一：普通采样得到参考，并冻结 reference_acts
x0_ref = mdm.sample(model_kwargs)                 # 注意你工程里的张量布局
guide.build_reference(to_BTC(x0_ref))             # 转成 (B, T, 263)

# 阶段二：带引导采样（见 §5.3 的 hook）
x0_path = mdm.sample(model_kwargs, guidance=guide, scale=..., t_start_guidance=...)
```

### 5.3 引导 hook（推荐：x0 空间引导，便宜又稳）

因为 MDM 每步给出 `x0_hat`，最稳的做法是**在干净空间里 nudge `x0_hat`**，梯度只经过代理、**不经过庞大的扩散网络**，所以很省。`muscle_guidance.py` 末尾给了可直接改的伪代码，要点：

```python
out   = diffusion.p_mean_variance(model, x_t, t, **kw)   # 拿 pred_xstart
x0    = out["pred_xstart"]
if int(t[0]) <= t_start_guidance:                        # 只在低噪声步引导
    grad, L = guide.guidance_grad(to_BTC(x0))            # +grad 指向病态
    x0 = x0 + scale * to_model_layout(grad)              # 梯度**上升**（最大化 loss）
    out["pred_xstart"] = x0
    out["mean"], _, _ = diffusion.q_posterior_mean_variance(x0, x_t, t)  # 用 nudge 后的 x0 重算后验
# 然后照常采样 x_{t-1}
```

**符号务必记牢**：loss 越大越病态，所以是 `x0 += scale * grad`（上升）。写成下降也行，但那时要对 `-loss` 求梯度。

### 5.4 可选：更强的 x_t 空间引导

如果 x0 空间引导太弱，可以让梯度也穿过 MDM（对 `x_t` 求梯度，p_mean_variance 之前不要 detach）。更强但更贵（每步要 backprop 整个扩散网络）。一般用不到，先试 §5.3。

---

## 6. 四个必须对齐的"握手项"（接入失败 90% 出在这里）

### 6.1 归一化（头号风险）

MDM 内部在**归一化空间**工作并输出。代理是在 Guo 预处理（含 z-score 归一化）后的特征上训练的。两者**必须**：要么用**同一套** `Mean.npy/Std.npy`（那就 `same_normalization=True` 直接喂）；要么显式地"用 MDM 统计反归一化 → 用代理统计重归一化"（`MuscleGuidance._mdm_to_proxy` 已实现）。**第一件事就是去 motion2muscle 的 dataloader 里确认它训练时到底用了哪套统计、是否做了 z-score。** 我会把代理那套统计随权重一起给你。

### 6.2 帧率与序列长度

代理在 **20 fps、T=28**（1.4s）上训练。HumanML3D / MDM 也是 20 fps，帧率天然对齐。但 MDM 生成长度可变（最长 196 帧）。transformer 理论上能吃任意 T，但**离训练长度越远越不准**。建议你先做一个验证：用代理在 28 帧和在完整长度上各跑一遍，看激活是否仍合理；必要时把长序列切成 28 帧窗口（可重叠）分别预测。

### 6.3 mint_cols（402 列的有序列名）

loss 完全靠 `mint_cols` 把代理输出的第 i 维对应到具体肌肉。**这个有序列名列表必须来自代理的训练数据头，且顺序一致**，否则 loss 全错但不报错。我需要从 musint 数据集导出这 402 个列名存成 `mint_cols.json` 交给你（§2 已标注待补交）。

### 6.4 选哪个 checkpoint

目录里有 `net_best_div/fid/loss/matching/top1.pth` 和 `net_last.pth`。div/fid/matching/top1 这些名字是 T2M 那套**生成评测**指标，对"肌肉回归代理"不一定是对的选择标准。对我们而言，**回归误差最低的 `net_best_loss.pth` 最可能是正确的代理权重**（或按 PCC 最好的那个）。我会确认后明确告诉你用哪一个——**先别随便挑**。

---

## 7. 验证 / Sanity checks（接好后请逐条过）

1. **参考自洽**：把阶段一的 `x0_ref` 再喂回 `guide.loss(x0_ref)`，应当 ≈ 0。不为 0 → 多半是归一化或 mint_cols 错了。
2. **梯度存在且有限**：`guidance_grad` 返回的 grad 非 None、`isfinite().all()`、范数随 `scale` 起作用。
3. **方向正确**：对一段已知正常运动，开引导若干步后，`guide.loss` 应当**上升**（朝病态），且可视化骨架确实出现该病态特征。
4. **激活合理**：代理输出全在 [0,1]、没有大面积 NaN/饱和。
5. **合成扰动对照**：可用 `posture_loss.make_synthetic_distortion` 对正常激活造一个该病态的"标准答案"，确认 loss 行为与之一致（我已验证 numpy/torch 两版数值一致）。

---

## 8. 调参建议（先从这里起步再扫）

- `scale`（引导强度）：从个位数起，按对数尺度扫；太大→运动崩坏/不自然，太小→看不出病态。
- `t_start_guidance`：只在**低噪声后段**（如最后 30–50% 步）引导，此时 `x0_hat` 才有意义。
- 必要时对 `x0_hat` 做 HumanML3D 合法范围裁剪，避免推到分布外。
- 想强调某一类机制（如更强调代偿链）就调 `component_weights`。

---

## 9. 注意事项

- **代理只冻结参数、不要包 `no_grad` 前向**，否则梯度断了引导失效。`build_reference` 里用了 `no_grad`（对的，参考是常量）；`loss/guidance_grad` 里用 `enable_grad`（对的）。
- **病态是持续模式**：antagonist/chain/synergy 三类都先做时间平均再算 loss；stabilizer 是逐帧惩罚。这是有意为之，别改。
- 合成数据是仿真，存在 sim-to-real gap；这里只用于"相对 reference 的方向性引导"，不要当作真实 EMG 解读。
- 这套 loss 不依赖具体病态种类的硬编码——换病态只需换 `posture_name`（前提是该病态已在 `POSTURE_PRIORS` 里）。
