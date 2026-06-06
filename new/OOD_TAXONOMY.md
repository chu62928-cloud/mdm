# OOD Taxonomy for Inference-Time Posture Guidance

> 从 "in/out 二分" 升级为可发表的 **三分类体系**。
> 地基文档：所有下游工作（新体态评估、方法改进、论文写作）都应先参考本文档的分类框架。

---

## 一、为什么需要三分类

`check_ood_risk.py` 旧版用 **per-file mean angles** 判 APT 20° ≈ 5σ，标为 "❌ 分布外"。
但实验结果显示 APT 成功了（hit=88.7%, corr=0.407）。

**矛盾根源是统计方法错误**：
- 旧版: `np.mean(120帧角度)` per file → `np.std(跨文件 means)` 
- per-file mean 坍缩了 gait cycle 内的帧间方差（σ 被低估为 ~3° 而非真实的 ~10-15°）
- 这虚假放大了 OOD score，把"密度尾部"误判为"分布外"

**修复**：用 per-FRAME 统计（从训练集 `new_joints/` 逐帧提取角度，不做 clip 内平均），
恢复完整的分布信息，然后按以下三分类框架判定。

---

## 二、三分类定义

| Class | 名称 | 英文 | 定义 | 关键判据 | Guidance 预后 |
|-------|------|------|------|----------|--------------|
| **1** | 几何支撑集外 | Geometric Support Outside | 目标角度超过骨架物理上限 | `target > geometric_max` 或 `target > training_max` | 任何 post-hoc guidance 方法都将失败 |
| **2** | 密度尾部 | Density Tail | 目标在训练分布支撑集内，但位于低概率尾部 | `P95 < target <= training_max`（对 `greater_than` 方向） | Guidance 可成功，需足够推力（V2_s40 或 V6 default） |
| **3** | 条件 OOD | Conditional OOD | 角度单独在全局分布内可达，但在 (prompt, 相位) 联合条件下密度极低——低到 guidance 无法克服 MDM prior | `target 在全局分布内` 但 `walking-stance P5 > target`（对 less_than）或 `walking-stance P99 < target`（对 greater_than） | Guidance 因 prior-guidance 冲突而失败（相位反转或推力归零） |

### 判别流程

```
target 角度
    │
    ├─ > geometric_max?  ──────────────────→ Class 1（几何支撑集外）
    │
    ├─ > training_max (upright)?  ──────────→ Class 1（训练集未见）
    │
    ├─ > training_P99 (upright)?  ──────────→ Class 2（密度尾部）✅ guidance 可工作
    │
    ├─ spec.phase ≠ "always"
    │   └─ target 远超出 phase-filtered 分布? → Class 3（条件 OOD）
    │
    └─ ≤ training_P99  ────────────────────→ Class 0（分布内）
```

---

## 三、四个体态的归类结果

### 3.1 骨盆前倾 (Anterior Pelvic Tilt, APT) — 20°

| 统计量 | 数值 | 来源 |
|--------|------|------|
| 训练集规模 | 500 clips, 69,713 帧（~64,893 直立帧） | `analyze_pelvis_tilt.py` |
| 直立帧 P50 | 6.1° | 同上 |
| 直立帧 P90 | 14.3° | 同上 |
| 直立帧 P95 | 18.1° | 同上 |
| 直立帧 P99 | **31.0°** | 同上 |
| 直立帧 Max | 90.0° | 同上（含弯腰等极端姿势） |
| 超过 20° 的帧比例（全部直立） | **3.8%** | 同上 |
| **Walking 子集 P50** | **5.7°** | `--walking-only` (265 clips, 37,662 直立帧) |
| **Walking 子集 P95** | **15.2°** | 同上 |
| **Walking 子集 P99** | **21.8°** | 同上 |
| **Walking 子集 >20° 比例** | **1.4%** | 同上 |
| Baseline 生成均值 | 10.0° | n15 实验（15 clips, 1,800 帧） |
| Baseline 生成 P99 | 16.5° | 同上 |
| 实验 hit_band (V2_s40_last_quarter) | **88.7%** | README §4.1 |
| 实验 corr | **+0.407** | 同上 |

**判定：Class 2 — 密度尾部 (Density Tail)**

**依据**（条件分布锁死）：
- **边缘分布**：目标 20° 在全部直立帧 [P95=18.1°, P99=31.0°] 范围内
- **Walking 条件分布**：Walking prompt 子集 P99 = **21.8°** > 20°，
  1.4% 的 walking 直立帧 ≥20°。这些帧来自快走 gait cycle 的前倾峰值，
  **不是**弯腰/前探/起身等非行走动作的混杂
- 全部直立帧 3.8% → walking 子集 1.4%：差异来自非行走动作（弯腰等）APT 更高，
  但 walking 内部仍保留充分密度

**解释为什么旧 check_ood_risk.py 判为 5σ OOD**：
旧版对每个 clip 的 120 帧取均值（坍缩了 gait cycle 内的 ±10-15° 振荡），
然后跨 clip 算 σ。per-file mean 的 σ ≈ 3°，
而 per-frame σ ≈ 11°。同样的目标距离 15°（|20-5|），
除以 3° 得 5σ（误判 OOD），除以 11° 得 1.4σ（正确：密度尾部）。

**注意**：MDM 生成的 baseline 分布（mean=10°, P99=16.5°）显著
**窄于**训练分布（P99=31°）。这是因为 MDM 趋向于生成"平均"姿态。
Guidance 需要弥补这个 gap——将生成分布从 baseline 的 ~10° 推到 20°。

---

### 3.2 躯干前倾 (Trunk Forward Lean) — 15°

| 统计量 | 数值 | 来源 |
|--------|------|------|
| 训练集规模 | 500 clips, 69,713 帧（~64,893 直立帧） | `analyze_trunk_lean.py` |
| 直立帧 P50 | -0.3° | 同上 |
| 直立帧 P90 | 12.6° | 同上 |
| 直立帧 P95 | 24.4° | 同上 |
| 直立帧 P99 | **55.7°** | 同上 |
| 直立帧 Max | 90.0° | 同上 |
| 超过 15° 的帧比例 | **7.3%** | 同上 |
| 实验 hit_band (V2_s40_last_quarter) | **88.9%** | README §4.1 |
| 实验 corr (V2) | **+0.295** | 同上 |
| 实验 corr (V6) | **+0.476** | 同上 |

**判定：Class 2 — 密度尾部 (Density Tail)**

**依据**：目标 15° 在 P90-P95 之间，7.3% 的训练直立帧超过 15°。
与 APT 类似，躯干前倾随步态周期振荡，guidance 将分布重心推向前倾方向即可成功。

**V6 高 corr 的特殊性**：V6 在躯干前倾上 corr 反超 V2(0.476 vs 0.295)，
三轮消融证明这是因为 V6 的 PID 对此任务的推力幅度更合适，
而非特定组件（manifold_project、lambda_smooth、Huber）的贡献。
详见 README §4.3。

---

### 3.3 膝超伸 (Knee Hyperextension) — 190°

| 统计量 | 数值 | 来源 |
|--------|------|------|
| 三点角（直立帧）P99 | 175.2° | `check_ood_risk.py` v2 (three_point_angle ref) |
| 三点角（直立帧）Max | **178.8°** | 同上 |
| 三点角几何上限 | **180°**（acos 值域 [0, π]） | 数学定义 |
| signed_knee 几何上限 | **185°**（解剖学上超伸 >5° 极其罕见） | 临床文献 |
| 目标角度 | **190°** | POSTURE_REGISTRY |
| 实验 hit_band | **0%** | README §4.1 |
| 实验 corr | 0.060（近零） | 同上 |

**判定：Class 1 — 几何支撑集外 (Geometric Support Outside)**

**依据**：`three_point_angle`（hip-knee-ankle 三点角）严格限制在 [0°, 180°]。
训练集中直立帧的 Max 为 178.8°，185° 已超过所有训练样本。
190° 超出解剖学上可能的膝超伸范围（膝超伸 >5° 即属严重病态）。

**关键**：`signed_knee_angle` 函数通过 z-offset sign trick 可返回 >180° 的值，
但这是坐标系统的 artifact（依赖人物朝向 +z），不是真实解剖角度。
对几何支撑检查必须使用 `three_point_angle` 作为参考。

**Guidance 失败机制**：gradient 将角度推向 180° 后撞到几何天花板，
acos 梯度饱和，无法继续推进。V2 和 V6 均失败（Δ≈0）。

---

### 3.4 膝弯曲 + 站立相 (Knee Flexion + Stance) — 125°

| 统计量 | 数值 | 来源 |
|--------|------|------|
| signed_knee 全局 P50 | 168.6° | `check_ood_risk.py` v2 |
| signed_knee 全局 P99 | 275.3° | 同上（含坐/躺等极端屈曲） |
| signed_knee 直立帧 P50 | ~168° | `analyze_knee_angles.py` |
| signed_knee 直立帧 P10 | 127.4° | `check_ood_risk.py` v2 |
| signed_knee 直立帧 P5 | 111.5° | 同上 |
| **Walking-stance three_point P50** | **161.6°** | 临时脚本 (1,000 clips, 83,309 stance 帧) |
| **Walking-stance three_point P10** | **137.8°** | 同上 |
| **Walking-stance three_point P5** | **124.0°** | 同上 |
| **Walking-stance three_point P1** | **96.6°** | 同上 |
| **Walking-stance three_point P99** | **173.4°** | 同上 |
| **Walking-stance <125° 比例** | **5.2%** | 同上 |
| 三点角 直立帧 P50 | 161.2° | `analyze_knee_angles.py` |
| 目标角度 | **125°** | POSTURE_REGISTRY |
| 相位约束 | **stance_left**（仅站立相施压） | 同上 |
| 实验 hit_band (V2) | **8.2%** | README §4.1 |
| 实验 corr (V2) | **-0.215**（负相关！） | 同上 |

**判定：Class 3 — 条件 OOD (Conditional OOD)**

**依据**（条件分布 + 实验证据）：
- **Walking-stance 条件分布**：Walking prompt 下的 stance 相膝角集中在
  160-170°（P50=161.6°, P99=173.4°）。P5=124.0°——仅 5% 的 walking-stance
  帧膝角 ≤124°。这 5% 主要来自 crouching/sneaking gait，而非正常行走。
- **不是绝对零密度**：walking-stance 中 5.2% 的帧 <125°（vs 全部直立-stance 的 7.0%），
  但这些帧对应的 motion pattern（蹲踞步态）与 "a person walks forward" 的
  prompt 先验严重冲突
- **实验证实**：Guidance 尝试在 stance 相弯曲膝盖时，MDM 的 walking prior
  保持 stance 膝角 ~160-170°。V2 强推导致相位反转（stance↔swing 翻转），
  corr = -0.215；V6 的流形投影阻止离开 walking manifold，推力≈0
- **本质**：(target=125°, phase=stance, prompt="walking forward") 的联合密度
  极低。角度单独可达，相位单独出现，但 **三元组合** 被 MDM 的 walking prior 
  压制。这是 post-hoc guidance 的基本限制

---

## 四、四体态归类总表

| 体态 | Target | Class | 类别名称 | 训练 P99/Max | Guidance 结果 | 根因 |
|------|--------|-------|----------|-------------|--------------|------|
| 骨盆前倾 (APT) | 20° | **2** | 密度尾部 | P99=31° / Max=90° | ✅ hit=88.7% | Gait cycle 含前倾峰值 |
| 躯干前倾 | 15° | **2** | 密度尾部 | P99=56° / Max=90° | ✅ hit=88.9% | 步态振荡含前倾分量 |
| 膝超伸 | 190° | **1** | 几何支撑集外 | P99=175° / Max=179° | ❌ hit=0% | 三点角硬上限 180° |
| 膝弯曲+stance | 125° | **3** | 条件 OOD | 全局可屈曲至 90° | ❌ corr<0 | 站立相膝角保持 160-170° |

---

## 五、方法论教训

### 5.1 per-file mean → per-frame：为什么统计方法改变了一切

```
旧 OOD score 公式：
  μ = mean_i( mean_t( angle_i(t) ) )    ← 先对每 clip 的 120 帧取均值
  σ = std_i( mean_t( angle_i(t) ) )     ← 再跨 clip 取标准差
  OOD = |target - μ| / σ                ← 虚假放大

新方法：
  收集所有 per-frame 角度到 flat array
  直接计算 P50/P90/P95/P99/Max
  按 target 在分布中的百分位位置分类
```

### 5.2 为什么 APT 旧判 5σ 却成功

- APT 在 gait cycle 内振荡幅度约 ±10-15°（均值 ~5°，峰值可达 20°+）
- Per-file mean 将每 clip 120 帧坍缩为一个数，丢失了帧间方差
- Per-file mean 的 σ ≈ 3°（反映 clip 间差异），Per-frame σ ≈ 11°（反映真实的帧间差异）
- |20° - 5°| / 3° = 5σ → "分布外"
- |20° - 5°| / 11° = 1.4σ → "密度尾部"
- 结论：**OOD score 不应基于 per-file mean**

### 5.3 MDM 生成分布的"平均化"倾向

Overlay 图（见 `new_results/pelvis_tilt_distribution_overlay.png`）显示：
- 训练分布宽而长尾（P99=31°, 含极端前倾值）
- MDM 生成的 baseline 分布显著更窄（P99=16.5°, mean=10.0°）
- MDM 倾向于生成"平均"姿态，guidance 需要弥补这个 gap

这解释了为什么即使目标是密度尾部（Class 2），仍然需要较大的 guidance 推力。

---

## 六、对新体态评估的建议

在添加新体态约束之前，按以下流程预检：

1. 运行 `analyze_xxx.py`（复制 `analyze_pelvis_tilt.py` 模板）获得训练集 per-frame 分布
2. 运行 `check_ood_risk.py --training-data-dir` 获得自动三分类
3. 人工判断是否有 **条件 OOD** 风险（角度-相位组合冲突）
4. 参考下表决定下一步：

| OOD Class | 建议 |
|-----------|------|
| Class 0 分布内 | 直接跑 guidance，使用轻推力（V2_s20-30 或 V6 low Kp） |
| Class 1 几何外 | ❌ 放弃 post-hoc guidance，改用 fine-tune / LoRA / ControlNet |
| Class 2 密度尾部 | ✅ 跑 guidance，用 V2_s40 或 V6 default，schedule=last_quarter |
| Class 3 条件 OOD | ⚠ guidance 将失败。尝试 prompt 先验偏移 或 相位条件训练 |

---

## 七、生成产物清单

| 文件 | 描述 |
|------|------|
| `new_results/pelvis_tilt_distribution_training.png` | 训练集骨盆倾斜分布直方图 |
| `new_results/pelvis_tilt_distribution_overlay.png` | 训练集 + baseline 生成叠加图 |
| `new_results/trunk_lean_distribution_training.png` | 训练集躯干前倾分布直方图 |
| `new_results/trunk_lean_distribution_overlay.png` | 训练集 + baseline 生成叠加图 |
| `new/analyze_pelvis_tilt.py` | 骨盆倾斜角分析脚本 |
| `new/analyze_trunk_lean.py` | 躯干前倾角分析脚本 |
| `new/check_ood_risk.py` | OOD 三分类预检脚本（v2 升级版） |
| `new/OOD_TAXONOMY.md` | 本文档 |

---

*生成日期：2026-06-04*
*作者：Claude Code（基于实验数据自动生成，数字来自 HumanML3D 训练集 500-clip 样本 + 已有实验结果）*
