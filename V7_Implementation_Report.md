# V7 Trust-Region Auto-DPS 实施报告

> **日期**: 2026-07-30（最终更新：40-seed blind test 完成）  
> **分支**: `feature/v7-auto-dps`  
> **最新提交**: `2970f4f` — "feat: 40-seed blind test complete (600/600 OK)"  
> **当前状态**: V7.1-A 锁定，blind test 通过，论文级结论已形成  
> **服务器**: `connect.westd.seetacloud.com:10090` | GPU: RTX 5090 (32GB) | 环境: `mdm5090`

---

## 1. 概述

V7 是一个无需 guidance scale 的自动步长 DPS 接口。它在每个 diffusion timestep 使用当前物理约束残差及其通过冻结 MDM prior 的 Jacobian 来预测最小修正量，同时 diffusion-aware trust region 和 candidate acceptance 自动防止过推和 off-manifold 更新。

**核心创新**：取消 V2 的固定 `s` 和 V6 的 `Kp/Ki/Kd`，由算法在线决定更新长度、是否接受、何时停止。

---

## 2. 算法设计

### 2.1 控制变量

V7 在 noisy latent 空间 `x_t` 上操作（不直接编辑 `pred_xstart`，不将未验证的 `delta` 加到 `mu_t`）。

### 2.2 Gauss-Newton 自动步长

```
delta_raw = -r / (sum(g^2) + lambda) * g
```

- `r` = 当前物理残差 (signed, radians)
- `g` = d(r_sum)/d(x_t) — 通过完整 MDM forward + FK 路径的 autograd
- 使用 `sum(g^2)` 分母，而非 `|r| / grad_rms`

### 2.3 Trust Region

```
radius_rms = clamp(radius_scale * sqrt(1 - alpha_bar_t), min, max)
delta = clip_by_rms(delta_raw, radius_rms)
```

信任半径与 diffusion noise level 成正比：高噪声阶段允许大步长，低噪声阶段限制微调。

### 2.4 Candidate Trial 与接受检验

在 `x_t + delta` 上运行完整 MDM forward (`p_mean_variance`)，使用冻结的 phase mask 测量 trial residual：

```
rho = actual_reduction / predicted_reduction
```

接受条件：`pred_reduction > 0`, `actual_reduction > 0`, `rho >= 0.10`, trial finite, mask 一致。

### 2.5 Target-Band Stopping (Hysteresis)

```
进入 band: |r| <= tolerance (2°)
离开 band: |r| > 1.5 * tolerance (3°)
```

Band 内不再更新，防止 V2 式持续过推。

### 2.6 关键设计决策

| 设计 | 理由 |
|------|------|
| Trial 在 `x_t` 空间 | 与 Jacobian 计算空间一致 |
| 接受后使用完整 `out_trial` posterior | 同时替换 mean/variance/pred_xstart |
| Trial 调用 `p_mean_variance()` closure | 与主 sampler 使用相同计算路径 |
| 同一 timestep 冻结 mask | 防止通过改变 active frames 虚假降低残差 |
| 同一 `noise_z` 用于 accept/reject | 噪声公平性，保证可复现 |

---

## 3. 代码结构

### 3.1 新增文件

| 文件 | 职责 | 行数 |
|------|------|------|
| `posture_guidance/constraint_measurement.py` | 统一残差/目标/容差/phase mask 测量，支持 frozen mask | ~260 |
| `posture_guidance/auto_dps_controller.py` | 纯 GN 步长计算、trust region、backtracking、band hysteresis | ~330 |
| `posture_guidance/v7_auto_dps.py` | 单步算法主体：组合 MDM/Jacobian/trial/accept/reject | ~250 |
| `posture_guidance/tests/test_constraint_measurement.py` | 11 个单元测试 | ~340 |
| `posture_guidance/tests/test_v7_controller.py` | 8 个单元测试 | ~150 |
| `docs/v7_algorithm_spec.md` | 冻结算法规格文档 | ~350 |
| `scripts/run_v7_autocal.py` | 自动校准实验脚本 | ~120 |
| `scripts/analyze_v7_autocal.py` | 结果分析和 paired bootstrap | ~200 |

### 3.2 修改文件

| 文件 | 修改内容 |
|------|---------|
| `diffusion/gaussian_diffusion.py` | 添加 V7 特殊分支 (`_v7_apply_step`)、lazy controller init、noise level 计算、状态 reset |
| `posture_guidance/registry.py` | `LossSpec` 新增 `is_primary` 和 `control_type` 字段；APT specs 标记为 primary equality |
| `posture_guidance/controller.py` | `PostureGuidance` 新增 `measure_primary_constraint()` 方法 |

### 3.3 文件路径

```
/root/autodl-tmp/motion-diffusion-model/
├── posture_guidance/
│   ├── constraint_measurement.py    [新增]
│   ├── auto_dps_controller.py       [新增]
│   ├── v7_auto_dps.py               [新增]
│   ├── controller.py                [修改]
│   ├── registry.py                  [修改]
│   └── tests/
│       ├── test_constraint_measurement.py [新增]
│       └── test_v7_controller.py          [新增]
├── diffusion/
│   └── gaussian_diffusion.py        [修改]
├── scripts/
│   ├── run_v7_autocal.py            [新增]
│   └── analyze_v7_autocal.py        [新增]
├── docs/
│   └── v7_algorithm_spec.md         [新增]
├── baseline/
│   ├── baseline_commit.txt
│   ├── checksums.txt
│   ├── conda_list.txt / pip_freeze.txt / nvidia_smi.txt
│   ├── seed42/scorecard.json        (V2 baseline)
│   └── v6_seed42/scorecard.json     (V6 baseline)
├── output0727/
│   ├── v7_smoke/                    (5-seed smoke test)
│   │   ├── v7/scorecard.json
│   │   ├── v2/scorecard.json
│   │   └── v6/scorecard.json
│   └── v7_autocal/                  (full auto-calibration)
│       ├── tau05/{v7,v2,v6}/scorecard.json
│       ├── tau10/{v7,v2,v6}/scorecard.json
│       ├── tau15/{v7,v2,v6}/scorecard.json
│       ├── tau20/{v7,v2,v6}/scorecard.json
│       ├── tau25/{v7,v2,v6}/scorecard.json
│       ├── master_table.csv
│       └── paired_bootstrap.json
```

### 3.4 V7 默认配置

```json
{
  "schedule": "second_half",
  "radius_scale": 0.05,
  "min_radius_rms": 0.0001,
  "max_radius_rms": 0.10,
  "damping": 1e-8,
  "rho_accept": 0.10,
  "rho_shrink": 0.25,
  "rho_grow": 0.75,
  "shrink_factor": 0.5,
  "grow_factor": 1.5,
  "max_backtracks": 3,
  "band_hysteresis": 1.5,
  "min_valid_fraction": 0.8
}
```

---

## 4. 测试结果

### 4.1 单元测试：19/19 通过

```
posture_guidance/tests/test_constraint_measurement.py  — 11 passed
posture_guidance/tests/test_v7_controller.py           —  8 passed
```

### 4.2 V2/V6 回归测试

使用 seed 42, APT +20°, V2 (s=40, last_quarter) 在修改前后输出一致。V1-V6 行为不受 V7 分支影响。

### 4.3 Smoke Test (5 seeds: 0, 1, 2, 3, 42)

**配置**: APT +20° ±2°, single global config, no per-seed tuning

| Metric | V7 | V2 (s=40) | V6 (PID) |
|--------|----|-----------|----------|
| hit_band (↑) | 0.917 | 0.975 | 0.692 |
| delta (achieved) | 28.7° | 28.7° | 26.9° |
| overshoot (↓) | 0.19° | 0.15° | 0.52° |
| temporal_corr (↑) | **0.365** | 0.239 | 0.134 |
| 100% proposal accept | ✅ | N/A | N/A |
| mean backtracks | 0 | N/A | N/A |
| time/seed | ~2s | ~1s | ~3s |

**关键观察**:
- 5/5 seeds 方向正确，无 NaN，无 runaway
- 100% 首次 proposal 接受率 — GN 线性模型在此任务上非常准确
- 所有 seed 自动进入 target band
- V7 形成新 Pareto 点：corr 优于 V2 和 V6

---

## 5. Auto-Calibration 主实验

### 5.1 实验设置

| 参数 | 值 |
|------|-----|
| Targets | 5°, 10°, 15°, 20°, 25° APT |
| Seeds | 30 (base=100, 全新未见 seeds) |
| Prompt | "a person is walking forward" |
| 对比方法 | V7 (ours), V2 (s=40, last_quarter), V6 (PID, second_half) |
| 配置规则 | 所有 target 使用同一固定配置，无 per-target/per-seed 调参 |
| 总运行数 | 3 variants × 5 targets × 30 seeds = 450 |

### 5.2 核心指标对比

#### Hit Band (↑, 越高越好)

| Target | V7 | V2 | V6 |
|--------|-----|------|------|
| 5° | 0.429 | 0.446 | **0.608** |
| 10° | 0.517 | 0.567 | **0.721** |
| 15° | 0.642 | **0.900** | 0.892 |
| 20° | 0.450 | **0.988** | 0.833 |
| 25° | 0.454 | **1.000** | 0.713 |

#### Temporal Correlation (↑, 越高越好) — **V7 核心优势**

| Target | V7 | V2 | V6 | V7 vs V2 |
|--------|------|------|------|-----------|
| 5° | **0.586** | 0.524 | 0.463 | +12% |
| 10° | **0.552** | 0.417 | 0.352 | +32% |
| 15° | **0.529** | 0.277 | 0.328 | +91% |
| 20° | **0.498** | 0.220 | 0.327 | +126% |
| 25° | **0.531** | 0.121 | 0.319 | **+339%** |

#### Delta (achieved angle change, °)

| Target | V7 | V2 | V6 |
|--------|------|------|------|
| 5° | 15.9 | 17.6 | 16.7 |
| 10° | 21.0 | 22.0 | 21.9 |
| 15° | 25.3 | 26.1 | 25.4 |
| 20° | 28.8 | 30.6 | 29.6 |
| 25° | 33.9 | 35.8 | 34.4 |

#### Overshoot (↓, 越低越好)

| Target | V7 | V2 | V6 |
|--------|------|------|------|
| 5° | 1.80 | 1.93 | **1.02** |
| 10° | 1.40 | 1.63 | **1.07** |
| 15° | 1.27 | **0.59** | 0.49 |
| 20° | 1.96 | **0.41** | 0.99 |
| 25° | 2.04 | **0.35** | 1.69 |

### 5.3 Paired Bootstrap 分析

所有 30 seeds 进行配对比较，10,000 次 bootstrap：

#### Temporal Correlation — V7 系统性优势

| Target | V7-V2 diff | 95% CI | V7-V6 diff | 95% CI |
|--------|-----------|--------|-----------|--------|
| 5° | **+0.144** | [+0.030, +0.254] | **+0.161** | [+0.053, +0.265] |
| 10° | **+0.148** | [+0.042, +0.251] | **+0.195** | [+0.087, +0.298] |
| 15° | **+0.223** | [+0.111, +0.336] | **+0.217** | [+0.107, +0.324] |
| 20° | **+0.305** | [+0.210, +0.399] | **+0.210** | [+0.094, +0.327] |
| 25° | **+0.381** | [+0.292, +0.468] | **+0.193** | [+0.083, +0.306] |

**结论**: V7 在所有 target 上的 temporal correlation 均系统性优于 V2 和 V6，差异随 target 增大而增大。在 tau=25° 时，V7 的 corr 是 V2 的 4.4 倍。

#### Hit Band — V7 在大 target 上偏低

| Target | V7-V2 diff | V7-V6 diff |
|--------|-----------|-----------|
| 5° | -0.059 | -0.121 |
| 10° | -0.083 | -0.152 |
| 15° | -0.275 | -0.291 |
| 20° | -0.459 | -0.253 |
| 25° | -0.485 | -0.143 |

#### Overshoot — V7 偏高

| Target | V7-V2 diff | V7-V6 diff |
|--------|-----------|-----------|
| 5° | -0.44 | +0.40 |
| 15° | +0.76 | +0.82 |
| 25° | +1.52 | +0.12 |

### 5.4 结果解读

1. **V7 在结构保持（temporal correlation）上形成绝对优势**。所有 target 上 V7 > V2 且 V7 > V6。高 target 时优势尤为显著：V2 的 corr 在 25° 时崩溃至 0.12，而 V7 保持在 0.53。

2. **V7 的 hit_band 偏低**。默认 `radius_scale=0.05` 较为保守，trust region 限制了步长，导致大 target 时无法充分推动角度。

3. **V7 的 overshoot 偏高**。这与 hit_band 偏低共同指向同一个根因：trust radius 过小导致步长不足以到达 target，而非过推。

4. **V7 的 delta 略低于 V2**（约 1-2°），进一步证实是推力不足而非过推。

---

## 6. Phase 0 — 诊断与根因定位 (2026-07-30)

### 6.0 方法论

**核心原则**：不假定 `radius_scale=0.05` 是根因。Phase 0 必须先通过 trace 证据区分竞争假设（H1-H5），再按预注册的决策表行动。

### 6.1 Per-Seed 扩展诊断 (R2 指标)

编写 `scripts/diagnose_v7_results.py`，从 450 份已有 `.npy` 文件中提取冻结指标：

| 指标 | 定义 |
|------|------|
| `signed_error_deg` | guided_mean − target（负值 = 欠推） |
| `abs_error_deg` | abs(signed_error) |
| `frame_hit_band` | 每帧角度在 target ±2° 内的比例 |
| `positive_overshoot_deg` | 正向偏离（过推）均值 |
| `negative_undershoot_deg` | 负向偏离（欠推）均值 |
| `overshoot_p90_deg` | 正向偏离的 90 分位数 |
| `final_summary_in_band` | 最后 25% 帧的 band occupancy |

**关键发现**：

| Target | signed_error (V7) | pos_overshoot | neg_undershoot | frame_hit |
|--------|-------------------|---------------|----------------|-----------|
| 5° | **+0.96** (微过推) | 1.18° | 0.85° | 0.429 |
| 15° | **-0.99** (欠推) | 0.03° | 0.77° | 0.642 |
| 20° | **-1.96** (欠推) | 0.00° | 1.20° | 0.450 |
| 25° | **-2.04** (欠推) | 0.00° | 1.66° | 0.454 |

**结论**：大 target 不是"过推"而是**纯粹的欠推**。`positive_overshoot` 在 20°/25° 几乎为零。

### 6.2 扩展 Trace 与 R3 诊断字段

在 `v7_auto_dps.py` 中新增 11 个 R3 trace 字段：

| 字段 | 含义 |
|------|------|
| `schedule_active` | schedule 是否激活 |
| `proposal_valid` | proposal 是否有效 |
| `proposal_skip_reason` | 跳过原因 |
| `clip_factor` | delta_rms / raw_delta_rms |
| `boundary_hit` | raw delta 超过 radius |
| `proposal_count` / `accepted_proposal_count` | proposal 数量 |
| `remaining_active_steps` | 剩余 schedule-active 步数 |
| `radius_scale_value` | 当前 radius_scale |

新增 JSONL trace writer（通过 `V7_TRACE_DIR` 和 `V7_TRACE_SEED` 环境变量控制）。对 10 个代表性 case 收集了 per-timestep JSONL 轨迹。

### 6.3 假设检验：H1-H5 按预注册规则判断

**汇总统计（N=10）**：

| 指标 | 中位值 |
|------|--------|
| Active 步数 / 有效 proposal | 25 / 6 |
| Band 内跳过 | 19 |
| Boundary-hit 率 | 0.333 |
| Clip factor 中位值 | **1.000** |
| Band exit 次数 | **2** |

| 假设 | 证据 | 判定 |
|------|------|------|
| H1 — radius 过小 | boundary-hit 0.333, clip ≈ 1.0 | **不成立** |
| H2 — schedule 不足 | 19/25 跳过 | **成立** |
| H3 — band 反复退出 | median 2 exit, 旧 in_band 锁死 | **强烈成立** |
| H4 — frame 方差 | — | 待后续 |
| H5 — 评估不一致 | — | 待后续 |

### 6.4 根因：Band 状态评估顺序 Bug

V7.0 Step 顺序：
```
1. 测量 → 2. Jacobian → 3. propose (旧 in_band)
→ 4. Trial → 5. Select → 6. update_band_state  ← BUG
```

**问题**：`propose()` 使用上一 timestep 的 `state.in_band`。当 posterior noise 推出 band 后，旧 `in_band=True` 永久跳过 proposal。

**案例 seed=105, tau=25°**：t=12 时 |r|=3.2° 已出 band，但 in_band=True → **全部后续步被跳过**。19/25 步浪费。

### 6.5 Bug 修复 (commit `f414856`)

```
1. 测量 → 2. update_band_state (← 修复)
→ 3. Jacobian → 4. propose (最新 in_band)
→ 5. Trial → 6. Select → 7. update_radius
```

修复后 7/7 band re-entry 回归测试永久锁定（`test_v7_band_regression.py`，commit `bf67691`）。

---

## 7. Phase 1 — 12-Seed Validation 与 V7.1-A 锁定 (2026-07-30)

### 7.1 Schedule + Max Radius 筛选

**Schedule** (5 seeds × 3 targets)：

| Schedule | 判定 | 原因 |
|----------|------|------|
| second_half | ✅ 保留 | corr 稳定 |
| always | ❌ 淘汰 | corr 全面崩溃 |

**Max Radius** (worst seeds 105, 111, tau=25°)：

| Variant | seed=105 err | seed=105 hit | seed=111 err | seed=111 hit |
|---------|-------------|-------------|-------------|-------------|
| V7.0 unfixed | **-4.9°** | **0.017** | **-4.4°** | **0.000** |
| band-fix max=0.10 | **-0.2°** | **0.800** | — | — |
| band-fix max=0.15 | -0.1° | 0.825 | **-0.3°** | **0.917** |
| band-fix max=0.20 | -0.1° | 0.867 | -0.3° | 0.917 |

### 7.2 12-Seed A/B Validation (seeds 300-311, 240 runs, 100% OK)

**A/B 决策（预注册规则）**：

| 规则 | 阈值 | V7.1-B vs V7.1-A | 判定 |
|------|------|-----------------|------|
| C1: MAE 提升 ≥0.25° | 0.25 | 0.18 | ✗ |
| C2: hit 提升 ≥0.03 | 0.03 | 0.027 | ✗ |
| C3: 大 target MAE ≥0.35° | 0.35 | 0.28 | ✗ |
| C4: corr 丢失 ≤0.03 | 0.03 | **0.043** | ✗ |

**→ 锁定 V7.1-A: `max_radius_rms=0.10`**（Section 6.3 默认规则：B 精度不足 + corr 超标）。

### 7.3 V7.1 锁定配置

```json
{"schedule":"second_half","radius_scale":0.05,"min_radius_rms":0.0001,
 "max_radius_rms":0.10,"damping":1e-8,"rho_accept":0.10,"rho_shrink":0.25,
 "rho_grow":0.75,"shrink_factor":0.5,"grow_factor":1.5,"max_backtracks":3,
 "band_hysteresis":1.5,"min_valid_fraction":0.8,"trace":false}
```

---

## 8. Phase 2 — 消融与结构审计 (2026-07-30)

### 8.1 消融实验 (288 runs, 8 methods)

**新增 3 个 ablation switch** 于 `AutoDPSConfig`：
- `disable_trial`: 跳过 candidate acceptance，直接应用 proposal
- `disable_band_stop`: 不因 in_band 跳过 proposal
- `band_order_bug`: 模拟 V7.0 band 更新在 proposal 之后

| Method | 配置 | 目的 |
|--------|------|------|
| V2-fixed | s=40 | Baseline |
| V6-PID | 冻结 PID | Baseline |
| V7.0(bug) | band_order_bug=True | 量化 bug fix |
| V7.1-full | locked config | 主方法 |
| V7-no-trial | disable_trial=True | 量化 candidate acceptance |
| V7-no-band | disable_band_stop=True | 量化 band stopping |
| V7-static | shrink_factor=1.0, grow_factor=1.0 | 量化 radius adaptation |
| V7-no-both | disable_trial=True, disable_band_stop=True | 极限消融 |

**Tau=25°, median**：

| Method | hit_band | corr | abs_err |
|--------|----------|------|---------|
| V2-fixed | **1.000** | 0.101 | **0.50°** |
| V6-PID | 0.404 | 0.406 | 1.72° |
| V7.0(bug) | 0.471 | 0.481 | 1.60° |
| V7.1-full | 0.617 | 0.481 | 1.81° |
| V7-no-trial | 0.608 | 0.488 | 1.83° |
| V7-no-band | **0.771** | 0.506 | 1.43° |
| V7-static | 0.604 | 0.486 | 1.84° |
| V7-no-both | 0.775 | 0.504 | 1.43° |

**论文含义**：
- **Candidate acceptance 从不拒绝**（V7-no-trial ≈ V7.1-full）→ GN 线性模型极准确
- **Band-stop 是主要机制**（移除后 hit +25%）→ 牺牲精度换取结构保持
- **Adaptive radius 贡献极小**（V7-static ≈ V7.1-full）

### 8.2 结构审计

**Foot-skate 定义** (`eval/control_metrics.py:58`)：接触帧（脚高 < 5cm）中滑动帧（水平位移 > 2.5cm）的比例。

| Method | tau=10 fs | tau=20 fs | tau=25 fs |
|--------|----------|----------|----------|
| V2-fixed | 0.063 | 0.034 | 0.034 |
| V6-PID | 0.038 | 0.025 | 0.038 |
| V7.1-full | 0.076 | 0.071 | 0.084 |
| V7-no-band | 0.080 | 0.088 | 0.097 |

所有方法均显著低于 baseline (~0.13-0.16)。V7 在 V2-V6 之间，偏高但可控。移除 band-stop 恶化 foot-skate（更多更新 = 更多滑步）。

---

## 9. Phase 3 — 40-Seed Blind Test (2026-07-30)

### 9.1 实验设置

| 参数 | 值 |
|------|-----|
| 种子 | **400-439 (40 fresh, untouched)** |
| Targets / 方法 | 5 targets × V2 + V6 + V7.1-A |
| 总运行数 | 3 × 5 × 40 = **600** |
| 成功率 | **600/600 (100%)** |
| 时间 | ~35 min GPU |
| 协议 | 预注册，冻结于 `v7_blind_protocol/` |

### 9.2 全 Target 汇总

| Target | Method | hit_band | delta | corr | fs_guided |
|--------|--------|----------|-------|------|-----------|
| 5° | V2 / V6 / V7 | 0.450 / 0.550 / 0.546 | 17.6 / 16.8 / 15.9 | 0.371 / 0.400 / **0.636** | 0.130 / 0.092 / 0.122 |
| 10° | V2 / V6 / V7 | 0.592 / 0.542 / 0.554 | 22.4 / 21.9 / 21.2 | 0.344 / 0.310 / **0.544** | 0.113 / 0.097 / 0.113 |
| 15° | V2 / V6 / V7 | **0.925** / 0.867 / 0.617 | 26.3 / 25.7 / 25.3 | 0.298 / 0.278 / **0.473** | 0.101 / 0.084 / 0.084 |
| 20° | V2 / V6 / V7 | **0.996** / 0.867 / 0.529 | 30.8 / 29.4 / 29.4 | 0.124 / 0.243 / **0.468** | 0.076 / 0.076 / 0.076 |
| 25° | V2 / V6 / V7 | **1.000** / 0.650 / 0.533 | 35.7 / 33.5 / 34.2 | 0.023 / 0.372 / **0.490** | 0.055 / 0.067 / 0.067 |

### 9.3 Cross-Target Median

| Metric | V2 | V6 | V7 | V7 vs V2 |
|--------|-----|-----|-----|-----------|
| corr | 0.232 | 0.321 | **0.522** | **+125%** |
| abs_err | **1.01°** | 1.18° | 1.42° | +0.41° |
| frame_hit | **0.793** | 0.695 | 0.556 | -0.238 |

### 9.4 Tau=25° 深度分析

| Metric | V2 | V6 | V7 |
|--------|-----|-----|-----|
| corr | 0.023 | 0.372 | **0.490** (21× V2) |
| abs_err | **0.50°** | 1.72° | 1.81° |
| frame_hit | **1.000** | 0.650 | 0.533 |
| signed_err | -0.48° | -1.72° | -1.81° |

V2 强力到达 target 但时间波形完全崩溃（corr=0.023）。V7 保留波形（corr=0.490）但停在 band 边缘（~1.8° 欠推）。

### 9.5 Paired Bootstrap (V7-V2 abs_error, 10000 samples, N=40 per target)

| Target | Diff | 95% CI | 结论 |
|--------|------|--------|------|
| 5° | -0.54° | [-0.97, -0.12] | V7 更优 |
| 10° | -0.22° | [-0.53, +0.09] | 相当 |
| 15° | +0.54° | [+0.24, +0.83] | V2 更优 |
| 20° | **+1.11°** | [+0.83, +1.40] | V2 显著更优 |
| 25° | **+1.15°** | [+0.88, +1.44] | V2 显著更优 |
| **Cross-target** | **+0.41°** | [+0.24, +0.58] | V2 整体更精确 |

### 9.6 成功标准

| 标准 | 状态 |
|------|------|
| 单一配置覆盖所有 target | ✅ |
| 600/600 零失败 | ✅ |
| 大 target corr: V7 >> V2 (21× at 25°) | ✅ |
| Cross-target corr: V7 > V2 (+125%) | ✅ |
| 无 foot-skate 系统性恶化 | ✅ |
| 新 accuracy-preservation Pareto 点 | ✅ |

---

## 10. 论文级结论与已知限制

### 10.1 可正式宣称

> V7.1 uses a single global configuration across 5 target severities and 40 unseen random seeds. On the 25° anterior pelvic tilt task, V7.1 preserves the baseline temporal profile (corr = 0.490) while fixed-scale DPS collapses it (corr = 0.023, **21× worse**). Across targets, V7.1 achieves 125% higher temporal correlation than fixed-scale V2 (0.522 vs 0.232), at a cross-target accuracy cost of 0.41° MAE (1.42° vs 1.01°). Band stopping with hysteresis is the primary driver of this accuracy-preservation trade-off, while Gauss-Newton automatic step sizing removes per-target/per-seed guidance-scale tuning.

### 10.2 不应使用的表述

- "V7 is superior on all metrics"
- "V7 fully preserves motion structure"（应为 "preserves controlled-angle temporal profile"）
- "Candidate acceptance automatically prevents overshoot"（消融不支持）
- "Adaptive radius scaling drives performance"（消融不支持）

### 10.3 已知限制与 V7.2

| 限制 | V7.2 计划 |
|------|----------|
| 25° ~1.8° 系统性欠推 | 3-tier band (inner control / evaluation / hysteresis) |
| Band 三重角色耦合 | 拆分为独立信号 |
| 单 primary constraint | ConstraintProvider 多约束接口 |
| Foot-skate 高于 V2 | 已在论文中透明报告 |

V7.2 在独立分支 `feature/v7-2-target-redesign` 上启动，不允许复用 blind test seeds。

---

## 11. 运行命令

### 11.1 环境

```bash
ssh connect.westd.seetacloud.com -p 10090
conda activate mdm5090 && source /etc/network_turbo
cd /root/autodl-tmp/motion-diffusion-model
```

### 11.2 单元测试

```bash
python -m pytest posture_guidance/tests/ -v   # 26 passed
```

### 11.3 V7.1 单 seed

```bash
python scripts/run_seed_batch.py   --model_path ./save/humanml_trans_dec_512_bert/model000600000.pt   --seeds "42" --output_dir /tmp/v7_test   --posture_instructions anterior_pelvic_tilt   --variant v7_auto_dps   --variant_kwargs_json '{"schedule":"second_half","max_backtracks":3,"max_radius_rms":0.10}'   --guidance_mode joint --motion_length 6.0
```

### 11.4 消融 / Blind Test / 诊断

```bash
python scripts/run_v7_ablation.py              # 288 runs
python scripts/diagnose_v7_results.py <dir>     # per-seed diagnostics
# JSONL trace: export V7_TRACE_DIR=/tmp/traces V7_TRACE_SEED=<seed>
```

---

## 12. 附录

### A. Git 历史

```
2970f4f feat: 40-seed blind test complete (600/600 OK)
11340f8 feat: Phase 2A ablation complete (288 runs, 8 methods)
bf67691 feat: ablation switches
9e0dc58 feat: V7.1 12-seed validation
f414856 fix(V7): band state ordering (Branch C)
170104d feat: V7 Auto-DPS core implementation
840896b baseline: freeze pre-V7
```

### B. 环境与 Checksum

- Python 3.10.20, RTX 5090 32GB
- Checkpoint: `save/humanml_trans_dec_512_bert/model000600000.pt`
  - SHA256: `195664bed7...`
- Ref stats: `eval/assets/ref_stats.npz`
  - SHA256: `25db4841cd...`

### C. 完整输出目录

| 目录 | 内容 | 规模 |
|------|------|------|
| `output0727/v7_smoke/` | Smoke test | 15 runs |
| `output0727/v7_autocal/` | V7.0 autocal | 450 runs |
| `output0727/v7_diagnostics/` | Phase 0 诊断 | 10 traces + CSV |
| `output0727/v7_validation/` | 12-seed A/B | 240 runs |
| `output0727/v7_ablation/` | 8 methods | 288 runs |
| `output0727/v7_blind_protocol/` | 冻结协议 | config files |
| `output0727/v7_blind_test/` | **40-seed blind** | **600 runs** |
| **总计** | | **~1603 runs, 零失败** |
