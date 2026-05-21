# MDM 22-Joint Skeleton: Posture Representability Analysis

> 这份文档评估在 HumanML3D 22-joint SMPL 骨架表示下，哪些病态体态可以被
> 角度/几何量约束精确表征，哪些只能粗略近似，哪些根本无法被表征。
> 用于指导后续病态体态约束的设计范围。

## 一、22 关节的解剖映射

| Idx | SMPL 名 | 临床对应 | 用途 |
|---|---|---|---|
| 0 | pelvis | 骶骨中点 | 躯干根 |
| 3 | spine1 | L1-L3（腰胸交界） | 腰椎-下胸 |
| 6 | spine2 | T6-T8（中胸） | **胸椎驼背峰值区** |
| 9 | spine3 | T1-T3（上胸-肩胛上沿） | 上胸 |
| 12 | neck | C7-T1（颈胸交界） | 颈根 |
| 15 | head | 颅顶 | 头位 |
| 1,2 | L/R hip | 髋关节中心 | 髋屈伸/外展 |
| 4,5 | L/R knee | 膝关节中心 | 膝屈伸 |
| 7,8 | L/R ankle | 踝关节中心 | 踝屈伸 |
| 10,11 | L/R foot | 跖骨前部 | 足触地点 |
| 13,14 | L/R collar | 锁骨 | 肩带姿态 |
| 16,17 | L/R shoulder | 肱骨头 | 肩屈伸/外展 |
| 18,19 | L/R elbow | 肘 | 肘屈伸 |
| 20,21 | L/R wrist | 腕 | 腕屈伸 |

**注意**：手指、足趾、面部、肩胛骨细节、长骨内/外旋 都**未采样**。

## 二、强可表征体态（推荐主实验）

这些体态用 3+ 个空间分散的关节，几何定义清晰，约束饱满。

| 体态 | 关节 | 度量 | 当前是否注册 |
|---|---|---|---|
| **骨盆前倾 (APT)** | pelvis-hip_mid-spine1 | sagittal atan2 | ✅ |
| **骨盆后倾 (PPT)** | 同 APT，方向反 | 同上 (target=−15°) | ⚠ 可加 |
| **膝超伸 (knee hyperextension)** | hip-knee-ankle | 3-point angle | ✅ |
| **膝弯曲 (knee flexion)** | 同膝超伸，方向反 | 同上 | ⚠ 可加 |
| **躯干前倾 (trunk forward lean)** | pelvis-spine1-neck | sagittal angle | ⚠ 可加 |
| **躯干侧倾 (trunk lateral lean)** | pelvis-spine1, frontal | frontal angle | ⚠ 可加 |
| **骨盆侧倾 (pelvic obliquity)** | L_hip vs R_hip 高度差 | scalar y-diff | ⚠ 可加 |
| **膝内扣/外翻 (valgus/varus)** | hip-knee-ankle, frontal | frontal 3-point | ⚠ 可加 |
| **足距过宽/过窄 (wide/narrow stance)** | L_foot vs R_foot 距离 | x-distance | ⚠ 可加 |
| **肩部不对称 (shoulder asymmetry)** | L_shoulder vs R_shoulder y | scalar | ⚠ 可加 |

## 三、弱可表征体态（可做，但预期噪声大）

骨架采样点不足以精细刻画病灶位置，只能提供"有/无"的粗判断。

| 体态 | 关节 | 局限 | 当前 |
|---|---|---|---|
| **驼背 (thoracic kyphosis)** | spine1-spine3-neck（跳过 spine2） | T6-T8 病灶区采样不足；无法区分上胸 vs 中胸 vs 全胸驼 | ✅ 已注册 |
| **头前伸 (forward head posture)** | neck-head 两点 | 只能测头-颈 z 偏移；无法区分颈椎前移 vs 胸椎前倾合成 | ✅ 已注册 |
| **腰椎前凸 (lumbar lordosis)** | pelvis-spine1-spine2 | 与 APT 高度耦合，难以单独度量 | ❌ 未做 |
| **圆肩 (rounded shoulders)** | shoulder-collar 前移 | 两侧 collar 采样不太敏感于肩胛前倾 | ❌ 未做 |

**改进方向**：驼背可以加 spine2 进度量（用三点角 spine1-spine2-spine3 或 spine1-spine2-neck）。

## 四、不可表征体态（不要尝试）

骨架表示从根本上丢失这些信息：

- **局部椎段病变**：C5 不稳 / T4 楔形变 / L5 滑脱——MDM 无相邻椎间分辨率
- **长骨旋转**：股骨内旋（婴幼儿 in-toeing 的核心特征）——骨架是位置不是朝向
- **足部细节**：足内翻/外翻、扁平足、足弓塌陷——足只有 foot 一个采样点
- **手腕/手指畸形**：腕骨脱位、手指鹅颈样畸形——无指骨采样
- **面部姿态**：颅前突 / 下颌后缩——head 是单个 blob
- **呼吸 / 腹胀**：躯干是刚体段
- **眼球运动 / 视线**：无采样
- **微表情**：无采样

## 五、对本项目的实验建议

### 短期（已规划的跨体态验证）

| 实验 | 体态 | 预期 |
|---|---|---|
| 主对照 | 骨盆前倾 (APT) | 已有结果 |
| 强可表征任务 | **膝超伸** | V6/V2 都应稳定 |
| 弱可表征任务 | **驼背**（可选） | 预期 corr 低、CV 高；作为"任务上限"参照 |

### 中期（论文 ablation 扩展）

加 2-3 个 frontal-plane 体态以验证方法跨任务维度泛化：
- 骨盆侧倾（最简单：单标量）
- 膝内扣（验证 3-point angle 跨平面）
- 足距过宽（验证空间约束 vs 角度约束）

这些都只需在 `posture_guidance/registry.py` 加 `LossSpec(...)` + 在
`angle_ops.py` 加对应可微角度函数，约 30 行/体态。

### 不建议尝试的方向

- 给驼背换更复杂的多段曲线度量——表示本身就缺信息，再精细的度量也是无米之炊
- 颈椎细分体态（FHP 亚型）——同上
- 任何依赖足部细节、手指、面部的临床体征
