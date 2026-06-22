"""
Posture loss registry — 阶段一的核心。

每个体态问题对应一个 LossSpec，描述：
- 怎么计算角度/几何量
- 目标值是多少
- 何时激活（时间调度 + 相位 mask）
- 是否有解剖连带约束
"""
import math
from dataclasses import dataclass, field
from typing import Callable, Optional

import torch
import torch.nn.functional as F

from . import angle_ops as ops
from .phase_detector import PhaseDetector, PHASE_FUNCTIONS


# ============================================================
# Loss spec 数据结构
# ============================================================

@dataclass
class LossSpec:
    """单个体态约束的完整规格"""
    name: str
    angle_fn: Callable                       # q → angle 的可微函数
    angle_fn_kwargs: dict = field(default_factory=dict)
    target_deg: float = 0.0                  # 目标角度（度）
    direction: str = "greater_than"          # greater_than | less_than | equal
    tolerance_deg: float = 2.0               # 死区
    phase: str = "always"                    # 相位条件名
    schedule: str = "always"                 # 时间调度名
    base_weight: float = 1.0                 # 该 loss 的基准权重
    unit: str = "deg"                        # deg | meter
    companion_specs: list = field(default_factory=list)  # 连带约束


# ============================================================
# 时间调度策略
# ============================================================

def schedule_always(t, T):       return 1.0
def schedule_second_half(t, T):  return 1.0 if t < T / 2  else 0.0
def schedule_last_quarter(t, T): return 1.0 if t < T / 4  else 0.0
def schedule_final(t, T):        return 1.0 if t < T / 10 else 0.0

def schedule_decay(t, T):
    """随 t 减小而权重增大的二次衰减，参考 proposal"""
    return (1.0 - t / T) ** 2

SCHEDULE_FUNCTIONS = {
    "always":        schedule_always,
    "second_half":   schedule_second_half,
    "last_quarter":  schedule_last_quarter,
    "final":         schedule_final,
    "decay":         schedule_decay,
}


# ============================================================
# 通用 hinge loss 计算器
# ============================================================

def compute_hinge_loss(
    angle: torch.Tensor,        # (..., N) 当前角度（弧度或米）
    target: float,               # 目标值
    direction: str,              # greater_than | less_than | equal
    tolerance: float,            # 死区
    mask: torch.Tensor,          # (..., N) 相位 mask
) -> torch.Tensor:
    """
    通用 hinge loss：到位就松手，没到位就施压。

    direction 语义：
    - greater_than: 当前值应当 > target，未达到时施压
    - less_than:    当前值应当 < target，未达到时施压
    - equal:        当前值应当 ≈ target，双侧施压
    """
    target_t = torch.tensor(target, device=angle.device, dtype=angle.dtype)
    tol_t    = torch.tensor(tolerance, device=angle.device, dtype=angle.dtype)

    if direction == "greater_than":
        # angle 应当 > target，loss = max(0, target - tolerance - angle)
        loss = F.relu(target_t - tol_t - angle)
    elif direction == "less_than":
        # angle 应当 < target，loss = max(0, angle - target - tolerance)
        loss = F.relu(angle - target_t - tol_t)
    elif direction == "equal":
        # 双侧 hinge
        loss_low  = F.relu(target_t - tol_t - angle)
        loss_high = F.relu(angle - target_t - tol_t)
        loss = loss_low + loss_high
    else:
        raise ValueError(f"Unknown direction: {direction}")

    # 应用相位 mask，并对帧维度求平均
    masked_loss = loss * mask
    # 防止 mask 全为 0 时除零
    mask_sum = mask.sum().clamp(min=1.0)
    return masked_loss.sum() / mask_sum


# ============================================================
# 对称 Huber loss — 供 V6 闭环控制器使用
# ============================================================

def compute_huber_loss(
    angle: torch.Tensor,        # (..., N) 当前角度（弧度或米）
    target: float,               # 目标值
    direction: str,              # greater_than | less_than | equal
    tolerance: float,            # 软死区（Huber delta 默认值的参考）
    mask: torch.Tensor,          # (..., N) 相位 mask
    delta: float = 0.05,         # Huber 转折点（与 angle 同单位）
) -> torch.Tensor:
    """
    Symmetric Huber loss — 给 V6 闭环 PID 控制器用的对称损失。

    与 compute_hinge_loss 的关键差异：
      hinge：到位后 L=0, grad=0 → controller 失明，无法感知过推
      huber：永远 differentiable，过推时 grad 反转 → controller 能拉回

    direction 语义：
      "greater_than": 仍按"应当 > target"，但在 angle > target 时给一个反向小推
                      L = huber(target − angle) when (target − angle) > 0
                      L = 0.1 · huber(angle − target) when (angle − target) > 0   ← 弱反推
      "less_than":    对称
      "equal":        双边均推（V6 推荐）
                      L = huber(angle − target)

    Huber form (cushion = delta)：
      |r| ≤ delta:  L = 0.5 · r² / delta
      |r| > delta:  L = |r| − 0.5 · delta
    梯度 ∈ [−1, +1]，远目标时不爆炸；近目标时平滑过零。

    Ref:
      - Huber 1964, "Robust Estimation of a Location Parameter", §3
      - Bansal et al., CVPR 2024 §4.2 (training-free guidance 需 bounded-grad)
    """
    target_t = torch.tensor(target, device=angle.device, dtype=angle.dtype)
    delta_t  = torch.tensor(delta,  device=angle.device, dtype=angle.dtype)

    def _huber(r: torch.Tensor) -> torch.Tensor:
        abs_r = r.abs()
        quad  = 0.5 * (r ** 2) / delta_t
        lin   = abs_r - 0.5 * delta_t
        return torch.where(abs_r <= delta_t, quad, lin)

    if direction == "equal":
        r = angle - target_t
        loss = _huber(r)
    elif direction == "greater_than":
        # 双侧但不对称：未达目标时全力推，过目标时弱反推（保留向上倾向）
        r_down = target_t - angle              # 未达目标 → r_down > 0
        r_up   = angle - target_t              # 过目标 → r_up > 0
        loss_main = _huber(torch.clamp(r_down, min=0.0))    # 主推力
        loss_back = 0.1 * _huber(torch.clamp(r_up, min=0.0))  # 弱回拉
        loss = loss_main + loss_back
    elif direction == "less_than":
        r_up   = angle - target_t
        r_down = target_t - angle
        loss_main = _huber(torch.clamp(r_up, min=0.0))
        loss_back = 0.1 * _huber(torch.clamp(r_down, min=0.0))
        loss = loss_main + loss_back
    else:
        raise ValueError(f"Unknown direction: {direction}")

    # 与 hinge 同样的 mask + 归一化
    masked_loss = loss * mask
    mask_sum = mask.sum().clamp(min=1.0)
    return masked_loss.sum() / mask_sum


# ============================================================
# 体态注册表 ★ 阶段一手动注册的体态都在这里 ★
# ============================================================

POSTURE_REGISTRY: dict[str, LossSpec] = {}


def register_posture(spec: LossSpec):
    """注册一个体态 spec"""
    POSTURE_REGISTRY[spec.name] = spec


# --- 骨盆前倾 ---
register_posture(LossSpec(
    name="骨盆前倾",
    angle_fn=ops.pelvis_tilt_angle,           # 注意要用修复后的版本
    target_deg=20.0,                          # 病态前倾 20°
    direction="greater_than",
    tolerance_deg=2.0,
    phase="always",
    schedule="last_quarter",                         # ← 改：用衰减调度，前期不施压
    base_weight=20.0,                          # ← 改：从 1.0 降到 0.3
    companion_specs=[],
))

register_posture(LossSpec(
    name="骨盆前倾_深蹲",
    angle_fn=ops.pelvis_tilt_angle,
    target_deg=40.0,          # ← 深蹲时正常 25°，目标 40° 才能凸显病态
    direction="greater_than",
    tolerance_deg=3.0,
    phase="always",
    schedule="last_quarter",   # 深蹲动作变化大，太晚施压会破坏动作连贯
    base_weight=15.0,
    unit="deg",
))

register_posture(LossSpec(
    name="脚不离地_左",
    angle_fn=ops.foot_floor_distance,
    angle_fn_kwargs={"side": "left"},
    target_deg=0.05,                          # 支撑相足部离地 < 5cm
    direction="less_than",
    tolerance_deg=0.02,
    phase="stance_left",                      # 只在左足支撑相约束
    schedule="always",
    base_weight=0.5,
    unit="meter",                             # 单位是米！不是度
))
 
register_posture(LossSpec(
    name="脚不离地_右",
    angle_fn=ops.foot_floor_distance,
    angle_fn_kwargs={"side": "right"},
    target_deg=0.05,
    direction="less_than",
    tolerance_deg=0.02,
    phase="stance_right",
    schedule="always",
    base_weight=0.5,
    unit="meter",
))

# --- 膝超伸（左） ---
register_posture(LossSpec(
    name="膝超伸_左",
    angle_fn=ops.signed_knee_angle,
    angle_fn_kwargs={"side": "left"},
    target_deg=190.0,               # 超伸 5°
    direction="greater_than",
    tolerance_deg=1.5,
    phase="stance_left",            
    schedule="last_quarter",
    base_weight=15.0,
    unit="deg",
))

# --- 膝超伸（右） ---
register_posture(LossSpec(
    name="膝超伸_右",
    angle_fn=ops.signed_knee_angle,
    angle_fn_kwargs={"side": "right"},
    target_deg=190.0,
    direction="greater_than",
    tolerance_deg=1.5,
    phase="stance_right",
    schedule="last_quarter",
    base_weight=15.0,
))


# --- 膝超伸_dist（左）--- 有符号矢状面距离，无acos梯度饱和 ---
register_posture(LossSpec(
    name="膝超伸_dist_左",
    angle_fn=ops.signed_knee_distance_sagittal,
    angle_fn_kwargs={"side": "left"},
    target_deg=-0.05,               # 约等效超伸 6°，负值=膝在后
    direction="less_than",         # dist < target
    tolerance_deg=0.01,             # 1cm 容许带
    phase="always",
    schedule="last_quarter",
    base_weight=30.0,               # 距离量纲小，需更大 weight
    unit="meter",                  # ★ 距离单位
))

# --- 膝超伸_dist（右）---
register_posture(LossSpec(
    name="膝超伸_dist_右",
    angle_fn=ops.signed_knee_distance_sagittal,
    angle_fn_kwargs={"side": "right"},
    target_deg=-0.05,
    direction="less_than",
    tolerance_deg=0.01,
    phase="always",
    schedule="last_quarter",
    base_weight=30.0,
    unit="meter",
))

# --- 膝弯曲（左） --- 分布内，慢走常见，屈曲目标 125°
register_posture(LossSpec(
    name="膝弯曲_左",
    angle_fn=ops.signed_knee_angle,
    angle_fn_kwargs={"side": "left"},
    target_deg=125.0,
    direction="less_than",
    tolerance_deg=2.0,
    phase="stance_left",
    schedule="last_quarter",
    base_weight=15.0,
    unit="deg",
))

# --- 膝弯曲（右） ---
register_posture(LossSpec(
    name="膝弯曲_右",
    angle_fn=ops.signed_knee_angle,
    angle_fn_kwargs={"side": "right"},
    target_deg=125.0,
    direction="less_than",
    tolerance_deg=2.0,
    phase="stance_right",
    schedule="last_quarter",
    base_weight=15.0,
    unit="deg",
))

# --- 膝超伸（双侧别名，用户可以直接说"膝超伸"） ---
# 在 controller 里展开成左右两个

# --- 膝弯曲_A：target=145°，station 相位（轻度弯膝步态，分布内） ---
# 正常站立相膝角 ~160-170°，145° 需要 Δ≈-15~25°，不触发相位反转
register_posture(LossSpec(
    name="膝弯曲_A_左",
    angle_fn=ops.signed_knee_angle,
    angle_fn_kwargs={"side": "left"},
    target_deg=145.0,
    direction="less_than",
    tolerance_deg=2.0,
    phase="stance_left",
    schedule="last_quarter",
    base_weight=15.0,
    unit="deg",
))

register_posture(LossSpec(
    name="膝弯曲_A_右",
    angle_fn=ops.signed_knee_angle,
    angle_fn_kwargs={"side": "right"},
    target_deg=145.0,
    direction="less_than",
    tolerance_deg=2.0,
    phase="stance_right",
    schedule="last_quarter",
    base_weight=15.0,
    unit="deg",
))

# --- 膝弯曲_B：target=125°，phase=always（去掉相位门控） ---
# 原 spec 的相位限制 stance_left/right 与 125° 形成相位-角度矛盾；
# 改为 always 后 loss 在摆动相自然为 0（角度已 <125°），只在站立相有效
register_posture(LossSpec(
    name="膝弯曲_B_左",
    angle_fn=ops.signed_knee_angle,
    angle_fn_kwargs={"side": "left"},
    target_deg=125.0,
    direction="less_than",
    tolerance_deg=2.0,
    phase="always",
    schedule="last_quarter",
    base_weight=15.0,
    unit="deg",
))

register_posture(LossSpec(
    name="膝弯曲_B_右",
    angle_fn=ops.signed_knee_angle,
    angle_fn_kwargs={"side": "right"},
    target_deg=125.0,
    direction="less_than",
    tolerance_deg=2.0,
    phase="always",
    schedule="last_quarter",
    base_weight=15.0,
    unit="deg",
))

# --- 躯干前倾（全身性前倾，髋中点→双肩中点矢状面夹角） ---
# 正常快走约 5-10°，病理性前倾（Parkinson's、老年屈曲步态）约 15-30°
# phase=always：全步态周期躯干均前倾，无相位冲突（不同于膝弯曲在站立相的相位-角度矛盾）
register_posture(LossSpec(
    name="躯干前倾",
    angle_fn=ops.trunk_forward_lean,
    target_deg=15.0,
    direction="greater_than",
    tolerance_deg=2.0,
    phase="always",
    schedule="last_quarter",
    base_weight=20.0,
    unit="deg",
))

# --- 骨盆侧倾（Trendelenburg 步态，右髋高） ---
# 正常步态均值≈0°（双侧对称振荡）；病态 Trendelenburg 均值 > 3-5°（系统性单侧偏移）
# 目标 5°：mild Trendelenburg，分布边界但仍在 MDM 训练集范围内
# phase=always：全步态周期均需系统性偏移（不只在某一相）
register_posture(LossSpec(
    name="骨盆侧倾",
    angle_fn=ops.pelvis_lateral_tilt,
    target_deg=5.0,
    direction="greater_than",
    tolerance_deg=1.0,
    phase="always",
    schedule="last_quarter",
    base_weight=20.0,
    unit="deg",
))

# --- 驼背 ---
register_posture(LossSpec(
    name="驼背",
    angle_fn=ops.spine_posterior_bulge,  
    target_deg=0.10,     # 目标 10cm 偏离（baseline 约 3-5cm，驼背约 10-15cm）
    direction="greater_than",
    tolerance_deg=0.01,
    phase="always",
    schedule="last_quarter",
    base_weight=30.0,
    unit="meter",
))

# --- 头前伸 ---
register_posture(LossSpec(
    name="头前伸",
    angle_fn=ops.head_forward_offset,
    target_deg=0.04,                # 4 cm 前伸
    direction="greater_than",
    tolerance_deg=0.005,
    phase="always",
    schedule="always",
    base_weight=1.0,
    unit="meter",                   # 注意单位是米
))


# ============================================================
# 双侧别名展开
# ============================================================

# ---- 骨盆后倾 (PPT) ----
register_posture(LossSpec(
    name="骨盆后倾",
    angle_fn=ops.pelvis_tilt_angle,
    target_deg=-20.0,
    direction="less_than",
    tolerance_deg=2.0,
    phase="always",
    schedule="last_quarter",
    base_weight=20.0,
    companion_specs=[],
))

POSTURE_ALIASES = {
    # English -> Chinese (joint+muscle unified): set POSTURE=english_name for both modules
    "anterior_pelvic_tilt":  ["骨盆前倾"],
    "posterior_pelvic_tilt": ["骨盆后倾"],
    "forward_head_posture":  ["头前倾"],
    "trendelenburg":         ["特伦德伦堡"],
    # Chinese expansion aliases (knee/leg)
    "膝盖超伸_dist": ["膝盖超伸_dist_左", "膝盖超伸_dist_右"],
    "膝盖超伸":   ["膝盖超伸_左",   "膝盖超伸_右"],
    "膝盖弯曲":   ["膝盖弯曲_左",   "膝盖弯曲_右"],
    "膝盖弯曲_A": ["膝盖弯曲_A_左", "膝盖弯曲_A_右"],
    "膝盖弯曲_B": ["膝盖弯曲_B_左", "膝盖弯曲_B_右"],
    "骨盆侧倾": ["骨盆侧倾_左", "骨盆侧倾_右"],
}


def resolve_instruction(instruction: str) -> list[str]:
    """把高层指令展开成具体的 spec 名字列表"""
    if instruction in POSTURE_ALIASES:
        return POSTURE_ALIASES[instruction]
    if instruction in POSTURE_REGISTRY:
        return [instruction]
    raise KeyError(
        f"Unknown posture: '{instruction}'. "
        f"Available: {list(POSTURE_REGISTRY.keys())} "
        f"+ aliases {list(POSTURE_ALIASES.keys())}"
    )