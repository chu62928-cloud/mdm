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
    phase="always",            
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
    phase="always",
    schedule="last_quarter",
    base_weight=15.0,
))

# --- 膝超伸（双侧别名，用户可以直接说"膝超伸"） ---
# 在 controller 里展开成左右两个

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

POSTURE_ALIASES = {
    "膝超伸": ["膝超伸_左", "膝超伸_右"],
    "脚不离地": ["脚不离地_左", "脚不离地_右"],
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