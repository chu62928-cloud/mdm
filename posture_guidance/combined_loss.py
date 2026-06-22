"""
posture_guidance/combined_loss.py

统一组合 loss：把"关节角约束（Module 1, PostureGuidance）"和"肌肉激活约束
（Module 2, motion2muscle 的 MuscleGuidance）"合成一个可微标量，供 MDM 采样循环
里的 guidance variant（统一走 v2_dps，保留 v6）调用。

符号约定（全部统一成"最小化"，因为现有 variant 都对 loss 做梯度下降 μ ← μ − s·∇L）：
    - 关节项 L_joint：hinge/huber，到位即 0 —— 方向是"最小化"。
    - 肌肉项 L_muscle：MuscleGuidance.loss，>0 = 越病态 —— 方向是"最大化"。
    => total = w_joint * L_joint  −  w_muscle * L_muscle

模式（mode）：
    "joint"  : 只用关节项（默认，与历史行为逐位等价）
    "muscle" : 只用肌肉项
    "both"   : 两者同时

输入约定：
    motion : MDM 原生布局 (B, 263, 1, T)，MDM 归一化空间，requires_grad。
        - 关节项内部用 fk_fn(motion) → (B, T, J, 3) 再算角度 loss。
        - 肌肉项内部 permute 成 (B, T, 263) 交给 MuscleGuidance（其内部再做
          MDM→proxy 的归一化握手 + 冻结代理 + 四分量 posture loss）。
"""
from __future__ import annotations
from typing import Optional


def motion_mdm_to_btc(motion):
    """(B, 263, 1, T)  ->  (B, T, 263)。与 make_fk_fn 的 permute 口径一致。"""
    return motion.permute(0, 3, 2, 1).squeeze(2)


class CombinedGuidance:
    """
    组合关节 + 肌肉两路约束。对外只暴露一个方法 `motion_loss`，
    variant 拿到 motion（x0_hat 或 mu_t）后调用它得到标量 loss，再求梯度。

    Args:
        posture     : PostureGuidance 实例（关节项）。可为 None（纯肌肉模式）。
        muscle      : motion2muscle 的 MuscleGuidance 实例（肌肉项）。可为 None。
        fk_fn       : make_fk_fn 构建的可微 FK 闭包 (B,263,1,T) -> (B,T,J,3)。
        mode        : "joint" | "muscle" | "both"。
        w_joint     : 关节项权重。
        w_muscle    : 肌肉项权重。
    """

    def __init__(self, posture=None, muscle=None, fk_fn=None,
                 mode: str = "joint", w_joint: float = 1.0, w_muscle: float = 1.0,
                 muscle_loss_kind: str = "guidance", muscle_margin: float = 0.3):
        mode = (mode or "joint").lower()
        if mode not in ("joint", "muscle", "both"):
            raise ValueError(f"mode must be joint|muscle|both, got {mode!r}")
        self.posture = posture
        self.muscle = muscle
        self.fk_fn = fk_fn
        self.mode = mode
        self.w_joint = float(w_joint)
        self.w_muscle = float(w_muscle)
        # "guidance" = dense 目标匹配（处处可微、最小化即朝病态）；"clinical" = 原检测损失（>0=病态）
        self.muscle_loss_kind = muscle_loss_kind
        self.muscle_margin = float(muscle_margin)
        # 诊断用：记录上一次各分量的标量值（detached）
        self.last_joint = 0.0
        self.last_muscle = 0.0

    # -- 各项是否启用 ------------------------------------------------------
    @property
    def joint_active(self) -> bool:
        return (self.mode in ("joint", "both")
                and self.posture is not None
                and len(getattr(self.posture, "specs", [])) > 0
                and self.fk_fn is not None)

    @property
    def muscle_active(self) -> bool:
        return self.mode in ("muscle", "both") and self.muscle is not None

    # -- 组合 loss ---------------------------------------------------------
    def motion_loss(
        self,
        motion,
        t: int,
        T: int,
        *,
        loss_form: str = "hinge",
        huber_delta: float = 0.05,
        huber_direction_override: Optional[str] = None,
        temporal_smoothness_weight: float = 0.0,
        spec_schedule_override: Optional[str] = None,
    ):
        """
        Args:
            motion : (B, 263, 1, T) MDM 归一化空间，带梯度。
            t, T   : 当前 / 总去噪步。
            其余    : 透传给 PostureGuidance.compute_loss（仅关节项使用）。
        Returns:
            标量 tensor（已乘各自权重并合并），grad 连接到 motion。
        """
        # anchor 保证返回值始终有 grad_fn，即便所有项当前都为 0
        total = motion.sum() * 0.0

        if self.joint_active:
            q = self.fk_fn(motion)
            L_joint = self.posture.compute_loss(
                q, t, T,
                temporal_smoothness_weight=temporal_smoothness_weight,
                loss_form=loss_form,
                huber_delta=huber_delta,
                huber_direction_override=huber_direction_override,
                spec_schedule_override=spec_schedule_override,
            )
            total = total + self.w_joint * L_joint
            self.last_joint = float(L_joint.detach())

        if self.muscle_active:
            x_btc = motion_mdm_to_btc(motion)
            if self.muscle_loss_kind == "clinical":
                # 原检测损失：>0=病态，最小化框架里取负号（梯度上升）。注意它在引导起点
                # 梯度恒为 0（死区），不建议用于生成，仅为兼容保留。
                L_muscle = self.muscle.loss(x_btc)
                total = total - self.w_muscle * L_muscle
            else:
                # dense 目标匹配损失：处处可微、起点梯度非零，**最小化即朝病态** → 直接相加
                from muscle_guidance_mdm import muscle_guidance_loss
                L_muscle = muscle_guidance_loss(
                    self.muscle, x_btc, kind="guidance", margin=self.muscle_margin)
                total = total + self.w_muscle * L_muscle
            self.last_muscle = float(L_muscle.detach())

        return total
