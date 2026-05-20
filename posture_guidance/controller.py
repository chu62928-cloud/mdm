"""
Top-level posture guidance controller.
对外接口：传入指令列表 + 当前 q + 时间步 t，返回总 loss。
"""
import math
import torch

from .registry import (
    POSTURE_REGISTRY,
    SCHEDULE_FUNCTIONS,
    LossSpec,
    compute_hinge_loss,
    compute_huber_loss,
    resolve_instruction,
)
from .phase_detector import PhaseDetector, PHASE_FUNCTIONS


class PostureGuidance:
    """
    阶段一的统一 loss 入口。

    使用：
        guidance = PostureGuidance(["骨盆前倾", "膝超伸"])
        # 在每个去噪步内：
        loss = guidance(q, t=current_step, T=total_steps)
        loss.backward()  # 梯度通过 FK 反传到 μₜ
    """

    def __init__(
        self,
        instructions: list[str],
        phase_detector: PhaseDetector = None,
        global_scale: float = 1.0,
        verbose: bool = False,
    ):
        self.detector = phase_detector or PhaseDetector()
        self.global_scale = global_scale
        self.verbose = verbose

        # 解析指令 → 展开 → 收集所有 spec
        self.specs: list[LossSpec] = []
        for inst in instructions:
            spec_names = resolve_instruction(inst)
            for sn in spec_names:
                self.specs.append(POSTURE_REGISTRY[sn])

        if self.verbose:
            print(f"[PostureGuidance] activated specs: "
                  f"{[s.name for s in self.specs]}")

    def __call__(
        self,
        q: torch.Tensor,
        t: int,
        T: int,
        spec_schedule_override: str = None,
    ) -> torch.Tensor:
        """
        Args:
            q: (B, N, J, 3) 或 (N, J, 3) 全局关节坐标，要求保持梯度
            t: 当前去噪步索引
            T: 总去噪步数
            spec_schedule_override: 若非 None，强制覆盖每个 spec.schedule（V6 用）
        Returns:
            total_loss: 标量 tensor
        """
        total_loss = torch.zeros((), device=q.device, dtype=q.dtype)

        for spec in self.specs:
            # 1. 时间调度：判断是否在当前 t 激活
            spec_schedule = spec_schedule_override if spec_schedule_override else spec.schedule
            schedule_w = SCHEDULE_FUNCTIONS[spec_schedule](t, T)
            if schedule_w == 0.0:
                continue

            # 2. 计算几何量
            angle = spec.angle_fn(q, **spec.angle_fn_kwargs)

            # 3. 计算相位 mask
            mask = PHASE_FUNCTIONS[spec.phase](self.detector, q)
            # 确保 mask shape 与 angle 一致
            if mask.dim() < angle.dim():
                # mask 可能是 (B, N) 而 angle 是 (B, N, ...)
                while mask.dim() < angle.dim():
                    mask = mask.unsqueeze(-1)
            elif mask.dim() > angle.dim():
                mask = mask.squeeze(-1)

            # 4. 单位转换
            if spec.unit == "deg":
                target_val = spec.target_deg * math.pi / 180.0
                tol_val    = spec.tolerance_deg * math.pi / 180.0
            else:  # meter or other
                target_val = spec.target_deg
                tol_val    = spec.tolerance_deg

            # 5. 计算 hinge loss
            loss_val = compute_hinge_loss(
                angle=angle,
                target=target_val,
                direction=spec.direction,
                tolerance=tol_val,
                mask=mask,
            )

            # 6. 加权累加
            weighted = self.global_scale * spec.base_weight * schedule_w * loss_val
            total_loss = total_loss + weighted

            if self.verbose:
                print(f"  [{spec.name}] angle_mean={angle.mean().item():.4f}, "
                      f"loss={loss_val.item():.6f}, weight={weighted.item():.6f}")

        return total_loss
    
    def set_variant(self, variant: str = "v1_mu_sgd",
                    variant_kwargs: dict = None,
                    diagnostic: bool = False):
        """配置 sampling 时用哪个 guidance variant。"""
        self.variant_config = {
            "variant": variant,
            "variant_kwargs": variant_kwargs or {},
            "diagnostic": diagnostic,
        }

    def compute_loss(
        self, q, t, T,
        temporal_smoothness_weight: float = 0.0,
        loss_form: str = "hinge",
        huber_delta: float = 0.05,
        huber_direction_override: str = None,
        spec_schedule_override: str = None,
    ):
        """
        兼容 diffusion 内部的调用接口。

        Args:
            q: (B, N, J, 3) 或 (N, J, 3) 关节坐标
            t, T: 当前 / 总去噪步
            temporal_smoothness_weight:
                λ_smooth ≥ 0. 在主 loss 上叠加
                  λ · mean(‖angle[k+1] − angle[k]‖²)
                逐 spec 累加（每个 spec 自己的 angle_fn 算一次）。
                设为 0（默认）则完全等价于 self(q, t, T)。
                推荐范围 0.01–0.05 (rad²)。
            loss_form:
                "hinge" — 默认，走 __call__ 用 compute_hinge_loss（V1-V5 行为不变）
                "huber" — 给 V6 闭环用，过目标时梯度反转能主动拉回
            huber_delta:
                Huber 转折点（与 angle 单位一致，rad 或 m）。默认 0.05 rad ≈ 2.9°。
            huber_direction_override:
                若指定（"equal" / "greater_than" / "less_than"），则强制覆盖每个
                spec.direction 使用此方向。V6 默认走 "equal"（双边推），
                因为闭环 PID 必须有过推拉回信号。设 None 则尊重 spec.direction。
            spec_schedule_override:
                若指定（"always" / "last_quarter" / ...），则强制覆盖每个
                spec.schedule。V6 推荐设 "always"，让控制器在全部去噪步都有梯度信号
                （c_t 时间步衰减自带早期软系数，不需要 spec schedule 二次 mask）。
                None 则尊重 spec.schedule（V1-V5 默认）。
        """
        if loss_form == "hinge":
            loss = self(q, t, T, spec_schedule_override=spec_schedule_override)
        elif loss_form == "huber":
            loss = self._compute_total_huber(
                q, t, T,
                huber_delta=huber_delta,
                direction_override=huber_direction_override,
                spec_schedule_override=spec_schedule_override,
            )
        else:
            raise ValueError(f"Unknown loss_form: {loss_form}")

        if temporal_smoothness_weight <= 0.0:
            return loss

        smooth = torch.zeros((), device=q.device, dtype=q.dtype)
        for spec in self.specs:
            spec_schedule = spec_schedule_override if spec_schedule_override else spec.schedule
            schedule_w = SCHEDULE_FUNCTIONS[spec_schedule](t, T)
            if schedule_w == 0.0:
                continue
            angle = spec.angle_fn(q, **spec.angle_fn_kwargs)
            # angle 可能是 (B, N) 或 (N,)；取最后一维做一阶差分
            if angle.dim() == 0:
                continue
            diff = angle[..., 1:] - angle[..., :-1]
            smooth = smooth + (diff ** 2).mean()
        return loss + temporal_smoothness_weight * smooth

    def _compute_total_huber(
        self, q, t, T,
        huber_delta: float = 0.05,
        direction_override: str = None,
        spec_schedule_override: str = None,
    ):
        """
        与 __call__ 同样的 spec/schedule/mask/单位转换逻辑，
        把内层 compute_hinge_loss 换成 compute_huber_loss。
        给 V6 闭环 PID 使用。

        spec_schedule_override: 若非 None，则强制每个 spec 走该 schedule（V6 推荐 "always"）。
        """
        total_loss = torch.zeros((), device=q.device, dtype=q.dtype)

        for spec in self.specs:
            spec_schedule = spec_schedule_override if spec_schedule_override else spec.schedule
            schedule_w = SCHEDULE_FUNCTIONS[spec_schedule](t, T)
            if schedule_w == 0.0:
                continue

            angle = spec.angle_fn(q, **spec.angle_fn_kwargs)

            mask = PHASE_FUNCTIONS[spec.phase](self.detector, q)
            if mask.dim() < angle.dim():
                while mask.dim() < angle.dim():
                    mask = mask.unsqueeze(-1)
            elif mask.dim() > angle.dim():
                mask = mask.squeeze(-1)

            if spec.unit == "deg":
                target_val = spec.target_deg * math.pi / 180.0
                tol_val    = spec.tolerance_deg * math.pi / 180.0
            else:
                target_val = spec.target_deg
                tol_val    = spec.tolerance_deg

            direction = direction_override if direction_override else spec.direction

            loss_val = compute_huber_loss(
                angle=angle,
                target=target_val,
                direction=direction,
                tolerance=tol_val,
                mask=mask,
                delta=huber_delta,
            )

            weighted = self.global_scale * spec.base_weight * schedule_w * loss_val
            total_loss = total_loss + weighted

            if self.verbose:
                print(f"  [{spec.name}] (huber dir={direction}) "
                      f"angle_mean={angle.mean().item():.4f}, "
                      f"loss={loss_val.item():.6f}, weight={weighted.item():.6f}")

        return total_loss