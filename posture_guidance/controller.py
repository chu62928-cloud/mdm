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
    ) -> torch.Tensor:
        """
        Args:
            q: (B, N, J, 3) 或 (N, J, 3) 全局关节坐标，要求保持梯度
            t: 当前去噪步索引
            T: 总去噪步数
        Returns:
            total_loss: 标量 tensor
        """
        total_loss = torch.zeros((), device=q.device, dtype=q.dtype)

        for spec in self.specs:
            # 1. 时间调度：判断是否在当前 t 激活
            schedule_w = SCHEDULE_FUNCTIONS[spec.schedule](t, T)
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

    def compute_loss(self, q, t, T):
        """兼容 diffusion 内部的调用接口"""
        return self(q, t, T)