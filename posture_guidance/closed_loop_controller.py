"""
Closed-loop PID step-size controller for posture guidance.

每个 (B, sample) 独立维护 PID 状态。在 V6 闭环采样里每个去噪步调用 update()，
返回当前应该使用的步长 s_t（已经做了时间步衰减、anti-windup、扩散维 EMA 平滑）。

References:
  - Åström & Hägglund, "Advanced PID Control", ISA 2006 (anti-windup)
  - Karras et al., NeurIPS 2022 (PID 控制 ODE solver 步长)
  - Efron, JASA 2011 (Tweedie's formula — 解释为什么早期 x0_hat 不可信)
"""
from dataclasses import dataclass, field
from typing import Optional

import torch


@dataclass
class PIDState:
    """单个 batch 的 PID 状态，跨 diffusion timestep 累积。"""
    integral: Optional[torch.Tensor] = None      # (B,)
    prev_err: Optional[torch.Tensor] = None      # (B,)
    s_prev:   Optional[torch.Tensor] = None      # (B,) — for EMA
    saturated: bool = False                       # last step saturated to s_max?
    step_count: int = 0
    history: list = field(default_factory=list)   # debug only


class ClosedLoopController:
    """
    PID step-size controller for DPS-style posture guidance.

    Usage:
        ctrl = ClosedLoopController(Kp=30, Ki=1, Kd=5, ...)
        # in each diffusion step:
        s_t = ctrl.update(err=target - a_now, sigma_t=sigma_t, t_int=t_int, T=T)
        mu_t_new = mu_t - s_t * grad / grad_norm

    All scalar args become per-batch tensors (B,) automatically.
    """

    def __init__(
        self,
        Kp: float = 30.0,
        Ki: float = 1.0,
        Kd: float = 5.0,
        s_min: float = 5.0,
        s_max: float = 120.0,
        I_max: float = 20.0,
        beta_ema: float = 0.8,
        i_start_frac: float = 0.5,
        sigma_min: float = 0.01,
        sigma_max: float = 1.0,
        use_time_decay: bool = True,
    ):
        # gains
        self.Kp = Kp
        self.Ki = Ki
        self.Kd = Kd
        # saturation
        self.s_min = s_min
        self.s_max = s_max
        self.I_max = I_max
        # smoothing
        self.beta_ema = beta_ema
        # anti-windup delay
        self.i_start_frac = i_start_frac
        # sigma normalization range (for time-step-decay gain)
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max
        # c_t time-decay: if False, c_t=1.0 always (ablation use)
        self.use_time_decay = use_time_decay

        self.state = PIDState()

    def reset(self):
        self.state = PIDState()

    def update(
        self,
        err: torch.Tensor,
        sigma_t: float,
        t_int: int,
        T: int,
    ) -> torch.Tensor:
        """
        Args:
            err:     (B,) target − current_angle，已是同一单位（rad 或 deg）
            sigma_t: scalar，当前 timestep 的 posterior std（用于时间步衰减增益）
            t_int:   当前去噪步索引（T-1 倒数到 0）
            T:       总去噪步数
        Returns:
            s_t: (B,) 当前步的步长
        """
        device = err.device
        dtype = err.dtype

        # 跨采样状态隔离：t_int = T-1 是新一次采样的第一步，自动 reset
        if t_int >= T - 1:
            self.reset()

        # ---- 对策 A1: 时间步衰减增益 ----------------------------------------
        # σ_t 大 → c_t 小 → 弱控制；σ_t 小 → c_t 大 → 强校准
        # 早期 x0_hat 估计偏差大（Efron 2011），不该让闭环吃这些噪声
        if self.use_time_decay:
            sigma_norm_range = max(self.sigma_max - self.sigma_min, 1e-6)
            c_t = max(
                0.0,
                min(1.0, (self.sigma_max - float(sigma_t)) / sigma_norm_range),
            )
        else:
            c_t = 1.0

        # ---- 对策 A2: anti-windup ------------------------------------------
        # (1) I 项启动延迟：t_int > T·i_start_frac 时不累积
        # (2) I 项 clamp 到 ±I_max
        # (3) conditional integration：上一步已饱和到 s_max 且同方向时不累积
        # 注意：t_int 从 T-1 倒数到 0，"后期"=t_int 小
        i_active = (t_int < T * self.i_start_frac)

        if self.state.integral is None:
            self.state.integral = torch.zeros_like(err)
        if self.state.prev_err is None:
            self.state.prev_err = torch.zeros_like(err)
            d_err = torch.zeros_like(err)
        else:
            d_err = err - self.state.prev_err

        if i_active and not self.state.saturated:
            self.state.integral = (self.state.integral + err).clamp(
                -self.I_max, self.I_max
            )

        # ---- PID 主公式 -----------------------------------------------------
        s_raw = c_t * (
            self.Kp * err
            + self.Ki * self.state.integral
            + self.Kd * d_err
        )
        s_t_raw = s_raw.clamp(self.s_min, self.s_max)

        # 记录饱和状态（同方向触发后续 conditional integration）
        self.state.saturated = bool((s_t_raw >= self.s_max - 1e-6).any().item())

        # ---- 对策 B1: 扩散维 EMA 平滑 --------------------------------------
        if self.state.s_prev is None:
            s_t = s_t_raw
        else:
            s_t = self.beta_ema * self.state.s_prev + (1.0 - self.beta_ema) * s_t_raw

        # 更新状态
        self.state.s_prev = s_t.detach()
        self.state.prev_err = err.detach()
        self.state.step_count += 1
        self.state.history.append({
            "t": t_int, "c_t": c_t,
            "err": float(err.mean().item()),
            "integral": float(self.state.integral.mean().item()),
            "s_raw": float(s_t_raw.mean().item()),
            "s_t": float(s_t.mean().item()),
        })

        return s_t


def orthogonal_project(
    grad: torch.Tensor,
    x0_hat: torch.Tensor,
    alpha: float = 1.0,
    eps: float = 1e-8,
) -> torch.Tensor:
    """
    Manifold-preserving orthogonal projection (MPGD, He et al. ICLR 2024 §3.2).

    把 grad 中沿 x0_hat 方向的分量减去 α 倍，保留正交于 x0 manifold 的方向。
    α=1.0 是完整投影（严格去 drift），α=0 等价不投影。

    Args:
        grad:   (B, ...) gradient w.r.t. x_t
        x0_hat: (B, ...) MDM 预测的 clean sample，同 shape
        alpha:  [0, 1] 投影强度
    Returns:
        grad_proj: (B, ...) 同 shape
    """
    # per-batch flatten 计算投影系数
    g_flat = grad.flatten(1)
    x_flat = x0_hat.flatten(1)
    num = (g_flat * x_flat).sum(dim=1, keepdim=True)               # (B, 1)
    den = (x_flat * x_flat).sum(dim=1, keepdim=True).clamp_min(eps)
    coef = num / den                                                # (B, 1)
    # 还原回原 shape 做减法
    coef_shape = [coef.shape[0]] + [1] * (grad.dim() - 1)
    coef_b = coef.view(*coef_shape)
    return grad - alpha * coef_b * x0_hat
