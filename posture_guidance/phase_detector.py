"""
Gait phase detection from joint coordinates.
不依赖任何外部传感器或学习模型，纯几何判断。
"""
import torch
import torch.nn.functional as F
from .joint_indices import get_joint_idx


class PhaseDetector:
    """
    从关节坐标序列检测步态相位。

    所有 mask 都是可微的（用 sigmoid 软化），这样梯度可以
    通过 mask 反传，避免硬阈值在边界帧产生梯度突变。
    """

    def __init__(
        self,
        height_thresh: float = 0.05,   # 足部离地高度阈值（米）
        vel_thresh: float = 0.10,      # 足部速度阈值（米/秒）
        fps: int = 20,                 # HumanML3D 是 20fps
        soft_k: float = 50.0,          # sigmoid 锐度，越大越接近硬阈值
    ):
        self.h_thresh = height_thresh
        self.v_thresh = vel_thresh
        self.dt = 1.0 / fps
        self.soft_k = soft_k

        self.foot_idx = {
            "left":  get_joint_idx("left_foot"),
            "right": get_joint_idx("right_foot"),
        }

    def _soft_below(self, value, threshold):
        """
        软化的 (value < threshold)，返回 [0,1] 之间的值。
        value < threshold 时趋近 1，反之趋近 0。
        """
        return torch.sigmoid(self.soft_k * (threshold - value))

    def get_stance_mask(self, q: torch.Tensor) -> torch.Tensor:
        """
        Args:
            q: (B, N, J, 3) 全局关节坐标，或 (N, J, 3)
        Returns:
            stance_mask: 和 q 同 batch shape，最后一维是 (left, right)
                        每个值在 [0,1]，1 表示该足处于支撑相
        """
        # 统一 batch 处理
        squeeze_batch = q.dim() == 3
        if squeeze_batch:
            q = q.unsqueeze(0)

        B, N, J, _ = q.shape
        stance = []

        for side in ["left", "right"]:
            foot_y = q[:, :, self.foot_idx[side], 1]   # (B, N) 高度

            # 高度条件：足部贴近地面
            height_mask = self._soft_below(foot_y, self.h_thresh)

            # 速度条件：足部相对静止
            # 用中心差分，端点 padding
            foot_y_padded = F.pad(
                foot_y.unsqueeze(1), (1, 1), mode="replicate"
            ).squeeze(1)
            foot_vel = (foot_y_padded[:, 2:] - foot_y_padded[:, :-2]) / (2 * self.dt)
            vel_mask = self._soft_below(foot_vel.abs(), self.v_thresh)

            # 两个条件软 AND（乘积）
            side_mask = height_mask * vel_mask
            stance.append(side_mask)

        stance_mask = torch.stack(stance, dim=-1)  # (B, N, 2)

        if squeeze_batch:
            stance_mask = stance_mask.squeeze(0)
        return stance_mask

    def get_swing_mask(self, q):
        return 1.0 - self.get_stance_mask(q)

    def get_double_support_mask(self, q):
        """双足同时接触相"""
        stance = self.get_stance_mask(q)
        return stance[..., 0] * stance[..., 1]

    def get_always_mask(self, q):
        """全程激活，返回全 1 mask"""
        return torch.ones(q.shape[:-2], device=q.device, dtype=q.dtype)


PHASE_FUNCTIONS = {
    "always":         lambda d, q: d.get_always_mask(q),
    "stance_left":    lambda d, q: d.get_stance_mask(q)[..., 0],
    "stance_right":   lambda d, q: d.get_stance_mask(q)[..., 1],
    "stance_either":  lambda d, q: d.get_stance_mask(q).max(dim=-1).values,
    "stance_both":    lambda d, q: d.get_double_support_mask(q),
    "swing_left":     lambda d, q: d.get_swing_mask(q)[..., 0],
    "swing_right":    lambda d, q: d.get_swing_mask(q)[..., 1],
}