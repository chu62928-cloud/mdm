"""
MDM diffusion sampling 集成 PostureGuidance。
fk_fn 通过 make_fk_fn 工厂函数构建，绑定 t2m_dataset，
在 generate.py 里创建一次，传入采样循环复用。
"""
import torch
from torch.optim import LBFGS
from data_loaders.humanml.scripts.motion_process import recover_from_ric

from .controller import PostureGuidance


def make_fk_fn(t2m_dataset, n_joints: int = 22):
    """
    工厂函数：把 t2m_dataset 绑定进闭包，返回可在 guidance 里调用的 fk_fn。

    在 generate.py 里调用一次：
        fk_fn = make_fk_fn(data.dataset.t2m_dataset, n_joints=22)
    然后把 fk_fn 传给 apply_posture_guidance。

    Args:
        t2m_dataset: data.dataset.t2m_dataset，提供 inv_transform
        n_joints:    22（hml_vec 263-d）或 21
    Returns:
        fk_fn: (mu: Tensor[B,263,1,T]) -> q: Tensor[B,T,J,3]，梯度链完整
    """
    # 提前把 mean/std 转成 tensor，避免每次 forward 都做类型转换
    # 这也保证 inv_transform 是纯 tensor 运算，梯度不会断
    mean = torch.tensor(t2m_dataset.mean, dtype=torch.float32)  # (263,)
    std  = torch.tensor(t2m_dataset.std,  dtype=torch.float32)  # (263,)

    def fk_fn(mu: torch.Tensor) -> torch.Tensor:
        """
        Args:
            mu: (B, 263, 1, T)  MDM 原生格式，requires_grad=True
        Returns:
            q:  (B, T, J, 3)    全局关节坐标，梯度链保留
        """
        device = mu.device

        # step1: (B, 263, 1, T) → (B, T, 1, 263)
        mu_perm = mu.permute(0, 3, 2, 1)   # (B, T, 1, 263)

        # step2: 可微的 inv_transform
        # x = x_norm * std + mean，纯 tensor 运算，梯度完整
        mean_d = mean.to(device)
        std_d  = std.to(device)
        mu_inv = mu_perm * std_d + mean_d   # (B, T, 1, 263)

        # step3: recover_from_ric → (B, T, 1, J, 3)
        q = recover_from_ric(mu_inv, n_joints)

        # step4: squeeze nfeats 维度 → (B, T, J, 3)
        q = q.squeeze(2)

        return q

    return fk_fn


def apply_posture_guidance(
    mu_t: torch.Tensor,
    guidance: PostureGuidance,
    t: int,
    T: int,
    n_lbfgs_steps: int = 5,
    lr: float = 0.05,
    fk_fn=None,
) -> torch.Tensor:
    """
    在单个去噪步内对 μₜ 施加 posture guidance。

    Args:
        mu_t:          (B, 263, 1, T) MDM posterior mean，不需要提前 requires_grad
        guidance:      PostureGuidance 实例
        t:             当前去噪步索引（从 T-1 倒数到 0）
        T:             总去噪步数
        n_lbfgs_steps: L-BFGS 内层迭代次数
        lr:            L-BFGS 学习率
        fk_fn:         由 make_fk_fn 构建的正向运动学函数
    Returns:
        mu_t_updated:  同 shape，已被 guidance 推向目标体态
    """
    if fk_fn is None:
        raise ValueError(
            "fk_fn 不能为 None。请在 generate.py 里用 make_fk_fn 构建后传入。"
        )

    # 从 MDM 计算图切断，新建 leaf tensor 作为优化变量
    # 这样 L-BFGS 的梯度只更新 mu_t_var，不会流回 MDM 权重
    mu_t_var = mu_t.detach().clone().contiguous().requires_grad_(True)

    from torch.optim import SGD
    optimizer = SGD([mu_t_var], lr=lr)

    mu_before = mu_t_var.detach().clone()

    for _ in range(n_lbfgs_steps):
        optimizer.zero_grad()
        q    = fk_fn(mu_t_var)
        loss = guidance(q, t=t, T=T)
        if not loss.requires_grad:
            break
        loss.backward()
        optimizer.step()

    delta = (mu_t_var.detach() - mu_before).norm().item()
    if True:
        print(f"  [guidance t={t:4d}] "
              f"loss: {loss.item():.4f}  "
              f"|Δμ|={delta:.4f}")

    return mu_t_var.detach()