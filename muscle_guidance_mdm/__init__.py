"""
muscle_guidance_mdm —— MDM 侧对接 motion2muscle 冻结代理 + posture loss 的适配层。

命名为 muscle_guidance_mdm（而非 muscle_guidance）以避免与 motion2muscle/ 内部的
muscle_guidance.py 模块重名（后者在 build_muscle_guidance 里被加入 sys.path 后 import）。

对外主要入口：
    build_muscle_guidance(...) -> motion2muscle.MuscleGuidance（已就绪，待 build_reference）
    load_frozen_proxy(...)     -> 冻结的 nn.Module 代理
    muscle_guidance_loss(...)  -> 给采样循环用的可微肌肉损失（默认 dense 引导版）
"""
from .build import build_muscle_guidance, load_mint_cols
from .loader import load_frozen_proxy
from .dense_loss import dense_posture_guidance_loss


def muscle_guidance_loss(mg, x_btc, kind: str = "guidance", margin: float = 0.3):
    """
    统一肌肉损失入口（供 CombinedGuidance / 评估调用）。

    Args:
        mg     : motion2muscle.MuscleGuidance 实例（已 build_reference）。
        x_btc  : (B,T,263) MDM 归一化空间的运动，带梯度。
        kind   :
            "guidance" —— dense 目标匹配损失（处处可微、起点梯度非零）。
                          **最小化 = 朝病态**（生成时用）。
            "clinical" —— 肌肉队原版 detection 损失（带阈值，>0=病态）。
                          仅用于评估/复刻 midterm Table 3；不要用于引导。
        margin : dense 版的方向性目标幅度。
    Returns:
        标量 tensor。
    """
    if kind == "clinical":
        return mg.loss(x_btc)
    if kind != "guidance":
        raise ValueError(f"kind must be 'guidance' or 'clinical', got {kind!r}")
    assert mg.reference_acts is not None, "先调用 mg.build_reference() / set_reference()"
    acts = mg._activations(x_btc)          # (B,T,402)，梯度连到 x_btc（经冻结代理）
    return dense_posture_guidance_loss(
        acts, mg.group_index, mg.posture_name, mg.reference_acts,
        component_weights=mg.component_weights, margin=margin)


__all__ = [
    "build_muscle_guidance", "load_mint_cols", "load_frozen_proxy",
    "dense_posture_guidance_loss", "muscle_guidance_loss",
]
