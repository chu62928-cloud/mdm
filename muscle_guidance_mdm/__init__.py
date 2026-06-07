"""
muscle_guidance_mdm —— MDM 侧对接 motion2muscle 冻结代理 + posture loss 的适配层。

命名为 muscle_guidance_mdm（而非 muscle_guidance）以避免与 motion2muscle/ 内部的
muscle_guidance.py 模块重名（后者在 build_muscle_guidance 里被加入 sys.path 后 import）。

对外主要入口：
    build_muscle_guidance(...) -> motion2muscle.MuscleGuidance（已就绪，待 build_reference）
    load_frozen_proxy(...)     -> 冻结的 nn.Module 代理
"""
from .build import build_muscle_guidance, load_mint_cols
from .loader import load_frozen_proxy

__all__ = ["build_muscle_guidance", "load_mint_cols", "load_frozen_proxy"]
