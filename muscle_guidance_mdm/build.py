"""
muscle_guidance_mdm/build.py

把 motion2muscle 的 MuscleGuidance 一站式装配好（代理 + 列名 + 归一化握手 + 病态名），
返回一个待 build_reference() 的对象，供 MDM 采样循环使用。

依赖的交付物（都在仓库 motion2muscle/ 目录）：
    muscle_guidance.py     -> MuscleGuidance 类
    posture_loss*.py       -> 四分量 posture loss + POSTURE_PRIORS
    muscle_rollup.py       -> 功能肌群映射
    muscle_names.txt       -> 402 条有序肌束列名（= mint_cols）
运行时还需要（由肌肉队补交）：
    代理模型类 + net_best_*.pth 权重；可选 proxy Mean/Std（或 same_normalization）。
"""
from __future__ import annotations
import os
import sys
from typing import Callable, Optional

import numpy as np
import torch

from .loader import load_frozen_proxy

# 仓库内 motion2muscle 目录（默认资产位置）
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_DEFAULT_ASSETS = os.path.join(_REPO_ROOT, "motion2muscle")


def _ensure_on_path(assets_dir: str):
    """把 motion2muscle/ 加入 sys.path，以便 import MuscleGuidance / posture_loss 等。"""
    assets_dir = os.path.abspath(assets_dir)
    if assets_dir not in sys.path:
        sys.path.insert(0, assets_dir)


def load_mint_cols(assets_dir: str = _DEFAULT_ASSETS) -> list[str]:
    """读取 muscle_names.txt → 402 条有序列名（去空白、去空行）。"""
    path = os.path.join(assets_dir, "muscle_names.txt")
    if not os.path.exists(path):
        raise FileNotFoundError(f"找不到 muscle_names.txt：{path}")
    with open(path, "r") as f:
        cols = [ln.strip() for ln in f if ln.strip()]
    return cols


def _load_stats(path: Optional[str], device) -> Optional[torch.Tensor]:
    if not path:
        return None
    if not os.path.exists(path):
        raise FileNotFoundError(f"归一化统计文件不存在：{path}")
    arr = np.load(path)
    return torch.as_tensor(arr, dtype=torch.float32, device=device)


def build_muscle_guidance(
    ckpt_path: str,
    posture_name: str = "anterior_pelvic_tilt",
    *,
    assets_dir: str = _DEFAULT_ASSETS,
    model_builder: Optional[Callable[[], "torch.nn.Module"]] = None,
    mdm_mean: Optional[torch.Tensor] = None,
    mdm_std: Optional[torch.Tensor] = None,
    proxy_mean_path: Optional[str] = None,
    proxy_std_path: Optional[str] = None,
    same_normalization: bool = True,
    component_weights: Optional[dict] = None,
    device: torch.device | str = "cuda",
):
    """
    返回一个就绪的 motion2muscle.MuscleGuidance（还需调用 build_reference）。

    Args:
        ckpt_path          : 代理权重 net_best_*.pth。
        posture_name       : POSTURE_PRIORS 里的病态名（默认 anterior_pelvic_tilt）。
        assets_dir         : motion2muscle 资产目录。
        model_builder      : 构造空代理的回调（见 loader）。
        mdm_mean/mdm_std   : MDM 训练用的 Mean/Std（shape (263,)）。
                             同时也是 fk_fn 用的那套（来自 t2m_dataset）。
        proxy_mean_path/std: 代理训练用的 Mean.npy/Std.npy 路径（same_normalization=False 时必填）。
        same_normalization : True = 代理与 MDM 用同一套统计，跳过换算（默认先试这条，
                             见 HANDOFF §6.1）。
        component_weights  : 覆盖默认四分量权重（antag/chain/syn/stab）。
        device             : 设备。
    """
    _ensure_on_path(assets_dir)
    # 延迟 import：必须在 sys.path 注入之后
    from muscle_guidance import MuscleGuidance  # type: ignore

    device = torch.device(device)
    mint_cols = load_mint_cols(assets_dir)
    if len(mint_cols) != 402:
        print(f"[build_muscle_guidance] WARNING: muscle_names.txt 有 {len(mint_cols)} 行，"
              f"预期 402。请核对列名顺序（HANDOFF §6.3）。")

    proxy = load_frozen_proxy(ckpt_path, model_builder=model_builder, device=device)

    proxy_mean = proxy_std = None
    if not same_normalization:
        proxy_mean = _load_stats(proxy_mean_path, device)
        proxy_std = _load_stats(proxy_std_path, device)
        if proxy_mean is None or proxy_std is None:
            raise ValueError(
                "same_normalization=False 时必须提供 proxy_mean_path 和 proxy_std_path。"
            )
        if mdm_mean is None or mdm_std is None:
            raise ValueError(
                "same_normalization=False 时必须提供 mdm_mean / mdm_std（来自 t2m_dataset）。"
            )

    mg = MuscleGuidance(
        proxy=proxy,
        mint_cols=mint_cols,
        posture_name=posture_name,
        mdm_mean=mdm_mean,
        mdm_std=mdm_std,
        proxy_mean=proxy_mean,
        proxy_std=proxy_std,
        same_normalization=same_normalization,
        component_weights=component_weights,
        device=device,
    )
    print(f"[build_muscle_guidance] ready: posture={posture_name} "
          f"same_norm={same_normalization} cols={len(mint_cols)}")
    return mg
