"""
muscle_guidance_mdm/loader.py

加载并冻结 motion2muscle 代理（motion -> muscle 的 transformer）。

⚠ 注意：代理的**模型类定义**（transformer 网络结构）来自 motion2muscle-main，
不在本仓库的 motion2muscle/ 目录里。要端到端运行 muscle / both 模式，需要：
  1. 把代理模型定义文件放进 motion2muscle/（或可被 import 的位置）；
  2. 提供 net_best_*.pth 权重（HANDOFF §6.4 推荐回归 loss 最优的那个）。

本模块通过一个 `model_builder` 回调解耦"如何构造空模型"这件事：
    proxy = load_frozen_proxy(ckpt_path, model_builder=lambda: MyTransformer(...))
若不传 model_builder，会尝试若干常见的约定式 import；都失败则抛出清晰错误，
指明需要补交模型类。
"""
from __future__ import annotations
import os
from typing import Callable, Optional

import torch
import torch.nn as nn


def _default_model_builder() -> nn.Module:
    """
    约定式构造：尝试从 motion2muscle 包里 import 代理模型类。
    支持几种常见命名；都失败时抛出可操作的错误。
    """
    candidates = [
        # (module, attr)
        ("models", "MotionToMuscleTransformer"),
        ("model", "MotionToMuscleTransformer"),
        ("transformer", "TransformerModel"),
        ("net", "Net"),
        ("motion2muscle.models", "MotionToMuscleTransformer"),
    ]
    errs = []
    for mod_name, attr in candidates:
        try:
            mod = __import__(mod_name, fromlist=[attr])
            cls = getattr(mod, attr)
            return cls()  # 注意：无参构造往往不对，多半需要外部传 model_builder
        except Exception as e:  # noqa: BLE001
            errs.append(f"  {mod_name}.{attr}: {type(e).__name__}: {e}")
    raise ImportError(
        "无法自动构造 motion2muscle 代理模型类。请通过 model_builder 显式传入构造函数，"
        "例如：\n"
        "    from my_proxy_def import MyTransformer\n"
        "    load_frozen_proxy(ckpt, model_builder=lambda: MyTransformer(...))\n"
        "尝试过的候选都失败：\n" + "\n".join(errs)
    )


def _extract_state_dict(ckpt):
    """从 checkpoint 里取出 state_dict（兼容直接存 state_dict 或包了一层 dict 的情况）。"""
    if isinstance(ckpt, dict):
        for key in ("state_dict", "model", "net", "model_state_dict"):
            if key in ckpt and isinstance(ckpt[key], dict):
                return ckpt[key]
    return ckpt


def load_frozen_proxy(
    ckpt_path: str,
    model_builder: Optional[Callable[[], nn.Module]] = None,
    device: torch.device | str = "cuda",
    strict: bool = False,
) -> nn.Module:
    """
    构造代理 → 载入权重 → eval() → 冻结参数（只冻参数，不包 no_grad，
    以便梯度能从 loss 经代理回传到运动 x0）。

    Args:
        ckpt_path     : net_best_*.pth 路径。
        model_builder : 返回"空"代理 nn.Module 的回调。强烈建议显式提供。
        device        : 设备。
        strict        : load_state_dict 的 strict（默认 False，容忍键不完全匹配）。
    Returns:
        冻结好的 proxy（requires_grad_=False，eval 模式）。
    """
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(
            f"代理权重不存在：{ckpt_path}\n"
            "请把 net_best_*.pth 放到该路径（见 HANDOFF.md §6.4）。"
        )

    model = model_builder() if model_builder is not None else _default_model_builder()

    ckpt = torch.load(ckpt_path, map_location="cpu")
    state = _extract_state_dict(ckpt)
    missing, unexpected = model.load_state_dict(state, strict=strict)
    if missing:
        print(f"[load_frozen_proxy] missing keys: {len(missing)} (strict={strict})")
    if unexpected:
        print(f"[load_frozen_proxy] unexpected keys: {len(unexpected)} (strict={strict})")

    model = model.to(device).eval()
    for p in model.parameters():
        p.requires_grad_(False)
    print(f"[load_frozen_proxy] loaded & frozen proxy from {ckpt_path} on {device}")
    return model
