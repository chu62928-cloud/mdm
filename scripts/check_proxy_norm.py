#!/usr/bin/env python3
"""
scripts/check_proxy_norm.py

诊断 proxy 的"输出退化 + Jacobian 极小"问题，定位是 (1) 多余的最终 sigmoid，
还是 (2) 归一化握手，还是 (3) proxy 本身迟钝。

它对比 final_activation ∈ {none, sigmoid}（可选再对比原始模型类）下：
  - 激活分布 min/mean/max/std（预期正确值落在 ~0.03–0.15，而非 0.50–0.53）
  - baseline→guided 的平均激活变化 |Δ|（用 comparison.npy 时）
  - 输入→输出 Jacobian 范数 ‖∂ mean(act) / ∂ motion‖（越大越灵敏）

用法：
    python scripts/check_proxy_norm.py --muscle_ckpt <net_best_loss.pth> \
        [--comparison output/.../comparison.npy] [--device cuda] \
        [--orig_module /root/autodl-tmp/motion2muscle/models/m2m_transformer.py \
         --orig_class MotionToMuscleModel]
"""
import argparse
import importlib.util
import os
import sys

import numpy as np
import torch

_THIS = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_THIS)
_ASSETS = os.path.join(_ROOT, "motion2muscle")
for p in (_ROOT, _ASSETS):
    if p not in sys.path:
        sys.path.insert(0, p)

from muscle_guidance_mdm.loader import load_frozen_proxy             # noqa: E402


def _import_class_from_file(path, cls_name):
    spec = importlib.util.spec_from_file_location("orig_proxy_mod", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return getattr(mod, cls_name)


def stats(t):
    t = t.detach().float()
    return (f"min={t.min():.4f} mean={t.mean():.4f} max={t.max():.4f} "
            f"std={t.std():.4f}")


def jacobian_norm(model, x):
    """‖∂ mean(activation) / ∂ x‖ —— 越大越灵敏。"""
    x = x.detach().clone().requires_grad_(True)
    a = model(x)
    g = torch.autograd.grad(a.mean(), x)[0]
    return float(g.norm()), a.detach()


def run_config(label, model, x_base, x_guided):
    jn, a_base = jacobian_norm(model, x_base)
    line = f"[{label:30s}] act({stats(a_base)})  Jacobian‖∂a/∂x‖={jn:.5f}"
    if x_guided is not None:
        with torch.no_grad():
            a_g = model(x_guided)
        d = (a_g - a_base).abs().mean().item()
        line += f"  mean|Δact base→guided|={d:.5e}"
    print(line)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--muscle_ckpt", required=True)
    ap.add_argument("--comparison", default=None, help="comparison.npy（取真实 baseline/guided motion）")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--orig_module", default=None, help="原始 m2m_transformer.py 路径（可选对比）")
    ap.add_argument("--orig_class", default="MotionToMuscleModel")
    ap.add_argument("--T", type=int, default=28)
    args = ap.parse_args()

    device = torch.device(args.device if (args.device == "cpu" or torch.cuda.is_available()) else "cpu")

    # ---- 取输入 motion（MDM 归一化空间，(B,T,263)）----
    if args.comparison and os.path.exists(args.comparison):
        data = np.load(args.comparison, allow_pickle=True).item()
        x_base = torch.from_numpy(data["motion_hml_tj"]).float().to(device)
        x_guided = torch.from_numpy(data["motion_hml_tj_guided"]).float().to(device)
        print(f"输入来自 {args.comparison}  shape={tuple(x_base.shape)}")
    else:
        x_base = torch.randn(1, args.T, 263, device=device)
        x_guided = None
        print(f"输入为随机张量 (1,{args.T},263)（未提供 --comparison）")

    from models import MotionToMuscleModel  # 重建类（motion2muscle/models.py）

    print("\n== 重建类 motion2muscle/models.py ==")
    for fa in ("none", "sigmoid"):
        model = load_frozen_proxy(
            args.muscle_ckpt,
            model_builder=lambda fa=fa: MotionToMuscleModel(final_activation=fa),
            device=device)
        run_config(f"reconstructed final={fa}", model, x_base, x_guided)

    if args.orig_module and os.path.exists(args.orig_module):
        print("\n== 原始类（对照） ==")
        try:
            OrigCls = _import_class_from_file(args.orig_module, args.orig_class)
            model = load_frozen_proxy(
                args.muscle_ckpt, model_builder=lambda: OrigCls(), device=device)
            run_config(f"original {args.orig_class}", model, x_base, x_guided)
        except Exception as e:  # noqa: BLE001
            print(f"  加载原始类失败：{type(e).__name__}: {e}")

    print("\n判读：")
    print("  - 若 final=none 的激活落在 ~0.03–0.15、Jacobian 明显大于 final=sigmoid，")
    print("    则证实'多余 sigmoid'是瓶颈 → 用 final_activation=none（已设为默认）。")
    print("  - 若两者都 ~0.5 / Jacobian 都极小，再查归一化（proxy 训练 Mean/Std vs MDM）。")
    print("  - 若与原始类输出一致且仍迟钝，才是 proxy 固有灵敏度问题（midterm §5.2）。")


if __name__ == "__main__":
    main()
