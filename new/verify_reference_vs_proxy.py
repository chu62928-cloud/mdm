#!/usr/bin/env python3
"""
new/verify_reference_vs_proxy.py

一锤定音：把"方向反转"的两个候选病因分离开——
  (H_A) proxy 逆映射：proxy 把 APT 几何直接映射成 PPT 激活形态；
  (H_ref) reference 污染：baseline 轻微后倾 → reference 偏，使"减 baseline 后的 Δ"假反向。

为什么要这个脚本
----------------
之前所有判读都用「guided - baseline 的 Δ」，而 baseline(=reference) 本身可能被污染，
Δ 的符号不可信。本脚本只看**绝对激活**（完全不减 baseline、不经过 reference / dense loss），
直接问：APT 几何运动喂进冻结 proxy，输出的肌肉形态到底是 APT 还是 PPT？

判读（关键肌群：髋屈肌 vs 髋伸肌）
  临床 APT 形态 = 髋屈肌(iliopsoas, rectus_femoris) 高、髋伸肌(gluteus_maximus) 低。
  - 若 APT几何运动的【绝对】激活就是 PPT 形态（髋屈低 / 髋伸高）
        → 锁定 H_A（proxy 逆映射）。reference 改不改都救不了方向。
  - 若 APT几何运动的【绝对】激活其实接近 APT 形态（髋屈高 / 髋伸低），
    只是因为后倾 baseline 太高/太低、减出反向 Δ
        → H_ref（reference 污染）才是主因，换中性 reference 能修复。

用法
    python new/verify_reference_vs_proxy.py <comparison.npy 或其目录> \
        --muscle_ckpt motion2muscle/checkpoints/.../net_best_loss.pth \
        [--device cuda]

    # 同时传多个目录（如 joint / muscle / both 三个模式）逐个对比：
    python new/verify_reference_vs_proxy.py ./output/apt_integrated_v2 --muscle_ckpt ...
    （若传的是 output_root，会自动遍历其下每个含 comparison.npy 的子目录）
"""
import argparse
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

from muscle_rollup import get_indices  # noqa: E402

# 关键肌群（侧无关名，后面展开 _R/_L 平均）。
# expect = 'high' / 'low' = 临床 APT 下该肌群的绝对形态期望。
KEY_GROUPS = [
    ("iliopsoas",        "high", "髋屈肌（APT 应高）"),
    ("rectus_femoris",   "high", "髋屈肌（APT 应高）"),
    ("erector_spinae",   "high", "腰伸肌（APT 应高）"),
    ("gluteus_maximus",  "low",  "髋伸肌（APT 应低）"),
    ("gluteus_medius",   "low",  "髋外展肌（APT 应低/抑制）"),
]


def group_abs_mean(acts_np, mint_cols, group):
    """acts_np (T,402) -> 该肌群（左右合并）的标量绝对平均激活；缺失返回 None。"""
    idx = []
    for side in ("_R", "_L"):
        idx += get_indices(group + side, mint_cols)
    if not idx:
        return None
    return float(acts_np[..., idx].mean())


def load_hml(data, key):
    if key not in data:
        return None
    arr = data[key]  # (B,T,263)
    return np.asarray(arr, dtype=np.float32)


def analyze_one(mg, npy_path, device):
    data = np.load(npy_path, allow_pickle=True).item()
    hml_base = load_hml(data, "motion_hml_tj")
    hml_guided = load_hml(data, "motion_hml_tj_guided")
    if hml_base is None or hml_guided is None:
        print(f"[skip] {npy_path}: 缺少 motion_hml_tj(_guided)，现有 keys={list(data.keys())}")
        return
    mode = data.get("guidance_mode", os.path.basename(os.path.dirname(npy_path)))

    xb = torch.from_numpy(hml_base).float().to(device)
    xg = torch.from_numpy(hml_guided).float().to(device)
    with torch.no_grad():
        ab = mg._activations(xb).detach().cpu().numpy()  # (B,T,402)
        ag = mg._activations(xg).detach().cpu().numpy()

    print(f"\n{'='*82}")
    print(f"{npy_path}   (guidance_mode={mode})")
    print(f"{'='*82}")
    print(f"{'肌群':<18}{'期望形态':<12}{'baseline绝对':>14}{'guided绝对':>14}{'Δ':>12}")
    print("-" * 82)

    hipflex_g, hipext_g = [], []
    for group, expect, _desc in KEY_GROUPS:
        mb = group_abs_mean(ab, mg.mint_cols, group)
        mg_ = group_abs_mean(ag, mg.mint_cols, group)
        if mb is None or mg_ is None:
            print(f"{group:<18}{'(缺列)':<12}")
            continue
        d = mg_ - mb
        tag = "高" if expect == "high" else "低"
        print(f"{group:<18}{tag:<12}{mb:>14.4e}{mg_:>14.4e}{d:>+12.2e}")
        if group in ("iliopsoas", "rectus_femoris"):
            hipflex_g.append(mg_)
        if group == "gluteus_maximus":
            hipext_g.append(mg_)

    # --- 决定性判读：只看 guided 的绝对形态（髋屈 vs 髋伸 比值） ---
    if hipflex_g and hipext_g:
        flex = float(np.mean(hipflex_g))
        ext = float(np.mean(hipext_g))
        ratio = flex / (ext + 1e-12)
        print("-" * 82)
        print(f"guided 绝对形态：髋屈肌均值={flex:.4e}  髋伸肌(glmax)={ext:.4e}  "
              f"屈/伸比={ratio:.3f}")
        print("判读：临床 APT 期望 屈/伸比 > 1（髋屈高、髋伸低）。")
        if ratio < 1.0:
            print("  → ✗ guided 绝对形态是【髋伸主导 = PPT 形态】。")
            print("     即使绕过 reference，proxy 也把 APT 几何映射成 PPT 激活。")
            print("     结论：H_A（proxy 逆映射）主导，换 reference 救不了方向。")
        else:
            print("  → ✓ guided 绝对形态其实是【髋屈主导 = APT 形态】。")
            print("     若此前的 Δ 显示反向，则反向来自 baseline/reference 污染。")
            print("     结论：H_ref（reference 污染）主导，换中性 reference 可修复。")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("path", help="comparison.npy、其所在目录，或包含多个子目录的 output_root")
    ap.add_argument("--muscle_ckpt", required=True)
    ap.add_argument("--muscle_posture", default="anterior_pelvic_tilt")
    ap.add_argument("--muscle_assets_dir", default=_ASSETS)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--same_norm", action="store_true", default=True)
    ap.add_argument("--diff_norm", dest="same_norm", action="store_false")
    ap.add_argument("--proxy_mean_path", default=None)
    ap.add_argument("--proxy_std_path", default=None)
    args = ap.parse_args()

    device = torch.device(
        args.device if (torch.cuda.is_available() or args.device == "cpu") else "cpu")

    from muscle_guidance_mdm import build_muscle_guidance
    mg = build_muscle_guidance(
        ckpt_path=args.muscle_ckpt,
        posture_name=args.muscle_posture,
        assets_dir=args.muscle_assets_dir,
        same_normalization=args.same_norm,
        proxy_mean_path=args.proxy_mean_path,
        proxy_std_path=args.proxy_std_path,
        device=device,
    )
    # 本脚本只用 _activations，不需要 reference，但 mint_cols 必须就绪
    assert mg.mint_cols is not None

    # 收集待分析的 comparison.npy
    targets = []
    p = args.path
    if os.path.isfile(p) and p.endswith(".npy"):
        targets = [p]
    elif os.path.isdir(p):
        direct = os.path.join(p, "comparison.npy")
        if os.path.isfile(direct):
            targets = [direct]
        else:
            for entry in sorted(os.listdir(p)):
                npy = os.path.join(p, entry, "comparison.npy")
                if os.path.isfile(npy):
                    targets.append(npy)
    if not targets:
        print(f"未找到 comparison.npy：{p}")
        return

    for npy in targets:
        analyze_one(mg, npy, device)


if __name__ == "__main__":
    main()
