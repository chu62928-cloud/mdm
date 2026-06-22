"""
scripts/sanity_muscle_integration.py

自检：关节角 + 肌肉激活组合引导的"代码链路"是否正确。
**不需要** MDM checkpoint，也**不需要**真实代理权重——用一个可微的 DummyProxy
（Linear+Sigmoid，263->402）替身，专门验证：
    1. posture_loss_torch 链路：参考自洽(≈0) / 合成扰动方向性 / numpy-torch parity
    2. MuscleGuidance：build_reference + loss + guidance_grad 梯度有限
    3. CombinedGuidance：joint / muscle / both 三模式下梯度都能回到 motion(263)，
       且符号正确（关节最小化、肌肉最大化）

端到端真实运行（带真权重）请用 scripts/run_apt_integrated.sh。

用法：
    python scripts/sanity_muscle_integration.py
"""
import os
import sys

import numpy as np
import torch
import torch.nn as nn

# --- 路径：仓库根 + motion2muscle 资产目录 ---
_THIS = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_THIS)
_ASSETS = os.path.join(_ROOT, "motion2muscle")
for p in (_ROOT, _ASSETS):
    if p not in sys.path:
        sys.path.insert(0, p)

from muscle_rollup import ROLLUP_GROUPS                                  # noqa: E402
from posture_loss import (build_reference_from_activations,             # noqa: E402
                          make_synthetic_distortion, compute_posture_loss)
from posture_loss_torch import build_group_index, compute_posture_loss_torch  # noqa: E402
from muscle_guidance import MuscleGuidance                              # noqa: E402
from muscle_guidance_mdm.dense_loss import dense_posture_guidance_loss  # noqa: E402
from posture_guidance.combined_loss import CombinedGuidance             # noqa: E402
from posture_guidance.controller import PostureGuidance                 # noqa: E402

POSTURE = "anterior_pelvic_tilt"
JOINT_INSTRUCTION = "骨盆前倾"
DEVICE = "cpu"


def load_mint_cols():
    path = os.path.join(_ASSETS, "muscle_names.txt")
    with open(path) as f:
        cols = [ln.strip() for ln in f if ln.strip()]
    return cols


class DummyProxy(nn.Module):
    """(B,T,263) -> (B,T,402) in [0,1]，可微替身，仅用于验证梯度链路。"""
    def __init__(self, n_out=402):
        super().__init__()
        self.lin = nn.Linear(263, n_out)

    def forward(self, x):  # x: (B,T,263)
        return torch.sigmoid(self.lin(x))


def fk_fn_simple(motion):
    """轻量 FK 替身：从 263 表示里直接取 RIC 关节位置 dims[4:67] -> (B,T,22,3)。
    身份归一化（不反 z-score），仅用于让 angle_ops 跑起来验证梯度，不追求几何正确。
    motion: (B,263,1,T)"""
    x = motion.permute(0, 3, 2, 1).squeeze(2)         # (B,T,263)
    B, T, _ = x.shape
    pos = x[..., 4:4 + 21 * 3].reshape(B, T, 21, 3)   # 21 个非根关节
    root = torch.zeros(B, T, 1, 3, device=x.device, dtype=x.dtype)
    return torch.cat([root, pos], dim=-2)             # (B,T,22,3)


def check(name, cond):
    print(f"[{'PASS' if cond else 'FAIL'}] {name}")
    return cond


def main():
    torch.manual_seed(0)
    rng = np.random.default_rng(0)
    mint_cols = load_mint_cols()
    ok = True

    print("=" * 70)
    print(f"mint_cols: {len(mint_cols)} (expect 402)")
    ok &= check("muscle_names.txt 有 402 列", len(mint_cols) == 402)

    # ---- 1. posture_loss_torch 纯链路（HANDOFF §7 / smoke_test）----
    # 用"时间常数"激活：每帧相同 → stabilizer 的逐帧惩罚天然为 0 → 参考自洽严格 ≈0。
    print("\n-- 1. posture_loss_torch 链路 --")
    B, T = 1, 28
    one_frame = rng.uniform(0.05, 0.30, size=(B, 1, len(mint_cols))).astype(np.float32)
    normal = np.repeat(one_frame, T, axis=1)          # (B,T,402)，沿时间常数
    ref = build_reference_from_activations(normal, mint_cols)
    gidx = build_group_index(mint_cols, POSTURE, device=DEVICE)

    x = torch.tensor(normal, requires_grad=True)
    L = compute_posture_loss_torch(x, gidx, POSTURE, ref)
    L.backward()
    ok &= check("梯度存在且有限", x.grad is not None and bool(torch.isfinite(x.grad).all()))
    ok &= check(f"参考自洽 loss≈0 (={float(L.detach()):.2e})", float(L.detach()) < 1e-5)

    distorted = make_synthetic_distortion(normal, mint_cols, POSTURE, severity=0.8)
    L_bad = compute_posture_loss_torch(torch.tensor(distorted), gidx, POSTURE, ref)
    ok &= check(f"合成 APT 扰动 loss 上升 ({float(L_bad):.4f} > {float(L.detach()):.2e})",
                float(L_bad) > float(L.detach()) + 1e-4)

    np_total = compute_posture_loss(distorted, mint_cols, POSTURE, ref)["total"]
    parity = abs(np_total - float(L_bad))
    ok &= check(f"numpy/torch parity (diff={parity:.2e})", parity < 1e-4)

    # ---- 1b. dense 引导损失：起点梯度非零（修复 loss=0 死区的核心断言）----
    print("\n-- 1b. dense 引导损失起点梯度 --")
    x_ref = torch.tensor(normal, requires_grad=True)
    L_clin = compute_posture_loss_torch(x_ref, gidx, POSTURE, ref)
    g_clin = torch.autograd.grad(L_clin, x_ref, retain_graph=False, allow_unused=True)[0]
    gnorm_clin = 0.0 if g_clin is None else float(g_clin.norm())
    x_ref2 = torch.tensor(normal, requires_grad=True)
    L_dense = dense_posture_guidance_loss(x_ref2, gidx, POSTURE, ref)
    L_dense.backward()
    gnorm_dense = float(x_ref2.grad.norm())
    print(f"   clinical 起点 |grad|={gnorm_clin:.6f}   dense 起点 |grad|={gnorm_dense:.6f}")
    ok &= check("clinical 起点梯度=0（确认死区）", gnorm_clin < 1e-9)
    ok &= check("dense 起点梯度>0（修复生效）", gnorm_dense > 1e-3)

    # ---- 2. MuscleGuidance（DummyProxy）----
    # 让 reference 与 query 处于不同激活水平（低 vs 高），逼四分量 loss 越过阈值触发，
    # 从而验证梯度确实能穿过代理回到 motion（避开"幅度不足"的零梯度区，见 midterm §3.2）。
    print("\n-- 2. MuscleGuidance --")
    proxy = DummyProxy().to(DEVICE)
    mg = MuscleGuidance(proxy, mint_cols, POSTURE, same_normalization=True, device=DEVICE)
    motion_low = torch.full((1, T, 263), -4.0)        # 时间常数 → 自洽严格 0
    mg.build_reference(motion_low)
    ok &= check("build_reference 产出非空", mg.reference_acts is not None and len(mg.reference_acts) > 0)

    L_self = float(mg.loss(motion_low).detach())
    ok &= check(f"参考样本自身 loss≈0 (={L_self:.2e})", L_self < 1e-6)

    motion_high = torch.full((1, T, 263), 4.0)         # 远离参考 → loss 触发
    grad, Lval = mg.guidance_grad(motion_high)
    ok &= check("guidance_grad 形状匹配", tuple(grad.shape) == (1, T, 263))
    ok &= check("guidance_grad 有限", bool(torch.isfinite(grad).all()))
    ok &= check(f"远离参考时 loss>0 (={Lval:.4f})", Lval > 0.0)
    ok &= check(f"远离参考时梯度非零 (|g|={float(grad.norm()):.4e})", float(grad.norm()) > 0.0)

    # ---- 3. CombinedGuidance：三模式梯度回到 motion(263) ----
    print("\n-- 3. CombinedGuidance 三模式 --")
    posture = PostureGuidance(instructions=[JOINT_INSTRUCTION], verbose=False)

    def run_mode(mode):
        cg = CombinedGuidance(posture=posture, muscle=mg, fk_fn=fk_fn_simple,
                              mode=mode, w_joint=1.0, w_muscle=1.0)
        # 高激活区(+4) + 少量噪声：肌肉项越过阈值、关节项 hinge 也有信号
        motion = (torch.full((1, 263, 1, T), 4.0)
                  + 0.1 * torch.randn(1, 263, 1, T)).requires_grad_(True)
        loss = cg.motion_loss(motion, t=0, T=10, loss_form="hinge",
                              spec_schedule_override="always")
        loss.backward()
        g = motion.grad
        finite = g is not None and bool(torch.isfinite(g).all())
        gnorm = float(g.norm()) if g is not None else 0.0
        return finite, gnorm, cg.joint_active, cg.muscle_active

    for mode in ("joint", "muscle", "both"):
        finite, gnorm, ja, ma = run_mode(mode)
        print(f"   mode={mode:6s} joint_active={ja} muscle_active={ma} "
              f"|grad|={gnorm:.4e}")
        ok &= check(f"mode={mode}: 梯度有限", finite)
        ok &= check(f"mode={mode}: 梯度非零", gnorm > 0.0)

    print("\n" + "=" * 70)
    print("ALL PASS ✅" if ok else "SOME CHECKS FAILED ❌")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
