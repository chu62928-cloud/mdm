"""
new/sdEdit_ood.py

路线 B：IK 关键帧注入 + SDEdit 重去噪，用于生成膝超伸（190°）这类全局 OOD 体态。

背景：
    诊断（new/analyze_knee_angles.py + retrieve_ood_prompts.py）已确认膝超伸
    190° 是「全局 OOD」——HumanML3D 训练集里 three_point_angle 物理上限为 180°，
    不存在膝向后弯的几何配置。因此 guidance（V2/V6）和组合式扩散（路线 A）
    都无法注入这块零密度区域（受 guidance 的 reshape-only 限制）。

    路线 B 绕开该限制：直接用 IK 把膝角强制设到 190°（IK 不受概率密度约束），
    得到运动学正确但动力学不自然的序列；再加噪到中间时间步 t0、用 MDM 去噪，
    让 MDM 恢复自然性。t0 决定「自然性 ↔ 超伸保留」的权衡：
        t0 高（噪声多）→ 自然但超伸被拉回
        t0 低（噪声少）→ 保住超伸但可能抖动 / 不自然
    存在一个最优甜点。可选叠加 V6 PID guidance（hybrid），在去噪过程中
    持续把膝角往 190° 推，抵抗 MDM 的回拉。

实现要点：
    - IK 在全局 xyz 空间做：绕膝关节刚体旋转小腿（knee→ankle），保持骨长，
      把 signed_knee_angle 推到 target（默认 190°）。foot 随 ankle 刚体平移。
    - 仅修改「接近伸直」的帧（baseline 三点角 > --min-base-deg），保留摆动相深屈曲。
    - 编码回 263-d 时只改 ric_data 通道（纯 torch 逆变换，避开 extract_features
      对已废弃 np.float 的依赖）；rot_data/vel 通道留旧值，加噪去噪时被 MDM 协调。
    - SDEdit 用 p_sample_loop 原生的 skip_timesteps + init_image（内部 q_sample 加噪）。

用法：
    # 纯 SDEdit（不加 guidance）
    python -m new.sdEdit_ood \\
        --model_path ./save/humanml_trans_enc_512/model000200000.pt \\
        --text_prompt "a person is walking forward" \\
        --t0_ratio 0.5 --seed 42 \\
        --output_dir ./output/sdedit_t05/

    # SDEdit + V6 PID hybrid（env 配置 V6）
    GUIDANCE_VARIANT=v6_closed_loop \\
    GUIDANCE_KWARGS_JSON='{"Kp":30,"Ki":1,"Kd":5,"s_min":5,"s_max":120,"manifold_project":false}' \\
    python -m new.sdEdit_ood --model_path ... --text_prompt "..." \\
        --posture_instructions 膝超伸 --t0_ratio 0.5 --output_dir ./output/sdedit_v6_t05/

    # t0 扫描：对 {0.3,0.5,0.7} 各跑一次，比较膝角保留 vs 自然性
"""
import sys
import os
import json

# V6 等 guidance 通过环境变量配置（被 diffusion 内部 dispatch 读取）
GUIDANCE_VARIANT = os.environ.get("GUIDANCE_VARIANT", "v1_mu_sgd")
_kw_json = os.environ.get("GUIDANCE_KWARGS_JSON", "{}")
GUIDANCE_KWARGS = json.loads(_kw_json)
DIAGNOSTIC = os.environ.get("DIAGNOSTIC", "0") == "1"

import copy
import math
import argparse
import numpy as np
import torch

from utils.fixseed import fixseed
from utils.parser_util import generate_args
from utils.model_util import create_model_and_diffusion, load_saved_model
from utils import dist_util
from utils.sampler_util import ClassifierFreeSampleModel
from data_loaders.get_data import get_dataset_loader
from data_loaders.humanml.scripts.motion_process import recover_from_ric, recover_root_rot_pos
from data_loaders.humanml.common.quaternion import qrot, qinv
from data_loaders.tensors import collate

from posture_guidance.mdm_integration import make_fk_fn
from posture_guidance import angle_ops
from posture_guidance.joint_indices import get_joint_idx


# 腿部关节索引（SMPL 22-joint）
LEG_JOINTS = {
    "left":  {"hip": "left_hip",  "knee": "left_knee",  "ankle": "left_ankle",  "foot": "left_foot"},
    "right": {"hip": "right_hip", "knee": "right_knee", "ankle": "right_ankle", "foot": "right_foot"},
}


# =========================================================
# 表示转换工具
# =========================================================

def hml_to_xyz(sample_hml, data, n_joints=22):
    """(B,263,1,T) hml_vec → (B,J,3,T) xyz，与 generate.py 后处理一致。"""
    sample = data.dataset.t2m_dataset.inv_transform(
        sample_hml.cpu().permute(0, 2, 3, 1)
    ).float()
    sample = recover_from_ric(sample, n_joints)
    sample = sample.view(-1, *sample.shape[2:]).permute(0, 2, 3, 1)
    return sample


def hml_raw_to_global(raw):
    """
    raw: (B,1,T,263) 反归一化后的 hml_vec。
    返回 global_pos (B,T,22,3), r_rot_quat (B,T,4), r_pos (B,T,3)。
    """
    r_rot_quat, r_pos = recover_root_rot_pos(raw)          # (B,1,T,4), (B,1,T,3)
    global_pos = recover_from_ric(raw, 22)                 # (B,1,T,22,3)
    return (global_pos.squeeze(1),
            r_rot_quat.squeeze(1),
            r_pos.squeeze(1))


def global_to_ric(global_pos, r_rot_quat, r_pos):
    """
    recover_from_ric 中 ric 部分的逆：global joints 1..21 → ric_data (B,T,63)。

    正向（recover_from_ric）：
        global = qrot(qinv(r_rot), local);  global.xz += r_pos.xz
    逆：
        local = qrot(r_rot, global - r_pos.xz)
    """
    local = global_pos[:, :, 1:, :].clone()               # (B,T,21,3)
    local[..., 0] -= r_pos[..., 0:1]
    local[..., 2] -= r_pos[..., 2:3]
    q = r_rot_quat[:, :, None, :].expand(local.shape[:-1] + (4,))
    local = qrot(q, local)                                # global→root frame
    return local.reshape(local.shape[0], local.shape[1], -1)  # (B,T,63)


# =========================================================
# IK：膝超伸
# =========================================================

def rotate_about_axis(v, axis, angle):
    """Rodrigues 旋转。v:(...,3), axis:(...,3) 单位向量, angle:(...,) 弧度。"""
    axis = torch.nn.functional.normalize(axis, dim=-1, eps=1e-8)
    cos = torch.cos(angle).unsqueeze(-1)
    sin = torch.sin(angle).unsqueeze(-1)
    dot = (axis * v).sum(-1, keepdim=True)
    cross = torch.cross(axis, v, dim=-1)
    return v * cos + cross * sin + axis * dot * (1.0 - cos)


def _signed_knee_from_pose(q, side):
    """在完整 (B,T,22,3) pose 上算 signed_knee_angle（度）。"""
    return angle_ops.signed_knee_angle(q, side=side) * (180.0 / math.pi)


def hyperextend_leg(global_pos, side, target_deg, min_base_deg):
    """
    把一条腿在全局空间超伸到 target_deg（signed 定义，>180°）。
    只修改「baseline 三点角 > min_base_deg」的帧（即接近伸直的支撑相）。

    几何：保持 hip、knee 不动；绕膝、沿内外侧轴刚体旋转小腿（knee→ankle）到
    目标内角；foot 随 ankle 同位移刚体平移。两个旋转方向都试，选 signed 角更
    接近 target 的那个（自动满足超伸方向）。

    Args:
        global_pos: (B,T,22,3)，原地修改并返回
        side: "left" / "right"
        target_deg: 目标 signed 膝角（如 190）
        min_base_deg: 仅修改三点角 > 此值的帧（度）
    """
    j = LEG_JOINTS[side]
    i_hip, i_knee  = get_joint_idx(j["hip"]),  get_joint_idx(j["knee"])
    i_ankle, i_foot = get_joint_idx(j["ankle"]), get_joint_idx(j["foot"])
    i_lhip, i_rhip = get_joint_idx("left_hip"), get_joint_idx("right_hip")

    H = global_pos[:, :, i_hip,   :]                      # (B,T,3)
    K = global_pos[:, :, i_knee,  :]
    A = global_pos[:, :, i_ankle, :]
    F = global_pos[:, :, i_foot,  :]

    thigh_dir = torch.nn.functional.normalize(K - H, dim=-1, eps=1e-8)   # hip→knee
    L2 = (A - K).norm(dim=-1, keepdim=True)              # 胫骨长（保持不变）

    # 内外侧旋转轴：髋连线（左右一致）
    ml_axis = global_pos[:, :, i_rhip, :] - global_pos[:, :, i_lhip, :]   # (B,T,3)

    # 目标内角（三点角）：signed>180 → 内角 = 360 - target；base_bend = 过直角度
    interior_deg = 360.0 - target_deg
    base_bend = math.radians(180.0 - interior_deg)

    # signed_knee_angle 的方向判定是 sigmoid 软函数，膝不够靠后时读数偏低。
    # 因此扫描多个弯曲幅度 × 两个旋转方向，逐帧选 signed 读数最接近 target 的候选，
    # 自校正软度量偏差。
    cand_ankles = []
    for mag in (1.0, 1.3, 1.6, 2.0, 2.5):
        bend_t = torch.full(thigh_dir.shape[:-1], base_bend * mag, device=global_pos.device)
        for sgn in (+1.0, -1.0):
            d = rotate_about_axis(thigh_dir, ml_axis, sgn * bend_t)
            cand_ankles.append(K + L2 * d)               # (B,T,3)

    # 逐候选评估 signed 角
    errs, newAs = [], []
    for newA in cand_ankles:
        pose = global_pos.clone()
        pose[:, :, i_ankle, :] = newA
        pose[:, :, i_foot,  :] = F + (newA - A)
        ang = _signed_knee_from_pose(pose, side)         # (B,T)
        errs.append((ang - target_deg).abs())
        newAs.append(newA)

    err_stack = torch.stack(errs, dim=0)                 # (C,B,T)
    newA_stack = torch.stack(newAs, dim=0)               # (C,B,T,3)
    best = err_stack.argmin(dim=0)                       # (B,T)
    sel_ankle = torch.gather(
        newA_stack, 0,
        best[None, ..., None].expand(1, *best.shape, 3)
    ).squeeze(0)                                         # (B,T,3)
    sel_foot = F + (sel_ankle - A)

    # 仅在接近伸直的帧应用（用 baseline 三点角判定，保留摆动相深屈曲）
    base_tp = angle_ops.three_point_angle(
        global_pos, j["hip"], j["knee"], j["ankle"]
    ) * (180.0 / math.pi)                                # (B,T)
    apply = base_tp > min_base_deg

    m = apply.unsqueeze(-1)
    global_pos[:, :, i_ankle, :] = torch.where(m, sel_ankle, A)
    global_pos[:, :, i_foot,  :] = torch.where(m, sel_foot,  F)
    return global_pos, apply


def ik_inject_hyperextension(baseline_hml, data, target_deg, min_base_deg, sides):
    """
    baseline_hml: (B,263,1,T) 归一化 hml_vec（MDM 输出）。
    返回 (ik_hml (B,263,1,T) 归一化, ik_xyz (B,J,3,T), diag dict)。
    只改 ric_data 通道，root/rot/vel/feet 保持 baseline。
    """
    t2m = data.dataset.t2m_dataset
    mean = torch.tensor(t2m.mean, dtype=torch.float32)   # (263,)
    std  = torch.tensor(t2m.std,  dtype=torch.float32)

    # (B,263,1,T) → (B,1,T,263) 反归一化
    raw = baseline_hml.cpu().permute(0, 2, 3, 1).float() * std + mean   # (B,1,T,263)

    global_pos, r_rot_quat, r_pos = hml_raw_to_global(raw)   # (B,T,22,3),(B,T,4),(B,T,3)

    # 记录注入前的膝角
    pre = {s: _signed_knee_from_pose(global_pos, s).mean().item() for s in sides}

    applied_frac = {}
    for side in sides:
        global_pos, applied = hyperextend_leg(
            global_pos, side, target_deg, min_base_deg
        )
        applied_frac[side] = applied.float().mean().item()

    post = {s: _signed_knee_from_pose(global_pos, s).mean().item() for s in sides}

    # 写回 ric_data 通道（dims 4:67）；ric_new (B,T,63) → (B,1,T,63) 匹配 raw
    ric_new = global_to_ric(global_pos, r_rot_quat, r_pos)  # (B,T,63)
    raw_ik = raw.clone()
    raw_ik[..., 4:4 + 63] = ric_new.unsqueeze(1)

    # 重新归一化 → (B,263,1,T)
    ik_norm = (raw_ik - mean) / std                      # (B,1,T,263)
    ik_hml = ik_norm.permute(0, 3, 1, 2).contiguous()    # (B,263,1,T)

    # IK 后的 xyz（直接从编辑过的 global_pos，最准确）
    ik_xyz = global_pos.permute(0, 2, 3, 1).contiguous() # (B,22,3,T)

    diag = {
        "pre_knee_deg":  pre,
        "post_knee_deg": post,
        "applied_frac":  applied_frac,
        "target_deg":    target_deg,
        "min_base_deg":  min_base_deg,
    }
    return ik_hml, ik_xyz.cpu().numpy(), diag


# =========================================================
# 采样
# =========================================================

def run_baseline(diffusion, model, motion_shape, model_kwargs, init_noise, seed):
    """标准采样（skip=0），返回干净 hml (B,263,1,T)。"""
    fixseed(seed)
    return diffusion.p_sample_loop(
        model, motion_shape,
        clip_denoised=False, model_kwargs=model_kwargs,
        skip_timesteps=0, init_image=None, progress=True,
        noise=init_noise.clone(), const_noise=False,
        posture_instructions=None, posture_fk_fn=None,
    )


def run_sdedit(diffusion, model, motion_shape, model_kwargs, ik_hml,
               t0_ratio, posture_instructions, fk_fn, seed,
               lbfgs_steps, lr):
    """
    SDEdit：把 ik_hml 加噪到 t0 再去噪。
        skip_timesteps = round(T_total * (1 - t0_ratio))
        → 去噪从 t0 = T_total*t0_ratio 起；t0_ratio 越小，保留 IK 越多。
    可选 posture_instructions 启用 V6 PID（hybrid）。
    """
    T_total = diffusion.num_timesteps
    skip = int(round(T_total * (1.0 - t0_ratio)))
    skip = max(0, min(T_total - 1, skip))
    print(f"[SDEdit] T_total={T_total}  t0_ratio={t0_ratio}  "
          f"→ skip_timesteps={skip}  起始 t0≈{T_total - skip}")

    init_image = ik_hml.to(dist_util.dev())              # (B,263,1,T)

    fixseed(seed)
    return diffusion.p_sample_loop(
        model, motion_shape,
        clip_denoised=False, model_kwargs=model_kwargs,
        skip_timesteps=skip, init_image=init_image, progress=True,
        noise=None, const_noise=False,
        posture_instructions=posture_instructions,
        posture_lbfgs_steps=lbfgs_steps, posture_lr=lr,
        posture_fk_fn=fk_fn,
    )


# =========================================================
# 参数
# =========================================================

def parse_extra_args():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--posture_instructions", nargs="+", default=[],
                        help="SDEdit 去噪时叠加的 V6 PID 约束（如 膝超伸）；留空=纯 SDEdit")
    parser.add_argument("--t0_ratio", type=float, default=0.5,
                        help="SDEdit 加噪起始水平占比（0-1）；小=保留 IK 多，大=更自然")
    parser.add_argument("--target_signed_deg", type=float, default=190.0,
                        help="IK 目标 signed 膝角（度），默认 190 超伸 10°")
    parser.add_argument("--min_base_deg", type=float, default=150.0,
                        help="仅对 baseline 三点角 > 此值的帧做超伸（度），保留摆动相屈曲")
    parser.add_argument("--sides", nargs="+", default=["left", "right"],
                        choices=["left", "right"], help="超伸哪几条腿")
    parser.add_argument("--posture_lbfgs_steps", type=int, default=5)
    parser.add_argument("--posture_lr", type=float, default=0.05)
    parser.add_argument("--comparison_output", type=str, default="comparison.npy")
    args, unknown = parser.parse_known_args()
    sys.argv = [sys.argv[0]] + unknown
    return args


# =========================================================
# 主流程
# =========================================================

def main():
    extra = parse_extra_args()
    args = generate_args()
    for k in ["posture_instructions", "t0_ratio", "target_signed_deg",
              "min_base_deg", "sides", "posture_lbfgs_steps", "posture_lr",
              "comparison_output"]:
        setattr(args, k, getattr(extra, k))

    fixseed(args.seed)
    print(f"[run] variant={GUIDANCE_VARIANT}  kwargs={GUIDANCE_KWARGS}")
    print(f"[run] t0_ratio={args.t0_ratio}  target={args.target_signed_deg}°  "
          f"posture={args.posture_instructions or '(纯SDEdit)'}")

    out_dir = args.output_dir or "./output/sdedit_run"
    os.makedirs(out_dir, exist_ok=True)

    dist_util.setup_dist(args.device)
    n_joints = 22 if args.dataset == "humanml" else 21
    max_frames = 196 if args.dataset in ["kit", "humanml"] else 60
    fps = 12.5 if args.dataset == "kit" else 20
    n_frames = min(max_frames, int(args.motion_length * fps))
    args.batch_size = args.num_samples

    print("Loading dataset...")
    data = get_dataset_loader(
        name=args.dataset, batch_size=args.batch_size, num_frames=max_frames,
        split="test", hml_mode="text_only",
    )
    data.fixed_length = n_frames

    print("Creating model and diffusion...")
    model, diffusion = create_model_and_diffusion(args, data)
    print(f"Loading checkpoint from [{args.model_path}]...")
    load_saved_model(model, args.model_path, use_avg=args.use_ema)
    if args.guidance_param != 1:
        model = ClassifierFreeSampleModel(model)
    model.to(dist_util.dev())
    model.eval()

    motion_shape = (args.batch_size, model.njoints, model.nfeats, n_frames)

    # model_kwargs（baseline 与 SDEdit 共用）
    text_prompt = args.text_prompt or "a person is walking forward"
    texts = [text_prompt] * args.num_samples
    collate_args = [{"inp": torch.zeros(n_frames), "tokens": None, "lengths": n_frames}] * args.num_samples
    collate_args = [dict(a, text=t) for a, t in zip(collate_args, texts)]
    _, model_kwargs = collate(collate_args)
    model_kwargs["y"] = {k: (v.to(dist_util.dev()) if torch.is_tensor(v) else v)
                         for k, v in model_kwargs["y"].items()}
    if args.guidance_param != 1:
        model_kwargs["y"]["scale"] = torch.ones(args.batch_size, device=dist_util.dev()) * args.guidance_param
    if "text" in model_kwargs["y"]:
        model_kwargs["y"]["text_embed"] = model.encode_text(model_kwargs["y"]["text"])

    # 共享初始噪声（baseline 用）
    fixseed(args.seed)
    init_noise = torch.randn(*motion_shape, device=dist_util.dev())

    # V6 hybrid 的 fk_fn：仅当指定 posture_instructions 时构建并传入，
    # diffusion 内部会据此 + 环境变量（GUIDANCE_VARIANT/KWARGS）自建 PostureGuidance。
    fk_fn = None
    if args.posture_instructions:
        fk_fn = make_fk_fn(t2m_dataset=data.dataset.t2m_dataset, n_joints=n_joints)

    # ---- Pass 1: baseline ----
    print("\n" + "=" * 70 + "\nPass 1: baseline (clean walking, IK 源)\n" + "=" * 70)
    baseline_hml = run_baseline(
        diffusion, model, motion_shape,
        copy.deepcopy(model_kwargs), init_noise, args.seed,
    )
    baseline_xyz = hml_to_xyz(baseline_hml, data, n_joints).cpu().numpy()

    # ---- IK 注入 ----
    print("\n" + "=" * 70 + "\nIK: 膝超伸注入\n" + "=" * 70)
    ik_hml, ik_xyz, diag = ik_inject_hyperextension(
        baseline_hml, data, args.target_signed_deg, args.min_base_deg, args.sides,
    )
    print(f"  注入前膝角(均值): {diag['pre_knee_deg']}")
    print(f"  注入后膝角(均值): {diag['post_knee_deg']}")
    print(f"  修改帧占比:       {diag['applied_frac']}")

    # ---- Pass 2: SDEdit 重去噪 ----
    print("\n" + "=" * 70 + f"\nPass 2: SDEdit (t0_ratio={args.t0_ratio})"
          + (f" + V6 {args.posture_instructions}" if args.posture_instructions else " 纯SDEdit")
          + "\n" + "=" * 70)
    sdedit_hml = run_sdedit(
        diffusion, model, motion_shape, copy.deepcopy(model_kwargs), ik_hml,
        args.t0_ratio, args.posture_instructions or None, fk_fn, args.seed,
        args.posture_lbfgs_steps, args.posture_lr,
    )
    sdedit_xyz = hml_to_xyz(sdedit_hml, data, n_joints).cpu().numpy()

    # 报告最终膝角
    sd_q = torch.from_numpy(sdedit_xyz).permute(0, 3, 1, 2).float()  # (B,T,J,3)
    final_knee = {s: _signed_knee_from_pose(sd_q, s).mean().item() for s in args.sides}
    print(f"\n[结果] SDEdit 后膝角(均值): {final_knee}  (目标 {args.target_signed_deg}°)")

    # ---- 保存（与现有分析管线兼容：motion_xyz / motion_xyz_guided）----
    save_dict = {
        "motion_hml":           baseline_hml.detach().cpu().numpy(),
        "motion_xyz":           baseline_xyz,                # baseline
        "motion_xyz_guided":    sdedit_xyz,                  # SDEdit 输出（供 aggregate/compare）
        "motion_xyz_ik":        ik_xyz,                      # IK 注入目标（诊断用）
        "motion_hml_guided":    sdedit_hml.detach().cpu().numpy(),
        "text_prompt":          text_prompt,
        "posture_instructions": args.posture_instructions,
        "seed":                 args.seed,
        "num_samples":          args.num_samples,
        "motion_length":        args.motion_length,
        "fps":                  fps,
        "sdedit_config": {
            "t0_ratio":         args.t0_ratio,
            "target_signed_deg": args.target_signed_deg,
            "min_base_deg":     args.min_base_deg,
            "sides":            args.sides,
            "variant":          GUIDANCE_VARIANT,
            "variant_kwargs":   GUIDANCE_KWARGS,
        },
        "ik_diag":              diag,
        "final_knee_deg":       final_knee,
    }
    out_path = os.path.join(out_dir, args.comparison_output)
    np.save(out_path, save_dict, allow_pickle=True)

    print("\n" + "=" * 70)
    print(f"✓ 保存到 {out_path}")
    print("=" * 70)
    print("后续：")
    print(f"  python -m new.quantitative_compare {out_path}")
    print(f"  python -m new.visualize_compare    {out_path}")


if __name__ == "__main__":
    main()
