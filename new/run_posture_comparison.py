"""
scripts/run_posture_comparison.py

对照实验主脚本：
    用同一个 text prompt + 同一个 seed + 同一份初始噪声，
    分别跑两次：
        Pass 1 — baseline (无 guidance)
        Pass 2 — guided   (启用 posture guidance)
    所有结果（hml + xyz + 元数据）存到一个 npy 文件，方便后续分析。

调用示例：
    python -m scripts.run_posture_comparison \
        --model_path ./save/humanml_trans_enc_512/model000200000.pt \
        --text_prompt "a person is walking forward" \
        --posture_instructions 骨盆前倾 膝超伸 \
        --num_repetitions 1 \
        --num_samples 1 \
        --motion_length 6.0 \
        --seed 42 \
        --output_dir ./output/comparison_run \
        --posture_lbfgs_steps 5 \
        --posture_lr 0.05

设计要点：
    1. 严格固定 seed + 复用初始噪声，保证两次采样除 guidance 外完全一致
    2. 两次 pass 共用同一份 model_kwargs，避免 text encoding 漂移
    3. baseline 和 guided 的输出存到同一个 npy 的不同 key 下
"""
import sys
import os, json

# 读 ablation 配置 (根据图示要求添加)
GUIDANCE_VARIANT = os.environ.get("GUIDANCE_VARIANT", "v1_mu_sgd")
_kw_json = os.environ.get("GUIDANCE_KWARGS_JSON", "{}")
print(f"[DEBUG] _kw_json 的原始字符串是: {_kw_json}")
GUIDANCE_KWARGS = json.loads(_kw_json)
DIAGNOSTIC = os.environ.get("DIAGNOSTIC", "0") == "1"

print(f"[run] variant={GUIDANCE_VARIANT}")
print(f"[run] variant_kwargs={GUIDANCE_KWARGS}")

import copy
import argparse
import numpy as np
import torch
from pathlib import Path

# 复用项目内的工具
from utils.fixseed import fixseed
from utils.parser_util import generate_args
from utils.model_util import create_model_and_diffusion, load_saved_model
from utils import dist_util
from utils.sampler_util import ClassifierFreeSampleModel
from data_loaders.get_data import get_dataset_loader
from data_loaders.humanml.scripts.motion_process import recover_from_ric
from data_loaders.tensors import collate

# Posture guidance 模块
from posture_guidance.controller import PostureGuidance
from posture_guidance.mdm_integration import make_fk_fn


# =========================================================
# 工具函数
# =========================================================

def hml_to_xyz(sample_hml, data, n_joints=22):
    """
    将 hml_vec 表示 (B, 263, 1, T) 转为 xyz 关节坐标 (B, J, 3, T)。
    与 generate.py 里的后处理逻辑一致。
    """
    sample = data.dataset.t2m_dataset.inv_transform(
        sample_hml.cpu().permute(0, 2, 3, 1)
    ).float()
    sample = recover_from_ric(sample, n_joints)
    sample = sample.view(-1, *sample.shape[2:]).permute(0, 2, 3, 1)
    return sample


def run_single_pass(
    model, diffusion, data,
    motion_shape, model_kwargs,
    init_noise,
    posture_instructions,
    fk_fn,
    posture_lbfgs_steps,
    posture_lr,
    seed,
):
    """
    跑一次完整的采样，返回 hml + xyz 两种格式。

    严格固定随机数：
        - 进入采样前重新 fixseed(seed)
        - noise 参数显式传入 init_noise 的副本，避免被内部修改
    """

    print(f">>> run_single_pass: posture={posture_instructions}, "
          f"lr={posture_lr}, steps={posture_lbfgs_steps}, "
          f"fk_fn={'None' if fk_fn is None else 'set'}")

    fixseed(seed)

    sample_hml = diffusion.p_sample_loop(
        model,
        motion_shape,
        clip_denoised=False,
        model_kwargs=model_kwargs,
        skip_timesteps=0,
        init_image=None,
        progress=True,
        dump_steps=None,
        noise=init_noise.clone(),
        const_noise=False,
        posture_instructions=posture_instructions,
        posture_lbfgs_steps=posture_lbfgs_steps,
        posture_lr=posture_lr,
        posture_fk_fn=fk_fn,
    )
    # sample_hml: (B, 263, 1, T)

    # 转为 xyz
    n_joints = 22 if sample_hml.shape[1] == 263 else 21
    sample_xyz = hml_to_xyz(sample_hml, data, n_joints=n_joints)

    return {
        "hml":      sample_hml.detach().cpu().numpy(),                     # (B, 263, 1, T)
        "hml_tj":   sample_hml.detach().cpu().numpy().squeeze(2).transpose(0, 2, 1),  # (B, T, 263)
        "xyz":      sample_xyz.cpu().numpy(),                              # (B, J, 3, T)
    }


# =========================================================
# 命令行参数（继承 generate_args 并扩展）
# =========================================================

def parse_extra_args():
    """
    在 generate_args 之外新增对照实验需要的参数。
    用 sys.argv 直接修改的方式，复用 generate_args 的所有原有参数。
    """
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--posture_instructions", nargs="+", default=[],
                        help="要施加的体态约束列表，例如 骨盆前倾 膝超伸")
    parser.add_argument("--posture_lbfgs_steps", type=int, default=5)
    parser.add_argument("--posture_lr", type=float, default=0.05)
    parser.add_argument("--comparison_output", type=str, default="comparison.npy",
                        help="对照实验结果保存文件名（在 output_dir 下）")
    args, unknown = parser.parse_known_args()
    sys.argv = [sys.argv[0]] + unknown
    return args


# =========================================================
# 主流程
# =========================================================

def main():
    extra_args = parse_extra_args()
    args = generate_args()

    # 把 extra_args 的字段附加到 args 上
    args.posture_instructions = extra_args.posture_instructions
    args.posture_lbfgs_steps  = extra_args.posture_lbfgs_steps
    args.posture_lr           = extra_args.posture_lr
    args.comparison_output    = extra_args.comparison_output

    if not args.posture_instructions:
        print("[Warning] 没有传入 --posture_instructions，"
              "对照实验将退化为两次相同的 baseline 采样。")

    fixseed(args.seed)

    # ----- 路径准备 -----
    out_dir = args.output_dir
    if out_dir == "":
        out_dir = "./output/comparison_run"
    os.makedirs(out_dir, exist_ok=True)

    # ----- 设备 / 数据集 / 模型 -----
    dist_util.setup_dist(args.device)
    print("Loading dataset...")

    n_joints = 22 if args.dataset == "humanml" else 21
    max_frames = 196 if args.dataset in ["kit", "humanml"] else 60
    fps = 12.5 if args.dataset == "kit" else 20
    n_frames = min(max_frames, int(args.motion_length * fps))

    args.batch_size = args.num_samples

    data = get_dataset_loader(
        name=args.dataset,
        batch_size=args.batch_size,
        num_frames=max_frames,
        split="test",
        hml_mode="text_only",
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

    # ----- 准备 model_kwargs（两次 pass 共用） -----
    text_prompt = args.text_prompt if args.text_prompt else "a person is walking"
    texts = [text_prompt] * args.num_samples

    collate_args = [{"inp": torch.zeros(n_frames),
                     "tokens": None,
                     "lengths": n_frames}] * args.num_samples
    collate_args = [dict(arg, text=txt) for arg, txt in zip(collate_args, texts)]
    _, model_kwargs = collate(collate_args)

    model_kwargs["y"] = {
        k: (v.to(dist_util.dev()) if torch.is_tensor(v) else v)
        for k, v in model_kwargs["y"].items()
    }

    if args.guidance_param != 1:
        model_kwargs["y"]["scale"] = torch.ones(
            args.batch_size, device=dist_util.dev()
        ) * args.guidance_param

    if "text" in model_kwargs["y"]:
        model_kwargs["y"]["text_embed"] = model.encode_text(model_kwargs["y"]["text"])

    # ----- 关键：预生成共享的初始噪声 -----
    # 在所有采样调用前生成，两次 pass 都用这份噪声的副本
    fixseed(args.seed)
    init_noise = torch.randn(*motion_shape, device=dist_util.dev())
    print(f"[Init noise] shape={init_noise.shape}, "
          f"mean={init_noise.mean().item():.4f}, "
          f"std={init_noise.std().item():.4f}")

    # ----- 准备 fk_fn（仅 guided pass 用到，但提前构建一次） -----
    fk_fn = make_fk_fn(
        t2m_dataset=data.dataset.t2m_dataset,
        n_joints=n_joints,
    )

    posture_guidance = PostureGuidance(instructions=args.posture_instructions)
    posture_guidance.set_variant(
        variant=GUIDANCE_VARIANT,
        variant_kwargs=GUIDANCE_KWARGS,
        diagnostic=DIAGNOSTIC,
    )


    # =========================================================
    # Pass 1: baseline（无 guidance）
    # =========================================================
    print("\n" + "=" * 70)
    print("Pass 1: baseline (no guidance)")
    print("=" * 70)
    baseline_out = run_single_pass(
        model=model,
        diffusion=diffusion,
        data=data,
        motion_shape=motion_shape,
        model_kwargs=copy.deepcopy(model_kwargs),  # 防止内部修改
        init_noise=init_noise,
        posture_instructions=None,                 # ★ 关键：不施加约束
        fk_fn=None,
        posture_lbfgs_steps=args.posture_lbfgs_steps,
        posture_lr=args.posture_lr,
        seed=args.seed,
    )

    # =========================================================
    # Pass 2: guided（带约束）
    # =========================================================
    print("\n" + "=" * 70)
    print(f"Pass 2: guided with {args.posture_instructions}")
    print("=" * 70)
    guided_out = run_single_pass(
        model=model,
        diffusion=diffusion,
        data=data,
        motion_shape=motion_shape,
        model_kwargs=copy.deepcopy(model_kwargs),
        init_noise=init_noise,
        posture_instructions=args.posture_instructions,
        fk_fn=fk_fn,
        posture_lbfgs_steps=args.posture_lbfgs_steps,
        posture_lr=args.posture_lr,
        seed=args.seed,
    )

    # =========================================================
    # 保存对照结果
    # =========================================================
    save_dict = {
        # baseline
        "motion_hml":           baseline_out["hml"],
        "motion_hml_tj":        baseline_out["hml_tj"],
        "motion_xyz":           baseline_out["xyz"],
        # guided
        "motion_hml_guided":    guided_out["hml"],
        "motion_hml_tj_guided": guided_out["hml_tj"],
        "motion_xyz_guided":    guided_out["xyz"],
        # 元数据
        "text_prompt":          text_prompt,
        "posture_instructions": args.posture_instructions,
        "seed":                 args.seed,
        "num_samples":          args.num_samples,
        "motion_length":        args.motion_length,
        "fps":                  fps,
        "guidance_config": {
            "lbfgs_steps": args.posture_lbfgs_steps,
            "lr":          args.posture_lr,
            "variant": GUIDANCE_VARIANT,
            "variant_kwargs": GUIDANCE_KWARGS,
        },
    }

    out_path = os.path.join(out_dir, args.comparison_output)
    np.save(out_path, save_dict, allow_pickle=True)

    print("\n" + "=" * 70)
    print(f"✓ Saved comparison results to {out_path}")
    print("=" * 70)
    print(f"  baseline hml shape: {baseline_out['hml'].shape}")
    print(f"  guided   hml shape: {guided_out['hml'].shape}")
    print(f"  baseline xyz shape: {baseline_out['xyz'].shape}")
    print(f"  guided   xyz shape: {guided_out['xyz'].shape}")
    print()
    print("Next steps:")
    print(f"  python -m new.quantitative_compare {out_path}")
    print(f"  python -m new.visualize_compare    {out_path}")


if __name__ == "__main__":
    main()