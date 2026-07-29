#!/usr/bin/env python3
"""scripts/run_seed_batch.py -- batch mode: one model load, N seeds.

Avoids the argparse collision by NOT calling generate_args().
All MDM args are set directly on a Namespace object.
"""

import os, sys, json, copy, time, argparse
from pathlib import Path
import numpy as np
import torch
from argparse import Namespace

PROJ = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJ))

# ---- Parse custom args ----
ap = argparse.ArgumentParser()
ap.add_argument("--model_path", required=True)
ap.add_argument("--text_prompt", default="a person is walking forward")
ap.add_argument("--seeds", required=True)
ap.add_argument("--motion_length", type=float, default=6.0)
ap.add_argument("--output_dir", required=True)
ap.add_argument("--posture_instructions", nargs="+", default=["anterior_pelvic_tilt"])
ap.add_argument("--variant", default="v2_dps")
ap.add_argument("--variant_kwargs_json", default="{}")
ap.add_argument("--guidance_mode", default="joint")
# Muscle guidance args
ap.add_argument("--muscle_ckpt", default="")
ap.add_argument("--muscle_posture", default="anterior_pelvic_tilt")
ap.add_argument("--muscle_assets_dir", default="motion2muscle")
ap.add_argument("--joint_weight", type=float, default=1.0)
ap.add_argument("--muscle_weight", type=float, default=1.0)
ap.add_argument("--device", default="cuda")
args = ap.parse_args()

OUT = Path(args.output_dir)
OUT.mkdir(parents=True, exist_ok=True)
SEEDS = [int(s.strip()) for s in args.seeds.split(",")]
VARIANT_KWARGS = json.loads(args.variant_kwargs_json)
POSTURE = args.posture_instructions or None
POSTURE_TAG = POSTURE[0].replace("_", "-")[:30] if POSTURE else "none"
VNAME = args.variant.replace("_", "-")

print(f"[batch] variant={args.variant}  posture={POSTURE_TAG}  "
      f"seeds={SEEDS}  n={len(SEEDS)}  out={OUT}", flush=True)

# ---- Build gen_args manually (avoids generate_args sys.argv conflict) ----
# Load model checkpoint args.json for architecture params
model_dir = Path(args.model_path).parent
with open(model_dir / "args.json") as f:
    model_args = json.load(f)

gen_args = Namespace()
gen_args.dataset = "humanml"
gen_args.data_dir = str(PROJ / "dataset" / "HumanML3D")
gen_args.arch = model_args.get("arch", "trans_dec")
gen_args.text_encoder_type = model_args.get("text_encoder_type", "bert")
gen_args.emb_trans_dec = model_args.get("emb_trans_dec", False)
gen_args.layers = model_args.get("layers", 8)
gen_args.latent_dim = model_args.get("latent_dim", 512)
gen_args.cond_mask_prob = model_args.get("cond_mask_prob", 0.1)
gen_args.mask_frames = False
gen_args.lambda_rcxyz = 0.0
gen_args.lambda_vel = 0.0
gen_args.lambda_fc = 0.0
gen_args.lambda_target_loc = 0.0
gen_args.unconstrained = model_args.get("unconstrained", False)
gen_args.pos_embed_max_len = 200
gen_args.use_ema = True
gen_args.model_path = args.model_path
gen_args.output_dir = str(OUT)
gen_args.batch_size = 1
gen_args.train_platform_type = "NoPlatform"
gen_args.cuda = True
gen_args.device = 0
gen_args.seed = 42
gen_args.num_samples = 1
gen_args.num_repetitions = 1
gen_args.guidance_param = 1.0
gen_args.autoregressive = False
gen_args.autoregressive_include_prefix = False
gen_args.autoregressive_init = "data"
gen_args.motion_length = args.motion_length
gen_args.input_text = ""
gen_args.dynamic_text_path = ""
gen_args.action_file = ""
gen_args.text_prompt = args.text_prompt
gen_args.action_name = ""
gen_args.noise_schedule = "cosine"
gen_args.diffusion_steps = 50
gen_args.sigma_small = True
gen_args.context_len = 0
gen_args.pred_len = 0
gen_args.multi_target_cond = False
gen_args.multi_encoder_type = "single"
gen_args.target_enc_layers = 1
gen_args.external_mode = False
gen_args.target_joint_names = "DIMP_FINAL"
gen_args.overwrite = False
gen_args.save_dir = str(OUT)

# Set cond_mode based on text/action presence
gen_args.cond_mode = "text"

from utils.fixseed import fixseed
from utils.model_util import create_model_and_diffusion, load_saved_model
from utils import dist_util
from utils.sampler_util import ClassifierFreeSampleModel
from data_loaders.get_data import get_dataset_loader
from data_loaders.humanml.scripts.motion_process import recover_from_ric
from data_loaders.tensors import collate
from posture_guidance.mdm_integration import make_fk_fn

# ---- Setup device/dataloader/model ----
dist_util.setup_dist(0)  # GPU 0
device = dist_util.dev()
n_frames = min(196, int(args.motion_length * 20))

data = get_dataset_loader(name="humanml", batch_size=1, num_frames=196,
                          split="test", hml_mode="text_only")
data.fixed_length = n_frames

print("[batch] Loading model ...", flush=True)
t0 = time.time()
model, diffusion = create_model_and_diffusion(gen_args, data)
load_saved_model(model, args.model_path, use_avg=True)
if gen_args.guidance_param != 1:
    model = ClassifierFreeSampleModel(model)
model.to(device)
model.eval()
print(f"[batch] Model loaded in {time.time()-t0:.1f}s", flush=True)

n_joints = 22
fk_fn = make_fk_fn(t2m_dataset=data.dataset.t2m_dataset, n_joints=n_joints)
motion_shape = (1, model.njoints, model.nfeats, n_frames)

# ---- Muscle guidance setup (lazy, only if muscle/both mode) ----
muscle_guidance = None
want_muscle = args.guidance_mode in ("muscle", "both") and args.muscle_ckpt
if want_muscle:
    from muscle_guidance_mdm import build_muscle_guidance
    t2m = data.dataset.t2m_dataset
    mdm_mean_t = torch.tensor(t2m.mean, dtype=torch.float32)
    mdm_std_t = torch.tensor(t2m.std, dtype=torch.float32)
    muscle_guidance = build_muscle_guidance(
        ckpt_path=args.muscle_ckpt,
        posture_name=args.muscle_posture,
        assets_dir=args.muscle_assets_dir,
        mdm_mean=mdm_mean_t, mdm_std=mdm_std_t,
        same_normalization=True,
        device=device,
    )
    print("[batch] Muscle guidance loaded", flush=True)

# ---- Precompute model_kwargs ----
collate_args = [{"inp": torch.zeros(n_frames), "tokens": None, "lengths": n_frames}]
collate_args = [dict(arg, text=t) for arg, t in zip(collate_args, [args.text_prompt])]
_, model_kwargs = collate(collate_args)
model_kwargs["y"] = {k: (v.to(device) if torch.is_tensor(v) else v)
                      for k, v in model_kwargs["y"].items()}
if gen_args.guidance_param != 1:
    model_kwargs["y"]["scale"] = torch.ones(1, device=device) * gen_args.guidance_param
if "text" in model_kwargs["y"]:
    model_kwargs["y"]["text_embed"] = model.encode_text(model_kwargs["y"]["text"])

# ---- Precompute noises ----
noises = {}
for seed in SEEDS:
    fixseed(seed)
    noises[seed] = torch.randn(*motion_shape, device=device)


def run_pass(init_noise, seed, posture_inst):
    fixseed(seed)
    os.environ["GUIDANCE_VARIANT"] = args.variant
    os.environ["GUIDANCE_KWARGS_JSON"] = args.variant_kwargs_json
    os.environ["GUIDANCE_MODE"] = args.guidance_mode
    hml = diffusion.p_sample_loop(
        model, motion_shape, clip_denoised=False,
        model_kwargs=copy.deepcopy(model_kwargs),
        skip_timesteps=0, init_image=None, progress=False,
        dump_steps=None, noise=init_noise.clone(), const_noise=False,
        posture_instructions=posture_inst,
        posture_lbfgs_steps=5, posture_lr=0.05, posture_fk_fn=fk_fn,
        muscle_guidance=None, guidance_mode=args.guidance_mode,
        joint_weight=1.0, muscle_weight=1.0,
    )
    sample = data.dataset.t2m_dataset.inv_transform(hml.cpu().permute(0,2,3,1)).float()
    sample = recover_from_ric(sample, n_joints)
    sample_xyz = sample.view(-1, *sample.shape[2:]).permute(0,2,3,1)
    return {"hml": hml.detach().cpu().numpy(),
            "hml_tj": hml.detach().cpu().numpy().squeeze(2).transpose(0,2,1),
            "xyz": sample_xyz.cpu().numpy()}


# ---- Run ----
done, ok = 0, 0
for seed in SEEDS:
    out_dir = OUT / f"{POSTURE_TAG}__{VNAME}_joint_seed{seed}"
    if (out_dir / "comparison.npy").exists():
        print(f"  [skip] seed={seed}", flush=True)
        done += 1; continue
    t1 = time.time()
    try:
        bl = run_pass(noises[seed], seed, None)
        gd = run_pass(noises[seed], seed, POSTURE)
        save_dict = {
            "motion_hml": bl["hml"], "motion_hml_tj": bl["hml_tj"],
            "motion_xyz": bl["xyz"],
            "motion_hml_guided": gd["hml"], "motion_hml_tj_guided": gd["hml_tj"],
            "motion_xyz_guided": gd["xyz"],
            "text_prompt": args.text_prompt,
            "posture_instructions": [POSTURE] if POSTURE else [],
            "seed": seed, "num_samples": 1,
            "motion_length": args.motion_length, "fps": 20,
            "guidance_mode": args.guidance_mode,
            "guidance_config": {"variant": args.variant,
                "variant_kwargs": VARIANT_KWARGS,
                "joint_weight": 1.0, "muscle_weight": 1.0},
        }
        out_dir.mkdir(parents=True, exist_ok=True)
        np.save(out_dir / "comparison.npy", save_dict, allow_pickle=True)
        print(f"  [OK] seed={seed:<5d}  ({time.time()-t1:.0f}s)  [{done+1}/{len(SEEDS)}]",
              flush=True)
        ok += 1
    except Exception as e:
        print(f"  [FAIL] seed={seed}: {e}", flush=True)
        import traceback; traceback.print_exc()
    done += 1

print(f"[batch] DONE: {ok}/{len(SEEDS)} seeds OK, failed/skipped={done-ok}", flush=True)