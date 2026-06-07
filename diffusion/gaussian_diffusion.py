# This code is based on https://github.com/openai/guided-diffusion
"""
This code started out as a PyTorch port of Ho et al's diffusion models:
https://github.com/hojonathanho/diffusion/blob/1e0dceb3b3495bbe19116a5e1b3596cd0706c543/diffusion_tf/diffusion_utils_2.py

Docstrings have been added, as well as DDIM sampling and a new collection of beta schedules.

----------------------------------------------------------------------
[MODIFIED] Posture-guidance variant dispatch
----------------------------------------------------------------------
This file has been extended to support 5 IK-style guidance variants
(see posture_guidance/guidance_variants.py for the implementations):

  V1  v1_mu_sgd       — SGD on mu_t (current method, default for backward
                        compatibility)
  V2  v2_dps          — DPS-style: gradient flows through MDM via x_0_hat
  V3  v3_x0_direct    — direct edit of predicted x_0_hat
  V4  v4_omni         — OmniControl-style dynamic K_e<<K_l on mu_t
  V5  v5_lgd          — LGD-style: DPS + Monte Carlo smoothing

Variant selection is read from environment variable GUIDANCE_VARIANT
(default 'v1_mu_sgd') OR from posture_instructions kwargs of
p_sample_loop. JSON kwargs come from GUIDANCE_KWARGS_JSON.

If posture_instructions is None, NONE of the new code paths fire,
keeping the file fully backward-compatible with non-guidance runs.
"""

import enum
import json
import math
import os

import numpy as np
import torch
import torch as th
from copy import deepcopy
from diffusion.nn import mean_flat, sum_flat
from diffusion.losses import normal_kl, discretized_gaussian_log_likelihood
from data_loaders.humanml.scripts import motion_process
from utils.loss_util import masked_l2, masked_goal_l2
from data_loaders.humanml.scripts.motion_process import get_target_location
# 新增
from posture_guidance.controller import PostureGuidance
from posture_guidance.mdm_integration import apply_posture_guidance
from data_loaders.humanml.scripts.motion_process import recover_from_ric


def get_named_beta_schedule(schedule_name, num_diffusion_timesteps, scale_betas=1.):
    """
    Get a pre-defined beta schedule for the given name.
    """
    if schedule_name == "linear":
        scale = scale_betas * 1000 / num_diffusion_timesteps
        beta_start = scale * 0.0001
        beta_end = scale * 0.02
        return np.linspace(
            beta_start, beta_end, num_diffusion_timesteps, dtype=np.float64
        )
    elif schedule_name == "cosine":
        return betas_for_alpha_bar(
            num_diffusion_timesteps,
            lambda t: math.cos((t + 0.008) / 1.008 * math.pi / 2) ** 2,
        )
    else:
        raise NotImplementedError(f"unknown beta schedule: {schedule_name}")


def betas_for_alpha_bar(num_diffusion_timesteps, alpha_bar, max_beta=0.999):
    betas = []
    for i in range(num_diffusion_timesteps):
        t1 = i / num_diffusion_timesteps
        t2 = (i + 1) / num_diffusion_timesteps
        betas.append(min(1 - alpha_bar(t2) / alpha_bar(t1), max_beta))
    return np.array(betas)


class ModelMeanType(enum.Enum):
    PREVIOUS_X = enum.auto()
    START_X = enum.auto()
    EPSILON = enum.auto()


class ModelVarType(enum.Enum):
    LEARNED = enum.auto()
    FIXED_SMALL = enum.auto()
    FIXED_LARGE = enum.auto()
    LEARNED_RANGE = enum.auto()


class LossType(enum.Enum):
    MSE = enum.auto()
    RESCALED_MSE = enum.auto()
    KL = enum.auto()
    RESCALED_KL = enum.auto()

    def is_vb(self):
        return self == LossType.KL or self == LossType.RESCALED_KL


# ============================================================
# Helper to read guidance variant config from env
# ============================================================
def _read_variant_config_from_env():
    """Returns (variant_name, variant_kwargs_dict, diagnostic_bool)."""
    variant = os.environ.get("GUIDANCE_VARIANT", "v1_mu_sgd")
    kw_json = os.environ.get("GUIDANCE_KWARGS_JSON", "{}")
    try:
        variant_kwargs = json.loads(kw_json)
    except json.JSONDecodeError:
        print(f"[WARN] failed to parse GUIDANCE_KWARGS_JSON='{kw_json}', using {{}}")
        variant_kwargs = {}
    diagnostic = os.environ.get("GUIDANCE_DIAGNOSTIC", "0") == "1"
    return variant, variant_kwargs, diagnostic


def _read_muscle_config_from_env():
    """
    组合 loss 的模式与权重配置（与 GUIDANCE_VARIANT 同风格）。
    Returns (mode, joint_weight, muscle_weight)。
        GUIDANCE_MODE  : "joint" | "muscle" | "both"（默认 joint，行为同历史）
        JOINT_WEIGHT   : 关节项权重（默认 1.0）
        MUSCLE_WEIGHT  : 肌肉项权重（默认 1.0）
    显式传入的 CLI 值（经 p_sample_loop 参数）优先于 env；这里只在 env 设置了才覆盖。
    """
    mode = os.environ.get("GUIDANCE_MODE", None)
    if mode is not None:
        mode = mode.lower()
    def _f(name):
        v = os.environ.get(name, None)
        return None if v is None else float(v)
    return mode, _f("JOINT_WEIGHT"), _f("MUSCLE_WEIGHT")


class GaussianDiffusion:
    """
    Utilities for training and sampling diffusion models.
    """

    def __init__(
        self,
        *,
        betas,
        model_mean_type,
        model_var_type,
        loss_type,
        rescale_timesteps=False,
        lambda_rcxyz=0.,
        lambda_vel=0.,
        lambda_pose=1.,
        lambda_orient=1.,
        lambda_loc=1.,
        data_rep='rot6d',
        lambda_root_vel=0.,
        lambda_vel_rcxyz=0.,
        lambda_fc=0.,
        lambda_target_loc=0.,
        **kargs,
    ):
        self.model_mean_type = model_mean_type
        self.model_var_type = model_var_type
        self.loss_type = loss_type
        self.rescale_timesteps = rescale_timesteps
        self.data_rep = data_rep

        if data_rep != 'rot_vel' and lambda_pose != 1.:
            raise ValueError('lambda_pose is relevant only when training on velocities!')
        self.lambda_pose = lambda_pose
        self.lambda_orient = lambda_orient
        self.lambda_loc = lambda_loc

        self.lambda_rcxyz = lambda_rcxyz
        self.lambda_target_loc = lambda_target_loc
        self.lambda_vel = lambda_vel
        self.lambda_root_vel = lambda_root_vel
        self.lambda_vel_rcxyz = lambda_vel_rcxyz
        self.lambda_fc = lambda_fc

        if self.lambda_rcxyz > 0. or self.lambda_vel > 0. or self.lambda_root_vel > 0. or \
                self.lambda_vel_rcxyz > 0. or self.lambda_fc > 0. or self.lambda_target_loc > 0.:
            assert self.loss_type == LossType.MSE, 'Geometric losses are supported by MSE loss type only!'

        betas = np.array(betas, dtype=np.float64)
        self.betas = betas
        assert len(betas.shape) == 1, "betas must be 1-D"
        assert (betas > 0).all() and (betas <= 1).all()

        self.num_timesteps = int(betas.shape[0])

        alphas = 1.0 - betas
        self.alphas_cumprod = np.cumprod(alphas, axis=0)
        self.alphas_cumprod_prev = np.append(1.0, self.alphas_cumprod[:-1])
        self.alphas_cumprod_next = np.append(self.alphas_cumprod[1:], 0.0)
        assert self.alphas_cumprod_prev.shape == (self.num_timesteps,)

        self.sqrt_alphas_cumprod = np.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = np.sqrt(1.0 - self.alphas_cumprod)
        self.log_one_minus_alphas_cumprod = np.log(1.0 - self.alphas_cumprod)
        self.sqrt_recip_alphas_cumprod = np.sqrt(1.0 / self.alphas_cumprod)
        self.sqrt_recipm1_alphas_cumprod = np.sqrt(1.0 / self.alphas_cumprod - 1)

        self.posterior_variance = (
            betas * (1.0 - self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
        )
        self.posterior_log_variance_clipped = np.log(
            np.append(self.posterior_variance[1], self.posterior_variance[1:])
        )
        self.posterior_mean_coef1 = (
            betas * np.sqrt(self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
        )
        self.posterior_mean_coef2 = (
            (1.0 - self.alphas_cumprod_prev)
            * np.sqrt(alphas)
            / (1.0 - self.alphas_cumprod)
        )

        self.masked_l2 = masked_l2

    def q_mean_variance(self, x_start, t):
        mean = (
            _extract_into_tensor(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start
        )
        variance = _extract_into_tensor(1.0 - self.alphas_cumprod, t, x_start.shape)
        log_variance = _extract_into_tensor(
            self.log_one_minus_alphas_cumprod, t, x_start.shape
        )
        return mean, variance, log_variance

    def q_sample(self, x_start, t, noise=None):
        if noise is None:
            noise = th.randn_like(x_start)
        assert noise.shape == x_start.shape
        return (
            _extract_into_tensor(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start
            + _extract_into_tensor(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape)
            * noise
        )

    def q_posterior_mean_variance(self, x_start, x_t, t):
        assert x_start.shape == x_t.shape
        posterior_mean = (
            _extract_into_tensor(self.posterior_mean_coef1, t, x_t.shape) * x_start
            + _extract_into_tensor(self.posterior_mean_coef2, t, x_t.shape) * x_t
        )
        posterior_variance = _extract_into_tensor(self.posterior_variance, t, x_t.shape)
        posterior_log_variance_clipped = _extract_into_tensor(
            self.posterior_log_variance_clipped, t, x_t.shape
        )
        assert (
            posterior_mean.shape[0]
            == posterior_variance.shape[0]
            == posterior_log_variance_clipped.shape[0]
            == x_start.shape[0]
        )
        return posterior_mean, posterior_variance, posterior_log_variance_clipped

    def p_mean_variance(
        self, model, x, t, clip_denoised=True, denoised_fn=None, model_kwargs=None
    ):
        if model_kwargs is None:
            model_kwargs = {}

        B, C = x.shape[:2]
        assert t.shape == (B,)
        model_output = model(x, self._scale_timesteps(t), **model_kwargs)

        if 'inpainting_mask' in model_kwargs['y'].keys() and 'inpainted_motion' in model_kwargs['y'].keys():
            inpainting_mask, inpainted_motion = model_kwargs['y']['inpainting_mask'], model_kwargs['y']['inpainted_motion']
            assert self.model_mean_type == ModelMeanType.START_X, 'This feature supports only X_start pred for mow!'
            assert model_output.shape == inpainting_mask.shape == inpainted_motion.shape
            model_output = (model_output * ~inpainting_mask) + (inpainted_motion * inpainting_mask)

        if self.model_var_type in [ModelVarType.LEARNED, ModelVarType.LEARNED_RANGE]:
            assert model_output.shape == (B, C * 2, *x.shape[2:])
            model_output, model_var_values = th.split(model_output, C, dim=1)
            if self.model_var_type == ModelVarType.LEARNED:
                model_log_variance = model_var_values
                model_variance = th.exp(model_log_variance)
            else:
                min_log = _extract_into_tensor(
                    self.posterior_log_variance_clipped, t, x.shape
                )
                max_log = _extract_into_tensor(np.log(self.betas), t, x.shape)
                frac = (model_var_values + 1) / 2
                model_log_variance = frac * max_log + (1 - frac) * min_log
                model_variance = th.exp(model_log_variance)
        else:
            model_variance, model_log_variance = {
                ModelVarType.FIXED_LARGE: (
                    np.append(self.posterior_variance[1], self.betas[1:]),
                    np.log(np.append(self.posterior_variance[1], self.betas[1:])),
                ),
                ModelVarType.FIXED_SMALL: (
                    self.posterior_variance,
                    self.posterior_log_variance_clipped,
                ),
            }[self.model_var_type]

            model_variance = _extract_into_tensor(model_variance, t, x.shape)
            model_log_variance = _extract_into_tensor(model_log_variance, t, x.shape)

        def process_xstart(x):
            if denoised_fn is not None:
                x = denoised_fn(x)
            if clip_denoised:
                return x.clamp(-1, 1)
            return x

        if self.model_mean_type == ModelMeanType.PREVIOUS_X:
            pred_xstart = process_xstart(
                self._predict_xstart_from_xprev(x_t=x, t=t, xprev=model_output)
            )
            model_mean = model_output
        elif self.model_mean_type in [ModelMeanType.START_X, ModelMeanType.EPSILON]:
            if self.model_mean_type == ModelMeanType.START_X:
                pred_xstart = process_xstart(model_output)
            else:
                pred_xstart = process_xstart(
                    self._predict_xstart_from_eps(x_t=x, t=t, eps=model_output)
                )
            model_mean, _, _ = self.q_posterior_mean_variance(
                x_start=pred_xstart, x_t=x, t=t
            )
        else:
            raise NotImplementedError(self.model_mean_type)

        assert (
            model_mean.shape == model_log_variance.shape == pred_xstart.shape == x.shape
        )
        return {
            "mean": model_mean,
            "variance": model_variance,
            "log_variance": model_log_variance,
            "pred_xstart": pred_xstart,
        }

    def _predict_xstart_from_eps(self, x_t, t, eps):
        assert x_t.shape == eps.shape
        return (
            _extract_into_tensor(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t
            - _extract_into_tensor(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape) * eps
        )

    def _predict_xstart_from_xprev(self, x_t, t, xprev):
        assert x_t.shape == xprev.shape
        return (
            _extract_into_tensor(1.0 / self.posterior_mean_coef1, t, x_t.shape) * xprev
            - _extract_into_tensor(
                self.posterior_mean_coef2 / self.posterior_mean_coef1, t, x_t.shape
            )
            * x_t
        )

    def _predict_eps_from_xstart(self, x_t, t, pred_xstart):
        return (
            _extract_into_tensor(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t
            - pred_xstart
        ) / _extract_into_tensor(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape)

    def _scale_timesteps(self, t):
        if self.rescale_timesteps:
            return t.float() * (1000.0 / self.num_timesteps)
        return t

    def condition_mean(self, cond_fn, p_mean_var, x, t, model_kwargs=None):
        gradient = cond_fn(x, self._scale_timesteps(t), **model_kwargs)
        new_mean = (
            p_mean_var["mean"].float() + p_mean_var["variance"] * gradient.float()
        )
        return new_mean

    def condition_mean_with_grad(self, cond_fn, p_mean_var, x, t, model_kwargs=None):
        gradient = cond_fn(x, t, p_mean_var, **model_kwargs)
        new_mean = (
            p_mean_var["mean"].float() + p_mean_var["variance"] * gradient.float()
        )
        return new_mean

    def condition_score(self, cond_fn, p_mean_var, x, t, model_kwargs=None):
        alpha_bar = _extract_into_tensor(self.alphas_cumprod, t, x.shape)
        eps = self._predict_eps_from_xstart(x, t, p_mean_var["pred_xstart"])
        eps = eps - (1 - alpha_bar).sqrt() * cond_fn(
            x, self._scale_timesteps(t), **model_kwargs
        )
        out = p_mean_var.copy()
        out["pred_xstart"] = self._predict_xstart_from_eps(x, t, eps)
        out["mean"], _, _ = self.q_posterior_mean_variance(
            x_start=out["pred_xstart"], x_t=x, t=t
        )
        return out

    def condition_score_with_grad(self, cond_fn, p_mean_var, x, t, model_kwargs=None):
        alpha_bar = _extract_into_tensor(self.alphas_cumprod, t, x.shape)
        eps = self._predict_eps_from_xstart(x, t, p_mean_var["pred_xstart"])
        eps = eps - (1 - alpha_bar).sqrt() * cond_fn(
            x, t, p_mean_var, **model_kwargs
        )
        out = p_mean_var.copy()
        out["pred_xstart"] = self._predict_xstart_from_eps(x, t, eps)
        out["mean"], _, _ = self.q_posterior_mean_variance(
            x_start=out["pred_xstart"], x_t=x, t=t
        )
        return out

    def p_sample(
        self,
        model,
        x,
        t,
        clip_denoised=True,
        denoised_fn=None,
        cond_fn=None,
        model_kwargs=None,
        const_noise=False,
    ):
        out = self.p_mean_variance(
            model,
            x,
            t,
            clip_denoised=clip_denoised,
            denoised_fn=denoised_fn,
            model_kwargs=model_kwargs,
        )
        noise = th.randn_like(x)
        if const_noise:
            noise = noise[[0]].repeat(x.shape[0], 1, 1, 1)

        nonzero_mask = (
            (t != 0).float().view(-1, *([1] * (len(x.shape) - 1)))
        )
        if cond_fn is not None:
            out["mean"] = self.condition_mean(
                cond_fn, out, x, t, model_kwargs=model_kwargs
            )
        sample = out["mean"] + nonzero_mask * th.exp(0.5 * out["log_variance"]) * noise
        return {"sample": sample, "pred_xstart": out["pred_xstart"]}

    def p_sample_with_grad(
        self,
        model,
        x,
        t,
        clip_denoised=True,
        denoised_fn=None,
        cond_fn=None,
        model_kwargs=None,
    ):
        with th.enable_grad():
            x = x.detach().requires_grad_()
            out = self.p_mean_variance(
                model,
                x,
                t,
                clip_denoised=clip_denoised,
                denoised_fn=denoised_fn,
                model_kwargs=model_kwargs,
            )
            noise = th.randn_like(x)
            nonzero_mask = (
                (t != 0).float().view(-1, *([1] * (len(x.shape) - 1)))
            )
            if cond_fn is not None:
                out["mean"] = self.condition_mean_with_grad(
                    cond_fn, out, x, t, model_kwargs=model_kwargs
                )
        sample = out["mean"] + nonzero_mask * th.exp(0.5 * out["log_variance"]) * noise
        return {"sample": sample, "pred_xstart": out["pred_xstart"].detach()}

    def p_sample_loop(
        self,
        model,
        shape,
        noise=None,
        clip_denoised=True,
        denoised_fn=None,
        cond_fn=None,
        model_kwargs=None,
        device=None,
        progress=False,
        skip_timesteps=0,
        init_image=None,
        randomize_class=False,
        cond_fn_with_grad=False,
        dump_steps=None,
        const_noise=False,
        posture_instructions=None,
        posture_lbfgs_steps=5,
        posture_lr=0.05,
        posture_fk_fn=None,
        muscle_guidance=None,
        guidance_mode=None,
        joint_weight=None,
        muscle_weight=None,
    ):
        final = None
        if dump_steps is not None:
            dump = []

        if 'text' in model_kwargs['y'].keys():
            model_kwargs['y']['text_embed'] = model.encode_text(model_kwargs['y']['text'])

        for i, sample in enumerate(self.p_sample_loop_progressive(
            model,
            shape,
            noise=noise,
            clip_denoised=clip_denoised,
            denoised_fn=denoised_fn,
            cond_fn=cond_fn,
            model_kwargs=model_kwargs,
            device=device,
            progress=progress,
            skip_timesteps=skip_timesteps,
            init_image=init_image,
            randomize_class=randomize_class,
            cond_fn_with_grad=cond_fn_with_grad,
            const_noise=const_noise,
            posture_instructions=posture_instructions,
            posture_lbfgs_steps=posture_lbfgs_steps,
            posture_lr=posture_lr,
            posture_fk_fn=posture_fk_fn,
            muscle_guidance=muscle_guidance,
            guidance_mode=guidance_mode,
            joint_weight=joint_weight,
            muscle_weight=muscle_weight,
        )):
            if dump_steps is not None and i in dump_steps:
                dump.append(deepcopy(sample["sample"]))
            final = sample
        if dump_steps is not None:
            return dump
        return final["sample"]

    def p_sample_loop_progressive(
        self,
        model,
        shape,
        noise=None,
        clip_denoised=False,
        denoised_fn=None,
        cond_fn=None,
        model_kwargs=None,
        device=None,
        progress=False,
        skip_timesteps=0,
        init_image=None,
        randomize_class=False,
        cond_fn_with_grad=False,
        dump_steps=None,
        const_noise=False,
        # ---------- 新增的 4 个参数，全部有默认值 ----------
        posture_instructions=None,
        posture_lbfgs_steps=5,
        posture_lr=0.05,
        posture_fk_fn=None,
        # ---------- 肌肉/组合引导参数（全部有默认值，不传等同纯关节）----------
        muscle_guidance=None,
        guidance_mode=None,
        joint_weight=None,
        muscle_weight=None,
    ):
        """
        Generate samples from the model and yield intermediate samples from
        each timestep of diffusion.

        ----------------------------------------------------------------------
        [MODIFIED] Posture-guidance with 5 variant dispatch
        ----------------------------------------------------------------------
        When posture_instructions is non-empty AND posture_fk_fn is not None,
        we apply IK-style guidance at each reverse step. The variant is
        selected by the GUIDANCE_VARIANT environment variable:

          v1_mu_sgd     — SGD on mu_t (default; original behavior)
          v2_dps        — DPS-style: gradient through MDM via x_0_hat
          v3_x0_direct  — direct edit of x_0_hat, then recompose mu_t
          v4_omni       — OmniControl-style dynamic K_e<<K_l on mu_t
          v5_lgd        — LGD-style: DPS + Monte Carlo smoothing

        Per-variant kwargs are read from GUIDANCE_KWARGS_JSON (a JSON dict).

        MDM weights are NEVER touched.
        """
        # -------- 设备、初始噪声、indices 准备 --------
        if device is None:
            device = next(model.parameters()).device
        assert isinstance(shape, (tuple, list))

        if noise is not None:
            img = noise
        else:
            img = th.randn(*shape, device=device)

        if skip_timesteps and init_image is None:
            init_image = th.zeros_like(img)

        indices = list(range(self.num_timesteps - skip_timesteps))[::-1]

        print(f">>> p_sample_loop: total_steps={len(indices)}, "
              f"num_timesteps={self.num_timesteps}")

        if init_image is not None:
            my_t = th.ones([shape[0]], device=device, dtype=th.long) * indices[0]
            img = self.q_sample(init_image, my_t, img)

        if progress:
            from tqdm.auto import tqdm
            indices = tqdm(indices)

        # -------- 解析组合 loss 模式 / 权重（env 覆盖 CLI 默认）--------
        env_mode, env_jw, env_mw = _read_muscle_config_from_env()
        guidance_mode = (env_mode or guidance_mode or "joint").lower()
        joint_weight = env_jw if env_jw is not None else (joint_weight if joint_weight is not None else 1.0)
        muscle_weight = env_mw if env_mw is not None else (muscle_weight if muscle_weight is not None else 1.0)

        has_joint = (posture_instructions is not None
                     and len(posture_instructions) > 0
                     and posture_fk_fn is not None)
        has_muscle = muscle_guidance is not None and guidance_mode in ("muscle", "both")

        # -------- 初始化 guidance --------
        guidance = None
        use_guidance = (has_joint or has_muscle) and posture_fk_fn is not None
        if use_guidance:
            from posture_guidance.controller import PostureGuidance
            # 纯肌肉模式下 instructions 可能为空，PostureGuidance([]) 的 specs 为空、loss≈0
            guidance = PostureGuidance(instructions=posture_instructions or [])

        # 读取 variant 配置（从环境变量）
        variant_name, variant_kwargs, diagnostic = _read_variant_config_from_env()

        if use_guidance:
            print(f">>> [posture-guidance] variant={variant_name}")
            print(f">>> [posture-guidance] kwargs={variant_kwargs}")
            print(f">>> [posture-guidance] diagnostic={diagnostic}")

        T_total = self.num_timesteps

        # -------- 主循环 --------
        for i in indices:
            t = th.tensor([i] * shape[0], device=device)

            if randomize_class and 'y' in model_kwargs:
                model_kwargs['y'] = deepcopy(model_kwargs['y'])
                model_kwargs['y']['class'] = th.randint(
                    low=0,
                    high=model.num_classes,
                    size=model_kwargs['y']['class'].shape,
                    device=model_kwargs['y']['class'].device,
                )

            with th.no_grad():
                sample_fn = (
                    self.p_sample_with_grad if cond_fn_with_grad else self.p_sample
                )
                out = sample_fn(
                    model,
                    img,
                    t,
                    clip_denoised=clip_denoised,
                    denoised_fn=denoised_fn,
                    cond_fn=cond_fn,
                    model_kwargs=model_kwargs,
                    const_noise=const_noise,
                )

            # ============================================================
            # POSTURE GUIDANCE — variant dispatch
            # ============================================================
            if use_guidance and guidance is not None:
                # Compute mu_t from pred_xstart (needed by all variants)
                mu_t, _, model_log_variance = self.q_posterior_mean_variance(
                    x_start=out["pred_xstart"],
                    x_t=img,
                    t=t,
                )
                mu_t = mu_t.contiguous()

                # Dispatch by variant
                mu_t_updated = self._apply_guidance_variant(
                    variant_name=variant_name,
                    variant_kwargs=variant_kwargs,
                    mu_t=mu_t,
                    x_t=img,
                    pred_xstart=out["pred_xstart"],
                    t_int=int(i),
                    t_tensor=t,
                    T_total=T_total,
                    model=model,
                    model_kwargs=model_kwargs,
                    guidance=guidance,
                    posture_fk_fn=posture_fk_fn,
                    legacy_lbfgs_steps=posture_lbfgs_steps,
                    legacy_lr=posture_lr,
                    diagnostic=diagnostic,
                    muscle_guidance=muscle_guidance,
                    muscle_mode=guidance_mode,
                    joint_weight=joint_weight,
                    muscle_weight=muscle_weight,
                )

                # Resample x_{t-1} = mu_t_updated + sigma_t * z
                noise_z = th.randn_like(mu_t_updated)
                nonzero_mask = (
                    (t != 0).float().view(-1, *([1] * (len(mu_t_updated.shape) - 1)))
                )
                out["sample"] = (
                    mu_t_updated
                    + nonzero_mask * th.exp(0.5 * model_log_variance) * noise_z
                )
            # ============================================================

            if dump_steps is not None and i in dump_steps:
                yield out

            yield out
            img = out["sample"]

        # ======== Save conflict log at end of sampling ========
        if os.environ.get("LOG_PRIOR_CONFLICT", "0") == "1" and hasattr(guidance, "_conflict_log") and guidance._conflict_log:
            import json as _json
            log_path = os.environ.get("CONFLICT_LOG_PATH", "/tmp/prior_conflict_log.json")
            os.makedirs(os.path.dirname(log_path) if os.path.dirname(log_path) else ".", exist_ok=True)
            with open(log_path, "w") as _f:
                _json.dump(guidance._conflict_log, _f)
            print(f"[LOG_PRIOR_CONFLICT] saved {len(guidance._conflict_log)} records to {log_path}")
        # =======================================================

    # ============================================================
    # Variant dispatch — internal helper
    # ============================================================
    def _apply_guidance_variant(
        self,
        variant_name,
        variant_kwargs,
        mu_t,
        x_t,
        pred_xstart,
        t_int,
        t_tensor,
        T_total,
        model,
        model_kwargs,
        guidance,
        posture_fk_fn,
        legacy_lbfgs_steps,
        legacy_lr,
        diagnostic,
        muscle_guidance=None,
        muscle_mode="joint",
        joint_weight=1.0,
        muscle_weight=1.0,
    ):
        """
        Dispatch to one of 5 guidance variants. Returns updated mu_t.

        Each variant reads its own kwargs from variant_kwargs dict.
        Variants V2/V3/V5 require model + model_kwargs to compute x0_hat
        with gradients enabled.

        组合 loss（关节 + 肌肉）只接到统一接口 v2_dps 和 v6（见 plan）。其余
        variant（v1/v2b/v3/v4/v5/v2_dps_norm）保持纯关节行为不变。
        """
        # Build a guidance loss closure that all variants share（纯关节，向后兼容）
        def guidance_loss_fn(q, t_in, T_in, spec_schedule_override=None):
            # anchor 保证返回值始终连接到 q（grad_fn 不为 None），
            # 即使所有 target 在当前时间步都不激活（loss 值为 0）。
            anchor = q.sum() * 0.0
            return guidance.compute_loss(q, t_in, T_in, spec_schedule_override=spec_schedule_override) + anchor

        # 组合引导：把关节 + 肌肉两路合成一个可微 motion_loss（v2_dps / v6 用）
        from posture_guidance.combined_loss import CombinedGuidance
        combined = CombinedGuidance(
            posture=guidance,
            muscle=muscle_guidance,
            fk_fn=posture_fk_fn,
            mode=muscle_mode,
            w_joint=joint_weight,
            w_muscle=muscle_weight,
        )

        if variant_name == "v1_mu_sgd":
            mu_new = self._guidance_v1_mu_sgd(
                mu_t=mu_t, t_int=t_int, T=T_total,
                fk_fn=posture_fk_fn,
                guidance_loss_fn=guidance_loss_fn,
                guidance=guidance,
                legacy_lbfgs_steps=legacy_lbfgs_steps,
                legacy_lr=legacy_lr,
                **variant_kwargs,
            )

        elif variant_name == "v2_dps":
            mu_new = self._guidance_v2_dps(
                mu_t=mu_t, x_t=x_t, t_int=t_int, t_tensor=t_tensor, T=T_total,
                model=model, model_kwargs=model_kwargs,
                fk_fn=posture_fk_fn,
                guidance=guidance,
                guidance_loss_fn=guidance_loss_fn,
                combined=combined,
                **variant_kwargs,
            )

        elif variant_name == "v2_x0_edit":
            # V2b: 直接在 x0_hat 上做 SGD，再重组 mu_t（不穿 MDM）
            mu_new = self._guidance_v2b_x0_edit(
                mu_t=mu_t, x_t=x_t, pred_xstart=pred_xstart,
                t_int=t_int, t_tensor=t_tensor, T=T_total,
                fk_fn=posture_fk_fn,
                guidance_loss_fn=guidance_loss_fn,
                **variant_kwargs,
            )

        elif variant_name == "v3_x0_direct":
            mu_new = self._guidance_v3_x0_direct(
                mu_t=mu_t, x_t=x_t, pred_xstart=pred_xstart,
                t_int=t_int, t_tensor=t_tensor, T=T_total,
                fk_fn=posture_fk_fn,
                guidance_loss_fn=guidance_loss_fn,
                **variant_kwargs,
            )

        elif variant_name == "v4_omni":
            mu_new = self._guidance_v4_omni(
                mu_t=mu_t, t_int=t_int, T=T_total,
                fk_fn=posture_fk_fn,
                guidance_loss_fn=guidance_loss_fn,
                **variant_kwargs,
            )

        elif variant_name == "v5_lgd":
            mu_new = self._guidance_v5_lgd(
                mu_t=mu_t, x_t=x_t, t_int=t_int, t_tensor=t_tensor, T=T_total,
                model=model, model_kwargs=model_kwargs,
                fk_fn=posture_fk_fn,
                guidance_loss_fn=guidance_loss_fn,
                **variant_kwargs,
            )

        elif variant_name == "v2_dps_norm":
            # Step 1: gradient-normalized DPS — s 解耦于 ‖∇L‖
            mu_new = self._guidance_v2_dps_norm(
                mu_t=mu_t, x_t=x_t, t_int=t_int, t_tensor=t_tensor, T=T_total,
                model=model, model_kwargs=model_kwargs,
                fk_fn=posture_fk_fn,
                guidance_loss_fn=guidance_loss_fn,
                **variant_kwargs,
            )

        elif variant_name == "v6_closed_loop":
            # Step 2+3: PID closed-loop + manifold projection + huber loss
            mu_new = self._guidance_v6_closed_loop(
                mu_t=mu_t, x_t=x_t, pred_xstart=pred_xstart,
                t_int=t_int, t_tensor=t_tensor, T=T_total,
                model=model, model_kwargs=model_kwargs,
                fk_fn=posture_fk_fn,
                guidance=guidance,
                guidance_loss_fn=guidance_loss_fn,
                combined=combined,
                **variant_kwargs,
            )

        else:
            print(f"[WARN] unknown variant '{variant_name}', falling back to v1_mu_sgd")
            mu_new = self._guidance_v1_mu_sgd(
                mu_t=mu_t, t_int=t_int, T=T_total,
                fk_fn=posture_fk_fn,
                guidance_loss_fn=guidance_loss_fn,
                guidance=guidance,
                legacy_lbfgs_steps=legacy_lbfgs_steps,
                legacy_lr=legacy_lr,
            )

        if diagnostic:
            with th.no_grad():
                delta = (mu_new - mu_t).norm().item()
            print(f"  [t={t_int:3d}] variant={variant_name} |Δμ|={delta:.4f}")

        return mu_new

    # ------------------------------------------------------------
    # V1 — SGD on detached mu_t (current method, default)
    # ------------------------------------------------------------
    def _guidance_v1_mu_sgd(
        self, *, mu_t, t_int, T, fk_fn, guidance_loss_fn, guidance,
        legacy_lbfgs_steps=5, legacy_lr=0.05,
        n_inner_steps=None, lr=None, schedule="final",
        base_weight=30.0,
    ):
        """
        V1: Original behavior — call apply_posture_guidance from
        posture_guidance.mdm_integration. This preserves backward
        compatibility with the existing mdm_integration.py.

        If you supply n_inner_steps / lr in variant_kwargs, they override
        the legacy posture_lbfgs_steps / posture_lr.
        """
        if not _schedule_active(schedule, t_int, T):
            return mu_t

        n_steps = n_inner_steps if n_inner_steps is not None else legacy_lbfgs_steps
        lr_val = lr if lr is not None else legacy_lr

        # Path A: try the existing apply_posture_guidance (preserves V1 exactly)
        try:
            from posture_guidance.mdm_integration import apply_posture_guidance
            return apply_posture_guidance(
                mu_t=mu_t,
                guidance=guidance,
                t=t_int,
                T=T,
                n_lbfgs_steps=n_steps,
                lr=lr_val,
                fk_fn=fk_fn,
            )
        except Exception as e:
            # Path B: fallback to inline SGD if the legacy fn signature changed
            print(f"[V1] legacy apply_posture_guidance failed ({e}), using inline SGD")
            mu_var = mu_t.detach().clone().contiguous().requires_grad_(True)
            optimizer = th.optim.SGD([mu_var], lr=lr_val)
            for _ in range(n_steps):
                optimizer.zero_grad()
                q = fk_fn(mu_var)
                loss = base_weight * guidance_loss_fn(q, t_int, T)
                loss.backward()
                optimizer.step()
            return mu_var.detach()

    # ------------------------------------------------------------
    # V2 — DPS: gradient flows through MDM via x_0_hat
    # ------------------------------------------------------------
    # ------------------------------------------------------------
    # V2a — DPS-style: gradient through MDM, update mu_t
    # 标准 DPS：∇_{x_t} L 穿过 MDM 反传，更新 mu_t
    # ------------------------------------------------------------
    def _guidance_v2_dps(
        self, *, mu_t, x_t, t_int, t_tensor, T,
        model, model_kwargs, fk_fn, guidance_loss_fn,
        guidance=None, combined=None,
        s=30.0, schedule="last_quarter", base_weight=1.0,
        spec_schedule_override=None,
        manifold_project=False, manifold_alpha=1.0,
        loss_form="hinge", huber_delta=0.05,
        sigma_schedule="constant",  # "constant" | "linear_decay" | "sqrt_decay"
    ):
        """
        V2a: DPS-style (Chung et al. NeurIPS 2022).

        梯度路径：x_t → MDM → x0_hat → FK → loss （穿过 MDM 网络）
        更新对象：mu_t
        mu_t_new = mu_t - s * grad_{x_t} L

        关键参数说明：
            s            : 步长。直接控制每步推多远，需要较大值（默认 30）
                           以对抗 MDM 的自愈效应。
            schedule     : 建议用 'last_quarter' 或 'final'。
                           'always' 在高噪声步（t>12）loss=0，grad=0，白跑一次 MDM forward。
            base_weight  : 在 guidance_loss_fn 里已经乘了，这里保持 1.0。
            manifold_project : 【消融用】是否对 grad 做 MPGD 正交投影（剥离 x0_hat 方向分量）。
                           默认 False = 原始 V2 行为。设 True 可单独隔离"流形投影"这一项的
                           贡献，用于 V6 vs V2 的受控消融（其余 PID/Huber/平滑均不引入）。
            manifold_alpha : 投影强度 ∈ [0,1]，1.0 = 完全投影。

        与 V1 的效果差异根因：
            V1 用 15 步 inner loop，每步推 lr * base_weight * grad；
            V2 只有 1 步但梯度穿过 MDM（全 263 维），用大 s 补偿。
        """
        if not _schedule_active(schedule, t_int, T):
            return mu_t

        with th.enable_grad():
            x_t_var = x_t.detach().clone().contiguous().requires_grad_(True)

            model_output = model(x_t_var, self._scale_timesteps(t_tensor), **model_kwargs)

            if model_output.grad_fn is None:
                if hasattr(model, "model"):
                    model_output = model.model(
                        x_t_var, self._scale_timesteps(t_tensor), **model_kwargs
                    )
                if model_output.grad_fn is None:
                    print(f"[V2 SKIP t={t_int}] model_output.grad_fn=None")
                    return mu_t

            if self.model_mean_type == ModelMeanType.START_X:
                x0_hat = model_output
            elif self.model_mean_type == ModelMeanType.EPSILON:
                x0_hat = self._predict_xstart_from_eps(x_t_var, t_tensor, model_output)
            else:
                return mu_t

            q = fk_fn(x0_hat)
            if q.grad_fn is None:
                print(f"[V2 SKIP t={t_int}] q.grad_fn=None")
                return mu_t

            # 将外层 schedule 透传给 controller，保持内外一致
            _spec_sched = spec_schedule_override if spec_schedule_override is not None else schedule
            if combined is not None:
                # 统一接口：组合 loss = w_joint·关节 − w_muscle·肌肉（梯度均穿过 MDM）
                loss = base_weight * combined.motion_loss(
                    x0_hat, t_int, T,
                    loss_form=loss_form,
                    huber_delta=huber_delta,
                    spec_schedule_override=_spec_sched,
                )
                muscle_active = combined.muscle_active
            else:
                # 向后兼容（无组合对象时）：纯关节
                if loss_form == "huber" and guidance is not None:
                    anchor = q.sum() * 0.0
                    loss = base_weight * guidance.compute_loss(
                        q, t_int, T,
                        loss_form="huber",
                        huber_delta=huber_delta,
                        spec_schedule_override=_spec_sched,
                    ) + anchor
                else:
                    loss = base_weight * guidance_loss_fn(q, t_int, T, spec_schedule_override=_spec_sched)
                muscle_active = False
            if loss.grad_fn is None:
                return mu_t

            # loss=0 说明当前 x0_hat 预测的角度已满足约束，无需推（仅 hinge 且无肌肉项时可早退）
            if loss_form == "hinge" and not muscle_active and float(loss.detach()) == 0.0:
                return mu_t

            grad = th.autograd.grad(loss, x_t_var)[0]

        grad = grad.detach()

        # 【消融用】MPGD 流形正交投影：剥离梯度中沿 x0_hat 的分量
        # 与 V6 使用同一个 orthogonal_project，确保对比口径一致
        if manifold_project:
            from posture_guidance.closed_loop_controller import orthogonal_project
            grad = orthogonal_project(grad, x0_hat.detach(), alpha=manifold_alpha)

        # σ-aware step size: scale s by denoising progress
        if sigma_schedule != "constant":
            sigma_t = float(self.posterior_variance[t_int]) ** 0.5
            sigma_max = float(max(self.posterior_variance)) ** 0.5
            sigma_min = float(min(v for v in self.posterior_variance if v > 0)) ** 0.5
            sigma_range = max(sigma_max - sigma_min, 1e-6)
            progress = max(0.0, min(1.0, (sigma_max - sigma_t) / sigma_range))
            if sigma_schedule == "linear_decay":
                c_t = progress
            elif sigma_schedule == "sqrt_decay":
                c_t = progress ** 0.5
            else:
                c_t = 1.0
            s_eff = s * c_t
        else:
            s_eff = s

        loss_val  = loss.item()
        delta_mu  = (s_eff * grad).norm().item()
        grad_norm = grad.norm().item()
        print(f"[V2 UPDATE t={t_int:3d}] loss={loss_val:.4f}  "
              f"grad_norm={grad_norm:.5f}  |s*grad|={delta_mu:.4f}  s_eff={s_eff:.2f}  "
              f"loss_form={loss_form}  sigma_sched={sigma_schedule}  mproj={manifold_project}")

        mu_t_new = mu_t.detach() - s_eff * grad
        return mu_t_new

    # ------------------------------------------------------------
    # V2b — x0 direct edit: SGD on x0_hat, recompose mu_t
    # 直接在 MDM 预测的 x0_hat 上做 SGD，再重组 mu_t
    # ------------------------------------------------------------
    def _guidance_v2b_x0_edit(
        self, *, mu_t, x_t, pred_xstart, t_int, t_tensor, T,
        fk_fn, guidance_loss_fn,
        n_inner_steps=15, lr=0.5, schedule="always", base_weight=30.0,
    ):
        """
        V2b: direct x0 editing.

        梯度路径：x0_hat → FK → loss（不穿 MDM）
        更新对象：x0_hat（SGD），再用 q_posterior_mean_variance 重组 mu_t

        mu_t_new = coef1 * x0_hat_updated + coef2 * x_t

        与 V2a 的区别：
            - 不需要重新跑 MDM forward（更快）
            - 梯度只覆盖 FK 可微的维度（约 24%），其余维度靠重组传播
            - 每步多次迭代（inner loop），比 V2a 的单步梯度更激进
        """
        if not _schedule_active(schedule, t_int, T):
            return mu_t

        # 从 pred_xstart 出发（p_sample 已经算好的 MDM 预测值）
        x0_var = pred_xstart.detach().clone().contiguous().requires_grad_(True)
        optimizer = th.optim.SGD([x0_var], lr=lr)

        for _ in range(n_inner_steps):
            optimizer.zero_grad()
            q = fk_fn(x0_var)
            loss = base_weight * guidance_loss_fn(q, t_int, T)
            if loss.grad_fn is None:
                break
            loss.backward()
            optimizer.step()

        # 用更新后的 x0_hat 重组 mu_t
        x0_delta = (x0_var.detach() - pred_xstart).norm().item()
        print(f"[V2b UPDATE t={t_int:3d}] x0_delta={x0_delta:.5f}  "
              f"steps={n_inner_steps}  lr={lr}")

        mu_t_new, _, _ = self.q_posterior_mean_variance(
            x_start=x0_var.detach(),
            x_t=x_t,
            t=t_tensor,
        )
        return mu_t_new

    # ------------------------------------------------------------
    # V3 — direct edit of x_0_hat, then recompose mu_t
    # ------------------------------------------------------------
    def _guidance_v3_x0_direct(
        self, *, mu_t, x_t, pred_xstart, t_int, t_tensor, T,
        fk_fn, guidance_loss_fn,
        n_inner_steps=5, lr=0.05, schedule="always", base_weight=1.0,
    ):
        """
        V3: Direct gradient descent on x_0_hat, then recompose mu_t via
        posterior_mean_coef1 * x0 + posterior_mean_coef2 * x_t.
        """
        if not _schedule_active(schedule, t_int, T):
            return mu_t

        x0_var = pred_xstart.detach().clone().contiguous().requires_grad_(True)
        optimizer = th.optim.SGD([x0_var], lr=lr)
        for _ in range(n_inner_steps):
            optimizer.zero_grad()
            q = fk_fn(x0_var)
            loss = base_weight * guidance_loss_fn(q, t_int, T)
            loss.backward()
            optimizer.step()

        # Recompose mu_t with the modified x0
        mu_t_new, _, _ = self.q_posterior_mean_variance(
            x_start=x0_var.detach(), x_t=x_t, t=t_tensor,
        )
        return mu_t_new

    # ------------------------------------------------------------
    # V4 — OmniControl-style: dynamic Ke<<Kl iteration on mu_t
    # ------------------------------------------------------------
    def _guidance_v4_omni(
        self, *, mu_t, t_int, T, fk_fn, guidance_loss_fn,
        K_early=1, K_late=10, T_split=None, lr=0.5,
        base_weight=30.0, schedule="always",
    ):
        """
        V4: OmniControl-style. Few iters early, many iters late.
        """
        if not _schedule_active(schedule, t_int, T):
            return mu_t

        if T_split is None:
            T_split = T // 3

        n_steps = K_late if t_int < T_split else K_early
        if n_steps == 0:
            return mu_t

        mu_var = mu_t.detach().clone().contiguous().requires_grad_(True)
        optimizer = th.optim.SGD([mu_var], lr=lr)
        for _ in range(n_steps):
            optimizer.zero_grad()
            q = fk_fn(mu_var)
            loss = base_weight * guidance_loss_fn(q, t_int, T)
            loss.backward()
            optimizer.step()
        return mu_var.detach()

    # ------------------------------------------------------------
    # V5 — LGD: DPS + Monte Carlo smoothing
    # ------------------------------------------------------------
    def _guidance_v5_lgd(
        self, *, mu_t, x_t, t_int, t_tensor, T,
        model, model_kwargs, fk_fn, guidance_loss_fn,
        n_mc=4, s=1.0, schedule="always", base_weight=1.0,
        mc_noise_scale=0.1,
    ):
        """
        V5: LGD-style. Average n_mc DPS gradients with Gaussian-perturbed
        x_0_hat for variance reduction.
        """
        if not _schedule_active(schedule, t_int, T):
            return mu_t

        sigma_t = float(self.posterior_variance[t_int]) ** 0.5
        grads = []

        for _ in range(n_mc):
            with th.enable_grad():
                x_t_var = x_t.detach().clone().contiguous().requires_grad_(True)
                model_output = model(
                    x_t_var, self._scale_timesteps(t_tensor), **model_kwargs
                )

                if model_output.grad_fn is None:
                    if hasattr(model, "model"):
                        model_output = model.model(
                            x_t_var, self._scale_timesteps(t_tensor), **model_kwargs
                        )
                    if model_output.grad_fn is None:
                        return mu_t

                if self.model_mean_type == ModelMeanType.START_X:
                    x0_hat = model_output
                elif self.model_mean_type == ModelMeanType.EPSILON:
                    x0_hat = self._predict_xstart_from_eps(x_t_var, t_tensor, model_output)
                else:
                    return mu_t

                eps = th.randn_like(x0_hat) * sigma_t * mc_noise_scale
                x0_perturbed = x0_hat + eps

                q = fk_fn(x0_perturbed)
                if q.grad_fn is None:
                    return mu_t

                loss = base_weight * guidance_loss_fn(q, t_int, T)
                if loss.grad_fn is None:
                    return mu_t

                g = th.autograd.grad(loss, x_t_var)[0]
                grads.append(g.detach())

        if not grads:
            return mu_t

        grad_mean = th.stack(grads).mean(dim=0)

        # [FIX] 去掉 sigma_sq 缩放（与 V2 保持一致）
        # 原因：final schedule 下 sigma_sq < 0.01，会让推力小 100 倍
        delta_mu = (s * grad_mean.detach()).norm().item()
        print(f"[V5 UPDATE t={t_int:3d}] grad_norm={grad_mean.norm():.4f}  "
              f"|s*grad|={delta_mu:.4f}  s={s}  n_mc={n_mc}")

        mu_t_new = mu_t.detach() - s * grad_mean
        return mu_t_new

    # ------------------------------------------------------------
    # V2-norm — DPS with per-batch gradient normalization
    # 解决跨 seed 不稳定的第一层修复：step = s · ∇L / ‖∇L‖
    # ------------------------------------------------------------
    def _guidance_v2_dps_norm(
        self, *, mu_t, x_t, t_int, t_tensor, T,
        model, model_kwargs, fk_fn, guidance_loss_fn,
        s=2.0, schedule="always", base_weight=1.0,
        grad_clip_norm=None, spec_schedule_override=None,
    ):
        """
        Step 1: gradient-normalized DPS.

        与 V2 (_guidance_v2_dps) 的唯一区别：
            V2:       mu_t_new = mu_t - s * grad
            V2-norm:  mu_t_new = mu_t - s * grad / ‖grad‖_2 (per-batch)

        参数 s 的语义从 V2 的"梯度乘子（依赖 ‖∇L‖ 量级）"
        变成"单位方向上的推动距离"，跨 seed 等效推力一致。

        典型范围 s ∈ [1, 5]（V2 的 s ∈ [30, 80] 已不再适用）。

        Ref: Chung et al. ICLR 2023, eq. 16.
        """
        if not _schedule_active(schedule, t_int, T):
            return mu_t

        with th.enable_grad():
            x_t_var = x_t.detach().clone().contiguous().requires_grad_(True)
            model_output = model(x_t_var, self._scale_timesteps(t_tensor), **model_kwargs)
            if model_output.grad_fn is None:
                if hasattr(model, "model"):
                    model_output = model.model(
                        x_t_var, self._scale_timesteps(t_tensor), **model_kwargs
                    )
                if model_output.grad_fn is None:
                    return mu_t

            if self.model_mean_type == ModelMeanType.START_X:
                x0_hat = model_output
            elif self.model_mean_type == ModelMeanType.EPSILON:
                x0_hat = self._predict_xstart_from_eps(x_t_var, t_tensor, model_output)
            else:
                return mu_t

            q = fk_fn(x0_hat)
            if q.grad_fn is None:
                return mu_t
            _spec_sched = spec_schedule_override if spec_schedule_override is not None else schedule
            loss = base_weight * guidance_loss_fn(q, t_int, T, spec_schedule_override=_spec_sched)
            if loss.grad_fn is None or loss.item() == 0.0:
                return mu_t
            grad = th.autograd.grad(loss, x_t_var)[0]

        grad = grad.detach()
        # per-batch L2 归一化（不是全局，否则跨 batch 元素互相耦合）
        gn = grad.flatten(1).norm(dim=1).clamp_min(1e-8)
        gn_shape = [gn.shape[0]] + [1] * (grad.dim() - 1)
        grad_unit = grad / gn.view(*gn_shape)

        if grad_clip_norm is not None:
            # 可选的总范数 clip（safety net）
            total = grad_unit.flatten(1).norm(dim=1).clamp_min(1e-8)
            scale = (grad_clip_norm / total).clamp_max(1.0)
            grad_unit = grad_unit * scale.view(*gn_shape)

        print(f"[V2-norm UPDATE t={t_int:3d}] loss={loss.item():.4f}  "
              f"raw_gn={gn.mean().item():.5f}  s={s}")

        return mu_t.detach() - s * grad_unit

    # ------------------------------------------------------------
    # V6 — Closed-loop PID + manifold projection
    # 解决跨 seed 不稳定的核心方案
    # ------------------------------------------------------------
    def _guidance_v6_closed_loop(
        self, *, mu_t, x_t, pred_xstart, t_int, t_tensor, T,
        model, model_kwargs, fk_fn, guidance, guidance_loss_fn,
        combined=None,
        # PID gains (calibrated for RAW grad — |grad|~0.2, s_t~80 → |Δμ|~16)
        Kp=80.0, Ki=1.0, Kd=5.0,
        s_min=0.5, s_max=50.0, I_max=20.0,
        # smoothing
        beta_ema=0.8,
        # anti-windup
        i_start_frac=0.5,
        # temporal smoothness term on the loss (motion-dim regularizer)
        lambda_smooth=0.02,
        # manifold projection (Step 3)
        manifold_project=True,
        manifold_alpha=1.0,
        # ---- 迭代 2 新增：对称损失 + 容差带停推 ----
        loss_form="huber",                  # "huber" | "hinge"
        huber_delta=0.05,                   # rad
        huber_direction="equal",            # 强制 V6 走双边
        # ---- 迭代 4：spec schedule 覆盖 ----
        # spec 注册的 last_quarter 会把 V6 的早期 5 步全 mask 掉（loss=0, grad=0）
        # PID 的 c_t 时间步衰减已经自带早期软系数，spec schedule 二次 mask 多余
        # 默认 "always" 让控制器在全部去噪步都有梯度信号
        spec_schedule_override="always",
        # ---- 迭代 5：sigma 硬过滤 ----
        # "always" + s_min 同时存在时，最早期高 σ 步（c_t≈0 但 s_min 顶住）
        # 仍会注入小推力，per-frame 一致性受损。sigma_cutoff 直接跳过
        # σ > cutoff 的步，比 spec_schedule 更原理化（按噪声水平而非 step 索引）。
        # 推荐 0.4–0.6。None=禁用，等价于"always"全程引导。
        sigma_cutoff=None,
        # ---- 迭代 3 关键变更 ----
        # 默认不归一化梯度 + 默认关闭 band_gate
        # Huber loss 的 grad 自带衰减（远目标=1，近目标→0），归一化反而破坏自调节
        normalize_grad=False,
        band_gate=False,
        band_gate_factor=0.5,               # 若开启，只在 |err|<tol/2 时停（很保守）
        # 单步位移上限（防止大冲击；与 normalize_grad=False 配合作为 safety）
        delta_max=None,                     # None=不限；典型值 5.0
        # schedule and base scale
        schedule="always",
        base_weight=1.0,
        # target — 自动从 guidance 第一个 spec 推断；外部可显式传入覆盖
        target_value=None,
        target_unit=None,         # "deg" or other (None → 从 spec 读)
        angle_to_err_scale=1.0,
        # 消融用：False → c_t=1.0 全程（去掉时衰增益，隔离 c_t 的贡献）
        use_time_decay=True,
        # 容差带内冻结：|err| < tol 的样本置零梯度，避免 Huber 在带内持续扰动
        freeze_in_band=False,
        # ---- 路线 C：score suppression ----
        # 在 t/T < score_suppress_ratio 的低噪声步，把 MDM posterior mu_t
        # 向 x_t 渐进混合（权重 1-frac/ratio → frac=0 时完全用 x_t），
        # 削弱训练分布的"自愈"拉力，让 guidance delta 相对更强。
        # 0.0 = 不抑制（默认，行为不变）；推荐 0.1-0.2 做 OOD 实验。
        score_suppress_ratio=0.0,
        # diagnostic
        verbose=True,
    ):
        """
        迭代 3：raw-grad + Huber 自调节，移除 band-gate。

        失败的迭代 2 暴露了一个反直觉的事：grad 归一化 + band-gate 让 V6 退化
        成"前 2 步大冲击 + 后 11 步完全放任"，corr 从 0.236 崩到 0.022（CV=1860%）。

        诊断：
          - Huber 的梯度本身就是有界自调节的（远=1，近→0，准=0），
            做 grad/||grad|| 归一化把这个自调节杀了 → 每步固定大冲击
          - band-gate 一旦触发就锁死后面所有步 → 模型无机会平滑整合约束
          - 早期 pred_xstart 仍含强噪，冲击方向几乎随机 → 50% 概率搞反时间结构

        迭代 3 修正：
          - 默认 normalize_grad=False（用 raw grad，让 Huber 自带衰减生效）
          - 默认 band_gate=False（让 grad 自然衰减来收敛，不强行截断）
          - 默认 Kp=80（raw grad |≈0.2|，需要更大 s_t 才能给足早期推力）
          - 保留 delta_max 作为 safety clip（防偶发大冲击）

        每步去噪：
          1. (可选 band-gate，默认关) 检查 pred_xstart 的 angle 是否在带内
          2. DPS forward 拿 x0_hat, grad, angle_now
          3. PID controller 给出 s_t（基于 err = target − angle_now）
          4. (可选) manifold orthogonal projection
          5. mu_t_new = mu_t − s_t · grad           # raw grad！不归一化
        """
        from posture_guidance.closed_loop_controller import (
            ClosedLoopController, orthogonal_project,
        )

        if not _schedule_active(schedule, t_int, T):
            return mu_t

        # ---- 0. (迭代 5) sigma 硬过滤：跳过过早的高噪声步 ----
        # 在 sigma > sigma_cutoff 时 pred_xstart 仍嘈杂，推力方向不可信
        # 直接 return mu_t 避免噪声注入；PID 状态保留以便后续步使用
        if sigma_cutoff is not None:
            sigma_now = float(self.posterior_variance[t_int]) ** 0.5
            if sigma_now > sigma_cutoff:
                if verbose:
                    print(f"[V6 t={t_int:3d}] SIGMA-CUTOFF skip "
                          f"(σ={sigma_now:.3f} > cutoff={sigma_cutoff:.3f})")
                return mu_t

        # ---- 1. lazy-init controller in guidance object ----
        if not hasattr(guidance, "_v6_ctrl") or guidance._v6_ctrl is None:
            guidance._v6_ctrl = ClosedLoopController(
                Kp=Kp, Ki=Ki, Kd=Kd,
                s_min=s_min, s_max=s_max, I_max=I_max,
                beta_ema=beta_ema, i_start_frac=i_start_frac,
                use_time_decay=use_time_decay,
            )
        ctrl = guidance._v6_ctrl

        # ---- 2. 推断 target 和 tolerance（rad，若 spec 是 deg）----
        if len(guidance.specs) == 0:
            return mu_t
        spec0 = guidance.specs[0]
        if target_value is None:
            if spec0.unit == "deg":
                target_value = spec0.target_deg * math.pi / 180.0
            else:
                target_value = spec0.target_deg
        if spec0.unit == "deg":
            tol_rad = spec0.tolerance_deg * math.pi / 180.0
        else:
            tol_rad = spec0.tolerance_deg

        # ---- 3. 廉价预测当前 angle（用 dispatcher 已算好的 pred_xstart）做 band-gate ----
        # pred_xstart 是主循环 p_sample 已经算出的 MDM x0_hat 预测，零额外成本
        # fk_fn 是纯 tensor 运算，无梯度也能跑
        if band_gate:
            with th.no_grad():
                q_pre = fk_fn(pred_xstart)
                angle_pre = spec0.angle_fn(q_pre, **spec0.angle_fn_kwargs)
                a_pre = angle_pre.mean()
                err_pre = float(target_value - a_pre.item())
                if abs(err_pre) < tol_rad * band_gate_factor:
                    if verbose:
                        print(f"[V6 t={t_int:3d}] BAND-GATE skip "
                              f"(|err|={abs(err_pre):.4f} < tol={tol_rad*band_gate_factor:.4f})")
                    return mu_t

        # ---- 4. DPS forward：算 grad + 拿到 x0_hat 和 angle_now ----
        with th.enable_grad():
            x_t_var = x_t.detach().clone().contiguous().requires_grad_(True)
            model_output = model(x_t_var, self._scale_timesteps(t_tensor), **model_kwargs)
            if model_output.grad_fn is None:
                if hasattr(model, "model"):
                    model_output = model.model(
                        x_t_var, self._scale_timesteps(t_tensor), **model_kwargs
                    )
                if model_output.grad_fn is None:
                    return mu_t

            if self.model_mean_type == ModelMeanType.START_X:
                x0_hat = model_output
            elif self.model_mean_type == ModelMeanType.EPSILON:
                x0_hat = self._predict_xstart_from_eps(x_t_var, t_tensor, model_output)
            else:
                return mu_t

            q = fk_fn(x0_hat)
            if q.grad_fn is None:
                return mu_t

            # 主 loss + 时域平滑。loss_form="huber" → 过推时 grad 反转可拉回。
            # combined.motion_loss 在关节项之上叠加 −w_muscle·肌肉项（both 模式），
            # 纯关节（mode=joint）时与原 guidance.compute_loss 等价。
            anchor = q.sum() * 0.0
            if combined is not None:
                loss = base_weight * combined.motion_loss(
                    x0_hat, t_int, T,
                    temporal_smoothness_weight=lambda_smooth,
                    loss_form=loss_form,
                    huber_delta=huber_delta,
                    huber_direction_override=huber_direction if loss_form == "huber" else None,
                    spec_schedule_override=spec_schedule_override,
                ) + anchor
            else:
                loss = base_weight * guidance.compute_loss(
                    q, t_int, T,
                    temporal_smoothness_weight=lambda_smooth,
                    loss_form=loss_form,
                    huber_delta=huber_delta,
                    huber_direction_override=huber_direction if loss_form == "huber" else None,
                    spec_schedule_override=spec_schedule_override,
                ) + anchor
            if loss.grad_fn is None:
                return mu_t
            # 旧的 `loss.item()==0.0` 早退已删除：
            # - Huber 永远 differentiable，loss 不会真为 0
            # - band_gate 已在更上游用 err 信号做了清晰的收敛判定

            grad = th.autograd.grad(loss, x_t_var, retain_graph=True)[0]

            # 同时读出当前 angle 均值（不需要梯度回传）
            with th.no_grad():
                angle = spec0.angle_fn(q, **spec0.angle_fn_kwargs)
                if angle.dim() >= 2:
                    a_now = angle.mean(dim=tuple(range(1, angle.dim())))
                else:
                    a_now = angle.mean().unsqueeze(0)

        grad = grad.detach()
        x0_hat = x0_hat.detach()

        # ======== LOG_PRIOR_CONFLICT hook ========
        if os.environ.get("LOG_PRIOR_CONFLICT", "0") == "1":
            with th.no_grad():
                eps_for_log = self._predict_eps_from_xstart(x_t, t_tensor, x0_hat)
                grad_flat = grad.flatten(1)
                eps_flat = eps_for_log.flatten(1)
                cos_global = torch.nn.functional.cosine_similarity(grad_flat, eps_flat, dim=1)
                grad_j66 = grad_flat[:, :66]
                eps_j66 = eps_flat[:, :66]
                cos_j_all = torch.nn.functional.cosine_similarity(grad_j66, eps_j66, dim=1)
                JM = {"骨盆前倾":[0],"骨盆前倾_深蹲":[0],"躯干前倾":[0,3,6,9,12,16,17],"膝超伸_左":[4],"膝超伸_右":[5],"膝弯曲_左":[4],"膝弯曲_右":[5],"膝弯曲_A_左":[4],"膝弯曲_A_右":[5],"膝弯曲_B_左":[4],"膝弯曲_B_右":[5]}
                jds = []
                for j in JM.get(spec0.name, list(range(22))):
                    jds.extend([j*3, j*3+1, j*3+2])
                if jds:
                    cos_joint = torch.nn.functional.cosine_similarity(grad_flat[:, jds], eps_flat[:, jds], dim=1)
                else:
                    cos_joint = cos_j_all
                gn = grad_flat.norm(dim=1)
                en = eps_flat.norm(dim=1)
                nr = gn / (en + 1e-8)
                hn = torch.isnan(grad).any().item()
                gnr = grad.nan_to_num(0).norm().item()
                if not hasattr(guidance, "_conflict_log"):
                    guidance._conflict_log = []
                B = grad.shape[0]
                for b in range(B):
                    guidance._conflict_log.append({
                        "t":t_int,"T":T,
                        "cos_global":cos_global[b].item(),
                        "cos_joint":cos_joint[b].item() if cos_joint.numel()>1 else cos_joint.item(),
                        "grad_norm":gn[b].item(),
                        "eps_norm":en[b].item(),
                        "norm_ratio":nr[b].item(),
                        "loss":loss.item() if hasattr(loss,"item") else float(loss),
                        "has_nan":hn,
                        "grad_norm_raw":gnr,
                    })
        # ==================================================

        # ---- 4. PID controller → s_t ----
        # err 形状要和 grad 的 batch 维匹配
        B = grad.shape[0]
        if a_now.shape[0] != B:
            a_now = a_now.expand(B)
        err = (target_value - a_now) * angle_to_err_scale          # (B,)
        sigma_t = float(self.posterior_variance[t_int]) ** 0.5
        s_t = ctrl.update(err=err, sigma_t=sigma_t, t_int=t_int, T=T)   # (B,)

        # ---- 5. (Step 3) manifold orthogonal projection ----
        if manifold_project:
            grad = orthogonal_project(grad, x0_hat, alpha=manifold_alpha)

        # ---- 6. 计算 delta：默认用 raw grad（Huber 自带衰减）----
        B = grad.shape[0]
        gn = grad.flatten(1).norm(dim=1).clamp_min(1e-8)            # (B,) 仅诊断
        gn_shape = [B] + [1] * (grad.dim() - 1)

        if normalize_grad:
            # 旧路径（迭代 1/2 行为，保留可选）：grad/||grad|| 单位向量
            grad_for_step = grad / gn.view(*gn_shape)
        else:
            # 默认路径（迭代 3）：raw grad，让 Huber 的有界自调节梯度起效
            grad_for_step = grad

        # 容差带冻结：在带内的样本梯度置零，避免 Huber 双侧软推持续扰动已收敛的结果
        if freeze_in_band:
            tol_rad = spec0.tolerance_deg * (math.pi / 180.0 if spec0.unit == "deg" else 1.0)
            in_band = (err.abs() < tol_rad)      # (B,) bool
            if in_band.any():
                mask = (~in_band).float().view(*gn_shape)
                grad_for_step = grad_for_step * mask
                if verbose:
                    print(f"  [freeze_in_band] {in_band.sum().item()}/{B} samples frozen")

        delta = s_t.view(*gn_shape) * grad_for_step

        # ---- 6b. (可选 safety) delta_max clip 防偶发大冲击 ----
        if delta_max is not None:
            delta_norm = delta.flatten(1).norm(dim=1).clamp_min(1e-8)
            clip_scale = (delta_max / delta_norm).clamp_max(1.0)
            delta = delta * clip_scale.view(*gn_shape)

        if verbose:
            mode = "norm" if normalize_grad else "raw"
            print(
                f"[V6 t={t_int:3d}] err={err.mean().item():+.4f}rad "
                f"sigma={sigma_t:.4f} s_t={s_t.mean().item():.3f} "
                f"loss={loss.item():.4f} grad_norm={gn.mean().item():.5f} "
                f"|Δμ|={delta.norm().item():.4f} ({mode}"
                f"{', proj' if manifold_project else ''}"
                f"{', clip' if delta_max is not None else ''})"
            )

        # ---- Route C: score suppression at low-noise steps ----
        # At t→0, MDM posterior strongly pulls x toward training distribution.
        # Blend mu_t toward x_t so guidance gradient has more relative influence.
        mu_base = mu_t.detach()
        if score_suppress_ratio > 0.0:
            frac = t_int / T  # 1.0=pure noise, 0.0=clean
            if frac < score_suppress_ratio:
                suppress_w = 1.0 - frac / score_suppress_ratio  # 1 at t=0, 0 at boundary
                mu_base = (1.0 - suppress_w) * mu_base + suppress_w * x_t.detach()
                if verbose:
                    print(f"  [score-suppress] t={t_int} suppress_w={suppress_w:.3f} "
                          f"(frac={frac:.3f} < ratio={score_suppress_ratio:.3f})")

        return mu_base - delta

    # ============================================================
    # DDIM sampling (unchanged)
    # ============================================================
    def ddim_sample(
        self,
        model,
        x,
        t,
        clip_denoised=True,
        denoised_fn=None,
        cond_fn=None,
        model_kwargs=None,
        eta=0.0,
    ):
        out_orig = self.p_mean_variance(
            model,
            x,
            t,
            clip_denoised=clip_denoised,
            denoised_fn=denoised_fn,
            model_kwargs=model_kwargs,
        )
        if cond_fn is not None:
            out = self.condition_score(cond_fn, out_orig, x, t, model_kwargs=model_kwargs)
        else:
            out = out_orig

        eps = self._predict_eps_from_xstart(x, t, out["pred_xstart"])

        alpha_bar = _extract_into_tensor(self.alphas_cumprod, t, x.shape)
        alpha_bar_prev = _extract_into_tensor(self.alphas_cumprod_prev, t, x.shape)
        sigma = (
            eta
            * th.sqrt((1 - alpha_bar_prev) / (1 - alpha_bar))
            * th.sqrt(1 - alpha_bar / alpha_bar_prev)
        )
        noise = th.randn_like(x)
        mean_pred = (
            out["pred_xstart"] * th.sqrt(alpha_bar_prev)
            + th.sqrt(1 - alpha_bar_prev - sigma ** 2) * eps
        )
        nonzero_mask = (
            (t != 0).float().view(-1, *([1] * (len(x.shape) - 1)))
        )
        sample = mean_pred + nonzero_mask * sigma * noise
        return {"sample": sample, "pred_xstart": out_orig["pred_xstart"]}

    def ddim_sample_with_grad(
        self,
        model,
        x,
        t,
        clip_denoised=True,
        denoised_fn=None,
        cond_fn=None,
        model_kwargs=None,
        eta=0.0,
    ):
        with th.enable_grad():
            x = x.detach().requires_grad_()
            out_orig = self.p_mean_variance(
                model,
                x,
                t,
                clip_denoised=clip_denoised,
                denoised_fn=denoised_fn,
                model_kwargs=model_kwargs,
            )
            if cond_fn is not None:
                out = self.condition_score_with_grad(cond_fn, out_orig, x, t,
                                                     model_kwargs=model_kwargs)
            else:
                out = out_orig

        out["pred_xstart"] = out["pred_xstart"].detach()

        eps = self._predict_eps_from_xstart(x, t, out["pred_xstart"])

        alpha_bar = _extract_into_tensor(self.alphas_cumprod, t, x.shape)
        alpha_bar_prev = _extract_into_tensor(self.alphas_cumprod_prev, t, x.shape)
        sigma = (
            eta
            * th.sqrt((1 - alpha_bar_prev) / (1 - alpha_bar))
            * th.sqrt(1 - alpha_bar / alpha_bar_prev)
        )
        noise = th.randn_like(x)
        mean_pred = (
            out["pred_xstart"] * th.sqrt(alpha_bar_prev)
            + th.sqrt(1 - alpha_bar_prev - sigma ** 2) * eps
        )
        nonzero_mask = (
            (t != 0).float().view(-1, *([1] * (len(x.shape) - 1)))
        )
        sample = mean_pred + nonzero_mask * sigma * noise
        return {"sample": sample, "pred_xstart": out_orig["pred_xstart"].detach()}

    def ddim_reverse_sample(
        self,
        model,
        x,
        t,
        clip_denoised=True,
        denoised_fn=None,
        model_kwargs=None,
        eta=0.0,
    ):
        assert eta == 0.0, "Reverse ODE only for deterministic path"
        out = self.p_mean_variance(
            model,
            x,
            t,
            clip_denoised=clip_denoised,
            denoised_fn=denoised_fn,
            model_kwargs=model_kwargs,
        )
        eps = (
            _extract_into_tensor(self.sqrt_recip_alphas_cumprod, t, x.shape) * x
            - out["pred_xstart"]
        ) / _extract_into_tensor(self.sqrt_recipm1_alphas_cumprod, t, x.shape)
        alpha_bar_next = _extract_into_tensor(self.alphas_cumprod_next, t, x.shape)

        mean_pred = (
            out["pred_xstart"] * th.sqrt(alpha_bar_next)
            + th.sqrt(1 - alpha_bar_next) * eps
        )
        return {"sample": mean_pred, "pred_xstart": out["pred_xstart"]}

    def ddim_sample_loop(
        self,
        model,
        shape,
        noise=None,
        clip_denoised=True,
        denoised_fn=None,
        cond_fn=None,
        model_kwargs=None,
        device=None,
        progress=False,
        eta=0.0,
        skip_timesteps=0,
        init_image=None,
        randomize_class=False,
        cond_fn_with_grad=False,
        dump_steps=None,
        const_noise=False,
    ):
        if dump_steps is not None:
            raise NotImplementedError()
        if const_noise == True:
            raise NotImplementedError()

        final = None
        for sample in self.ddim_sample_loop_progressive(
            model,
            shape,
            noise=noise,
            clip_denoised=clip_denoised,
            denoised_fn=denoised_fn,
            cond_fn=cond_fn,
            model_kwargs=model_kwargs,
            device=device,
            progress=progress,
            eta=eta,
            skip_timesteps=skip_timesteps,
            init_image=init_image,
            randomize_class=randomize_class,
            cond_fn_with_grad=cond_fn_with_grad,
        ):
            final = sample
        return final["sample"]

    def ddim_sample_loop_progressive(
        self,
        model,
        shape,
        noise=None,
        clip_denoised=True,
        denoised_fn=None,
        cond_fn=None,
        model_kwargs=None,
        device=None,
        progress=False,
        eta=0.0,
        skip_timesteps=0,
        init_image=None,
        randomize_class=False,
        cond_fn_with_grad=False,
    ):
        if device is None:
            device = next(model.parameters()).device
        assert isinstance(shape, (tuple, list))
        if noise is not None:
            img = noise
        else:
            img = th.randn(*shape, device=device)

        if skip_timesteps and init_image is None:
            init_image = th.zeros_like(img)

        indices = list(range(self.num_timesteps - skip_timesteps))[::-1]

        if init_image is not None:
            my_t = th.ones([shape[0]], device=device, dtype=th.long) * indices[0]
            img = self.q_sample(init_image, my_t, img)

        if progress:
            from tqdm.auto import tqdm
            indices = tqdm(indices)

        for i in indices:
            t = th.tensor([i] * shape[0], device=device)
            if randomize_class and 'y' in model_kwargs:
                model_kwargs['y'] = th.randint(low=0, high=model.num_classes,
                                               size=model_kwargs['y'].shape,
                                               device=model_kwargs['y'].device)
            with th.no_grad():
                sample_fn = self.ddim_sample_with_grad if cond_fn_with_grad else self.ddim_sample
                out = sample_fn(
                    model,
                    img,
                    t,
                    clip_denoised=clip_denoised,
                    denoised_fn=denoised_fn,
                    cond_fn=cond_fn,
                    model_kwargs=model_kwargs,
                    eta=eta,
                )
                yield out
                img = out["sample"]

    # ============================================================
    # PLMS sampling (unchanged)
    # ============================================================
    def plms_sample(
        self,
        model,
        x,
        t,
        clip_denoised=True,
        denoised_fn=None,
        cond_fn=None,
        model_kwargs=None,
        cond_fn_with_grad=False,
        order=2,
        old_out=None,
    ):
        if not int(order) or not 1 <= order <= 4:
            raise ValueError('order is invalid (should be int from 1-4).')

        def get_model_output(x, t):
            with th.set_grad_enabled(cond_fn_with_grad and cond_fn is not None):
                x = x.detach().requires_grad_() if cond_fn_with_grad else x
                out_orig = self.p_mean_variance(
                    model,
                    x,
                    t,
                    clip_denoised=clip_denoised,
                    denoised_fn=denoised_fn,
                    model_kwargs=model_kwargs,
                )
                if cond_fn is not None:
                    if cond_fn_with_grad:
                        out = self.condition_score_with_grad(cond_fn, out_orig, x, t, model_kwargs=model_kwargs)
                        x = x.detach()
                    else:
                        out = self.condition_score(cond_fn, out_orig, x, t, model_kwargs=model_kwargs)
                else:
                    out = out_orig

            eps = self._predict_eps_from_xstart(x, t, out["pred_xstart"])
            return eps, out, out_orig

        alpha_bar = _extract_into_tensor(self.alphas_cumprod, t, x.shape)
        alpha_bar_prev = _extract_into_tensor(self.alphas_cumprod_prev, t, x.shape)
        eps, out, out_orig = get_model_output(x, t)

        if order > 1 and old_out is None:
            old_eps = [eps]
            mean_pred = out["pred_xstart"] * th.sqrt(alpha_bar_prev) + th.sqrt(1 - alpha_bar_prev) * eps
            eps_2, _, _ = get_model_output(mean_pred, t - 1)
            eps_prime = (eps + eps_2) / 2
            pred_prime = self._predict_xstart_from_eps(x, t, eps_prime)
            mean_pred = pred_prime * th.sqrt(alpha_bar_prev) + th.sqrt(1 - alpha_bar_prev) * eps_prime
        else:
            old_eps = old_out["old_eps"]
            old_eps.append(eps)
            cur_order = min(order, len(old_eps))
            if cur_order == 1:
                eps_prime = old_eps[-1]
            elif cur_order == 2:
                eps_prime = (3 * old_eps[-1] - old_eps[-2]) / 2
            elif cur_order == 3:
                eps_prime = (23 * old_eps[-1] - 16 * old_eps[-2] + 5 * old_eps[-3]) / 12
            elif cur_order == 4:
                eps_prime = (55 * old_eps[-1] - 59 * old_eps[-2] + 37 * old_eps[-3] - 9 * old_eps[-4]) / 24
            else:
                raise RuntimeError('cur_order is invalid.')
            pred_prime = self._predict_xstart_from_eps(x, t, eps_prime)
            mean_pred = pred_prime * th.sqrt(alpha_bar_prev) + th.sqrt(1 - alpha_bar_prev) * eps_prime

        if len(old_eps) >= order:
            old_eps.pop(0)

        nonzero_mask = (t != 0).float().view(-1, *([1] * (len(x.shape) - 1)))
        sample = mean_pred * nonzero_mask + out["pred_xstart"] * (1 - nonzero_mask)

        return {"sample": sample, "pred_xstart": out_orig["pred_xstart"], "old_eps": old_eps}

    def plms_sample_loop(
        self,
        model,
        shape,
        noise=None,
        clip_denoised=True,
        denoised_fn=None,
        cond_fn=None,
        model_kwargs=None,
        device=None,
        progress=False,
        skip_timesteps=0,
        init_image=None,
        randomize_class=False,
        cond_fn_with_grad=False,
        order=2,
    ):
        final = None
        for sample in self.plms_sample_loop_progressive(
            model,
            shape,
            noise=noise,
            clip_denoised=clip_denoised,
            denoised_fn=denoised_fn,
            cond_fn=cond_fn,
            model_kwargs=model_kwargs,
            device=device,
            progress=progress,
            skip_timesteps=skip_timesteps,
            init_image=init_image,
            randomize_class=randomize_class,
            cond_fn_with_grad=cond_fn_with_grad,
            order=order,
        ):
            final = sample
        return final["sample"]

    def plms_sample_loop_progressive(
        self,
        model,
        shape,
        noise=None,
        clip_denoised=True,
        denoised_fn=None,
        cond_fn=None,
        model_kwargs=None,
        device=None,
        progress=False,
        skip_timesteps=0,
        init_image=None,
        randomize_class=False,
        cond_fn_with_grad=False,
        order=2,
    ):
        if device is None:
            device = next(model.parameters()).device
        assert isinstance(shape, (tuple, list))
        if noise is not None:
            img = noise
        else:
            img = th.randn(*shape, device=device)

        if skip_timesteps and init_image is None:
            init_image = th.zeros_like(img)

        indices = list(range(self.num_timesteps - skip_timesteps))[::-1]

        if init_image is not None:
            my_t = th.ones([shape[0]], device=device, dtype=th.long) * indices[0]
            img = self.q_sample(init_image, my_t, img)

        if progress:
            from tqdm.auto import tqdm
            indices = tqdm(indices)

        old_out = None

        for i in indices:
            t = th.tensor([i] * shape[0], device=device)
            if randomize_class and 'y' in model_kwargs:
                model_kwargs['y'] = th.randint(low=0, high=model.num_classes,
                                               size=model_kwargs['y'].shape,
                                               device=model_kwargs['y'].device)
            with th.no_grad():
                out = self.plms_sample(
                    model,
                    img,
                    t,
                    clip_denoised=clip_denoised,
                    denoised_fn=denoised_fn,
                    cond_fn=cond_fn,
                    model_kwargs=model_kwargs,
                    cond_fn_with_grad=cond_fn_with_grad,
                    order=order,
                    old_out=old_out,
                )
                yield out
                old_out = out
                img = out["sample"]

    # ============================================================
    # Loss / VLB / training (unchanged)
    # ============================================================
    def _vb_terms_bpd(
        self, model, x_start, x_t, t, clip_denoised=True, model_kwargs=None
    ):
        true_mean, _, true_log_variance_clipped = self.q_posterior_mean_variance(
            x_start=x_start, x_t=x_t, t=t
        )
        out = self.p_mean_variance(
            model, x_t, t, clip_denoised=clip_denoised, model_kwargs=model_kwargs
        )
        kl = normal_kl(
            true_mean, true_log_variance_clipped, out["mean"], out["log_variance"]
        )
        kl = mean_flat(kl) / np.log(2.0)

        decoder_nll = -discretized_gaussian_log_likelihood(
            x_start, means=out["mean"], log_scales=0.5 * out["log_variance"]
        )
        assert decoder_nll.shape == x_start.shape
        decoder_nll = mean_flat(decoder_nll) / np.log(2.0)

        output = th.where((t == 0), decoder_nll, kl)
        return {"output": output, "pred_xstart": out["pred_xstart"]}

    def training_losses(self, model, x_start, t, model_kwargs=None, noise=None, dataset=None):
        enc = model.model
        mask = model_kwargs['y']['mask']
        get_xyz = lambda sample: enc.rot2xyz(sample, mask=None, pose_rep=enc.pose_rep, translation=enc.translation,
                                             glob=enc.glob,
                                             jointstype='smpl',
                                             vertstrans=False)

        if model_kwargs is None:
            model_kwargs = {}
        if noise is None:
            noise = th.randn_like(x_start)
        x_t = self.q_sample(x_start, t, noise=noise)

        terms = {}

        if self.loss_type == LossType.KL or self.loss_type == LossType.RESCALED_KL:
            terms["loss"] = self._vb_terms_bpd(
                model=model,
                x_start=x_start,
                x_t=x_t,
                t=t,
                clip_denoised=False,
                model_kwargs=model_kwargs,
            )["output"]
            if self.loss_type == LossType.RESCALED_KL:
                terms["loss"] *= self.num_timesteps
        elif self.loss_type == LossType.MSE or self.loss_type == LossType.RESCALED_MSE:
            model_output = model(x_t, self._scale_timesteps(t), **model_kwargs)

            if self.model_var_type in [
                ModelVarType.LEARNED,
                ModelVarType.LEARNED_RANGE,
            ]:
                B, C = x_t.shape[:2]
                assert model_output.shape == (B, C * 2, *x_t.shape[2:])
                model_output, model_var_values = th.split(model_output, C, dim=1)
                frozen_out = th.cat([model_output.detach(), model_var_values], dim=1)
                terms["vb"] = self._vb_terms_bpd(
                    model=lambda *args, r=frozen_out: r,
                    x_start=x_start,
                    x_t=x_t,
                    t=t,
                    clip_denoised=False,
                )["output"]
                if self.loss_type == LossType.RESCALED_MSE:
                    terms["vb"] *= self.num_timesteps / 1000.0

            target = {
                ModelMeanType.PREVIOUS_X: self.q_posterior_mean_variance(
                    x_start=x_start, x_t=x_t, t=t
                )[0],
                ModelMeanType.START_X: x_start,
                ModelMeanType.EPSILON: noise,
            }[self.model_mean_type]
            assert model_output.shape == target.shape == x_start.shape

            terms["rot_mse"] = self.masked_l2(target, model_output, mask)

            target_xyz, model_output_xyz = None, None

            if self.lambda_rcxyz > 0.:
                target_xyz = get_xyz(target)
                model_output_xyz = get_xyz(model_output)
                terms["rcxyz_mse"] = self.masked_l2(target_xyz, model_output_xyz, mask)

            if self.lambda_vel_rcxyz > 0.:
                if self.data_rep == 'rot6d' and dataset.dataname in ['humanact12', 'uestc']:
                    target_xyz = get_xyz(target) if target_xyz is None else target_xyz
                    model_output_xyz = get_xyz(model_output) if model_output_xyz is None else model_output_xyz
                    target_xyz_vel = (target_xyz[:, :, :, 1:] - target_xyz[:, :, :, :-1])
                    model_output_xyz_vel = (model_output_xyz[:, :, :, 1:] - model_output_xyz[:, :, :, :-1])
                    terms["vel_xyz_mse"] = self.masked_l2(target_xyz_vel, model_output_xyz_vel, mask[:, :, :, 1:])

            if self.lambda_fc > 0.:
                with torch.autograd.set_detect_anomaly(True):
                    if self.data_rep == 'rot6d' and dataset.dataname in ['humanact12', 'uestc']:
                        target_xyz = get_xyz(target) if target_xyz is None else target_xyz
                        model_output_xyz = get_xyz(model_output) if model_output_xyz is None else model_output_xyz
                        l_ankle_idx, r_ankle_idx, l_foot_idx, r_foot_idx = 7, 8, 10, 11
                        relevant_joints = [l_ankle_idx, l_foot_idx, r_ankle_idx, r_foot_idx]
                        gt_joint_xyz = target_xyz[:, relevant_joints, :, :]
                        gt_joint_vel = torch.linalg.norm(gt_joint_xyz[:, :, :, 1:] - gt_joint_xyz[:, :, :, :-1], axis=2)
                        fc_mask = torch.unsqueeze((gt_joint_vel <= 0.01), dim=2).repeat(1, 1, 3, 1)
                        pred_joint_xyz = model_output_xyz[:, relevant_joints, :, :]
                        pred_vel = pred_joint_xyz[:, :, :, 1:] - pred_joint_xyz[:, :, :, :-1]
                        pred_vel[~fc_mask] = 0
                        terms["fc"] = self.masked_l2(pred_vel,
                                                     torch.zeros(pred_vel.shape, device=pred_vel.device),
                                                     mask[:, :, :, 1:])
            if self.lambda_vel > 0.:
                target_vel = (target[..., 1:] - target[..., :-1])
                model_output_vel = (model_output[..., 1:] - model_output[..., :-1])
                terms["vel_mse"] = self.masked_l2(target_vel[:, :-1, :, :],
                                                  model_output_vel[:, :-1, :, :],
                                                  mask[:, :, :, 1:])

            if self.lambda_target_loc > 0.:
                assert self.model_mean_type == ModelMeanType.START_X, 'This feature supports only X_start pred for now!'
                ref_target = model_kwargs['y']['target_cond']
                pred_target = get_target_location(model_output, dataset.mean_gpu, dataset.std_gpu,
                                            model_kwargs['y']['lengths'], dataset.t2m_dataset.opt.joints_num, model.all_goal_joint_names,
                                            model_kwargs['y']['target_joint_names'], model_kwargs['y']['is_heading'])
                terms["target_loc"] = masked_goal_l2(pred_target, ref_target, model_kwargs['y'], model.all_goal_joint_names)

            terms["loss"] = terms["rot_mse"] + terms.get('vb', 0.) +\
                            (self.lambda_vel * terms.get('vel_mse', 0.)) +\
                            (self.lambda_rcxyz * terms.get('rcxyz_mse', 0.)) + \
                            (self.lambda_target_loc * terms.get('target_loc', 0.)) + \
                            (self.lambda_fc * terms.get('fc', 0.))

        else:
            raise NotImplementedError(self.loss_type)

        return terms

    def fc_loss_rot_repr(self, gt_xyz, pred_xyz, mask):
        l_ankle_idx, r_ankle_idx = 7, 8
        l_foot_idx, r_foot_idx = 10, 11

        gt_joint_xyz = gt_xyz[:, [l_ankle_idx, l_foot_idx, r_ankle_idx, r_foot_idx], :, :]
        gt_joint_vel = torch.linalg.norm(gt_joint_xyz[:, :, :, 1:] - gt_joint_xyz[:, :, :, :-1], axis=2)
        fc_mask = (gt_joint_vel <= 0.01)
        pred_joint_xyz = pred_xyz[:, [l_ankle_idx, l_foot_idx, r_ankle_idx, r_foot_idx], :, :]
        pred_joint_vel = torch.linalg.norm(pred_joint_xyz[:, :, :, 1:] - pred_joint_xyz[:, :, :, :-1], axis=2)
        pred_joint_vel[~fc_mask] = 0
        pred_joint_vel = torch.unsqueeze(pred_joint_vel, dim=2)

        return self.masked_l2(pred_joint_vel, torch.zeros(pred_joint_vel.shape, device=pred_joint_vel.device),
                              mask[:, :, :, 1:])

    def foot_contact_loss_humanml3d(self, target, model_output):
        target_fc = target[:, -4:, :, :]
        root_rot_velocity = target[:, :1, :, :]
        root_linear_velocity = target[:, 1:3, :, :]
        root_y = target[:, 3:4, :, :]
        ric_data = target[:, 4:67, :, :]
        rot_data = target[:, 67:193, :, :]
        local_velocity = target[:, 193:259, :, :]
        contact = target[:, 259:, :, :]
        contact_mask_gt = contact > 0.5
        vel_lf_7 = local_velocity[:, 7 * 3:8 * 3, :, :]
        vel_rf_8 = local_velocity[:, 8 * 3:9 * 3, :, :]
        vel_lf_10 = local_velocity[:, 10 * 3:11 * 3, :, :]
        vel_rf_11 = local_velocity[:, 11 * 3:12 * 3, :, :]

        calc_vel_lf_7 = ric_data[:, 6 * 3:7 * 3, :, 1:] - ric_data[:, 6 * 3:7 * 3, :, :-1]
        calc_vel_rf_8 = ric_data[:, 7 * 3:8 * 3, :, 1:] - ric_data[:, 7 * 3:8 * 3, :, :-1]
        calc_vel_lf_10 = ric_data[:, 9 * 3:10 * 3, :, 1:] - ric_data[:, 9 * 3:10 * 3, :, :-1]
        calc_vel_rf_11 = ric_data[:, 10 * 3:11 * 3, :, 1:] - ric_data[:, 10 * 3:11 * 3, :, :-1]

        for chosen_vel_foot_calc, chosen_vel_foot, joint_idx, contact_mask_idx in zip(
                [calc_vel_lf_7, calc_vel_rf_8, calc_vel_lf_10, calc_vel_rf_11],
                [vel_lf_7, vel_lf_10, vel_rf_8, vel_rf_11],
                [7, 10, 8, 11],
                [0, 1, 2, 3]):
            tmp_mask_gt = contact_mask_gt[:, contact_mask_idx, :, :].cpu().detach().numpy().reshape(-1).astype(int)
            chosen_vel_norm = np.linalg.norm(chosen_vel_foot.cpu().detach().numpy().reshape((3, -1)), axis=0)
            chosen_vel_calc_norm = np.linalg.norm(chosen_vel_foot_calc.cpu().detach().numpy().reshape((3, -1)),
                                                  axis=0)

            print(tmp_mask_gt.shape)
            print(chosen_vel_foot.shape)
            print(chosen_vel_calc_norm.shape)
            import matplotlib.pyplot as plt
            plt.plot(tmp_mask_gt, label='FC mask')
            plt.plot(chosen_vel_norm, label='Vel. XYZ norm (from vector)')
            plt.plot(chosen_vel_calc_norm, label='Vel. XYZ norm (calculated diff XYZ)')

            plt.title(f'FC idx {contact_mask_idx}, Joint Index {joint_idx}')
            plt.legend()
            plt.show()
        return 0

    def velocity_consistency_loss_humanml3d(self, target, model_output):
        target_fc = target[:, -4:, :, :]
        root_rot_velocity = target[:, :1, :, :]
        root_linear_velocity = target[:, 1:3, :, :]
        root_y = target[:, 3:4, :, :]
        ric_data = target[:, 4:67, :, :]
        rot_data = target[:, 67:193, :, :]
        local_velocity = target[:, 193:259, :, :]
        contact = target[:, 259:, :, :]

        calc_vel_from_xyz = ric_data[:, :, :, 1:] - ric_data[:, :, :, :-1]
        velocity_from_vector = local_velocity[:, 3:, :, 1:]
        r_rot_quat, r_pos = motion_process.recover_root_rot_pos(target.permute(0, 2, 3, 1).type(th.FloatTensor))
        print(f'r_rot_quat: {r_rot_quat.shape}')
        print(f'calc_vel_from_xyz: {calc_vel_from_xyz.shape}')
        calc_vel_from_xyz = calc_vel_from_xyz.permute(0, 2, 3, 1)
        calc_vel_from_xyz = calc_vel_from_xyz.reshape((1, 1, -1, 21, 3)).type(th.FloatTensor)
        r_rot_quat_adapted = r_rot_quat[..., :-1, None, :].repeat((1, 1, 1, 21, 1)).to(calc_vel_from_xyz.device)
        print(f'calc_vel_from_xyz: {calc_vel_from_xyz.shape} , {calc_vel_from_xyz.device}')
        print(f'r_rot_quat_adapted: {r_rot_quat_adapted.shape}, {r_rot_quat_adapted.device}')

        calc_vel_from_xyz = motion_process.qrot(r_rot_quat_adapted, calc_vel_from_xyz)
        calc_vel_from_xyz = calc_vel_from_xyz.reshape((1, 1, -1, 21 * 3))
        calc_vel_from_xyz = calc_vel_from_xyz.permute(0, 3, 1, 2)
        print(f'calc_vel_from_xyz: {calc_vel_from_xyz.shape} , {calc_vel_from_xyz.device}')

        import matplotlib.pyplot as plt
        for i in range(21):
            plt.plot(np.linalg.norm(calc_vel_from_xyz[:, i * 3:(i + 1) * 3, :, :].cpu().detach().numpy().reshape((3, -1)), axis=0), label='Calc Vel')
            plt.plot(np.linalg.norm(velocity_from_vector[:, i * 3:(i + 1) * 3, :, :].cpu().detach().numpy().reshape((3, -1)), axis=0), label='Vector Vel')
            plt.title(f'Joint idx: {i}')
            plt.legend()
            plt.show()
        print(calc_vel_from_xyz.shape)
        print(velocity_from_vector.shape)
        diff = calc_vel_from_xyz - velocity_from_vector
        print(np.linalg.norm(diff.cpu().detach().numpy().reshape((63, -1)), axis=0))

        return 0

    def _prior_bpd(self, x_start):
        batch_size = x_start.shape[0]
        t = th.tensor([self.num_timesteps - 1] * batch_size, device=x_start.device)
        qt_mean, _, qt_log_variance = self.q_mean_variance(x_start, t)
        kl_prior = normal_kl(
            mean1=qt_mean, logvar1=qt_log_variance, mean2=0.0, logvar2=0.0
        )
        return mean_flat(kl_prior) / np.log(2.0)

    def calc_bpd_loop(self, model, x_start, clip_denoised=True, model_kwargs=None):
        device = x_start.device
        batch_size = x_start.shape[0]

        vb = []
        xstart_mse = []
        mse = []
        for t in list(range(self.num_timesteps))[::-1]:
            t_batch = th.tensor([t] * batch_size, device=device)
            noise = th.randn_like(x_start)
            x_t = self.q_sample(x_start=x_start, t=t_batch, noise=noise)
            with th.no_grad():
                out = self._vb_terms_bpd(
                    model,
                    x_start=x_start,
                    x_t=x_t,
                    t=t_batch,
                    clip_denoised=clip_denoised,
                    model_kwargs=model_kwargs,
                )
            vb.append(out["output"])
            xstart_mse.append(mean_flat((out["pred_xstart"] - x_start) ** 2))
            eps = self._predict_eps_from_xstart(x_t, t_batch, out["pred_xstart"])
            mse.append(mean_flat((eps - noise) ** 2))

        vb = th.stack(vb, dim=1)
        xstart_mse = th.stack(xstart_mse, dim=1)
        mse = th.stack(mse, dim=1)

        prior_bpd = self._prior_bpd(x_start)
        total_bpd = vb.sum(dim=1) + prior_bpd
        return {
            "total_bpd": total_bpd,
            "prior_bpd": prior_bpd,
            "vb": vb,
            "xstart_mse": xstart_mse,
            "mse": mse,
        }


# ================================================================
# Module-level helpers
# ================================================================
def _schedule_active(schedule, t, T):
    """
    Return True if guidance should fire at diffusion step t.
    Used by V1/V3/V4/V5.
    """
    if schedule == "always":
        return True
    if schedule == "decay":
        return True
    if schedule == "last_quarter":
        return t < T // 4
    if schedule == "second_half":
        return t < T // 2
    if schedule == "final":
        return t < 5
    return True


def _extract_into_tensor(arr, timesteps, broadcast_shape):
    """
    Extract values from a 1-D numpy array for a batch of indices.
    """
    res = th.from_numpy(arr).to(device=timesteps.device)[timesteps].float()
    while len(res.shape) < len(broadcast_shape):
        res = res[..., None]
    return res.expand(broadcast_shape)