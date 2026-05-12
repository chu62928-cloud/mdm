"""
debug_script/debug_v2_grad.py

诊断 V2 (DPS) 梯度链断在哪一步。
逐步前向，每一步检查 .requires_grad / .grad_fn。

★ 修复：使用 load_saved_model（与 run_posture_comparison.py 完全一致），
  不再用 load_model_wo_clip（会触发 KeyError: sequence_pos_encoder.pe）

使用方式（在项目根目录执行）：
    cd /root/autodl-tmp/motion-diffusion-model
    python debug_script/debug_v2_grad.py
"""

import os
import sys
import torch

# 确保能找到项目根目录的模块
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)


def check_grad(name, tensor):
    """打印 tensor 的梯度状态，返回是否有梯度。"""
    if tensor is None:
        print(f"  ❌ {name}: None")
        return False
    has_grad = tensor.requires_grad
    has_fn   = tensor.grad_fn is not None
    is_leaf  = tensor.is_leaf
    status   = "✅" if (has_grad or has_fn) else "❌ BROKEN"
    print(f"  {status} {name}: shape={tuple(tensor.shape)} "
          f"requires_grad={has_grad} "
          f"grad_fn={'YES' if has_fn else 'NO'} "
          f"is_leaf={is_leaf}")
    return has_grad or has_fn


def main():
    # ----------------------------------------------------------------
    # 导入：与 run_posture_comparison.py 完全对齐
    # ----------------------------------------------------------------
    from utils.parser_util import generate_args
    from utils.fixseed import fixseed
    from utils.model_util import create_model_and_diffusion, load_saved_model  # ★ 修复
    from utils import dist_util
    from utils.sampler_util import ClassifierFreeSampleModel
    from data_loaders.get_data import get_dataset_loader
    from data_loaders.tensors import collate
    from posture_guidance.mdm_integration import make_fk_fn
    from posture_guidance.controller import PostureGuidance

    # ----------------------------------------------------------------
    # sys.argv：与 run_posture_comparison.py 同参数格式
    # ★ 根据你的实际路径修改 --model_path
    # ----------------------------------------------------------------
    sys.argv = [
        "debug_v2_grad",
        "--model_path",    "./save/humanml_trans_dec_512_bert/model000200000.pt",
        "--text_prompt",   "a person is walking forward",
        "--motion_length", "6.0",
        "--seed",          "42",
        "--num_samples",   "1",
        "--num_repetitions", "1",
        "--guidance_param", "2.5",
        "--output_dir",    "/tmp/debug_out",
        "--dataset",       "humanml",
    ]
    args = generate_args()
    fixseed(args.seed)

    # ----------------------------------------------------------------
    # 数据集 / 模型加载（完全复用 run_posture_comparison.py 的路径）
    # ----------------------------------------------------------------
    dist_util.setup_dist(args.device)

    fps      = 20
    n_frames = min(196, int(args.motion_length * fps))
    args.batch_size = args.num_samples

    print("=" * 60)
    print("Loading dataset ...")
    print("=" * 60)
    data = get_dataset_loader(
        name=args.dataset,
        batch_size=args.batch_size,
        num_frames=196,
        split="test",
        hml_mode="text_only",
    )
    data.fixed_length = n_frames

    print("Creating model and diffusion ...")
    model, diffusion = create_model_and_diffusion(args, data)

    print(f"Loading checkpoint from [{args.model_path}] ...")
    load_saved_model(model, args.model_path, use_avg=args.use_ema)  # ★ 修复

    if args.guidance_param != 1:
        model = ClassifierFreeSampleModel(model)
    model.to(dist_util.dev())
    model.eval()

    # ----------------------------------------------------------------
    # model_kwargs（与 run_posture_comparison.py 完全相同）
    # ----------------------------------------------------------------
    text_prompt = "a person is walking forward"
    texts       = [text_prompt] * args.num_samples

    collate_args = [{"inp": torch.zeros(n_frames), "tokens": None,
                     "lengths": n_frames}] * args.num_samples
    collate_args = [dict(arg, text=txt) for arg, txt in zip(collate_args, texts)]
    _, model_kwargs = collate(collate_args)

    model_kwargs["y"] = {
        k: (v.to(dist_util.dev()) if torch.is_tensor(v) else v)
        for k, v in model_kwargs["y"].items()
    }
    if args.guidance_param != 1:
        model_kwargs["y"]["scale"] = (
            torch.ones(args.batch_size, device=dist_util.dev()) * args.guidance_param
        )
    if hasattr(model, "encode_text"):
        # ★ 与 run_posture_comparison.py 一致：提前缓存 text_embed
        model_kwargs["y"]["text_embed"] = model.encode_text(model_kwargs["y"]["text"])

    # ----------------------------------------------------------------
    # FK 函数
    # ----------------------------------------------------------------
    fk_fn = make_fk_fn(data.dataset.t2m_dataset, n_joints=22)

    # ----------------------------------------------------------------
    # 模拟 sampling 的一步（用 t=10 作为代表时间步）
    # ----------------------------------------------------------------
    device = dist_util.dev()
    B = args.num_samples
    C = model.njoints   # 应该是 263
    T = n_frames        # 应该是 120

    print(f"\n模型信息: njoints={model.njoints}, nfeats={model.nfeats}")
    print(f"采样参数: B={B}, C={C}, T={T}, device={device}")

    print("\n" + "=" * 60)
    print("Diagnostic: tracing gradient chain step-by-step (t=10)")
    print("=" * 60)

    # --------- 步骤 1：leaf tensor x_t ---------
    print("\n[Step 1] 构造 x_t leaf tensor")
    x_t = torch.randn(B, C, 1, T, device=device)
    x_t.requires_grad_(True)
    check_grad("x_t", x_t)

    # --------- 步骤 2：MDM forward ---------
    print("\n[Step 2] model(x_t, t=10) 前向")
    t_tensor = torch.tensor([10] * B, device=device)
    t_scaled = diffusion._scale_timesteps(t_tensor)
    try:
        model_output = model(x_t, t_scaled, **model_kwargs)
        ok = check_grad("model_output", model_output)
        if not ok:
            print("\n  ⚠ model_output 没有 grad_fn！")
            print("  可能原因：")
            print("    a) MDM forward 内部某处用了 torch.no_grad() 或 @torch.inference_mode()")
            print("    b) model.module 的某个子模块被 .requires_grad_(False)")
            print("  → 下面会尝试 model.model(...) 绕过 ClassifierFreeSampleModel")
    except Exception as e:
        print(f"  ❌ model forward 报错: {e}")
        return

    # 如果是 ClassifierFreeSampleModel 包了一层，尝试内部 model
    if not model_output.requires_grad and hasattr(model, "model"):
        print("\n  [尝试] 直接调用 model.model(...) 绕过 ClassifierFree wrapper")
        try:
            inner_out = model.model(x_t, t_scaled, **model_kwargs)
            ok2 = check_grad("inner_out (model.model)", inner_out)
            if ok2:
                print("  ✅ 内部 model 有梯度，问题在 ClassifierFreeSampleModel 包装层")
                print("  修复方向: V2 的 forward 改为调用 model.model(...) 而不是 model(...)")
                model_output = inner_out
        except Exception as e:
            print(f"  ❌ model.model forward 报错: {e}")

    # --------- 步骤 3：推导 x0_hat ---------
    print("\n[Step 3] 推导 x0_hat")
    from diffusion.gaussian_diffusion import ModelMeanType
    if diffusion.model_mean_type == ModelMeanType.START_X:
        x0_hat = model_output
        print("  model_mean_type = START_X, x0_hat = model_output")
    elif diffusion.model_mean_type == ModelMeanType.EPSILON:
        x0_hat = diffusion._predict_xstart_from_eps(x_t, t_tensor, model_output)
        print("  model_mean_type = EPSILON, x0_hat = _predict_xstart_from_eps(...)")
    else:
        print(f"  ❌ 不支持的 model_mean_type: {diffusion.model_mean_type}")
        return
    check_grad("x0_hat", x0_hat)

    # --------- 步骤 4：fk_fn ---------
    print("\n[Step 4] fk_fn(x0_hat)")
    try:
        q = fk_fn(x0_hat)
        ok = check_grad("q (joint xyz)", q)
        if not ok:
            print("\n  ⚠ fk_fn 断掉了梯度！")
            print("  可能原因：")
            print("    a) recover_from_ric 内部有 .detach() 或 .cpu().numpy() 转换")
            print("    b) inv_transform 里用了 numpy 操作")
            print("  → 检查 make_fk_fn 的实现，确保全程是 pure torch 操作")
    except Exception as e:
        print(f"  ❌ fk_fn 报错: {e}")
        return

    # --------- 步骤 5：guidance loss ---------
    print("\n[Step 5] guidance.compute_loss(q)")
    guidance = PostureGuidance(instructions=["骨盆前倾"])
    try:
        loss = guidance.compute_loss(q, t=10, T=50)
        ok = check_grad("loss", loss)
        if not ok:
            print("\n  ⚠ compute_loss 断掉了梯度！")
            print("  可能原因：angle_ops 里有 .item()、.detach() 或 numpy 操作")
    except Exception as e:
        print(f"  ❌ compute_loss 报错: {e}")
        return

    # --------- 步骤 6：autograd.grad ---------
    print("\n[Step 6] autograd.grad(loss, x_t)")
    try:
        grad = torch.autograd.grad(loss, x_t)[0]
        print(f"  ✅ 梯度正常! shape={tuple(grad.shape)} norm={grad.norm():.4f}")
        grad_ok = True
    except RuntimeError as e:
        print(f"  ❌ autograd.grad 失败: {e}")
        grad_ok = False

    # ================================================================
    # 独立子测试：精确定位断点
    # ================================================================
    print("\n" + "=" * 60)
    print("Sub-tests: 精确定位断点")
    print("=" * 60)

    # 子测试 A：fk_fn 单独测试
    print("\n[Sub-A] fk_fn 独立梯度测试（用随机输入）")
    test_x0 = torch.randn(B, C, 1, T, device=device, requires_grad=True)
    try:
        test_q = fk_fn(test_x0)
        ok = check_grad("test_q", test_q)
        if ok:
            try:
                g = torch.autograd.grad(test_q.sum(), test_x0)[0]
                print(f"  ✅ fk_fn 完全可微 — grad norm={g.norm():.4f}")
                fk_ok = True
            except RuntimeError as e:
                print(f"  ❌ fk_fn backward 失败: {e}")
                fk_ok = False
        else:
            print("  ❌ fk_fn 输出无 grad_fn → fk_fn 内部断了梯度链")
            fk_ok = False
    except Exception as e:
        print(f"  ❌ fk_fn 前向报错: {e}")
        fk_ok = False

    # 子测试 B：compute_loss 单独测试
    print("\n[Sub-B] compute_loss 独立梯度测试（用随机 q）")
    test_q2 = torch.randn(B, T, 22, 3, device=device, requires_grad=True)
    try:
        test_loss = guidance.compute_loss(test_q2, t=10, T=50)
        ok = check_grad("test_loss", test_loss)
        if ok:
            try:
                g = torch.autograd.grad(test_loss, test_q2)[0]
                print(f"  ✅ compute_loss 完全可微 — grad norm={g.norm():.4f}")
                loss_ok = True
            except RuntimeError as e:
                print(f"  ❌ compute_loss backward 失败: {e}")
                loss_ok = False
        else:
            print("  ❌ compute_loss 输出无 grad_fn → compute_loss 内部断了梯度链")
            loss_ok = False
    except Exception as e:
        print(f"  ❌ compute_loss 前向报错: {e}")
        loss_ok = False

    # ================================================================
    # 最终诊断报告
    # ================================================================
    print("\n" + "=" * 60)
    print("最终诊断报告")
    print("=" * 60)

    if grad_ok:
        print("🎉 V2 完整梯度链正常！可以直接运行 v2_dps。")
    else:
        print("梯度链断点分析：")
        if not model_output.requires_grad:
            print("  ❌ 断点在 MDM forward")
            print("  ➜ 修复：在 _guidance_v2_dps 里改用 model.model(...)")
            print("          或检查 ClassifierFreeSampleModel 是否截断了梯度")
        elif not x0_hat.requires_grad:
            print("  ❌ 断点在 _predict_xstart_from_eps")
            print("  ➜ 这几乎不可能，检查 _extract_into_tensor 返回的类型")
        elif not fk_ok:
            print("  ❌ 断点在 fk_fn（recover_from_ric 等）")
            print("  ➜ 修复：让 fk_fn 走 pure torch 路径，见下方说明")
        elif not loss_ok:
            print("  ❌ 断点在 compute_loss（angle_ops）")
            print("  ➜ 修复：检查 angle_ops.py，去掉 .item()/.detach()/numpy 操作")
        else:
            print("  ❓ 断点位置未知，请把上面的 Step 输出贴给 Claude 分析")

    print("\n" + "=" * 60)
    print("fk_fn 修复参考（如果 Sub-A 失败）")
    print("=" * 60)
    print("""
如果 fk_fn 断梯度，原因是 make_fk_fn 里用了类似：

    def fk_fn(mu_t):
        # ❌ 这三行会断梯度
        data_np = mu_t.detach().cpu().numpy()     # detach !!!
        sample  = dataset.inv_transform(data_np)
        ...

修复方法：确保 inv_transform 和 recover_from_ric 用 pure torch：

    def fk_fn(mu_t):
        # ✅ pure torch，保留梯度
        # mu_t: (B, 263, 1, T)
        sample = mu_t.squeeze(2).permute(0, 3, 1, 2)   # (B, T, 1, 263)
        sample = sample * std_tensor + mean_tensor       # inv_transform
        xyz = recover_from_ric(sample, n_joints)         # (B, T, J, 3)
        return xyz

其中 std_tensor 和 mean_tensor 是从 dataset 预先提取的 torch tensor，
不做 detach，但它们是常量所以不影响梯度流向 mu_t。
""")


if __name__ == "__main__":
    main()