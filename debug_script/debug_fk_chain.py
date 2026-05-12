import torch
import math
import numpy as np
import sys
sys.path.insert(0, "/root/autodl-tmp/motion-diffusion-model")

from posture_guidance.angle_ops import pelvis_tilt_angle
from posture_guidance.mdm_integration import make_fk_fn
from data_loaders.get_data import get_dataset_loader


# 1. 加载数据集（拿到 t2m_dataset 用于 fk_fn）
print("Loading dataset...")
data = get_dataset_loader(
    name="humanml", batch_size=1, num_frames=120,
    split="test", hml_mode="text_only",
)
fk_fn = make_fk_fn(data.dataset.t2m_dataset, n_joints=22)


# 2. 加载之前的 baseline 结果（hml 格式）
result = np.load(
    "./new_results/posture_exp_20260503_232934/comparison.npy",
    allow_pickle=True
).item()

# motion_hml shape: (B, 263, 1, T)
mu_t_baseline = torch.from_numpy(result["motion_hml"]).float().cuda()
print(f"mu_t shape: {mu_t_baseline.shape}")


# 3. 用 fk_fn 算关节坐标和骨盆角
mu_var = mu_t_baseline.detach().clone().requires_grad_(True)
q = fk_fn(mu_var)                             # (1, T, 22, 3)
angle_per_frame = pelvis_tilt_angle(q[0])     # (T,)
angle_mean = angle_per_frame.mean()

print(f"\n=== fk_fn 出来的骨盆角 ===")
print(f"mean: {(angle_mean * 180 / math.pi).item():+.2f}°")
print(f"max:  {(angle_per_frame * 180 / math.pi).max().item():+.2f}°")
print(f"min:  {(angle_per_frame * 180 / math.pi).min().item():+.2f}°")


# 4. 反传梯度
angle_mean.backward()
grad = mu_var.grad
print(f"\n=== 梯度 ===")
print(f"grad norm: {grad.norm().item():.4f}")
print(f"grad max:  {grad.abs().max().item():.4f}")


# 5. 关键测试：手动用梯度做一步 SGD，看 angle 是否真的增加
# 注意：我们要让 angle 增大（梯度上升），所以 mu_t = mu_t + lr * grad
print(f"\n=== 梯度上升测试（让 angle 增大）===")
for lr in [0.01, 0.1, 0.5, 1.0]:
    mu_after = mu_var.detach() + lr * grad
    q_after = fk_fn(mu_after.requires_grad_(False))
    angle_after = pelvis_tilt_angle(q_after[0]).mean()
    angle_after_deg = (angle_after * 180 / math.pi).item()
    angle_before_deg = (angle_mean * 180 / math.pi).item()
    delta = angle_after_deg - angle_before_deg
    print(f"  lr={lr}: angle {angle_before_deg:+.2f}° → {angle_after_deg:+.2f}° "
          f"(Δ={delta:+.2f}°)")

print()
print("如果 Δ 是正值 → 梯度方向正确，guidance 应该工作")
print("如果 Δ 是负值 → fk_fn 或 angle_fn 里有反符号 bug")
