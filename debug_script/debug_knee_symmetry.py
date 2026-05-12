"""
诊断左右膝 guidance 不对称的根因。
"""
import torch, math, numpy as np, sys
sys.path.insert(0, "/root/autodl-tmp/motion-diffusion-model")

from posture_guidance.angle_ops import signed_knee_angle
from posture_guidance.mdm_integration import make_fk_fn
from data_loaders.get_data import get_dataset_loader

print("Loading dataset...")
data_loader = get_dataset_loader(
    name="humanml", batch_size=1, num_frames=120,
    split="test", hml_mode="text_only",
)
fk_fn = make_fk_fn(data_loader.dataset.t2m_dataset, n_joints=22)

# 用 baseline 的 mu_t
result = np.load(
    "/root/autodl-tmp/motion-diffusion-model/new_results/posture_walking_膝超伸/comparison.npy",
    allow_pickle=True
).item()
mu_t = torch.from_numpy(result["motion_hml"]).float().cuda()

# ========== 测试 1：左右膝角度的逐帧统计 ==========
print("\n=== 测试 1：baseline 左右膝角度分布 ===")
mu_var = mu_t.detach().clone().requires_grad_(True)
q = fk_fn(mu_var)
left_angles  = signed_knee_angle(q[0], side="left")  * 180 / math.pi
right_angles = signed_knee_angle(q[0], side="right") * 180 / math.pi

print(f"左膝: mean={left_angles.mean():.1f}°  "
      f"min={left_angles.min():.1f}°  max={left_angles.max():.1f}°")
print(f"右膝: mean={right_angles.mean():.1f}°  "
      f"min={right_angles.min():.1f}°  max={right_angles.max():.1f}°")

# 步态周期内最大伸展时刻
left_peak  = left_angles.max().item()
right_peak = right_angles.max().item()
print(f"\n左膝峰值伸展: {left_peak:.1f}°")
print(f"右膝峰值伸展: {right_peak:.1f}°")
print("如果两者接近（差<5°），说明步态对称")
print("如果两者差距很大（>10°），说明 angle_fn 左右不对称")


# ========== 测试 2：guidance 梯度的左右分布 ==========
print("\n=== 测试 2：guidance 梯度对左右腿的影响 ===")
mu_var = mu_t.detach().clone().requires_grad_(True)
q = fk_fn(mu_var)

# 左膝 loss
loss_l = torch.nn.functional.relu(
    190 * math.pi / 180 - signed_knee_angle(q[0], side="left")
).mean()
loss_l.backward()
grad_l = mu_var.grad.clone()
print(f"左膝 loss 梯度 norm: {grad_l.norm().item():.4f}")

# 重新算右膝
mu_var = mu_t.detach().clone().requires_grad_(True)
q = fk_fn(mu_var)
loss_r = torch.nn.functional.relu(
    190 * math.pi / 180 - signed_knee_angle(q[0], side="right")
).mean()
loss_r.backward()
grad_r = mu_var.grad.clone()
print(f"右膝 loss 梯度 norm: {grad_r.norm().item():.4f}")

ratio = grad_l.norm().item() / max(grad_r.norm().item(), 1e-8)
print(f"\n左/右梯度比: {ratio:.2f}")
print("如果接近 1.0 → 梯度对称")
print("如果远离 1.0 → fk_fn 在左右腿上稀释程度不同")


# ========== 测试 3：左右膝符号判断的信号强度 ==========
print("\n=== 测试 3：判断超伸方向的 z_offset 分布 ===")
hip_l   = q[0, :, 1, :]   # (T, 3) left_hip
knee_l  = q[0, :, 4, :]
ankle_l = q[0, :, 7, :]
hip_r   = q[0, :, 2, :]
knee_r  = q[0, :, 5, :]
ankle_r = q[0, :, 8, :]

mid_z_l = (hip_l[:, 2] + ankle_l[:, 2]) / 2
mid_z_r = (hip_r[:, 2] + ankle_r[:, 2]) / 2
offset_l = (knee_l[:, 2] - mid_z_l).detach().cpu().numpy()
offset_r = (knee_r[:, 2] - mid_z_r).detach().cpu().numpy()

print(f"左膝 z_offset:  mean={offset_l.mean():+.4f}m  std={offset_l.std():.4f}")
print(f"右膝 z_offset:  mean={offset_r.mean():+.4f}m  std={offset_r.std():.4f}")
print()
print("正值 → 膝盖在前（屈曲）→ angle 保持 base_angle")
print("负值 → 膝盖在后（超伸）→ angle 转换为 2π-base_angle")
print()
print("如果两者均值都是正的且接近，符号判断对称")
print("如果一边经常负一边经常正，符号判断有偏差")
