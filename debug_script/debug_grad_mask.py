import torch
import math
import numpy as np
import sys
sys.path.insert(0, "/root/autodl-tmp/motion-diffusion-model")

from posture_guidance.angle_ops import pelvis_tilt_angle
from posture_guidance.mdm_integration import make_fk_fn
from data_loaders.get_data import get_dataset_loader

print("Loading dataset...")
data = get_dataset_loader(
    name="humanml", batch_size=1, num_frames=120,
    split="test", hml_mode="text_only",
)
fk_fn = make_fk_fn(data.dataset.t2m_dataset, n_joints=22)

result = np.load(
    "./new_results/posture_exp_20260503_232934/comparison.npy",
    allow_pickle=True
).item()
mu_t = torch.from_numpy(result["motion_hml"]).float().cuda()
# shape: (1, 263, 1, 120)

mu_var = mu_t.detach().clone().requires_grad_(True)
q = fk_fn(mu_var)
angle = pelvis_tilt_angle(q[0]).mean()
angle.backward()

grad = mu_var.grad.squeeze(0).squeeze(1)   # (263, 120)
grad_per_dim = grad.abs().mean(dim=1)       # (263,)

# 263 维分解：
# 0       : root rot vel
# 1-2     : root linear vel xz
# 3       : root y
# 4-66    : 21 joints × 3 (relative positions)
# 67-192  : 21 joints × 6 (rotations)
# 193-258 : 22 joints × 3 (velocities)
# 259-262 : 4 foot contacts

print("=== 梯度按维度组分布 ===")
print(f"root rot vel       (dim 0):     {grad_per_dim[0:1].mean():.6f}")
print(f"root linear vel xz (dim 1-2):   {grad_per_dim[1:3].mean():.6f}")
print(f"root y             (dim 3):     {grad_per_dim[3:4].mean():.6f}")
print(f"joint positions    (dim 4-66):  {grad_per_dim[4:67].mean():.6f}")
print(f"joint rotations    (dim 67-192): {grad_per_dim[67:193].mean():.6f}")
print(f"joint velocities   (dim 193-258):{grad_per_dim[193:259].mean():.6f}")
print(f"foot contacts      (dim 259-262):{grad_per_dim[259:263].mean():.6f}")

# 找梯度最大的 10 个维度
top_idx = grad_per_dim.argsort(descending=True)[:10]
print(f"\n=== 梯度最大的 10 个维度 ===")
for i, idx in enumerate(top_idx):
    idx_int = idx.item()
    if idx_int < 1:           name = "root rot vel"
    elif idx_int < 3:          name = f"root linear vel ({idx_int})"
    elif idx_int < 4:          name = "root y"
    elif idx_int < 67:         
        joint_id = (idx_int - 4) // 3
        axis = ['x','y','z'][(idx_int - 4) % 3]
        name = f"joint_{joint_id+1} pos {axis}"
    elif idx_int < 193:
        joint_id = (idx_int - 67) // 6
        name = f"joint_{joint_id+1} rot dim{(idx_int-67)%6}"
    elif idx_int < 259:
        joint_id = (idx_int - 193) // 3
        axis = ['x','y','z'][(idx_int - 193) % 3]
        name = f"joint_{joint_id} vel {axis}"
    else:
        name = f"foot contact {idx_int - 259}"
    print(f"  {i+1}. dim {idx_int:3d}: {grad_per_dim[idx_int]:.6f}  ({name})")

# 计算梯度的"集中度"
total = grad_per_dim.sum()
top_10_share = grad_per_dim[top_idx].sum() / total
print(f"\n前 10 维梯度占总梯度的 {top_10_share*100:.1f}%")
print("（如果 < 30% 说明梯度极度分散，guidance 难以集中发力）")
