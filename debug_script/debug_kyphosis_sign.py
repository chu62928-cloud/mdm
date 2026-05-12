import torch, math, sys
sys.path.insert(0, "/root/autodl-tmp/motion-diffusion-model")
from posture_guidance.angle_ops import spine3_posterior_offset
import numpy as np

data = np.load(
    "./new_results/posture_exp_20260504_150636/comparison.npy",
    allow_pickle=True
).item()

q = torch.from_numpy(data["motion_xyz"][0]).permute(2,0,1).float()
# q: (T, 22, 3)

offset = spine3_posterior_offset(q)   # (T,)
print(f"mean: {offset.mean().item():+.4f} m")
print(f"max:  {offset.max().item():+.4f} m")
print(f"min:  {offset.min().item():+.4f} m")
print()

# 直接打印 spine3 的 z 位置 vs spine1 的 z 位置
spine1_z = q[:, 3,  2]   # spine1 z
spine3_z = q[:, 9,  2]   # spine3 z
neck_z   = q[:, 12, 2]   # neck z

print(f"spine1 z mean: {spine1_z.mean().item():+.4f}")
print(f"spine3 z mean: {spine3_z.mean().item():+.4f}")
print(f"neck   z mean: {neck_z.mean().item():+.4f}")
print()
print(f"spine3 - spine1 (z): {(spine3_z - spine1_z).mean().item():+.4f}")
print("正值=spine3在spine1前方，负值=spine3在spine1后方")
