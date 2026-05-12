import torch
import math
import numpy as np
import sys
sys.path.insert(0, "/root/autodl-tmp/motion-diffusion-model")

from posture_guidance.angle_ops import pelvis_tilt_angle
from posture_guidance.joint_indices import JOINT_IDX

# 用真实数据
data = np.load(
    "./new_results/posture_exp_20260503_232934/comparison.npy",
    allow_pickle=True
).item()

xyz = data["motion_xyz"][0]   # (J, 3, T)
T = xyz.shape[-1]

# 取第 60 帧作为代表（动作中段）
frame_idx = 60
q = torch.from_numpy(xyz[:, :, frame_idx]).float()  # (22, 3)

print(f"=== 第 {frame_idx} 帧的关键关节坐标 ===")
print(f"pelvis    [y,z]: y={q[0, 1]:+.3f}, z={q[0, 2]:+.3f}")
print(f"left_hip  [y,z]: y={q[1, 1]:+.3f}, z={q[1, 2]:+.3f}")
print(f"right_hip [y,z]: y={q[2, 1]:+.3f}, z={q[2, 2]:+.3f}")
print(f"spine1    [y,z]: y={q[3, 1]:+.3f}, z={q[3, 2]:+.3f}")
print(f"spine3    [y,z]: y={q[9, 1]:+.3f}, z={q[9, 2]:+.3f}")
print(f"head      [y,z]: y={q[15, 1]:+.3f}, z={q[15, 2]:+.3f}")

# 关键判断：spine1 相对 hip_center 的 z 偏移方向
hip_center = (q[1] + q[2]) / 2
pelvis_to_spine = q[3] - hip_center
print(f"\n=== pelvis_to_spine 向量 ===")
print(f"x:{pelvis_to_spine[0]:+.4f}, y:{pelvis_to_spine[1]:+.4f}, z:{pelvis_to_spine[2]:+.4f}")
print(f"z 分量 > 0 表示 spine1 在骨盆中心的 +z 方向（前方）")
print(f"z 分量 < 0 表示 spine1 在骨盆中心的 -z 方向（后方）")

# 看人物前进方向（前 vs 后帧的 pelvis 位移）
pelvis_t0   = xyz[0, :, 0]
pelvis_tend = xyz[0, :, -1]
disp = pelvis_tend - pelvis_t0
print(f"\n=== 人物行走方向 ===")
print(f"位移 z 分量: {disp[2]:+.3f}m")
print(f"如果是正值，人物朝 +z 方向走（即 +z 是前方）")

# 算这一帧的骨盆角
angle = pelvis_tilt_angle(q) * 180 / math.pi
print(f"\n=== 算出的骨盆角 ===")
print(f"angle: {angle.item():+.2f}°")

# 关键判断
print(f"\n=== 判断 ===")
print(f"人朝 +z 方向走（前方=+z）。")
print(f"走路时人的脊柱（spine1）应该相对骨盆中心**轻微向后**（避免向前栽倒）。")
print(f"也可能 spine1 几乎在 pelvis 正上方（z=0）。")
print(f"")
print(f"  如果 pelvis_to_spine 的 z < 0（spine1 在 pelvis 后方）:")
print(f"      正常走路应该是 z<0，骨盆此时应该接近直立或轻度前倾（+5°）。")
print(f"      pelvis_tilt_angle 算成 -6°，**符号是反的**。")
print(f"")
print(f"  如果 pelvis_to_spine 的 z > 0（spine1 在 pelvis 前方）:")
print(f"      说明 z 在 SMPL 里的实际朝向和 +z 前进方向相反。")
print(f"      需要把 angle_fn 里的 z 取负。")
