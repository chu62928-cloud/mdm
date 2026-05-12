import torch
import math
import sys
sys.path.insert(0, "/root/autodl-tmp/motion-diffusion-model")
from posture_guidance.angle_ops import pelvis_tilt_angle

def make_pose(tilt_offset):
    """
    构造一个简化姿态。
    tilt_offset > 0: 让 spine1 相对 hip_center 向"前方"偏移
    """
    q = torch.zeros(1, 22, 3)
    # SMPL/HumanML3D: y=高度，z=前后（+z 是前方）
    q[0, 0]  = torch.tensor([0.0, 0.9, 0.0])                    # pelvis
    q[0, 1]  = torch.tensor([-0.1, 0.85, 0.0])                  # left_hip
    q[0, 2]  = torch.tensor([ 0.1, 0.85, 0.0])                  # right_hip
    q[0, 3]  = torch.tensor([0.0, 1.1, tilt_offset])            # spine1（关键）
    # 其他关节简单填充
    for j in range(4, 22):
        q[0, j] = torch.tensor([0.0, 1.2 + j * 0.05, 0.0])
    return q

print("=== pelvis_tilt_angle 符号测试 ===\n")

for offset, label in [
    (-0.2, "spine1 后偏（应该是后倾，负值）"),
    ( 0.0, "spine1 正上方（应该是 0°）"),
    ( 0.2, "spine1 前偏（应该是前倾，正值）"),
]:
    q = make_pose(offset)
    angle_rad = pelvis_tilt_angle(q[0])
    angle_deg = (angle_rad * 180 / math.pi).item() if angle_rad.dim() == 0 \
                else (angle_rad * 180 / math.pi).mean().item()
    print(f"{label}")
    print(f"  spine1 z-offset = {offset:+.2f}m")
    print(f"  算出角度        = {angle_deg:+.2f}°")
    print()
