import torch
import math
import sys
sys.path.insert(0, "/root/autodl-tmp/motion-diffusion-model")
from posture_guidance.angle_ops import signed_knee_angle


def make_pose(thigh_y, shank_y, knee_z=0.0):
    """
    构造一个简化姿态：
        髋 (0, 0.9, 0)
        膝 (0, knee_y, knee_z)
        踝 (0, ankle_y, 0)
    
    通过调整 knee_z 模拟膝盖前突或后突
    """
    q = torch.zeros(22, 3)
    
    # 左侧
    q[1]  = torch.tensor([-0.1, 0.9,   0.0])     # left_hip
    q[4]  = torch.tensor([-0.1, thigh_y, knee_z])  # left_knee
    q[7]  = torch.tensor([-0.1, shank_y, 0.0])   # left_ankle
    
    # 右侧（与左侧对称）
    q[2]  = torch.tensor([ 0.1, 0.9,   0.0])     # right_hip
    q[5]  = torch.tensor([ 0.1, thigh_y, knee_z])  # right_knee
    q[8]  = torch.tensor([ 0.1, shank_y, 0.0])   # right_ankle
    
    return q.unsqueeze(0)  # (1, 22, 3)


print("=== signed_knee_angle 对称性测试 ===\n")

test_cases = [
    ("完全直立", 0.5, 0.0,  0.0),
    ("正常弯曲（膝盖向前）", 0.5, 0.0,  0.1),
    ("超伸（膝盖向后）",     0.5, 0.0, -0.05),
]

for label, thigh_y, ankle_y, knee_z in test_cases:
    q = make_pose(thigh_y, ankle_y, knee_z)
    
    angle_l = signed_knee_angle(q, side="left")  * 180 / math.pi
    angle_r = signed_knee_angle(q, side="right") * 180 / math.pi
    
    print(f"{label}（knee_z={knee_z:+.2f}）:")
    print(f"  左膝: {angle_l.item():.2f}°")
    print(f"  右膝: {angle_r.item():.2f}°")
    print(f"  差值: {(angle_l - angle_r).item():.2f}°")
    print()

print("如果 '完全直立' 时左右膝角度差很大，说明算法有左右不对称的 bug")
print("如果 '正常弯曲' 和 '超伸' 给出相同符号的角度，说明 cross_z 判断失效")
