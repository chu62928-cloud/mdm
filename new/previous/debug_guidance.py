# 在项目根目录下运行：python debug_guidance.py
import torch
import math
import sys
sys.path.insert(0, "/root/autodl-tmp/motion-diffusion-model")

from posture_guidance.controller import PostureGuidance
from posture_guidance.angle_ops import pelvis_tilt_angle
from posture_guidance.registry import POSTURE_REGISTRY

# 1. 确认注册表里的值
spec = POSTURE_REGISTRY["骨盆前倾"]
print(f"=== registry 当前配置 ===")
print(f"target_deg  : {spec.target_deg}")
print(f"base_weight : {spec.base_weight}")
print(f"schedule    : {spec.schedule}")
print(f"angle_fn    : {spec.angle_fn.__name__}")

# 2. 构造一个"直立站姿"的假关节坐标，测试 guidance 是否能推动它
# 22 个关节，按近似直立姿态排列
q = torch.zeros(1, 1, 22, 3)
# pelvis=0, left_hip=1, right_hip=2, spine1=3
q[0, 0, 0] = torch.tensor([0.0, 0.9,  0.0])   # pelvis
q[0, 0, 1] = torch.tensor([-0.1, 0.8, 0.0])   # left_hip
q[0, 0, 2] = torch.tensor([ 0.1, 0.8, 0.0])   # right_hip
q[0, 0, 3] = torch.tensor([0.0, 1.1,  0.0])   # spine1
# 其他关节填充合理值（不影响骨盆角的计算）
for j in range(4, 22):
    q[0, 0, j] = torch.tensor([0.0, float(j) * 0.05, 0.0])

angle_before = pelvis_tilt_angle(q[0]) * 180 / math.pi
print(f"\n=== 测试前骨盆角 ===")
print(f"angle (deg): {angle_before.mean().item():.2f}°")

# 3. 运行 guidance，看角度是否被推动
guidance = PostureGuidance(["骨盆前倾"], verbose=True)
q_var = q.detach().clone().requires_grad_(True)
loss = guidance(q_var, t=0, T=1000)
print(f"\nloss value: {loss.item():.6f}")
loss.backward()
print(f"grad norm:  {q_var.grad.norm().item():.6f}")

if q_var.grad.norm().item() < 1e-8:
    print("\n✗ 梯度为 0！guidance 完全没有工作")
    print("  → 检查 angle_fn 是否可微，loss 是否为 0")
else:
    print(f"\n✓ 梯度正常，guidance 可以工作")