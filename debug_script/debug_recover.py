import torch
import numpy as np
import sys
sys.path.insert(0, "/root/autodl-tmp/motion-diffusion-model")

from posture_guidance.mdm_integration import make_fk_fn
from data_loaders.get_data import get_dataset_loader

data = get_dataset_loader(
    name="humanml", batch_size=1, num_frames=120,
    split="test", hml_mode="text_only",
)
fk_fn = make_fk_fn(data.dataset.t2m_dataset, n_joints=22)

result = np.load(
    "./new_results/posture_exp_20260503_232934/comparison.npy",
    allow_pickle=True
).item()
mu_orig = torch.from_numpy(result["motion_hml"]).float().cuda()
q_orig  = fk_fn(mu_orig).detach()    # 原始关节坐标

# 测试 1：只改 position 维度（dim 4-66）
mu_pos_changed = mu_orig.clone()
mu_pos_changed[:, 4:67, :, :] += 0.5    # 大幅扰动 position
q_pos = fk_fn(mu_pos_changed).detach()
diff_pos = (q_pos - q_orig).abs().mean().item()

# 测试 2：只改 rotation 维度（dim 67-192）
mu_rot_changed = mu_orig.clone()
mu_rot_changed[:, 67:193, :, :] += 0.5   # 同样幅度扰动 rotation
q_rot = fk_fn(mu_rot_changed).detach()
diff_rot = (q_rot - q_orig).abs().mean().item()

print("=== recover_from_ric 灵敏度测试 ===")
print(f"扰动幅度: 0.5（在归一化空间）")
print(f"")
print(f"只改 position (dim 4-66)：    关节坐标变化 = {diff_pos:.4f} m")
print(f"只改 rotation (dim 67-192)：  关节坐标变化 = {diff_rot:.4f} m")
print(f"")
if diff_pos > diff_rot * 5:
    print("→ position 维度对关节坐标影响显著大于 rotation")
    print("  说明 recover_from_ric 主要用 position 重建")
    print("  问题不在维度选择，需要检查为什么推动 position 后骨盆角不变")
elif diff_rot > diff_pos * 5:
    print("→ rotation 维度对关节坐标影响显著大于 position")
    print("  说明 recover_from_ric 主要用 rotation 重建")
    print("  guidance 的梯度走 position 路径完全无效，必须走 rotation")
else:
    print("→ 两个维度都有显著影响")
    print("  位置和旋转都参与重建，guidance 走 position 应该有效")
    print("  问题可能在于 MDM 的去噪步骤把 position 修改'拉回'了")
