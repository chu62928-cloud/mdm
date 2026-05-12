import numpy as np
import os

print("正在读取 .npy...")
data = np.load("/root/autodl-tmp/motion-diffusion-model/save/263_batch_single_outputs/0001_a_person_walks_forward/results.npy", allow_pickle=True).item()

# ================= 核心修复 =================
# 必须使用 263 维的特征向量
motion_data = data.get('motion_hml')
# ============================================

mdm_format = {
    'motion': motion_data,
    'text': [data.get('text_prompt', 'unnamed_motion')],
    'lengths': [data.get('motion_length', 120)],
    'num_samples': 1,
    'num_repetitions': 1
}

np.save("/root/autodl-tmp/motion-diffusion-model/save/263_batch_single_outputs/0001_a_person_walks_forward/results.npy", mdm_format)
print(f"成功生成 results.npy! 当前数据形状: {motion_data.shape}")