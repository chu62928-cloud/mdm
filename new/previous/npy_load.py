import numpy as np

# 加载文件
data = np.load("/root/autodl-tmp/motion-diffusion-model/new_results/posture_walking_骨盆前倾_18/comparison.npy", allow_pickle=True).item()

# 查看所有 key
print(data.keys())