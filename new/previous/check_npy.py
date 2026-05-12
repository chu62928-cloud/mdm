import numpy as np

def inspect_npy(file_path):
    print(f"========== 正在分析文件: {file_path} ==========\n")
    try:
        # allow_pickle=True 是为了防止数据被存成了 Python 字典对象
        data = np.load(file_path, allow_pickle=True)
        
        # 很多时候，字典会被包装在一个 0 维的 ndarray 中
        if isinstance(data, np.ndarray) and data.dtype == 'O':
            data = data.item()
            
        # 情况 1：数据是一个字典 (标准的数据集通常这么存，包含 poses, trans 等多项)
        if isinstance(data, dict):
            print("[结果] 文件是一个 字典 (Dictionary)")
            print(f"包含的键名 (Keys): {list(data.keys())}\n")
            for key, val in data.items():
                if isinstance(val, np.ndarray):
                    print(f" - 键 '{key}' | 形状 (Shape): {val.shape} | 数据类型: {val.dtype}")
                else:
                    print(f" - 键 '{key}' | 类型: {type(val)}")
                    
        # 情况 2：数据直接是一个纯多维数组 (纯粹的矩阵)
        elif isinstance(data, np.ndarray):
            print("[结果] 文件是一个 纯多维数组 (Ndarray)")
            print(f"数组形状 (Shape): {data.shape}")
            print(f"数据类型 (Dtype): {data.dtype}")
            print(f"数据极值: 最小值 {data.min():.4f}, 最大值 {data.max():.4f}")
            
        else:
            print(f"未知的数据结构: {type(data)}")
            
    except Exception as e:
        print(f"读取失败，请检查路径。错误信息: {e}")

if __name__ == "__main__":
    # 请确保 comparison.npy 和这个脚本在同一个文件夹，或者填入绝对路径
    inspect_npy("/root/autodl-tmp/motion-diffusion-model/new_results/posture_walking_骨盆前倾_18/comparison.npy")