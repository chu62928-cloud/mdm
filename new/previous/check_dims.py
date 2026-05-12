import numpy as np
import sys

# 你可以把路径写死，也可以在命令行传进来
npy_path = "/root/autodl-tmp/motion-diffusion-model/save/263_batch_single_outputs/0001_a_person_walks_forward/results.npy" 

print(f"========== 正在解剖文件: {npy_path} ==========")
try:
    data = np.load(npy_path, allow_pickle=True).item()
    
    # 统计是否找到符合要求的维度
    found_263 = False
    
    for key, value in data.items():
        if isinstance(value, np.ndarray):
            shape_str = str(value.shape)
            print(f"🔑 键名: {key:<25} | 📊 形状: {shape_str}")
            
            # MDM 的 SMPL 渲染强依赖 263 维特征
            if 263 in value.shape:
                found_263 = True
                print(f"   ---> [✅ 核心目标发现!] 这是一个符合 SMPL 渲染要求的特征向量！")
                
        elif isinstance(value, list):
            print(f"🔑 键名: {key:<25} | 📝 类型: 列表 (List) | 长度: {len(value)}")
        else:
            print(f"🔑 键名: {key:<25} | 🏷️ 类型: {type(value).__name__} | 值: {value}")

    print("==================================================")
    if found_263:
        print("🎉 诊断结论：文件中包含 263 维的特征向量，完全满足 render_mesh.py 的输入要求！")
        print("请在伪装脚本 (adapt.py) 中，将 motion_data 设为上面标有 [✅] 的那个键名。")
    else:
        print("⚠️ 诊断结论：未找到 263 维数据！这个文件可能被裁剪过，或者只是纯坐标输出。")

except Exception as e:
    print(f"读取失败，错误信息: {e}")