import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks

def analyze_gait_phase(q_baseline, q_guided, left_heel_idx, right_heel_idx):
    """
    通过分析脚后跟的高度来寻找落脚点 (Heel Strikes)，诊断步态相位。
    假设 q_baseline/guided 的 shape 为 (T, J, 3)，且 Y 轴是垂直地面的高度。
    """
    frames = q_baseline.shape[0]
    
    # 提取脚后跟高度 (越小说明越接近地面)
    # 我们对其取负值，这样落脚点就是局部极大值 (Peaks)
    lh_base_y = -q_baseline[:, left_heel_idx, 1] 
    lh_guid_y = -q_guided[:, left_heel_idx, 1]
    
    # 寻找落脚点 (Heel Strike) 的帧索引
    # distance=10 防止在一次落脚中检测出多个峰值
    peaks_base, _ = find_peaks(lh_base_y, distance=10)
    peaks_guid, _ = find_peaks(lh_guid_y, distance=10)
    
    # 可视化对比
    plt.figure(figsize=(12, 4))
    plt.plot(range(frames), lh_base_y, label='Baseline Left Heel (-Y)', color='gray')
    plt.plot(range(frames), lh_guid_y, label='Guided Left Heel (-Y)', color='pink')
    
    # 标记落脚点
    plt.plot(peaks_base, lh_base_y[peaks_base], "x", color='black', markersize=8, label='Baseline Heel Strike')
    plt.plot(peaks_guid, lh_guid_y[peaks_guid], "o", color='red', markersize=6, label='Guided Heel Strike')
    
    plt.title("Gait Phase Analysis: Left Heel Strike Timing")
    plt.xlabel("Frame")
    plt.ylabel("Inverted Height (Approaching Ground)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()
    
    # 计算均值步态周期 (Stride Time)
    if len(peaks_base) > 1 and len(peaks_guid) > 1:
        avg_stride_base = np.mean(np.diff(peaks_base))
        avg_stride_guid = np.mean(np.diff(peaks_guid))
        print(f"Baseline 平均步态周期: {avg_stride_base:.2f} 帧")
        print(f"Guided   平均步态周期: {avg_stride_guid:.2f} 帧")
        
        shift = peaks_guid[0] - peaks_base[0]
        print(f"初始相位差 (第一步落脚差): {shift} 帧")
        
        if abs(avg_stride_base - avg_stride_guid) > 1.0:
            print("结论: 引导不仅改变了姿态，还改变了行走的速度/步频！这是一个真实的代偿效应，不是渲染伪影。")
        else:
            print("结论: 步频基本一致。如果视觉上存在长期漂移，可能是因为初始相位有一点微小的偏差累积。")

# 假设 10 是左脚跟，11 是右脚跟 (具体索引请参考 HumanML3D 的骨架定义)
analyze_gait_phase(q_baseline, q_guided, left_heel_idx=10, right_heel_idx=11)