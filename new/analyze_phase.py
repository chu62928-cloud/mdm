import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks

# =====================================================================
# 1. 英文映射词典
# =====================================================================
POSTURE_MAP_ENG = {
    '骨盆前倾': 'Anterior Pelvic Tilt (APT)',
    '膝关节超伸': 'Knee Hyperextension',
    '脊柱后凸': 'Thoracic Kyphosis',
    '前倾头部姿势': 'Forward Head Posture'
}

def run_gait_analysis(file_path="./output/exp/comparison.npy", batch_idx=0):
    if not os.path.exists(file_path):
        print(f"Error: 找不到文件 {file_path}")
        return
        
    data = np.load(file_path, allow_pickle=True).item()
    
    text_prompt = data.get("text_prompt", "Unknown Prompt")
    posture_raw = data.get("posture_instructions", "Unknown Posture")
    
    # 提取并翻译标题
    if isinstance(posture_raw, list):
        posture_eng_list = [POSTURE_MAP_ENG.get(p, p) for p in posture_raw]
        posture_str = ", ".join(posture_eng_list)
    else:
        posture_str = POSTURE_MAP_ENG.get(posture_raw, posture_raw)

    baseline_xyz = data["motion_xyz"]
    guided_xyz = data["motion_xyz_guided"]
        
    q_base = np.transpose(baseline_xyz[batch_idx], (2, 0, 1))  # (120, 22, 3)
    q_guid = np.transpose(guided_xyz[batch_idx], (2, 0, 1))  # (120, 22, 3)

    # =================================================================
    # 核心修改区：改为寻找摆动期最高点 (Mid-Swing Peaks)
    # =================================================================
    left_ankle_idx = 7 
    frames = q_base.shape[0]
    
    # 直接使用真实高度 Y，不加负号！
    lh_base_y = q_base[:, left_ankle_idx, 1] 
    lh_guid_y = q_guid[:, left_ankle_idx, 1]
    
    # 寻找最高点：
    # distance=20: 走一步（脚从离地到再次离地）至少20帧
    # prominence=0.05: 脚抬起的高度必须比周围显著高出 0.05，过滤脚底打滑的噪声
    peaks_base, _ = find_peaks(lh_base_y, distance=20, prominence=0.05)
    peaks_guid, _ = find_peaks(lh_guid_y, distance=20, prominence=0.05)
    
    # =================================================================
    # 画图展示
    # =================================================================
    plt.figure(figsize=(12, 5))
    # 注意 label 变了，现在是真实物理高度
    plt.plot(range(frames), lh_base_y, label='Baseline Left Ankle Height (Y)', color='royalblue', alpha=0.8, linewidth=2)
    plt.plot(range(frames), lh_guid_y, label='Guided Left Ankle Height (Y)', color='crimson', alpha=0.8, linewidth=2)
    
    # 标记最高点
    plt.plot(peaks_base, lh_base_y[peaks_base], "x", color='navy', markersize=10, markeredgewidth=2, label='Baseline Mid-Swing')
    plt.plot(peaks_guid, lh_guid_y[peaks_guid], "o", color='darkred', markersize=8, label='Guided Mid-Swing')
    
    plt.title(f"Gait Phase Analysis: {text_prompt} ({posture_str})", fontsize=12)
    plt.xlabel("Frame", fontsize=10)
    plt.ylabel("Actual Ankle Height (Y)", fontsize=10)
    plt.legend(loc='upper right')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    output_dir = "/root/autodl-tmp/motion-diffusion-model/output/ablation_walking_apt_seed42/v1_mu_sgd/viz"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    output_img = os.path.join(output_dir, "gait_phase_analysis.png")
    plt.savefig(output_img, dpi=300)
    print(f">>> 分析图表已成功保存至: {output_img}")
    plt.close()
    
    # =================================================================
    # 打印定量分析
    # =================================================================
    print("\n" + "=" * 50)
    print("             步态相位定量分析报告 (基于摆动期峰值)")
    print("=" * 50)
    
    print(f"Baseline 峰值检测帧: {peaks_base}")
    print(f"Guided   峰值检测帧: {peaks_guid}\n")
    
    if len(peaks_base) > 1 and len(peaks_guid) > 1:
        avg_stride_base = np.mean(np.diff(peaks_base))
        avg_stride_guid = np.mean(np.diff(peaks_guid))
        print(f"Baseline 平均单步周期: {avg_stride_base:.2f} 帧")
        print(f"Guided   平均单步周期: {avg_stride_guid:.2f} 帧")
        
        shift = peaks_guid[0] - peaks_base[0]
        print(f"首步启动相位差 (Guided - Baseline): {shift:+.1f} 帧")
        
        if abs(avg_stride_base - avg_stride_guid) > 1.5:
            print("\n[ 诊断结论 ]:")
            print(">>> 引导力显著改变了步态周期！")
            if avg_stride_guid > avg_stride_base:
                print(">>> 现象：Guided 的步态周期变长，步频变慢。")
            else:
                print(">>> 现象：Guided 的步态周期变短，步频变快。")
            print(">>> 结论：属于生物力学中的代偿效应 (Compensation)。")
        else:
            print("\n[ 诊断结论 ]:")
            print(">>> 步态周期基本保持一致，未发生明显的时序代偿。")
    else:
        print("未检测到至少两个步态周期。如果动作是单步，可以通过上方的峰值检测帧判断动作时间点。")
    print("=" * 50)

if __name__ == "__main__":
    run_gait_analysis(file_path="/root/autodl-tmp/motion-diffusion-model/output/ablation_walking_apt_seed42/v1_mu_sgd/comparison.npy", batch_idx=0)