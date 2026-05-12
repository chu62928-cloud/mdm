import cv2
import os

def extract_keyframes(video_path, target_frames, output_dir="keyframes"):
    """
    从视频中提取指定帧数的高清图片。
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: 无法打开视频文件 {video_path}")
        return

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    for frame_idx in target_frames:
        if frame_idx >= total_frames:
            print(f"Warning: 请求的帧 {frame_idx} 超出视频总帧数 {total_frames}")
            continue
            
        # 设置读取位置到指定帧
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        
        if ret:
            output_path = os.path.join(output_dir, f"frame_{frame_idx:03d}.png")
            # 保存为无损 PNG
            cv2.imwrite(output_path, frame, [cv2.IMWRITE_PNG_COMPRESSION, 0])
            print(f"Saved: {output_path}")
        else:
            print(f"Error: 无法读取帧 {frame_idx}")

    cap.release()

# 使用示例
video_file = "/root/autodl-tmp/motion-diffusion-model/output/ablation_walking_apt_seed42/v1_mu_sgd/viz/anatomical_animation.mp4" 
# 提取第1帧(站立), 第29帧, 第54帧(支撑期)等
frames_to_extract = [1, 29, 54, 89] 
output_dir = "/root/autodl-tmp/motion-diffusion-model/output/ablation_walking_apt_seed42/v1_mu_sgd/viz"
extract_keyframes(video_file, frames_to_extract,output_dir)