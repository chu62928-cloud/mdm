import bpy
import numpy as np
import math
from mathutils import Vector
import os

# ================= 1. 环境与参数设置 =================
NPY_PATH = "/root/autodl-tmp/motion-diffusion-model/new_results/posture_walking_骨盆前倾_18/comparison.npy"
OUT_DIR = os.path.abspath("/root/autodl-tmp/motion-diffusion-model/new_results/posture_walking_骨盆前倾_18")
os.makedirs(OUT_DIR, exist_ok=True)

# 材质颜色 (论文常用色系)
COLOR_BONE = (0.9, 0.9, 0.9, 1)    # 哑光白
COLOR_JOINT = (0.2, 0.2, 0.2, 1)   # 深灰
COLOR_GROUND = (0.95, 0.95, 0.95, 1) # 背景/地面

# ================= 2. 加载数据 =================
data = np.load(NPY_PATH, allow_pickle=True).item()
motion_xyz = data.get('motion_xyz_guided', data.get('motion_xyz'))
xyz_data = motion_xyz[0].transpose(2, 0, 1) # (120, 22, 3)
xyz_data[..., [1, 2]] = xyz_data[..., [2, 1]]
num_frames = xyz_data.shape[0]

# ================= 3. 场景初始化 (Cycles 高级设置) =================
bpy.ops.wm.read_factory_settings(use_empty=True)
scene = bpy.context.scene
scene.frame_end = num_frames
scene.render.engine = 'BLENDER_EEVEE_NEXT'
# 5090 性能极强，直接开高采样
scene.cycles.samples = 128 
scene.cycles.use_denoising = True
scene.render.resolution_x = 1920
scene.render.resolution_y = 1080
# --- 强制激活 RTX 5090 (OptiX 引擎) ---
scene.cycles.device = 'GPU'
prefs = bpy.context.preferences
cprefs = prefs.addons['cycles'].preferences
cprefs.compute_device_type = 'OPTIX'

# 刷新并获取设备列表
cprefs.get_devices()

# 遍历所有设备，只勾选 OptiX 对应的 GPU，关闭 CPU
for device in cprefs.devices:
    if device.type == 'OPTIX':
        device.use = True
        print(f"已成功激活 GPU 加速: {device.name}")
    elif device.type == 'CPU':
        device.use = False
# --------------------------------------

# ================= 4. 建立哑光材质 =================
def create_matte_mat(name, color):
    mat = bpy.data.materials.new(name=name)
    mat.use_nodes = True
    nodes = mat.node_tree.nodes
    bsdf = nodes.get("Principled BSDF")
    
    # 设置基础颜色
    bsdf.inputs['Base Color'].default_value = color
    
    # 关键修改：针对 Blender 4.0+ 的参数名变更
    # 1. 粗糙度
    bsdf.inputs['Roughness'].default_value = 0.9 
    
    # 2. 高光 (旧版叫 'Specular', 4.0+ 叫 'Specular IOR Level')
    if 'Specular IOR Level' in bsdf.inputs:
        bsdf.inputs['Specular IOR Level'].default_value = 0.1
    elif 'Specular' in bsdf.inputs:
        bsdf.inputs['Specular'].default_value = 0.1
        
    return mat

mat_bone = create_matte_mat("BoneMat", COLOR_BONE)
mat_joint = create_matte_mat("JointMat", COLOR_JOINT)
mat_ground = create_matte_mat("GroundMat", COLOR_GROUND)

# ================= 5. 骨架拓扑 (HumanML3D) =================
parents = [-1, 0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 9, 9, 12, 13, 14, 16, 17, 18, 19]

# 生成几何体
spheres = []
for i in range(22):
    bpy.ops.mesh.primitive_uv_sphere_add(radius=0.035, segments=32, ring_count=16)
    obj = bpy.context.object
    obj.data.materials.append(mat_joint)
    bpy.ops.object.shade_smooth()
    spheres.append(obj)

bones = []
for i in range(1, 22):
    bpy.ops.mesh.primitive_cylinder_add(radius=0.018, depth=2)
    obj = bpy.context.object
    obj.data.materials.append(mat_bone)
    bpy.ops.object.shade_smooth()
    bones.append(obj)

# ================= 6. 动画关键帧 =================
for f in range(num_frames):
    positions = xyz_data[f]
    for i in range(22):
        spheres[i].location = Vector(positions[i])
        spheres[i].keyframe_insert(data_path="location", frame=f)
    for i in range(1, 22):
        p_idx = parents[i]
        p1, p2 = Vector(positions[p_idx]), Vector(positions[i])
        bone = bones[i-1]
        v = p2 - p1
        bone.location = (p1 + p2) / 2
        bone.rotation_mode = 'QUATERNION'
        bone.rotation_quaternion = Vector((0, 0, 1)).rotation_difference(v)
        bone.scale = (1, 1, v.length / 2)
        bone.keyframe_insert(data_path="location", frame=f)
        bone.keyframe_insert(data_path="rotation_quaternion", frame=f)
        bone.keyframe_insert(data_path="scale", frame=f)

# ================= 7. 论文级构图与光照 =================
# 地面 (接收阴影)
bpy.ops.mesh.primitive_plane_add(size=100, location=(0, 0, 0))
ground = bpy.context.object
ground.data.materials.append(mat_ground)

# 顶部三点光源系统 (提供立体感和柔和阴影)
lights_data = [
    ((5, -5, 10), 2000),  # 主光
    ((-5, -5, 8), 1000),  # 补光
    ((0, 10, 5), 500)     # 背光
]
for loc, energy in lights_data:
    bpy.ops.object.light_add(type='AREA', location=loc)
    light = bpy.context.object
    light.data.size = 5 # 增大尺寸使阴影更柔和
    light.data.energy = energy

# 摄像机 (45度侧俯视，论文标准视角)
bpy.ops.object.camera_add(location=(4, -4, 2.5))
cam = bpy.context.object
scene.camera = cam
# 自动盯住骨盆
track = cam.constraints.new(type='TRACK_TO')
track.target = spheres[0]
track.track_axis = 'TRACK_NEGATIVE_Z'
track.up_axis = 'UP_Y'

# ================= 8. 执行渲染 =================
scene.render.filepath = os.path.join(OUT_DIR, "frame_")
bpy.ops.render.render(animation=True)