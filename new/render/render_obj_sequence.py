import bpy
import os
import math
from mathutils import Vector

# ================= 1. 配置路径 =================
# 填入你 .obj 文件所在的文件夹路径 (请确保文件夹里只有这一个动作的 obj 序列)
OBJ_DIR = os.path.abspath("/root/autodl-tmp/motion-diffusion-model/save/263_batch_single_outputs/0001_a_person_walks_forward/sample00_rep00_obj") 
OUT_DIR = os.path.abspath("/root/autodl-tmp/motion-diffusion-model/save/263_batch_single_outputs/0001_a_person_walks_forward/mesh_render_output")
os.makedirs(OUT_DIR, exist_ok=True)

# ================= 2. 初始化环境 =================
bpy.ops.wm.read_factory_settings(use_empty=True)
scene = bpy.context.scene
scene.render.engine = 'BLENDER_EEVEE_NEXT' # 5090 用这个极速出图
scene.eevee.use_gtao = True 
scene.render.resolution_x = 1920
scene.render.resolution_y = 1080

# ================= 3. 建立材质 =================
mat = bpy.data.materials.new(name="MeshMat")
mat.use_nodes = True
mat.node_tree.nodes["Principled BSDF"].inputs['Base Color'].default_value = (0.2, 0.5, 0.8, 1) # 蓝色

# ================= 4. 导入并设置自动对焦 =================
obj_files = sorted([f for f in os.listdir(OBJ_DIR) if f.endswith('.obj')])
num_frames = len(obj_files)
scene.frame_end = num_frames - 1

imported_objs = []
for i, filename in enumerate(obj_files):
    bpy.ops.wm.obj_import(filepath=os.path.join(OBJ_DIR, filename))
    obj = bpy.context.selected_objects[0]
    obj.data.materials.append(mat)
    bpy.ops.object.shade_smooth()
    
    # 初始状态全隐藏
    obj.hide_viewport = True
    obj.hide_render = True
    obj.keyframe_insert(data_path="hide_viewport", frame=0)
    obj.keyframe_insert(data_path="hide_render", frame=0)
    
    # 仅在当前帧显示
    obj.hide_viewport = False
    obj.hide_render = False
    obj.keyframe_insert(data_path="hide_viewport", frame=i)
    obj.keyframe_insert(data_path="hide_render", frame=i)
    
    # 过了这帧立刻消失
    obj.hide_viewport = True
    obj.hide_render = True
    obj.keyframe_insert(data_path="hide_viewport", frame=i+1)
    obj.keyframe_insert(data_path="hide_render", frame=i+1)
    imported_objs.append(obj)

# ================= 5. 相机与灯光自动布局 =================
# 创建地面
bpy.ops.mesh.primitive_plane_add(size=100)

# 自动计算所有模型的中心位置，把相机搬过去
all_locations = [o.location for o in imported_objs]
center_loc = sum(all_locations, Vector((0,0,0))) / len(all_locations)

# 创建相机
bpy.ops.object.camera_add(location=(center_loc.x, center_loc.y - 5, center_loc.z + 1.5))
cam = bpy.context.object
scene.camera = cam
# 让相机始终盯着模型的中心点
track = cam.constraints.new(type='TRACK_TO')
track.target = imported_objs[0] # 盯着第一个模型
track.track_axis = 'TRACK_NEGATIVE_Z'
track.up_axis = 'UP_Y'

# 添加灯光
bpy.ops.object.light_add(type='AREA', location=(center_loc.x + 2, center_loc.y - 2, 5))
bpy.context.object.data.energy = 1000

# ================= 6. 渲染 =================
scene.render.filepath = os.path.join(OUT_DIR, "frame_")
bpy.ops.render.render(animation=True)