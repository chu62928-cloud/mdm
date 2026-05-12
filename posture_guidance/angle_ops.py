"""
Differentiable joint angle computations.
所有函数返回的张量都保持梯度，用于 backprop 到 μₜ。
"""
import math
import torch
import torch.nn.functional as F
from .joint_indices import get_joint_idx


EPS = 1e-7


def three_point_angle(
    q: torch.Tensor,
    joint_a: str,
    joint_b: str,
    joint_c: str,
) -> torch.Tensor:
    """
    计算 A-B-C 三点夹角（B 是顶点）。
    用于膝/肘/踝等单自由度关节角。

    Args:
        q: (..., J, 3) 全局关节坐标
        joint_a, joint_b, joint_c: 关节名
    Returns:
        angle: (...,) 弧度，范围 [0, π]
    """
    a = q[..., get_joint_idx(joint_a), :]
    b = q[..., get_joint_idx(joint_b), :]
    c = q[..., get_joint_idx(joint_c), :]

    v_ba = a - b   # 从顶点指向 a
    v_bc = c - b   # 从顶点指向 c

    v_ba = F.normalize(v_ba, dim=-1, eps=EPS)
    v_bc = F.normalize(v_bc, dim=-1, eps=EPS)

    cos_theta = (v_ba * v_bc).sum(dim=-1)
    # clamp 防止 acos 在边界处梯度爆炸
    cos_theta = cos_theta.clamp(-1.0 + EPS, 1.0 - EPS)
    return torch.acos(cos_theta)


def signed_knee_angle(q: torch.Tensor, side: str = "left") -> torch.Tensor:
    """
    带符号的膝关节角（修复版）。

    定义：
        完全伸展 = π (180°)
        屈曲     < π
        超伸     > π

    修复：
        不再用 x-z 叉积判断方向（左右不对称），
        改用膝盖在髋-踝连线"前方还是后方"来判断，
        只看 z 分量（人物前进方向），对左右腿都成立。
    """
    EPS = 1e-7

    hip   = q[..., get_joint_idx(f"{side}_hip"),   :]
    knee  = q[..., get_joint_idx(f"{side}_knee"),  :]
    ankle = q[..., get_joint_idx(f"{side}_ankle"), :]

    v_thigh = hip   - knee
    v_shank = ankle - knee

    v_thigh_n = F.normalize(v_thigh, dim=-1, eps=EPS)
    v_shank_n = F.normalize(v_shank, dim=-1, eps=EPS)

    cos_a      = (v_thigh_n * v_shank_n).sum(dim=-1).clamp(-1+EPS, 1-EPS)
    base_angle = torch.acos(cos_a)   # [0, π]

    # 判断超伸：
    # 正常屈曲时膝盖在髋-踝连线前方（z 大）
    # 超伸时膝盖在髋-踝连线后方（z 小）
    # 直接比较 knee.z 和 髋-踝连线在同高度处的 z 插值
    mid_z = (hip[..., 2] + ankle[..., 2]) / 2
    knee_offset_z = knee[..., 2] - mid_z

    # knee_offset_z > 0 → 膝盖在前 → 正常屈曲，保持 base_angle
    # knee_offset_z < 0 → 膝盖在后 → 超伸，转换为 2π - base_angle
    sign       = torch.sigmoid(-50.0 * knee_offset_z)
    knee_angle = base_angle + sign * 2 * (math.pi - base_angle)

    return knee_angle


def pelvis_tilt_angle(q: torch.Tensor) -> torch.Tensor:
    """
    骨盆前倾角的鲁棒版本。
 
    临床定义：矢状面内骨盆与水平面的夹角。
    实现思路：
      1. 用左右髋的中点 → spine1 定义 "骨盆 → 脊柱" 向量
      2. 把这个向量投影到矢状面（去掉左右分量）
      3. 用 atan2(前后分量, 上下分量) 得到有符号的前倾角
         - 正值：骨盆前倾
         - 负值：骨盆后倾
         - 直立站姿 ≈ 0°
 
    Args:
        q: (..., J, 3) 全局关节坐标
    Returns:
        tilt: (...,) 弧度，正前倾 / 负后倾
    """
    pelvis    = q[..., get_joint_idx("pelvis"),    :]
    left_hip  = q[..., get_joint_idx("left_hip"),  :]
    right_hip = q[..., get_joint_idx("right_hip"), :]
    spine1    = q[..., get_joint_idx("spine1"),    :]
 
    # 骨盆中心
    hip_center = (left_hip + right_hip) / 2.0
 
    # 骨盆 → 脊柱方向
    pelvis_to_spine = spine1 - hip_center                       # (..., 3)
 
    # 左右轴（人物自身的左右朝向）
    lr_axis = right_hip - left_hip
    lr_axis = F.normalize(lr_axis, dim=-1, eps=EPS)
 
    # 把 pelvis_to_spine 投影到矢状面：去掉 lr 分量
    lr_component = (pelvis_to_spine * lr_axis).sum(
        dim=-1, keepdim=True
    ) * lr_axis
    sagittal_vec = pelvis_to_spine - lr_component
 
    # 矢状面内的有符号夹角
    # 注意：HumanML3D 坐标系 y=上下，z=前后
    forward_proj = sagittal_vec[..., 2]
    upward_proj  = sagittal_vec[..., 1]
 
    # atan2(前后, 上下)：直立时 forward≈0 → tilt≈0
    # 前倾时 forward>0 → tilt 为正
    # 后倾时 forward<0 → tilt 为负
    tilt = torch.atan2(forward_proj, upward_proj.clamp(min=EPS))
 
    return -tilt

def foot_floor_distance(q: torch.Tensor, side: str = "left") -> torch.Tensor:
    """
    足部到估计地面的距离（米）。
 
    用于约束足部不要在支撑相离地太远。
    地面高度用整段动作里所有足部关节 y 坐标的低分位数估计，
    这样不依赖外部地面假设，对 root 漂移鲁棒。
 
    Args:
        q:    (..., N, J, 3) 全局关节坐标，N 是时间维
        side: "left" 或 "right"
    Returns:
        dist: (..., N) 该侧足部每帧到地面的距离（米）
    """
    foot_idx       = get_joint_idx(f"{side}_foot")
    left_foot_idx  = get_joint_idx("left_foot")
    right_foot_idx = get_joint_idx("right_foot")
 
    foot_y = q[..., foot_idx, 1]                                # (..., N)
 
    # 用左右足所有时刻 y 坐标的最低 5% 作为地面
    both_feet_y = torch.cat([
        q[..., left_foot_idx,  1],
        q[..., right_foot_idx, 1],
    ], dim=-1)                                                  # (..., 2N)
 
    # quantile 沿时间维（最后一维）
    floor = torch.quantile(both_feet_y, 0.05, dim=-1, keepdim=True)
 
    return (foot_y - floor).clamp(min=0.0)

def spine_kyphosis_angle(q: torch.Tensor, EPS=1e-8) -> torch.Tensor:
    spine1 = q[..., get_joint_idx("spine1"), :]
    spine3 = q[..., get_joint_idx("spine3"), :]
    neck   = q[..., get_joint_idx("neck"),   :]
 
    # 躯干基准线方向（spine1 → neck）
    trunk_vec = neck - spine1
    trunk_len = trunk_vec.norm(dim=-1, keepdim=True).clamp(min=EPS)
    trunk_dir = trunk_vec / trunk_len
 
    # spine3 相对 spine1 的向量
    s3_vec = spine3 - spine1
 
    # 沿躯干方向的投影（"沿轴"分量）
    proj_scalar = (s3_vec * trunk_dir).sum(dim=-1)          # (...,)
    proj_vec    = proj_scalar.unsqueeze(-1) * trunk_dir
 
    # 垂直于躯干的"横向"分量（后凸的物理量）
    perp_vec  = s3_vec - proj_vec                            # (..., 3)
    perp_dist = perp_vec.norm(dim=-1)                        # (...,)
 
    # 归一化为角度（用半段躯干长度做归一化）
    half_len = trunk_len.squeeze(-1) * 0.5
    kyphosis = torch.atan2(perp_dist, half_len.clamp(min=EPS))
 
    return kyphosis   # 弧度，始终 ≥ 0

def spine_posterior_bulge(q: torch.Tensor) -> torch.Tensor:
    """
    spine3 偏离 spine1-neck 基准线的垂直距离（米）。

    直立时 spine3 几乎在基准线上，bulge ≈ 0.01-0.03m（正常轻微弯曲）
    驼背时 spine3 向后凸出，bulge 增大到 0.06-0.12m

    始终 ≥ 0，无符号问题，direction="greater_than" 直接可用。
    """
    EPS = 1e-7

    spine1 = q[..., get_joint_idx("spine1"), :]
    spine3 = q[..., get_joint_idx("spine3"), :]
    neck   = q[..., get_joint_idx("neck"),   :]

    trunk_vec = neck - spine1
    trunk_len = trunk_vec.norm(dim=-1, keepdim=True).clamp(min=EPS)
    trunk_dir = trunk_vec / trunk_len

    s3_vec = spine3 - spine1

    # spine3 在 trunk 方向的投影
    proj  = (s3_vec * trunk_dir).sum(dim=-1, keepdim=True) * trunk_dir

    # 垂直分量的长度（始终 ≥ 0，不管向哪个方向偏）
    perp  = s3_vec - proj
    bulge = perp.norm(dim=-1)

    return bulge

def head_forward_offset(q: torch.Tensor) -> torch.Tensor:
    """
    头前伸偏移量（米），用于检测头前伸体态。

    取头部相对于颈椎在前后方向（z 轴）的偏移。
    """
    head = q[..., get_joint_idx("head"), :]
    neck = q[..., get_joint_idx("neck"), :]

    # 假设 z 轴是前方
    return (head[..., 2] - neck[..., 2])