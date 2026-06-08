import numpy as np, math, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Arc
from matplotlib.lines import Line2D
import torch, sys
sys.path.insert(0, ".")
from posture_guidance.angle_ops import pelvis_tilt_angle

data = np.load("output/posture_pipeline/both_w30/comparison.npy", allow_pickle=True).item()
xyz_b = data["motion_xyz"][0]
xyz_g = data["motion_xyz_guided"][0]

q_b = torch.from_numpy(xyz_b).permute(2, 0, 1).unsqueeze(0).float()
q_g = torch.from_numpy(xyz_g).permute(2, 0, 1).unsqueeze(0).float()
apt_b_all = (pelvis_tilt_angle(q_b) * 180.0 / math.pi).numpy()[0]
apt_g_all = (pelvis_tilt_angle(q_g) * 180.0 / math.pi).numpy()[0]

best_frame = 51
print("Frame {}: baseline={:.1f} deg, guided={:.1f} deg".format(best_frame, apt_b_all[best_frame], apt_g_all[best_frame]))

P, LH, RH = 0, 1, 2
S1, S2, S3 = 3, 6, 9
N, HD = 12, 15

def sag_2d(pts):
    return np.column_stack([pts[:, 2], pts[:, 1]])

def draw_skel(ax, p2, bc, fc, lw=3.0):
    si = [P, S1, S2, S3, N, HD]
    sp = p2[si]
    ax.plot(sp[:,0], sp[:,1], color=bc, lw=lw, alpha=0.95, zorder=4, solid_capstyle='round')
    ax.add_patch(Circle(p2[HD], 0.06, fc=fc, ec=bc, lw=1.5, alpha=0.9, zorder=3))
    for (i,j) in [(LH,4),(4,7),(7,10),(RH,5),(5,8),(8,11),(N,13),(13,16),(16,18),(18,20),(N,14),(14,17),(17,19),(19,21)]:
        ax.plot([p2[i,0],p2[j,0]],[p2[i,1],p2[j,1]],color=bc,lw=lw*0.55,alpha=0.8,zorder=3,solid_capstyle='round')

def apt_vec(pts):
    hc = (pts[LH] + pts[RH]) / 2.0
    v = pts[S1] - hc
    lr = pts[RH] - pts[LH]
    lr = lr / (np.linalg.norm(lr) + 1e-8)
    sag = v - np.dot(v, lr) * lr
    return np.array([sag[2], sag[1]]), hc

fig, (ax_l, ax_r) = plt.subplots(1, 2, figsize=(14, 8))

for ax, xyz, title, bc, fc in [
    (ax_l, xyz_b, "Baseline", "#3D4F8F", "#A0B2DC"),
    (ax_r, xyz_g, "Guided  (both, W=30)", "#8E3F61", "#D4A0B6")]:

    p3 = xyz[:, :, best_frame]
    p2 = sag_2d(p3)
    sv, hc = apt_vec(p3)
    hc2 = sag_2d(hc.reshape(1,3))[0]

    draw_skel(ax, p2, bc, fc)

    # Vertical reference
    rl = 0.25
    ax.plot([hc2[0], hc2[0]], [hc2[1], hc2[1]+rl], '--', color='gray', lw=1.5, alpha=0.5, zorder=2)
    ax.text(hc2[0]-0.03, hc2[1]+rl/2, "gravity", fontsize=7, color='gray', rotation=90, va='center')

    # Arrow
    vn = sv / (np.linalg.norm(sv) + 1e-8)
    vp = vn * rl
    ax.annotate('', xy=(hc2[0]+vp[0], hc2[1]+vp[1]), xytext=(hc2[0], hc2[1]),
                arrowprops=dict(arrowstyle='->', color=bc, lw=4.5, alpha=0.95), zorder=6)

    # Angle: -atan2(z,y) = pelvis_tilt_angle
    ar = math.atan2(sv[0], sv[1])
    ad = -ar * 180.0 / math.pi

    # Arc
    vad = 90.0 - ad
    rr = 0.12
    arc = Arc((hc2[0], hc2[1]), 2*rr, 2*rr, angle=0,
              theta1=min(90, vad), theta2=max(90, vad), color=bc, lw=2.5, zorder=5)
    ax.add_patch(arc)

    # Label
    ma = (90 + vad) / 2.0 * math.pi / 180.0
    ax.text(hc2[0]+rr*1.7*math.cos(ma), hc2[1]+rr*1.7*math.sin(ma),
            "{} deg".format(int(round(ad))), fontsize=12, color=bc,
            fontweight='bold', ha='center', va='center',
            bbox=dict(boxstyle='round,pad=0.1', fc='white', alpha=0.85, ec='none'))

    ax.scatter([hc2[0]], [hc2[1]], s=60, c=bc, zorder=7, marker='o', ec='white', lw=1.5)

    m = 0.3
    ax.set_xlim(p2[:,0].min()-m, p2[:,0].max()+m)
    ax.set_ylim(p2[:,1].min()-0.15, p2[:,1].max()+m)
    ax.set_aspect('equal'); ax.axis('off')
    ax.set_title(title, fontsize=14, fontweight='bold', color=bc, pad=8)

    ax.text(0.03, 0.97, "APT = {:.1f} deg".format(ad),
            transform=ax.transAxes, fontsize=12, fontweight='bold', color=bc, va='top',
            bbox=dict(boxstyle='round', fc='white', alpha=0.85, ec=bc, lw=1.2))

    # Annotation arrows
    ax.annotate('pelvis', xy=(p2[P,0], p2[P,1]), fontsize=7, color=bc, ha='center',
                xytext=(p2[P,0]-0.1, p2[P,1]-0.1),
                arrowprops=dict(arrowstyle='->', color='gray', lw=0.8, alpha=0.6))

fig.suptitle("Anterior Pelvic Tilt (APT): Baseline vs Guided", fontsize=15, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig("output/posture_pipeline/fig1_apt_comparison.png", dpi=200, bbox_inches="tight", facecolor='white')
plt.close()
print("Saved fig1_apt_comparison.png")
