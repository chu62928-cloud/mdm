"""
new/retrieve_ood_prompts.py

从 HumanML3D 训练集检索与高膝角 motion 配对的 text prompt，
作为组合式扩散（路线 A）的 text_ood 候选。

原理（Retrieval-Augmented Compositional Diffusion）：
    MDM 的文本条件分布以训练集文本为支撑。找出训练集里膝角最高的 clip，
    读取对应的文本标注——这些描述在模型看来天然与"较高膝角"相关联，
    比手工猜测的 prompt 有更可靠的密度支撑。

用法：
    # 基本用法
    python -m new.retrieve_ood_prompts --data-dir /path/to/HumanML3D

    # 只看超伸 clip（>178°），输出文本候选到文件
    python -m new.retrieve_ood_prompts --data-dir /path/to/HumanML3D \\
        --min-angle 178 --top-k 50 --output new/ood_prompt_candidates.txt

输出：
    - 膝角最高的 top-k clip 详情（clip ID + 膝角 + 文本）
    - 高频文本排名（用于挑选 text_ood）
    - 路线 A 的 GUIDANCE_KWARGS_JSON 使用示例
"""

import argparse
import math
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from posture_guidance.angle_ops import signed_knee_angle


# ──────────────────────────────────────────────────────────
# 数据加载
# ──────────────────────────────────────────────────────────

def load_xyz(npy_path: Path):
    """加载 new_joints .npy，返回 (T, 22, 3) tensor 或 None。"""
    try:
        arr = np.load(npy_path)
        if arr.ndim == 3 and arr.shape[-1] == 3 and arr.shape[1] >= 22:
            return torch.from_numpy(arr[:, :22, :]).float()
        return None
    except Exception:
        return None


def load_texts(text_path: Path) -> list:
    """
    读取 HumanML3D 单个 clip 的文本标注。
    格式：每行 "text#start_time#end_time"（时间可为 0.0）。
    返回去重后的文本列表。
    """
    if not text_path.exists():
        return []
    seen = set()
    texts = []
    for line in text_path.read_text(encoding="utf-8", errors="ignore").strip().split("\n"):
        line = line.strip()
        if not line:
            continue
        text = line.split("#")[0].strip().lower()
        if text and text not in seen:
            seen.add(text)
            texts.append(text)
    return texts


def clip_knee_score(q: torch.Tensor) -> float:
    """
    返回 clip 的膝角分数（度）：左右膝的最大值。
    用 P95（而非全局 max）减少异常帧的影响。
    """
    with torch.no_grad():
        left  = signed_knee_angle(q, side="left").numpy()   # (T,) rad
        right = signed_knee_angle(q, side="right").numpy()  # (T,) rad
    # 取 P95（而非 max），对噪声帧更鲁棒
    score_deg = max(
        float(np.percentile(left,  95)) * 180.0 / math.pi,
        float(np.percentile(right, 95)) * 180.0 / math.pi,
    )
    return score_deg


# ──────────────────────────────────────────────────────────
# 主函数
# ──────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="HumanML3D 高膝角 clip 文本检索（路线 A text_ood 候选）",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--data-dir", required=True,
        help="HumanML3D 根目录（含 new_joints/ 和 texts/ 子目录）",
    )
    parser.add_argument(
        "--split", default="train",
        choices=["train", "val", "all"],
        help="使用哪个 split（默认 train，避免测试集泄漏）",
    )
    parser.add_argument(
        "--top-k", type=int, default=20,
        help="输出膝角最高的前 N 个 clip（默认 20）",
    )
    parser.add_argument(
        "--min-angle", type=float, default=175.0,
        help="只纳入膝角 P95 超过此值的 clip 做文本统计（度，默认 175）",
    )
    parser.add_argument(
        "--output", default=None,
        help="把 top-k 文本候选写入此文件（每行一条，按频率降序）",
    )
    parser.add_argument(
        "--max-clips", type=int, default=0,
        help="只处理前 N 个 clip（0 = 全部，调试用）",
    )
    args = parser.parse_args()

    data_dir   = Path(args.data_dir)
    joints_dir = data_dir / "new_joints"
    texts_dir  = data_dir / "texts"

    # 路径检查
    missing = []
    if not joints_dir.exists():
        missing.append(str(joints_dir))
    if not texts_dir.exists():
        missing.append(str(texts_dir))
    if missing:
        print(f"\n❌ 找不到以下目录：")
        for m in missing:
            print(f"   {m}")
        print("   请确认 --data-dir 指向 HumanML3D 根目录并已完整解压")
        sys.exit(1)

    # 确定 clip 列表
    if args.split == "all":
        clip_ids = [p.stem for p in sorted(joints_dir.glob("*.npy"))]
    else:
        split_file = data_dir / f"{args.split}.txt"
        if not split_file.exists():
            print(f"⚠  找不到 {split_file}，改为扫描全部")
            clip_ids = [p.stem for p in sorted(joints_dir.glob("*.npy"))]
        else:
            clip_ids = [l.strip() for l in split_file.read_text().strip().split("\n") if l.strip()]

    if args.max_clips > 0:
        clip_ids = clip_ids[:args.max_clips]

    print(f"\n扫描 {len(clip_ids)} 个 clip（split={args.split}）...\n")

    # ── 第一遍：计算膝角分数 ──────────────────────────────
    scores = []   # [(score_deg, clip_id)]
    skipped = 0
    report_every = max(1, len(clip_ids) // 10)

    for i, clip_id in enumerate(clip_ids):
        if i > 0 and i % report_every == 0:
            pct = 100 * i // len(clip_ids)
            top_score = scores[0][0] if scores else 0.0
            print(f"  [{i:5d}/{len(clip_ids)}]  {pct}%  当前最高膝角={top_score:.1f}°")

        npy_path = joints_dir / f"{clip_id}.npy"
        q = load_xyz(npy_path)
        if q is None:
            skipped += 1
            continue

        score = clip_knee_score(q)
        scores.append((score, clip_id))

    scores.sort(reverse=True)
    print(f"\n完成。有效 {len(scores)} / {len(clip_ids)} 个，跳过 {skipped}。\n")

    # ── 统计 min_angle 以上的 clip ──────────────────────────
    high_clips = [(s, cid) for s, cid in scores if s >= args.min_angle]
    print(f"膝角 P95 ≥ {args.min_angle:.0f}° 的 clip：{len(high_clips)} 个  "
          f"（占 {100.0 * len(high_clips) / max(len(scores), 1):.2f}%）\n")

    # ── 第二遍：读取 top-k clip 的文本 ──────────────────────
    top_clips = scores[:args.top_k]
    all_texts = []          # 所有文本（可重复，用于词频统计）
    clip_details = []       # [(score, clip_id, [texts])]

    for score, clip_id in top_clips:
        text_path = texts_dir / f"{clip_id}.txt"
        texts = load_texts(text_path)
        all_texts.extend(texts)
        clip_details.append((score, clip_id, texts))

    # ── 输出 top-k clip 详情 ─────────────────────────────────
    print("=" * 80)
    print(f"  膝角最高的 top-{args.top_k} 个 clip（膝角 = P95 max(左,右)）")
    print("=" * 80)
    for rank, (score, clip_id, texts) in enumerate(clip_details, 1):
        print(f"\n  #{rank:2d}  [{clip_id}]  knee_p95 = {score:.1f}°"
              + ("  ← 超伸" if score > 180 else ""))
        for t in texts[:4]:
            print(f"        ┆ '{t}'")
        if len(texts) > 4:
            print(f"        ┆ ... 共 {len(texts)} 条")

    # ── 文本频率统计 ─────────────────────────────────────────
    counter = Counter(all_texts)
    unique_texts = [t for t, _ in counter.most_common()]

    print(f"\n{'=' * 80}")
    print(f"  文本候选（top-{args.top_k} clip 共 {len(all_texts)} 条，{len(unique_texts)} 种）")
    print(f"  按频率降序排列 ─ 推荐作为路线 A 的 text_ood")
    print(f"{'=' * 80}")
    for rank, (text, count) in enumerate(counter.most_common(25), 1):
        bar = "█" * min(count, 20)
        print(f"  {rank:2d}. ({count:3d}x) {bar}  '{text}'")

    # ── 可选写文件 ──────────────────────────────────────────
    if args.output:
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as f:
            f.write(f"# HumanML3D 高膝角 clip 文本候选（top-{args.top_k} clips）\n")
            f.write(f"# min_angle={args.min_angle}°, split={args.split}\n")
            f.write(f"# 按频率降序，共 {len(unique_texts)} 条\n\n")
            for text, count in counter.most_common():
                f.write(f"{text}\n")
        print(f"\n完整列表已写入 {out_path}（{len(unique_texts)} 条）")

    # ── 使用建议 ────────────────────────────────────────────
    top3 = [t for t, _ in counter.most_common(3)]
    print(f"\n{'=' * 80}")
    print("  路线 A 使用建议（GUIDANCE_KWARGS_JSON）")
    print(f"{'=' * 80}")
    print('  {')
    print('    "text_conditions": [')
    print('      "a person is walking forward",   <- w1: 步态结构（固定）')
    if top3:
        for t in top3:
            print(f'      "{t}",   <- w2: text_ood 候选（逐一测试）')
    print('    ],')
    print('    "compositional_weights": [1.0, 0.5],   <- 先从 w2=0.5 开始，上调至 2.0')
    print('    ...(其余 V6 参数保持不变)')
    print('  }')
    print()
    print("  验证 text_ood 有效性（单独生成，看膝角分布是否偏移）：")
    if top3:
        example = top3[0].replace('"', '\\"')
        print(f'  GUIDANCE_VARIANT=none python -m sample.generate \\')
        print(f'    --text_prompt "{example}" \\')
        print(f'    --num_samples 10 --output_dir ./output/test_ood_prompt/')
        print(f'  python -m new.check_ood_risk --posture 膝超伸_左 ./output/test_ood_prompt/')
    print("=" * 80)


if __name__ == "__main__":
    main()
