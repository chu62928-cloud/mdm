#!/usr/bin/env python3
import argparse
import os
import re
import sys
from pathlib import Path

from visualize import vis_utils


SINGLE_RE = re.compile(r"^sample(\d+)_rep(\d+)\.mp4$")
GROUPED_RE = re.compile(r"^samples_(\d+)_to_(\d+)\.mp4$")


def find_results_npy(search_dir: Path) -> Path:
    npy_path = search_dir / "results.npy"
    if npy_path.exists():
        return npy_path
    raise FileNotFoundError(f"results.npy not found in {search_dir}")


def export_one(
    npy_path: Path,
    out_dir: Path,
    sample_idx: int,
    rep_idx: int,
    use_cuda: bool,
    device,
    overwrite: bool,
):
    prefix = out_dir / f"sample{sample_idx}_rep{rep_idx}"
    obj_dir = Path(str(prefix) + "_obj")
    smpl_npy = Path(str(prefix) + "_smpl_params.npy")

    if (obj_dir.exists() or smpl_npy.exists()) and not overwrite:
        print(f"[skip] sample{sample_idx}_rep{rep_idx} already exists")
        return

    obj_dir.mkdir(parents=True, exist_ok=True)

    print(f"[run ] sample={sample_idx}, rep={rep_idx}, cuda={use_cuda}, device={device}")
    npy2obj = vis_utils.npy2obj(
        str(npy_path),
        sample_idx,
        rep_idx,
        device=device,
        cuda=use_cuda,
    )

    print(f"[save] obj -> {obj_dir}")
    for frame_i in range(npy2obj.real_num_frames):
        npy2obj.save_obj(str(obj_dir / f"frame{frame_i:03d}.obj"), frame_i)

    print(f"[save] smpl params -> {smpl_npy}")
    npy2obj.save_npy(str(smpl_npy))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input_dir",
        required=True,
        help="Directory containing mp4 files and results.npy",
    )
    parser.add_argument(
        "--cuda",
        action="store_true",
        help="Use CUDA for vis_utils.npy2obj. Default: CPU",
    )
    parser.add_argument(
        "--device",
        default="cpu",
        help="Device to use. Example: cpu or 0",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing outputs",
    )
    args = parser.parse_args()

    input_dir = Path(args.input_dir).resolve()
    if not input_dir.exists():
        raise FileNotFoundError(f"Input dir does not exist: {input_dir}")

    npy_path = find_results_npy(input_dir)
    print(f"[info] using results.npy: {npy_path}")

    mp4_files = sorted([p for p in input_dir.iterdir() if p.suffix.lower() == ".mp4"])
    if not mp4_files:
        print("[warn] no mp4 files found")
        return

    single_targets = []
    skipped_grouped = []
    skipped_other = []

    for mp4 in mp4_files:
        m = SINGLE_RE.match(mp4.name)
        if m:
            sample_idx = int(m.group(1))
            rep_idx = int(m.group(2))
            single_targets.append((mp4, sample_idx, rep_idx))
            continue

        g = GROUPED_RE.match(mp4.name)
        if g:
            skipped_grouped.append(mp4.name)
            continue

        skipped_other.append(mp4.name)

    if skipped_grouped:
        print("\n[warn] grouped mp4 files are not supported by render_mesh logic and were skipped:")
        for name in skipped_grouped:
            print("   ", name)

    if skipped_other:
        print("\n[warn] unrecognized mp4 filenames were skipped:")
        for name in skipped_other:
            print("   ", name)

    if not single_targets:
        print("\n[warn] no single-sample mp4 files found (expected names like sample0_rep0.mp4)")
        print("[hint] If your folder only has files like samples_00_to_02.mp4,")
        print("       create proper single-sample symlinks/copies first, or render directly from results.npy.")
        return

    use_cuda = bool(args.cuda)
    device = 0 if (use_cuda and str(args.device) == "cpu") else args.device

    for mp4, sample_idx, rep_idx in single_targets:
        print(f"\n[mp4 ] {mp4.name}")
        try:
            export_one(
                npy_path=npy_path,
                out_dir=input_dir,
                sample_idx=sample_idx,
                rep_idx=rep_idx,
                use_cuda=use_cuda,
                device=device,
                overwrite=args.overwrite,
            )
        except Exception as e:
            print(f"[fail] {mp4.name}: {e}")

    print("\n[done]")


if __name__ == "__main__":
    main()