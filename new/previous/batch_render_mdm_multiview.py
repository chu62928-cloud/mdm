import os
import sys
import glob
import argparse
import subprocess
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(
        description="Batch render multi-view videos for all results.npy under a base directory."
    )
    parser.add_argument(
        "--base_dir",
        type=str,
        default="/root/autodl-tmp/motion-diffusion-model/save/batch_single_outputs",
        help="Base directory to search results.npy recursively"
    )
    parser.add_argument(
        "--render_script",
        type=str,
        default="/root/autodl-tmp/motion-diffusion-model/new/render_mdm_multiview.py",
        help="Path to render_mdm_multiview.py"
    )
    parser.add_argument(
        "--out_subdir_name",
        type=str,
        default="multiview_render",
        help="Output subdirectory name created under each case folder"
    )
    parser.add_argument(
        "--sample_idx",
        type=int,
        default=0,
        help="Sample index in each results.npy"
    )
    parser.add_argument(
        "--up_axis",
        type=str,
        default="y",
        choices=["x", "y", "z"],
        help="Up axis"
    )
    parser.add_argument(
        "--fps",
        type=float,
        default=20.0,
        help="FPS for output videos"
    )
    parser.add_argument(
        "--width",
        type=int,
        default=900,
        help="Single-view width"
    )
    parser.add_argument(
        "--height",
        type=int,
        default=700,
        help="Single-view height"
    )
    parser.add_argument(
        "--trail",
        type=int,
        default=20,
        help="Root trail length"
    )
    parser.add_argument(
        "--draw_ground",
        action="store_true",
        help="Whether to draw ground plane"
    )
    parser.add_argument(
        "--skip_existing",
        action="store_true",
        help="Skip if multiview_2x2.mp4 already exists"
    )
    parser.add_argument(
        "--recursive",
        action="store_true",
        help="Recursively search results.npy"
    )
    parser.add_argument(
        "--python_exec",
        type=str,
        default=sys.executable,
        help="Python executable"
    )
    return parser.parse_args()


def discover_results(base_dir, recursive=True):
    base = Path(base_dir)
    if recursive:
        files = list(base.rglob("results.npy"))
    else:
        files = list(base.glob("*/results.npy"))
    return sorted([str(f.resolve()) for f in files])


def main():
    args = parse_args()

    if not os.path.exists(args.base_dir):
        print(f"[ERROR] base_dir not found: {args.base_dir}")
        return

    if not os.path.exists(args.render_script):
        print(f"[ERROR] render_script not found: {args.render_script}")
        return

    results_files = discover_results(args.base_dir, recursive=args.recursive)

    if len(results_files) == 0:
        print("[INFO] No results.npy found.")
        return

    print("=" * 90)
    print(f"Found {len(results_files)} results.npy files under:")
    print(args.base_dir)
    print("=" * 90)

    success = 0
    failed = 0
    skipped = 0

    for i, results_path in enumerate(results_files, start=1):
        case_dir = os.path.dirname(results_path)
        outdir = os.path.join(case_dir, args.out_subdir_name)
        target_check = os.path.join(outdir, "multiview_2x2.mp4")

        print("\n" + "-" * 90)
        print(f"[{i}/{len(results_files)}]")
        print(f"results.npy : {results_path}")
        print(f"output dir  : {outdir}")

        if args.skip_existing and os.path.exists(target_check):
            print("[SKIP] multiview_2x2.mp4 already exists.")
            skipped += 1
            continue

        os.makedirs(outdir, exist_ok=True)

        cmd = [
            args.python_exec,
            args.render_script,
            "--input", results_path,
            "--sample_idx", str(args.sample_idx),
            "--outdir", outdir,
            "--up_axis", args.up_axis,
            "--fps", str(args.fps),
            "--width", str(args.width),
            "--height", str(args.height),
            "--trail", str(args.trail),
        ]

        if args.draw_ground:
            cmd.append("--draw_ground")

        print("[RUN]", " ".join(cmd))

        try:
            result = subprocess.run(cmd, check=False, capture_output=True, text=True)

            log_path = os.path.join(outdir, "render_log.txt")
            with open(log_path, "w", encoding="utf-8") as f:
                f.write("COMMAND:\n")
                f.write(" ".join(cmd) + "\n\n")
                f.write("STDOUT:\n")
                f.write(result.stdout + "\n\n")
                f.write("STDERR:\n")
                f.write(result.stderr + "\n")

            if result.returncode == 0:
                print("[OK] Render finished.")
                success += 1
            else:
                print("[FAIL] Render failed.")
                print(result.stderr[:1000])
                failed += 1

        except Exception as e:
            print(f"[EXCEPTION] {e}")
            failed += 1

    print("\n" + "=" * 90)
    print("Batch rendering finished.")
    print(f"Success : {success}")
    print(f"Skipped : {skipped}")
    print(f"Failed  : {failed}")
    print("=" * 90)


if __name__ == "__main__":
    main()