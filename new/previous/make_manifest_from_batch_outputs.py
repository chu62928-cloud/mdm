import os
import csv
import argparse
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(
        description="Make manifest.csv from batch_single_outputs-style folders"
    )
    parser.add_argument("--mp4_root", type=str, required=True,
                        help="Root directory containing per-prompt subfolders with mp4 files")
    parser.add_argument("--results_root", type=str, default=None,
                        help="Root directory containing per-prompt subfolders with results.npy; default = mp4_root")
    parser.add_argument("--out_csv", type=str, required=True,
                        help="Output manifest csv path")
    parser.add_argument("--sample_idx", type=int, default=0,
                        help="Default sample_idx to use in manifest")
    parser.add_argument("--mp4_pattern", type=str, default="*.mp4",
                        help="Pattern to search for mp4 inside each subfolder")
    return parser.parse_args()


def find_first_file(folder: Path, pattern: str):
    files = sorted(folder.glob(pattern))
    return files[0] if files else None


def main():
    args = parse_args()

    mp4_root = Path(args.mp4_root)
    results_root = Path(args.results_root) if args.results_root else mp4_root
    out_csv = Path(args.out_csv)

    if not mp4_root.exists():
        raise FileNotFoundError(f"mp4_root not found: {mp4_root}")
    if not results_root.exists():
        raise FileNotFoundError(f"results_root not found: {results_root}")

    rows = []
    subdirs = sorted([p for p in mp4_root.iterdir() if p.is_dir()])

    for subdir in subdirs:
        mp4_file = find_first_file(subdir, args.mp4_pattern)
        if mp4_file is None:
            print(f"[SKIP] no mp4 found in {subdir}")
            continue

        results_dir = results_root / subdir.name
        results_file = results_dir / "results.npy"
        if not results_file.exists():
            print(f"[SKIP] no results.npy found for {subdir.name}: {results_file}")
            continue

        rows.append({
            "mp4_path": str(mp4_file.resolve()),
            "results_path": str(results_file.resolve()),
            "sample_idx": args.sample_idx,
            "output_name": subdir.name
        })

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["mp4_path", "results_path", "sample_idx", "output_name"]
        )
        writer.writeheader()
        for row in rows:
            writer.writerow(row)

    print(f"[OK] wrote manifest: {out_csv}")
    print(f"[OK] total rows: {len(rows)}")


if __name__ == "__main__":
    main()