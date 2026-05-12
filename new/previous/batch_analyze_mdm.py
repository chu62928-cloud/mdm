import os
import re
import csv
import sys
import json
import glob
import argparse
import subprocess
from pathlib import Path

import numpy as np


def parse_args():
    parser = argparse.ArgumentParser(
        description="Batch-run analyze_mdm_case_v2.py over official MDM results.npy files"
    )
    parser.add_argument(
        "--inputs",
        type=str,
        nargs="+",
        required=True,
        help="One or more results.npy files, directories, or glob patterns"
    )
    parser.add_argument(
        "--analyze_script",
        type=str,
        required=True,
        help="Path to analyze_mdm_case_v2.py"
    )
    parser.add_argument(
        "--outdir",
        type=str,
        required=True,
        help="Base output directory"
    )
    parser.add_argument(
        "--pattern",
        type=str,
        default="results.npy",
        help="Filename pattern to search for in directories"
    )
    parser.add_argument(
        "--recursive",
        action="store_true",
        help="Recursively search subdirectories"
    )
    parser.add_argument(
        "--python_exec",
        type=str,
        default=sys.executable,
        help="Python executable used to run the analyze script"
    )

    # passthrough args to analyze_mdm_case_v2.py
    parser.add_argument("--fps", type=float, default=20.0)
    parser.add_argument("--up_axis", type=str, default="y", choices=["x", "y", "z"])
    parser.add_argument("--mode", type=str, default="auto", choices=["auto", "walk", "bend"])
    parser.add_argument("--height_thresh", type=float, default=0.05)
    parser.add_argument("--speed_thresh", type=float, default=0.08)
    parser.add_argument("--foot_width", type=float, default=0.10)
    parser.add_argument("--ground_quantile", type=float, default=0.02)
    parser.add_argument("--root_low_speed_thresh", type=float, default=0.15)
    parser.add_argument("--foot_stable_speed_thresh", type=float, default=0.10)
    parser.add_argument("--smooth_win", type=int, default=5)

    parser.add_argument(
        "--max_samples_per_file",
        type=int,
        default=None,
        help="Optional cap for number of samples analyzed in each results.npy"
    )
    parser.add_argument(
        "--skip_existing",
        action="store_true",
        help="Skip cases whose metrics.csv already exists"
    )
    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="Only print planned actions without running analysis"
    )
    return parser.parse_args()


def discover_result_files(inputs, pattern="results.npy", recursive=False):
    found = []

    for item in inputs:
        p = Path(item)

        if p.is_file():
            if p.name == pattern or p.suffix == ".npy":
                found.append(str(p.resolve()))
            continue

        if p.is_dir():
            matches = list(p.rglob(pattern)) if recursive else list(p.glob(pattern))
            found.extend(str(m.resolve()) for m in matches)
            continue

        # treat as glob
        matches = glob.glob(item, recursive=recursive)
        for m in matches:
            mp = Path(m)
            if mp.is_file():
                found.append(str(mp.resolve()))

    return sorted(set(found))


def load_results_metadata(results_path):
    obj = np.load(results_path, allow_pickle=True)
    if isinstance(obj, np.ndarray) and obj.dtype == object and obj.shape == ():
        data = obj.item()
    elif isinstance(obj, dict):
        data = obj
    else:
        raise ValueError(f"Unsupported results.npy format: {results_path}")

    if "motion" not in data:
        raise KeyError(f"'motion' not found in {results_path}")

    motion = np.asarray(data["motion"])
    if motion.ndim != 4:
        raise ValueError(f"Expected motion shape (N, J, 3, T), got {motion.shape} in {results_path}")

    num_cases = motion.shape[0]
    texts = list(data.get("text", [""] * num_cases))
    lengths = list(np.asarray(data.get("lengths", [None] * num_cases)).tolist())

    return {
        "num_cases": num_cases,
        "motion_shape": tuple(motion.shape),
        "texts": texts,
        "lengths": lengths,
    }


def sanitize_filename(text: str, max_len: int = 80) -> str:
    text = (text or "").strip().replace("\n", " ")
    text = re.sub(r"\s+", "_", text)
    text = re.sub(r"[^A-Za-z0-9_\-]+", "", text)
    if not text:
        text = "sample"
    return text[:max_len]


def build_case_outdir(base_outdir, results_path, sample_idx, prompt_text):
    parent_name = Path(results_path).parent.name
    file_stem = Path(results_path).stem
    prompt_stub = sanitize_filename(prompt_text, max_len=60)
    case_dir_name = f"{parent_name}__{file_stem}__sample_{sample_idx:03d}__{prompt_stub}"
    return os.path.join(base_outdir, case_dir_name)


def read_metrics_csv(metrics_csv):
    metrics = {}
    if not os.path.exists(metrics_csv):
        return metrics

    with open(metrics_csv, "r", encoding="utf-8") as f:
        reader = csv.reader(f)
        next(reader, None)  # header
        for row in reader:
            if len(row) != 2:
                continue
            key, value = row
            try:
                if value.lower() == "nan":
                    metrics[key] = np.nan
                else:
                    metrics[key] = float(value)
            except Exception:
                metrics[key] = value
    return metrics


def run_single_case(args, results_path, sample_idx, case_outdir):
    os.makedirs(case_outdir, exist_ok=True)

    cmd = [
        args.python_exec,
        args.analyze_script,
        "--input", results_path,
        "--sample_idx", str(sample_idx),
        "--outdir", case_outdir,
        "--fps", str(args.fps),
        "--up_axis", args.up_axis,
        "--mode", args.mode,
        "--height_thresh", str(args.height_thresh),
        "--speed_thresh", str(args.speed_thresh),
        "--foot_width", str(args.foot_width),
        "--ground_quantile", str(args.ground_quantile),
        "--root_low_speed_thresh", str(args.root_low_speed_thresh),
        "--foot_stable_speed_thresh", str(args.foot_stable_speed_thresh),
        "--smooth_win", str(args.smooth_win),
        "--title", f"MDM Sample {sample_idx:03d}",
    ]

    result = subprocess.run(cmd, capture_output=True, text=True)
    return result, cmd


def safe_mean(vals):
    vals = [v for v in vals if v is not None and not (isinstance(v, float) and np.isnan(v))]
    if len(vals) == 0:
        return np.nan
    return float(np.mean(vals))


def make_grouped_summary(summary_rows):
    """
    生成更适合看总体趋势的 grouped summary:
    - 按 analysis_mode 分组
    - 统计常见指标均值
    """
    groups = {}
    for row in summary_rows:
        if row.get("status") != "ok":
            continue
        mode = row.get("analysis_mode", "unknown")
        groups.setdefault(mode, []).append(row)

    metric_candidates = [
        "left_skating_mean",
        "right_skating_mean",
        "left_skating_ratio",
        "right_skating_ratio",
        "root_jerk",
        "mos_mean",
        "mos_min",
        "mos_negative_ratio",
        "mos_mean_stance",
        "mos_negative_ratio_stance",
        "com_margin_mean_all",
        "com_margin_min_all",
        "com_margin_negative_ratio_all",
        "com_margin_mean_low_speed",
        "com_margin_min_low_speed",
        "com_margin_negative_ratio_low_speed",
        "num_low_speed_frames",
        "num_selected_quasistatic_frames",
    ]

    grouped_rows = []
    for mode, rows in groups.items():
        out = {
            "analysis_mode": mode,
            "num_cases": len(rows),
        }
        for m in metric_candidates:
            vals = []
            for r in rows:
                if m in r:
                    vals.append(r[m])
            out[m + "_mean"] = safe_mean(vals)
        grouped_rows.append(out)

    return grouped_rows


def main():
    args = parse_args()
    os.makedirs(args.outdir, exist_ok=True)

    result_files = discover_result_files(
        inputs=args.inputs,
        pattern=args.pattern,
        recursive=args.recursive
    )

    if not result_files:
        print("No results.npy files found.")
        return

    print(f"Discovered {len(result_files)} results.npy file(s):")
    for p in result_files:
        print("  ", p)

    summary_rows = []
    failures = []

    for results_path in result_files:
        print("\n" + "=" * 88)
        print(f"Processing: {results_path}")

        try:
            meta = load_results_metadata(results_path)
        except Exception as e:
            print(f"[ERROR] metadata load failed: {e}")
            failures.append({
                "results_path": results_path,
                "sample_idx": None,
                "error": str(e),
            })
            continue

        num_cases = meta["num_cases"]
        texts = meta["texts"]
        lengths = meta["lengths"]

        if args.max_samples_per_file is not None:
            num_cases = min(num_cases, args.max_samples_per_file)

        print(f"motion_shape = {meta['motion_shape']}")
        print(f"num_cases    = {num_cases}")

        for sample_idx in range(num_cases):
            prompt_text = texts[sample_idx] if sample_idx < len(texts) else ""
            motion_len = lengths[sample_idx] if sample_idx < len(lengths) else None

            case_outdir = build_case_outdir(
                base_outdir=args.outdir,
                results_path=results_path,
                sample_idx=sample_idx,
                prompt_text=prompt_text
            )

            metrics_csv = os.path.join(case_outdir, "metrics.csv")
            stdout_path = os.path.join(case_outdir, "analyze_stdout.txt")
            stderr_path = os.path.join(case_outdir, "analyze_stderr.txt")

            if args.skip_existing and os.path.exists(metrics_csv):
                print(f"[SKIP] sample {sample_idx:03d} already analyzed")
                metrics = read_metrics_csv(metrics_csv)
                summary_rows.append({
                    "results_path": results_path,
                    "sample_idx": sample_idx,
                    "prompt": prompt_text,
                    "length": motion_len,
                    "case_outdir": case_outdir,
                    "status": "skipped_existing",
                    **metrics,
                })
                continue

            print("-" * 88)
            print(f"sample_idx = {sample_idx}")
            print(f"prompt     = {prompt_text}")
            print(f"length     = {motion_len}")
            print(f"outdir     = {case_outdir}")

            if args.dry_run:
                summary_rows.append({
                    "results_path": results_path,
                    "sample_idx": sample_idx,
                    "prompt": prompt_text,
                    "length": motion_len,
                    "case_outdir": case_outdir,
                    "status": "dry_run",
                })
                continue

            run_result, cmd = run_single_case(args, results_path, sample_idx, case_outdir)

            with open(stdout_path, "w", encoding="utf-8") as f:
                f.write(run_result.stdout)
            with open(stderr_path, "w", encoding="utf-8") as f:
                f.write(run_result.stderr)

            if run_result.returncode != 0:
                print(f"[FAIL] sample {sample_idx:03d} returncode={run_result.returncode}")
                print(run_result.stderr[:600])
                failures.append({
                    "results_path": results_path,
                    "sample_idx": sample_idx,
                    "error": run_result.stderr.strip()[:1500],
                    "cmd": " ".join(cmd),
                })
                summary_rows.append({
                    "results_path": results_path,
                    "sample_idx": sample_idx,
                    "prompt": prompt_text,
                    "length": motion_len,
                    "case_outdir": case_outdir,
                    "status": "failed",
                })
                continue

            metrics = read_metrics_csv(metrics_csv)
            summary_rows.append({
                "results_path": results_path,
                "sample_idx": sample_idx,
                "prompt": prompt_text,
                "length": motion_len,
                "case_outdir": case_outdir,
                "status": "ok",
                **metrics,
            })
            print(f"[OK] sample {sample_idx:03d} done")

    # save summary_metrics.csv
    summary_csv = os.path.join(args.outdir, "summary_metrics.csv")
    fieldnames = sorted(set().union(*[row.keys() for row in summary_rows])) if summary_rows else []

    with open(summary_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if fieldnames:
            writer.writeheader()
            for row in summary_rows:
                writer.writerow(row)

    # save grouped_summary.csv
    grouped_rows = make_grouped_summary(summary_rows)
    grouped_csv = os.path.join(args.outdir, "grouped_summary.csv")
    grouped_fields = sorted(set().union(*[row.keys() for row in grouped_rows])) if grouped_rows else []

    with open(grouped_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=grouped_fields)
        if grouped_fields:
            writer.writeheader()
            for row in grouped_rows:
                writer.writerow(row)

    # save run_summary.json
    summary_json = os.path.join(args.outdir, "run_summary.json")
    with open(summary_json, "w", encoding="utf-8") as f:
        json.dump({
            "num_result_files": len(result_files),
            "num_cases_total": len(summary_rows),
            "num_failures": len(failures),
            "requested_mode": args.mode,
            "failures": failures,
        }, f, indent=2, ensure_ascii=False)

    # save failures.csv
    if failures:
        fail_csv = os.path.join(args.outdir, "failures.csv")
        fail_fields = sorted(set().union(*[d.keys() for d in failures]))
        with open(fail_csv, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fail_fields)
            writer.writeheader()
            for row in failures:
                writer.writerow(row)

    print("\n" + "=" * 88)
    print("Batch analysis finished.")
    print(f"Summary CSV      : {summary_csv}")
    print(f"Grouped Summary  : {grouped_csv}")
    print(f"Run Summary JSON : {summary_json}")
    if failures:
        print(f"Failures         : {len(failures)}")
        print(f"Failures CSV     : {os.path.join(args.outdir, 'failures.csv')}")
    else:
        print("Failures         : 0")


if __name__ == "__main__":
    main()