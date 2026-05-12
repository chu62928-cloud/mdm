import os
import csv
import math
import argparse
from collections import defaultdict

import numpy as np


def parse_args():
    parser = argparse.ArgumentParser(
        description="Judge suspicious motion cases from case_metrics.csv"
    )
    parser.add_argument("--input_csv", type=str, required=True,
                        help="Path to case_metrics.csv")
    parser.add_argument("--outdir", type=str, required=True,
                        help="Output directory")
    parser.add_argument("--topk", type=int, default=30,
                        help="How many top suspicious cases to keep in the ranked output")

    # contact risk thresholds
    parser.add_argument("--slide_ratio_warn", type=float, default=0.10)
    parser.add_argument("--slide_ratio_bad", type=float, default=0.25)
    parser.add_argument("--slide_mean_warn", type=float, default=0.03)
    parser.add_argument("--slide_mean_bad", type=float, default=0.08)
    parser.add_argument("--penetration_warn", type=float, default=0.01)
    parser.add_argument("--penetration_bad", type=float, default=0.03)

    # walk-like balance thresholds
    parser.add_argument("--mos_neg_ratio_warn", type=float, default=0.10)
    parser.add_argument("--mos_neg_ratio_bad", type=float, default=0.30)
    parser.add_argument("--mos_min_warn", type=float, default=-0.05)
    parser.add_argument("--mos_min_bad", type=float, default=-0.15)

    # bend-like balance thresholds
    parser.add_argument("--qs_neg_ratio_warn", type=float, default=0.10)
    parser.add_argument("--qs_neg_ratio_bad", type=float, default=0.30)
    parser.add_argument("--qs_margin_min_warn", type=float, default=-0.02)
    parser.add_argument("--qs_margin_min_bad", type=float, default=-0.05)

    # overall score thresholds
    parser.add_argument("--overall_warn", type=float, default=0.60,
                        help="If overall_suspicion_score exists, >= this means suspicious")
    parser.add_argument("--overall_bad", type=float, default=0.80,
                        help="If overall_suspicion_score exists, >= this means highly suspicious")

    return parser.parse_args()


def try_float(x):
    if x is None:
        return np.nan
    s = str(x).strip()
    if s == "" or s.lower() == "nan":
        return np.nan
    try:
        return float(s)
    except Exception:
        return np.nan


def read_csv(path):
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)
    return rows


def write_csv(path, rows):
    if not rows:
        with open(path, "w", newline="", encoding="utf-8") as f:
            pass
        return

    fieldnames = sorted(set().union(*[r.keys() for r in rows]))
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def risk_level_from_two_thresholds(value, warn_thr, bad_thr, larger_is_worse=True):
    if not np.isfinite(value):
        return "unknown", 0

    if larger_is_worse:
        if value >= bad_thr:
            return "high", 2
        elif value >= warn_thr:
            return "medium", 1
        else:
            return "low", 0
    else:
        if value <= bad_thr:
            return "high", 2
        elif value <= warn_thr:
            return "medium", 1
        else:
            return "low", 0


def summarize_reasons(reasons):
    if not reasons:
        return "no strong rule triggered"
    return "; ".join(reasons)


def judge_one_row(row, args):
    mode = str(row.get("analysis_mode", "")).strip().lower()
    prompt = row.get("prompt", "")

    slide_ratio = try_float(row.get("slide_ratio_pivot"))
    slide_mean = try_float(row.get("slide_mean_pivot"))
    penetration = try_float(row.get("penetration_max"))
    overall_score = try_float(row.get("overall_suspicion_score"))

    mos_neg_ratio = try_float(row.get("mos_negative_ratio"))
    mos_min = try_float(row.get("mos_min"))

    qs_neg_ratio = try_float(row.get("qs_negative_ratio"))
    qs_margin_min = try_float(row.get("qs_margin_min"))

    reasons = []
    contact_scores = []
    balance_scores = []

    # -----------------------------
    # contact risk
    # -----------------------------
    lv, sc = risk_level_from_two_thresholds(
        slide_ratio, args.slide_ratio_warn, args.slide_ratio_bad, larger_is_worse=True
    )
    contact_scores.append(sc)
    if lv == "high":
        reasons.append(f"high slide_ratio_pivot={slide_ratio:.4f}")
    elif lv == "medium":
        reasons.append(f"elevated slide_ratio_pivot={slide_ratio:.4f}")

    lv, sc = risk_level_from_two_thresholds(
        slide_mean, args.slide_mean_warn, args.slide_mean_bad, larger_is_worse=True
    )
    contact_scores.append(sc)
    if lv == "high":
        reasons.append(f"high slide_mean_pivot={slide_mean:.4f}")
    elif lv == "medium":
        reasons.append(f"elevated slide_mean_pivot={slide_mean:.4f}")

    lv, sc = risk_level_from_two_thresholds(
        penetration, args.penetration_warn, args.penetration_bad, larger_is_worse=True
    )
    contact_scores.append(sc)
    if lv == "high":
        reasons.append(f"high penetration_max={penetration:.4f}")
    elif lv == "medium":
        reasons.append(f"elevated penetration_max={penetration:.4f}")

    contact_score = int(max(contact_scores) if contact_scores else 0)
    if contact_score >= 2:
        contact_risk = "high"
    elif contact_score == 1:
        contact_risk = "medium"
    else:
        contact_risk = "low"

    # -----------------------------
    # balance risk
    # -----------------------------
    if mode == "walk":
        lv, sc = risk_level_from_two_thresholds(
            mos_neg_ratio, args.mos_neg_ratio_warn, args.mos_neg_ratio_bad, larger_is_worse=True
        )
        balance_scores.append(sc)
        if lv == "high":
            reasons.append(f"high mos_negative_ratio={mos_neg_ratio:.4f}")
        elif lv == "medium":
            reasons.append(f"elevated mos_negative_ratio={mos_neg_ratio:.4f}")

        lv, sc = risk_level_from_two_thresholds(
            mos_min, args.mos_min_warn, args.mos_min_bad, larger_is_worse=False
        )
        balance_scores.append(sc)
        if lv == "high":
            reasons.append(f"very low mos_min={mos_min:.4f}")
        elif lv == "medium":
            reasons.append(f"low mos_min={mos_min:.4f}")

    elif mode == "bend":
        lv, sc = risk_level_from_two_thresholds(
            qs_neg_ratio, args.qs_neg_ratio_warn, args.qs_neg_ratio_bad, larger_is_worse=True
        )
        balance_scores.append(sc)
        if lv == "high":
            reasons.append(f"high qs_negative_ratio={qs_neg_ratio:.4f}")
        elif lv == "medium":
            reasons.append(f"elevated qs_negative_ratio={qs_neg_ratio:.4f}")

        lv, sc = risk_level_from_two_thresholds(
            qs_margin_min, args.qs_margin_min_warn, args.qs_margin_min_bad, larger_is_worse=False
        )
        balance_scores.append(sc)
        if lv == "high":
            reasons.append(f"very low qs_margin_min={qs_margin_min:.4f}")
        elif lv == "medium":
            reasons.append(f"low qs_margin_min={qs_margin_min:.4f}")

    else:
        # mode unknown
        balance_scores.append(0)

    balance_score = int(max(balance_scores) if balance_scores else 0)
    if balance_score >= 2:
        balance_risk = "high"
    elif balance_score == 1:
        balance_risk = "medium"
    else:
        balance_risk = "low"

    # -----------------------------
    # overall risk
    # -----------------------------
    rule_score = max(contact_score, balance_score)

    # 如果 overall_suspicion_score 存在，就也参考它
    overall_rule_label = "low"
    if np.isfinite(overall_score):
        if overall_score >= args.overall_bad:
            overall_rule_label = "high"
            reasons.append(f"high overall_suspicion_score={overall_score:.4f}")
            rule_score = max(rule_score, 2)
        elif overall_score >= args.overall_warn:
            overall_rule_label = "medium"
            reasons.append(f"elevated overall_suspicion_score={overall_score:.4f}")
            rule_score = max(rule_score, 1)
        else:
            overall_rule_label = "low"

    if rule_score >= 2:
        overall_risk = "high"
    elif rule_score == 1:
        overall_risk = "medium"
    else:
        overall_risk = "low"

    judged = dict(row)
    judged["contact_risk"] = contact_risk
    judged["balance_risk"] = balance_risk
    judged["overall_risk"] = overall_risk
    judged["rule_reason"] = summarize_reasons(reasons)

    # 方便排序
    judged["_rule_numeric"] = rule_score
    judged["_overall_score_numeric"] = overall_score if np.isfinite(overall_score) else -1.0

    # 简单建议
    if overall_risk == "high":
        judged["suggestion"] = "strong candidate failure case; inspect mp4 + analysis plots"
    elif overall_risk == "medium":
        judged["suggestion"] = "possible issue; compare against nearby cases"
    else:
        judged["suggestion"] = "not a priority suspect"

    return judged


def prompt_level_summary(rows):
    groups = defaultdict(list)
    for r in rows:
        key = (r.get("prompt", ""), r.get("analysis_mode", ""))
        groups[key].append(r)

    out = []
    for (prompt, mode), items in groups.items():
        n = len(items)
        high_n = sum(1 for x in items if x.get("overall_risk") == "high")
        med_n = sum(1 for x in items if x.get("overall_risk") == "medium")
        low_n = sum(1 for x in items if x.get("overall_risk") == "low")

        def mean_of(col):
            vals = [try_float(x.get(col)) for x in items]
            vals = [v for v in vals if np.isfinite(v)]
            return float(np.mean(vals)) if vals else np.nan

        out.append({
            "prompt": prompt,
            "analysis_mode": mode,
            "num_cases": n,
            "num_high_risk": high_n,
            "num_medium_risk": med_n,
            "num_low_risk": low_n,
            "high_risk_ratio": high_n / n if n > 0 else np.nan,
            "medium_or_higher_ratio": (high_n + med_n) / n if n > 0 else np.nan,
            "mean_overall_suspicion_score": mean_of("overall_suspicion_score"),
            "mean_slide_ratio_pivot": mean_of("slide_ratio_pivot"),
            "mean_slide_mean_pivot": mean_of("slide_mean_pivot"),
            "mean_penetration_max": mean_of("penetration_max"),
            "mean_mos_negative_ratio": mean_of("mos_negative_ratio"),
            "mean_mos_min": mean_of("mos_min"),
            "mean_qs_negative_ratio": mean_of("qs_negative_ratio"),
            "mean_qs_margin_min": mean_of("qs_margin_min"),
        })

    # 先按高风险比例，再按中高风险比例排序
    out.sort(
        key=lambda x: (
            -(x["high_risk_ratio"] if np.isfinite(x["high_risk_ratio"]) else -1),
            -(x["medium_or_higher_ratio"] if np.isfinite(x["medium_or_higher_ratio"]) else -1),
            -(x["mean_overall_suspicion_score"] if np.isfinite(x["mean_overall_suspicion_score"]) else -1),
        )
    )
    return out


def main():
    args = parse_args()
    os.makedirs(args.outdir, exist_ok=True)

    rows = read_csv(args.input_csv)
    if not rows:
        print("No rows found in input_csv.")
        return

    judged_rows = [judge_one_row(r, args) for r in rows]

    # 排序：先按 overall_risk，再按 overall_suspicion_score
    risk_order = {"high": 2, "medium": 1, "low": 0}
    judged_rows.sort(
        key=lambda x: (
            -risk_order.get(x.get("overall_risk", "low"), 0),
            -x.get("_overall_score_numeric", -1.0),
            -x.get("_rule_numeric", 0),
        )
    )

    # 全部写出
    all_csv = os.path.join(args.outdir, "suspect_cases_by_rule.csv")
    rows_to_save = []
    for r in judged_rows:
        rr = dict(r)
        rr.pop("_rule_numeric", None)
        rr.pop("_overall_score_numeric", None)
        rows_to_save.append(rr)
    write_csv(all_csv, rows_to_save)

    # topk
    topk_csv = os.path.join(args.outdir, "top_suspect_cases_by_rule.csv")
    write_csv(topk_csv, rows_to_save[:args.topk])

    # 只保留 medium/high
    filtered_csv = os.path.join(args.outdir, "medium_high_risk_cases.csv")
    filtered_rows = [r for r in rows_to_save if r.get("overall_risk") in ["medium", "high"]]
    write_csv(filtered_csv, filtered_rows)

    # prompt级别汇总
    prompt_rows = prompt_level_summary(rows_to_save)
    prompt_csv = os.path.join(args.outdir, "suspect_prompt_summary_by_rule.csv")
    write_csv(prompt_csv, prompt_rows)

    print("=" * 72)
    print("Judging finished.")
    print(f"All judged cases      : {all_csv}")
    print(f"Top suspect cases     : {topk_csv}")
    print(f"Medium/high risk only : {filtered_csv}")
    print(f"Prompt-level summary  : {prompt_csv}")
    print("=" * 72)


if __name__ == "__main__":
    main()