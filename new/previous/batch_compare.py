"""
scripts/batch_compare.py

批量对照实验脚本：
    遍历多个 (text_prompt, posture_combination) 组合，
    自动跑 run_posture_comparison + quantitative_compare，
    最终把所有结果汇总到一个 CSV 表格。

适用于：
    - 系统性验证 guidance 在不同 prompt / 不同体态下的鲁棒性
    - 调参时快速扫描 lr / lbfgs_steps 的敏感性
    - 写论文时一次性生成所有对照实验数据

调用：
    python -m scripts.batch_compare \
        --model_path ./save/humanml_trans_enc_512/model000200000.pt \
        --output_root ./output/batch_experiments \
        --config scripts/batch_config.yaml

YAML 配置示例（batch_config.yaml）：

    prompts:
      - "a person is walking forward"
      - "a person is standing still"
      - "a person walks slowly"

    posture_combinations:
      - ["骨盆前倾"]
      - ["膝超伸"]
      - ["骨盆前倾", "膝超伸"]
      - ["驼背", "头前伸"]

    seeds: [42, 123, 2024]

    guidance_configs:
      - { lr: 0.05, steps: 5 }
      - { lr: 0.10, steps: 5 }
      - { lr: 0.05, steps: 10 }
"""

import os
import sys
import math
import json
import argparse
import subprocess
import itertools
import csv
from datetime import datetime
from pathlib import Path

import numpy as np
import torch

try:
    import yaml
except ImportError:
    print("[Warning] pyyaml 未安装，仅支持 JSON 配置")
    yaml = None

from posture_guidance.registry import POSTURE_REGISTRY, resolve_instruction


# =========================================================
# 配置加载
# =========================================================

def load_config(path):
    """支持 YAML 或 JSON 配置文件"""
    ext = os.path.splitext(path)[1].lower()
    with open(path, "r", encoding="utf-8") as f:
        if ext in (".yaml", ".yml"):
            assert yaml is not None, "需要安装 pyyaml: pip install pyyaml"
            return yaml.safe_load(f)
        elif ext == ".json":
            return json.load(f)
        else:
            raise ValueError(f"不支持的配置格式: {ext}")


def default_config():
    """默认扫描配置（用户没传 --config 时用）"""
    return {
        "prompts": [
            "a person is walking forward",
            "a person is standing still",
        ],
        "posture_combinations": [
            ["骨盆前倾"],
            ["膝超伸"],
            ["骨盆前倾", "膝超伸"],
        ],
        "seeds": [42],
        "guidance_configs": [
            {"lr": 0.05, "steps": 5},
        ],
    }


# =========================================================
# 单次实验执行（直接调用 Python 函数，不通过 subprocess）
# =========================================================

def run_one_experiment(
    base_args, prompt, posture_list, seed, lr, steps, exp_dir,
):
    """
    跑一次对照实验 + 量化分析。
    返回该实验的指标字典。
    """
    from copy import deepcopy

    # 修改 args
    args = deepcopy(base_args)
    args.text_prompt           = prompt
    args.posture_instructions  = posture_list
    args.seed                  = seed
    args.posture_lr            = lr
    args.posture_lbfgs_steps   = steps
    args.output_dir            = exp_dir
    args.comparison_output     = "comparison.npy"

    os.makedirs(exp_dir, exist_ok=True)

    # 直接调用 run_posture_comparison.main()
    # 这里我们 import 子模块的 main 而不是 subprocess，避免重复加载模型
    from scripts.run_posture_comparison import main as run_main
    sys.argv = [
        "run_posture_comparison",
        "--model_path",            base_args.model_path,
        "--text_prompt",           prompt,
        "--seed",                  str(seed),
        "--num_samples",           str(getattr(base_args, "num_samples", 1)),
        "--num_repetitions",       str(getattr(base_args, "num_repetitions", 1)),
        "--motion_length",         str(getattr(base_args, "motion_length", 6.0)),
        "--output_dir",            exp_dir,
        "--posture_instructions",  *posture_list,
        "--posture_lr",            str(lr),
        "--posture_lbfgs_steps",   str(steps),
    ]

    try:
        run_main()
    except SystemExit:
        pass  # argparse 在 main 里可能调 sys.exit
    except Exception as e:
        print(f"[Error] 实验失败: {e}")
        return None

    # 读结果，计算指标
    npy_path = os.path.join(exp_dir, "comparison.npy")
    if not os.path.exists(npy_path):
        print(f"[Error] 输出文件不存在: {npy_path}")
        return None

    return compute_summary_metrics(npy_path)


# =========================================================
# 简化的指标计算（不依赖 quantitative_compare 的打印逻辑）
# =========================================================

def compute_summary_metrics(npy_path):
    """
    从 comparison.npy 读结果，计算每个体态的关键指标。
    返回 dict，每个体态对应一组指标。
    """
    data = np.load(npy_path, allow_pickle=True).item()

    posture_instructions = data.get("posture_instructions", [])
    xyz_base   = data["motion_xyz"]
    xyz_guided = data["motion_xyz_guided"]

    # (B, J, 3, T) → (B, T, J, 3)
    q_base   = torch.from_numpy(xyz_base).permute(0, 3, 1, 2).float()
    q_guided = torch.from_numpy(xyz_guided).permute(0, 3, 1, 2).float()

    metrics = {
        "rmse_overall": float(np.sqrt(np.mean((xyz_guided - xyz_base) ** 2))),
        "postures": {},
    }

    for inst in posture_instructions:
        for sn in resolve_instruction(inst):
            spec = POSTURE_REGISTRY[sn]

            angle_b = spec.angle_fn(q_base,   **spec.angle_fn_kwargs)
            angle_g = spec.angle_fn(q_guided, **spec.angle_fn_kwargs)

            if spec.unit == "deg":
                angle_b = (angle_b * 180.0 / math.pi).cpu().numpy()
                angle_g = (angle_g * 180.0 / math.pi).cpu().numpy()
                target  = spec.target_deg
                tol     = spec.tolerance_deg
            else:
                angle_b = angle_b.cpu().numpy()
                angle_g = angle_g.cpu().numpy()
                target  = spec.target_deg
                tol     = spec.tolerance_deg

            # 命中率
            if spec.direction == "greater_than":
                hit = float((angle_g.flatten() >= target - tol).mean())
            elif spec.direction == "less_than":
                hit = float((angle_g.flatten() <= target + tol).mean())
            else:
                hit = float((np.abs(angle_g.flatten() - target) <= tol).mean())

            metrics["postures"][sn] = {
                "baseline_mean": float(np.mean(angle_b)),
                "guided_mean":   float(np.mean(angle_g)),
                "delta":         float(np.mean(angle_g) - np.mean(angle_b)),
                "target":        float(target),
                "hit_rate":      hit,
                "unit":          spec.unit,
            }

    return metrics


# =========================================================
# 结果汇总到 CSV
# =========================================================

def write_summary_csv(all_results, csv_path):
    """
    把所有实验结果汇总成一个 CSV，方便 Excel 分析。
    每行一个 (prompt, posture_combo, seed, config, posture) 组合。
    """
    rows = []
    for r in all_results:
        if r["metrics"] is None:
            continue
        base = {
            "prompt":      r["prompt"],
            "posture_set": ", ".join(r["posture_list"]),
            "seed":        r["seed"],
            "lr":          r["lr"],
            "steps":       r["steps"],
            "rmse":        r["metrics"]["rmse_overall"],
        }
        for posture_name, pm in r["metrics"]["postures"].items():
            row = {
                **base,
                "posture":       posture_name,
                "baseline_mean": pm["baseline_mean"],
                "guided_mean":   pm["guided_mean"],
                "delta":         pm["delta"],
                "target":        pm["target"],
                "hit_rate":      pm["hit_rate"],
                "unit":          pm["unit"],
            }
            rows.append(row)

    if not rows:
        print("[Warning] 没有任何成功的实验，CSV 为空")
        return

    keys = list(rows[0].keys())
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)
    print(f"\n✓ Summary CSV → {csv_path}")
    print(f"  total rows: {len(rows)}")


# =========================================================
# 主流程
# =========================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path",   type=str, required=True)
    parser.add_argument("--config",       type=str, default=None,
                        help="YAML/JSON 配置文件，定义扫描空间")
    parser.add_argument("--output_root",  type=str, required=True)
    parser.add_argument("--num_samples",  type=int, default=1)
    parser.add_argument("--motion_length", type=float, default=6.0)
    parser.add_argument("--dataset",      type=str, default="humanml")
    parser.add_argument("--device",       type=int, default=0)
    args, _ = parser.parse_known_args()

    # 加载扫描配置
    if args.config:
        cfg = load_config(args.config)
    else:
        cfg = default_config()

    prompts          = cfg.get("prompts", [])
    posture_combos   = cfg.get("posture_combinations", [])
    seeds            = cfg.get("seeds", [42])
    guidance_configs = cfg.get("guidance_configs", [{"lr": 0.05, "steps": 5}])

    total = (
        len(prompts) * len(posture_combos) *
        len(seeds)   * len(guidance_configs)
    )
    print(f"\nBatch experiment plan:")
    print(f"  prompts:           {len(prompts)}")
    print(f"  posture combos:    {len(posture_combos)}")
    print(f"  seeds:             {len(seeds)}")
    print(f"  guidance configs:  {len(guidance_configs)}")
    print(f"  total experiments: {total}")

    # 准备根目录
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    root = os.path.join(args.output_root, f"batch_{timestamp}")
    os.makedirs(root, exist_ok=True)

    # 备份配置
    with open(os.path.join(root, "config.json"), "w") as f:
        json.dump(cfg, f, indent=2, ensure_ascii=False)

    all_results = []
    idx = 0

    for prompt, posture_list, seed, gc in itertools.product(
        prompts, posture_combos, seeds, guidance_configs
    ):
        idx += 1
        lr    = gc["lr"]
        steps = gc["steps"]

        # 实验子目录命名
        prompt_slug = prompt.replace(" ", "_")[:30]
        posture_slug = "+".join(posture_list)
        exp_name = (
            f"{idx:03d}_{prompt_slug}_{posture_slug}_"
            f"seed{seed}_lr{lr}_s{steps}"
        )
        exp_dir = os.path.join(root, exp_name)

        print()
        print("█" * 70)
        print(f"  [{idx}/{total}] {exp_name}")
        print("█" * 70)

        # 用一个最小化的 args 对象传给 run_one_experiment
        class A: pass
        base_args = A()
        base_args.model_path     = args.model_path
        base_args.num_samples    = args.num_samples
        base_args.num_repetitions = 1
        base_args.motion_length  = args.motion_length
        base_args.dataset        = args.dataset
        base_args.device         = args.device

        metrics = run_one_experiment(
            base_args, prompt, posture_list,
            seed, lr, steps, exp_dir,
        )

        all_results.append({
            "prompt":       prompt,
            "posture_list": posture_list,
            "seed":         seed,
            "lr":           lr,
            "steps":        steps,
            "exp_dir":      exp_dir,
            "metrics":      metrics,
        })

    # 写汇总 CSV
    csv_path = os.path.join(root, "summary.csv")
    write_summary_csv(all_results, csv_path)

    # 写汇总 JSON（保留完整信息）
    json_path = os.path.join(root, "summary.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)
    print(f"✓ Summary JSON → {json_path}")

    print(f"\n✓ Batch root → {root}")


if __name__ == "__main__":
    main()