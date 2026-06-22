"""
muscle_guidance_mdm/dense_loss.py

Dense（处处可微、起点梯度非零）的肌肉病态引导损失。

为什么需要它
-----------
`motion2muscle/posture_loss_torch.py` 的 `compute_posture_loss_torch` 是**检测型**损失：
四个分量每一项都是 `relu(偏离 − δ)` 或门控乘积，阈值 δ=0.2–0.5 是为"真实病理数据的大偏差"
设计的。在 MDM 引导起点，guided 激活 == baseline 参考，每一项都落在死区 →
**loss≡0、grad≡0**；grad=0 → 运动不动 → loss 一直 0（自锁，整段采样毫无推力）。

本模块把它翻转成**目标匹配**：保留 POSTURE_PRIORS 的四机制 + 肌群结构 + 方向，但用单边
`relu(target − current)` 把激活朝病理目标拉、到目标即止。性质：
  - 起点（current==ref）梯度非零 → guidance 能启动；
  - 到达方向性目标 ref·(1±margin) 后该项归零 → 不过冲；
  - **最小化该损失 = 朝病态**（与关节 hinge 同向，组合时直接相加，不需取负号）。

不修改肌肉队交付的文件：复用 `posture_loss_torch._group_act/_expand_sides` 与
`posture_loss.POSTURE_PRIORS`。
"""
from __future__ import annotations
from typing import Dict, Optional

import torch

# 复用肌肉队的肌群池化助手与病态模板（唯一真值来源）
from posture_loss_torch import _group_act, _expand_sides           # type: ignore
from posture_loss import POSTURE_PRIORS                            # type: ignore

DEFAULT_WEIGHTS = {"antagonist": 1.0, "chain": 1.5, "synergy": 0.5, "stabilizer": 0.5}


def _relu(x: torch.Tensor) -> torch.Tensor:
    return torch.clamp(x, min=0.0)


def dense_posture_guidance_loss(
    activations: torch.Tensor,
    group_index: Dict[str, torch.Tensor],
    posture_name: str,
    reference_acts: Dict[str, float],
    component_weights: Optional[Dict[str, float]] = None,
    margin: float = 0.3,
    normalize_by_count: bool = True,
    eps: float = 1e-6,
    return_components: bool = False,
):
    """
    Args:
        activations    : (B,T,402) 或 (T,402)，代理输出 [0,1]，带梯度。
        group_index    : build_group_index(...) 的结果（肌群→列下标）。
        posture_name   : POSTURE_PRIORS 中的病态名。
        reference_acts : {肌群: 标量}，来自正常样本，常量。
        component_weights / normalize_by_count : 同 clinical 版语义。
        margin         : 方向性目标幅度 m（target = ref·(1±m)）。默认 0.3。
    Returns:
        标量 tensor。**最小化 → 朝病态**。return_components=True 时附带各分量明细。
    """
    if posture_name not in POSTURE_PRIORS:
        raise ValueError(f"Unknown posture '{posture_name}'. Available: {list(POSTURE_PRIORS)}")

    cfg = POSTURE_PRIORS[posture_name]
    bilateral = cfg.get("bilateral", True)
    weights = dict(DEFAULT_WEIGHTS)
    if component_weights:
        weights.update(component_weights)

    zero = activations.new_zeros(())

    def gmean(name):
        """肌群逐帧池化后再做时间平均 -> (B,) 或标量；缺失返回 None。"""
        a = _group_act(activations, group_index, name)
        return None if a is None else a.mean(dim=-1)

    # ---- 1. antagonist：over-active 拉高到 ref·(1+m)，under-active 压低到 ref·(1−m) ----
    L_antag, n_antag = zero.clone(), 0
    for over, under, _ in cfg.get("antagonist_imbalances", []):
        for s_over, s_under in zip(_expand_sides(over, bilateral),
                                   _expand_sides(under, bilateral)):
            a_o, a_u = gmean(s_over), gmean(s_under)
            r_o, r_u = reference_acts.get(s_over), reference_acts.get(s_under)
            if a_o is None or a_u is None or r_o is None or r_u is None:
                continue
            up = _relu(r_o * (1.0 + margin) - a_o)     # 想更高
            down = _relu(a_u - r_u * (1.0 - margin))   # 想更低
            L_antag = L_antag + up.mean() + down.mean()
            n_antag += 1

    # ---- 2. chain：primary 压低、compensator 拉高 ----
    L_chain, n_chain = zero.clone(), 0
    for primary, comp, _ in cfg.get("compensation_chains", []):
        for s_p, s_c in zip(_expand_sides(primary, bilateral),
                            _expand_sides(comp, bilateral)):
            a_p, a_c = gmean(s_p), gmean(s_c)
            r_p, r_c = reference_acts.get(s_p), reference_acts.get(s_c)
            if a_p is None or a_c is None or r_p is None or r_c is None:
                continue
            down = _relu(a_p - r_p * (1.0 - margin))   # primary 想更低
            up = _relu(r_c * (1.0 + margin) - a_c)     # compensator 想更高
            L_chain = L_chain + down.mean() + up.mean()
            n_chain += 1

    # ---- 3. synergy：主导份额拉低到 (normal_share − shift) ----
    L_syn, n_syn = zero.clone(), 0
    for syn in cfg.get("synergy_imbalances", []):
        sides = ["_R", "_L"] if bilateral else [""]
        for side in sides:
            dom = [a for n in syn["dominant"] if (a := gmean(n + side)) is not None]
            comp = [a for n in syn["compensator"] if (a := gmean(n + side)) is not None]
            if not dom or not comp:
                continue
            dom_sum = torch.stack(dom).sum(dim=0)
            comp_sum = torch.stack(comp).sum(dim=0)
            dom_share = dom_sum / (dom_sum + comp_sum + eps)
            ref_dom = sum(reference_acts.get(n + side, 0.0) for n in syn["dominant"])
            ref_comp = sum(reference_acts.get(n + side, 0.0) for n in syn["compensator"])
            if ref_dom + ref_comp > eps:
                normal_share = ref_dom / (ref_dom + ref_comp + eps)
            else:
                normal_share = syn["normal_dominant_share"]
            target_share = normal_share - syn["shift"]
            L_syn = L_syn + _relu(dom_share - target_share).mean()   # 想更低
            n_syn += 1

    # ---- 4. stabilizer：压低到 ref·(1−frac) ----
    L_stab, n_stab = zero.clone(), 0
    for name, fraction_below in cfg.get("inhibited_stabilizers", []):
        for s in _expand_sides(name, bilateral):
            a = gmean(s)
            r = reference_acts.get(s)
            if a is None or r is None:
                continue
            L_stab = L_stab + _relu(a - r * (1.0 - fraction_below)).mean()  # 想更低
            n_stab += 1

    if normalize_by_count:
        if n_antag > 0: L_antag = L_antag / n_antag
        if n_chain > 0: L_chain = L_chain / n_chain
        if n_syn > 0:   L_syn = L_syn / n_syn
        if n_stab > 0:  L_stab = L_stab / n_stab

    total = (weights["antagonist"] * L_antag + weights["chain"] * L_chain
             + weights["synergy"] * L_syn + weights["stabilizer"] * L_stab)

    if not return_components:
        return total
    return total, {
        "antagonist": float(L_antag.detach()),
        "chain":      float(L_chain.detach()),
        "synergy":    float(L_syn.detach()),
        "stabilizer": float(L_stab.detach()),
        "rule_counts": {"antagonist": n_antag, "chain": n_chain,
                        "synergy": n_syn, "stabilizer": n_stab},
    }
