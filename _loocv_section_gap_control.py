# -*- coding: utf-8 -*-
"""LOOCV 关键对照 2（V7 + MSE 聚合口径，2026-09-23 重写）

问题：`loocv_summary.csv` 的 `Gap_RMSE_last_pct`（折内"留出断面 vs 其余断面"的差）
究竟反映**空间泛化间隙**，还是仅仅是**断面固有难度差**？

做法：在**未掩码**的 V7 主实验（`Ablation_New_5d/H{hz}/V7_wo_VerticalGCN`）上，
用与 LOOCV **完全相同**的口径计算同样的 gap。若未掩码时 gap 就已存在，且其大小
与 LOOCV 的 gap 相当（并与断面自身难度高度相关），则该量由断面难度构成决定，
不能当作泛化证据。

口径（2026-09-23 统一，与 `Spatial_LOOCV.compute_loocv_summary` 一致）：
  一律 **MSE 聚合** —— 逐 (断面, 种子) 先算末步 MSE，聚合后再取
      RMSE = sqrt(mean MSE)。
  两侧均直接读 `predictions.npz`（`pred_cum`/`true_cum`，形状 (ns, hz, N)），
  故口径逐位可比。**不再**使用"逐节点 RMSE_last 取中位数"或"逐节点 R² 中位数"
  （旧口径在小变形节点上会给出 −20 量级的伪值）。

  切换点：旧口径读 `node_metrics.xlsx` 的逐节点中位数；新口径读 `predictions.npz`
  的场量。若 `Spatial_LOOCV_Results` 尚无 `section_metrics_all.csv`（新脚本产物），
  脚本判定其为**旧口径（留出断面仍参与训练）结果**并告警，且不写 CSV。

用法：
  python _loocv_section_gap_control.py [--loocv-root <dir>] [--abl-root <dir>]
"""
from _env import AUX_ROOT, WORK_ROOT

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

ABL_DEFAULT = AUX_ROOT / "Ablation_New_5d"
LOOCV_DEFAULT = WORK_ROOT / "Spatial_LOOCV_Results"
SECTIONS = ["CX2", "CX3", "CX8", "CX13", "CX15"]
SEEDS5 = [42, 15, 77, 85, 120]
HZ = [2, 6, 8, 10, 12]
BASE_VARIANT = "V7_wo_VerticalGCN"      # 2026-09-23：口径由 V5 改为 V7（论文主模型）


# -----------------------------------------------------------------------------
# 共享小工具（与 Spatial_LOOCV 同口径；本脚本独立运行故就地实现）
# -----------------------------------------------------------------------------
def section_blocks(nodes):
    """按节点顺序切出各断面的列索引。"""
    blocks = {}
    for j, n in enumerate(nodes):
        blocks.setdefault(str(n).split('_')[0], []).append(j)
    return blocks


def section_mse(pred_cum, true_cum, blocks):
    """逐断面末步 MSE（(ns, hz, N) → {sec: mse_last}）。"""
    out = {}
    for sec, idx in blocks.items():
        p, t = pred_cum[:, -1, idx], true_cum[:, -1, idx]
        out[sec] = float(np.mean((p - t) ** 2))
    return out


def read_nodes(abl_root, hz):
    """从主实验 V7 的 node_metrics.xlsx 读节点顺序（155 个，权威顺序）。"""
    for s in SEEDS5:
        f = abl_root / f"H{hz}" / BASE_VARIANT / f"run_{s}" / "node_metrics.xlsx"
        if f.exists():
            return pd.read_excel(f)["Node"].astype(str).tolist()
    return None


def load_preds(root, hz, fold=None):
    """读 5 个种子的 predictions.npz（fold=None 时按主实验 run_ 目录布局）。

    返回 list[np.ndarray] 或 None（缺任一种子）。
    """
    base = (root / f"H{hz}" / BASE_VARIANT) if fold is None \
        else (root / f"Fold_{fold}" / f"H{hz}")
    zs = []
    for s in SEEDS5:
        f = base / f"run_{s}" / "predictions.npz"
        if not f.exists():
            return None
        zs.append(np.load(f))
    return zs


def agg_rmse(mse_list):
    """MSE 聚合：RMSE = sqrt(mean MSE)。"""
    return float(np.sqrt(np.mean(mse_list))) if len(mse_list) else np.nan


def warn_if_old_caliber(loocv_root):
    """新口径的 LOOCV 会写 section_metrics_all.csv；缺失即判为旧（含泄露）结果。"""
    if not loocv_root.exists():
        print(f"!! LOOCV 根目录不存在：{loocv_root}")
        return True
    if (loocv_root / "Fold_CX2" / "H10" / "section_metrics_all.csv").exists():
        return False
    print("!" * 104)
    print("!! 警告：未找到 section_metrics_all.csv —— 该结果目录来自**旧口径**：")
    print("!!   ① 留出断面的**预测目标仍参与训练** ⇒ 实际任务是「由可见断面插补该断面」")
    print("!!      （模型已见过该断面的变形历史），而非「预测一个从未见过的断面」。")
    print("!!      与脚本/论文的「空间泛化」表述不符，故 gap 不能读作泛化间隙。")
    print("!!   ② 汇总曾用逐节点 RMSE_last 中位数（小变形节点会给出 −20 量级伪值）。")
    print("!!   请先重跑：python Spatial_LOOCV.py（约 8 分钟）。")
    print("!" * 104)
    return True


def main():
    ap = argparse.ArgumentParser(description="LOOCV 断面 gap 对照（V7 + MSE 聚合）")
    ap.add_argument('--loocv-root', type=Path, default=LOOCV_DEFAULT)
    ap.add_argument('--abl-root', type=Path, default=ABL_DEFAULT)
    args = ap.parse_args()
    loocv_root, abl_root = args.loocv_root, args.abl_root

    stale = warn_if_old_caliber(loocv_root)

    print("=" * 104)
    print(f"未掩码 {BASE_VARIANT}（主实验）逐视界的『断面间 gap』"
          f" vs LOOCV 的『留出断面 gap』—— MSE 聚合口径")
    print("=" * 104)

    rows = []
    for hz in HZ:
        nodes = read_nodes(abl_root, hz)
        if nodes is None:
            print(f"H={hz}: 未找到 {BASE_VARIANT} 的 node_metrics.xlsx，跳过")
            continue
        blocks = section_blocks(nodes)

        base_zs = load_preds(abl_root, hz)
        if base_zs is None:
            print(f"H={hz}: 未找到 {BASE_VARIANT} 的 predictions.npz，跳过")
            continue

        # ---- 未掩码 V7：sec -> [该断面逐种子 MSE] ----
        base_sec = {}
        for z in base_zs:
            for sec, mse in section_mse(z['pred_cum'], z['true_cum'], blocks).items():
                base_sec.setdefault(sec, []).append(mse)

        # ---- LOOCV：fold -> {sec -> [逐种子 MSE]} ----
        loo = {}
        for fold in SECTIONS:
            zs = load_preds(loocv_root, hz, fold=fold)
            if zs is None:
                continue
            per = {}
            for z in zs:
                for sec, mse in section_mse(z['pred_cum'], z['true_cum'],
                                            blocks).items():
                    per.setdefault(sec, []).append(mse)
            loo[fold] = per

        def gap_of(sec_mse_map, holdout):
            """(留出断面 RMSE − 其余 4 断面池化 RMSE) / 其余 × 100。"""
            if holdout not in sec_mse_map:
                return np.nan
            held = agg_rmse(sec_mse_map[holdout])
            others = [m for s2, v in sec_mse_map.items() if s2 != holdout for m in v]
            if not others:
                return np.nan
            oth = agg_rmse(others)
            return (held - oth) / oth * 100

        print(f"\nH = {hz}")
        print(f"  {'断面':<8} {'未掩码V7 gap%':>15} {'LOOCV gap%':>12} "
              f"{'两者差':>9}   {'该断面未掩码RMSE':>18} {'该断面LOOCV RMSE':>18}")
        base_gaps, loo_gaps, diffs, intr, loo_gap_by_sec = [], [], [], [], []
        for sec in SECTIONS:
            gb = gap_of(base_sec, sec)
            gl = gap_of(loo[sec], sec) if sec in loo else np.nan
            r_b = agg_rmse(base_sec.get(sec, []))
            r_l = agg_rmse(loo[sec][sec]) if sec in loo and sec in loo[sec] else np.nan
            base_gaps.append(gb)
            if np.isfinite(gl):
                loo_gaps.append(gl)
                diffs.append(gl - gb)
                intr.append(r_b)
                loo_gap_by_sec.append(gl)
            print(f"  {sec:<8} {gb:>+15.2f} "
                  f"{(f'{gl:+.2f}' if np.isfinite(gl) else '(无)'):>12} "
                  f"{(f'{gl - gb:+.2f}' if np.isfinite(gl) else ''):>9}   "
                  f"{r_b:>18.4f} {r_l:>18.4f}")
        print(f"  {'|均值|':<8} {np.mean(np.abs(base_gaps)):>+15.2f} "
              f"{np.mean(np.abs(loo_gaps)) if loo_gaps else np.nan:>+12.2f} "
              f"{np.mean(np.abs(diffs)) if diffs else np.nan:>+9.2f}")

        # 关键诊断：LOOCV 的 gap 是否只是"该断面本来就难"的单调函数
        r_corr = np.nan
        if len(intr) >= 3:
            r_corr = float(np.corrcoef(intr, loo_gap_by_sec)[0, 1])
            print(f"  → corr(该断面未掩码 RMSE, LOOCV gap) = {r_corr:+.3f}"
                  f"   （接近 ±1 ⇒ LOOCV 的 gap 主要由断面难度构成决定）")

        rows.append(dict(
            H=hz,
            base_gap_mean_pct=round(float(np.mean(base_gaps)), 2),
            base_gap_abs_mean_pct=round(float(np.abs(base_gaps).mean()), 2),
            loocv_gap_mean_pct=(round(float(np.mean(loo_gaps)), 2)
                                if loo_gaps else np.nan),
            loocv_gap_abs_mean_pct=(round(float(np.abs(loo_gaps).mean()), 2)
                                    if loo_gaps else np.nan),
            corr_difficulty_vs_loocv_gap=(round(r_corr, 3)
                                          if np.isfinite(r_corr) else np.nan),
        ))

    df = pd.DataFrame(rows)
    print("\n" + "=" * 104)
    print("汇总：未掩码 V7 的断面间 gap vs LOOCV 的 gap（|gap| 均值反映难度构成离散度）")
    print("=" * 104)
    print(df.to_string(index=False))

    if stale:
        print("\n（检测到旧口径 LOOCV 结果，未写出 CSV）")
    else:
        out = loocv_root / "_section_gap_control.csv"
        df.to_csv(out, index=False, encoding="utf-8-sig")
        print(f"\n已存：{out}")


if __name__ == "__main__":
    main()