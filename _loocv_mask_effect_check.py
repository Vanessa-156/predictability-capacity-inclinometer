# -*- coding: utf-8 -*-
"""LOOCV 关键对照 1（V7 + MSE 聚合口径，2026-09-23 重写）

问题：折内"掩码留出断面"这一操作**是否真的产生了效应**？

两问：
  Q1（整体） 折内**全局**指标（全 155 节点）与未掩码 V7 主实验相比是否劣化？
  Q2（靶点） 折内**留出断面**的指标是否比未掩码 V7 中同一断面更差？

判读：
  - Q2 为正（留出断面确实更差）⇒ 掩码生效，`Spatial_LOOCV` 测的确实是"未见断面"。
  - Q2 ≈ 0 或为负 ⇒ 掩码未形成实质扰动，该实验不能支撑"空间泛化"主张。
  - Q1 幅度通常远小于 Q2（留出断面仅占 31/155 ≈ 20% 节点），故 Q1 不应作为主判据。

口径（2026-09-23 统一）：一律 **MSE 聚合** —— 逐 (对象, 种子) 先算末步 MSE，
聚合后取 RMSE = sqrt(mean MSE)；配对检验用逐种子的 RMSE_last。
两侧均读 `predictions.npz`，口径逐位可比。不再用逐节点 RMSE_last 中位数。

  V7 = `Ablation_New_5d/H{hz}/V7_wo_VerticalGCN`（2026-09-23 由 V5 改为 V7）。
  若 `Spatial_LOOCV_Results` 尚无 `section_metrics_all.csv`，脚本判定其为
  **旧口径（留出断面仍参与训练）结果**并告警，且不写 CSV。

用法：
  python _loocv_mask_effect_check.py [--loocv-root <dir>] [--abl-root <dir>]
"""
from _env import AUX_ROOT, WORK_ROOT

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

ABL_DEFAULT = AUX_ROOT / "Ablation_New_5d"
LOOCV_DEFAULT = WORK_ROOT / "Spatial_LOOCV_Results"
SECTIONS = ["CX2", "CX3", "CX8", "CX13", "CX15"]
SEEDS5 = [42, 15, 77, 85, 120]
HZ = [2, 6, 8, 10, 12]
BASE_VARIANT = "V7_wo_VerticalGCN"


def section_blocks(nodes):
    blocks = {}
    for j, n in enumerate(nodes):
        blocks.setdefault(str(n).split('_')[0], []).append(j)
    return blocks


def rmse_last_global(pred_cum, true_cum):
    """全场末步逐种子 RMSE（用于配对检验）。"""
    p, t = pred_cum[:, -1, :], true_cum[:, -1, :]
    return float(np.sqrt(np.mean((p - t) ** 2)))


def rmse_last_section(pred_cum, true_cum, idx):
    """指定断面列末步逐种子 RMSE。"""
    p, t = pred_cum[:, -1, idx], true_cum[:, -1, idx]
    return float(np.sqrt(np.mean((p - t) ** 2)))


def read_nodes(abl_root, hz):
    for s in SEEDS5:
        f = abl_root / f"H{hz}" / BASE_VARIANT / f"run_{s}" / "node_metrics.xlsx"
        if f.exists():
            return pd.read_excel(f)["Node"].astype(str).tolist()
    return None


def load_preds(root, hz, fold=None):
    base = (root / f"H{hz}" / BASE_VARIANT) if fold is None \
        else (root / f"Fold_{fold}" / f"H{hz}")
    zs = []
    for s in SEEDS5:
        f = base / f"run_{s}" / "predictions.npz"
        if not f.exists():
            return None
        zs.append(np.load(f))
    return zs


def warn_if_old_caliber(loocv_root):
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
    ap = argparse.ArgumentParser(description="LOOCV 掩码效应检验（V7 + MSE 聚合）")
    ap.add_argument('--loocv-root', type=Path, default=LOOCV_DEFAULT)
    ap.add_argument('--abl-root', type=Path, default=ABL_DEFAULT)
    args = ap.parse_args()
    loocv_root, abl_root = args.loocv_root, args.abl_root

    stale = warn_if_old_caliber(loocv_root)

    print("=" * 104)
    print("掩码效应检验：LOOCV 各折 vs 未掩码 " + BASE_VARIANT +
          "（同 5 种子，MSE 聚合；配对检验用逐种子 RMSE_last）")
    print("=" * 104)
    print(f"  Q1 = 全局（155 节点）劣化%    Q2 = 留出断面劣化%   正 = 掩码后更差")

    rows = []
    for hz in HZ:
        nodes = read_nodes(abl_root, hz)
        if nodes is None:
            print(f"H={hz}: 未找到 node_metrics.xlsx，跳过")
            continue
        blocks = section_blocks(nodes)
        base_zs = load_preds(abl_root, hz)
        if base_zs is None:
            print(f"H={hz}: 未找到 predictions.npz，跳过")
            continue

        base_g = np.array([rmse_last_global(z['pred_cum'], z['true_cum'])
                           for z in base_zs])
        base_sec = {sec: np.array([rmse_last_section(z['pred_cum'], z['true_cum'], idx)
                                   for z in base_zs])
                    for sec, idx in blocks.items()}

        print(f"\nH = {hz}")
        print(f"  {'留出断面':<9} {'Q1 全局 未掩码→折内':>24} {'Q1%':>8} {'p':>8}"
              f"   {'Q2 该断面 未掩码→折内':>24} {'Q2%':>8} {'p':>8} {'：V7 胜出':>8}")

        for fold in SECTIONS:
            zs = load_preds(loocv_root, hz, fold=fold)
            if zs is None or fold not in blocks:
                continue
            fol_g = np.array([rmse_last_global(z['pred_cum'], z['true_cum'])
                              for z in zs])
            fol_s = np.array([rmse_last_section(z['pred_cum'], z['true_cum'],
                                                blocks[fold]) for z in zs])
            bs = base_sec[fold]

            # MSE 聚合口径的整体劣化
            q1 = (np.sqrt(np.mean(fol_g ** 2)) - np.sqrt(np.mean(base_g ** 2))) \
                / np.sqrt(np.mean(base_g ** 2)) * 100
            q2 = (np.sqrt(np.mean(fol_s ** 2)) - np.sqrt(np.mean(bs ** 2))) \
                / np.sqrt(np.mean(bs ** 2)) * 100
            p_g = stats.ttest_rel(fol_g, base_g).pvalue
            p_s = stats.ttest_rel(fol_s, bs).pvalue
            win = "V7" if q2 > 0 else "折内"

            print(f"  {fold:<9} {np.sqrt(np.mean(base_g**2)):>10.4f}→"
                  f"{np.sqrt(np.mean(fol_g**2)):<10.4f} {q1:>+8.2f} {p_g:>8.3f}"
                  f"   {np.sqrt(np.mean(bs**2)):>10.4f}→"
                  f"{np.sqrt(np.mean(fol_s**2)):<10.4f} {q2:>+8.2f} {p_s:>8.3f}"
                  f" {win:>8}")
            rows.append(dict(H=hz, Fold=fold,
                             base_global_rmse=round(float(np.sqrt(np.mean(base_g**2))), 4),
                             masked_global_rmse=round(float(np.sqrt(np.mean(fol_g**2))), 4),
                             q1_global_pct=round(float(q1), 2), p_global=round(float(p_g), 4),
                             base_held_rmse=round(float(np.sqrt(np.mean(bs**2))), 4),
                             masked_held_rmse=round(float(np.sqrt(np.mean(fol_s**2))), 4),
                             q2_held_pct=round(float(q2), 2), p_held=round(float(p_s), 4)))

    df = pd.DataFrame(rows)
    if len(df):
        print("\n" + "=" * 104)
        print("汇总（MSE 聚合）：正 = 掩码后更差；Q2 应显著为正才说明掩码生效")
        print("=" * 104)
        print(df.to_string(index=False))
        n_pos_g = int((df['q1_global_pct'] > 0).sum())
        n_pos_h = int((df['q2_held_pct'] > 0).sum())
        print(f"\n  Q1 为正的折数 = {n_pos_g}/{len(df)}"
              f"   |  Q2 为正的折数 = {n_pos_h}/{len(df)}")
        print(f"  Q2 均值 = {df['q2_held_pct'].mean():+.2f}%"
              f"   |  Q2 显著（p<0.05）折数 = "
              f"{int((df['p_held'] < 0.05).sum())}/{len(df)}")

        if stale:
            print("\n（检测到旧口径 LOOCV 结果，未写出 CSV）")
        else:
            out = loocv_root / "_mask_effect_check.csv"
            df.to_csv(out, index=False, encoding="utf-8-sig")
            print(f"\n已存：{out}")
    else:
        print("\n未收集到任何折（确认 LOOCV 是否已跑）。")


if __name__ == "__main__":
    main()