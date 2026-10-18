# -*- coding: utf-8 -*-
"""_vicd_layer2_criteria.py — VICD **层 2 判据化**数值。

背景：现 `_vicd_null_distribution.py::verification_position` 的 `cum_over` 路径
**实测永不可达** —— 保谱代理施加在**累计量**上，代理仍保留趋势 ⇒ 零分布与观测同量级
（沉降 0.786 vs 阈 0.812）。根因不是采样不足，而是**代理的施加域与待判量不匹配**。
故层 2 目前只能按"物理角色"定位，不做显著性主张。

本脚本实现 §3.40 A 的两个零假设，把层 2 从"定位"升级为"**判据**"：

  **零假设 A（主）：差分域保谱代理**
      H0 = "u 与 y 在**增量**尺度上无关，累计耦合全部来自各自独立的长期趋势"。
      构造：Δu → `phase_surrogate(Δu)` → `cumsum(·)` = u_cum^null。
      代理序列仍随机游走（谱保真），但**趋势走向与 y 独立** ⇒ 阈值可达。
      统计量 S = |ρ(y_cum, u_cum)|；S_obs > S_null(95%) ⇒ 层 2 判"验证资格成立"。

  **零假设 B（独立佐证）：空间错配置换**
      H0 = "任意沉降测点与任意位移孔的累计耦合与真配对无差别"。
      参照集 = 3 真配对（S6↔CX2 / S18↔CX8 / S31↔CX15）之外的 **12 个错配对**（3×5−3）；
      统计量 = 真配对 mean(|ρ|)；**精确枚举**全部 C(15,3)=455 个三元组算 p 值。

  **判定**：两个零假设都越界 ⇒ 升为"定量一致性验证"；任一未越界 ⇒ 维持"仅定性"。
  **必做标定**：FPR（独立 AR(1) 游走，φ=0/0.5/0.9，要求 ≈5%）+ 判别力正对照
  （注入真实增量驱动 + 共享趋势 ⇒ 应判越界）。

诚实边界（须一并写入论文）：
  · 仅适用"同量异器"（沉降↔位移）；异质变量（水位/锚索）仍只作定位。
  · 差分域代理会同时破坏 u 的真实趋势走向 —— 这是 H0 的**定义**，但意味着判据只回答
    "是否超出共存趋势"，**不能**区分"纯趋势耦合"与"趋势 + 机制耦合"。
  · 空间置换 p 值下界受组合数限制（3 vs 12，最小 p ≈ 1/455 ≈ 0.0022）。

零训练、纯数值，分钟级。
用法：
  python -u _vicd_layer2_criteria.py

环境变量：B（代理次数，默认 1000）、N_REP（标定重复数，默认 300）、ALPHA（默认 0.95）
"""
from _env import DATA_ROOT, WORK_ROOT

import os
import sys
import argparse
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(WORK_ROOT))

import Ablation_Experiment as AE
import _vicd_null_distribution as V1      # 复用 phase_surrogate / safe_corr / ar1 / 常量

SEED = V1.SEED
TOL_DAYS = V1.TOL_DAYS
OUT = WORK_ROOT / "_vicd_layer2"

# 3 个真配对（与 V1.build_records 的 settlement 分支一致）
TRUE_PAIRS = {"S6": "CX2", "S18": "CX8", "S31": "CX15"}
SECTIONS = ["CX2", "CX3", "CX8", "CX13", "CX15"]
SETTLE_DIR = DATA_ROOT / "shuju/沉降"
LAYERS = ("全深度", "浅(<3m)", "中(3-8m)", "深(8-16m)")


# ============================================================ 零假设 A
def diff_domain_test(y_cum, u_cum, B, rng, alpha=0.95):
    """差分域保谱代理：返回 (S_obs, thr, S_null)。

    H0 构造：对**增量**做相位随机化后再累积。代理仍是随机游走（谱/自相关保真），
    但其趋势走向与 y 独立 ⇒ 与"累计域直接打代理"不同，阈值是可达的。
    """
    y = np.asarray(y_cum, float)
    u = np.asarray(u_cum, float)
    S_obs = abs(V1.safe_corr(y, u))
    du = np.diff(u)
    if len(du) < 10:
        return S_obs, np.nan, np.empty(0)
    S_null = np.empty(B, float)
    for b in range(B):
        u_null = np.concatenate([[u[0]], u[0] + np.cumsum(V1.phase_surrogate(du, rng))])
        S_null[b] = abs(V1.safe_corr(y, u_null))
    thr = float(np.percentile(S_null[np.isfinite(S_null)], alpha * 100))
    return S_obs, thr, S_null


# ============================================================ 标定
def fpr_ar1(phi, n, B, n_rep, rng, alpha=0.95):
    """H0 为真：y 与 u 为**相互独立**的 AR(1) 累积游走（无任何耦合）⇒ 期望 FPR ≈ 5%。"""
    hit = 0
    for _ in range(n_rep):
        y = np.cumsum(V1.ar1(n, phi, rng))
        u = np.cumsum(V1.ar1(n, phi, rng))
        S_obs, thr, _ = diff_domain_test(y, u, B, rng, alpha)
        if np.isfinite(thr) and S_obs > thr:
            hit += 1
    return hit / n_rep


def power_injected(n, B, n_rep, rng, alpha=0.95, w=0.5):
    """备择为真：u 的**增量**含 y 增量的真实驱动（+ 各自趋势）⇒ 期望检出率高。"""
    hit = 0
    for _ in range(n_rep):
        dy = V1.ar1(n, 0.5, rng)
        y = np.cumsum(dy)
        u = np.cumsum(w * dy + V1.ar1(n, 0.5, rng))
        S_obs, thr, _ = diff_domain_test(y, u, B, rng, alpha)
        if np.isfinite(thr) and S_obs > thr:
            hit += 1
    return hit / n_rep


# ============================================================ 零假设 B
def spatial_permutation(rho_all, true_idx):
    """精确枚举 C(15,3)：统计量 = 选中 3 个配对的 mean(|ρ|)。"""
    rho = np.abs(np.asarray(rho_all, float))
    obs = float(rho[list(true_idx)].mean())
    idx = list(range(len(rho)))
    stats = np.array([rho[list(c)].mean() for c in combinations(idx, 3)])
    p = float((stats >= obs - 1e-12).mean())
    return obs, p, len(stats), stats


# ============================================================ 数据装配
def load_settlement(sname, ts):
    fp = SETTLE_DIR / f"{sname[1:]}.xlsx"
    if not fp.exists():
        print(f"[warn] 缺失 {fp}")
        return None
    sd = pd.read_excel(fp)
    sd["监测时间"] = pd.to_datetime(sd["监测时间"])
    sd = sd.sort_values("监测时间").reset_index(drop=True)
    dt_days = (sd["监测时间"].values[:, None] - ts.values[None, :]) \
        .astype("timedelta64[h]").astype(float) / 24.0
    idx = np.abs(dt_days).argmin(1)
    gap = np.abs(dt_days).min(1)
    keep = (gap <= TOL_DAYS) & sd["累计沉降量_mm"].notna().values
    if keep.sum() < 20:
        print(f"[warn] {sname} 对齐后样本 {keep.sum()} < 20，跳过")
        return None
    return idx[keep], sd["累计沉降量_mm"].values[keep].astype(float)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true", help="B=200 / N_REP=60 冒烟")
    ap.add_argument("--B", type=int, default=int(os.environ.get("B", "1000")))
    ap.add_argument("--n-rep", type=int, default=int(os.environ.get("N_REP", "300")))
    ap.add_argument("--alpha", type=float, default=float(os.environ.get("ALPHA", "0.95")))
    args = ap.parse_args()
    B = 200 if args.quick else args.B
    N_REP = 60 if args.quick else args.n_rep
    alpha = args.alpha

    cfg = AE.Config()
    cfg.RESUME = 0
    AE.set_seed(SEED)
    rng = np.random.default_rng(SEED)
    full_df = AE.load_raw_data_with_meta(cfg)
    disp_raw, _, meta = AE.build_node_features(full_df, cfg)
    ts = pd.to_datetime(meta["timestamps"])
    depths = np.asarray(meta["depths"], float)
    point_names = np.asarray(meta["point_names"])
    OUT.mkdir(parents=True, exist_ok=True)

    print("=" * 110)
    print(f"VICD 层 2 判据化  |  差分域保谱代理 + 空间错配置换  |  B={B} N_REP={N_REP} "
          f"alpha={alpha:.2f} seed={SEED}")
    print(f"位移场 T={disp_raw.shape[0]} 节点={disp_raw.shape[1]} 对齐容差={TOL_DAYS}天")
    print("=" * 110)

    # ---------- 装配：3 沉降点 × 5 断面 × 4 层 ----------
    lay_mask = {"全深度": np.ones(len(depths), bool), "浅(<3m)": depths < 3.0,
                "中(3-8m)": (depths >= 3.0) & (depths < 8.0), "深(8-16m)": depths >= 8.0}
    series = {}          # (sname, cx, layer) -> (y_cum, u_cum)
    for sname in TRUE_PAIRS:
        got = load_settlement(sname, ts)
        if got is None:
            continue
        t_map, u = got
        for cx in SECTIONS:
            pidx = np.where(point_names == cx)[0]
            for lname, lm in lay_mask.items():
                lsel = pidx[lm[pidx]]
                if len(lsel) == 0:
                    continue
                series[(sname, cx, lname)] = (disp_raw[t_map][:, lsel].mean(1), u)
    if not series:
        raise SystemExit("未装配到任何序列（检查沉降数据路径）")

    # ---------- A. 差分域保谱代理 ----------
    print(f"\n{'#' * 110}\n# A. 零假设 A：差分域保谱代理（S_obs > 阈 ⇒ 层 2 判'验证资格成立'）\n{'#' * 110}")
    rows = []
    for (sname, cx, lname), (y, u) in sorted(series.items()):
        S_obs, thr, S_null = diff_domain_test(y, u, B, rng, alpha)
        over = bool(np.isfinite(thr) and S_obs > thr)
        rows.append(dict(part="A", settle=sname, section=cx, layer=lname, n=len(y),
                         is_true_pair=int(TRUE_PAIRS.get(sname) == cx),
                         S_obs=S_obs, S_thr95=thr, layer2_A="越界" if over else "未越界"))
    dA = pd.DataFrame(rows)

    print(f"\n【3 个真配对（全深度）—— 判据主结论】")
    print(f"  {'配对':<14}{'n':>5}{'S_obs=|ρ|':>11}{'S阈95':>9}   层2A")
    for _, r in dA[(dA.is_true_pair == 1) & (dA.layer == "全深度")].iterrows():
        print(f"  {r['settle']+'↔'+r['section']:<14}{r['n']:>5}{r['S_obs']:>11.4f}"
              f"{r['S_thr95']:>9.4f}   {r['layer2_A']}")

    print(f"\n【全部 15 配对 × 各层 越界率（作为对照：越界应集中于真配对）】")
    piv = dA.assign(ov=(dA.layer2_A == "越界").astype(int)) \
        .groupby(["is_true_pair", "layer"], as_index=False)["ov"].mean()
    for _, r in piv.iterrows():
        tag = "真配对" if r["is_true_pair"] else "错配对"
        print(f"  [{tag}] {r['layer']:<10} 越界率 {r['ov']:.3f}")

    # ---------- B. 空间错配置换 ----------
    print(f"\n{'#' * 110}\n# B. 零假设 B：空间错配置换（精确枚举 C(15,3)=455）\n{'#' * 110}")
    perm_rows = []
    for lname in LAYERS:
        rho_all, true_idx = [], []
        for sname in TRUE_PAIRS:
            for j, cx in enumerate(SECTIONS):
                key = (sname, cx, lname)
                if key not in series:
                    continue
                y, u = series[key]
                rho_all.append(V1.safe_corr(y, u))
                if TRUE_PAIRS[sname] == cx:
                    true_idx.append(len(rho_all) - 1)
        if len(rho_all) != 15:
            print(f"  [{lname}] 配对数 {len(rho_all)}≠15，跳过")
            continue
        obs, p, n_perc, _ = spatial_permutation(rho_all, true_idx)
        perm_rows.append(dict(part="B", layer=lname, stat_true3_meanabsrho=obs,
                              n_perm=n_perc, p_exact=p,
                              layer2_B="越界" if p < 0.05 else "未越界",
                              n_true=len(true_idx)))
        print(f"  [{lname:<10}] 真配对 mean|ρ|={obs:.4f}  枚举 {n_perc} 个三元组  "
              f"p={p:.4f}  最小可能 p={1/n_perc:.4f}  → {perm_rows[-1]['layer2_B']}")
    dB = pd.DataFrame(perm_rows)

    # ---------- 标定 ----------
    print(f"\n{'#' * 110}\n# C. 标定：FPR（H0 为真）+ 判别力（备择为真），n=200, 重复 {N_REP}\n{'#' * 110}")
    calib_rows = []
    n_cal = int(np.median([len(v[0]) for v in series.values()]))
    for phi in (0.0, 0.5, 0.9):
        f = fpr_ar1(phi, n_cal, B, N_REP, rng, alpha)
        calib_rows.append(dict(kind="FPR", phi=phi, n=n_cal, value=f, n_rep=N_REP))
        print(f"  FPR   φ={phi:<4} n={n_cal}  → {f:.3f}   （期望 ≈ {1-alpha:.2f}）")
    for w in (0.3, 0.5, 1.0):
        pw = power_injected(n_cal, B, N_REP, rng, alpha, w=w)
        calib_rows.append(dict(kind="power", phi=np.nan, n=n_cal, value=pw, n_rep=N_REP,
                               inj_w=w))
        print(f"  判别力 注入 w={w:<4} n={n_cal}  → 检出率 {pw:.3f}   （期望接近 1）")
    dC = pd.DataFrame(calib_rows)

    # ---------- 判定 ----------
    print(f"\n{'=' * 110}\n# D. 层 2 判定\n{'=' * 110}")
    a_true = dA[(dA.is_true_pair == 1) & (dA.layer == "全深度")]
    nA = int((a_true.layer2_A == "越界").sum())
    nB = int((dB.layer2_B == "越界").sum()) if not dB.empty else 0
    verdict = ("**升为定量一致性验证**" if (nA == len(a_true) and len(a_true) > 0 and nB > 0)
               else "**维持'仅定性'**（至少一个零假设未越界）")
    print(f"  零假设 A：真配对越界 {nA}/{len(a_true)}")
    print(f"  零假设 B：越界层数 {nB}/{len(dB)}")
    print(f"  ⇒ 层 2 判定：{verdict}")

    dA.to_csv(OUT / "layer2_partA.csv", index=False, encoding="utf-8-sig")
    dB.to_csv(OUT / "layer2_partB.csv", index=False, encoding="utf-8-sig")
    dC.to_csv(OUT / "layer2_calibration.csv", index=False, encoding="utf-8-sig")
    print(f"\n已落盘：{OUT}\\layer2_{{partA,partB,calibration}}.csv")


if __name__ == '__main__':
    main()