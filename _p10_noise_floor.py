# -*- coding: utf-8 -*-
r"""_p10_noise_floor.py —— 判别增量目标是否由仪器噪声主导。

仪器规格（作者自提供文件核实）
------------------------------
测斜仪型号 **RDIS1030S**，分辨率 **0.01 mm / 500 mm**（即每 500 mm 测杆，
最小可分辨读数 0.01 mm；本场地测点深度间距恰为 0.5 m，一一对应）。
回程差 / 系统精度（如 ±2 mm/30 m 一类）在提供文件中未给出 ⇒ 只用分辨率，
不臆造其他规格。

本脚本做什么（零训练、只读）
----------------------------
1) 主口径 = **模型所用栅格**（`<DATA_ROOT>\ceshi14\MoXin\data\CX*.xlsx`，
   2 天栅格 213 时点）：逐节点原始增量（累计位移跨时点差分）与去噪增量
   （db4 单层软阈值，镜像锁定管线）的统计量；
2) **仪器噪声底**：分辨率量化噪声 σ_q = 0.01/√12（均匀量化）按深度累加——
   深度 z 处累计位移由 n = z/0.5 个测杆读数累加，增量噪声底
   σ_floor(z) = √2·σ_q·√n（两次独立读数之差）；并给出峰值界 σ_q=0.01 的保守档；
3) 逐节点 SNR = 原始增量 std / σ_floor(z)，按深度带（浅 <3 / 中 3–8 / 深 8–16 m）
   汇总：增量 std、中位 |Δ|、|Δ|≤0.01 mm（分辨率）的时间占比、SNR；
4) 辅助口径 = 原始非栅格数据（`date\raw\CX*.xlsx`）的"本次水平位移"列：
   实测分辨率步长（经验量化步长）与 |Δ|≤0.01 占比，佐证 0.01 mm 分辨率；
5) 判读：SNR<1 ⇒ 该带增量被仪器噪声主导（测量层面不可分辨）；1–2 弱可辨；
   >2 可辨。输出 `_paper_tables\T_p10_noise_floor.csv`（逐节点+逐带）+ 判读文字。

铁律
----
· 零训练、只读数据，不改任何锁定脚本；输出仅写 `_paper_tables\`。
· 栅格数据含 2 天插值平滑（会把增量噪声摊薄），故另给原始数据口径作保守对照；
  主口径结论若依赖栅格平滑，须声明。
· 补材编号预留 **S25**（S23=术语表、S24=学习曲线）。

运行（PowerShell，只读分析）
------------------------------------------------------
  python -u _p10_noise_floor.py
"""
from _env import DATA_ROOT, WORK_ROOT

import json
import numpy as np
import pandas as pd
import pywt
from pathlib import Path

PROJ = WORK_ROOT
TABLES = PROJ / "_paper_tables"
TABLES.mkdir(parents=True, exist_ok=True)

GRID_DIR = DATA_ROOT / "ceshi14/MoXin/data"     # 2 天栅格 213 时点（模型所用）
RAW_DIR = DATA_ROOT / "ceshi14/date/raw"        # 原始非规则时点（保守对照）
POINTS = ["CX2", "CX3", "CX8", "CX13", "CX15"]
DEPTHS = np.arange(0.5, 16.0, 0.5)
RES_MM = 0.01          # 分辨率 0.01 mm / 500 mm
GAUGE_M = 0.5          # 每测杆 500 mm
SIGMA_Q = RES_MM / np.sqrt(12.0)   # 均匀量化噪声 std
BANDS = [("Shallow", 0.5, 3.0), ("Middle", 3.0, 8.0), ("Deep", 8.0, 16.0)]


def wavelet_denoise(signal, wavelet="db4", level=1, thresh_mode="soft"):
    """与锁定管线 denoise_split 相同的逐序列 db4 单层软阈值去噪。"""
    signal = np.asarray(signal, dtype=np.float64)
    level = min(level, int(np.log2(len(signal))) - 1)
    if level < 1:
        return signal
    coeffs = pywt.wavedec(signal, wavelet, level=level)
    sigma = np.median(np.abs(coeffs[-1])) / 0.6745
    thr = sigma * np.sqrt(2 * np.log(len(signal)))
    coeffs[1:] = [pywt.threshold(c, thr, thresh_mode) for c in coeffs[1:]]
    rec = pywt.waverec(coeffs, wavelet)
    if len(rec) > len(signal):
        return rec[:len(signal)]
    return np.pad(rec, (0, len(signal) - len(rec)), "edge")


def load_grid(section):
    """2 天栅格：213 时点 × 31 深度的累计位移矩阵（按 测量时间 × 深度_m 透视）。"""
    df = pd.read_excel(GRID_DIR / f"{section}.xlsx")
    df["测量时间"] = pd.to_datetime(df["测量时间"])
    piv = df.pivot_table(index="测量时间", columns="深度_m",
                         values="累计水平位移_mm", aggfunc="first")
    piv = piv.reindex(columns=DEPTHS).sort_index()
    return piv.values.astype(np.float64), piv.index


def load_raw_increment(section):
    """原始非栅格数据：本次水平位移列（真实测量读数，含 0.01 mm 量化）。"""
    df = pd.read_excel(RAW_DIR / f"{section}.xlsx")
    cand = [c for c in df.columns if "本次" in str(c)]
    col = cand[0] if cand else next(
        c for c in df.columns if "水平位移" in str(c) and "累计" not in str(c))
    df["深度_m"] = pd.to_numeric(df["深度_m"], errors="coerce")
    df = df[df["深度_m"].isin(DEPTHS)].sort_values("深度_m")
    inc = (df.groupby("深度_m")[col]
           .apply(lambda s: pd.to_numeric(s, errors="coerce")
                  .astype(float).dropna().values))
    return inc


def incr_stats(inc):
    """逐节点增量统计：std、中位 |Δ|、|Δ|≤0.01 占比、|Δ|≤0.02 占比。"""
    a = np.abs(inc)
    return {"incr_std": float(np.std(inc, ddof=1)),
            "median_abs": float(np.median(a)),
            "frac_le_0p01": float(np.mean(a <= RES_MM)),
            "frac_le_0p02": float(np.mean(a <= 2 * RES_MM))}


def floor_std(depth):
    """深度 z 处累计位移增量噪声底（两次独立读数之差）：
    σ_floor(z) = √2 · σ_q · √(z/0.5)（量化档，σ_q=0.01/√12）；并给峰值界档。"""
    n = depth / GAUGE_M
    quant = np.sqrt(2.0) * SIGMA_Q * np.sqrt(n)
    peak = np.sqrt(2.0) * RES_MM * np.sqrt(n)
    return quant, peak


def band_rows(records, inc_key="incr_raw"):
    rows = []
    for name, lo, hi in BANDS:
        sub = [r for r in records if lo <= r["depth"] < hi]
        for stat in ("incr_std", "median_abs", "frac_le_0p01", "frac_le_0p02",
                     "snr_quant", "snr_peak"):
            rows.append({"band": name, "depth_lo": lo, "depth_hi": hi,
                         "stat": stat, "n_nodes": len(sub),
                         "value": float(np.mean([r[stat] for r in sub]))})
    return rows


def main():
    print("=" * 100)
    print("增量目标噪声底判别 | RDIS1030S 分辨率 0.01 mm/500 mm | 零训练只读")
    print("=" * 100)
    records = []

    for sec in POINTS:
        disp, ts = load_grid(sec)
        T, N = disp.shape
        inc_raw = np.diff(disp, axis=0)                       # (212, 31) 原始增量
        disp_den = np.column_stack([wavelet_denoise(disp[:, j]) for j in range(N)])
        inc_den = np.diff(disp_den, axis=0)                   # (212, 31) 去噪增量
        # 测试段 [181,213) 对应的增量行（栅格差分后第 181..211 行）
        test_slice = slice(180, 212)
        for j, d in enumerate(DEPTHS):
            fq, fp = floor_std(d)
            rec = {"section": sec, "depth": float(d), "floor_quant": fq,
                   "floor_peak": fp}
            for tag, arr in (("raw", inc_raw), ("den", inc_den)):
                s = incr_stats(arr[:, j])
                st = incr_stats(arr[test_slice, j])
                rec[f"incr_std_{tag}"] = s["incr_std"]
                rec[f"median_abs_{tag}"] = s["median_abs"]
                rec[f"frac_le_0p01_{tag}"] = s["frac_le_0p01"]
                rec[f"frac_le_0p02_{tag}"] = s["frac_le_0p02"]
                rec[f"incr_std_{tag}_test"] = st["incr_std"]
                rec[f"frac_le_0p01_{tag}_test"] = st["frac_le_0p01"]
            rec["snr_quant"] = rec["incr_std_raw"] / fq
            rec["snr_peak"] = rec["incr_std_raw"] / fp
            records.append(rec)

    df = pd.DataFrame(records)
    df.to_csv(TABLES / "T_p10_noise_floor.csv", index=False, encoding="utf-8-sig")

    # ---- 逐带汇总 ----
    print("\n逐带汇总（31 深度 × 5 断面 = 155 节点；栅格 2 天步长）")
    print(f"{'带':<10}{'深度':<10}{'原始σ':>10}{'去噪σ':>10}{'中位|Δ|':>10}"
          f"{'|Δ|≤0.01':>10}{'测试段|Δ|≤0.01':>14}{'σ_floor(q)':>11}{'SNR(q)':>9}")
    band_rows_out = []
    for name, lo, hi in BANDS:
        sub = df[(df.depth >= lo) & (df.depth < hi)]
        row = {"band": name, "depth": f"{lo}-{hi}",
               "raw_sigma": sub.incr_std_raw.mean(),
               "den_sigma": sub.incr_std_den.mean(),
               "med_abs": sub.median_abs_raw.mean(),
               "le01": sub.frac_le_0p01_raw.mean(),
               "le01_test": sub.frac_le_0p01_raw_test.mean(),
               "floor_q": sub.floor_quant.mean(),
               "snr_q": sub.snr_quant.mean()}
        print(f"{name:<10}{row['depth']:<10}{row['raw_sigma']:>10.4f}"
              f"{row['den_sigma']:>10.4f}{row['med_abs']:>10.4f}"
              f"{row['le01']:>10.2%}{row['le01_test']:>14.2%}"
              f"{row['floor_q']:>11.4f}{row['snr_q']:>9.2f}")
        band_rows_out.append(row)

    # ---- 原始数据分辨率佐证 ----
    print("\n辅助：原始非栅格数据'本次水平位移'量化佐证（date\\raw）")
    for sec in POINTS[:2]:
        inc = load_raw_increment(sec)
        vals = np.concatenate([np.asarray(v) for v in inc.values])
        uniq = np.unique(vals)
        step = np.min(np.diff(np.sort(uniq))) if len(uniq) > 1 else np.nan
        print(f"  {sec}: n={len(vals)} | 唯一值 {len(uniq)} | 最小量化步长 "
              f"{step:.3f} mm | |Δ|≤0.01 占比 {np.mean(np.abs(vals) <= 0.01):.2%}")

    # ---- 判读 ----
    print("\n" + "=" * 100)
    print("判读（SNR = 原始增量σ / 仪器增量噪声底σ；σ_floor(z)=√2·0.01/√12·√(z/0.5)）")
    print("=" * 100)
    for r in band_rows_out:
        snr = r["snr_q"]
        if snr < 1:
            verdict = "噪声主导（测量层面不可分辨）"
        elif snr < 2:
            verdict = "弱可辨（信号与噪声同阶）"
        else:
            verdict = "可辨（噪声不是瓶颈）"
        print(f"  {r['band']:<8} SNR={snr:.2f}  =>  {verdict}")
    print("\n说明：① 栅格数据含 2 天插值平滑，σ 与 |Δ|≤0.01 占比为摊薄后口径；"
          "原始数据口径见 T_p10 辅助段。② 回程差/系统精度未提供，只用分辨率。"
          "③ 本表只判增量目标量级，不直接等于模型预测误差。")
    print(f"\n[已写出] {TABLES / 'T_p10_noise_floor.csv'}（155 节点逐节点 + 逐带汇总）")


if __name__ == "__main__":
    main()
