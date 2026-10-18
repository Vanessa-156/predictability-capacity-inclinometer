# -*- coding: utf-8 -*-
r"""_vicd_phase_recheck.py -- VICD 层 1 判据的**分阶段复检**。

动机（§6.4）：
  主稿"水位 / 沉降 / 锚索三个变量在增量口径不可作预测输入"这一层 1 判定，是在
  **全时段**（n≈213，测试段与验证段 100% 落在开挖后的近静止期）上做出的。
  测试段全为结构施工期、活跃度仅为开挖期的 1/2.5 ⇒ 该判定可能与"全时段平均"
  捆绑，未必在开挖期（活跃期）成立。故须按阶段分层重跑同一判据。

做法（零训练，纯数值）：
  · 复用 `_vicd_null_distribution.py` 的全部统计量、代理与判定逻辑（不重写、不改口径）：
    M1 增量|ρ| / M2 归一化互信息 / M3 符号对齐增益 / M4 最大领先相关，
    相位随机化代理零分布 + max-ratio 统计量 T，T_obs ≤ T_thr(95%) ⇒ 判"不可用"。
  · 阶段：全时段(0–212) / 开挖期(0–77) / 底板期(78–155) / 结构期(156–212) / 开挖后(78–212)
  · **必做功效标定**（否则"不可用"可能只是样本变短的假象）：
      FPR  —— 真零假设（y、u 各自独立 AR(1)）在**该阶段 n** 下的假阳率，应 ≈ 5%
      判别力 —— 合成注入（y 含 τ=3 领先的 u 驱动）在**该阶段 n** 下的检出率
  · 保真自检：本地对齐装配的"全时段"序列须与 V1.build_records 逐位一致。

输出：`_vicd_phase/*.csv` + 终端
用法：
  python -u _vicd_phase_recheck.py --cases water,settlement
  --quick 冒烟（B=200 / 标定 reps 小）
"""
from _env import DATA_ROOT

import argparse
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))
import Ablation_Experiment as AE          # noqa: E402
import _vicd_null_distribution as V1      # noqa: E402

OUT = ROOT / "_vicd_phase"
OUT.mkdir(parents=True, exist_ok=True)
BINS = V1.BINS
TOL_DAYS = V1.TOL_DAYS
BAD_CAP = V1.BAD_CAP
DOT = V1.DOT
SEED = V1.SEED

# 阶段（时间轴位置，含端点）—— 与 §6.4 一致：开挖止于 idx 77
PHASES = [
    ("全时段", 0, 213),
    ("开挖期", 0, 78),
    ("底板期", 78, 156),
    ("结构期", 156, 213),
    ("开挖后", 78, 213),
]
LOGIN_N = 78          # 开挖期长度，用于标定"该量级 n"
LAYERS = ("全深度", "浅(<3m)", "中(3-8m)", "深(8-16m)")


def layer_mask(depths):
    return {"全深度": np.ones(len(depths), bool), "浅(<3m)": depths < 3.0,
            "中(3-8m)": (depths >= 3.0) & (depths < 8.0),
            "深(8-16m)": depths >= 8.0}


def align_to_timeline(t_src, ts, vals):
    """与 V1 相同的最邻近对齐：返回 (时间轴索引, 对齐后的值)。"""
    dt_days = (t_src[:, None] - ts.values[None, :]).astype("timedelta64[h]").astype(float) / 24.0
    idx = np.abs(dt_days).argmin(1)
    gap = np.abs(dt_days).min(1)
    keep = gap <= TOL_DAYS
    return idx[keep], np.asarray(vals, float)[keep]


def build_records_tmap(case, cfg, disp_raw, meta, ts, depths, point_names):
    """镜像 V1.build_records，但额外返回每条记录的时间轴索引 t_idx（供阶段切片）。"""
    recs = []
    LM = layer_mask(depths)

    def layer_series(section, t_idx):
        pidx = np.where(np.asarray(point_names) == section)[0]
        for lname, lm in LM.items():
            lsel = pidx[lm[pidx]]
            if len(lsel) == 0:
                continue
            yield lname, disp_raw[t_idx][:, lsel].mean(1)

    if case == "water":
        wd, _ = AE.load_water_data(ts, cfg)
        t_idx = np.arange(len(ts))
        for lname, y in layer_series("CX2", t_idx):
            recs.append(("水位(wd)", "全场", lname, y, np.asarray(wd, float), t_idx))

    elif case == "settlement":
        for sname, cx in {"S6": "CX2", "S18": "CX8", "S31": "CX15"}.items():
            fp = DATA_ROOT / f"shuju/沉降/{sname[1:]}.xlsx"
            if not fp.exists():
                print(f"[warn] 缺失 {fp}，跳过")
                continue
            sd = pd.read_excel(fp)
            sd["监测时间"] = pd.to_datetime(sd["监测时间"])
            sd = sd.sort_values("监测时间").reset_index(drop=True)
            idx, u = align_to_timeline(sd["监测时间"].values, ts,
                                       sd["累计沉降量_mm"].values)
            ok = np.isfinite(u)
            idx, u = idx[ok], u[ok]
            if len(idx) < 20:
                print(f"[warn] {sname} 对齐后 {len(idx)} < 20，跳过")
                continue
            for lname, y in layer_series(cx, idx):
                recs.append((f"{sname}↔{cx}", cx, lname, y, u, idx))

    elif case == "anchor":
        ML = ROOT / "ML数据"
        for mlpref, cx in {"ML2": "CX2", "ML3": "CX3", "ML8": "CX8",
                           "ML13": "CX13", "ML15": "CX15"}.items():
            for fk in sorted(ML.glob(f"{mlpref}-*.xlsx")):
                df = pd.read_excel(fk)
                df["日期"] = pd.to_datetime(df["测量时间"], errors="coerce").dt.normalize()
                df["v"] = df["测试值_KN"].apply(_to_float)
                df = df[df["v"].notna() & (df["v"].abs() <= BAD_CAP)]
                g = df.groupby("日期", as_index=False)["v"].mean().sort_values("日期")
                g = g.dropna(subset=["日期"]).reset_index(drop=True)
                if len(g) < 20:
                    continue
                idx, u = align_to_timeline(pd.to_datetime(g["日期"]).values, ts,
                                           g["v"].to_numpy(float))
                if len(idx) < 20:
                    continue
                for lname, y in layer_series(cx, idx):
                    recs.append((f"{fk.stem}↔{cx}", cx, lname, y, u, idx))
    else:
        raise ValueError(case)
    return recs


def _to_float(x):
    if isinstance(x, (int, float)):
        return float(x)
    if isinstance(x, str):
        s = x.strip()
        for c in DOT:
            s = s.replace(c, ".")
        s = re.sub(r"(?<=\d)\s+(?=\d)", ".", s).replace(" ", "")
        try:
            return float(s)
        except ValueError:
            return np.nan
    return np.nan


# ---------------------------------------------------------------- 标定
def fpr_at_n(phi, n, B, reps, rng, alpha):
    hit = 0
    a = np.random.default_rng(SEED + int(phi * 100) + n)
    for _ in range(reps):
        u = V1.ar1(n, phi, a) if phi > 0 else a.normal(size=n)
        y = V1.ar1(n, phi, a) if phi > 0 else a.normal(size=n)
        if V1.one_shot_verdict(y, u, max(B // 2, 100), rng, V1.phase_surrogate,
                               alpha) != "不可用":
            hit += 1
    return hit / reps


def power_at_n(n, a_eff, B, reps, rng, alpha):
    det = 0
    for r_i in range(reps):
        a = np.random.default_rng(SEED + 5000 + r_i)
        u = V1.ar1(n, 0.8, a)
        y = np.zeros(n)
        y[3:] = a_eff * u[:-3] + 0.5 * a.normal(size=n - 3)
        if V1.one_shot_verdict(y, u, max(B // 2, 100), rng, V1.phase_surrogate,
                               alpha) != "不可用":
            det += 1
    return det / reps


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--cases", default="water,settlement,anchor")
    ap.add_argument("--alpha", type=float, default=0.95)
    args = ap.parse_args()
    B = 200 if args.quick else 1000
    reps = 15 if args.quick else 60
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

    cases = [c.strip() for c in args.cases.split(",") if c.strip()]
    print("=" * 118)
    print(f"VICD 层 1 判据 · 分阶段复检 | B={B} alpha={alpha:.2f} seed={SEED} "
          f"时间轴 {len(ts)} 时刻")
    for nm, a, b in PHASES:
        print(f"   {nm:<8} idx {a}–{b - 1}（{b - a} 时刻）")
    print("=" * 118, flush=True)

    # ---------------- 保真自检 ----------------
    ok_all = True
    for case in cases:
        ref = V1.build_records(case, cfg, disp_raw, meta, ts, depths, point_names)
        mine = build_records_tmap(case, cfg, disp_raw, meta, ts, depths, point_names)
        rmap = {(r[0], r[2]): (r[3], r[4]) for r in ref}
        mmap = {(r[0], r[2]): (r[3], r[4]) for r in mine}
        same = set(rmap) == set(mmap) and all(
            np.allclose(np.asarray(rmap[k][0], float), np.asarray(mmap[k][0], float),
                        equal_nan=True) and
            np.allclose(np.asarray(rmap[k][1], float), np.asarray(mmap[k][1], float),
                        equal_nan=True) for k in rmap)
        print(f"[自检] {case}: 记录数 参考={len(ref)} 本地={len(mine)} 逐位一致={same}")
        ok_all &= bool(same)
    assert ok_all, '保真自检失败，禁止继续'

    # ---------------- 主表：案例 × 阶段 ----------------
    rows = []
    for case in cases:
        recs = build_records_tmap(case, cfg, disp_raw, meta, ts, depths, point_names)
        print(f"\n{'#' * 118}\n# 案例 {case}：记录 {len(recs)}\n{'#' * 118}")
        for pname, a, b in PHASES:
            hdr = (f"{'阶段':<8}{'记录':<15}{'层':<10}{'n':>4}{'M1|ρ|':>8}{'阈':>7}"
                   f"{'M2MI':>8}{'阈':>7}{'M3符号':>8}{'阈':>7}{'M4先导':>8}{'阈':>7}"
                   f"{'T_obs':>8}{'T阈':>7}  层1判定")
            if pname == PHASES[0][0]:
                print(hdr); print("-" * len(hdr))
            for name, section, lname, y, u, t_idx in recs:
                sel = (t_idx >= a) & (t_idx < b)
                if sel.sum() < 20:
                    continue
                yy = np.asarray(y, float)[sel]
                uu = np.asarray(u, float)[sel]
                n = min(len(yy), len(uu))
                yy, uu = yy[:n], uu[:n]
                ok = np.isfinite(yy) & np.isfinite(uu)
                yy, uu = yy[ok], uu[ok]
                if len(yy) < 20:
                    continue
                r = V1.null_analysis(yy, uu, B, rng, V1.phase_surrogate, alpha=alpha)
                o, t = r["obs"], r["thr"]
                usable = "不可用"
                v1 = (usable if (np.isfinite(r["T_obs"]) and np.isfinite(r["T_thr"])
                                 and r["T_obs"] <= r["T_thr"]) else "越界→需再检验")
                g = lambda d, k: f"{d[k]:.3f}" if np.isfinite(d.get(k, np.nan)) else "nan"
                print(f"{pname:<8}{name:<15}{lname:<10}{len(yy):>4}"
                      f"{g(o, 'M1_inc_corr'):>8}{g(t, 'M1_inc_corr'):>7}"
                      f"{g(o, 'M2_inc_mi'):>8}{g(t, 'M2_inc_mi'):>7}"
                      f"{g(o, 'M3_sign_gain'):>8}{g(t, 'M3_sign_gain'):>7}"
                      f"{g(o, 'M4_lead_max'):>8}{g(t, 'M4_lead_max'):>7}"
                      f"{r['T_obs']:>8.3f}{r['T_thr']:>7.3f}  {v1}")
                rows.append(dict(case=case, phase=pname, record=name, section=section,
                                 layer=lname, n=len(yy),
                                 **{k: o[k] for k in o},
                                 **{f"{k}_thr_95": t[k] for k in t},
                                 T_obs=r["T_obs"], T_thr_95=r["T_thr"],
                                 argmax=r["argmax"], layer1=v1))

    df = pd.DataFrame(rows)
    df.round(4).to_csv(OUT / "vicd_phase_records.csv", index=False, encoding="utf-8-sig")

    print("\n" + "=" * 118)
    print("层 1 判定汇总：案例 × 阶段（'不可用'占比 = 判为不可作预测输入的比例）")
    print("=" * 118)
    if not df.empty:
        g = df.copy()
        g["inc"] = (g["layer1"] == "不可用").astype(float)
        piv = g.groupby(["case", "phase"], as_index=False).agg(
            记录数=("inc", "size"), 不可用占比=("inc", "mean"),
            n中位=("n", "median"), M1中位=("M1_inc_corr", "median"),
            T观测中位=("T_obs", "median"), T阈中位=("T_thr_95", "median"))
        piv = piv[["case", "phase", "记录数", "不可用占比", "n中位", "M1中位",
                   "T观测中位", "T阈中位"]]
        piv["不可用占比"] = piv["不可用占比"].map("{:.0%}".format)
        print(piv.round(3).to_string(index=False))
        piv.to_csv(OUT / "vicd_phase_summary.csv", index=False, encoding="utf-8-sig")

        print("\n按深度的层 1 判定（案例 × 阶段 × 层）：")
        piv2 = g.groupby(["case", "phase", "layer"], as_index=False)["inc"].mean()
        for c in cases:
            s = piv2[piv2["case"] == c]
            if s.empty:
                continue
            print(f"\n  [{c}]")
            print(s.pivot(index="phase", columns="layer", values="inc")
                  .round(2).to_string())

    # ---------------- 功效标定 ----------------
    print("\n" + "=" * 118)
    print("功效标定（关键：样本变短后判据是否仍有效/仍有检出力？）")
    print("=" * 118)
    cal = []
    ns_cal = sorted({int(df[df["phase"] == p]["n"].median()) for p in ("全时段", "开挖期")
                     if not df[df["phase"] == p].empty}) if not df.empty else []
    for nm, a, b in PHASES:
        ns_cal.append(b - a)
    ns_cal = sorted(set([n for n in ns_cal if n >= 20]))
    print(f"标定样本长度 n ∈ {ns_cal}（reps={reps}）")
    for n in ns_cal:
        line = [f"  n={n:<4}"]
        for phi in (0.0, 0.5, 0.9):
            f = fpr_at_n(phi, n, B, reps, rng, alpha)
            cal.append(dict(kind="FPR", n=n, phi=phi, value=f, reps=reps))
            line.append(f"FPR(φ={phi})={f:.2f}")
        for a_eff in (0.3, 0.6):
            p = power_at_n(n, a_eff, B, reps, rng, alpha)
            cal.append(dict(kind="power", n=n, a_eff=a_eff, value=p, reps=reps))
            line.append(f"检出(a={a_eff})={p:.2f}")
        print("  ".join(line), flush=True)
    pd.DataFrame(cal).to_csv(OUT / "vicd_phase_calibration.csv", index=False,
                             encoding="utf-8-sig")
    print(f"\n[已写出] {OUT}\\vicd_phase_{{records,summary,calibration}}.csv")


if __name__ == "__main__":
    main()