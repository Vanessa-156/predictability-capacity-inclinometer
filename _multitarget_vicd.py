# -*- coding: utf-8 -*-
"""_multitarget_vicd.py -- 方案1：VICD 换目标重跑（零训练，只读）

预注册协议见 _multitarget_protocol.md §7（本次为 §7 之前的方案1修订）。
原 VICD（§3.4 / _vicd_null_distribution.py）问"候选输入 u 对位移目标 y 是否携带可利用信息"，
三案例 u∈{水位,沉降,锚索}、y=位移。本脚本**掉转目标**：y∈{沉降,锚索}，u∈{位移(分层均值),水位}，
问"位移/水位 对 沉降/锚索增量 是否携带可利用信息"——检验变量准入判据是否不依赖目标选择。

口径：完全镜像 V1.build_records 的装配（对齐容差/坏点剔除/层定义），仅 y、u 对调；
    判据复用 V1.null_analysis + phase_surrogate + max-ratio（层1），不重写、不改阈值逻辑。

用法：python -u _multitarget_vicd.py [--quick]
"""
from _env import DATA_ROOT, WORK_ROOT

import argparse
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = WORK_ROOT
sys.path.insert(0, str(ROOT))
import Ablation_Experiment as AE          # noqa: E402
import _vicd_null_distribution as V1      # noqa: E402

OUT = ROOT / "_vicd_swapped"
OUT.mkdir(parents=True, exist_ok=True)

DOT = V1.DOT
TOL_DAYS = V1.TOL_DAYS
BAD_CAP = V1.BAD_CAP
SEED = V1.SEED
INCREMENT_KEYS = V1.INCREMENT_KEYS


def _to_float(x):
    if isinstance(x, (int, float)):
        return float(x)
    if isinstance(x, str):
        s = x.strip()
        for c in DOT:
            s = s.replace(c, '.')
        s = re.sub(r'(?<=\d)\s+(?=\d)', '.', s).replace(' ', '')
        try:
            return float(s)
        except ValueError:
            return np.nan
    return np.nan


def _align(t_src, ts):
    dt = (t_src[:, None] - ts.values[None, :]).astype('timedelta64[h]').astype(float) / 24.0
    idx = np.abs(dt).argmin(1)
    gap = np.abs(dt).min(1)
    return idx, gap <= TOL_DAYS


def disp_layer_series(disp_raw, point_names, depths, section, t_map):
    """位移分层均值序列（与 V1.build_records 的 layer_series 逐位一致）。"""
    pidx = np.where(np.asarray(point_names) == section)[0]
    for lname, lmask in {
            "全深度": np.ones(len(depths), bool),
            "浅(<3m)": depths < 3.0,
            "中(3-8m)": (depths >= 3.0) & (depths < 8.0),
            "深(8-16m)": depths >= 8.0}.items():
        lsel = pidx[lmask[pidx]]
        if len(lsel) == 0:
            continue
        yield lname, disp_raw[t_map][:, lsel].mean(1)


def build_swapped(cfg, disp_raw, meta, ts, depths, point_names, target_case, input_kind):
    """返回 [(name, section, layer, y=目标序列, u=输入序列), ...]。"""
    recs = []
    u_water = None
    if input_kind == "water":
        wd, _ = AE.load_water_data(ts, cfg)
        u_water = np.asarray(wd, float)

    if target_case == "settlement":
        for sname, cx in {"S6": "CX2", "S18": "CX8", "S31": "CX15"}.items():
            sd = pd.read_excel(DATA_ROOT / "shuju/沉降" / f"{sname[1:]}.xlsx")
            sd["监测时间"] = pd.to_datetime(sd["监测时间"])
            sd = sd.sort_values("监测时间").reset_index(drop=True)
            idx, gkeep = _align(sd["监测时间"].values, ts)
            keep = gkeep & np.isfinite(sd["累计沉降量_mm"].values.astype(float))
            t_map = idx[keep]
            if len(t_map) < 20:
                continue
            y = sd["累计沉降量_mm"].values[keep].astype(float)
            if input_kind == "water":
                recs.append((f"{sname}↔{cx}", cx, "全场", y, u_water))
            else:
                for lname, u in disp_layer_series(disp_raw, point_names, depths, cx, t_map):
                    recs.append((f"{sname}↔{cx}", cx, lname, y, u))

    elif target_case == "anchor":
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
                idx, gkeep = _align(pd.to_datetime(g["日期"]).values, ts)
                t_map = idx[gkeep]
                if len(t_map) < 20:
                    continue
                y = g["v"].to_numpy(float)[gkeep]
                if input_kind == "water":
                    recs.append((f"{fk.stem}↔{cx}", cx, "全场", y, u_water))
                else:
                    for lname, u in disp_layer_series(disp_raw, point_names, depths, cx, t_map):
                        recs.append((f"{fk.stem}↔{cx}", cx, lname, y, u))
    else:
        raise ValueError(target_case)
    return recs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args()
    B = 200 if args.quick else 1000
    alpha = 0.95

    cfg = AE.Config()
    cfg.RESUME = 0
    AE.set_seed(SEED)
    rng = np.random.default_rng(SEED)
    full_df = AE.load_raw_data_with_meta(cfg)
    disp_raw, _, meta = AE.build_node_features(full_df, cfg)
    ts = pd.to_datetime(meta["timestamps"])
    depths = np.asarray(meta["depths"], float)
    point_names = np.asarray(meta["point_names"])

    rows = []
    for target_case, input_kind in [("settlement", "disp"), ("settlement", "water"),
                                    ("anchor", "disp"), ("anchor", "water")]:
        recs = build_swapped(cfg, disp_raw, meta, ts, depths, point_names,
                             target_case, input_kind)
        tag = f"y={target_case}, u={input_kind}"
        print(f"\n{'=' * 110}\n{tag}  记录数={len(recs)}\n{'=' * 110}")
        hdr = (f"{'记录':<16}{'层':<10}{'n':>4} | {'M1增量ρ':>8}{'阈':>7} | "
               f"{'M2增量MI':>9}{'阈':>7} | {'M3符号':>8}{'阈':>7} | {'M4先导':>8}{'阈':>7} | "
               f"{'T_obs':>7}{'T阈':>7} | 层1判定")
        print(hdr); print("-" * len(hdr))
        for name, section, lname, y_cum, u_cum in recs:
            y = np.asarray(y_cum, float); u = np.asarray(u_cum, float)
            n = min(len(y), len(u)); y, u = y[:n], u[:n]
            ok = np.isfinite(y) & np.isfinite(u); y, u = y[ok], u[ok]
            if len(y) < 20:
                continue
            r = V1.null_analysis(y, u, B, rng, V1.phase_surrogate, alpha=alpha)
            o, t = r["obs"], r["thr"]
            v1 = ("不可用" if (np.isfinite(r["T_obs"]) and np.isfinite(r["T_thr"])
                              and r["T_obs"] <= r["T_thr"]) else "越界→需再检验")
            g = lambda d, k: f"{d[k]:.3f}" if np.isfinite(d.get(k, np.nan)) else "nan"
            print(f"{name:<16}{lname:<10}{len(y):>4} | "
                  f"{g(o,'M1_inc_corr'):>8}{g(t,'M1_inc_corr'):>7} | "
                  f"{g(o,'M2_inc_mi'):>9}{g(t,'M2_inc_mi'):>7} | "
                  f"{g(o,'M3_sign_gain'):>8}{g(t,'M3_sign_gain'):>7} | "
                  f"{g(o,'M4_lead_max'):>8}{g(t,'M4_lead_max'):>7} | "
                  f"{r['T_obs']:>7.3f}{r['T_thr']:>7.3f} | {v1}")
            rows.append(dict(target=target_case, input=input_kind, record=name,
                             section=section, layer=lname, n=len(y),
                             **{k: o[k] for k in o}, **{f"{k}_thr_95": t[k] for k in t},
                             T_obs=r["T_obs"], T_thr_95=r["T_thr"], argmax=r["argmax"],
                             layer1=v1))

    df = pd.DataFrame(rows)
    df.round(4).to_csv(OUT / "vicd_swapped_records.csv", index=False, encoding="utf-8-sig")

    print("\n" + "=" * 110)
    print("层1 汇总（'不可用'占比 = 判为不可作预测输入的比例）· 按 目标×输入×层")
    print("=" * 110)
    if not df.empty:
        g = df.copy()
        g["inc"] = (g["layer1"] == "不可用").astype(float)
        piv = g.groupby(["target", "input", "layer"], as_index=False).agg(
            记录数=("inc", "size"), 不可用占比=("inc", "mean"), n中位=("n", "median"),
            M1中位=("M1_inc_corr", "median"), T中位=("T_obs", "median"))
        piv["不可用占比"] = piv["不可用占比"].map("{:.0%}".format)
        print(piv.round(3).to_string(index=False))
        piv.to_csv(OUT / "vicd_swapped_summary.csv", index=False, encoding="utf-8-sig")

    print(f"\n[已写出] {OUT}\\vicd_swapped_{{records,summary}}.csv")


if __name__ == "__main__":
    main()