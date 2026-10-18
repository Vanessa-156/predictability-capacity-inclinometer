# -*- coding: utf-8 -*-
r"""_p16_vicd_denoise_check.py —— VICD 判据的序列口径声明 + 去噪敏感性对照。

代码核查（本脚本第 1 步打印，可复核）
----------------------------------
主场地三案例（水位 / 沉降 / 锚索）的 $y,u$ 均由
`_vicd_null_distribution.build_records` 装配：
  · $y$（位移分层均值）来自 `AE.build_node_features` —— **原始**，不含 wavelet_denoise；
  · $u$（水位 / 沉降 / 锚索）同样取**原始**观测列。
⇒ **VICD 判据在原始序列上计算**，"去噪致假不可用"这一机制不成立。
但"判据口径 = 原始、载体口径 = db4 去噪"这一差异仍须量化，本脚本即给出对照。

做法（零训练、只读）
------------------
同一批记录上跑两臂，判据协议完全不变（复用 `V1.null_analysis` + 相位随机化代理 +
max-ratio + held-out 两半、$B=1000$、$\alpha=0.95$）：
  臂 A（raw）：$y$ = 原始位移分层均值；$u$ = 原始变量序列
  臂 B（den）：$y$ = **逐节点** db4 去噪后的分层均值；$u$ = db4 去噪后的变量序列
同一记录、两臂使用**同一随机种子**（配对），比较判定与四项指标。

判读
----
· 判定一致率高 ⇒ 口径差异不影响 VICD 结论，只需在 §3.1 声明口径；
· 出现"raw 不可用 / den 越界"或反向的翻转 ⇒ 必须在正文披露翻转方向与占比。

铁律
----
· 零训练、只读；**不覆盖旧结果**，输出写新建目录 `_vicd_denoise_check/`。
· 不改编据协议：阈值逻辑、代理构造、统计量一律复用 `_vicd_null_distribution`。

运行（PowerShell）
-----------------
  python -u _p16_vicd_denoise_check.py
"""
from __future__ import annotations

from _env import WORK_ROOT

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = WORK_ROOT
sys.path.insert(0, str(ROOT))

import Ablation_Experiment as AE            # noqa: E402
import _vicd_null_distribution as V1        # noqa: E402

OUT = ROOT / "_vicd_denoise_check"
OUT.mkdir(parents=True, exist_ok=True)

B = 1000
ALPHA = 0.95
CASES = ("water", "settlement", "anchor")
CASE_LABEL = {"water": "水位", "settlement": "沉降", "anchor": "锚索力"}
KEYS = V1.INCREMENT_KEYS + ("MLEV_corr",)


def verdict_of(r: dict) -> str:
    """T_obs > T_thr ⇒ 越界（判定为可作预测输入）；否则不可用。"""
    if not (np.isfinite(r["T_obs"]) and np.isfinite(r["T_thr"])):
        return "nan"
    return "越界" if r["T_obs"] > r["T_thr"] else "不可用"


def denoise_disp(disp_raw, cfg):
    """对 (T,N) 位移场**逐节点**做 db4 去噪（与载体 `AB.denoise_split` 同一函数）。"""
    out = np.zeros_like(disp_raw)
    for j in range(disp_raw.shape[1]):
        out[:, j] = AE.wavelet_denoise(disp_raw[:, j], cfg)
    return out


def main():
    cfg = AE.Config()
    cfg.RESUME = 0
    AE.set_seed(V1.SEED)

    print("=" * 100)
    print("第 1 步：代码核查 —— VICD 用的是原始序列还是去噪序列？")
    print("=" * 100)
    full_df = AE.load_raw_data_with_meta(cfg)
    disp_raw, _, meta = AE.build_node_features(full_df, cfg)
    ts = pd.to_datetime(meta["timestamps"])
    depths = np.asarray(meta["depths"], float)
    point_names = np.asarray(meta["point_names"])
    print("  y（位移分层均值）：AE.build_node_features 的 pivot 值 = 原始实测（未经去噪）")
    print("  u（水位/沉降/锚索）：build_records 内直接读原始列（未经去噪）")
    print("  => VICD 判据口径 = 原始序列；载体（§2.3 预处理）口径 = db4 单层去噪")
    print(f"  B={B}, alpha={ALPHA}, 代理=phase_surrogate, 统计量=max-ratio(4 增量口径)")

    print("\n" + "=" * 100)
    print("第 2 步：逐节点 db4 去噪（构造臂 B 的输入）")
    print("=" * 100)
    disp_den = denoise_disp(disp_raw, cfg)
    rms = float(np.sqrt(np.mean((disp_den - disp_raw) ** 2)))
    print(f"  去噪前后位移场残差 RMS = {rms:.6f} mm（同一函数 AB.wavelet_denoise，"
          f"{cfg.WAVELET}/level={cfg.WAVELET_LEVEL}/{cfg.THRESH_MODE}）")

    rows = []
    for case in CASES:
        recs_raw = V1.build_records(case, cfg, disp_raw, meta, ts, depths, point_names)
        recs_den = V1.build_records(case, cfg, disp_den, meta, ts, depths, point_names)
        print("\n" + "=" * 100)
        print(f"第 3 步：{CASE_LABEL[case]}（y=位移） · 记录数 = {len(recs_raw)} vs {len(recs_den)}")
        print("=" * 100)
        hdr = (f"{'记录':<16}{'层':<10}{'n':>4} | {'T_obs(raw)':>10}{'T_thr':>8}"
               f"{'判定':>7} | {'T_obs(den)':>10}{'T_thr':>8}{'判定':>7} | {'ΔM1(增量ρ)':>10}"
               f"{'ΔM2(MI)':>9}{'ΔM3(符号)':>10} | 一致")
        print(hdr)
        print("-" * len(hdr))

        if len(recs_raw) != len(recs_den):
            print("  [warn] 两臂记录数不一致，跳过该案例")
            continue

        for (name, section, lname, y_r, u_r), (_, _, _, y_d, u_d) in zip(recs_raw, recs_den):
            # 臂 B 的 u 也去噪（u 为单序列）
            u_d = AE.wavelet_denoise(np.asarray(u_d, float), cfg)
            y_r, u_r = np.asarray(y_r, float), np.asarray(u_r, float)
            y_d = np.asarray(y_d, float)
            n = min(len(y_r), len(u_r))
            if n < 20:
                continue
            yy_r, uu_r = y_r[:n], u_r[:n]
            ok = np.isfinite(yy_r) & np.isfinite(uu_r)
            yy_r, uu_r = yy_r[ok], uu_r[ok]
            yy_d, uu_d = y_d[:len(yy_r)], u_d[:len(yy_r)]
            if len(yy_r) < 20 or not np.all(np.isfinite(yy_d)) \
                    or not np.all(np.isfinite(uu_d)):
                continue

            r_r = V1.null_analysis(yy_r, uu_r, B, np.random.default_rng(V1.SEED),
                                   V1.phase_surrogate, alpha=ALPHA)
            r_d = V1.null_analysis(yy_d, uu_d, B, np.random.default_rng(V1.SEED),
                                   V1.phase_surrogate, alpha=ALPHA)
            v_r, v_d = verdict_of(r_r), verdict_of(r_d)
            same = (v_r == v_d)
            d_m1 = r_d["obs"]["M1_inc_corr"] - r_r["obs"]["M1_inc_corr"]
            d_m2 = r_d["obs"]["M2_inc_mi"] - r_r["obs"]["M2_inc_mi"]
            d_m3 = r_d["obs"]["M3_sign_gain"] - r_r["obs"]["M3_sign_gain"]
            print(f"{name:<16}{lname:<10}{len(yy_r):>4} | "
                  f"{r_r['T_obs']:>10.3f}{r_r['T_thr']:>8.3f}{v_r:>7} | "
                  f"{r_d['T_obs']:>10.3f}{r_d['T_thr']:>8.3f}{v_d:>7} | "
                  f"{d_m1:>+10.3f}{d_m2:>+9.3f}{d_m3:>+10.3f} | {'是' if same else '否←翻转'}")
            rows.append(dict(case=case, case_label=CASE_LABEL[case], record=name,
                             section=section, layer=lname, n=len(yy_r),
                             T_obs_raw=r_r["T_obs"], T_thr_raw=r_r["T_thr"],
                             verdict_raw=v_r, ver_raw_usable=int(v_r == "越界"),
                             T_obs_den=r_d["T_obs"], T_thr_den=r_d["T_thr"],
                             verdict_den=v_d, ver_den_usable=int(v_d == "越界"),
                             flip=int(not same), agree=int(same),
                             dM1=d_m1, dM2=d_m2, dM3=d_m3,
                             dM4=r_d["obs"]["M4_lead_max"] - r_r["obs"]["M4_lead_max"],
                             dMLEV=r_d["obs"]["MLEV_corr"] - r_r["obs"]["MLEV_corr"]))

    df = pd.DataFrame(rows)
    if df.empty:
        raise SystemExit("无有效记录，停止（请检查数据路径）。")
    csv = OUT / "vicd_denoise_compare.csv"
    df.round(6).to_csv(csv, index=False, encoding="utf-8-sig")

    print("\n" + "=" * 110)
    print("第 4 步：汇总（判定一致性 = 两臂判定相同的记录占比）")
    print("=" * 110)
    hdr = (f"{'案例':<10}{'记录':>6}{'raw 越界':>10}{'den 越界':>10}{'一致':>8}"
           f"{'一致率':>9} | {'ΔM1中位':>10}{'ΔM2中位':>10}{'ΔM3中位':>10}{'ΔM4中位':>10}")
    print(hdr)
    print("-" * len(hdr))
    for case in CASES:
        g = df[df.case == case]
        if g.empty:
            continue
        print(f"{CASE_LABEL[case]:<10}{len(g):>6}{int(g.ver_raw_usable.sum()):>10}"
              f"{int(g.ver_den_usable.sum()):>10}{int(g.agree.sum()):>8}"
              f"{g.agree.mean():>9.0%} | "
              f"{g.dM1.median():>+10.3f}{g.dM2.median():>+10.3f}"
              f"{g.dM3.median():>+10.3f}{g.dM4.median():>+10.3f}")
    g = df
    print("-" * len(hdr))
    print(f"{'合计':<10}{len(g):>6}{int(g.ver_raw_usable.sum()):>10}"
          f"{int(g.ver_den_usable.sum()):>10}{int(g.agree.sum()):>8}{g.agree.mean():>9.0%}")
    nflip = int(g.flip.sum())
    print(f"\n判定翻转记录数 = {nflip}/{len(g)}")
    if nflip:
        f = g[g.flip == 1]
        print("翻转明细（须在正文/补充材料披露）：")
        for _, r in f.iterrows():
            print(f"  [{r.case_label}] {r.record} / {r.layer}: "
                  f"raw={r.verdict_raw}(T={r.T_obs_raw:.3f}) → "
                  f"den={r.verdict_den}(T={r.T_obs_den:.3f})")
    else:
        print("  ⇒ 去噪前后判定完全一致：口径差异不改变 VICD 结论，"
              "只需在 §3.1 声明判据口径为原始序列。")
    print(f"\n[已写出] {csv}")


if __name__ == "__main__":
    main()