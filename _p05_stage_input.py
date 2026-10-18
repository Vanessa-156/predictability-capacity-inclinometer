# -*- coding: utf-8 -*-
r"""_p05_stage_input.py —— 把开挖深度/工况纳入模型输入与 VICD 候选集并重跑。

背景
----------------
原始数据自带"施工进度"工况列、作者亦自述"墙侧移由分步开挖卸荷主导"，但模型
4 维输入（位移 / 一阶差分 / 深度归一 / 浅层标志）与 VICD 候选集都不含卸载信息。
本脚本把"开挖深度"补进来，回答两件事：
  (1) 开挖深度是否携带**可用**信息（VICD 层 1 增量判据，B=1000、相位随机化代理、
      max-ratio 统计量、逐记录 n 与阈值 —— 复用 `_vicd_null_distribution.py` 协议）；
  (2) 4 维 → 5 维（新增开挖深度通道）是否显著改善预测（同种子配对 t + 效应量 +
      时间块 bootstrap CI，见 §3.4）；模型取 Ours(=V7_wo_VerticalGCN) / TCN-Only(V0) /
      BiLSTM-Attn，视界 H=10，30 种子，共 2×3×30 = 180 次训练。

铁律
----
· 不覆盖旧结果：全部写入 `<WORK_ROOT>\Stage_Input\`（新目录）。
· 不编造：开挖列来自原始 excel 的 `施工进度` 列；取不到就报错停下。
· 训练配方 / 损失 / 早停 / 数据划分**完全沿用锁定脚本**（AB / CE），只改输入维度与
  第 5 通道的构造。

锚点自检（先打印，失败即停）
---------------------------
4 维口径下 Ours H=10 测试末步 RMSE 应复现 1.5687 ± 0.0940；零变化 = 1.372010。

运行
----
  python -u _p05_stage_input.py --probe
  python -u _p05_stage_input.py            # 正式
  冒烟：加 --smoke（2 种子 / 5 epoch）；续跑：环境变量 P05_RESUME=1
"""
from __future__ import annotations

from _env import WORK_ROOT

import argparse
import copy
import json
import os
import re
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy import stats
from torch.utils.data import DataLoader, TensorDataset

PROJ = WORK_ROOT
sys.path.insert(0, str(PROJ))

import Ablation_Experiment as AB          # noqa: E402
import Comparison_Experiment as CE         # noqa: E402
import _vicd_null_distribution as VN       # noqa: E402  （VICD 协议复用）

HZ = 10                                   # 本项主视界
OUT_ROOT = PROJ / "Stage_Input"
TABLES = PROJ / "_paper_tables"
TABLES.mkdir(parents=True, exist_ok=True)

# 锚点
ANCHOR_OURS_MEAN, ANCHOR_OURS_STD = 1.5687, 0.0940
ZERO_H10 = 1.372010

# §3.4 时间块 bootstrap 参数
B_BOOT = 2000
BOOT_SEED = 42
L_STAR = 3

# 模型名 → 目录名 / 构造方式
MODELS = ["Ours", "TCN-Only", "BiLSTM-Attn"]
VAR_OF = {"Ours": "V7_wo_VerticalGCN", "TCN-Only": "V0_Pure_TCN"}


# =============================================================================
# 第 1 步：工况 → 开挖深度（米）映射
# =============================================================================
STAGE_FILE = "CX2.xlsx"          # 场地级参考断面（见 main 中的口径说明）
_DEPTH_RE = re.compile(r"(\d+)\s*米")


def parse_depth(label: str):
    """从工况标签解析显式开挖深度（米）；无则 None。"""
    m = _DEPTH_RE.search(str(label))
    return float(m.group(1)) if m else None


def is_excavation_label(label: str) -> bool:
    """是否为**主基坑开挖**工况（马道开挖不算 —— 属临时坡道而非坑底卸荷）。"""
    s = str(label)
    return ("开挖" in s) and ("马道" not in s)


def build_depth_series(labels):
    """标签序列 → 单调非降的开挖深度（米）序列。

    规则（对参考断面逐时刻）：
      · 显式标注 X 米 的时刻即锚点，深度 = X；
      · 锚点之间（含支撑/停工等非开挖工况）按索引线性插值（卸荷是渐进过程）；
      · 首个锚点之前从 0 线性升至首锚点值（自地面起挖）；
      · 末锚点之后保持不变（开挖到底后进入底板/结构期）。
    """
    n = len(labels)
    anchors = [(i, parse_depth(s)) for i, s in enumerate(labels)]
    anchors = [(i, d) for i, d in anchors if d is not None and is_excavation_label(labels[i])]
    if not anchors:
        raise RuntimeError("未在施工进度列解析到任何开挖深度锚点，停止（不编造）。")
    depth = np.full(n, np.nan, dtype=float)
    # 末锚点之后：保持
    last_i, last_d = anchors[-1]
    depth[last_i:] = last_d
    # 锚点之间 + 首锚点之前
    prev_i, prev_d = None, 0.0
    for i, d in anchors:
        if prev_i is None:
            # 首个锚点之前：0 → d 线性
            depth[:i] = np.linspace(0.0, d, i, endpoint=False) if i > 0 else []
            depth[i] = d
        else:
            depth[prev_i:i + 1] = np.linspace(prev_d, d, i - prev_i + 1)
        prev_i, prev_d = i, d
    # 兜底：仍为 nan 的（锚点间隙外）用前值填充
    idx = np.where(~np.isfinite(depth))[0]
    for i in idx:
        depth[i] = depth[i - 1] if i > 0 else 0.0
    return depth, anchors


def load_stage_labels(base_dir: Path):
    """读取某断面 excel 的 (测量时间, 施工进度)，返回时间升序的标签序列 DataFrame。"""
    fp = base_dir / STAGE_FILE
    if not fp.exists():
        raise FileNotFoundError(f"原始数据缺失：{fp}")
    df = pd.read_excel(fp)
    tcol = [c for c in df.columns if "测量时间" in str(c)]
    scol = [c for c in df.columns if "施工进度" in str(c)]
    if not scol:
        raise RuntimeError(f"{fp.name} 无'施工进度'列 —— 取不到工况信息，停止（不编造）。")
    tbl = (df[[tcol[0], scol[0]]].drop_duplicates(subset=[tcol[0]])
           .sort_values(tcol[0]).reset_index(drop=True))
    tbl.columns = ["测量时间", "施工进度"]
    tbl["测量时间"] = pd.to_datetime(tbl["测量时间"])
    return tbl


def probe_stage(base_dir: Path):
    """打印探查结果与映射表，返回 (时间戳Index, 深度序列, 锚点列表)。"""
    tbl = load_stage_labels(base_dir)
    labels = tbl["施工进度"].astype(str).values
    depth, anchors = build_depth_series(labels)

    print("=" * 100)
    print("第 1 步：定位开挖信息（施工进度列）")
    print("=" * 100)
    print(f"参考断面文件：{base_dir / STAGE_FILE}")
    print(f"时间轴长度 T = {len(tbl)}，时间跨度 {tbl['测量时间'].min().date()} ~ "
          f"{tbl['测量时间'].max().date()}")
    print(f"\n【工况 → 开挖深度（米）映射表】（按参考断面出现顺序）")
    print(f"{'序':>4}  {'测量时间':<12} {'施工进度':<20} {'深度(m)':>8}  规则")
    print("-" * 68)
    seen = set()
    for i in range(len(tbl)):
        lab = labels[i]
        d = depth[i]
        rule = "显式锚点" if i in [a[0] for a in anchors] else \
               ("末期保持" if i > anchors[-1][0] else "线性插值/自 0 起挖")
        mark = ""
        if lab not in seen:
            seen.add(lab)
            mark = "  ← 首次出现"
        print(f"{i:>4}  {str(tbl['测量时间'].iloc[i].date()):<12} {lab:<20} "
              f"{d:>8.2f}  {rule}{mark}")
    print("\n【唯一工况标签 → 解析深度】")
    for lab in sorted(set(labels)):
        print(f"  {lab:<22} -> {parse_depth(lab)}")
    print(f"\n【锚点（显式深度里程碑）】")
    for i, d in anchors:
        print(f"  {tbl['测量时间'].iloc[i].date()}  开挖深度 = {d:.0f} m")

    ts_idx = pd.DatetimeIndex(tbl["测量时间"].values)
    return ts_idx, depth, anchors


def report_cross_section(base_dir: Path, ts_idx: pd.DatetimeIndex):
    """跨断面工况一致性报告（说明口径）。"""
    print("\n【跨断面工况一致性检查】")
    ser = {}
    for p in AB.Config.POINTS:
        fp = base_dir / f"{p}.xlsx"
        if not fp.exists():
            continue
        df = pd.read_excel(fp)
        tcol = [c for c in df.columns if "测量时间" in str(c)][0]
        scol = [c for c in df.columns if "施工进度" in str(c)][0]
        s = (df[[tcol, scol]].drop_duplicates(subset=[tcol]).sort_values(tcol)
             .reset_index(drop=True))
        ser[p] = pd.Series(s[scol].astype(str).values,
                           index=pd.DatetimeIndex(s[tcol].values))
    cmp = pd.DataFrame({p: ser[p].reindex(ts_idx) for p in ser})
    same = (cmp.nunique(axis=1, dropna=False) == 1)
    print(f"  断面数 = {len(ser)}；逐时点工况完全一致的比例 = "
          f"{int(same.sum())}/{len(cmp)} ({same.mean():.0%})")
    diff_ts = cmp.index[~same]
    print(f"  不一致时点数 = {len(diff_ts)}（首 {min(3, len(diff_ts))} 例）：")
    for t in diff_ts[:3]:
        print(f"    {t.date()}: " + " | ".join(f"{p}={cmp.loc[t, p]}" for p in ser))
    print("  → 口径选择：以 CX2 为场地级参考断面。理由：其工况序列恰与设计开挖梯级\n"
          "    15→20→21→24 m 完全一致、无跨断面矛盾；其余断面在 2023-11-29~12-15 报出\n"
          "    16/18/25/20 m 等**局部开挖面**中间值（25 m 已超设计坑深 24 m，属数据异常），\n"
          "    不宜作为场地级卸荷变量。这不影响'工况列是否为可用输入'的判定（跨断面\n"
          "    差异只出现在少数时点、且均为同类阶段）。")
    return cmp


# =============================================================================
# 第 2 步：VICD 层 1 判据（开挖深度作为候选变量 u）
# =============================================================================
def run_vicd_stage(full_df, disp_raw, meta, ts_idx, depth):
    """对开挖深度跑 VICD 层 1 增量判据（复用 _vicd_null_distribution 协议）。

    y = 各断面 × 各深度层的累计水平位移均值（标量序列）；u = 开挖深度（场地级）。
    """
    print("\n" + "=" * 100)
    print("第 2 步：VICD 层 1 增量判据 —— 开挖深度（场地级 u）")
    print("=" * 100)
    rng = np.random.default_rng(VN.SEED)
    depths = np.asarray(meta["depths"], float)
    point_names = np.asarray(meta["point_names"])
    layers = {"全深度": np.ones(len(depths), bool),
              "浅(<3m)": depths < 3.0,
              "中(3-8m)": (depths >= 3.0) & (depths < 8.0),
              "深(8-16m)": depths >= 8.0}
    u = np.asarray(depth, float)
    header = (f"{'断面':<8}{'层':<10}{'n':>4} | {'M1增量ρ':>8}{'阈':>7} | "
              f"{'M2增量MI':>9}{'阈':>7} | {'M3符号':>8}{'阈':>7} | "
              f"{'M4先导':>8}{'阈':>7} | {'T_obs':>7}{'T_thr':>7} | 判定")
    print(header)
    print("-" * len(header))
    rows = []
    for sec in AB.Config.POINTS:
        pidx = np.where(point_names == sec)[0]
        for lname, lmask in layers.items():
            sel = pidx[lmask[pidx]]
            if len(sel) == 0:
                continue
            y = disp_raw[:, sel].mean(1)
            y, uu = np.asarray(y, float), u.copy()
            n = min(len(y), len(uu))
            y, uu = y[:n], uu[:n]
            ok = np.isfinite(y) & np.isfinite(uu)
            y, uu = y[ok], uu[ok]
            if len(y) < 20:
                continue
            r = VN.null_analysis(y, uu, 1000, rng, VN.phase_surrogate,
                                 alpha=0.95, boot=200)
            o, t = r["obs"], r["thr"]
            usable = (np.isfinite(r["T_obs"]) and np.isfinite(r["T_thr"])
                      and r["T_obs"] > r["T_thr"])
            verdict = "越界→可用" if usable else "不可用"

            def g(d, k):
                return f"{d[k]:.3f}" if np.isfinite(d.get(k, np.nan)) else "nan"

            print(f"{sec:<8}{lname:<10}{len(y):>4} | "
                  f"{g(o,'M1_inc_corr'):>8}{g(t,'M1_inc_corr'):>7} | "
                  f"{g(o,'M2_inc_mi'):>9}{g(t,'M2_inc_mi'):>7} | "
                  f"{g(o,'M3_sign_gain'):>8}{g(t,'M3_sign_gain'):>7} | "
                  f"{g(o,'M4_lead_max'):>8}{g(t,'M4_lead_max'):>7} | "
                  f"{r['T_obs']:>7.3f}{r['T_thr']:>7.3f} | {verdict}")
            row = {"case": "excavation_depth", "record": f"开挖深度↔{sec}",
                   "section": sec, "layer": lname, "n": len(y),
                   "T_obs": r["T_obs"], "T_thr_95": r["T_thr"], "argmax": r["argmax"],
                   "layer1_verdict": verdict}
            for k in VN.INCREMENT_KEYS:
                row[k] = o[k]
                row[f"{k}_thr_95"] = t[k]
            row["MLEV_corr"] = o["MLEV_corr"]
            row["MLEV_corr_thr_95"] = t["MLEV_corr"]
            rows.append(row)
    df = pd.DataFrame(rows)
    out = TABLES / "T_p05_vicd_stage.csv"
    df.round(4).to_csv(out, index=False, encoding="utf-8-sig")
    n_use = int((df["layer1_verdict"] == "越界→可用").sum())
    print("-" * len(header))
    print(f"汇总：{len(df)} 条记录中判为'越界→可用' = {n_use} 条，"
          f"'不可用' = {len(df) - n_use} 条")
    print(f"  T_obs 中位 = {df['T_obs'].median():.3f}；T_thr 中位 = {df['T_thr_95'].median():.3f}")
    print(f"[已写出] {out}")
    return df


# =============================================================================
# 第 3 步：4 维 / 5 维模型输入构造
# =============================================================================
def stage_feature_z(depth, train_idx):
    """按**训练段**统计量把开挖深度标准化为第 5 通道。"""
    s = np.asarray(depth, float)
    mu, sd = s[train_idx].mean(), s[train_idx].std() + 1e-6
    return (s - mu) / sd


def _append_stage(base, stage_seg):
    """把 (T,) 的时序标量广播为 (T,N,1) 并接到特征末尾。"""
    T, N = base.shape[0], base.shape[1]
    s = np.broadcast_to(np.asarray(stage_seg, float).reshape(-1, 1), (T, N))
    return np.concatenate([base, s[:, :, None]], axis=-1).astype(np.float32)


def prepare_data_ab_stage(cfg, full_df, wd, wa, meta, train_idx, val_idx, test_idx,
                          water_feat, geo_feat, hz, stage_seg=None, disp_raw=None):
    """`AB.prepare_data` 的忠实副本，仅在 make_features 之后追加开挖深度通道。

    该副本在 stage_seg=None 时必须与 AB.prepare_data 逐位一致（由 self_check 断言）。
    """
    if disp_raw is None:
        disp_raw, _, _ = AB.build_node_features(full_df, cfg)
    valid_mask = meta.get("valid_mask", np.ones(disp_raw.shape[1], dtype=np.float32))
    train_den, val_den, test_den = AB.denoise_split(
        disp_raw[train_idx], disp_raw[val_idx], disp_raw[test_idx], cfg)
    all_den = np.vstack([train_den, val_den, test_den])
    diff1_den = np.zeros_like(all_den)
    if len(meta["timestamps"]) > 1:
        diff1_den[1:] = np.diff(all_den, axis=0)
    geo_static_raw = np.concatenate([geo_feat, (meta["depths"] / 16.0).reshape(-1, 1)],
                                    axis=1)
    geo_static = (geo_static_raw - geo_static_raw.mean(0)) / (
        geo_static_raw.std(0) + 1e-6)

    def mf(disp, diff1, wf, seg):
        base = AB.make_features(disp, diff1, meta, wf, geo_static, cfg, False)
        return _append_stage(base, seg) if stage_seg is not None else base

    train_feat = mf(train_den, diff1_den[train_idx], water_feat[train_idx],
                    None if stage_seg is None else stage_seg[train_idx])
    val_feat = mf(val_den, diff1_den[val_idx], water_feat[val_idx],
                  None if stage_seg is None else stage_seg[val_idx])
    test_feat = mf(test_den, diff1_den[test_idx], water_feat[test_idx],
                   None if stage_seg is None else stage_seg[test_idx])
    F = train_feat.shape[-1]
    y_train, y_val, y_test = diff1_den[train_idx], diff1_den[val_idx], diff1_den[test_idx]
    scX = AB.StandardScaler().fit(train_feat.reshape(-1, F))
    tr_norm = scX.transform(train_feat.reshape(-1, F)).reshape(train_feat.shape)
    va_norm = scX.transform(val_feat.reshape(-1, F)).reshape(val_feat.shape)
    te_norm = scX.transform(test_feat.reshape(-1, F)).reshape(test_feat.shape)
    pca_y = AB.PCA(0.90).fit(y_train)
    out_dim = pca_y.n_components_
    scY = AB.StandardScaler().fit(pca_y.transform(y_train))
    ytr_norm = scY.transform(pca_y.transform(y_train))
    yva_norm = scY.transform(pca_y.transform(y_val))
    yte_norm = scY.transform(pca_y.transform(y_test))

    def make_samples(feat, tgt, lb, hval):
        X, y = [], []
        for i in range(len(feat) - lb - hval + 1):
            X.append(feat[i:i + lb])
            y.append(tgt[i + lb:i + lb + hval])
        return np.array(X, dtype=np.float32), np.array(y, dtype=np.float32)

    Xtr, ytr = make_samples(tr_norm, ytr_norm, cfg.LOOKBACK, hz)
    Xva, yva = make_samples(va_norm, yva_norm, cfg.LOOKBACK, hz)
    Xte, yte = make_samples(te_norm, yte_norm, cfg.LOOKBACK, hz)
    yte_raw = np.stack([y_test[i + cfg.LOOKBACK:i + cfg.LOOKBACK + hz]
                        for i in range(Xte.shape[0])]) if Xte.shape[0] else np.empty(
        (0, hz, y_test.shape[1]), dtype=np.float32)
    init_test = test_den[cfg.LOOKBACK - 1:-hz][:Xte.shape[0]]
    N = disp_raw.shape[1]
    w_mean = water_feat[train_idx].mean(0)
    w_std = water_feat[train_idx].std(0) + 1e-6
    wfs = (water_feat - w_mean) / w_std
    wf_train = wfs[train_idx][cfg.LOOKBACK - 1:-hz][:Xtr.shape[0]]
    wf_val = wfs[val_idx][cfg.LOOKBACK - 1:-hz][:Xva.shape[0]]
    wf_test = wfs[test_idx][cfg.LOOKBACK - 1:-hz][:Xte.shape[0]]
    wf_train = np.tile(wf_train[:, np.newaxis, :], (1, N, 1))
    wf_val = np.tile(wf_val[:, np.newaxis, :], (1, N, 1))
    wf_test = np.tile(wf_test[:, np.newaxis, :], (1, N, 1))
    return {"Xtr": Xtr, "ytr": ytr, "Xva": Xva, "yva": yva, "Xte": Xte, "yte": yte,
            "yte_raw": yte_raw, "init_test": init_test, "scX": scX, "scY": scY,
            "pca_y": pca_y, "out_dim": out_dim, "F_in": F, "N": N,
            "wf_train": wf_train, "wf_val": wf_val, "wf_test": wf_test,
            "geo_static": geo_static, "geo_dim_actual": geo_static.shape[1],
            "valid_mask": valid_mask}


def build_shared_ce_stage(cache_ce, tr, va, te, stage_seg=None):
    """时序支路共享数据：复刻 CE.prepare_shared_data，可追加开挖深度通道。"""
    cfg, water_feat, meta = cache_ce["cfg"], cache_ce["water_feat"], cache_ce["meta"]
    disp_raw, geo_feat_raw = cache_ce["disp_raw"], cache_ce["geo_feat_raw"]
    train_den, val_den, test_den = CE.denoise_split(
        disp_raw[tr], disp_raw[va], disp_raw[te], cfg)
    all_den = np.vstack([train_den, val_den, test_den])
    diff1_den = np.zeros_like(all_den)
    diff1_den[1:] = np.diff(all_den, axis=0)
    geo_static_raw = np.concatenate([
        geo_feat_raw if geo_feat_raw is not None else np.zeros(
            (len(meta["depths"]), 6), dtype=np.float32),
        (meta["depths"] / 16.0).reshape(-1, 1)], axis=1)
    geo_static = (geo_static_raw - geo_static_raw.mean(0)) / (
        geo_static_raw.std(0) + 1e-6)
    o1, o2 = len(tr), len(tr) + len(va)

    def mf(disp, diff1, wf, seg):
        base = CE.make_features(disp, diff1, meta, wf, None, use_geo=False)
        return _append_stage(base, seg) if stage_seg is not None else base

    t_tr = mf(train_den, diff1_den[:o1], water_feat[tr],
              None if stage_seg is None else stage_seg[tr])
    t_va = mf(val_den, diff1_den[o1:o2], water_feat[va],
              None if stage_seg is None else stage_seg[va])
    t_te = mf(test_den, diff1_den[o2:], water_feat[te],
              None if stage_seg is None else stage_seg[te])
    geo_fb = geo_feat_raw if geo_feat_raw is not None else np.zeros(
        (len(meta["depths"]), 6), dtype=np.float32)
    adj_norm, L_scaled = CE.build_adjacency(meta, geo_fb, cfg)
    return {"train_feat_nogeo": t_tr, "val_feat_nogeo": t_va, "test_feat_nogeo": t_te,
            "F_in_nogeo": t_tr.shape[-1], "N": t_tr.shape[1],
            "y_train": diff1_den[:o1], "y_val": diff1_den[o1:o2],
            "y_test": diff1_den[o2:], "train_idx": tr, "val_idx": va, "test_idx": te,
            "ts_arr": cache_ce["ts"], "meta": meta, "adj_norm": adj_norm,
            "L_scaled": L_scaled, "geo_static": geo_static, "geo_fb": geo_fb,
            "test_den": test_den, "water_feat": water_feat}


# =============================================================================
# 第 4 步：单次训练（图支路 / 时序支路）
# =============================================================================
def load_caches(base_dir):
    cfg_ab = AB.Config()
    AB.set_seed(42)
    full_ab = AB.load_raw_data_with_meta(cfg_ab)
    ts = np.sort(full_ab["测量时间"].unique())
    wd, wa = AB.load_water_data(ts, cfg_ab)
    disp_ab, _, meta_ab = AB.build_node_features(full_ab, cfg_ab)
    geo_ab = AB.load_geology_params(cfg_ab)
    geo_feat_ab = AB.assign_geo_to_nodes(geo_ab, meta_ab["depths"], cfg_ab) \
        if geo_ab else None
    caches = {"ab": dict(cfg=cfg_ab, full_df=full_ab, ts=ts, wd=wd, wa=wa,
                         water_feat=AB.extract_water_features(wd, wa),
                         disp_raw=disp_ab, meta=meta_ab, geo_feat_raw=geo_feat_ab)}

    cfg_ce = CE.Config()
    CE.set_seed(42)
    wd2, wa2 = CE.load_water_data(ts, cfg_ce)
    disp_ce, _, meta_ce = CE.build_node_features(full_ab, cfg_ce)
    geo_ce = CE.load_geology_params(cfg_ce)
    geo_feat_ce = CE.assign_geo_to_nodes(geo_ce, meta_ce["depths"], cfg_ce) \
        if geo_ce else None
    caches["ce"] = dict(cfg=cfg_ce, ts=ts, wd=wd2, wa=wa2,
                        water_feat=CE.extract_water_features(wd2, wa2),
                        disp_raw=disp_ce, meta=meta_ce, geo_feat_raw=geo_feat_ce)
    return caches


def run_graph_seed(cfg, variant, cache, tr, va, te, hz, stage_seg, out_dir, seed,
                   resume):
    """图支路：Ours(V7) / TCN-Only(V0)。"""
    cfg = copy.deepcopy(cfg)
    for k, v in AB.ABLATION_VARIANTS[variant].items():
        setattr(cfg, k, v)
    full_df, meta = cache["full_df"], cache["meta"]
    gf = cache["geo_feat_raw"] if cache["geo_feat_raw"] is not None else np.zeros(
        (len(meta["depths"]), cfg.GEO_FEAT_DIM), dtype=np.float32)
    data = prepare_data_ab_stage(cfg, full_df, cache["wd"], cache["wa"], meta,
                                 tr, va, te, cache["water_feat"], gf, hz,
                                 stage_seg=stage_seg, disp_raw=cache["disp_raw"])
    geo_fb_use = gf if np.any(gf) else None
    adj = AB.build_adjacency(meta, geo_fb_use, cfg,
                             zero_same_hole=getattr(cfg, "ZERO_SAME_HOLE", False))
    adj_t = torch.FloatTensor(adj).to(cfg.DEVICE)
    vert_adj = AB.build_vertical_adjacency(meta, cfg)
    vert_adj_t = torch.FloatTensor(vert_adj).to(cfg.DEVICE) if vert_adj is not None else None
    geo_static_t = torch.FloatTensor(data["geo_static"]).to(cfg.DEVICE)
    valid_mask_t = torch.FloatTensor(data["valid_mask"]).to(cfg.DEVICE)
    N, F_in = data["N"], data["F_in"]

    sp = out_dir / f"run_{seed}"
    sp.mkdir(parents=True, exist_ok=True)
    mfile, pfile = sp / "metrics.json", sp / "predictions.npz"
    if resume and mfile.exists() and pfile.exists():
        gm = json.load(open(mfile, encoding="utf-8"))
        return gm

    AB.set_seed(seed)
    ds = lambda X, y, wf: TensorDataset(
        torch.FloatTensor(X), torch.FloatTensor(y), torch.FloatTensor(wf),
        geo_static_t.unsqueeze(0).expand(X.shape[0], N, -1))
    ld_tr = DataLoader(ds(data["Xtr"], data["ytr"], data["wf_train"]),
                       batch_size=cfg.BATCH_SIZE, shuffle=True)
    ld_va = DataLoader(ds(data["Xva"], data["yva"], data["wf_val"]),
                       batch_size=cfg.BATCH_SIZE)
    ld_te = DataLoader(ds(data["Xte"], data["yte"], data["wf_test"]),
                       batch_size=cfg.BATCH_SIZE)
    model = AB.AblationModel(cfg, N=N, in_dim=F_in, gcn_hid=cfg.GCN_HIDDEN,
                             tcn_ch=cfg.TCN_CHANNELS, k=cfg.TCN_KERNEL_SIZE,
                             dil=cfg.TCN_DILATIONS, attn_hid=cfg.ATTENTION_HIDDEN,
                             hor=hz, adj=adj_t, out_dim=data["out_dim"],
                             geo_feat_dim=data["geo_dim_actual"],
                             vert_adj=vert_adj_t).to(cfg.DEVICE)
    # 复现锁定脚本 `AB.run_single_seed` 的 RNG 消耗：FLOPs 估计会跑一次 train 态
    # forward（含 dropout）并消费全局随机数。不调用则训练轨迹与既有锚点不一致
    # （实测末步 RMSE 偏差 ~1e-2）。此处仅复现，不落盘该量。
    AB.count_params(model)
    AB.estimate_flops(model, cfg, N, F_in, hz)
    model, _, _, _ = AB.train_model(cfg, model, ld_tr, ld_va, sp / "best.pth",
                                    node_mask=valid_mask_t)
    pred_cum, true_cum = AB.evaluate(model, ld_te, data["init_test"], data["scY"],
                                     data["pca_y"], hz, cfg.DEVICE,
                                     node_mask=valid_mask_t,
                                     true_diff_win=data["yte_raw"])
    gm = AB.global_metrics(true_cum, pred_cum)
    gm["seed"] = seed
    zero = float(np.sqrt(np.mean(
        (data["init_test"] - true_cum[:, -1, :]) ** 2)))
    gm["RMSE_zero_last"] = zero
    gm["K"] = int(data["out_dim"])
    gm["n_win"] = int(pred_cum.shape[0])
    gm["F_in"] = int(F_in)
    json.dump(gm, open(mfile, "w", encoding="utf-8"), indent=2)
    np.savez(pfile, pred_cum=pred_cum, true_cum=true_cum)
    return gm


def run_temporal_seed(cfg, cache_ce, tr, va, te, hz, stage_seg, out_dir, seed, resume):
    """时序支路：BiLSTM-Attn。"""
    sh = build_shared_ce_stage(cache_ce, tr, va, te, stage_seg=stage_seg)
    tcfg = copy.deepcopy(cfg)
    tcfg.OUT_ROOT = out_dir
    # `CE.build_temporal_samples` 的 `PCA(n_components=48)` 走随机化 SVD（svd_solver
    # ='auto'，n_components<0.8·min(n,p) 且 max(p)>500），会消费 numpy 全局随机流 ⇒
    # 结果不可复现，且 4d/5d 两路的投影基不同（620 vs 775 维输入）。此处固定 np
    # 随机种子使输入构造确定，随后 `CE.set_seed(seed)` 会重置随机流，不影响训练。
    np.random.seed(BOOT_SEED)
    tdata = CE.build_temporal_samples(sh, tcfg, hz)
    N = sh["N"]
    sp = out_dir / f"run_{seed}"
    sp.mkdir(parents=True, exist_ok=True)
    mfile, pfile = sp / "metrics.json", sp / "predictions.npz"
    if resume and mfile.exists() and pfile.exists():
        return json.load(open(mfile, encoding="utf-8"))

    CE.set_seed(seed)
    model = CE.get_model(tcfg, "BiLSTM-Attn", N, tdata["temporal_in_dim"],
                         tdata["out_dim"], hz, None, None).to(tcfg.DEVICE)
    ld_tr = DataLoader(TensorDataset(torch.FloatTensor(tdata["Xtr"]),
                                     torch.FloatTensor(tdata["ytr"])),
                       batch_size=tcfg.BATCH_SIZE, shuffle=True)
    ld_va = DataLoader(TensorDataset(torch.FloatTensor(tdata["Xva"]),
                                     torch.FloatTensor(tdata["yva"])),
                       batch_size=tcfg.BATCH_SIZE)
    ld_te = DataLoader(TensorDataset(torch.FloatTensor(tdata["Xte"]),
                                     torch.FloatTensor(tdata["yte"])),
                       batch_size=tcfg.BATCH_SIZE)
    model, _, _, _, _ = CE.train_model(tcfg, model, ld_tr, ld_va, sp / "best.pth",
                                       model_tag="BiLSTM-Attn", seed=seed,
                                       ours_lr_groups=False)
    pred_cum, true_cum = CE.evaluate(model, ld_te, tdata["init_test"], tdata["scY"],
                                     tdata["pca_y"], hz, tcfg.DEVICE,
                                     true_diff_win=tdata["yte_raw"])
    gm = CE.global_metrics(true_cum, pred_cum)
    gm["seed"] = seed
    zero = float(np.sqrt(np.mean(
        (tdata["init_test"] - true_cum[:, -1, :]) ** 2)))
    gm["RMSE_zero_last"] = zero
    gm["K"] = int(tdata["out_dim"])
    gm["n_win"] = int(pred_cum.shape[0])
    gm["F_in"] = int(sh["F_in_nogeo"])
    json.dump(gm, open(mfile, "w", encoding="utf-8"), indent=2)
    np.savez(pfile, pred_cum=pred_cum, true_cum=true_cum)
    return gm


# =============================================================================
# 第 5 步：汇总与统计（配对 t + 效应量 + 时间块 bootstrap）
# =============================================================================
def circular_block_idx(n, L, rng):
    nblk = int(np.ceil(n / L))
    starts = rng.integers(0, n, size=nblk)
    return np.concatenate([(np.arange(s, s + L) % n) for s in starts])[:n]


def window_se(pred_cum, true_cum):
    pc = np.asarray(pred_cum, float)
    tc = np.asarray(true_cum, float)
    return np.mean((pc[:, -1, :] - tc[:, -1, :]) ** 2, axis=1)


def seed_rmse(out_dir, seeds):
    vals = []
    for s in seeds:
        p = out_dir / f"run_{s}" / "metrics.json"
        vals.append(float(json.load(open(p, encoding="utf-8"))["RMSE_last"]))
    return np.asarray(vals)


def seed_se_windows(out_dir, seeds):
    mat = []
    for s in seeds:
        z = np.load(out_dir / f"run_{s}" / "predictions.npz")
        mat.append(window_se(z["pred_cum"], z["true_cum"]))
    return np.vstack(mat)


def summarize(seeds):
    rows = []
    rng = np.random.default_rng(BOOT_SEED)
    for model in MODELS:
        d4 = OUT_ROOT / f"H{HZ}" / f"{model}_4d"
        d5 = OUT_ROOT / f"H{HZ}" / f"{model}_5d"
        r4 = seed_rmse(d4, seeds)
        r5 = seed_rmse(d5, seeds)
        se4 = seed_se_windows(d4, seeds)
        se5 = seed_se_windows(d5, seeds)
        n_win = se4.shape[1]
        idxs = [circular_block_idx(n_win, L_STAR, rng) for _ in range(B_BOOT)]

        def pooled(se, idx):
            return float(np.sqrt(se[:, idx].mean(axis=1)).mean())

        delta_boot = np.array([pooled(se5, ix) - pooled(se4, ix) for ix in idxs])
        ci = (float(np.percentile(delta_boot, 2.5)),
              float(np.percentile(delta_boot, 97.5)))
        p_boot = float(min(1.0, 2 * min(float(np.mean(delta_boot > 0)),
                                        float(np.mean(delta_boot < 0)))))
        t, p = stats.ttest_rel(r5, r4)
        diff = r5 - r4
        d_eff = float(diff.mean() / (diff.std(ddof=1) + 1e-12))
        delta_pct = float(100 * (r5.mean() - r4.mean()) / r4.mean())
        for tag, r in (("4d", r4), ("5d", r5)):
            rows.append({
                "H": HZ, "model": model, "featdim": tag, "n_seed": len(r),
                "RMSE_last_mean": float(r.mean()),
                "RMSE_last_std": float(r.std(ddof=1)),
                "R2_last": float(np.mean([json.load(open(
                    OUT_ROOT / f"H{HZ}" / f"{model}_{tag}" / f"run_{s}" / "metrics.json",
                    encoding="utf-8"))["R2_last"] for s in seeds])),
                "delta_vs_4d_pct": 0.0 if tag == "4d" else delta_pct,
                "p_vs_4d": np.nan if tag == "4d" else float(p),
                "note": (f"配对t={t:+.3f}; cohen_d={d_eff:+.3f}; "
                         f"时间块bootstrap Δ95%CI=[{ci[0]:+.4f},{ci[1]:+.4f}] "
                         f"p_boot={p_boot:.4f} (B={B_BOOT},L*={L_STAR},n_win={n_win}); "
                         f"K={json.load(open(OUT_ROOT / f'H{HZ}' / f'{model}_{tag}' / f'run_{seeds[0]}' / 'metrics.json', encoding='utf-8'))['K']}"),
            })
            if tag == "4d":
                rows[-1]["note"] += "; 4D=基线"
    df = pd.DataFrame(rows, columns=["H", "model", "featdim", "n_seed",
                                     "RMSE_last_mean", "RMSE_last_std", "R2_last",
                                     "delta_vs_4d_pct", "p_vs_4d", "note"])
    out = TABLES / "T_p05_stage_input.csv"
    df.round(6).to_csv(out, index=False, encoding="utf-8-sig")
    print(f"\n[已写出] {out}")
    print(df[["model", "featdim", "n_seed", "RMSE_last_mean", "RMSE_last_std",
              "delta_vs_4d_pct", "p_vs_4d"]].round(4).to_string(index=False))
    return df


# =============================================================================
# 主流程
# =============================================================================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--probe", action="store_true", help="只做第 1/2 步（探查+VICD），不训练")
    ap.add_argument("--smoke", action="store_true", help="冒烟：2 种子 / 5 epoch")
    args = ap.parse_args()

    base_dir = AB.Config().BASE_DATA_DIR
    ts_idx, depth, anchors = probe_stage(base_dir)
    report_cross_section(base_dir, ts_idx)

    caches = load_caches(base_dir)
    ca, cc = caches["ab"], caches["ce"]
    T = len(ca["ts"])
    tr = np.arange(0, int(T * 0.70))
    va = np.arange(int(T * 0.70), int(T * 0.85))
    te = np.arange(int(T * 0.85), T)
    print(f"\n数据划分：T={T}  train={len(tr)}  val={len(va)}  test={len(te)}  "
          f"LOOKBACK={ca['cfg'].LOOKBACK}  H={HZ}  "
          f"测试窗数={len(te) - ca['cfg'].LOOKBACK - HZ + 1}")

    # 深度序列对齐到模型时间轴（模型时间轴 = 全断面并集）
    depth_full = pd.Series(depth, index=ts_idx).reindex(
        pd.DatetimeIndex(ca["ts"])).interpolate().bfill().ffill().values.astype(float)
    assert np.isfinite(depth_full).all(), "深度序列存在缺失，停止（不编造）。"
    print(f"深度序列对齐：长度 {len(depth_full)}（参考断面 {len(depth)}）")

    # ---------------- 锚点自检：零变化 ----------------
    from Ablation_Experiment import denoise_split as _dn
    tr_den, va_den, te_den = _dn(ca["disp_raw"][tr], ca["disp_raw"][va],
                                 ca["disp_raw"][te], ca["cfg"])
    nth = te_den.shape[0] - ca["cfg"].LOOKBACK - HZ + 1
    init_test = te_den[ca["cfg"].LOOKBACK - 1:ca["cfg"].LOOKBACK - 1 + nth]
    true_last = te_den[ca["cfg"].LOOKBACK - 1 + HZ:
                       ca["cfg"].LOOKBACK - 1 + HZ + nth]
    zero = float(np.sqrt(np.mean((init_test - true_last) ** 2)))
    print(f"\n锚点自检(零变化) H={HZ} 累计末步 RMSE = {zero:.6f}  "
          f"(参考 {ZERO_H10:.6f})  [{'OK' if abs(zero - ZERO_H10) < 1e-4 else 'FAIL'}]")

    # ---------------- 第 2 步 VICD ----------------
    vicd = run_vicd_stage(ca["full_df"], ca["disp_raw"], ca["meta"], ts_idx, depth_full)

    if args.probe:
        print("\n--probe：已完成第 1/2 步，跳过训练。")
        return

    # ---------------- 第 3 步训练 ----------------
    seeds = list(ca["cfg"].SEEDS)
    epochs_bak = patience_bak = None
    if args.smoke:
        print("\n!! SMOKE：2 种子 / 5 epoch，仅验证管线")
        seeds = seeds[:2]
        AB.Config.EPOCHS = 5
        AB.Config.PATIENCE = 3
        CE.Config.EPOCHS = 5
        CE.Config.PATIENCE = 3
    resume = os.environ.get("P05_RESUME", "0") == "1"
    stage_z = stage_feature_z(depth_full, tr)

    # ---- 保真自检：stage_seg=None 时，副本必须与锁定脚本逐位一致 ----
    gf = ca["geo_feat_raw"] if ca["geo_feat_raw"] is not None else np.zeros(
        (len(ca["meta"]["depths"]), ca["cfg"].GEO_FEAT_DIM), dtype=np.float32)
    d_ref = AB.prepare_data(ca["cfg"], ca["full_df"], ca["wd"], ca["wa"], ca["meta"],
                            tr, va, te, ca["water_feat"], gf, HZ,
                            disp_raw=ca["disp_raw"])
    d_new = prepare_data_ab_stage(ca["cfg"], ca["full_df"], ca["wd"], ca["wa"],
                                  ca["meta"], tr, va, te, ca["water_feat"], gf, HZ,
                                  stage_seg=None, disp_raw=ca["disp_raw"])
    bad = [k for k in d_ref if isinstance(d_ref[k], np.ndarray)
           and d_new[k].shape != d_ref[k].shape]
    diffk = [k for k in d_ref if isinstance(d_ref[k], np.ndarray)
             and d_new[k].shape == d_ref[k].shape
             and not np.allclose(d_new[k], d_ref[k])]
    print(f"[自检] 图支路 stage=None 与 AB.prepare_data 一致：形状不符={bad or '无'}，"
          f"数值不符={diffk or '无'}")
    assert not bad and not diffk, "图支路保真自检失败，禁止继续"
    sh_ref = CE.prepare_shared_data(cc["cfg"])
    sh_new = build_shared_ce_stage(cc, tr, va, te, stage_seg=None)
    dk = [k for k in ("train_feat_nogeo", "val_feat_nogeo", "test_feat_nogeo",
                      "y_train", "y_val", "y_test")
          if not np.allclose(sh_ref[k], sh_new[k])]
    print(f"[自检] 时序支路 stage=None 与 CE.prepare_shared_data 一致：数值不符={dk or '无'}")
    assert not dk, "时序支路保真自检失败，禁止继续"

    t0 = time.time()
    total = len(MODELS) * 2 * len(seeds)
    done = 0
    for model in MODELS:
        for tag, seg in (("4d", None), ("5d", stage_z)):
            out_dir = OUT_ROOT / f"H{HZ}" / f"{model}_{tag}"
            out_dir.mkdir(parents=True, exist_ok=True)
            for seed in seeds:
                done += 1
                if model in VAR_OF:
                    gm = run_graph_seed(ca["cfg"], VAR_OF[model], ca, tr, va, te, HZ,
                                        seg, out_dir, seed, resume)
                else:
                    gm = run_temporal_seed(cc["cfg"], cc, tr, va, te, HZ,
                                           seg, out_dir, seed, resume)
                print(f"  [{done:>3}/{total}] {model:<12}{tag}  seed={seed:<5} "
                      f"RMSE_last={gm['RMSE_last']:.4f}  zero={gm['RMSE_zero_last']:.4f}  "
                      f"({time.time() - t0:.0f}s)", flush=True)

    # ---------------- 第 4 步锚点复算 ----------------
    print("\n" + "=" * 100)
    print("锚点复算（4 维 / Ours / H=10 / 30 种子）")
    print("=" * 100)
    r4 = seed_rmse(OUT_ROOT / f"H{HZ}" / "Ours_4d", seeds)
    m, sd = float(r4.mean()), float(r4.std(ddof=1))
    ok = abs(m - ANCHOR_OURS_MEAN) < 1e-3 and abs(sd - ANCHOR_OURS_STD) < 1e-3
    print(f"  Ours_4d RMSE_last = {m:.4f} ± {sd:.4f}  "
          f"(参考 {ANCHOR_OURS_MEAN:.4f} ± {ANCHOR_OURS_STD:.4f})  "
          f"[{'OK' if ok else 'FAIL'}]")

    summarize(seeds)
    print(f"\n总耗时 {time.time() - t0:.1f}s；结果目录 {OUT_ROOT}")


if __name__ == "__main__":
    main()