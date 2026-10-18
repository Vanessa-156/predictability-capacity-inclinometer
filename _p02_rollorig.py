# -*- coding: utf-8 -*-
"""_p02_rollorig.py — rolling-origin（walk-forward）验证（需重训）。

背景
----
主表 T1/T2 的全部数值来自**单一固定划分** train[0,149) / val[149,181) / test[181,213)
（时间轴共 n=213 步）。这里检验结论是否对**时间划分**稳健。
本脚本按**扩展窗口（expanding-window / walk-forward）**协议，把 [0,213) 切成 6 个
**不重叠**的连续测试块，逐折在其之前的数据上重训，从而给出"跨折 RMSE 分布"、
"最优变体选择稳定性"与"跨折配对检验"。

铁律（见任务书）
----------------
1. 不覆盖任何旧结果：**全部输出写入 `<WORK_ROOT>\WalkForward\\`**；
   汇总 CSV 写入既有的 `_paper_tables\\`（仅新增文件，不改旧文件）。
2. 每折必须**在其自身训练段内**重新拟合：去噪(wavelet)、标准化(scX/scY)、
   PCA(K)、水位统计量 —— 由 `Ablation_Experiment.prepare_data` 天然保证
   （它只在 train 切片上 fit），本脚本只改时间划分，不改管线。
3. 不编造：跑不完/某折失败如实报告（逐 run 的 note 记录失败原因）。

折定义（可复现，写入脚本与报告）
--------------------------------
时间轴 n=213，划分 7 个连续段：1 个 warm-up 训练段 S0 与 6 个测试块 S1..S6：

    边界 BOUNDS = [57, 83, 109, 135, 161, 187, 213]
    S0 = [0, 57)                                    (warm-up，仅作早期训练数据)
    Sk = [BOUNDS[k-1], BOUNDS[k])   k=1..6          (每块 26 步)

第 k 折：
    测试段 test_k = Sk                              (26 步，各折互不重叠)
    训练段 train_region_k = [0, BOUNDS[k-1])        (严格早于 test_k，随 k 扩展)
    内部再切：val_k = 训练段末 30 步，train_k = 其余（固定 val 长度 30 步）
原因：n=213 与 6 折下无法用原比例(0.70/0.15)切 val——H=10 时 val 需 ≥22 步才有
一个验证窗，按比例折 1 的 val 仅 ~10 步不可用。故固定 val=30 步，train_k 长度
随折从 27 增至 157。全部折的 train_k/test_k 均满足 LOOKBACK(12)+H(10)+1 ≤ 段长。

模型 / 视界 / 种子
------------------
模型 2 个：Ours（=V7_wo_VerticalGCN 架构与配方）、V0（=V0_Pure_TCN，纯 TCN）。
视界 H ∈ {2, 10}；每折每格取 30 种子清单的**前 5 个**（见 `_gen_seeds.py`）：
    SEEDS5 = [42, 15, 77, 85, 120]
规模：6 折 × 2 模型 × 2 视界 × 5 种子 = 120 次训练。

⚠️ 本协议改变时间划分，**其数值不得与主表 T1/T2/T3 混用**。

用法
----
  # 锚点自检 + 单折(fold1, H=10)单种子 快速验证
  python _p02_rollorig.py --stage single
  # 锚点自检 + 全量 120 runs（写逐 run 结果与汇总）
  python _p02_rollorig.py --stage full
  # 跳过锚点（仅在锚点已单独验证过时使用）
  ... --stage full --skip-anchor
"""
from _env import AUX_ROOT, WORK_ROOT

import argparse
import copy
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

PROJ = WORK_ROOT
if str(PROJ) not in sys.path:
    sys.path.insert(0, str(PROJ))

import _robust_io  # noqa: F401  （须早于 numpy/pandas/torch）
import torch
from torch.utils.data import DataLoader, TensorDataset

import Ablation_Experiment as AE

# =============================================================================
# 常量与折定义
# =============================================================================
WF_ROOT = PROJ / "WalkForward"                     # 全部新结果（绝不覆盖旧目录）
TABLES = PROJ / "_paper_tables"
SUPP = PROJ / "补充材料"

HORIZONS = [2, 10]
# 目录名 / 汇总表中的模型标签 -> Ablation_Experiment 变体键
MODELS = {"Ours": "V7_wo_VerticalGCN", "V0": "V0_Pure_TCN"}
MODEL_ORDER = ["Ours", "V0"]
SEEDS5 = list(AE.Config.SEEDS[:5])                 # [42, 15, 77, 85, 120]

N_TOTAL = 213
BOUNDS = [57, 83, 109, 135, 161, 187, 213]         # 7 段边界，见模块 docstring
FOLDS = list(range(1, 7))                          # 第 k 折用 BOUNDS[k-1:k+1]
VAL_SIZE = 30                                      # 每折内部验证段长度（固定）

# 锚点：原锁定划分 train[0,149) val[149,181) test[181,213)，Ours H=10 的既有结果
ANCHOR_TR_END = int(N_TOTAL * 0.70)                # 149
ANCHOR_VA_END = int(N_TOTAL * 0.85)                # 181
ANCHOR_MEAN_H10 = 1.5687
ANCHOR_STD_H10 = 0.0940
# 既有 30 种子逐 run 结果（用于逐种子精确比对；缺失则回退到均值比对）
ANCHOR_CSV = AUX_ROOT / "Ablation_New_5d/H10/V7_wo_VerticalGCN/summary/global_30runs.csv"


# =============================================================================
# 数据静态段（与时间划分无关）
# =============================================================================
def build_static(cfg):
    AE.set_seed(42)
    full_df = AE.load_raw_data_with_meta(cfg)
    ts = np.sort(full_df["测量时间"].unique())
    wd, wa = AE.load_water_data(ts, cfg)
    water_feat = AE.extract_water_features(wd, wa)
    disp_raw, _, meta = AE.build_node_features(full_df, cfg)
    geo_records = AE.load_geology_params(cfg)
    geo_feat_raw = (AE.assign_geo_to_nodes(geo_records, meta["depths"], cfg)
                    if geo_records else None)
    geo_fb = (geo_feat_raw if geo_feat_raw is not None
              else np.zeros((len(meta["depths"]), cfg.GEO_FEAT_DIM), dtype=np.float32))
    assert len(ts) == N_TOTAL, f"时间轴步数 {len(ts)} != {N_TOTAL}"
    print(f"[data] n={len(ts)}  N_nodes={disp_raw.shape[1]}  "
          f"geo={'有' if geo_feat_raw is not None else '无'}")
    return {"full_df": full_df, "ts": ts, "wd": wd, "wa": wa,
            "water_feat": water_feat, "disp_raw": disp_raw, "meta": meta,
            "geo_feat_raw": geo_feat_raw, "geo_fb": geo_fb}


# =============================================================================
# 单格（fold × H × model × seed）训练 + 测试/验证评估
# =============================================================================
def _val_arrays(static, cfg, train_idx, val_idx, test_idx, hz, data):
    """重建验证段的累计口径真值，用于"验证段最优变体"选择（不参与训练）。"""
    tr_den, va_den, te_den = AE.denoise_split(
        static["disp_raw"][train_idx], static["disp_raw"][val_idx],
        static["disp_raw"][test_idx], cfg)
    all_den = np.vstack([tr_den, va_den, te_den])
    diff1_den = np.zeros_like(all_den)
    diff1_den[1:] = np.diff(all_den, axis=0)
    y_val = diff1_den[val_idx]
    Xva = data["Xva"]
    ns = Xva.shape[0]
    init_val = va_den[cfg.LOOKBACK - 1:-hz][:ns]
    y_val_raw = (np.stack([y_val[i + cfg.LOOKBACK:i + cfg.LOOKBACK + hz]
                           for i in range(ns)])
                 if ns else np.empty((0, hz, y_val.shape[1]), dtype=np.float32))
    return Xva, init_val, y_val_raw


def run_cell(static, base_cfg, hz, model_label, train_idx, val_idx, test_idx,
             out_root, seed):
    """训练单个网格单元；沿用 AE.run_single_seed（写 metrics.json / predictions.npz）。"""
    variant = MODELS[model_label]
    cfg = copy.deepcopy(base_cfg)
    for k, v in AE.ABLATION_VARIANTS[variant].items():
        setattr(cfg, k, v)
    cfg.OUT_ROOT = out_root
    cfg.OUT_ROOT.mkdir(parents=True, exist_ok=True)

    data = AE.prepare_data(cfg, static["full_df"], static["wd"], static["wa"],
                           static["meta"], train_idx, val_idx, test_idx,
                           static["water_feat"], static["geo_fb"], hz,
                           disp_raw=static["disp_raw"])
    geo_fb_use = (static["geo_feat_raw"]
                  if static["geo_feat_raw"] is not None
                  and np.any(static["geo_feat_raw"]) else None)
    adj = AE.build_adjacency(static["meta"], geo_fb_use, cfg,
                             zero_same_hole=getattr(cfg, "ZERO_SAME_HOLE", False))
    adj_t = torch.FloatTensor(adj).to(cfg.DEVICE)
    vert_adj_t = torch.FloatTensor(
        AE.build_vertical_adjacency(static["meta"], cfg)).to(cfg.DEVICE)
    geo_static_t = torch.FloatTensor(data["geo_static"]).to(cfg.DEVICE)

    gm, pred_cum, true_cum, _, model = AE.run_single_seed(
        cfg, seed, data, static["meta"], adj_t, geo_static_t,
        train_idx, val_idx, test_idx, static["ts"], hz, vert_adj_t=vert_adj_t)

    # ---- 验证段评估（仅用于"最优变体"选择，不写盘、不参与训练）----
    Xva, init_val, y_val_raw = _val_arrays(static, cfg, train_idx, val_idx,
                                           test_idx, hz, data)
    N = data["N"]
    valid_mask_t = torch.FloatTensor(data["valid_mask"]).to(cfg.DEVICE)
    val_ds = TensorDataset(
        torch.FloatTensor(Xva), torch.FloatTensor(data["yva"]),
        torch.FloatTensor(data["wf_val"]),
        geo_static_t.unsqueeze(0).expand(Xva.shape[0], N, -1))
    val_ld = DataLoader(val_ds, batch_size=cfg.BATCH_SIZE)
    pv, tv = AE.evaluate(model, val_ld, init_val, data["scY"], data["pca_y"],
                         hz, cfg.DEVICE, node_mask=valid_mask_t,
                         true_diff_win=y_val_raw)
    gm_v = AE.global_metrics(tv, pv)

    return {
        "RMSE_last": float(gm["RMSE_last"]), "R2_last": float(gm["R2_last"]),
        "MAE_last": float(gm["MAE_last"]),
        "RMSE_last_val": float(gm_v["RMSE_last"]),
        "R2_last_val": float(gm_v["R2_last"]),
        "n_win": int(pred_cum.shape[0]), "n_win_val": int(pv.shape[0]),
    }


# =============================================================================
# 锚点自检：原锁定划分复跑 Ours H=10
# =============================================================================
def anchor_check(static, base_cfg, only_seed=None):
    print("\n" + "=" * 88)
    print("锚点自检：原锁定划分 train[0,%d) val[%d,%d) test[%d,%d)，Ours H=10"
          % (ANCHOR_TR_END, ANCHOR_TR_END, ANCHOR_VA_END, ANCHOR_VA_END, N_TOTAL))
    print("=" * 88)
    train_idx = np.arange(0, ANCHOR_TR_END)
    val_idx = np.arange(ANCHOR_TR_END, ANCHOR_VA_END)
    test_idx = np.arange(ANCHOR_VA_END, N_TOTAL)
    seeds = [only_seed] if only_seed is not None else SEEDS5
    out_root = WF_ROOT / "_anchor" / "H10" / "Ours"

    stored = {}
    if ANCHOR_CSV.exists():
        d = pd.read_csv(ANCHOR_CSV)
        stored = {int(r["seed"]): float(r["RMSE_last"]) for _, r in d.iterrows()}
        all_vals = np.array(list(stored.values()))
        print(f"  既有 30 种子 Ours H=10 主表：mean={all_vals.mean():.4f} "
              f"std={all_vals.std(ddof=1):.4f}（参考 {ANCHOR_MEAN_H10}±{ANCHOR_STD_H10}）")

    got = {}
    for s in seeds:
        row = run_cell(static, base_cfg, 10, "Ours", train_idx, val_idx, test_idx,
                       out_root, s)
        got[s] = row["RMSE_last"]
        ref = stored.get(s)
        tag = ""
        if ref is not None:
            tag = f"  (既有 run_{s}={ref:.6f}, 差={row['RMSE_last']-ref:+.2e})"
        print(f"    seed={s}: 复跑 RMSE_last={row['RMSE_last']:.6f}{tag}")

    diffs = [abs(got[s] - stored[s]) for s in got if s in stored]
    max_diff = max(diffs) if diffs else float("nan")
    sub_mean = np.mean([stored[s] for s in got if s in stored]) if diffs else float("nan")
    seed_mean = np.mean(list(got.values()))
    print(f"  复跑 {len(got)} 种子均值={seed_mean:.4f}；既有对应子集均值={sub_mean:.4f}；"
          f"逐种子最大绝对差={max_diff:.2e}")
    ok = (not diffs) or (max_diff < 5e-3)
    print(f"  锚点判定：{'通过 ✓' if ok else '未通过 ✗（改动可能影响管线）'}")
    return ok, seed_mean, max_diff, stored


# =============================================================================
# 单折 / 全量执行
# =============================================================================
def fold_indices(k):
    test_idx = np.arange(BOUNDS[k - 1], BOUNDS[k])
    region_end = BOUNDS[k - 1]
    val_idx = np.arange(region_end - VAL_SIZE, region_end)
    train_idx = np.arange(0, region_end - VAL_SIZE)
    return train_idx, val_idx, test_idx


def run_stage(static, base_cfg, stage):
    """stage='single'：fold1×H10×2 模型×1 种子；stage='full'：全部 120 runs。"""
    if stage == "single":
        jobs = [(1, 10, m, SEEDS5[0]) for m in MODEL_ORDER]
    else:
        jobs = [(k, hz, m, s) for k in FOLDS for hz in HORIZONS
                for m in MODEL_ORDER for s in SEEDS5]

    print("\n" + "=" * 88)
    print(f"执行阶段：{stage}  共 {len(jobs)} 个训练单元")
    print("=" * 88)
    rows = []
    t0 = time.time()
    for i, (k, hz, m, s) in enumerate(jobs, 1):
        train_idx, val_idx, test_idx = fold_indices(k)
        out_root = WF_ROOT / f"H{hz}" / m / f"fold{k}"
        note = (f"variant={MODELS[m]}; train=[0,{train_idx.shape[0]})+val="
                f"[{val_idx[0]},{val_idx[-1]+1}) test=[{test_idx[0]},{test_idx[-1]+1})")
        try:
            row = run_cell(static, base_cfg, hz, m, train_idx, val_idx, test_idx,
                           out_root, s)
            rows.append(dict(fold=k, H=hz, model=m, seed=s,
                             RMSE_last=row["RMSE_last"], R2_last=row["R2_last"],
                             RMSE_last_val=row["RMSE_last_val"],
                             R2_last_val=row["R2_last_val"],
                             n_win=row["n_win"], n_win_val=row["n_win_val"],
                             note=note))
            print(f"[{i}/{len(jobs)}] fold{k} H{hz} {m} seed{s}: "
                  f"test RMSE_last={row['RMSE_last']:.4f} | "
                  f"val RMSE_last={row['RMSE_last_val']:.4f} "
                  f"(n_win={row['n_win']}/{row['n_win_val']}, {time.time()-t0:.0f}s)")
        except Exception as e:                              # 如实记录失败
            import traceback
            traceback.print_exc()
            rows.append(dict(fold=k, H=hz, model=m, seed=s,
                             RMSE_last=np.nan, R2_last=np.nan,
                             RMSE_last_val=np.nan, R2_last_val=np.nan,
                             n_win=0, n_win_val=0,
                             note=note + f"; FAILED={type(e).__name__}: {e}"))
            print(f"[{i}/{len(jobs)}] fold{k} H{hz} {m} seed{s}: FAILED {e}")
    return pd.DataFrame(rows)


# =============================================================================
# 汇总与统计
# =============================================================================
def summarize(df, stage):
    print("\n" + "=" * 88)
    print("汇总")
    print("=" * 88)
    TABLES.mkdir(parents=True, exist_ok=True)
    per_run = df[["fold", "H", "model", "seed", "RMSE_last", "R2_last",
                  "n_win", "note"]].copy()
    per_run.to_csv(TABLES / "T_p02_walkforward.csv", index=False,
                   float_format="%.6f")
    print(f"→ {TABLES / 'T_p02_walkforward.csv'}（{len(per_run)} 行）")

    g = (df.groupby(["fold", "H", "model"])
         .agg(n_seed=("seed", "count"),
              RMSE_last_mean=("RMSE_last", "mean"),
              RMSE_last_std=("RMSE_last", "std"),
              R2_last_mean=("R2_last", "mean"),
              RMSE_last_val_mean=("RMSE_last_val", "mean"),
              RMSE_last_val_std=("RMSE_last_val", "std"),
              n_win=("n_win", "first"))
         .reset_index())
    g["note"] = "rolling-origin; 数值不与主表 T1/T2/T3 混用"
    g.to_csv(TABLES / "T_p02_walkforward_fold.csv", index=False,
             float_format="%.6f")
    print(f"→ {TABLES / 'T_p02_walkforward_fold.csv'}（{len(g)} 行）")

    # ---- (1) 跨折 RMSE 分布 ----
    print("\n【1】跨折 test RMSE_last 分布（每折 5 种子均值，再跨 6 折均值/标准差）")
    print(f"{'H':>3} {'model':>6} {'fold-means (1..6)':>52} {'跨折mean':>10} {'跨折std':>9}")
    dist_rows = []
    for hz in HORIZONS:
        for m in MODEL_ORDER:
            sub = g[(g["H"] == hz) & (g["model"] == m)].sort_values("fold")
            fm = sub["RMSE_last_mean"].values
            s = " ".join(f"{x:.4f}" for x in fm)
            print(f"{hz:>3} {m:>6} {s:>52} {np.nanmean(fm):>10.4f} {np.nanstd(fm, ddof=1):>9.4f}")
            dist_rows.append(dict(H=hz, model=m,
                                  fold_means=";".join(f"{x:.4f}" for x in fm),
                                  crossfold_mean=float(np.nanmean(fm)),
                                  crossfold_std=float(np.nanstd(fm, ddof=1))))

    # ---- (2) 最优变体选择稳定性 ----
    print("\n【2】验证段最优变体 vs 测试段最优（逐折 argmin）")
    sel_rows = []
    for hz in HORIZONS:
        n_sel_ours = 0
        for k in FOLDS:
            sub = g[(g["H"] == hz) & (g["fold"] == k)]
            if sub.empty or sub["RMSE_last_val_mean"].isna().any():
                continue
            vd = dict(zip(sub["model"], sub["RMSE_last_val_mean"]))
            td = dict(zip(sub["model"], sub["RMSE_last_mean"]))
            sel = min(vd, key=vd.get)
            win = min(td, key=td.get)
            n_sel_ours += int(sel == "Ours")
            print(f"    H={hz} fold{k}: val选={sel} (Ours={vd['Ours']:.4f}/V0={vd['V0']:.4f})"
                  f" | test更优={win} (Ours={td['Ours']:.4f}/V0={td['V0']:.4f})"
                  f" | {'一致' if sel == win else '不一致'}")
            sel_rows.append(dict(H=hz, fold=k, val_selected=sel, test_winner=win,
                                 consistent=bool(sel == win)))
        print(f"  → H={hz}: 6 折中 val 选中 Ours 的折数 = {n_sel_ours}/6")
    sel_df = pd.DataFrame(sel_rows)
    sel_df.to_csv(TABLES / "T_p02_walkforward_selection.csv", index=False)

    # ---- (3) 跨折配对检验 Ours vs V0 ----
    print("\n【3】跨折配对检验 Ours vs V0（RMSE_last，负值=Ours 更优）")
    for hz in HORIZONS:
        per_fold_p = []
        per_fold_d = []
        pooled_d = []
        for k in FOLDS:
            o = df[(df["H"] == hz) & (df["fold"] == k) & (df["model"] == "Ours")] \
                .sort_values("seed")["RMSE_last"].values
            v = df[(df["H"] == hz) & (df["fold"] == k) & (df["model"] == "V0")] \
                .sort_values("seed")["RMSE_last"].values
            if len(o) < 2 or len(o) != len(v):
                continue
            d = o - v
            t, p = stats.ttest_rel(o, v)
            per_fold_p.append(p)
            per_fold_d.append(float(d.mean()))
            pooled_d.extend(list(d))
            print(f"    H={hz} fold{k}: Δmean={d.mean():+.4f}  t={t:+.3f}  p={p:.4f}")
        per_fold_d = np.array(per_fold_d)
        pooled_d = np.array(pooled_d)
        if len(per_fold_d) >= 2:
            t1, p1 = stats.ttest_1samp(per_fold_d, 0.0)     # 以折为单元（n=6）
        else:
            t1 = p1 = float("nan")
        if len(pooled_d) >= 2:
            t2, p2 = stats.ttest_1samp(pooled_d, 0.0)       # 池化 (fold,seed)，n=30
        else:
            t2 = p2 = float("nan")
        print(f"  → H={hz}: 折级配对(以折为单元,n={len(per_fold_d)}): "
              f"Δmean={per_fold_d.mean():+.4f} t={t1:+.3f} p={p1:.4f}")
        print(f"          池化(fold,seed)配对(n={len(pooled_d)}): "
              f"Δmean={pooled_d.mean():+.4f} t={t2:+.3f} p={p2:.4f}")

    return per_run, g


# =============================================================================
# 主流程
# =============================================================================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=["single", "full"], default="single")
    ap.add_argument("--skip-anchor", action="store_true")
    args = ap.parse_args()

    if torch.cuda.is_available():
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    print(f"[env] device={AE.Config.DEVICE}  torch={torch.__version__}")

    base_cfg = AE.Config()
    base_cfg.LOOKBACK = 12
    static = build_static(base_cfg)

    anchor_ok = True
    if not args.skip_anchor:
        anchor_ok, _, _, _ = anchor_check(static, base_cfg)
        if not anchor_ok:
            print("\n*** 锚点未通过：停止，不写产物。***")
            sys.exit(1)

    df = run_stage(static, base_cfg, args.stage)
    # 单折阶段不覆盖正式汇总（其行数不完整）；仅打印
    if args.stage == "single":
        print("\n单折单种子结果（不写正式汇总，避免不完整数据落表）：")
        print(df[["fold", "H", "model", "seed", "RMSE_last", "R2_last",
                  "RMSE_last_val", "n_win"]].to_string(index=False))
        print("\n单折验证通过后可运行 --stage full 生成 120 runs 与正式汇总。")
    else:
        summarize(df, args.stage)
        print(f"\n完成：{len(df)} runs。产物根目录 {WF_ROOT}")


if __name__ == "__main__":
    main()