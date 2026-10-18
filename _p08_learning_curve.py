# -*- coding: utf-8 -*-
r"""_p08_learning_curve.py —— 学习曲线 + 模型规模扫描，排除"样本—容量失配"竞争解释。

本脚本做什么
------------
**固定数据划分（val/test 不动），只变训练数据量与模型规模**，$H=10$：
  · 训练窗档：$f\in\{10\%,25\%,50\%,75\%,100\%\}$（占训练段可用的 128 个窗）
  · 规模档：隐藏宽度 $w\in\{16,32,64,128\}$（同时设 `TCN_CHANNELS=[w,w]` 与 `GCN_HIDDEN=w`）
  · 重复：每格 $N$ 个种子（默认 5），报告 mean ± std
  · 记录 `K`（PCA 保留成分数）、训练时间戳/窗数、参数量、末步 RMSE、零变化、skill

判读
----
· skill 随训练数据量与参数量**均趋于饱和且仍为负** ⇒ **排除**"样本—容量失配"解释；
· skill 随任一维度**显著上升并趋近 0/转正** ⇒ 该维度是当前的真实瓶颈，须如实报告。

为什么必须写"忠实副本"而不是直接用 `AB.prepare_data`
--------------------------------------------------
`AB.prepare_data` 把三段去噪序列 `vstack` 后用**原始绝对索引**回索差分
（`diff1_den[train_idx]`）。缩减训练段会让 `all_den` 行数与索引不再对齐（越界/错位），
故此处用一份**逐行忠实副本** `prepare_data_reduced`：改为按**段内相对位置**回索，
在 $f=100\%$（训练段 = 完整 $[0,149)$）时与原实现**逐位等价**（脚本启动时自检断言）。

同理，**标准化与 PCA 只用缩减后的训练段拟合**——若先在全训练段拟合再切窗，会把
"看到更多数据"的泄漏混进学习曲线，曲线将不可解释。

铁律
----
· 不覆盖旧结果：写入新建目录 `<WORK_ROOT>\Stage_LearningCurve\`；
  表写 `_paper_tables\T_p08_learning_curve.csv`（逐 run）与 `T_p08_lc_summary.csv`（逐格）。
· 不改训练配方：损失 / 优化器 / 早停 / LOOKBACK / 划分 val[149,181) test[181,213) 全沿用。
· 锚点自检：(f=100%, w=64) 应与锁定管线**逐种子**一致（对照权威表容差 1e-6）。

运行（PowerShell）
-----------------
  $py = "python"
  & $py -u _p08_learning_curve.py --smoke    # 冒烟：1 格 / 1 种子 / 5 epoch
  & $py -u _p08_learning_curve.py --probe    # 锚点：f=100%,w=64 / 3 种子
  & $py -u _p08_learning_curve.py            # 正式：5 档 × 4 规模 × 5 种子 = 100 次
  续跑：$env:P08_RESUME="1"   自定义：--fractions 0.25,1.0 --widths 32,64,128 --seeds 42,15,77
"""
from __future__ import annotations

from _env import AUX_ROOT, WORK_ROOT

import argparse
import copy
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset

PROJ = WORK_ROOT
sys.path.insert(0, str(PROJ))

import Ablation_Experiment as AB          # noqa: E402

HZ = 10
FRACTIONS = [0.10, 0.25, 0.50, 0.75, 1.00]
WIDTHS = [16, 32, 64, 128]
OUT_ROOT = PROJ / "Stage_LearningCurve"
TABLES = PROJ / "_paper_tables"
TABLES.mkdir(parents=True, exist_ok=True)

RESUME = os.environ.get("P08_RESUME", "0") == "1"
VARIANT_OF = {"ours": "V7_wo_VerticalGCN", "v0": "V0_Pure_TCN"}
VAR_LABEL = {"V7_wo_VerticalGCN": "Ours(V7)", "V0_Pure_TCN": "TCN-Only(V0)"}

# 锚点：权威逐种子表（与 _p03 共用）
def _resolve_summary_csv(d):
    """取该 summary 目录下种子数最多的 global_*runs.csv（真实=30，示例可能更少）。"""
    cand = sorted(Path(d).glob("global_*runs.csv"))
    if not cand:
        return Path(d) / "global_30runs.csv"
    return max(cand, key=lambda p: len(p.read_text(encoding="utf-8-sig").splitlines()))


AUTH_TABLE = _resolve_summary_csv(
    AUX_ROOT / "Ablation_New_5d_mid/H10/V7_wo_VerticalGCN/summary")
ZERO_H10 = 1.372010
PER_SEED_TOL = 1e-6


# -----------------------------------------------------------------------------
# 数据缓存与划分
# -----------------------------------------------------------------------------
def load_cache():
    cfg = AB.Config()
    AB.set_seed(42)
    full_df = AB.load_raw_data_with_meta(cfg)
    ts = np.sort(full_df["测量时间"].unique())
    wd, wa = AB.load_water_data(ts, cfg)
    disp_raw, _, meta = AB.build_node_features(full_df, cfg)
    geo = AB.load_geology_params(cfg)
    geo_feat_raw = AB.assign_geo_to_nodes(geo, meta["depths"], cfg) if geo else None
    return dict(cfg=cfg, full_df=full_df, ts=ts, wd=wd, wa=wa,
                water_feat=AB.extract_water_features(wd, wa),
                disp_raw=disp_raw, meta=meta, geo_feat_raw=geo_feat_raw)


def base_split(n):
    """锁定划分（连续区间）：train [0,149) / val [149,181) / test [181,213)。"""
    t0, t1 = int(n * AB.Config.TRAIN_RATIO), int(n * (AB.Config.TRAIN_RATIO
                                                       + AB.Config.VAL_RATIO))
    return np.arange(0, t0), np.arange(t0, t1), np.arange(t1, n)


def train_prefix(tr, va, te, hz, frac):
    """按"训练窗占位比例"截取训练段前缀，val/test 保持不变。

    可用训练窗数 = len(tr) - LOOKBACK - hz + 1（f=1.0 时为 128）。
    k = max(1, round(frac * 可用窗数))；所需时间戳 = k + LOOKBACK + hz - 1，再截到 len(tr)。
    """
    avail = len(tr) - AB.Config.LOOKBACK - hz + 1
    k = max(1, int(round(frac * avail)))
    n_need = min(len(tr), k + AB.Config.LOOKBACK + hz - 1)
    return np.arange(0, n_need), va, te


# -----------------------------------------------------------------------------
# AB.prepare_data 的忠实副本：三段按"段内相对位置"回索差分
# -----------------------------------------------------------------------------
def prepare_data_reduced(cfg, full_df, wd, wa, meta, train_idx, val_idx, test_idx,
                         water_feat, geo_feat, hz, disp_raw=None):
    """与 `AB.prepare_data` 逐行等价，唯一差别：差分按**段内相对位置**回索。

    在 train_idx=[0,149) / val=[149,181) / test=[181,213) 时，段内相对位与绝对索引
    完全一致 ⇒ 与原实现**逐位相同**（启动自检断言）。缩减训练段时仍保持 val/test 不动。
    """
    if disp_raw is None:
        disp_raw, _, _ = AB.build_node_features(full_df, cfg)
    valid_mask = meta.get("valid_mask", np.ones(disp_raw.shape[1], dtype=np.float32))
    train_disp, val_disp, test_disp = disp_raw[train_idx], disp_raw[val_idx], disp_raw[test_idx]
    train_den, val_den, test_den = AB.denoise_split(train_disp, val_disp, test_disp, cfg)
    all_den = np.vstack([train_den, val_den, test_den])
    diff1_den = np.zeros_like(all_den)
    if all_den.shape[0] > 1:
        diff1_den[1:] = np.diff(all_den, axis=0)
    k0, k1 = 0, len(train_idx)
    k2 = k1 + len(val_idx)
    sl_tr = slice(k0, k0 + len(train_idx))
    sl_va = slice(k1, k1 + len(val_idx))
    sl_te = slice(k2, k2 + len(test_idx))

    geo_static_raw = np.concatenate([geo_feat, (meta["depths"] / 16.0).reshape(-1, 1)],
                                    axis=1)
    geo_static = (geo_static_raw - geo_static_raw.mean(0)) / (
        geo_static_raw.std(0) + 1e-6)
    train_feat = AB.make_features(train_den, diff1_den[sl_tr], meta,
                                  water_feat[train_idx], geo_static, cfg, False)
    val_feat = AB.make_features(val_den, diff1_den[sl_va], meta,
                                water_feat[val_idx], geo_static, cfg, False)
    test_feat = AB.make_features(test_den, diff1_den[sl_te], meta,
                                 water_feat[test_idx], geo_static, cfg, False)
    F = train_feat.shape[-1]
    y_train, y_val, y_test = diff1_den[sl_tr], diff1_den[sl_va], diff1_den[sl_te]
    scX = AB.StandardScaler().fit(train_feat.reshape(-1, F))
    tr_norm = scX.transform(train_feat.reshape(-1, F)).reshape(train_feat.shape)
    va_norm = scX.transform(val_feat.reshape(-1, F)).reshape(val_feat.shape)
    te_norm = scX.transform(test_feat.reshape(-1, F)).reshape(test_feat.shape)
    # 标准化与 PCA **只用缩减后的训练段**拟合（否则学习曲线含泄漏）
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
    N = train_disp.shape[1]
    w_mean, w_std = water_feat[train_idx].mean(0), water_feat[train_idx].std(0) + 1e-6
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


def selfcheck_copy(cache, tr, va, te, hz):
    """断言：`prepare_data_reduced` 在完整训练段上与 `AB.prepare_data` 逐位等价。"""
    cfg, meta, geo = cache["cfg"], cache["meta"], cache["geo_feat_raw"]
    geo_fb = geo if geo is not None else np.zeros(
        (len(meta["depths"]), cfg.GEO_FEAT_DIM), dtype=np.float32)
    args = (cache["full_df"], cache["wd"], cache["wa"], meta, tr, va, te,
            cache["water_feat"], geo_fb, hz)
    d_ref = AB.prepare_data(cfg, *args, disp_raw=cache["disp_raw"])
    d_new = prepare_data_reduced(cfg, *args, disp_raw=cache["disp_raw"])
    md = 0.0
    for k in ("Xtr", "ytr", "Xva", "yva", "Xte", "yte", "yte_raw", "init_test",
              "wf_train", "wf_val", "wf_test", "geo_static"):
        a, b = np.asarray(d_ref[k], np.float64), np.asarray(d_new[k], np.float64)
        md = max(md, float(np.max(np.abs(a - b))) if a.size else 0.0)
    pc = max(float(np.max(np.abs(d_ref["pca_y"].components_
                                 - d_new["pca_y"].components_))), 0.0)
    print(f"[自检] 忠实副本 vs AB.prepare_data：数组最大偏差 = {md:.3e}；"
          f"PCA 载荷最大偏差 = {pc:.3e}；K = {d_ref['out_dim']} vs {d_new['out_dim']}")
    if md > 1e-6 or pc > 1e-9 or d_ref["out_dim"] != d_new["out_dim"]:
        raise SystemExit("[自检失败] 副本与原实现不等价，停止（不得据此出结论）。")
    print("[自检] 通过 —— 副本在完整训练段上与锁定实现逐位等价。")


# -----------------------------------------------------------------------------
# 单次训练
# -----------------------------------------------------------------------------
def run_one(base_cfg, variant, width, cache, tr, va, te, hz, seed, out_dir):
    cfg = copy.deepcopy(base_cfg)
    for k, v in AB.ABLATION_VARIANTS[variant].items():
        setattr(cfg, k, v)
    cfg.TCN_CHANNELS = [width, width]
    cfg.GCN_HIDDEN = width

    run_dir = out_dir / f"run_{seed}"
    run_dir.mkdir(parents=True, exist_ok=True)
    mfile = run_dir / "metrics.json"
    if RESUME and mfile.exists():
        return json.load(open(mfile, encoding="utf-8"))

    meta, geo = cache["meta"], cache["geo_feat_raw"]
    geo_fb = geo if geo is not None else np.zeros(
        (len(meta["depths"]), cfg.GEO_FEAT_DIM), dtype=np.float32)
    data = prepare_data_reduced(cfg, cache["full_df"], cache["wd"], cache["wa"], meta,
                                tr, va, te, cache["water_feat"], geo_fb, hz,
                                disp_raw=cache["disp_raw"])
    N, F_in = data["N"], data["F_in"]
    adj = AB.build_adjacency(meta, geo_fb, cfg,
                             zero_same_hole=getattr(cfg, "ZERO_SAME_HOLE", False))
    vert = AB.build_vertical_adjacency(meta, cfg)
    adj_t = torch.FloatTensor(adj).to(cfg.DEVICE)
    vert_t = torch.FloatTensor(vert).to(cfg.DEVICE)
    geo_t = torch.FloatTensor(data["geo_static"]).to(cfg.DEVICE)
    mask_t = torch.FloatTensor(data["valid_mask"]).to(cfg.DEVICE)

    def ld(X, y, wf, sh):
        return DataLoader(TensorDataset(
            torch.FloatTensor(X), torch.FloatTensor(y), torch.FloatTensor(wf),
            geo_t.unsqueeze(0).expand(X.shape[0], N, -1)),
            batch_size=cfg.BATCH_SIZE, shuffle=sh)

    ld_tr = ld(data["Xtr"], data["ytr"], data["wf_train"], True)
    ld_va = ld(data["Xva"], data["yva"], data["wf_val"], False)
    ld_te = ld(data["Xte"], data["yte"], data["wf_test"], False)

    AB.set_seed(seed)
    model = AB.AblationModel(cfg, N=N, in_dim=F_in, gcn_hid=cfg.GCN_HIDDEN,
                             tcn_ch=cfg.TCN_CHANNELS, k=cfg.TCN_KERNEL_SIZE,
                             dil=cfg.TCN_DILATIONS, attn_hid=cfg.ATTENTION_HIDDEN,
                             hor=hz, adj=adj_t, out_dim=data["out_dim"],
                             geo_feat_dim=data["geo_dim_actual"],
                             vert_adj=vert_t).to(cfg.DEVICE)
    n_par = int(AB.count_params(model))
    # 复现锁定脚本的 RNG 消耗（FLOPs 估计会跑一次 train 态 forward）
    AB.estimate_flops(model, cfg, N, F_in, hz)
    model, _, _, _ = AB.train_model(cfg, model, ld_tr, ld_va,
                                    run_dir / "best.pth", node_mask=mask_t)
    pred, true = AB.evaluate(model, ld_te, data["init_test"], data["scY"],
                             data["pca_y"], hz, cfg.DEVICE, node_mask=mask_t,
                             true_diff_win=data["yte_raw"])
    gm = AB.global_metrics(true, pred)
    rmse = float(gm["RMSE_last"])
    zero = float(np.sqrt(np.mean((data["init_test"] - true[:, -1, :]) ** 2)))
    rec = dict(variant=variant, model=VAR_LABEL.get(variant, variant),
               width=int(width), K=int(data["out_dim"]), seed=int(seed),
               n_train_ts=int(len(tr)), n_train_win=int(data["Xtr"].shape[0]),
               n_val_win=int(data["Xva"].shape[0]), n_test_win=int(pred.shape[0]),
               params=n_par, RMSE_last=rmse, R2_last=float(gm["R2_last"]),
               RMSE_zero_last=zero, skill=1.0 - (rmse / zero) ** 2)
    json.dump(rec, open(mfile, "w", encoding="utf-8"), indent=2)
    np.savez(run_dir / "predictions.npz", pred_cum=pred, true_cum=true)
    try:
        (run_dir / "best.pth").unlink()
    except OSError:
        pass
    return rec


# -----------------------------------------------------------------------------
# 汇总与判读
# -----------------------------------------------------------------------------
def summarize(df):
    g = df.groupby(["variant", "width", "fraction"], as_index=False).agg(
        n_seeds=("seed", "size"),
        K=("K", "first"),
        n_train_ts=("n_train_ts", "first"), n_train_win=("n_train_win", "first"),
        params=("params", "first"),
        RMSE_last_mean=("RMSE_last", "mean"), RMSE_last_std=("RMSE_last", "std"),
        skill_mean=("skill", "mean"), skill_std=("skill", "std"),
        RMSE_zero_mean=("RMSE_zero_last", "mean"))
    return g.sort_values(["variant", "fraction", "width"])


def print_verdict(sm):
    print("\n" + "=" * 104)
    print("判读：skill 随训练数据量 / 模型规模的变化")
    print("=" * 104)
    for v, g in sm.groupby("variant", sort=True):
        print(f"\n【{VAR_LABEL.get(v, v)}】  (H={HZ}, 末步口径, 零变化 "
              f"{g.RMSE_zero_mean.iloc[0]:.6f})")
        print(f"{'训练窗':>7}{'时间戳':>8}{'宽度':>6}{'参数':>9}{'K':>4}"
              f"{'RMSE_last':>18}{'skill':>10}{'skill std':>11}")
        for _, r in g.iterrows():
            print(f"{int(r.n_train_win):>7}{int(r.n_train_ts):>8}{int(r.width):>6}"
                  f"{int(r.params):>9}{int(r.K):>4}"
                  f"{r.RMSE_last_mean:>11.4f}±{r.RMSE_last_std:.4f}"
                  f"{r.skill_mean:>+10.4f}{r.skill_std:>11.4f}")
        # 两个方向的最大改善量（以最差格为基准）
        s_min = g.skill_mean.min()
        by_win = g.groupby("n_train_win").skill_mean.mean()
        by_w = g.groupby("width").skill_mean.mean()
        print(f"  数据量方向：skill 均值 {by_win.iloc[0]:+.4f}（{int(by_win.index[0])} 窗）"
              f" → {by_win.iloc[-1]:+.4f}（{int(by_win.index[-1])} 窗），改善 "
              f"{by_win.iloc[-1] - by_win.iloc[0]:+.4f}")
        print(f"  规模方向  ：skill 均值 {by_w.iloc[0]:+.4f}（w={int(by_w.index[0])}）"
              f" → {by_w.iloc[-1]:+.4f}（w={int(by_w.index[-1])}），改善 "
              f"{by_w.iloc[-1] - by_w.iloc[0]:+.4f}")
        best = g.loc[g.skill_mean.idxmax()]
        print(f"  最优格：{int(best.n_train_win)} 窗 / w={int(best.width)} / "
              f"参数量 {int(best.params)} → skill {best.skill_mean:+.4f}"
              f"（仍{'为负' if best.skill_mean < 0 else '为正'}）")
    print("-" * 104)
    print("判读准则：两个方向的改善量都接近 0 且最优 skill 仍为负 ⇒ 排除"
          "'样本—容量失配'竞争解释；若某方向改善显著并趋近 0/转正 ⇒ 该方向是真实瓶颈。")


def check_anchor(df):
    a = df[(df.variant == VARIANT_OF["ours"]) & (df.n_train_ts == 149)
           & (df.width == 64)]
    if a.empty:
        return
    print("\n" + "-" * 104)
    print(f"[锚点自检] f=100% · w=64 · Ours(V7) · H={HZ} · n={len(a)} 种子")
    z = (a.RMSE_zero_last - ZERO_H10).abs().max()
    print(f"  零变化与期望 {ZERO_H10} 的最大偏差 = {z:.2e}   "
          f"{'[OK]' if z <= 1e-3 else '[WARN]'}")
    if not AUTH_TABLE.exists():
        print(f"  [WARN] 未找到权威逐种子表：{AUTH_TABLE}（跳过逐种子核验）")
        print("-" * 104)
        return
    ref = pd.read_csv(AUTH_TABLE).set_index("seed")["RMSE_last"]
    m = a.set_index("seed")["RMSE_last"]
    common = m.index.intersection(ref.index)
    diff = (m[common] - ref[common]).abs()
    for s in common:
        print(f"    seed={s:<5} 本批 {m[s]:.6f}  权威 {ref[s]:.6f}  Δ={m[s] - ref[s]:+.2e}")
    ok = (len(common) == len(m)) and float(diff.max()) <= PER_SEED_TOL
    print(f"  匹配 {len(common)}/{len(m)}；最大偏差 {float(diff.max()):.2e}  => "
          f"{'[OK] 逐位复现锁定管线' if ok else '[WARN] 存在偏差，本批结论不可用'}")
    print("-" * 104)


# -----------------------------------------------------------------------------
# main
# -----------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hz", type=int, default=HZ)
    ap.add_argument("--fractions", type=str, default=",".join(map(str, FRACTIONS)))
    ap.add_argument("--widths", type=str, default=",".join(map(str, WIDTHS)))
    ap.add_argument("--models", type=str, default="ours")
    ap.add_argument("--seeds", type=str, default=None)
    ap.add_argument("--smoke", action="store_true", help="1 格 / 1 种子 / 5 epoch")
    ap.add_argument("--probe", action="store_true",
                    help="只跑 f=100%, w=64（锚点自检）")
    args = ap.parse_args()

    fracs = [float(x) for x in args.fractions.split(",") if x.strip()]
    widths = [int(x) for x in args.widths.split(",") if x.strip()]
    model_keys = [m.strip() for m in args.models.split(",") if m.strip()]
    seeds = ([int(s) for s in args.seeds.split(",")] if args.seeds
             else list(AB.Config.SEEDS)[:5])
    epochs = None
    if args.smoke:
        fracs, widths, seeds, epochs = [1.0], [64], seeds[:1], 5
    if args.probe:
        fracs, widths, seeds = [1.0], [64], seeds[:3]

    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    print("=" * 104)
    print(f"学习曲线 + 规模扫描 | H={args.hz} | 训练窗档={fracs} | "
          f"规模档={widths} | 模型={model_keys} | 种子数={len(seeds)}")
    print(f"  输出目录：{OUT_ROOT}（新建，不覆盖旧结果）")
    print("=" * 104)

    cache = load_cache()
    n = len(cache["ts"])
    tr_full, va, te = base_split(n)
    print(f"  时间戳 n={n}；锁定划分 train/val/test = {len(tr_full)}/{len(va)}/{len(te)}"
          f"（val [{va[0]},{va[-1]}]、test [{te[0]},{te[-1]}] 全程不动）")
    selfcheck_copy(cache, tr_full, va, te, args.hz)

    print("\n[准备] 训练窗档 → 时间戳数 / 实际训练窗数")
    print(f"{'fraction':>10}{'时间戳':>9}{'训练窗':>9}")
    plan = {}
    for f in fracs:
        trf, _, _ = train_prefix(tr_full, va, te, args.hz, f)
        nwin = len(trf) - AB.Config.LOOKBACK - args.hz + 1
        plan[f] = trf
        print(f"{f:>10.2f}{len(trf):>9}{nwin:>9}")

    rows = []
    for m in model_keys:
        variant = VARIANT_OF[m]
        for f in fracs:
            trf = plan[f]
            for w in widths:
                out_dir = OUT_ROOT / f"{m}_f{int(round(f * 100))}_w{w}"
                out_dir.mkdir(parents=True, exist_ok=True)
                print("\n" + "=" * 104)
                print(f"[{VAR_LABEL.get(variant, m)}] f={f:.2f}（{len(trf)} 时间戳）· "
                      f"w={w} · {len(seeds)} 种子 → {out_dir}")
                print("=" * 104)
                for si, seed in enumerate(seeds, 1):
                    t0 = time.time()
                    rec = run_one(cache["cfg"], variant, w, cache, trf, va, te,
                                  args.hz, seed, out_dir)
                    rec["fraction"] = float(f)
                    rows.append(rec)
                    print(f"  [{si:>2}/{len(seeds)}] seed={seed:<5} K={rec['K']:<3} "
                          f"params={rec['params']:<7} RMSE_last={rec['RMSE_last']:.4f} "
                          f"skill={rec['skill']:+.4f}  ({time.time() - t0:.1f}s)")

    df = pd.DataFrame(rows)
    df_out = TABLES / "T_p08_learning_curve.csv"
    df.round(8).to_csv(df_out, index=False, encoding="utf-8-sig")
    sm = summarize(df)
    sm_out = TABLES / "T_p08_lc_summary.csv"
    sm.round(8).to_csv(sm_out, index=False, encoding="utf-8-sig")

    if len(seeds) >= 2 and not args.smoke:
        print_verdict(sm)
        check_anchor(df)
    print(f"\n[已写出] {df_out}（逐 run）\n[已写出] {sm_out}（逐格汇总）")


if __name__ == "__main__":
    main()