# -*- coding: utf-8 -*-
r"""_p06_stage2_nestedcv.py — 嵌套 CV + 损失权重档（重训管线）。

目的
----
阶段 1（`_p06_selection_check.py`，零训练前向）已在单一 val 段 [149,181) 上证明：
把「架构 / PCA 维数 K」的选择限定在验证集内会得到 **V5_Full_Model 而非正文的
V7（Ours）**，K 也未选中 13（val 选择 ≠ test 选择，方向翻转）。
本脚本是阶段 2：**在训练段内做时间前向 walk-forward 嵌套 CV 并真正重训**，
把"验证选择"从单一固定段推广到多段滚动，检验 V5 / K≠13 的偏好是否**跨折稳定**，
并补上阶段 1 未能做的 **损失权重档**（无既有候选权重，必须重训）。

设计（2026-10-06 定案，用户已确认三项决策）
--------------------------------------------
· 候选集 = **架构 × K 网格**（默认 2 架构 × 3 K = 6 候选）：
    V7_wo_VerticalGCN / V5_Full_Model × K ∈ {7, 13, 21}
  —— 损失权重档做成 `--weight-tiers` 开关（默认关闭，只训锁定配方 1.0/0.5/0.2）；
    开启时并入候选（锁定/均匀 1/1/1/主项 1/0/0），候选数 ×3。
· 折结构 = **训练段内时间前向 walk-forward**：
    外层折只使用 [0, 181)（锁定 train+val）段数据，每折 train=前缀、val=紧邻块
    （宽 30，可产生 9 个验证窗）、test 槽 = 剩余段（不评估）；锁定测试段
    [181,213) 全程不参与选择。
· 终评 = **完整 train+val [0,181) 重训选定配置 → 测试段 [181,213) 评估一次**
    （符合"测试只评一次"铁律）。

复用接口（全部 import 自锁定主文件 `Ablation_Experiment.py`，**不改主文件一字**）
-----------------------------------------------------------------------------
· `AB.prepare_data` 的逐行忠实副本 `prepare_wf`（差分按段内相对位置回索，
  与 `_p08_learning_curve.prepare_data_reduced` 同源；完整划分下逐位等价）。
· 固定 K：monkey-patch `AB.PCA`（`_p03_config_capacity._fixed_pca` 同款），
  在 `AB.PCA(0.90)` 的方差阈值写法上强制成分数 = 候选 K。
· 损失权重档：`train_model(..., criterion=TieredLoss(...))` 注入（主文件 L915 已支持
  criterion 参数，Spatial_LOOCV 即用此法，未改主文件）。
· 训练配方（损失之外的优化器 / 早停 / LOOKBACK / 种子）全沿用锁定 Config。

锚点自检（`--probe`）
--------------------
完整锁定划分 [0,149)/[149,181)/[181,213) + Ours(V7) + K=13 + 锁定权重，
应逐种子复现权威表 `Ablation_New_5d_mid\H10\V7_wo_VerticalGCN\summary\global_30runs.csv`
（容差 1e-6）。K=13 时 PCA(0.90) 数据本身落在 13 成分 → 补丁恒等，锚点必须命中。

规模
----
正式默认：5 折 × 6 候选 × 30 种子 = **900 次训练**（+ 终评 30 次）。
`--weight-tiers` 开启：5 折 × 18 候选 × 30 种子 = 2700 次（重，建议先 probe）。
`--smoke`：1 折 / 1 候选 / 1 种子 / 5 epoch；`--probe`：锚点 3 种子。

运行（PowerShell）
---------------------------------
  $py = "python"
  & $py -u _p06_stage2_nestedcv.py --smoke        # 冒烟（管线可用性）
  & $py -u _p06_stage2_nestedcv.py --probe        # 锚点自检（Ours/K13 逐位复现）
  & $py -u _p06_stage2_nestedcv.py                # 正式 900 次（5 折）
  续跑：$env:P06_RESUME="1"
  选项：--folds 5 --seeds 42,15,77 --weight-tiers --eval-all-final

输出
----
`Stage_P06_NestedCV\`（新建，不覆盖旧结果）
  fold_runs.csv         逐（折 × 候选 × 种子）val 评估明细
  fold_selection.csv    每折选中的最优候选（val RMSE_last 跨种子均值最小）
  final_test.csv        终评：选定配置在测试段的 30 种子 RMSE_last / skill
  anchor_check.csv      `--probe` 锚点逐种子核验
`_paper_tables\T_p06_stage2_{fold,selection,final,anchor}.csv`（同步副本）
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
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

PROJ = WORK_ROOT
sys.path.insert(0, str(PROJ))

import Ablation_Experiment as AB          # noqa: E402

# -----------------------------------------------------------------------------
# 全局口径
# -----------------------------------------------------------------------------
HZ_DEFAULT = 10
OUT_ROOT = PROJ / "Stage_P06_NestedCV"
SMOKE_ROOT = PROJ / "Stage_P06_NestedCV_smoke"   # 冒烟输出分离，避免 RESUME 误跳
TABLES = PROJ / "_paper_tables"
TABLES.mkdir(parents=True, exist_ok=True)

RESUME = os.environ.get("P06_RESUME", "0") == "1"

# 候选网格
ARCHES = ["V7_wo_VerticalGCN", "V5_Full_Model"]
K_GRID = [7, 13, 21]
VAR_LABEL = {"V7_wo_VerticalGCN": "Ours(V7)", "V5_Full_Model": "V5_Full"}
# 损失权重档（--weight-tiers 时并入候选）
WEIGHT_TIERS = {
    "locked": (1.0, 0.5, 0.2),   # 锁定配方（论文口径）
    "uniform": (1.0, 1.0, 1.0),  # 均匀
    "main": (1.0, 0.0, 0.0),     # 仅 Huber 主项
}
DEFAULT_TIER = "locked"

# 锚点：权威逐种子表 + 期望值（与 _p03 / _p08 共用）
def _resolve_summary_csv(d):
    """取该 summary 目录下种子数最多的 global_*runs.csv（真实=30，示例可能更少）。"""
    cand = sorted(Path(d).glob("global_*runs.csv"))
    if not cand:
        return Path(d) / "global_30runs.csv"
    return max(cand, key=lambda p: len(p.read_text(encoding="utf-8-sig").splitlines()))


AUTH_TABLE = _resolve_summary_csv(
    AUX_ROOT / "Ablation_New_5d_mid/H10/V7_wo_VerticalGCN/summary")
ZERO_H10 = 1.372010
ANCHOR_RMSE, ANCHOR_STD = 1.5687, 0.0940
PER_SEED_TOL = 1e-6


# -----------------------------------------------------------------------------
# 固定 K 补丁（必须在任何 prepare_* 调用之前生效；AB.PCA 被替换）
# -----------------------------------------------------------------------------
_ORIG_PCA = AB.PCA
_K_FIXED = 13          # 由候选 K 逐 run 设置


def _fixed_pca(n_components=None, **kw):
    if isinstance(n_components, float) and 0.0 < n_components < 1.0:
        n_components = _K_FIXED
    return _ORIG_PCA(n_components=n_components, **kw)


def _apply_pca_patch(k_fixed):
    """把 AB.PCA 替换为固定成分数版本，K_FIXED 同步为 k_fixed。"""
    global _K_FIXED
    _K_FIXED = int(k_fixed)
    AB.PCA = _fixed_pca


# -----------------------------------------------------------------------------
# 数据缓存（与 _p08 同款，零训练加载）
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


def walkforward_folds(n_ts, n_folds, val_w=30):
    """时间前向 walk-forward（训练段 [0,181) 内）。

    折 i：train = [0, a_i)，val = [a_i, a_i+val_w)（末折截到 n_ts），
          test 槽 = [a_i+val_w, n_ts)（不评估，仅为满足 prepare 三段签名）。
    a 在 [n_ts - val_w*n_folds, n_ts - val_w] 间等步长。
    每折 val 宽 30 → 验证窗 = 30 - LOOKBACK - hz + 1 = 9 个。
    """
    if n_folds == 1:
        a0 = max(60, n_ts - val_w)
        return [(np.arange(0, a0), np.arange(a0, n_ts), np.arange(n_ts, n_ts))]
    e0 = n_ts - val_w * n_folds
    if e0 < 60:
        e0 = 60
    starts = np.linspace(e0, n_ts - val_w, n_folds).astype(int)
    folds = []
    for i, a in enumerate(starts):
        b = n_ts if i == n_folds - 1 else a + val_w
        folds.append((np.arange(0, a), np.arange(a, b), np.arange(b, n_ts)))
    return folds


# -----------------------------------------------------------------------------
# AB.prepare_data 的忠实副本（差分按段内相对位置回索）+ val 评估所需的额外字段
# -----------------------------------------------------------------------------
def prepare_wf(cfg, full_df, wd, wa, meta, train_idx, val_idx, test_idx,
               water_feat, geo_feat, hz, disp_raw=None):
    """与 `AB.prepare_data` 逐行等价；在完整锁定划分下逐位相同。

    唯一差别：三段差分按**段内相对位置**回索（而非绝对索引），使 walk-forward
    折（train=前缀、val=紧邻块）也能正确构造。额外返回 val 段的
    `init_val` / `yval_raw`，供折内验证评估（与 test 段对称）。
    """
    if disp_raw is None:
        disp_raw, _, _ = AB.build_node_features(full_df, cfg)
    valid_mask = meta.get("valid_mask", np.ones(disp_raw.shape[1], dtype=np.float32))
    train_disp, val_disp, test_disp = disp_raw[train_idx], disp_raw[val_idx], disp_raw[test_idx]
    # 末折 walk-forward 的 test 槽为空（时间上已无更后数据）——占位空段即可：
    # denoise_split 无法处理空段（log2(0) 溢出），用 val 占位跑通后丢弃 test 输出。
    if len(test_disp):
        train_den, val_den, test_den = AB.denoise_split(train_disp, val_disp, test_disp, cfg)
    else:
        train_den, val_den, _ = AB.denoise_split(train_disp, val_disp, val_disp, cfg)
        test_den = np.zeros((0, val_disp.shape[1]), dtype=np.float32)
    all_den = np.vstack([train_den, val_den, test_den])
    diff1_den = np.zeros_like(all_den)
    if all_den.shape[0] > 1:
        diff1_den[1:] = np.diff(all_den, axis=0)
    k0, k1, k2 = 0, len(train_idx), len(train_idx) + len(val_idx)
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
    te_norm = (scX.transform(test_feat.reshape(-1, F)).reshape(test_feat.shape)
               if len(test_feat) else np.zeros((0,) + test_feat.shape[1:],
                                               dtype=np.float32))
    # 标准化与 PCA 只用本折训练段拟合（避免选择泄漏）
    pca_y = AB.PCA(0.90).fit(y_train)
    out_dim = pca_y.n_components_
    scY = AB.StandardScaler().fit(pca_y.transform(y_train))
    ytr_norm = scY.transform(pca_y.transform(y_train))
    yva_norm = scY.transform(pca_y.transform(y_val))
    yte_norm = (scY.transform(pca_y.transform(y_test))
                if len(y_test) else np.zeros((0, out_dim), dtype=np.float32))

    def make_samples(feat, tgt, lb, hval):
        X, y = [], []
        for i in range(len(feat) - lb - hval + 1):
            X.append(feat[i:i + lb])
            y.append(tgt[i + lb:i + lb + hval])
        return np.array(X, dtype=np.float32), np.array(y, dtype=np.float32)

    Xtr, ytr = make_samples(tr_norm, ytr_norm, cfg.LOOKBACK, hz)
    Xva, yva = make_samples(va_norm, yva_norm, cfg.LOOKBACK, hz)
    Xte, yte = make_samples(te_norm, yte_norm, cfg.LOOKBACK, hz)

    def raw_wins(den_seg, diff_seg, n_win):
        """与 AB.prepare_data 的 yte_raw / init_test 同构（测试段），此处给 val 段。"""
        wins = np.stack([diff_seg[i + cfg.LOOKBACK:i + cfg.LOOKBACK + hz]
                         for i in range(n_win)]) if n_win else np.empty(
            (0, hz, diff_seg.shape[1]), dtype=np.float32)
        init = den_seg[cfg.LOOKBACK - 1:-hz][:n_win]
        return wins, init

    n_va = Xva.shape[0]
    yval_raw, init_val = raw_wins(val_den, diff1_den[sl_va], n_va)
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
            "yte_raw": yte_raw, "init_test": init_test,
            "yval_raw": yval_raw, "init_val": init_val,
            "scX": scX, "scY": scY, "pca_y": pca_y, "out_dim": out_dim,
            "F_in": F, "N": N,
            "wf_train": wf_train, "wf_val": wf_val, "wf_test": wf_test,
            "geo_static": geo_static, "geo_dim_actual": geo_static.shape[1],
            "valid_mask": valid_mask}


# -----------------------------------------------------------------------------
# 损失权重档（criterion 注入；train_model 的 criterion 参数已由主文件支持）
# -----------------------------------------------------------------------------
class TieredLoss(nn.Module):
    def __init__(self, w_hub=1.0, w_diff=0.5, w_freq=0.2):
        super().__init__()
        self.huber = nn.HuberLoss(delta=1.0)
        self.mse = nn.MSELoss()
        self.w = (float(w_hub), float(w_diff), float(w_freq))

    def forward(self, pred, target, epoch=None):
        B, hor, dim = pred.shape
        l_hub = self.huber(pred, target)
        l_diff = torch.tensor(0., device=pred.device)
        if hor > 1:
            l_diff = self.mse(pred[:, 1:] - pred[:, :-1], target[:, 1:] - target[:, :-1])
        pred_fft = torch.fft.rfft(pred, dim=1)
        target_fft = torch.fft.rfft(target, dim=1)
        l_freq = self.mse(torch.abs(pred_fft), torch.abs(target_fft))
        return self.w[0] * l_hub + self.w[1] * l_diff + self.w[2] * l_freq


# -----------------------------------------------------------------------------
# 单次训练 + val / test 评估
# -----------------------------------------------------------------------------
def _build_loader(cfg, data, Xk, yk, wfk, geo_t, shuffle):
    N = data["N"]
    return DataLoader(TensorDataset(
        torch.FloatTensor(Xk), torch.FloatTensor(yk), torch.FloatTensor(wfk),
        geo_t.unsqueeze(0).expand(Xk.shape[0], N, -1)),
        batch_size=cfg.BATCH_SIZE, shuffle=shuffle)


def run_cell(cfg0, cache, variant, k_fixed, tier, tr, va, te, hz, seed, out_dir,
             eval_mode):
    """训练单个（候选 × 种子）格，返回评估明细。

    eval_mode: "val"（折内验证，评 val 段，窗口数少）或 "final"（完整
    train+val 重训，评锁定 test 段）。锁定划分下 eval_mode="val" 即评 [149,181)。
    """
    cfg = copy.deepcopy(cfg0)
    for kk, vv in AB.ABLATION_VARIANTS[variant].items():
        setattr(cfg, kk, vv)
    _apply_pca_patch(k_fixed)

    run_dir = out_dir / f"run_{seed}"
    run_dir.mkdir(parents=True, exist_ok=True)
    mfile = run_dir / "metrics.json"
    if RESUME and mfile.exists():
        return json.load(open(mfile, encoding="utf-8"))

    meta, geo = cache["meta"], cache["geo_feat_raw"]
    geo_fb = geo if geo is not None else np.zeros(
        (len(meta["depths"]), cfg.GEO_FEAT_DIM), dtype=np.float32)
    data = prepare_wf(cfg, cache["full_df"], cache["wd"], cache["wa"], meta,
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

    ld_tr = _build_loader(cfg, data, data["Xtr"], data["ytr"], data["wf_train"],
                          geo_t, True)
    ld_va = _build_loader(cfg, data, data["Xva"], data["yva"], data["wf_val"],
                          geo_t, False)

    AB.set_seed(seed)
    model = AB.AblationModel(cfg, N=N, in_dim=F_in, gcn_hid=cfg.GCN_HIDDEN,
                             tcn_ch=cfg.TCN_CHANNELS, k=cfg.TCN_KERNEL_SIZE,
                             dil=cfg.TCN_DILATIONS, attn_hid=cfg.ATTENTION_HIDDEN,
                             hor=hz, adj=adj_t, out_dim=data["out_dim"],
                             geo_feat_dim=data["geo_dim_actual"],
                             vert_adj=vert_t).to(cfg.DEVICE)
    n_par = int(AB.count_params(model))
    AB.estimate_flops(model, cfg, N, F_in, hz)   # 复现锁定脚本的 RNG 消耗顺序

    tier_w = WEIGHT_TIERS[tier]
    criterion = TieredLoss(*tier_w).to(cfg.DEVICE) if tier_w != WEIGHT_TIERS[DEFAULT_TIER] else None
    model, _, _, _ = AB.train_model(cfg, model, ld_tr, ld_va,
                                    run_dir / "best.pth", node_mask=mask_t,
                                    criterion=criterion)

    if eval_mode == "final":
        # 完整 [0,181) 重训：评锁定 test 段
        ld_te = _build_loader(cfg, data, data["Xte"], data["yte"], data["wf_test"],
                              geo_t, False)
        pred, true = AB.evaluate(model, ld_te, data["init_test"], data["scY"],
                                 data["pca_y"], hz, cfg.DEVICE, node_mask=mask_t,
                                 true_diff_win=data["yte_raw"])
        zero = float(np.sqrt(np.mean((data["init_test"] - true[:, -1, :]) ** 2)))
    else:
        # 折内验证：评 val 段（同样走 PCA 逆变换 + 原始差分真值口径）
        pred, true = AB.evaluate(model, ld_va, data["init_val"], data["scY"],
                                 data["pca_y"], hz, cfg.DEVICE, node_mask=mask_t,
                                 true_diff_win=data["yval_raw"])
        zero = float(np.sqrt(np.mean((data["init_val"] - true[:, -1, :]) ** 2)))
    gm = AB.global_metrics(true, pred)
    rmse = float(gm["RMSE_last"])
    rec = dict(variant=variant, model=VAR_LABEL.get(variant, variant),
               K=int(data["out_dim"]), tier=tier, seed=int(seed),
               n_train_ts=int(len(tr)), n_val_win=int(pred.shape[0]),
               n_test_ts=int(len(te)), params=n_par,
               RMSE_last=rmse, R2_last=float(gm["R2_last"]),
               RMSE_zero_last=zero, skill=1.0 - (rmse / zero) ** 2)
    json.dump(rec, open(mfile, "w", encoding="utf-8"), indent=2)
    np.savez(run_dir / "predictions.npz", pred_cum=pred, true_cum=true)
    try:
        (run_dir / "best.pth").unlink()   # 释放磁盘
    except OSError:
        pass
    return rec


# -----------------------------------------------------------------------------
# 锚点自检（--probe）：完整锁定划分 + Ours + K=13 + 锁定权重
# -----------------------------------------------------------------------------
def check_anchor(cache, hz, seeds):
    tr, va, te = base_split(len(cache["ts"]))
    rows = []
    for seed in seeds:
        rec = run_cell(cache["cfg"], cache, "V7_wo_VerticalGCN", 13,
                       DEFAULT_TIER, tr, va, te, hz, seed,
                       OUT_ROOT / "anchor", "final")
        rows.append(rec)
    df = pd.DataFrame(rows)
    df.round(8).to_csv(OUT_ROOT / "anchor_check.csv", index=False,
                       encoding="utf-8-sig")
    print("\n" + "=" * 104)
    print("[锚点自检] Ours(V7) · K=13 · 锁定权重 · 完整划分 · H=%d · n=%d 种子"
          % (hz, len(seeds)))
    z = (df.RMSE_zero_last - ZERO_H10).abs().max()
    print(f"  零变化与期望 {ZERO_H10} 的最大偏差 = {z:.2e}   "
          f"{'[OK]' if z <= 1e-3 else '[WARN]'}")
    if AUTH_TABLE.exists():
        ref = pd.read_csv(AUTH_TABLE).set_index("seed")["RMSE_last"]
        m = df.set_index("seed")["RMSE_last"]
        common = m.index.intersection(ref.index)
        diff = (m[common] - ref[common]).abs()
        for s in common:
            print(f"    seed={s:<5} 本批 {m[s]:.6f}  权威 {ref[s]:.6f}  "
                  f"Δ={m[s] - ref[s]:+.2e}")
        ok = (len(common) == len(m)) and float(diff.max()) <= PER_SEED_TOL
        print(f"  匹配 {len(common)}/{len(m)}；最大偏差 {float(diff.max()):.2e} => "
              f"{'[OK] 逐位复现锁定管线' if ok else '[WARN] 存在偏差，本批结论不可用'}")
    else:
        print(f"  [WARN] 未找到权威逐种子表：{AUTH_TABLE}")
    print("=" * 104)
    return df


# -----------------------------------------------------------------------------
# main
# -----------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hz", type=int, default=HZ_DEFAULT)
    ap.add_argument("--folds", type=int, default=5, help="walk-forward 折数")
    ap.add_argument("--seeds", type=str, default=None)
    ap.add_argument("--arches", type=str, default=",".join(ARCHES))
    ap.add_argument("--k-grid", type=str, default=",".join(map(str, K_GRID)))
    ap.add_argument("--weight-tiers", action="store_true",
                    help="并入损失权重档（3 档）候选")
    ap.add_argument("--eval-all-final", action="store_true",
                    help="对全部候选做终评（默认只对折最优候选）")
    ap.add_argument("--smoke", action="store_true", help="1 折 / 1 候选 / 1 种子 / 5 epoch")
    ap.add_argument("--probe", action="store_true", help="锚点自检（Ours/K13）")
    args = ap.parse_args()

    arches = [a.strip() for a in args.arches.split(",") if a.strip()]
    k_grid = [int(k) for k in args.k_grid.split(",") if k.strip()]
    tiers = [DEFAULT_TIER] + list(WEIGHT_TIERS) if args.weight_tiers else [DEFAULT_TIER]
    if args.weight_tiers and DEFAULT_TIER in tiers:
        tiers = list(dict.fromkeys(tiers))
    seeds = ([int(s) for s in args.seeds.split(",")] if args.seeds
             else list(AB.Config.SEEDS))
    if args.smoke:
        args.folds = 1
        arches, k_grid, seeds = arches[:1], k_grid[:1], seeds[:1]
        cfg0 = AB.Config(); cfg0.EPOCHS, cfg0.PATIENCE = 5, 3
    else:
        cfg0 = AB.Config()

    # 冒烟输出与正式分离（RESUME 判据按 run 目录，混用会让正式误跳冒烟格子）
    out_root = SMOKE_ROOT if args.smoke else OUT_ROOT
    out_root.mkdir(parents=True, exist_ok=True)

    if args.probe:
        cache = load_cache()
        df = check_anchor(cache, args.hz, seeds[:3])
        df.round(8).to_csv(TABLES / "T_p06_stage2_anchor.csv", index=False,
                           encoding="utf-8-sig")
        print(f"\n[已写出] {TABLES / 'T_p06_stage2_anchor.csv'}")
        return

    print("=" * 104)
    print(f"嵌套 CV + 损失权重档 | H={args.hz} | "
          f"{args.folds} 折 | 候选 {len(arches)}×{len(k_grid)}×{len(tiers)} = "
          f"{len(arches) * len(k_grid) * len(tiers)} | 种子 {len(seeds)}")
    print(f"  输出目录：{out_root}（新建，不覆盖旧结果）")
    if args.smoke:
        print("  [SMOKE] 1 折 / 1 候选 / 1 种子 / 5 epoch")
    print("=" * 104)

    cache = load_cache()
    n = len(cache["ts"])
    tr_locked, va_locked, te_locked = base_split(n)
    print(f"  时间戳 n={n}；锁定划分 train/val/test = "
          f"{len(tr_locked)}/{len(va_locked)}/{len(te_locked)}"
          f"（val [{va_locked[0]},{va_locked[-1]}]、test [{te_locked[0]},"
          f"{te_locked[-1]}] 全程不参与选择）")
    n_cv = len(tr_locked) + len(va_locked)   # 嵌套 CV 可用 = 锁定 train+val = 181
    folds = walkforward_folds(n_cv, args.folds)
    print("\n[折结构] 训练段 [0,%d) 内 walk-forward（val 宽 30 → %d 验证窗）"
          % (n_cv, 30 - AB.Config.LOOKBACK - args.hz + 1))
    for i, (ftr, fva, fte) in enumerate(folds):
        te_show = f"[{fte[0]},{n_cv})" if len(fte) else "∅"
        print(f"  fold{i}: train=[0,{len(ftr)})  val=[{fva[0]},{fva[-1] + 1})  "
              f"(test槽={te_show} 不评估)")

    # ---- 外层循环：逐折 × 逐候选训练，val 评估 ----
    rows = []
    selections = []
    agg_val = {}          # 候选 → (fold×seed) 的 val RMSE list（跨折聚合，用于最终选定）
    for i, (ftr, fva, fte) in enumerate(folds):
        fold_dir = out_root / f"fold{i}"
        fold_dir.mkdir(parents=True, exist_ok=True)
        print("\n" + "=" * 104)
        print(f"[fold{i}] train {len(ftr)} 时点 / val {len(fva)} 时点"
              f"（{len(fva) - AB.Config.LOOKBACK - args.hz + 1} 验证窗）")
        print("=" * 104)
        cell_scores = {}
        for arch in arches:
            for k_fixed in k_grid:
                for tier in tiers:
                    key = f"{arch}|{k_fixed}|{tier}"
                    cell_dir = fold_dir / f"{VAR_LABEL.get(arch, arch)}_K{k_fixed}_{tier}"
                    cell_dir.mkdir(parents=True, exist_ok=True)
                    scores = []
                    for si, seed in enumerate(seeds, 1):
                        t0 = time.time()
                        rec = run_cell(cfg0, cache, arch, k_fixed, tier,
                                       ftr, fva, fte, args.hz, seed,
                                       cell_dir, "val")
                        rec["fold"] = i
                        rec["arch"] = arch
                        rows.append(rec)
                        scores.append(float(rec["RMSE_last"]))
                        agg_val.setdefault(key, []).append(float(rec["RMSE_last"]))
                        print(f"    [{arch}|K{k_fixed}|{tier}] seed={seed:<5} "
                              f"K={rec['K']:<3} val_RMSE={rec['RMSE_last']:.4f} "
                              f"skill={rec['skill']:+.4f}  ({time.time() - t0:.1f}s)")
                    cell_scores[key] = np.mean(scores)
        best_key = min(cell_scores, key=cell_scores.get)
        best_arch, best_k, best_tier = best_key.split("|")
        selections.append(dict(fold=i, n_train_ts=int(len(ftr)),
                               n_val_ts=int(len(fva)),
                               best_variant=best_arch,
                               best_model=VAR_LABEL.get(best_arch, best_arch),
                               best_K=int(best_k), best_tier=best_tier,
                               best_val_RMSE=float(cell_scores[best_key])))
        print(f"\n  → fold{i} 选中（折内）：{best_arch} / K={best_k} / {best_tier} "
              f"(val 跨种子均值 RMSE_last={cell_scores[best_key]:.4f})")

    df_runs = pd.DataFrame(rows)
    df_runs.round(8).to_csv(out_root / "fold_runs.csv", index=False,
                            encoding="utf-8-sig")
    df_sel = pd.DataFrame(selections)
    df_sel.round(8).to_csv(out_root / "fold_selection.csv", index=False,
                           encoding="utf-8-sig")

    # ---- 最终选定：跨折（fold×seed）聚合 val RMSE 均值最小 ----
    agg_mean = {k: float(np.mean(v)) for k, v in agg_val.items()}
    agg_sem = {k: float(np.std(v, ddof=0)) for k, v in agg_val.items()}
    sel_all_key = min(agg_mean, key=agg_mean.get)
    sel_arch, sel_k, sel_tier = sel_all_key.split("|")
    sel_arch = int(sel_arch) if sel_arch.isdigit() else sel_arch

    print("\n" + "=" * 104)
    print("[嵌套 CV 选择结果]（逐折折内最优 —— 稳定性诊断）")
    print(df_sel.to_string(index=False))
    vc = df_sel.groupby(["best_variant", "best_K", "best_tier"]).size()
    print("\n[跨折稳定性] 各候选被选中的折数（诊断用，不直接决定最终配置）：")
    for key, cnt in vc.items():
        print(f"  {key[0]} / K={key[1]} / {key[2]}：{cnt}/{len(selections)} 折")
    print("\n[最终选定] 跨折聚合（fold×seed）val RMSE 均值最小：")
    for k, m in sorted(agg_mean.items(), key=lambda x: x[1]):
        star = "  <-- 选定" if k == sel_all_key else ""
        print(f"  {k}: 均值 {m:.4f}（SD {agg_sem[k]:.4f}）{star}")

    # ---- 终评：完整 train+val 重训选定配置，测 test 一次 ----
    sel_variant, sel_k, sel_tier = sel_arch, int(sel_k), sel_tier
    print("\n" + "=" * 104)
    print(f"[终评] 完整 train+val [0,{n_cv}) 重训 {sel_variant} / K={sel_k} / "
          f"{sel_tier}，测试段 [{te_locked[0]},{te_locked[-1]}] 评估一次（{len(seeds)} 种子）")
    print("=" * 104)

    final_variants = [(sel_variant, sel_k, sel_tier)]
    if args.eval_all_final:
        final_variants = [(a, k, t) for a in arches for k in k_grid for t in tiers]
    final_rows = []
    for (arch, k_fixed, tier) in final_variants:
        fin_dir = out_root / f"final_{VAR_LABEL.get(arch, arch)}_K{k_fixed}_{tier}"
        fin_dir.mkdir(parents=True, exist_ok=True)
        for si, seed in enumerate(seeds, 1):
            t0 = time.time()
            rec = run_cell(cfg0, cache, arch, k_fixed, tier,
                           tr_locked, va_locked, te_locked, args.hz, seed,
                           fin_dir, "final")
            rec["arch"] = arch
            final_rows.append(rec)
            print(f"  [{arch}|K{k_fixed}|{tier}] seed={seed:<5} "
                  f"test_RMSE={rec['RMSE_last']:.4f} skill={rec['skill']:+.4f} "
                  f"  ({time.time() - t0:.1f}s)")

    df_fin = pd.DataFrame(final_rows)
    df_fin.round(8).to_csv(out_root / "final_test.csv", index=False,
                           encoding="utf-8-sig")
    fin_agg = df_fin.groupby(["arch", "K", "tier"], as_index=False).agg(
        n_seeds=("seed", "size"),
        RMSE_mean=("RMSE_last", "mean"), RMSE_std=("RMSE_last", "std"),
        skill_mean=("skill", "mean"), skill_std=("skill", "std"),
        zero_mean=("RMSE_zero_last", "mean"))
    print("\n[终评汇总]（测试段 RMSE_last，跨种子均值±std）")
    print(fin_agg.round(4).to_string(index=False))

    # ---- 关键判读：val 选择 vs test 最优 ----
    best_test = fin_agg.loc[fin_agg.RMSE_mean.idxmin()]
    print("\n" + "-" * 104)
    print("[判读] 嵌套 CV 在 val 上选出的配置 vs 测试段真实最优：")
    print(f"  · val 聚合选定        ：{sel_variant} / K={sel_k} / {sel_tier}")
    print(f"  · test 段 RMSE 最小  ：{best_test.arch} / K={best_test.K} / "
          f"{best_test.tier}（RMSE {best_test.RMSE_mean:.4f}）")
    consistent = (sel_variant == best_test.arch and int(sel_k) == int(best_test.K)
                  and sel_tier == best_test.tier)
    print(f"  · val/test 选择一致？ {'是' if consistent else '否'}")
    if sel_variant == "V7_wo_VerticalGCN" and int(sel_k) == 13:
        print("  · 注意：val 选中正文配置 V7/K13 —— 与阶段 1（val 选 V5）不同，"
              "须如实并报，不改主稿结论。")
    print("-" * 104)

    # ---- 同步副本到 _paper_tables ----
    df_runs.round(8).to_csv(TABLES / "T_p06_stage2_fold.csv", index=False,
                            encoding="utf-8-sig")
    df_sel.round(8).to_csv(TABLES / "T_p06_stage2_selection.csv", index=False,
                           encoding="utf-8-sig")
    df_fin.round(8).to_csv(TABLES / "T_p06_stage2_final.csv", index=False,
                           encoding="utf-8-sig")
    print("\n[已写出]")
    print(f"  {out_root / 'fold_runs.csv'} / fold_selection.csv / final_test.csv")
    print(f"  {TABLES / 'T_p06_stage2_fold.csv'} / selection / final")


if __name__ == "__main__":
    main()
