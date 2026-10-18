# -*- coding: utf-8 -*-
r"""_p12_p15_baseline_budget.py —— 外部基线统一预算/调参记录 + 现代时序基线补训。

本脚本做什么
------------
1) **统一预算 + 调参记录**：写 `T_p1215_budget.{json,md}` —— 宽度 64、输入（4 维无地质；
   temporal 路径展平 → PCA 48 维）、划分（锁定 [0,149)/[149,181)/[181,213)）、优化器
   AdamW / LR 1e-3（图/物理模块 5e-4）/ WD 1e-3 / 400 epoch / patience 40 / batch 32 /
   MultiLoss（1.0·Huber + 0.5·diff + 0.2·freq）/ grad clip 1.0；**无逐模型手工调参**，
   全部共享同一配方，架构固有超参按文献标准设定并固定（= "同一调参预算"）。
2) **补训现代时序基线（默认）**：DLinear / PatchTST / iTransformer 走 **temporal 路径**
   （与 LSTM/Transformer 完全同一输入：(B, L=12, 48)），输出 (B, hz, K=13) 标准化 PCA 系数，
   5 视界 × 30 种子。GBDT-inc：同一输入 + **滞后 PCA 系数（lag=1,2,3）+ 滚动均值/标准差
   （w=3,6,12）**，增量目标（PCA 系数 = 一阶差分），H=10 × 30 种子。
   结果写入 `<WORK_ROOT>\Baseline_Budget_w64\`（新建，不覆盖旧结果）。
3) **可选（--also_classic）**：把经典外部基线（LSTM / BiLSTM-Attn / Transformer / V-Attn /
   STGCN / GraphWaveNet / V-MLP / GBDT）按同一预算**重训**进 `Classic_Rerun_w64\`（满足
   "全部重训"的严格口径）；默认**复用**已完成的 w64 对比结果（2026-09-24 容量对齐已完成，
   BiLSTM-Attn 已是宽度 64 口径）。
4) **合并汇总 + 显著性**：Ours / TCN-Only 取权威源 `Ablation_New_5d_mid`，经典基线取 w64，
   新基线取本脚本；按种子对齐对 Ours 做配对 t 检验（末步口径 RMSE_last，附可达分量），
   输出 `T_p1215_summary.csv` / `T_p1215_significance.csv` / `T_p1215_new_runs.csv`。
5) **锚点自检**：H=10 零变化 = 1.372010；Ours 权威均值 = 1.5687 ± 0.0940（逐位核对）。

铁律
----
· 不覆盖旧结果：新基线写 `Baseline_Budget_w64\`，经典重训写 `Classic_Rerun_w64\`。
· 不改训练配方与数据管线：直接复用 `Comparison_Experiment.py` 的 prepare_shared_data /
  build_temporal_samples / train_model / evaluate / global_metrics / output_space_floor /
  node_metrics —— 保证与表 5 完全同口径。
· 锚点依赖锁定管线：一旦 `Ablation_Experiment.py` 被改动，本脚本同受影响
  （须重跑）。
· 训练类脚本须在可写盘环境运行。

运行（PowerShell）
-----------------
  $py = "python"
  & $py -u _p12_p15_baseline_budget.py --smoke    # 冒烟：H=10 / 1 种子 / 3 epoch（快速验证形状与链路）
  & $py -u _p12_p15_baseline_budget.py --probe    # 探针：H=10 / 2 种子 / 全 epoch（含锚点自检）
  & $py -u _p12_p15_baseline_budget.py            # 正式：3 新时序基线 × 5 视界 × 30 种子 + GBDT-inc(H10)
  续跑：$env:P1215_RESUME="1"
  可选：--models DLinear,PatchTST --horizons 2,10 --nseeds 30 --also_classic
"""
from __future__ import annotations

from _env import AUX_ROOT, EXAMPLE_MODE, WORK_ROOT

import argparse
import json
import os
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.multioutput import MultiOutputRegressor
from scipy.stats import ttest_rel

warnings.filterwarnings("ignore")

PROJ = WORK_ROOT
sys.path.insert(0, str(PROJ))

import Comparison_Experiment as CE          # noqa: E402
import _robust_io                          # noqa: F401,E402  (写盘重试层，须早于 IO)

# -----------------------------------------------------------------------------
# 路径与规模
# -----------------------------------------------------------------------------
OUT_ROOT = PROJ / "Baseline_Budget_w64"          # 新基线输出（新建，不覆盖）
CLASSIC_ROOT = PROJ / "Classic_Rerun_w64"        # --also_classic 的经典重训输出
TABLES = PROJ / "_paper_tables"
TABLES.mkdir(parents=True, exist_ok=True)

# 权威源（只读）
ABLATION_AUTH = AUX_ROOT / "Ablation_New_5d_mid"       # Ours(V7)/TCN-Only(V0) 逐种子权威表
# 回退源：_5d_mid 仅含 H6/H10 的 global_30runs.csv，H2/H8/H12 缺 —— 回退 _5d
# （与 _5d_mid 在 H6/H10 逐位一致，系同一批结果）
ABLATION_FALLBACK = AUX_ROOT / "Ablation_New_5d"
W64_ROOT = AUX_ROOT / "Comparison_Results_w64"          # 经典外部基线 w64 结果（2026-09-24 容量对齐）

NEW_TS_MODELS = ["DLinear", "PatchTST", "iTransformer"]
GBDT_INC_MODEL = "GBDT-inc"
ALL_NEW = NEW_TS_MODELS + [GBDT_INC_MODEL]

HORIZONS = [2, 6, 8, 10, 12]
GBDT_HZ = 10            # 树模型只在主展示步长评估（与 CE.GBDT_HORIZON 一致）

# 锚点（锁定管线权威数字）
ZERO_H10 = 1.372010
OURS_H10_EXPECT = (1.5687, 0.0940)   # mean, std（30 种子）

RESUME = os.environ.get("P1215_RESUME", "0") == "1"

# -----------------------------------------------------------------------------
# 新基线架构超参（文献标准设定、固定、共享同一训练预算）
# -----------------------------------------------------------------------------
D_MODEL = 64           # 与 CE.HIDDEN_DIM=64 对齐（统一宽度）
N_LAYERS = 2           # Transformer 编码器层数（与 CE Transformer 一致）
N_HEADS = 2            # 与 CE.N_HEADS 一致
PATCH_LEN = 4
PATCH_STRIDE = 2       # L=12 → n_patch = (12-4)//2 + 1 = 5
DLINEAR_TREND = "global_mean"   # L=12 短窗，移动平均核取全窗均值（无边界填充伪影）

# 统一预算记录（调参记录）
UNIFIED_BUDGET = {
    "experiment": "外部基线统一预算补训与调参记录",
    "date": "2026-10-05",
    "input": {
        "features": "4 维无地质：累计位移、一阶差分、归一化深度、浅层标志",
        "graph_path": "(B, L, N, F=4) 保留节点拓扑（Ours 框架 / 图基线：V-Attn/V-MLP/STGCN/GWN）",
        "temporal_path": "展平 (T, N*F) → PCA(TEMPORAL_PCA_DIM=48)（时序基线：LSTM/BiLSTM-Attn/"
                         "Transformer/DLinear/PatchTST/iTransformer/GBDT/GBDT-inc）",
        "target": "PCA(0.90) 一阶差分系数（K=13）；主口径 = 末步累计位移 RMSE_last",
    },
    "split": "213 时点两天栅格：train [0,149) / val [149,181) / test [181,213)（锁定划分）",
    "seeds": "30 个锁定种子（_gen_seeds.py，四脚本逐位一致）",
    "width": "64 = HIDDEN_DIM / GRAPH_HIDDEN_DIM / OURS_GCN_HIDDEN（与本文骨干一致）",
    "lookback": 12,
    "optimizer": {"name": "AdamW", "lr": "1e-3（图/物理相关模块 5e-4）",
                  "weight_decay": "1e-3"},
    "training": {"epochs": 400, "patience": 40,
                 "lr_scheduler": "ReduceLROnPlateau(factor=0.5, patience=15)",
                 "batch_size": 32, "grad_clip": 1.0,
                 "loss": "MultiLoss = 1.0*Huber + 0.5*diff + 0.2*freq"},
    "tuning_policy": ("无逐模型手工调参；全部模型共享同一配方（上表），架构固有超参"
                      "（层数/头数/核宽/patch 长等）按文献标准设定并固定 —— 即‘同一调参预算’"),
    "models": {},
}

NEW_LABELS = {"DLinear": "DLinear", "PatchTST": "PatchTST",
              "iTransformer": "iTransformer", "GBDT-inc": "GBDT-inc"}


# =============================================================================
# 新时序基线模型（temporal 路径：输入 (B, L, D)，输出 (B, hz, K) 标准化 PCA 系数）
# =============================================================================
class DLinearForecaster(nn.Module):
    """DLinear（Zeng et al. 2023，时序分解 + 线性）。

    短窗（L=12）下趋势 = 每通道全窗均值（等价于 kernel=12 的移动平均，无边界填充伪影），
    残差 = 输入 - 趋势；两支线性层各自从展平窗口直接映射到 hz·K，相加输出。
    与 CE 既有 temporal 基线完全同一输入，是"线性（含趋势分解）"的现代强基线。
    """

    def __init__(self, seq_len, input_dim, out_dim, horizon, dropout=0.3):
        super().__init__()
        self.horizon, self.out_dim = horizon, out_dim
        self.fc_trend = nn.Linear(seq_len * input_dim, horizon * out_dim)
        self.fc_res = nn.Linear(seq_len * input_dim, horizon * out_dim)

    def forward(self, x):
        B, L, D = x.shape
        trend = x.mean(dim=1, keepdim=True).expand(B, L, D)
        res = x - trend
        out = self.fc_trend(trend.reshape(B, -1)) + self.fc_res(res.reshape(B, -1))
        return out.reshape(B, self.horizon, self.out_dim)


class PatchTSTForecaster(nn.Module):
    """PatchTST（Nie et al. 2023，patch + 通道独立共享 Transformer）。

    每个通道（temporal 路径的 48 维分量）沿时间维切 patch，共享一个 Transformer 编码器，
    编码后按通道平均、再按 patch mean/max 聚合，映射到全局 PCA 系数 —— 保留 PatchTST
    的通道独立 + patch 建模思想，适配本任务的全局输出。
    """

    def __init__(self, seq_len, input_dim, d_model, out_dim, horizon,
                 patch_len=PATCH_LEN, stride=PATCH_STRIDE,
                 n_layers=N_LAYERS, n_heads=N_HEADS, dropout=0.3):
        super().__init__()
        self.horizon, self.out_dim = horizon, out_dim
        self.patch_len, self.stride = patch_len, stride
        self.n_patch = (seq_len - patch_len) // stride + 1
        assert self.n_patch >= 1, "patch 数 < 1，请调小 patch_len / 调大 stride"
        self.proj = nn.Linear(patch_len, d_model)
        layer = nn.TransformerEncoderLayer(
            d_model, n_heads, d_model * 4, dropout, batch_first=True,
            norm_first=True, activation="gelu")
        self.encoder = nn.TransformerEncoder(layer, n_layers)
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Sequential(
            nn.Linear(2 * d_model, 64), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(64, horizon * out_dim))

    def forward(self, x):
        B, L, D = x.shape
        x = x.transpose(1, 2)                              # (B, D, L)
        x = x.unfold(2, self.patch_len, self.stride)       # (B, D, n_patch, pl)
        toks = x.reshape(B * D, self.n_patch, self.patch_len)
        h = self.norm(self.encoder(self.proj(toks)))       # (B*D, n_patch, d_model)
        h = h.reshape(B, D, self.n_patch, -1).mean(dim=1)  # (B, n_patch, d_model)
        h = torch.cat([h.mean(dim=1), h.amax(dim=1)], dim=-1)
        return self.head(h).reshape(B, self.horizon, self.out_dim)


class iTransformerForecaster(nn.Module):
    """iTransformer（Liu et al. 2024，倒置 Transformer：变量为 token）。

    temporal 路径的 48 维分量作为变量 token，各变量的整段（L=12）序列先线性嵌入，
    在变量维做自注意力，聚合后映射到全局 PCA 系数。
    """

    def __init__(self, seq_len, input_dim, d_model, out_dim, horizon,
                 n_layers=N_LAYERS, n_heads=N_HEADS, dropout=0.3):
        super().__init__()
        self.horizon, self.out_dim = horizon, out_dim
        self.embed = nn.Linear(seq_len, d_model)
        layer = nn.TransformerEncoderLayer(
            d_model, n_heads, d_model * 4, dropout, batch_first=True,
            norm_first=True, activation="gelu")
        self.encoder = nn.TransformerEncoder(layer, n_layers)
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Sequential(
            nn.Linear(2 * d_model, 64), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(64, horizon * out_dim))

    def forward(self, x):
        B, L, D = x.shape
        toks = self.embed(x.transpose(1, 2))               # (B, D, d_model)
        h = self.norm(self.encoder(toks))                  # (B, D, d_model)
        h = torch.cat([h.mean(dim=1), h.amax(dim=1)], dim=-1)
        return self.head(h).reshape(B, self.horizon, self.out_dim)


def make_new_model(cfg, name, in_dim, out_dim, hz):
    """新时序基线工厂（temporal 路径）。"""
    if name == "DLinear":
        return DLinearForecaster(cfg.LOOKBACK, in_dim, out_dim, hz,
                                 dropout=cfg.DROPOUT).to(cfg.DEVICE)
    if name == "PatchTST":
        return PatchTSTForecaster(cfg.LOOKBACK, in_dim, D_MODEL, out_dim, hz,
                                  dropout=cfg.DROPOUT).to(cfg.DEVICE)
    if name == "iTransformer":
        return iTransformerForecaster(cfg.LOOKBACK, in_dim, D_MODEL, out_dim, hz,
                                      dropout=cfg.DROPOUT).to(cfg.DEVICE)
    raise ValueError(f"未知新基线：{name}")


# =============================================================================
# GBDT-inc：增量目标 + 滞后/滚动特征（树模型补强）
# =============================================================================
def full_pca_coefs(data, shared, seg="train"):
    """(len_seg, K) 标准化 PCA 系数时间序列（y 为段内一阶差分）。"""
    y = {"train": shared["y_train"], "val": shared["y_val"],
         "test": shared["y_test"]}[seg]
    return data["scY"].transform(data["pca_y"].transform(y))


def build_gbdt_inc_feats(X, coef_full, lb, hz, lags=(1, 2, 3), roll_ws=(3, 6, 12)):
    """X: (n, L, D) temporal 路径窗口；coef_full: (T, K) 标准化 PCA 系数（段内）。

    每窗 i 的"当前时刻" t = i + lb（目标覆盖 [t, t+hz)）：
      · 展平窗口特征（与 LSTM 同输入）        → D·L
      · 滞后 PCA 系数 y[t-1], y[t-2], y[t-3] → 3·K（AR 式滞后特征）
      · 首分量滚动均值/标准差 [t-w, t-1]      → 2·len(roll_ws)（滚动统计）
    返回 (n, D·L + 3·K + 2·len(roll_ws))。
    """
    n = X.shape[0]
    t_idx = np.arange(lb, lb + n)
    parts = [X.reshape(n, -1)]
    for k in lags:
        parts.append(np.asarray(coef_full[t_idx - k], dtype=np.float64))
    c0 = np.asarray(coef_full[:, 0], dtype=np.float64)
    for w in roll_ws:
        lo, hi = t_idx - w, t_idx
        mean = np.array([c0[lo[j]:hi[j]].mean() for j in range(n)])
        std = np.array([c0[lo[j]:hi[j]].std() for j in range(n)])
        parts.append(mean[:, None])
        parts.append(std[:, None])
    return np.hstack(parts).astype(np.float32)


def _make_gbdt(seed):
    """与 CE 同一树模型工厂（同预算）。"""
    try:
        from xgboost import XGBRegressor
        return "XGBoost", XGBRegressor(
            n_estimators=300, max_depth=4, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8,
            random_state=seed, n_jobs=-1, verbosity=0)
    except Exception:
        pass
    try:
        from lightgbm import LGBMRegressor
        return "LightGBM", LGBMRegressor(
            n_estimators=300, max_depth=4, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8,
            random_state=seed, n_jobs=-1, verbose=-1)
    except Exception:
        pass
    from sklearn.ensemble import HistGradientBoostingRegressor
    return "GBDT-sklearn", HistGradientBoostingRegressor(
        max_iter=120, max_depth=3, learning_rate=0.06, random_state=seed)


# =============================================================================
# 运行器
# =============================================================================
def run_new_temporal_model(cfg, name, shared, hz, out_root):
    """新时序基线（DLinear/PatchTST/iTransformer）：temporal 路径全流程。"""
    logger = CE.logger
    logger.info(f"\n{'=' * 60}\n{name}  H={hz}\n{'=' * 60}")
    data = CE.build_temporal_samples(shared, cfg, hz)
    Xtr, ytr = data["Xtr"], data["ytr"]
    Xva, yva = data["Xva"], data["yva"]
    Xte, yte = data["Xte"], data["yte"]
    out_dim, in_dim = data["out_dim"], data["temporal_in_dim"]
    init_test, scY, pca_y = data["init_test"], data["scY"], data["pca_y"]

    model_dir = out_root / f"H{hz}" / name
    model_dir.mkdir(parents=True, exist_ok=True)
    all_global, failed = [], []

    for si, seed in enumerate(cfg.SEEDS):
        run_dir = model_dir / f"run_{seed}"
        mfile = run_dir / "metrics.json"
        if RESUME and all((run_dir / f).exists()
                          for f in ("metrics.json", "best.pth",
                                    "predictions.npz", "losses.npy")):
            with open(mfile, "r", encoding="utf-8") as f:
                gm = json.load(f)
            gm["seed"] = seed
            all_global.append(gm)
            logger.info(f"    Seed {si + 1}/{len(cfg.SEEDS)} ({seed}): "
                        f"已完整，跳过（RESUME=1）")
            continue

        try:
            CE.set_seed(seed)
            model = make_new_model(cfg, name, in_dim, out_dim, hz)
            n_params = int(CE.count_params(model))
            # FLOPs 估计（temporal 输入）——保持与 CE 主流程相同的 RNG 消耗顺序
            flops = CE.estimate_flops(model, cfg, "temporal", Xtr[:1], None, None)

            train_ds = TensorDataset(torch.FloatTensor(Xtr), torch.FloatTensor(ytr))
            val_ds = TensorDataset(torch.FloatTensor(Xva), torch.FloatTensor(yva))
            test_ds = TensorDataset(torch.FloatTensor(Xte), torch.FloatTensor(yte))
            train_ld = DataLoader(train_ds, batch_size=cfg.BATCH_SIZE, shuffle=True)
            val_ld = DataLoader(val_ds, batch_size=cfg.BATCH_SIZE)
            test_ld = DataLoader(test_ds, batch_size=cfg.BATCH_SIZE)

            run_dir.mkdir(parents=True, exist_ok=True)
            save_path = run_dir / "best.pth"
            model, train_time, tr_losses, va_losses, best_ep = CE.train_model(
                cfg, model, train_ld, val_ld, save_path,
                model_tag=f"{name}_H{hz}", seed=seed, ours_lr_groups=False)

            t0 = time.time()
            pred_cum, true_cum = CE.evaluate(
                model, test_ld, init_test, scY, pca_y, hz, cfg.DEVICE,
                true_diff_win=data["yte_raw"])
            infer_time = (time.time() - t0) / max(len(test_ld.dataset), 1)

            gm = CE.global_metrics(true_cum, pred_cum)
            _fl = CE.output_space_floor(data["yte_raw"], init_test, scY, pca_y, hz,
                                        ns=pred_cum.shape[0])
            gm["RMSE_floor_last"] = _fl["RMSE_floor_last"]
            gm["RMSE_reach_last"] = float(np.sqrt(max(
                gm["RMSE_last"] ** 2 - _fl["MSE_floor_last"], 0.0)))
            gm["RMSE_floor"] = _fl["RMSE_floor_all"]
            gm["RMSE_reach"] = float(np.sqrt(max(
                gm["RMSE"] ** 2 - _fl["MSE_floor_all"], 0.0)))
            gm.update({"seed": int(seed), "Params": n_params, "FLOPs": flops,
                       "TrainTime": train_time, "InferTime": infer_time,
                       "BestEpoch": best_ep, "BestValLoss": float(min(va_losses)),
                       "ImplTag": name})
            all_global.append(gm)

            with open(mfile, "w", encoding="utf-8") as f:
                json.dump(gm, f, indent=2)
            np.savez(run_dir / "predictions.npz", pred_cum=pred_cum, true_cum=true_cum)
            loss_df = pd.DataFrame({"epoch": range(1, len(tr_losses) + 1),
                                    "train_loss": tr_losses, "val_loss": va_losses})
            loss_df.to_csv(run_dir / "loss_history.csv", index=False)
            np.save(run_dir / "losses.npy",
                    {"tr": tr_losses, "va": va_losses})

            logger.info(f"    Seed {si + 1}/{len(cfg.SEEDS)} ({seed}) "
                        f"RMSE(last)={gm['RMSE_last']:.4f} "
                        f"R2(last)={gm['R2_last']:.4f} params={n_params} "
                        f"BestEp={best_ep}")
        except Exception as e:
            logger.error(f"    Seed {seed} failed: {e}")
            failed.append(seed)
            import traceback
            traceback.print_exc()

        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    if not all_global:
        logger.error(f"  {name} H={hz}: ALL seeds failed!")
        return None
    if failed:
        logger.error(f"  !! {name} H={hz}: {len(failed)}/{len(cfg.SEEDS)} 个种子失败 {failed}")

    df_g = pd.DataFrame(all_global)
    sdir = model_dir / "summary"
    sdir.mkdir(exist_ok=True)
    df_g.to_csv(sdir / f"global_{len(all_global)}runs.csv", index=False)
    logger.info(f"  {name} H={hz}: {len(all_global)} seeds done. "
                f"RMSE(last)={df_g['RMSE_last'].mean():.4f}±{df_g['RMSE_last'].std():.4f}")
    return df_g


def run_gbdt_inc(cfg, shared, hz, out_root):
    """GBDT-inc：增量目标（PCA 系数）+ 滞后/滚动特征，同一树预算。"""
    logger = CE.logger
    logger.info(f"\n{'=' * 60}\n{GBDT_INC_MODEL}  H={hz}\n{'=' * 60}")
    data = CE.build_temporal_samples(shared, cfg, hz)
    out_dim = data["out_dim"]
    coef_tr = full_pca_coefs(data, shared, "train")
    coef_va = full_pca_coefs(data, shared, "val")
    coef_te = full_pca_coefs(data, shared, "test")

    Xtr = build_gbdt_inc_feats(data["Xtr"], coef_tr, cfg.LOOKBACK, hz)
    Xva = build_gbdt_inc_feats(data["Xva"], coef_va, cfg.LOOKBACK, hz)
    Xte = build_gbdt_inc_feats(data["Xte"], coef_te, cfg.LOOKBACK, hz)
    ytr = data["ytr"].reshape(data["ytr"].shape[0], -1)
    yva = data["yva"].reshape(data["yva"].shape[0], -1)
    logger.info(f"  GBDT-inc 特征维 = {Xtr.shape[1]}"
                f"（窗口展平 {data['Xtr'].shape[1] * data['Xtr'].shape[2]}"
                f" + 滞后 3·{out_dim} + 滚动 2·3）")

    model_dir = out_root / f"H{hz}" / GBDT_INC_MODEL
    model_dir.mkdir(parents=True, exist_ok=True)
    all_global, failed = [], []

    for si, seed in enumerate(cfg.SEEDS):
        run_dir = model_dir / f"run_{seed}"
        mfile = run_dir / "metrics.json"
        if RESUME and all((run_dir / f).exists()
                          for f in ("metrics.json", "predictions.npz")):
            with open(mfile, "r", encoding="utf-8") as f:
                gm = json.load(f)
            gm["seed"] = seed
            all_global.append(gm)
            logger.info(f"    Seed {si + 1}/{len(cfg.SEEDS)} ({seed}): "
                        f"已完整，跳过（RESUME=1）")
            continue
        try:
            CE.set_seed(seed)
            tag, base = _make_gbdt(seed)
            model = MultiOutputRegressor(base, n_jobs=1)
            t0 = time.time()
            model.fit(Xtr, ytr)
            train_time = time.time() - t0
            val_mse = float(np.mean((model.predict(Xva) - yva) ** 2))
            t1 = time.time()
            pred_flat = model.predict(Xte)
            infer_time = (time.time() - t1) / max(len(Xte), 1)
            pred_n = pred_flat.reshape(-1, hz, out_dim)

            pred_cum, true_cum = CE._decode_predict(
                pred_n, data["init_test"], data["scY"], data["pca_y"], hz,
                data["yte_raw"])
            gm = CE.global_metrics(true_cum, pred_cum)
            _fl = CE.output_space_floor(data["yte_raw"], data["init_test"],
                                        data["scY"], data["pca_y"], hz,
                                        ns=pred_cum.shape[0])
            gm["RMSE_floor_last"] = _fl["RMSE_floor_last"]
            gm["RMSE_reach_last"] = float(np.sqrt(max(
                gm["RMSE_last"] ** 2 - _fl["MSE_floor_last"], 0.0)))
            gm["RMSE_floor"] = _fl["RMSE_floor_all"]
            gm["RMSE_reach"] = float(np.sqrt(max(
                gm["RMSE"] ** 2 - _fl["MSE_floor_all"], 0.0)))
            gm.update({"seed": int(seed), "Params": 0, "FLOPs": 0.0,
                       "TrainTime": train_time, "InferTime": infer_time,
                       "BestEpoch": -1, "BestValLoss": val_mse, "ImplTag": tag})
            all_global.append(gm)

            run_dir.mkdir(parents=True, exist_ok=True)
            with open(mfile, "w", encoding="utf-8") as f:
                json.dump(gm, f, indent=2)
            np.savez(run_dir / "predictions.npz", pred_cum=pred_cum, true_cum=true_cum)
            logger.info(f"    Seed {si + 1}/{len(cfg.SEEDS)} ({seed}) "
                        f"RMSE(last)={gm['RMSE_last']:.4f}  [{tag}]")
        except Exception as e:
            logger.error(f"    Seed {seed} failed: {e}")
            failed.append(seed)
            import traceback
            traceback.print_exc()

    if not all_global:
        logger.error(f"  {GBDT_INC_MODEL} H={hz}: ALL seeds failed!")
        return None
    if failed:
        logger.error(f"  !! {GBDT_INC_MODEL} H={hz}: {len(failed)}/{len(cfg.SEEDS)} 个种子失败 {failed}")

    df_g = pd.DataFrame(all_global)
    sdir = model_dir / "summary"
    sdir.mkdir(exist_ok=True)
    df_g.to_csv(sdir / f"global_{len(all_global)}runs.csv", index=False)
    logger.info(f"  {GBDT_INC_MODEL} H={hz}: {len(all_global)} seeds done. "
                f"RMSE(last)={df_g['RMSE_last'].mean():.4f}±{df_g['RMSE_last'].std():.4f}")
    return df_g


# =============================================================================
# 合并汇总：权威 Ours/V0 + w64 经典基线 + 本脚本新基线
# =============================================================================
def load_global_csv(d: Path):
    """读取某模型 summary 目录下的 global_*runs.csv（取种子数最多者）。"""
    if not d.exists():
        return None
    cand = sorted(d.glob("global_*runs.csv"))
    if not cand:
        return None
    return pd.read_csv(cand[-1])


def assemble_models(cfg, out_root, classic_source="w64"):
    """逐视界加载全部模型逐种子结果（统一加 Model 键）。

    Ours / TCN-Only：先试权威 `_5d_mid`（仅 H6/H10 有 global_30runs.csv），
    缺失视界回退 `_5d`（与 _5d_mid 逐位一致）。返回 (root, dname) 二元组由
    加载循环按 ablation 特殊处理。
    """
    # 模型键 → (加载类型, 目录名)
    src = {}
    for m in NEW_TS_MODELS + [GBDT_INC_MODEL]:
        src[m] = ("new", m)
    # 权威消融（Ours / TCN-Only）：_5d_mid 优先、缺失视界回退 _5d
    src["V7_Ours"] = ("ablation", "V7_wo_VerticalGCN")
    src["V0_TCN-Only"] = ("ablation", "V0_Pure_TCN")
    # 经典外部基线（w64 或 --also_classic 重跑目录）
    classic_root = (W64_ROOT if classic_source == "w64" else CLASSIC_ROOT)
    for m in ["LSTM", "BiLSTM-Attn", "Transformer", "V-Attention",
              "STGCN", "GraphWaveNet", "V-MLP", "GBDT"]:
        src[m] = ("classic", m)

    all_rows = []
    for hz in cfg.HORIZONS:
        for key, (kind, dname) in src.items():
            if key == GBDT_INC_MODEL and hz != GBDT_HZ:
                continue
            if key == "GBDT" and hz != GBDT_HZ:
                continue
            if kind == "new":
                root = out_root
            elif kind == "classic":
                root = classic_root
            else:  # ablation：_5d_mid 优先、缺失回退 _5d
                root = ABLATION_AUTH
            df = load_global_csv(root / f"H{hz}" / dname / "summary")
            if df is None and kind == "ablation":
                root = ABLATION_FALLBACK
                df = load_global_csv(root / f"H{hz}" / dname / "summary")
            if df is None:
                CE.logger.warning(f"  缺结果：{key} H={hz} @ {root / dname}")
                continue
            if "seed" not in df.columns or "RMSE_last" not in df.columns:
                CE.logger.warning(f"  {key} H={hz}：缺 seed/RMSE_last 列，跳过")
                continue
            df = df.copy()
            df["Model"] = key
            df["Horizon"] = hz
            all_rows.append(df)
    return pd.concat(all_rows, ignore_index=True) if all_rows else pd.DataFrame()


def merged_summary(df, zeros):
    """按 (Horizon, Model) 聚合：末步主口径均值/标准差 + skill + 可达分量。

    zeros: {hz: 零变化 RMSE_last}（与模型无关，由数据重建，见 check_anchors）。
    """
    recs = []
    for (hz, key), g in df.groupby(["Horizon", "Model"]):
        z = zeros.get(hz)
        rec = {"Horizon": hz, "Model": key,
               "ModelLabel": CE.MODEL_LABELS.get(key, NEW_LABELS.get(key, key)),
               "Family": CE.MODEL_FAMILY.get(key, "external_temporal"
                                             if key not in ("V7_Ours", "V0_TCN-Only")
                                             else "ablation" if key == "V0_TCN-Only"
                                             else "ours"),
               "N_Seeds": len(g)}
        for col in ("RMSE_last", "R2_last", "MAE_last", "RMSE_reach_last",
                    "RMSE_floor_last", "Params", "TrainTime"):
            if col in g.columns:
                rec[f"{col}_mean"] = g[col].mean()
                rec[f"{col}_std"] = g[col].std()
        rec["RMSE_zero_last"] = float(z) if z else np.nan
        rec["skill"] = float(1.0 - (rec["RMSE_last_mean"] / z) ** 2) if z else np.nan
        recs.append(rec)
    sm = pd.DataFrame(recs)
    return sm.sort_values(["Horizon", "RMSE_last_mean"])


def significance_vs_ours(df):
    """按视界、按种子对齐对 Ours 做配对 t（末步口径 + 可达分量）。"""
    recs = []
    for hz, g in df.groupby("Horizon"):
        base = g[g["Model"] == "V7_Ours"]
        if base.empty or "seed" not in base.columns:
            continue
        for key, sub in g.groupby("Model"):
            if key == "V7_Ours":
                continue
            common = sorted(set(base["seed"]) & set(sub["seed"]))
            if len(common) < 3:
                continue
            b = base.set_index("seed").loc[common]
            c = sub.set_index("seed").loc[common]
            br, cr = b["RMSE_last"].values, c["RMSE_last"].values
            t, p = ttest_rel(br, cr)
            sig = ("***" if p < 0.001 else "**" if p < 0.01
                   else "*" if p < 0.05 else "ns")
            rec = {"Horizon": hz,
                   "Comparison": f"{CE.MODEL_LABELS.get(key, NEW_LABELS.get(key, key))}"
                                 f" vs {CE.MODEL_LABELS['V7_Ours']}",
                   "N_Seeds": len(common),
                   "RMSE_last_new": float(cr.mean()),
                   "RMSE_last_ours": float(br.mean()),
                   "Delta_RMSE_pct": float((cr.mean() - br.mean()) / br.mean() * 100),
                   "t": float(t), "p": float(p), "Sig": sig,
                   "Cohens_d": float(CE.cohens_d(br, cr))}
            if "RMSE_reach_last" in b.columns and "RMSE_reach_last" in c.columns:
                brr, crr = b["RMSE_reach_last"].values, c["RMSE_reach_last"].values
                tr, pr = ttest_rel(brr, crr)
                rec["RMSE_reach_last_new"] = float(crr.mean())
                rec["RMSE_reach_last_ours"] = float(brr.mean())
                rec["Delta_RMSE_reach_pct"] = float(
                    (crr.mean() - brr.mean()) / brr.mean() * 100)
                rec["reach_t"] = float(tr)
                rec["reach_p"] = float(pr)
            recs.append(rec)
    return pd.DataFrame(recs)


# =============================================================================
# 锚点自检 + 调参记录
# =============================================================================
def true_cum_win(init_test, yte_raw):
    """重建真值累计位移窗口（无模型）。"""
    ns = yte_raw.shape[0]
    tc = np.zeros_like(yte_raw, dtype=np.float64)
    for i in range(ns):
        c = np.array(init_test[i], dtype=np.float64, copy=True)
        for t in range(yte_raw.shape[1]):
            c = c + yte_raw[i, t]
            tc[i, t] = c
    return tc


def check_anchors(cfg, shared=None):
    """H=10 零变化 + Ours 权威均值核对（锚点依赖锁定管线）。"""
    print("\n" + "=" * 100)
    print("[锚点自检] H=10")
    shared = shared if shared is not None else CE.prepare_shared_data(cfg)
    data = CE.build_temporal_samples(shared, cfg, 10)
    tc = true_cum_win(data["init_test"], data["yte_raw"])
    zero = float(np.sqrt(np.mean((data["init_test"] - tc[:, -1, :]) ** 2)))
    ok_z = abs(zero - ZERO_H10) <= 1e-3
    print(f"  零变化 RMSE_last = {zero:.6f}（期望 {ZERO_H10}）"
          f"{'[OK]' if ok_z else '[WARN]'}")
    if not ok_z:
        if EXAMPLE_MODE:
            print("  [WARN] EXAMPLE_MODE=1（合成示例数据）：跳过硬锚点终止。")
        else:
            raise SystemExit("[锚点失败] 零变化与锁定口径不符，停止（不得据此出结论）。")

    a = ABLATION_AUTH / f"H{10}" / "V7_wo_VerticalGCN" / "summary"
    df = load_global_csv(a)
    if df is None:
        print(f"  [WARN] 未找到权威 Ours 表：{a}（跳过逐位核验）")
    else:
        m, s = df["RMSE_last"].mean(), df["RMSE_last"].std()
        ok = abs(m - OURS_H10_EXPECT[0]) <= 5e-3
        print(f"  Ours(V7) 权威均值 = {m:.4f} ± {s:.4f}"
              f"（期望 {OURS_H10_EXPECT[0]} ± {OURS_H10_EXPECT[1]}）"
              f"{'[OK]' if ok else '[WARN]'}")
    print("=" * 100)
    return data


def write_budget(out_root, cfg, classic_source):
    """调参记录：统一预算 + 逐模型条目。"""
    UNIFIED_BUDGET["models"] = {
        "LSTM": {"family": "external_temporal", "width": 64, "arch": "LSTM 1 层双向? 否",
                 "arch_detail": "LSTM(48→64), n_layers=1, dropout=0.5", "source": classic_source},
        "BiLSTM-Attn": {"family": "external_temporal", "width": 64,
                        "arch_detail": "BiLSTM(64//2) 双向 + 注意力 + head，n_layers=1, dropout=0.5",
                        "source": classic_source,
                        "note": "宽度 64 口径（Comparison_Results_w64）；压力矩阵内为 w48 历史口径，已在正文声明"},
        "Transformer": {"family": "external_temporal", "width": 64,
                        "arch_detail": "TransformerEncoder d=64, heads=2, layers=1, dropout=0.5",
                        "source": classic_source},
        "V-Attention": {"family": "ours_framework", "width": 64,
                        "arch_detail": "Ours 骨干 + DepthSelfAttention(4 头) 替换竖向 GCN",
                        "source": classic_source},
        "V-MLP": {"family": "ours_framework", "width": 64,
                  "arch_detail": "Ours 骨干 + DepthMLP 替换竖向 GCN", "source": classic_source},
        "STGCN": {"family": "external_graph", "width": 64,
                  "arch_detail": "ST-Conv Block, Cheb K=2, temp kernel=3", "source": classic_source},
        "GraphWaveNet": {"family": "external_graph", "width": 64,
                         "arch_detail": "自适应邻接 + 扩散 GCN + 膨胀 TCN (diff=1, layers=2, dil=[1,2])",
                         "source": classic_source},
        "GBDT": {"family": "external_ml", "arch_detail": "LightGBM/XGBoost/HistGBR 自动择一；"
                  "n_est=300, depth=4, lr=0.05；temporal 输入展平 576 维；增量目标（PCA 系数）",
                 "source": classic_source,
                 "note": "本脚本补 GBDT-inc（滞后/滚动特征）作树模型强基线对照"},
        "DLinear": {"family": "external_temporal", "width": 64,
                    "arch_detail": f"趋势(全窗均值)+残差双线性，{cfg.LOOKBACK}·48→hz·K",
                    "source": "本脚本（Baseline_Budget_w64）"},
        "PatchTST": {"family": "external_temporal", "width": 64,
                     "arch_detail": f"patch={PATCH_LEN}, stride={PATCH_STRIDE}（{PATCH_LEN} 个 patch），"
                     f"共享 Transformer d={D_MODEL}, heads={N_HEADS}, layers={N_LAYERS}, norm_first",
                     "source": "本脚本（Baseline_Budget_w64）"},
        "iTransformer": {"family": "external_temporal", "width": 64,
                         "arch_detail": f"倒置 Transformer：48 变量 token，d={D_MODEL}, "
                         f"heads={N_HEADS}, layers={N_LAYERS}, norm_first",
                         "source": "本脚本（Baseline_Budget_w64）"},
        "GBDT-inc": {"family": "external_ml",
                     "arch_detail": "同 GBDT 预算（n_est=300, depth=4, lr=0.05）+ "
                     "滞后 PCA 系数(lag=1,2,3) + 滚动均值/std(w=3,6,12)；增量目标（PCA 系数）",
                     "source": "本脚本（Baseline_Budget_w64）", "horizon": f"H{GBDT_HZ} 仅"},
    }
    UNIFIED_BUDGET["classic_source"] = (
        "Comparison_Results_w64（2026-09-24 容量对齐，宽度 64，与本文同一配方）"
        if classic_source == "w64" else "Classic_Rerun_w64（本脚本 --also_classic 重训）")
    for d in (TABLES, out_root):
        with open(d / "T_p1215_budget.json", "w", encoding="utf-8") as f:
            json.dump(UNIFIED_BUDGET, f, indent=2, ensure_ascii=False)
    md_lines = ["# 外部基线统一预算调参记录",
                "",
                "| 项 | 值 |",
                "|---|---|",
                ]
    for k, v in UNIFIED_BUDGET.items():
        if k in ("models",):
            continue
        md_lines.append(f"| {k} | {json.dumps(v, ensure_ascii=False)} |")
    md_lines += ["", "## 逐模型条目", ""]
    md_lines.append("| 模型 | 族 | 架构 | 来源 |")
    md_lines.append("|---|---|---|---|")
    for m, rec in UNIFIED_BUDGET["models"].items():
        md_lines.append(f"| {m} | {rec['family']} | {rec['arch_detail']} | {rec['source']} |")
    md_lines.append("")
    md_lines.append("> 依据：统一宽度/输入/划分/同一调参预算并报告调参记录；补充 2021–2024 现代时序基线。")
    for d in (TABLES, out_root):
        with open(d / "T_p1215_budget.md", "w", encoding="utf-8") as f:
            f.write("\n".join(md_lines))
    print(f"[调参记录] 已写入 T_p1215_budget.{{json,md}}（{out_root} 与 {TABLES}）")


# =============================================================================
# main
# =============================================================================
def main():
    ap = argparse.ArgumentParser(description="外部基线统一预算 + 现代时序基线")
    ap.add_argument("--models", type=str, default=",".join(ALL_NEW),
                    help="逗号分隔：DLinear / PatchTST / iTransformer / GBDT-inc")
    ap.add_argument("--horizons", type=str, default=",".join(map(str, HORIZONS)))
    ap.add_argument("--nseeds", type=int, default=30, help="取锁定 SEEDS 前 N 个")
    ap.add_argument("--seeds", type=str, default=None, help="显式种子列表（逗号分隔）")
    ap.add_argument("--smoke", action="store_true", help="H=10 / 1 种子 / 3 epoch")
    ap.add_argument("--probe", action="store_true", help="H=10 / 2 种子 / 全 epoch")
    ap.add_argument("--also_classic", action="store_true",
                    help="同时按统一预算重训经典外部基线进 Classic_Rerun_w64")
    ap.add_argument("--classic_source", type=str, default="w64",
                    choices=["w64", "classic_rerun"],
                    help="合并汇总时经典基线的来源（默认 w64）")
    args = ap.parse_args()

    models = [m.strip() for m in args.models.split(",") if m.strip()]
    horizons = [int(x) for x in args.horizons.split(",") if x.strip()]
    seeds = ([int(s) for s in args.seeds.split(",")] if args.seeds
             else list(CE.Config.SEEDS)[:args.nseeds])

    cfg = CE.Config()
    cfg.HORIZONS = horizons
    cfg.SEEDS = seeds
    cfg.RESUME = RESUME
    out_root = OUT_ROOT
    if args.smoke:
        cfg.HORIZONS = [10]
        cfg.SEEDS = seeds[:1]
        cfg.EPOCHS = 3
        cfg.PATIENCE = 3
        out_root = OUT_ROOT / "_smoke"
    elif args.probe:
        cfg.HORIZONS = [10]
        cfg.SEEDS = seeds[:2]
        if not models:
            models = ALL_NEW
    if not models:
        models = ALL_NEW

    out_root.mkdir(parents=True, exist_ok=True)
    print("=" * 100)
    print(f"外部基线统一预算 {('冒烟' if args.smoke else '探针' if args.probe else '正式')}")
    print(f"  模型 = {models} | 视界 = {cfg.HORIZONS} | 种子 = {len(cfg.SEEDS)}"
          f"（{cfg.SEEDS[:3]}{'…' if len(cfg.SEEDS) > 3 else ''}）")
    print(f"  Device = {cfg.DEVICE} | 输出 = {out_root}")
    print("=" * 100)

    shared = CE.prepare_shared_data(cfg)
    check_anchors(cfg, shared)
    write_budget(out_root, cfg, args.classic_source)

    run_rows = []
    for hz in cfg.HORIZONS:
        for name in models:
            if name == GBDT_INC_MODEL and hz != GBDT_HZ:
                print(f"  跳过 {name} @ H={hz}（仅 H{GBDT_HZ}）")
                continue
            if name in NEW_TS_MODELS:
                df_g = run_new_temporal_model(cfg, name, shared, hz, out_root)
            elif name == GBDT_INC_MODEL:
                df_g = run_gbdt_inc(cfg, shared, hz, out_root)
            else:
                print(f"  未知模型 {name}，跳过")
                df_g = None
            if df_g is not None:
                df_g["Model"] = name
                df_g["Horizon"] = hz
                run_rows.append(df_g)

    if args.also_classic:
        CLASSIC_ROOT.mkdir(parents=True, exist_ok=True)
        print("\n[--also_classic] 按统一预算重训经典外部基线 → Classic_Rerun_w64")
        for hz in cfg.HORIZONS:
            for m in ["LSTM", "BiLSTM-Attn", "Transformer", "V-Attention",
                      "STGCN", "GraphWaveNet", "V-MLP", "GBDT"]:
                if m == "GBDT" and hz != GBDT_HZ:
                    continue
                old = cfg.OUT_ROOT
                cfg.OUT_ROOT = CLASSIC_ROOT
                CE.run_external_model(cfg, m, shared, hz)
                cfg.OUT_ROOT = old

    # ---- 合并汇总 + 显著性 ----
    if run_rows:
        new_runs = pd.concat(run_rows, ignore_index=True)
        new_runs.to_csv(TABLES / "T_p1215_new_runs.csv", index=False,
                        encoding="utf-8-sig")
        print(f"\n[已写出] T_p1215_new_runs.csv（新基线逐 run，"
              f"{len(new_runs)} 行）")
    df_all = assemble_models(cfg, out_root, classic_source=args.classic_source)
    if df_all.empty:
        print("\n[警告] 无任何可汇总结果（检查各来源目录）。")
        return
    zeros = {}
    for hz in cfg.HORIZONS:
        d = CE.build_temporal_samples(shared, cfg, hz)
        tc = true_cum_win(d["init_test"], d["yte_raw"])
        zeros[hz] = float(np.sqrt(np.mean((d["init_test"] - tc[:, -1, :]) ** 2)))
    sm = merged_summary(df_all, zeros)
    sig = significance_vs_ours(df_all)
    sm.to_csv(TABLES / "T_p1215_summary.csv", index=False, encoding="utf-8-sig")
    if not sig.empty:
        sig.to_csv(TABLES / "T_p1215_significance.csv", index=False,
                   encoding="utf-8-sig")

    print("\n" + "=" * 100)
    print("合并汇总（H=10，末步口径，种子均值）")
    print("=" * 100)
    h10 = sm[sm["Horizon"] == 10]
    for _, r in h10.iterrows():
        print(f"  {r['ModelLabel']:<18} {r['RMSE_last_mean']:>8.4f}±{r['RMSE_last_std']:.4f}"
              f"  skill={r['skill']:+.4f}  params={r.get('Params_mean', float('nan')):.0f}"
              f"  seeds={int(r['N_Seeds'])}")
    if not sig.empty:
        print("\n显著性（vs Ours，种子配对 t，末步口径）：")
        h10s = sig[sig["Horizon"] == 10]
        for _, r in h10s.iterrows():
            print(f"  {r['Comparison']:<30} Δ={r['Delta_RMSE_pct']:+7.2f}%  "
                  f"p={r['p']:.4f}  {r['Sig']}")
    print(f"\n[已写出] T_p1215_summary.csv / T_p1215_significance.csv（{TABLES}）")


if __name__ == "__main__":
    main()
