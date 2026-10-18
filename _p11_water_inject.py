# -*- coding: utf-8 -*-
"""_p11_water_inject.py -- 水位注入的独立消融对照（轻量 pilot）。

背景：DA-6 指出 §4.1 声称"水位不纳入输入"有"独立的消融对照"支撑，但主文不存在该实验。
本脚本补上真实实验，协议与锚索注入 pilot（_anchor_gate_pilot.py）完全同构：
  水位深度（场地级标量，load_water_data 的 wd）按时间最近邻落在测斜栅格、
  训练期统计量 z-score 后广播到全部节点，作为额外 1 通道拼进 geo_static，
  PhysicsGate 门控宽度 +1，cross_attn 仍接收原地质特征（不改变其结构）。

实现要点（与 _anchor_gate_pilot.py 一致，不改动 Ablation_Experiment.py）：
  1. 子类 WaterAblationModel 覆盖 forward：拆包 geo -> gate(geo+water), cross_attn(geo)
  2. PhysicsGate 换成宽度 +1 的版本（WideGate）
  3. DataLoader 仍返回 4 元组（geo 列数 +1），复用 train_model / evaluate

运行（训练类）：
  python _p11_water_inject.py

预计 ~3 分钟（H6 × 5 种子 × 2 变体）。输出 <AUX_ROOT>\Water_Gate_Pilot\，不覆盖锚索目录。
"""
from _env import AUX_ROOT, WORK_ROOT

import sys, json, warnings
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import TensorDataset, DataLoader
warnings.filterwarnings('ignore')

sys.path.insert(0, str(WORK_ROOT))
import Ablation_Experiment as AE
from Ablation_Experiment import (Config, set_seed, logger, load_raw_data_with_meta,
                                 build_node_features, build_adjacency,
                                 load_geology_params, assign_geo_to_nodes,
                                 AblationModel, train_model, evaluate,
                                 global_metrics, aggregate_variant)

OUT = AUX_ROOT / "Water_Gate_Pilot"

cfg = Config()
cfg.HORIZONS = [6]
cfg.SEEDS = cfg.SEEDS[:5]
set_seed(42)

# ── 1) 实验数据 ──
logger.info("加载实验数据...")
full_df = load_raw_data_with_meta(cfg)
ts = pd.DatetimeIndex(np.sort(full_df['测量时间'].unique()))
disp_raw, _, meta = build_node_features(full_df, cfg)
geo_records = load_geology_params(cfg)
geo_feat_raw = assign_geo_to_nodes(geo_records, meta['depths'], cfg)
wd, wa = AE.load_water_data(ts, cfg)

n = len(ts)
tr_end = int(n * cfg.TRAIN_RATIO)
va_end = int(n * (cfg.TRAIN_RATIO + cfg.VAL_RATIO))
train_idx, val_idx, test_idx = np.arange(tr_end), np.arange(tr_end, va_end), np.arange(va_end, n)

N = len(meta['node_names'])

# ── 2) 水位特征 (T,) 场地级 -> (T, N) ──
logger.info("构建水位特征...")
wd_t = np.asarray(wd, float)
w_mu, w_sd = wd_t[train_idx].mean(), wd_t[train_idx].std() + 1e-6
wd_z = ((wd_t - w_mu) / w_sd).astype(np.float32)
water_t = torch.FloatTensor(np.tile(wd_z[:, None], (1, N)))      # (T, N)
logger.info(f"  水位深度 z-score 后范围 [{wd_z.min():.2f}, {wd_z.max():.2f}]")

# ── 3) 子类：gate 用 geo+water，cross_attn 用 geo ──
class WaterAblationModel(AblationModel):
    def forward(self, x, water_feat, geo_aug, vert_adj=None, node_mask=None):
        B, seq, Nodes, in_feat = x.shape
        tcn_out = self.tcn(x.permute(0, 2, 1, 3).reshape(B * Nodes, seq, in_feat))
        ctx = tcn_out.mean(dim=1) if self.use_attn else tcn_out[:, -1, :]
        ctx = ctx.reshape(B, Nodes, -1)
        if node_mask is not None and ctx.shape[1] == node_mask.shape[0]:
            ctx = ctx * node_mask.view(1, -1, 1).to(ctx.device)
        if self.use_vertical_gcn:
            vadj = vert_adj if vert_adj is not None else self.vert_adj
            if vadj is not None:
                ctx = self.vertical_gcn(ctx, vadj)
        if self.use_gcn:
            ctx = self.gcn(ctx, dynamic_adj=None)
            ctx = self.gcn_norm(ctx)
            ctx = self.gcn_drop(F.relu(ctx))
        geo_base, water_ch = geo_aug[..., :-1], geo_aug[..., -1:]
        if self.use_physics_gate:
            ctx = self.physics_gate(ctx, torch.cat([geo_base, water_ch], dim=-1))
        if self.use_cross_attn:
            ctx = self.cross_attn(ctx, geo_base)
        return self.head(ctx.mean(dim=1)).reshape(B, self.horizon, self.out_dim), None

class WideGate(nn.Module):
    def __init__(self, dim, extra_dim):
        super().__init__()
        self.gate = nn.Sequential(nn.Linear(dim + extra_dim, dim), nn.Sigmoid())

    def forward(self, x, feat):
        return x * self.gate(torch.cat([x, feat], dim=-1))

# ── 4) 训练 ──
OUT_H6 = OUT / "H6"

def build_water_windows(seg_idx):
    """把 (T,N) 水位序列按窗口切片 -> (n_win, N, 1)，取窗口最后一个输入时刻。"""
    sub = water_t[seg_idx]                     # (T_seg, N)
    n_win = len(seg_idx) - cfg.LOOKBACK - cfg.HORIZONS[0] + 1
    if n_win <= 0:
        return torch.zeros((0, sub.shape[1], 1))
    idx = torch.arange(n_win) + cfg.LOOKBACK - 1
    return sub[idx].unsqueeze(-1)               # (n_win, N, 1)

VARIANTS = {"V5_WaterInject": True, "V5_Baseline": False}
cfg.DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
logger.info(f"Device: {cfg.DEVICE}")

for v_name, use_water in VARIANTS.items():
    logger.info(f"\n======== {v_name} (water={use_water}) ========")
    out_dir = OUT_H6 / v_name
    out_dir.mkdir(parents=True, exist_ok=True)

    for seed in cfg.SEEDS:
        run_dir = out_dir / f"run_{seed}"
        if (run_dir / "metrics.json").exists():
            logger.info(f"  Skip {seed}")
            continue
        run_dir.mkdir(parents=True, exist_ok=True)
        set_seed(seed)
        cfg.DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        data = AE.prepare_data(cfg, full_df, wd, wa, meta,
                               train_idx, val_idx, test_idx,
                               AE.extract_water_features(wd, wa), geo_feat_raw,
                               cfg.HORIZONS[0])
        geo_t = torch.FloatTensor(data['geo_static']).to(cfg.DEVICE)   # (N, geo_dim)
        geo_dim = data['geo_dim_actual']

        def _make_ld(X, y, wf, seg_idx, shuffle=False, with_water=False):
            if with_water:
                ww = build_water_windows(seg_idx).to(cfg.DEVICE)       # (n_win, N, 1)
                g = torch.cat([geo_t.unsqueeze(0).expand(len(X), -1, -1), ww], dim=-1)
            else:
                g = geo_t.unsqueeze(0).expand(len(X), -1, -1)          # (n_win, N, geo_dim)
            ds = TensorDataset(torch.FloatTensor(X), torch.FloatTensor(y),
                               torch.FloatTensor(wf), g.cpu())
            return DataLoader(ds, batch_size=cfg.BATCH_SIZE, shuffle=shuffle)

        trl = _make_ld(data['Xtr'], data['ytr'], data['wf_train'], train_idx,
                       shuffle=True, with_water=use_water)
        vll = _make_ld(data['Xva'], data['yva'], data['wf_val'], val_idx,
                       with_water=use_water)
        tel = _make_ld(data['Xte'], data['yte'], data['wf_test'], test_idx,
                       with_water=use_water)

        adj = build_adjacency(meta, geo_feat_raw, cfg)
        vert = AE.build_vertical_adjacency(meta, cfg)
        base_kw = dict(N=N, in_dim=data['F_in'], gcn_hid=cfg.GCN_HIDDEN,
                       tcn_ch=cfg.TCN_CHANNELS, k=cfg.TCN_KERNEL_SIZE,
                       dil=cfg.TCN_DILATIONS, attn_hid=cfg.ATTENTION_HIDDEN,
                       hor=cfg.HORIZONS[0], adj=torch.FloatTensor(adj).to(cfg.DEVICE),
                       out_dim=data['out_dim'], geo_feat_dim=geo_dim,
                       vert_adj=torch.FloatTensor(vert).to(cfg.DEVICE))

        if use_water:
            model = WaterAblationModel(cfg, **base_kw).to(cfg.DEVICE)
            w0 = model.physics_gate.gate[0].weight
            cur_dim = w0.shape[0]
            wide = WideGate(cur_dim, geo_dim + 1).to(cfg.DEVICE)
            sd_new, sd_old = wide.state_dict(), model.physics_gate.state_dict()
            for k in sd_old:
                if k in sd_new and sd_old[k].shape == sd_new[k].shape:
                    sd_new[k].copy_(sd_old[k])
            wide.load_state_dict(sd_new)
            model.physics_gate = wide
        else:
            model = AblationModel(cfg, **base_kw).to(cfg.DEVICE)

        mt = torch.FloatTensor(data['valid_mask']).to(cfg.DEVICE)
        model, tr_l, va_l, t_time = train_model(cfg, model, trl, vll,
                                                run_dir / "best.pth", node_mask=mt)
        pred_cum, true_cum = evaluate(model, tel, data['init_test'], data['scY'],
                                       data['pca_y'], cfg.HORIZONS[0], cfg.DEVICE,
                                       node_mask=mt, true_diff_win=data['yte_raw'])
        gm = global_metrics(true_cum, pred_cum)
        gm['seed'] = seed
        gm['train_time'] = t_time
        with open(run_dir / "metrics.json", "w") as fh:
            json.dump(gm, fh, indent=2)
        logger.info(f"  {v_name} seed={seed} RMSE_last={gm.get('RMSE_last', 0):.4f}")

    aggregate_variant(out_dir)

# ── 5) 汇总与配对检验 ──
print(f"\n{'=' * 70}\n对照（H6 × 5 种子）\n{'=' * 70}")
for v_name in VARIANTS:
    d = OUT_H6 / v_name / "summary"
    cands = list(d.glob("global_*runs.csv"))
    if not cands:
        continue
    df = pd.read_csv(max(cands, key=lambda f: f.stat().st_mtime))
    r, r2 = df['RMSE_last'].values, df['R2_last'].values
    print(f"  {v_name:<18} RMSE_last={r.mean():.4f}±{r.std(ddof=1):.4f}  "
          f"R²_last={r2.mean():.4f}±{r2.std(ddof=1):.4f}")

def _load(v):
    d = OUT_H6 / v / "summary"
    c = list(d.glob("global_*runs.csv"))
    return pd.read_csv(max(c, key=lambda f: f.stat().st_mtime)).sort_values('seed') if c else None

a, b = _load("V5_WaterInject"), _load("V5_Baseline")
if a is not None and b is not None:
    from scipy import stats
    t, p = stats.ttest_rel(a['RMSE_last'].values, b['RMSE_last'].values)
    print(f"\n  配对 t 检验 (V5_WaterInject vs V5_Baseline): t={t:.3f}, p={p:.4f}")
    print(f"  变化: {(a['RMSE_last'].mean()/b['RMSE_last'].mean()-1)*100:+.2f}%")

print(f"\n  参考（Ablation_New_5d_mid H6, 30 种子）:")
for v in ['V5_Full_Model', 'V3_wo_PhysicsGate']:
    d = AUX_ROOT / "Ablation_New_5d_mid" / "H6" / v / "summary"
    c = list(d.glob("global_*runs.csv"))
    if c:
        df = pd.read_csv(max(c, key=lambda f: f.stat().st_mtime))
        print(f"  {v:<20} RMSE_last={df['RMSE_last'].mean():.4f}±{df['RMSE_last'].std(ddof=1):.4f}  n={len(df)}")
