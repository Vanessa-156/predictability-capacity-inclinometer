# -*- coding: utf-8 -*-
"""持续性基线（Persistence Baseline）——零训练，纯统计计算。

原理：预测 H 步后的累计位移 = 当前累计位移（即"不变"）。
      RMSE = sqrt(mean((cum[t+H] - cum[t])^2))

输出：与 T1 对比实验主表对齐的 RMSE_last / R^2_last / MAE_last 行。
"""
from _env import WORK_ROOT


import sys, json, logging
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

PROJ = WORK_ROOT
if str(PROJ) not in sys.path:
    sys.path.insert(0, str(PROJ))

from Ablation_Experiment import (
    Config, load_raw_data_with_meta, load_water_data, extract_water_features,
    build_node_features, load_geology_params, assign_geo_to_nodes,
    denoise_split, make_features
)

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger('persist')

# =============================================================================
# 1.  Config
# =============================================================================
cfg = Config()
cfg.HORIZONS = [2, 6, 8, 10, 12]
cfg.LOOKBACK = 12

OUT = PROJ / '_paper_tables'
OUT.mkdir(parents=True, exist_ok=True)

# =============================================================================
# 2.  加载数据（完全复用 Ablation_Experiment 管线）
# =============================================================================
logger.info("加载原始数据 ...")
full_df = load_raw_data_with_meta(cfg)
ts = np.sort(full_df['测量时间'].unique())
wd, wa = load_water_data(ts, cfg)
water_feat = extract_water_features(wd, wa)
disp_raw, _, meta = build_node_features(full_df, cfg)
geo_records = load_geology_params(cfg)
geo_feat_raw = assign_geo_to_nodes(geo_records, meta['depths'], cfg)

n = len(ts)
tr_end = int(n * cfg.TRAIN_RATIO)
va_end = int(n * (cfg.TRAIN_RATIO + cfg.VAL_RATIO))
train_idx = np.arange(tr_end)
val_idx = np.arange(tr_end, va_end)
test_idx = np.arange(va_end, n)
logger.info(f"时间划分: train[0,{tr_end}) val[{tr_end},{va_end}) test[{va_end},{n}) "
            f"(共 {n} 步)")

# 去噪
train_disp = disp_raw[train_idx]
val_disp = disp_raw[val_idx]
test_disp = disp_raw[test_idx]
train_den, val_den, test_den = denoise_split(train_disp, val_disp, test_disp, cfg)

N_nodes = test_den.shape[1]   # 155
T_test = test_den.shape[0]    # 32
LB = cfg.LOOKBACK
logger.info(f"测试集: T={T_test}, N={N_nodes}")

# =============================================================================
# 3.  每视界计算持续性基线
# =============================================================================
rows = []
for H in cfg.HORIZONS:
    n_win = T_test - LB - H + 1
    if n_win <= 0:
        logger.warning(f"  H={H}: n_win={n_win}<=0, skip")
        continue

    # 每窗口最后一帧的累计位移 → H 步后真实累计位移
    init_idx = np.arange(LB - 1, LB - 1 + n_win)        # (n_win,)
    init = test_den[init_idx, :]                          # (n_win, 155)

    true_last = test_den[init_idx + H, :]                 # (n_win, 155)

    # 全步口径
    true_all = np.vstack(
        [test_den[init_idx + t + 1, :] for t in range(H)])  # (n_win*H, 155)
    pred_all = np.tile(init, (H, 1))                        # (n_win*H, 155)

    rmse       = float(np.sqrt(mean_squared_error(true_all.ravel(),  pred_all.ravel())))
    mae        = float(mean_absolute_error(true_all.ravel(), pred_all.ravel()))
    r2         = float(r2_score(true_all.ravel(),         pred_all.ravel()))
    rmse_last  = float(np.sqrt(mean_squared_error(true_last.ravel(), init.ravel())))
    mae_last   = float(mean_absolute_error(true_last.ravel(), init.ravel()))
    r2_last    = float(r2_score(true_last.ravel(),         init.ravel()))

    rows.append(dict(H=H, n_win=n_win,
                     RMSE=rmse, MAE=mae, R2=r2,
                     RMSE_last=rmse_last, MAE_last=mae_last, R2_last=r2_last))

df = pd.DataFrame(rows)

# =============================================================================
# 4.  打印
# =============================================================================
print()
print("=" * 72)
print("持续性基线  ——  末步口径（与 T1 对比实验主表对齐）")
print("=" * 72)
fmt_hdr = f"{'H':>4}  {'n_win':>6}  {'RMSE_last':>10}  {'MAE_last':>10}  {'R2_last':>10}"
print(fmt_hdr)
print("-" * 72)
for _, r in df.iterrows():
    print(f"{int(r['H']):>4}  {int(r['n_win']):>6}  "
          f"{r['RMSE_last']:>10.4f}  {r['MAE_last']:>10.4f}  {r['R2_last']:>+10.4f}")

print()
print("=" * 72)
print("持续性基线  ——  全步口径")
print("=" * 72)
fmt_hdr2 = f"{'H':>4}  {'RMSE':>10}  {'MAE':>10}  {'R2':>10}  {'RMSE_last':>10}  {'R2_last':>10}"
print(fmt_hdr2)
print("-" * 72)
for _, r in df.iterrows():
    print(f"{int(r['H']):>4}  {r['RMSE']:>10.4f}  {r['MAE']:>10.4f}  {r['R2']:>+10.4f}  "
          f"{r['RMSE_last']:>10.4f}  {r['R2_last']:>+10.4f}")

# =============================================================================
# 5.  与 T1 主表 Ours(=V7) 行横向对比（末步口径）
# =============================================================================
ours = {
    2:  (0.5895, 0.0279),
    6:  (1.1239, 0.0650),
    8:  (1.3693, 0.1099),
    10: (1.5687, 0.0940),
    12: (1.7517, 0.0982),
}
print()
print("=" * 72)
print("T1 对齐：Persist vs Ours（末步口径）")
print("=" * 72)
print(f"{'H':>4}  {'Persist':>10}  {'Ours':>18}  {'Persist/Ours':>14}  {'R2(Persist)':>12}")
print("-" * 72)
for H in cfg.HORIZONS:
    r = df[df['H'] == H].iloc[0]
    ou, os = ours.get(H, (None, None))
    ratio = r['RMSE_last'] / ou
    logger.info(
        f"{H:>4}  {r['RMSE_last']:>10.4f}  {ou:>8.4f} +/- {os:.4f}     "
        f"{ratio:>8.2f} x      {r['R2_last']:>+12.4f}")

# =============================================================================
# 6.  落盘
# =============================================================================
csv_path = OUT / 'T_persist_baseline.csv'
df.to_csv(csv_path, index=False, float_format='%.6f')
logger.info(f"\n保存 → {csv_path}")
logger.info("完成。")