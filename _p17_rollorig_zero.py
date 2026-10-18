# _p17_rollorig_zero.py - rolling-origin 零变化基线（零训练）。
# 在 6 折 walk-forward 的各测试块上计算零变化(persistence) RMSE_last，
# 与既有 Ours/V0 跨折结果（T_p02_walkforward.csv）比较，给出跨折 skill。

from _env import WORK_ROOT
import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
sys.path.insert(0, str(WORK_ROOT))
import copy
from pathlib import Path
import numpy as np
import pandas as pd
import Ablation_Experiment as AE

PROJ = WORK_ROOT
N_TOTAL = 213
BOUNDS = [57, 83, 109, 135, 161, 187, 213]
VAL_SIZE = 30
LOOKBACK = 12
HORIZONS = [2, 10]

cfg = AE.Config()
cfg.LOOKBACK = LOOKBACK
AE.set_seed(42)
full_df = AE.load_raw_data_with_meta(cfg)
ts = np.sort(full_df['测量时间'].unique())
wd, wa = AE.load_water_data(ts, cfg)
water_feat = AE.extract_water_features(wd, wa)
disp_raw, _, meta = AE.build_node_features(full_df, cfg)
geo_records = AE.load_geology_params(cfg)
geo_fb = (AE.assign_geo_to_nodes(geo_records, meta['depths'], cfg)
          if geo_records else np.zeros((len(meta['depths']), cfg.GEO_FEAT_DIM), dtype=np.float32))

rows = []
for k in range(1, 7):
    test_idx = np.arange(BOUNDS[k-1], BOUNDS[k])
    region_end = BOUNDS[k-1]
    val_idx = np.arange(region_end - VAL_SIZE, region_end)
    train_idx = np.arange(0, region_end - VAL_SIZE)
    for hz in HORIZONS:
        data = AE.prepare_data(cfg, full_df, wd, wa, meta, train_idx, val_idx,
                               test_idx, water_feat, geo_fb, hz,
                               disp_raw=disp_raw)
        yraw = data['yte_raw']      # (n_win, H, N) 真值增量（去噪口径）
        init = data['init_test']    # (n_win, N) 每窗起始绝对位移
        n_win = yraw.shape[0]
        if n_win == 0:
            continue
        true_cum_last = init + yraw.sum(axis=1)          # (n_win, N)
        pred_zero_last = init.copy()                      # 零变化：末步=起始
        err = (pred_zero_last - true_cum_last)
        rmse_last_zero = float(np.sqrt(np.mean(err**2)))
        rows.append(dict(fold=k, H=hz, n_win=n_win, RMSE_last_zero=rmse_last_zero))

z = pd.DataFrame(rows)
z.to_csv(Path(PROJ) / "_paper_tables" / "T_p17_rollorig_zero.csv", index=False,
         float_format='%.6f')

wf = pd.read_csv(Path(PROJ) / "_paper_tables" / "T_p02_walkforward.csv")
print('fold  H  零变化RMSE | Ours均值 | V0均值 | Ours skill | V0 skill')
sumz = {'Ours': [], 'V0': [], 'foldz': []}
for k in range(1, 7):
    for hz in HORIZONS:
        zz = z[(z['fold'] == k) & (z['H'] == hz)]['RMSE_last_zero'].values[0]
        o = wf[(wf['fold'] == k) & (wf['H'] == hz) & (wf['model'] == 'Ours')]['RMSE_last'].values
        v = wf[(wf['fold'] == k) & (wf['H'] == hz) & (wf['model'] == 'V0')]['RMSE_last'].values
        om = np.nanmean(o); vm = np.nanmean(v)
        os_ = 1 - om**2/zz**2; vs_ = 1 - vm**2/zz**2
        sumz['Ours'].append(os_); sumz['V0'].append(vs_); sumz['foldz'].append(zz)
        print(f'{k}  {hz}  {zz:.4f} | {om:.4f} | {vm:.4f} | {os_:+.3f} | {vs_:+.3f}')
print('\n跨折均值(6折): 零变化RMSE=%.4f  Ours skill=%+.3f  V0 skill=%+.3f'
      % (np.mean(sumz['foldz']), np.mean(sumz['Ours']), np.mean(sumz['V0'])))
