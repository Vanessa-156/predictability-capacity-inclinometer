# -*- coding: utf-8 -*-
"""_k_scan_probe.py — 验证「提高 K 能否抬高 R² 天花板」。

两阶段：
  Stage 1（总是执行，零训练）：对一组 K 直接计算输出空间地板与 **R² 理论上限**
      R² 上限(K) = 1 − RMSE_floor_last(K)² / Var(末步累计位移)
  Stage 2（设 PROBE_K 时执行）：用 monkey-patch 把 `prepare_data` 里的 `PCA(0.90)`
      替换为固定 K，训练 V5（n 种子），报告实际 R²_last 并与该 K 的上限对照，
      判断"实际能否逼近新天花板"（若逼近 ⇒ 提高 K 有效；若远离 ⇒ 是模型能力瓶颈）。

用法：
  # Stage 1（天花板曲线）
  python -u _k_scan_probe.py
  # Stage 2（逐 K 训练；PowerShell 循环）
  $env:PROBE_K='30'; python -u _k_scan_probe.py

注意：patch 必须在 `import Ablation_Experiment` **之前**生效
（AE 里 `from sklearn.decomposition import PCA` 在导入时绑定）。
Stage 1 使用**未经 patch** 的原始 PCA 类（`_RawPCA`），以便自行扫描 K。
"""
from _env import WORK_ROOT

import os
import sys
from pathlib import Path

import numpy as np

_K_ENV = os.environ.get('PROBE_K', '').strip()
K_TARGET = int(_K_ENV) if _K_ENV.isdigit() else None

if K_TARGET:
    import sklearn.decomposition as _sd
    _OrigPCA = _sd.PCA

    def _PatchedPCA(n_components=None, **kw):
        """把方差阈值写法 PCA(0.90) 替换为固定成分数 K_TARGET。

        返回**真正的** sklearn PCA 实例（而非子类），以避开 sklearn 对
        `__init__` 签名的校验（不允许 *args / **kwargs）。
        """
        if isinstance(n_components, float) and 0.0 < n_components < 1.0:
            n_components = K_TARGET
        return _OrigPCA(n_components=n_components, **kw)

    _sd.PCA = _PatchedPCA
    _RawPCA = _OrigPCA
else:
    from sklearn.decomposition import PCA as _RawPCA

import torch
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset

sys.path.insert(0, str(WORK_ROOT))
import Ablation_Experiment as AE

HZ = 10
SEEDS = [42, 15, 77, 85, 120]
KS = [13, 20, 30, 50, 80, 120, 149]
OUT = WORK_ROOT / "_k_scan_probe"


def load_ctx():
    cfg = AE.Config()
    AE.set_seed(42)
    full_df = AE.load_raw_data_with_meta(cfg)
    ts = np.sort(full_df['测量时间'].unique())
    wd, wa = AE.load_water_data(ts, cfg)
    water_feat = AE.extract_water_features(wd, wa)
    disp_raw, _, meta = AE.build_node_features(full_df, cfg)
    geo = AE.load_geology_params(cfg)
    geo_raw = AE.assign_geo_to_nodes(geo, meta['depths'], cfg) if geo else None
    n = len(ts)
    tr_end = int(n * cfg.TRAIN_RATIO)
    va_end = int(n * (cfg.TRAIN_RATIO + cfg.VAL_RATIO))
    return {'cfg': cfg, 'full_df': full_df, 'wd': wd, 'wa': wa, 'meta': meta,
            'water_feat': water_feat, 'disp_raw': disp_raw,
            'geo_raw': geo_raw,
            'geo_fb': geo_raw if geo_raw is not None else np.zeros(
                (len(meta['depths']), cfg.GEO_FEAT_DIM), dtype=np.float32),
            'train_idx': np.arange(tr_end), 'val_idx': np.arange(tr_end, va_end),
            'test_idx': np.arange(va_end, n), 'n': n}


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    ctx = load_ctx()
    c = AE.Config()
    for k, v in AE.ABLATION_VARIANTS['V5_Full_Model'].items():
        setattr(c, k, v)
    d = AE.prepare_data(c, ctx['full_df'], ctx['wd'], ctx['wa'], ctx['meta'],
                        ctx['train_idx'], ctx['val_idx'], ctx['test_idx'],
                        ctx['water_feat'], ctx['geo_fb'], HZ, disp_raw=ctx['disp_raw'])
    yte_raw, init_test = d['yte_raw'], d['init_test']
    ns, hz, N = yte_raw.shape

    # 训练期真值（用于各 K 的 PCA 拟合）
    den_tr, den_va, den_te = AE.denoise_split(ctx['disp_raw'][ctx['train_idx']],
                                              ctx['disp_raw'][ctx['val_idx']],
                                              ctx['disp_raw'][ctx['test_idx']], c)
    all_den = np.vstack([den_tr, den_va, den_te])
    diff1 = np.zeros_like(all_den)
    diff1[1:] = np.diff(all_den, axis=0)
    y_train = diff1[ctx['train_idx']]

    # 末步累计真值方差（与 R² 定义一致，用于换算上限）
    cum = np.zeros((ns, hz, N), dtype=np.float64)
    for i in range(ns):
        cc = np.array(init_test[i], dtype=np.float64)
        for t in range(hz):
            cc = cc + yte_raw[i, t]
            cum[i, t] = cc
    var_last = float(cum[:, -1, :].var())

    print(f"{'='*96}")
    print(f"===== Stage 1：R² 理论天花板随 K 的变化（零训练，末步累计口径）=====")
    print(f"末步累计真值方差 Var = {var_last:.4f}（n_win={ns}, N={N}）")
    print(f"{'K':>5s} {'RMSE_floor_last':>17s} {'地板/Var':>10s} {'R² 上限':>10s} "
          f"{'相对 K=13 提升':>15s}")
    cap0 = None
    caps = {}
    for K in KS:
        kk = int(min(K, y_train.shape[0] - 1, N))
        pca_K = _RawPCA(n_components=kk).fit(y_train)
        scY_K = StandardScaler().fit(pca_K.transform(y_train))
        fl = AE.output_space_floor(yte_raw, init_test, scY_K, pca_K, hz, ns=ns)
        cap = 1 - fl['MSE_floor_last'] / var_last
        caps[K] = {'K_used': kk, 'rmse_floor_last': fl['RMSE_floor_last'],
                   'cap': cap, 'var': var_last}
        if cap0 is None:
            cap0 = cap
        if (not K_TARGET) or K in (13, K_TARGET):
            print(f"{K:5d} {fl['RMSE_floor_last']:17.4f} "
                  f"{fl['MSE_floor_last'] / var_last:10.4f} {cap:10.4f} "
                  f"{cap - cap0:+15.4f}" + ("   ← 主实验 K" if K == 13 else ""))

    print(f"\n结论：K 由 13 提到 80，R² 上限由 {caps[13]['cap']:.4f} 升至 "
          f"{caps[80]['cap']:.4f}（+{caps[80]['cap'] - caps[13]['cap']:.4f}）；"
          f"提到 149 为 {caps[149]['cap']:.4f}")

    # ---------------- Stage 2 ----------------
    if not K_TARGET:
        print(f"\n（未设 PROBE_K，跳过 Stage 2 训练验证）")
        return

    if K_TARGET not in caps:
        print(f"\n⚠️ PROBE_K={K_TARGET} 不在 Stage 1 扫描列表内，仅按同法自行计算其上限")
    cap_t = caps.get(K_TARGET, {}).get('cap', float('nan'))
    print(f"\n{'='*96}")
    print(f"===== Stage 2：以 K={K_TARGET} 训练 V5（n={len(SEEDS)}），"
          f"检验实际 R² 能否逼近上限 {cap_t:.4f} =====")
    d2 = AE.prepare_data(c, ctx['full_df'], ctx['wd'], ctx['wa'], ctx['meta'],
                         ctx['train_idx'], ctx['val_idx'], ctx['test_idx'],
                         ctx['water_feat'], ctx['geo_fb'], HZ, disp_raw=ctx['disp_raw'])
    print(f"patch 生效检查：prepare_data 的 out_dim = {d2['out_dim']}"
          f"（期望 {K_TARGET}）")
    assert d2['out_dim'] == K_TARGET, "monkey-patch 未生效"

    adj = AE.build_adjacency(ctx['meta'], ctx['geo_raw'], c)
    adj_t = torch.FloatTensor(adj).to(c.DEVICE)
    vert = AE.build_vertical_adjacency(ctx['meta'], c)
    vert_t = torch.FloatTensor(vert).to(c.DEVICE) if vert is not None else None
    g_t = torch.FloatTensor(d2['geo_static']).to(c.DEVICE)
    Nn = d2['N']
    mt = torch.FloatTensor(d2['valid_mask']).to(c.DEVICE)

    def mk(X, y, wf):
        return DataLoader(TensorDataset(
            torch.FloatTensor(X), torch.FloatTensor(y), torch.FloatTensor(wf),
            g_t.unsqueeze(0).expand(X.shape[0], Nn, -1)), batch_size=c.BATCH_SIZE)

    trl, vll, tel = (mk(d2['Xtr'], d2['ytr'], d2['wf_train']),
                     mk(d2['Xva'], d2['yva'], d2['wf_val']),
                     mk(d2['Xte'], d2['yte'], d2['wf_test']))
    rdir = OUT / f"K{K_TARGET}"
    rdir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'seed':>6s} {'RMSE_last':>11s} {'可达':>10s} {'地板':>10s} "
          f"{'R²_last':>9s} {'距上限':>9s}")
    rows = []
    for seed in SEEDS:
        AE.set_seed(seed)
        m = AE.AblationModel(c, N=Nn, in_dim=d2['F_in'], gcn_hid=c.GCN_HIDDEN,
                             tcn_ch=c.TCN_CHANNELS, k=c.TCN_KERNEL_SIZE,
                             dil=c.TCN_DILATIONS, attn_hid=c.ATTENTION_HIDDEN,
                             hor=HZ, adj=adj_t, out_dim=d2['out_dim'],
                             geo_feat_dim=d2['geo_dim_actual'],
                             vert_adj=vert_t).to(c.DEVICE)
        m, *_ = AE.train_model(c, m, trl, vll, rdir / f"run_{seed}.pth", node_mask=mt)
        pc, tc = AE.evaluate(m, tel, d2['init_test'], d2['scY'], d2['pca_y'], HZ,
                             c.DEVICE, node_mask=mt, true_diff_win=d2['yte_raw'])
        gm = AE.global_metrics(tc, pc)
        fl = AE.output_space_floor(d2['yte_raw'], d2['init_test'], d2['scY'],
                                   d2['pca_y'], HZ, ns=pc.shape[0])
        reach = float(np.sqrt(max(gm['RMSE_last'] ** 2 - fl['MSE_floor_last'], 0.0)))
        _num = ((tc[:, -1, :] - pc[:, -1, :]) ** 2).mean()
        _den = ((tc[:, -1, :] - tc[:, -1, :].mean()) ** 2).mean()
        r2 = float(1 - _num / _den)
        rows.append({'seed': seed, 'rmse': gm['RMSE_last'], 'reach': reach,
                     'floor': fl['RMSE_floor_last'], 'r2': r2})
        print(f"{seed:6d} {gm['RMSE_last']:11.4f} {reach:10.4f} "
              f"{fl['RMSE_floor_last']:10.4f} {r2:9.4f} {r2 - cap_t:9.4f}")
        del m
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    mr = float(np.mean([x['r2'] for x in rows]))
    mm = float(np.mean([x['rmse'] for x in rows]))
    print(f"\n均值：RMSE_last={mm:.4f}  R²_last={mr:.4f}  "
          f"（该 K 的理论上限 {cap_t:.4f}，差距 {mr - cap_t:+.4f}）")
    print(f"对照 K=13：RMSE_last=1.5890  R²_last=0.8798  上限 {caps[13]['cap']:.4f}")

    import json
    json.dump({'stage1': {str(k): v for k, v in caps.items()},
               'stage2': {'K': K_TARGET, 'cap': cap_t, 'rows': rows,
                          'mean_rmse': mm, 'mean_r2': mr, 'var_last': var_last}},
              open(OUT / f'k_scan_K{K_TARGET}.json', 'w', encoding='utf-8'),
              ensure_ascii=False, indent=2,
              default=lambda o: o.item() if hasattr(o, 'item') else str(o))
    print("->", OUT / f'k_scan_K{K_TARGET}.json')


if __name__ == '__main__':
    main()