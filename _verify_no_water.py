r"""方向B最小验证：真正"水位全关" vs V4(仅关投影) vs V5(完整)。

澄清：现行 Ablation_Experiment.make_features 把 8 维水位特征直接拼进 TCN 基准输入
(use_water 在 prepare_data 内写死 True)，因此 V4_wo_WaterMod 只是去掉了 GCN 前的
water_node_proj 投影分支，"水文增强"并未真正隔离。本脚本补一个在特征层就把水位
剔除的变体，用于回答："水位信息整体是否带来正贡献？"

策略：
  full        : 与消融 V5 一致（TCN 输入含水位 + water_node_proj 分支）
  v4_nowater  : 与消融 V4 一致（TCN 输入含水位，无投影像投影分支）
  true_nowater: 特征层 use_water=False（TCN 输入不含水位）+ 无投影分支

只做 H={2,6,10} x seed={42,15}，快速判断趋势，不下最终结论。
"""
import sys, json, copy
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

ROOT = Path(__file__).parent
OUT = ROOT / "_verify_no_water"
OUT.mkdir(parents=True, exist_ok=True)

sys.path.insert(0, str(ROOT))
from Ablation_Experiment import (Config, set_seed,
    load_raw_data_with_meta, load_water_data, extract_water_features,
    build_node_features, load_geology_params, assign_geo_to_nodes,
    build_adjacency, build_vertical_adjacency, denoise_split, make_features,
    AblationModel, MultiLoss, train_model, evaluate, global_metrics,
    count_params)


def prepare_data_uw(cfg, full_df, wd, wa, meta, train_idx, val_idx, test_idx,
                    water_feat, geo_feat, hz, use_water):
    """复制 prepare_data，额外带 use_water 开关控制 TCN 基准输入是否含水。"""
    disp_raw, diff1_all, _ = build_node_features(full_df, cfg)
    valid_mask = meta.get('valid_mask', np.ones(disp_raw.shape[1], dtype=np.float32))
    train_disp = disp_raw[train_idx]; val_disp = disp_raw[val_idx]; test_disp = disp_raw[test_idx]
    train_den, val_den, test_den = denoise_split(train_disp, val_disp, test_disp, cfg)
    all_den = np.vstack([train_den, val_den, test_den])
    diff1_den = np.zeros_like(all_den)
    if len(meta['timestamps']) > 1:
        diff1_den[1:] = np.diff(all_den, axis=0)
    geo_static_raw = np.concatenate([geo_feat, (meta['depths'] / 16.0).reshape(-1, 1)], axis=1)
    geo_static = (geo_static_raw - geo_static_raw.mean(0)) / (geo_static_raw.std(0) + 1e-6)
    tr = make_features(train_den, diff1_den[train_idx], meta, water_feat[train_idx],
                       geo_static, cfg, use_water)
    va = make_features(val_den, diff1_den[val_idx], meta, water_feat[val_idx],
                       geo_static, cfg, use_water)
    te = make_features(test_den, diff1_den[test_idx], meta, water_feat[test_idx],
                       geo_static, cfg, use_water)
    F = tr.shape[-1]
    y_train = diff1_den[train_idx]; y_val = diff1_den[val_idx]; y_test = diff1_den[test_idx]
    from sklearn.preprocessing import StandardScaler
    from sklearn.decomposition import PCA
    scX = StandardScaler().fit(tr.reshape(-1, F))
    tr_norm = scX.transform(tr.reshape(-1, F)).reshape(tr.shape)
    va_norm = scX.transform(va.reshape(-1, F)).reshape(va.shape)
    te_norm = scX.transform(te.reshape(-1, F)).reshape(te.shape)
    pca_y = PCA(0.90).fit(y_train); out_dim = pca_y.n_components_
    ytr_pca = pca_y.transform(y_train); yva_pca = pca_y.transform(y_val); yte_pca = pca_y.transform(y_test)
    scY = StandardScaler().fit(ytr_pca)
    ytr_norm = scY.transform(ytr_pca); yva_norm = scY.transform(yva_pca); yte_norm = scY.transform(yte_pca)

    def ms(feat, tgt, lb, hval):
        X, y = [], []
        for i in range(len(feat) - lb - hval + 1):
            X.append(feat[i:i + lb]); y.append(tgt[i + lb:i + lb + hval])
        return np.array(X, dtype=np.float32), np.array(y, dtype=np.float32)

    Xtr, ytr = ms(tr_norm, ytr_norm, cfg.LOOKBACK, hz)
    Xva, yva = ms(va_norm, yva_norm, cfg.LOOKBACK, hz)
    Xte, yte = ms(te_norm, yte_norm, cfg.LOOKBACK, hz)
    init_test = test_den[cfg.LOOKBACK - 1:-hz][:Xte.shape[0]]
    N = train_disp.shape[1]
    w_mean = water_feat[train_idx].mean(0); w_std = water_feat[train_idx].std(0) + 1e-6
    water_feat_s = (water_feat - w_mean) / w_std
    wf_train = water_feat_s[train_idx][cfg.LOOKBACK - 1:-hz][:Xtr.shape[0]]
    wf_val = water_feat_s[val_idx][cfg.LOOKBACK - 1:-hz][:Xva.shape[0]]
    wf_test = water_feat_s[test_idx][cfg.LOOKBACK - 1:-hz][:Xte.shape[0]]
    wf_train = np.tile(wf_train[:, np.newaxis, :], (1, N, 1))
    wf_val = np.tile(wf_val[:, np.newaxis, :], (1, N, 1))
    wf_test = np.tile(wf_test[:, np.newaxis, :], (1, N, 1))
    return {'Xtr': Xtr, 'ytr': ytr, 'Xva': Xva, 'yva': yva, 'Xte': Xte, 'yte': yte,
            'init_test': init_test, 'scX': scX, 'scY': scY, 'pca_y': pca_y,
            'out_dim': out_dim, 'F_in': F, 'N': N,
            'wf_train': wf_train, 'wf_val': wf_val, 'wf_test': wf_test,
            'geo_static': geo_static, 'geo_dim_actual': geo_static.shape[1],
            'valid_mask': valid_mask}


def run_one(cfg, seed, hz, use_water_gcn=1, water_in_feat=True,  tag=""):
    """use_water_gcn: 是否带 water_node_proj 投影分支；water_in_feat: TCN 输入是否含水。"""
    set_seed(42)
    full_df = load_raw_data_with_meta(cfg)
    ts = np.array(full_df['测量时间'].unique())
    wd, wa = load_water_data(ts, cfg)
    water_feat = extract_water_features(wd, wa)
    _, _, meta = build_node_features(full_df, cfg)
    geo_records = load_geology_params(cfg)
    geo_feat_raw = assign_geo_to_nodes(geo_records, meta['depths'], cfg) if geo_records else None
    n = len(ts)
    train_idx = np.arange(int(n * cfg.TRAIN_RATIO))
    val_idx = np.arange(int(n * cfg.TRAIN_RATIO), int(n * (cfg.TRAIN_RATIO + cfg.VAL_RATIO)))
    test_idx = np.arange(int(n * (cfg.TRAIN_RATIO + cfg.VAL_RATIO)), n)
    geo_fb = geo_feat_raw if geo_feat_raw is not None else np.zeros(
        (len(meta['depths']), cfg.GEO_FEAT_DIM), dtype=np.float32)
    data = prepare_data_uw(cfg, full_df, wd, wa, meta, train_idx, val_idx, test_idx,
                           water_feat, geo_fb, hz, use_water=water_in_feat)
    geo_fb_use = geo_feat_raw if geo_feat_raw is not None and np.any(geo_feat_raw) else None
    adj = build_adjacency(meta, geo_fb_use, cfg)
    adj_t = torch.FloatTensor(adj).to(cfg.DEVICE)
    vert_adj = build_vertical_adjacency(meta, cfg)
    vert_adj_t = torch.FloatTensor(vert_adj).to(cfg.DEVICE) if vert_adj is not None else None

    N = data['N']; F_in = data['F_in']; out_dim = data['out_dim']
    geo_static_t = torch.FloatTensor(data['geo_static']).to(cfg.DEVICE)
    valid_mask_t = torch.FloatTensor(data['valid_mask']).to(cfg.DEVICE)

    cfg_use = copy.deepcopy(cfg)
    cfg_use.USE_WATER_GCN_ENHANCE = bool(use_water_gcn)

    def ds(X, y, wf):
        return TensorDataset(torch.FloatTensor(X), torch.FloatTensor(y),
                             torch.FloatTensor(wf),
                             geo_static_t.unsqueeze(0).expand(X.shape[0], N, -1))
    train_ld = DataLoader(ds(data['Xtr'], data['ytr'], data['wf_train']), batch_size=cfg.BATCH_SIZE, shuffle=True)
    val_ld = DataLoader(ds(data['Xva'], data['yva'], data['wf_val']), batch_size=cfg.BATCH_SIZE)
    test_ld = DataLoader(ds(data['Xte'], data['yte'], data['wf_test']), batch_size=cfg.BATCH_SIZE)

    set_seed(seed)
    model = AblationModel(cfg_use, N, F_in, cfg.GCN_HIDDEN, cfg.TCN_CHANNELS,
                          cfg.TCN_KERNEL_SIZE, cfg.TCN_DILATIONS, cfg.ATTENTION_HIDDEN,
                          hz, adj_t, out_dim, data['geo_dim_actual'],
                          vert_adj=vert_adj_t).to(cfg.DEVICE)
    n_params = count_params(model)
    sv = OUT / tag
    sv.mkdir(parents=True, exist_ok=True)
    save_path = sv / f"seed{seed}_H{hz}.pth"
    model, _, _, _ = train_model(cfg_use, model, train_ld, val_ld, save_path, node_mask=valid_mask_t)
    pred_cum, true_cum = evaluate(model, test_ld, data['init_test'], data['scY'],
                                  data['pca_y'], hz, cfg.DEVICE, node_mask=valid_mask_t)
    gm = global_metrics(true_cum, pred_cum)
    gm['seed'] = seed; gm['Hz'] = hz; gm['Params'] = n_params
    gm['tag'] = tag; gm['F_in'] = F_in
    with open(sv / f"seed{seed}_H{hz}_metrics.json", 'w') as f:
        json.dump(gm, f, indent=2)
    del model
    return gm


if __name__ == '__main__':
    base = Config()
    # 环境变量控制：NO_WATER_DEV=cpu|auto|gpu；NO_WATER_HZS="2,6,10"；NO_WATER_SEEDS="42,15"
    import os
    dev = os.environ.get('NO_WATER_DEV', 'auto')
    if dev == 'cpu':
        base.DEVICE = torch.device('cpu')
    else:
        base.DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    HZS = [int(x) for x in os.environ.get('NO_WATER_HZS', '2,6,10').split(',')]
    SEEDS = [int(x) for x in os.environ.get('NO_WATER_SEEDS', '42,15').split(',')]
    TAGS = [
        ("full",        1, True),    # V5 完整：TCN含水 + 投影分支
        ("v4_nowater",  0, True),    # V4 现况：TCN含水，无投影
        ("true_nowater",0, False),   # 真正全关：TCN不含水 + 无投影
    ]
    print(f"设备: {base.DEVICE}")
    results = []
    for tag, uw, wf in TAGS:
        for hz in HZS:
            for seed in SEEDS:
                gm = run_one(base, seed, hz, use_water_gcn=uw, water_in_feat=wf, tag=tag)
                results.append(gm)
                print(f"  {tag:<14} H={hz:<3} seed={seed:<4} "
                      f"RMSE={gm['RMSE']:.4f}  R2={gm['R2']:.4f}  Params={gm['Params']}  F_in={gm['F_in']}")
                sys.stdout.flush()

    print("\n" + "=" * 78)
    print("汇总（每组 mean over 2 seeds）")
    print("=" * 78)
    print(f"  {'策略':<14}{'H':<5}{'RMSE mean':<14}{'RMSE sd':<12}{'F_in':<6}{'Params':<10}")
    for tag, uw, wf in TAGS:
        for hz in HZS:
            g = [r for r in results if r['tag'] == tag and r['Hz'] == hz]
            rm = np.mean([r['RMSE'] for r in g]); sd = np.std([r['RMSE'] for r in g])
            print(f"  {tag:<14}{hz:<5}{rm:<14.4f}{sd:<12.4f}{g[0]['F_in']:<6}{g[0]['Params']:<10}")
    # 排序 hint
    print("\n按 H=10 每组 RMSE 排序：")
    for hz in HZS:
        row = [(t[0], np.mean([r['RMSE'] for r in results if r['tag'] == t[0] and r['Hz'] == hz])) for t in TAGS]
        row.sort(key=lambda x: x[1])
        print(f"  H={hz:<3} " + "  <  ".join(f"{t}={v:.4f}" for t, v in row))
    print(f"\n输出: {OUT}")