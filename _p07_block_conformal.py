# -*- coding: utf-8 -*-
r"""_p07_block_conformal.py — 块 conformal 校准 + 分视界/分阶段条件覆盖 + 覆盖率 CI。

零训练：不训练任何模型。模型仅用于「验证集前向推理」取校准残差（best.pth 前向、
不反向传播、不更新权重）；测试集预测直接读既有 predictions.npz。

背景（诚实声明）：
  §4.4.3 现有区间数字（0.522 / 0.359 / 0.941 / 3.976 等）来自 `_uq_verify.py`
  → `_uq_out/ensemble_uq_summary.json`，其中：
    · 变体为 **V5_Full_Model**（并非论文 Ours=V7_wo_VerticalGCN）；
    · "半集 split-conformal" 在 **测试集内部按窗口奇偶切分**（even/odd）标定与评估，
      校准集与评估集同源 ⇒ 覆盖率是**构造性**的，不构成有效校准证据。
  本脚本改用**验证集残差**（与测试集时间互斥）标定，给出块 conformal / 分视界 /
  分施工阶段条件覆盖 / block bootstrap CI，并与旧口径并表对照。

用法（PowerShell）：
  $env:P07_HZS="2,6,8,10,12"; $env:P07_SEEDS="30"; python _p07_block_conformal.py
"""
from _env import EXAMPLE_MODE, WORK_ROOT

import os, sys, json, time
from pathlib import Path
from collections import OrderedDict

import numpy as np
import pandas as pd
import torch

ROOT = WORK_ROOT
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
import Ablation_Experiment as AB  # noqa: E402

VARIANT = 'V7_wo_VerticalGCN'          # Ours（论文最终模型）
OLD_VARIANT = 'V5_Full_Model'          # 旧区间数字所用变体
SWITCH = dict(AB.ABLATION_VARIANTS[VARIANT])

ABL_ROOT = AB.Config().OUT_ROOT
OUT_DIR = ROOT / '_paper_tables'
OUT_DIR.mkdir(parents=True, exist_ok=True)

H_LIST = [int(x) for x in os.environ.get('P07_HZS', '2,6,8,10,12').split(',')]
SEED_LIMIT = int(os.environ.get('P07_SEEDS', '30'))
B_BOOT = 2000
BOOT_SEED = 42
L_STAR = 3
ALPHAS = [0.10, 0.05]
GAMMA_ACI = 0.01
EPS = 1e-8

PHASE_NAMES = {'Excavation': '开挖期', 'Foundation': '底板期',
               'Structure': '结构期', 'Other': '开挖后(其他)'}


# =====================================================================
# 工具
# =====================================================================
def conf_quantile(vals, alpha):
    """保序 conformal 分位：level = ceil((n+1)(1-alpha))/n，取上侧分位（保守）。"""
    vals = np.asarray(vals).ravel()
    n = len(vals)
    if n == 0:
        return float('nan')
    lev = min(1.0, np.ceil((n + 1) * (1.0 - alpha)) / n)
    return float(np.quantile(vals, lev, method='higher'))


def block_aggregate(score_w, L, mode='maxw'):
    """对校准残差按块聚合：滑窗块长 L（窗级），块内只在**时间维**聚合。

    mode:
      'maxw'  — 逐节点取块内 L 个窗的最大（主口径；节点仍为同期重复，仅窗间相关
                被块内取最大吸收，避免跨 155 节点取 max 导致块分位退化为全局极值）
      'rms'   — 逐节点取块内 L 个窗的 RMS（块尺度替代口径）
      'maxall'— 整块取全 max（最保守上界，仅供参考）
    """
    nw = score_w.shape[0]
    out = []
    for b in range(nw - L + 1):
        blk = score_w[b:b + L]                       # (L, N)
        if mode == 'maxw':
            out.append(blk.max(0))
        elif mode == 'rms':
            out.append(np.sqrt((blk ** 2).mean(0)))
        else:
            out.append(np.asarray([blk.max()]))
    return np.concatenate(out)


def circular_block_idx(n, L, rng):
    """circular block bootstrap 重采样窗索引。"""
    out = []
    nblk = int(np.ceil(n / L))
    while len(out) < n:
        s = int(rng.integers(0, n))
        out.extend([(s + k) % n for k in range(L)])
    return np.asarray(out[:n], dtype=int)


def boot_ci(ok_w, L=L_STAR, B=B_BOOT, seed=BOOT_SEED):
    """对窗级布尔覆盖矩阵 ok_w (nw, N) 做 circular block bootstrap 的 95% CI。"""
    nw = ok_w.shape[0]
    if nw < L + 1:
        return float('nan'), float('nan')
    rng = np.random.default_rng(seed)
    covs = np.empty(B)
    for i in range(B):
        idx = circular_block_idx(nw, L, rng)
        covs[i] = ok_w[idx].mean()
    lo, hi = np.percentile(covs, [2.5, 97.5])
    return float(lo), float(hi)


# =====================================================================
# 施工阶段标签（复用 _phase_eval_probe 的分类逻辑）
# =====================================================================
def _is_excavation(s):
    return ('开挖' in s) or ('至' in s and '米' in s)


def _is_foundation(s):
    return '底板' in s


def _is_structure(s):
    return '结构' in s


def phase_of(s):
    if _is_excavation(s):
        return '开挖期'
    if _is_foundation(s):
        return '底板期'
    if _is_structure(s):
        return '结构期'
    return '开挖后(其他)'


def load_stage_labels(cfg, ts):
    raw0 = pd.read_excel(cfg.BASE_DATA_DIR / 'CX2.xlsx')
    tcol = [c for c in raw0.columns if '测量时间' in str(c)][0]
    scol = [c for c in raw0.columns if '施工进度' in str(c)][0]
    tbl = (raw0[[tcol, scol]].drop_duplicates(subset=[tcol])
           .sort_values(tcol).reset_index(drop=True))
    tbl[tcol] = pd.to_datetime(tbl[tcol])
    stage = tbl.set_index(tcol)[scol].astype(str)
    pres = np.array([stage.get(pd.Timestamp(x), '未知') for x in ts])
    return pres


# =====================================================================
# 主流程
# =====================================================================
def main():
    t0 = time.time()
    cfg = AB.Config()
    device = cfg.DEVICE
    print(f"设备: {device} | Ours={VARIANT} | H={H_LIST} | seeds<={SEED_LIMIT}")

    # ---------------- 数据（只加载一次） ----------------
    AB.set_seed(42)
    full_df = AB.load_raw_data_with_meta(cfg)
    ts = np.sort(full_df['测量时间'].unique())
    wd, wa = AB.load_water_data(ts, cfg)
    water_feat = AB.extract_water_features(wd, wa)
    disp_raw, _, meta = AB.build_node_features(full_df, cfg)
    geo_records = AB.load_geology_params(cfg)
    geo_feat_raw = AB.assign_geo_to_nodes(geo_records, meta['depths'], cfg) \
        if geo_records else None
    n = len(ts)
    tr_end = int(n * cfg.TRAIN_RATIO)
    va_end = int(n * (cfg.TRAIN_RATIO + cfg.VAL_RATIO))
    train_idx = np.arange(tr_end)
    val_idx = np.arange(tr_end, va_end)
    test_idx = np.arange(va_end, n)
    print(f"时间划分: train[0,{tr_end}) val[{tr_end},{va_end}) test[{va_end},{n})  共 {n} 步")

    pres = load_stage_labels(cfg, ts)
    print("施工阶段序列:", OrderedDict(
        (k, int((pres == k).sum())) for k in OrderedDict.fromkeys(pres)))
    print("测试段阶段分布:", OrderedDict(
        (k, int((pres[test_idx] == k).sum())) for k in OrderedDict.fromkeys(pres[test_idx])))

    geo_fb = geo_feat_raw if geo_feat_raw is not None else np.zeros(
        (len(meta['depths']), cfg.GEO_FEAT_DIM), dtype=np.float32)
    geo_fb_use = geo_feat_raw if geo_feat_raw is not None and np.any(geo_feat_raw) else None

    # 去噪（与 prepare_data 内部完全一致，用于取验证/测试绝对位移起点）
    train_den, val_den, test_den = AB.denoise_split(
        disp_raw[train_idx], disp_raw[val_idx], disp_raw[test_idx], cfg)
    all_den = np.vstack([train_den, val_den, test_den])
    diff1_den = np.zeros_like(all_den)
    diff1_den[1:] = np.diff(all_den, axis=0)

    rows = []          # 落表（覆盖面）
    ens_rows = []      # 落表（集成口径：可达误差下降 / 相对零变化落后）
    anchor_ok = True
    summary_lines = []

    for hz in H_LIST:
        print(f"\n{'='*70}\nH={hz}\n{'='*70}")
        data = AB.prepare_data(cfg, full_df, wd, wa, meta, train_idx, val_idx, test_idx,
                               water_feat, geo_fb, hz, disp_raw=disp_raw)
        geo_static_t = torch.FloatTensor(data['geo_static']).to(device)
        adj_t = torch.FloatTensor(AB.build_adjacency(meta, geo_fb_use, cfg)).to(device)
        vert_adj = AB.build_vertical_adjacency(meta, cfg)
        vert_adj_t = torch.FloatTensor(vert_adj).to(device) if vert_adj is not None else None
        valid_mask_t = torch.FloatTensor(data['valid_mask']).to(device)
        N, F_in, out_dim = data['N'], data['F_in'], data['out_dim']

        # 验证集真值窗（与 evaluate 口径一致）
        y_val = diff1_den[val_idx]
        Xva = data['Xva']
        yva_raw = np.stack([y_val[i + cfg.LOOKBACK:i + cfg.LOOKBACK + hz]
                            for i in range(Xva.shape[0])]) if Xva.shape[0] else \
            np.empty((0, hz, y_val.shape[1]), dtype=np.float32)
        init_val = val_den[cfg.LOOKBACK - 1:-hz][:Xva.shape[0]]

        def make_ds(X, y, wf, nwin):
            return torch.utils.data.TensorDataset(
                torch.FloatTensor(X), torch.FloatTensor(y), torch.FloatTensor(wf),
                geo_static_t.unsqueeze(0).expand(nwin, N, -1))

        val_ld = torch.utils.data.DataLoader(
            make_ds(Xva, data['yva'], data['wf_val'], Xva.shape[0]),
            batch_size=cfg.BATCH_SIZE)

        # ------- 30 种子：验证集前向（零训练） -------
        vcfg = AB.Config()
        for k, v in SWITCH.items():
            setattr(vcfg, k, v)
        vcfg.DEVICE = device

        # 测试集预测直接读既有产物
        run_dir = ABL_ROOT / f"H{hz}" / VARIANT
        runs = sorted(run_dir.glob("run_*/predictions.npz"))
        test_preds, test_true, seed_names = [], None, []
        for r in runs:
            d = np.load(r)
            test_preds.append(d["pred_cum"])
            test_true = d["true_cum"]
            seed_names.append(int(r.parent.name.split('_')[1]))
        test_preds = np.stack(test_preds, 0)          # (S, nw_t, hz, N)
        if SEED_LIMIT < test_preds.shape[0]:
            test_preds = test_preds[:SEED_LIMIT]
        m_t = test_preds.mean(0)
        s_t = test_preds.std(0)
        nw_t = m_t.shape[0]

        # 验证集：逐种子前向
        val_preds = []
        for i, seed in enumerate(seed_names[:test_preds.shape[0]]):
            pth = run_dir / f"run_{seed}" / "best.pth"
            model = AB.AblationModel(
                vcfg, N=N, in_dim=F_in, gcn_hid=vcfg.GCN_HIDDEN,
                tcn_ch=vcfg.TCN_CHANNELS, k=vcfg.TCN_KERNEL_SIZE,
                dil=vcfg.TCN_DILATIONS, attn_hid=vcfg.ATTENTION_HIDDEN,
                hor=hz, adj=adj_t, out_dim=out_dim,
                geo_feat_dim=data['geo_dim_actual'], vert_adj=vert_adj_t).to(device)
            state = torch.load(pth, map_location=device)
            # best.pth 由 thop profile 写入 total_ops/total_params 缓冲键，需放宽匹配
            model.load_state_dict(state, strict=False)
            pc, tc = AB.evaluate(model, val_ld, init_val, data['scY'], data['pca_y'],
                                 hz, device, node_mask=valid_mask_t,
                                 true_diff_win=yva_raw)
            val_preds.append(pc)
            del model
        val_preds = np.stack(val_preds, 0)            # (S, nw_v, hz, N)
        m_v = val_preds.mean(0)
        s_v = val_preds.std(0)
        # 验证集真值（evaluate 逆变换累加，与 m_v 同口径）
        true_v = tc
        nw_v = m_v.shape[0]

        # ---------------- 锚点自检 ----------------
        rmse_seed_last = np.array([
            np.sqrt(np.mean((test_preds[i, :, -1, :] - test_true[:, -1, :]) ** 2))
            for i in range(test_preds.shape[0])])
        ours_rmse = float(rmse_seed_last.mean())
        ours_std = float(rmse_seed_last.std(ddof=1))
        # 零变化基线（末步口径）
        nw_p = test_den.shape[0] - cfg.LOOKBACK - hz + 1
        init_p = test_den[np.arange(cfg.LOOKBACK - 1, cfg.LOOKBACK - 1 + nw_p), :]
        true_last_p = test_den[np.arange(cfg.LOOKBACK - 1, cfg.LOOKBACK - 1 + nw_p) + hz, :]
        persist = float(np.sqrt(np.mean((true_last_p - init_p) ** 2)))
        line = (f"[锚点] H={hz} Ours末步RMSE={ours_rmse:.4f}±{ours_std:.4f}"
                f" (期望 1.5687±0.0940) | 零变化末步RMSE={persist:.6f}"
                f" (期望 1.372010) | nw_t={nw_t} nw_v={nw_v}")
        print(line); summary_lines.append(line)
        if hz == 10:
            if abs(ours_rmse - 1.5687) > 5e-3 or abs(ours_std - 0.0940) > 5e-3:
                anchor_ok = False
            if abs(persist - 1.372010) > 5e-4:
                anchor_ok = False

        # ---------------- 校准分数（验证集，末步口径） ----------------
        e_v = np.abs(true_v[:, -1, :] - m_v[:, -1, :])          # (nw_v, N)
        sc_v = e_v / np.maximum(s_v[:, -1, :], EPS)             # 标定分数
        e_t = np.abs(test_true[:, -1, :] - m_t[:, -1, :])       # (nw_t, N)
        s_tl = np.maximum(s_t[:, -1, :], EPS)

        # 地板（数据侧已知量，变体无关；读同目录 metrics.json 已回填的 RMSE_floor_last）
        floor_vals = []
        for r in runs:
            mf = r.parent / 'metrics.json'
            if mf.exists():
                try:
                    gm_ = json.load(open(mf, encoding='utf-8'))
                    if 'RMSE_floor_last' in gm_:
                        floor_vals.append(gm_['RMSE_floor_last'])
                except Exception:
                    pass
        floor_last = float(np.mean(floor_vals)) if floor_vals else None

        # ---------------- 集成口径复算（_uq_verify.py 同式） ----------------
        # "集成" = 30 种子逐元素（窗×步×节点）预测的算术平均 m_t；
        # "可达误差" = sqrt(RMSE_last^2 - floor_last^2)（末步、逐视界）；
        # "降低可达误差%" = (1 - reach_ensemble / reach_single_mean)*100，
        # 其中 reach_single_mean 由"单种子末步 RMSE 的均值"再减地板得到。
        rmse_ens_last = float(np.sqrt(np.mean(
            (m_t[:, -1, :] - test_true[:, -1, :]) ** 2)))
        single_last = np.array([
            np.sqrt(np.mean((test_preds[i, :, -1, :] - test_true[:, -1, :]) ** 2))
            for i in range(test_preds.shape[0])])
        rmse_single_mean_last = float(single_last.mean())
        if floor_last is not None:
            reach_ens = float(np.sqrt(max(rmse_ens_last ** 2 - floor_last ** 2, 0.0)))
            reach_mean = float(np.sqrt(max(rmse_single_mean_last ** 2 - floor_last ** 2, 0.0)))
            drop_pct = float((1.0 - reach_ens / reach_mean) * 100.0) if reach_mean > 0 else float('nan')
        else:
            reach_ens = reach_mean = drop_pct = float('nan')
        lag_pct = float((rmse_ens_last / persist - 1.0) * 100.0)
        ens_rows.append(dict(
            H=hz, n_seeds=int(test_preds.shape[0]),
            ensemble_def='30种子逐元素(窗×步×节点)算术平均',
            rmse_ensemble_last=rmse_ens_last,
            rmse_single_mean_last=rmse_single_mean_last,
            rmse_single_best_last=float(single_last.min()),
            rmse_single_worst_last=float(single_last.max()),
            floor_last=floor_last,
            reach_ensemble_last=reach_ens,
            reach_single_mean_last=reach_mean,
            reach_drop_pct=drop_pct,
            persist_last=persist,
            lag_vs_persist_pct=lag_pct))
        print(f"  [集成] 末步RMSE: 集成={rmse_ens_last:.4f} vs 单种子均值={rmse_single_mean_last:.4f}"
              f" | 可达降幅={drop_pct:.2f}% | vs零变化={lag_pct:+.2f}%")

        # 测试窗 → 阶段（按末步目标时刻）
        last_ts_idx = np.array([test_idx[0] + cfg.LOOKBACK + w + hz - 1 for w in range(nw_t)])
        win_phase = np.array([phase_of(pres[j]) for j in last_ts_idx])
        print("测试窗末步阶段分布:", OrderedDict(
            (k, int((win_phase == k).sum())) for k in OrderedDict.fromkeys(win_phase)))

        for alpha in ALPHAS:
            tag_a = f"{1-alpha:.0%}"

            # ---- (1) split-conformal（验证集标定，correct proper） ----
            q_split = conf_quantile(sc_v, alpha)
            ok_split = (e_t <= q_split * s_tl)
            lo, hi = boot_ci(ok_split)
            w_split = float(2 * q_split * s_tl.mean())
            last_split_cov = float(ok_split.mean())
            rows.append(dict(H=hz, 方法='split', 覆盖对象='无条件', 名义水平=tag_a,
                             实测覆盖=last_split_cov, ci_lo=lo, ci_hi=hi,
                             区间宽度=w_split,
                             calibration_set='验证集残差(前向,与测试互斥)',
                             note=f'标定尺度q={q_split:.3f}(L=–)'))

            # ---- (1a) 未重校种子区间（把种子间 std 当不确定度） ----
            z = 1.6449 if alpha == 0.10 else 1.9600
            ok_raw = (e_t <= z * s_tl)
            lo, hi = boot_ci(ok_raw)
            rows.append(dict(H=hz, 方法='raw(未重校)', 覆盖对象='无条件', 名义水平=tag_a,
                             实测覆盖=float(ok_raw.mean()), ci_lo=lo, ci_hi=hi,
                             区间宽度=float(2 * z * s_tl.mean()),
                             calibration_set='测试集种子间std(不校准)',
                             note=f'未重校,宽度=z·s,z={z:.4f}'))

            # ---- (1b) 并入地板方差重校 ----
            if floor_last:
                sf = np.sqrt(s_tl ** 2 + floor_last ** 2)
                ok_floor = (e_t <= z * sf)
                lo, hi = boot_ci(ok_floor)
                rows.append(dict(H=hz, 方法='floor(+地板方差)', 覆盖对象='无条件',
                                 名义水平=tag_a, 实测覆盖=float(ok_floor.mean()),
                                 ci_lo=lo, ci_hi=hi,
                                 区间宽度=float(2 * z * sf.mean()),
                                 calibration_set='测试集种子间std+数据侧地板(无标签泄漏)',
                                 note=f'+floor_last={floor_last:.4f},z={z:.4f}'))

            # ---- (2) block conformal（EnbPI 式，L=3 主口径） ----
            for L in (2, 3, 4):
                q_blk = conf_quantile(block_aggregate(sc_v, L, 'maxw'), alpha)
                ok_blk = (e_t <= q_blk * s_tl)
                lo, hi = boot_ci(ok_blk)
                w_blk = float(2 * q_blk * s_tl.mean())
                rows.append(dict(H=hz, 方法=('block' if L == L_STAR else f'block(L={L})'),
                                 覆盖对象='无条件', 名义水平=tag_a,
                                 实测覆盖=float(ok_blk.mean()), ci_lo=lo, ci_hi=hi,
                                 区间宽度=w_blk,
                                 calibration_set='验证集残差(前向,与测试互斥)',
                                 note=f'块长L={L},逐节点块内窗最大后取分位,q={q_blk:.3f}'))
                if L == L_STAR:
                    q_blk_main, ok_blk_main = q_blk, ok_blk

            # ---- (2a') 最保守上界：整块全 max（仅供参考） ----
            q_bmax = conf_quantile(block_aggregate(sc_v, L_STAR, 'maxall'), alpha)
            ok_bmax = (e_t <= q_bmax * s_tl)
            rows.append(dict(H=hz, 方法='block(max-all,上界)', 覆盖对象='无条件',
                             名义水平=tag_a, 实测覆盖=float(ok_bmax.mean()),
                             ci_lo=float('nan'), ci_hi=float('nan'),
                             区间宽度=float(2 * q_bmax * s_tl.mean()),
                             calibration_set='验证集残差(前向,与测试互斥)',
                             note=f'整块(窗×节点)全max,q={q_bmax:.3f}(过保守,仅供上界)'))

            # ---- (2b) block-RMS 变体（块内 RMS 聚合） ----
            q_brms = conf_quantile(block_aggregate(sc_v, L_STAR, 'rms'), alpha)
            ok_brms = (e_t <= q_brms * s_tl)
            lo, hi = boot_ci(ok_brms)
            rows.append(dict(H=hz, 方法='block(rms)', 覆盖对象='无条件', 名义水平=tag_a,
                             实测覆盖=float(ok_brms.mean()), ci_lo=lo, ci_hi=hi,
                             区间宽度=float(2 * q_brms * s_tl.mean()),
                             calibration_set='验证集残差(前向,与测试互斥)',
                             note=f'块内RMS聚合,L={L_STAR},q={q_brms:.3f}'))

            # ---- (3) ACI（自适应在线覆盖，探索性） ----
            pool = sc_v.ravel().copy()
            at = alpha
            q_aci = conf_quantile(pool, at)
            oks = []
            for w in range(nw_t):
                ok = (e_t[w] <= q_aci * s_tl[w])
                oks.append(ok)
                err = 1.0 - float(ok.mean())
                at = float(np.clip(at + GAMMA_ACI * (alpha - err), 1e-3, 0.999))
                pool = np.concatenate([pool, (e_t[w] / s_tl[w])])
                q_aci = conf_quantile(pool, at)
            ok_aci = np.stack(oks, 0)
            lo, hi = boot_ci(ok_aci)
            rows.append(dict(H=hz, 方法='ACI', 覆盖对象='无条件', 名义水平=tag_a,
                             实测覆盖=float(ok_aci.mean()), ci_lo=lo, ci_hi=hi,
                             区间宽度=float(2 * q_aci * s_tl.mean()),
                             calibration_set='验证集残差+在线更新(γ=%.2f)' % GAMMA_ACI,
                             note='窗口级自适应α更新,序列短→探索性'))

            # ---- (4) 旧口径对照：测试集半集切分（构造性） ----
            idx = np.arange(nw_t)
            even, odd = idx[idx % 2 == 0], idx[idx % 2 == 1]
            covs, scales = [], []
            for cal_i, ev_i in ((even, odd), (odd, even)):
                a = conf_quantile(e_t[cal_i] / s_tl[cal_i], alpha)
                scales.append(a)
                covs.append(float((e_t[ev_i] <= a * s_tl[ev_i]).mean()))
            q_old = float(np.mean(scales))
            rows.append(dict(H=hz, 方法='split(test-half,旧口径)', 覆盖对象='无条件',
                             名义水平=tag_a, 实测覆盖=float(np.mean(covs)),
                             ci_lo=float('nan'), ci_hi=float('nan'),
                             区间宽度=float(2 * q_old * s_tl.mean()),
                             calibration_set='测试集内even/odd半集切分(构造性,信息泄漏)',
                             note='与_uq_verify.py一致；覆盖率非有效证据'))

            # ---- (5) 分阶段条件覆盖（split 与 block 主口径） ----
            for ph in PHASE_NAMES.values():
                sel = (win_phase == ph)
                if sel.sum() == 0:
                    # 该施工阶段在验证/测试窗内无样本（主划分下 test∩{开挖/底板}=∅）
                    for tag in ('split', 'block'):
                        rows.append(dict(H=hz, 方法=tag, 覆盖对象=f'阶段:{ph}',
                                         名义水平=tag_a, 实测覆盖=float('nan'),
                                         ci_lo=float('nan'), ci_hi=float('nan'),
                                         区间宽度=float('nan'),
                                         calibration_set='验证集残差(前向,与测试互斥)',
                                         note='该阶段在主划分 val/test 内无窗口→不可得'))
                    continue
                for tag, ok_m, qv in (('split', ok_split[sel], q_split),
                                      ('block', ok_blk_main[sel], q_blk_main)):
                    lo, hi = boot_ci(ok_m)
                    rows.append(dict(H=hz, 方法=tag, 覆盖对象=f'阶段:{ph}',
                                     名义水平=tag_a, 实测覆盖=float(ok_m.mean()),
                                     ci_lo=lo, ci_hi=hi,
                                     区间宽度=float(2 * qv * np.maximum(s_t[sel, -1, :], EPS).mean()),
                                     calibration_set='验证集残差(前向,与测试互斥)',
                                     note=f'阶段窗数={int(sel.sum())},L={L_STAR if tag=="block" else "–"}'))

        # 打印 H 摘要
        print(f"  split 90% 无条件覆盖={last_split_cov:.3f}")
        summary_lines.append(f"H={hz}: nw_v={nw_v}(val窗) nw_t={nw_t}(test窗) "
                             f"split90={last_split_cov:.3f}")

    # ---------------- 落表 ----------------
    df = pd.DataFrame(rows)
    csv_path = OUT_DIR / 'T_p07_conformal.csv'
    df.to_csv(csv_path, index=False, float_format='%.4f')
    print(f"\n已落盘: {csv_path}  ({len(df)} 行)")
    df_ens = pd.DataFrame(ens_rows)
    ens_path = OUT_DIR / 'T_p07_ensemble.csv'
    df_ens.to_csv(ens_path, index=False, float_format='%.6f')
    print(f"已落盘: {ens_path}  ({len(df_ens)} 行)")
    print("\n".join(summary_lines))
    print(f"锚点自检 {'通过' if anchor_ok else '失败'}")
    print(f"总耗时 {time.time()-t0:.0f}s")
    if not anchor_ok:
        if EXAMPLE_MODE:
            print("锚点自检未通过（EXAMPLE_MODE=1：合成示例数据，跳过终止）")
        else:
            raise SystemExit("锚点自检失败——停止，请核对数据/变体。")


if __name__ == '__main__':
    main()