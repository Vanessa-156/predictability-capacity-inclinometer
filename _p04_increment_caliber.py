# -*- coding: utf-8 -*-
"""_p04_increment_caliber.py — P04：增量口径 vs 累计口径的零训练重算。

背景：论文主表只报「累计位移」口径（预测 H 步后的累计位移与真值比），从未报模型
原生「增量」口径（逐帧一阶差分）。本脚本只读既有 `predictions.npz` / `metrics.json`
（30 种子），零训练，不覆盖任何既有产物，产出：

  1. 累计口径：末步 RMSE_last / 全步 RMSE_all（逐种子 mean±std）
  2. 增量口径：末步 / 全步 RMSE（逐种子 mean±std）
  3. 两种口径的 skill（论文约定 skill = 1 − (RMSE_model/RMSE_zero)²，种子均值口径）
     与次列的 MSE 聚合口径 skill_mse
  4. 增量 Pearson 相关 corr(pred_inc, true_inc)（逐种子 pooled，mean±std）
  5. 确定性基线（无种子）：零增量 / 末增量(匀速) / AR(1)逐节点 / PCA(0.90) oracle 投影
  6. 诊断：增量 ACF(lag1–5)、Ljung-Box（训练增量，逐节点）、Lo-MacKinlay VR(q)、
     ADF（测试累计序列，逐节点）
  7. 循环块 bootstrap(B=2000,L=3,seed=42) 的 95% CI（H=10 主，另给全部 H）

硬性锚点（任一不符即停，不调整数据）：
  (a) 零增量累计 RMSE_last 五个 H 与 T_persist_baseline.csv 完全一致(<1e-4)
  (b) Ours(=V7) H=10 种子均值 cum_last_rmse = 1.5687 ± 0.0940
  (c) 代表性 run：max|true_cum_npz − (init_test + cumsum(true_inc))| < 1e-6

用法（只读已有结果）：
  python _p04_increment_caliber.py
"""
from _env import AUX_ROOT, EXAMPLE_MODE, WORK_ROOT

import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings('ignore')

PROJ = WORK_ROOT
if str(PROJ) not in sys.path:
    sys.path.insert(0, str(PROJ))

from Ablation_Experiment import (          # noqa: E402
    Config, set_seed, load_raw_data_with_meta, load_water_data,
    extract_water_features, build_node_features, load_geology_params,
    assign_geo_to_nodes, denoise_split, prepare_data,
)

ROOT = AUX_ROOT
ABL_ROOT = ROOT / "Ablation_New_5d"
CMP_ROOT = ROOT / "Comparison_Results_w64"
OUT = PROJ / "_paper_tables"
OUT.mkdir(parents=True, exist_ok=True)

HZS = [2, 6, 8, 10, 12]
LB = 12                                   # LOOKBACK，与管线一致
N_BOOT = 2000
BOOT_L = 3
BOOT_SEED = 42
ALPHAS = [1e-4, 1e-3, 1e-2, 1e-1, 1, 10, 100, 1e3, 1e4, 1e5, 1e6, 1e7, 1e8, 1e9]

VARIANTS = ['V7_wo_VerticalGCN', 'V0_Pure_TCN', 'V5_Full_Model', 'V1_wo_MeanPool',
            'V2_wo_GCN', 'V3_wo_PhysicsGate', 'V6_wo_CrossNodeAttn', 'V8_wo_VerticalAll']
EXTERNALS = ['BiLSTM-Attn', 'GraphWaveNet', 'LSTM', 'STGCN', 'Transformer',
             'V-Attention', 'V-MLP']
OURS_KEY = 'V7_wo_VerticalGCN'            # 论文的 "Ours"

# 锚点参考值
PERSIST_LAST = {2: 0.526216, 6: 0.989323, 8: 1.192202, 10: 1.372010, 12: 1.524272}
OURS_H10_MEAN, OURS_H10_STD = 1.5687, 0.0940
FLOOR_LAST_H10 = 1.277868                 # _make_paper_tables.py FLOORS[10][1]

# 确定性基线的输出键
BASE_ZERO, BASE_LAST, BASE_AR1, BASE_ORACLE, BASE_RIDGE = (
    'ZERO_INC', 'LAST_INC', 'AR1_PER_NODE', 'ORACLE_PCA090', 'RIDGE_SAME_INPUT')


# =============================================================================
# 工具
# =============================================================================
def rmse(a, b):
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    return float(np.sqrt(np.mean((a - b) ** 2)))


def pearson(a, b):
    a = np.asarray(a, dtype=np.float64).ravel()
    b = np.asarray(b, dtype=np.float64).ravel()
    if a.std() < 1e-12 or b.std() < 1e-12:
        return float('nan')
    return float(np.corrcoef(a, b)[0, 1])


def agg(vals):
    a = np.asarray(vals, dtype=np.float64)
    if len(a) == 0:
        return float('nan'), float('nan')
    return float(a.mean()), (float(a.std(ddof=1)) if len(a) > 1 else 0.0)


def skill(rmse_val, zero_val):
    return 1.0 - (float(rmse_val) / float(zero_val)) ** 2


# =============================================================================
# 数据管线（复用 Ablation_Experiment，与 _persist_baseline / _predictability_probe 同源）
# =============================================================================
def load_pipeline():
    set_seed(42)
    cfg = Config()
    cfg.LOOKBACK = LB
    cfg.HORIZONS = HZS

    full_df = load_raw_data_with_meta(cfg)
    ts = np.sort(full_df['测量时间'].unique())
    wd, wa = load_water_data(ts, cfg)
    water_feat = extract_water_features(wd, wa)
    disp_raw, _, meta = build_node_features(full_df, cfg)
    recs = load_geology_params(cfg)
    geo_raw = assign_geo_to_nodes(recs, meta['depths'], cfg) if recs else None
    geo_fb = (geo_raw if (geo_raw is not None and np.any(geo_raw))
              else np.zeros((len(meta['depths']), cfg.GEO_FEAT_DIM), dtype=np.float32))

    n = len(ts)
    tr = np.arange(0, int(n * cfg.TRAIN_RATIO))
    va = np.arange(int(n * cfg.TRAIN_RATIO), int(n * (cfg.TRAIN_RATIO + cfg.VAL_RATIO)))
    te = np.arange(int(n * (cfg.TRAIN_RATIO + cfg.VAL_RATIO)), n)

    tr_den, va_den, te_den = denoise_split(disp_raw[tr], disp_raw[va], disp_raw[te], cfg)
    all_den = np.vstack([tr_den, va_den, te_den])
    diff1_den = np.zeros_like(all_den)
    diff1_den[1:] = np.diff(all_den, axis=0)
    y_train = diff1_den[tr]                       # 训练增量（含置 0 首行，与管线一致）
    pca_y = PCA(0.90).fit(y_train)                # 与 prepare_data 完全同参

    return dict(cfg=cfg, full_df=full_df, ts=ts, wd=wd, wa=wa, water_feat=water_feat,
                disp_raw=disp_raw, meta=meta, geo_fb=geo_fb, tr=tr, va=va, te=te,
                tr_den=tr_den, va_den=va_den, te_den=te_den, all_den=all_den,
                diff1_den=diff1_den, y_train=y_train, pca_y=pca_y)


# =============================================================================
# 窗口构造
# =============================================================================
def build_windows(te_den, H):
    """返回 (n_win, init_test, v_last, true_inc, true_cum)。

    init_test = 窗口起点累计（= 最后观测帧），v_last = 最后观测增量，
    true_inc[:,t] = te_den[init+t+1] − te_den[init+t]，true_cum 为累加还原。
    """
    T = te_den.shape[0]
    n_win = T - LB - H + 1
    init_idx = np.arange(LB - 1, LB - 1 + n_win)
    init_test = te_den[init_idx]                                  # (nw,N)
    v_last = te_den[init_idx] - te_den[init_idx - 1]              # (nw,N)
    true_inc = np.stack([te_den[init_idx + t + 1] - te_den[init_idx + t]
                         for t in range(H)], axis=1)              # (nw,H,N)
    true_cum = init_test[:, None, :] + np.cumsum(true_inc, axis=1)
    return n_win, init_test, v_last, true_inc, true_cum


def zero_rmse(init_test, true_inc, true_cum):
    H = true_inc.shape[1]
    cum_all_pred = np.broadcast_to(init_test[:, None, :], true_cum.shape)
    return dict(
        cum_last=rmse(init_test, true_cum[:, -1, :]),
        cum_all=rmse(cum_all_pred, true_cum),
        inc_last=float(np.sqrt(np.mean(true_inc[:, -1, :] ** 2))),
        inc_all=float(np.sqrt(np.mean(true_inc ** 2))),
    )


def metrics_from_cum(pred_cum, init_test, true_inc, true_cum):
    """由 pred_cum 推出两种口径的全部逐种子指标。"""
    pred_cum = np.asarray(pred_cum, dtype=np.float64)
    pred_inc = np.empty_like(pred_cum)
    pred_inc[:, 0, :] = pred_cum[:, 0, :] - init_test
    pred_inc[:, 1:, :] = np.diff(pred_cum, axis=1)
    return dict(
        cum_last=rmse(pred_cum[:, -1, :], true_cum[:, -1, :]),
        cum_all=rmse(pred_cum, true_cum),
        inc_last=rmse(pred_inc[:, -1, :], true_inc[:, -1, :]),
        inc_all=rmse(pred_inc, true_inc),
        corr=pearson(pred_inc, true_inc),
    )


def per_window_se(pred_cum, init_test, true_inc, true_cum):
    """逐窗口平方误差（末步 / 全步，累计 / 增量口径），供 bootstrap 重采样用。"""
    pred_cum = np.asarray(pred_cum, dtype=np.float64)
    pred_inc = np.empty_like(pred_cum)
    pred_inc[:, 0, :] = pred_cum[:, 0, :] - init_test
    pred_inc[:, 1:, :] = np.diff(pred_cum, axis=1)
    with np.errstate(all='ignore'):
        return dict(
            cum_last=np.mean((pred_cum[:, -1, :] - true_cum[:, -1, :]) ** 2, axis=1),
            cum_all=np.mean((pred_cum - true_cum) ** 2, axis=(1, 2)),
            inc_last=np.mean((pred_inc[:, -1, :] - true_inc[:, -1, :]) ** 2, axis=1),
            inc_all=np.mean((pred_inc - true_inc) ** 2, axis=(1, 2)),
        )


def zero_per_window_se(init_test, true_inc, true_cum):
    with np.errstate(all='ignore'):
        return dict(
            cum_last=np.mean((init_test - true_cum[:, -1, :]) ** 2, axis=1),
            cum_all=np.mean((init_test[:, None, :] - true_cum) ** 2, axis=(1, 2)),
            inc_last=np.mean(true_inc[:, -1, :] ** 2, axis=1),
            inc_all=np.mean(true_inc ** 2, axis=(1, 2)),
        )


# =============================================================================
# 确定性基线
# =============================================================================
def pred_zero(init_test, true_cum):
    return np.broadcast_to(init_test[:, None, :], true_cum.shape).copy()


def pred_last_inc(init_test, v_last, H):
    steps = np.arange(1, H + 1, dtype=np.float64)[None, :, None]
    return init_test[:, None, :] + steps * v_last[:, None, :]


def pred_ar1(init_test, v_last, diff1_train, H):
    """逐节点 AR(1)（x_t = c + phi·x_{t-1}），只用训练增量拟合，从 v_last 递归 H 步。"""
    x = diff1_train[1:]                                # 去掉置 0 首行
    N = x.shape[1]
    phi = np.zeros(N)
    c = np.zeros(N)
    for j in range(N):
        s = x[:, j]
        x0, x1 = s[:-1], s[1:]
        A = np.column_stack([np.ones_like(x0), x0])
        coef, *_ = np.linalg.lstsq(A, x1, rcond=None)
        c[j], phi[j] = coef
    nw = v_last.shape[0]
    pred_inc = np.zeros((nw, H, N))
    cur = v_last.copy()
    for t in range(H):
        cur = c[None, :] + phi[None, :] * cur
        pred_inc[:, t, :] = cur
    return init_test[:, None, :] + np.cumsum(pred_inc, axis=1)


def pred_oracle(true_inc, pca_y):
    """oracle = 真值增量投影到 PCA(0.90) 训练子空间（任何子空间约束预测的最优界）。"""
    N = true_inc.shape[2]
    flat = true_inc.reshape(-1, N)
    rec = pca_y.inverse_transform(pca_y.transform(flat)).reshape(true_inc.shape)
    return rec


def pred_ridge(data, H):
    """同输入空间（模型 4 维输入展平）岭回归，alpha 于 val 选，链与 evaluate 一致。"""
    Xtr = data['Xtr'].reshape(data['Xtr'].shape[0], -1)
    Xva = data['Xva'].reshape(data['Xva'].shape[0], -1)
    Xte = data['Xte'].reshape(data['Xte'].shape[0], -1)
    Ytr = data['ytr'].reshape(data['ytr'].shape[0], -1)
    Yva = data['yva'].reshape(data['yva'].shape[0], -1)
    sc = StandardScaler().fit(Xtr)
    a, b, c = sc.transform(Xtr), sc.transform(Xva), sc.transform(Xte)
    best_e, best_a = np.inf, ALPHAS[0]
    for al in ALPHAS:
        m = Ridge(alpha=al).fit(a, Ytr)
        e = float(np.mean((m.predict(b) - Yva) ** 2))
        if e < best_e:
            best_e, best_a = e, al
    m = Ridge(alpha=best_a).fit(a, Ytr)
    Yp = m.predict(c).reshape(data['yte'].shape)
    ns, hz_, out_dim = Yp.shape
    pca = data['scY'].inverse_transform(Yp.reshape(-1, out_dim)).reshape(ns, hz_, out_dim)
    pdiff = data['pca_y'].inverse_transform(
        pca.reshape(-1, out_dim)).reshape(ns, hz_, data['pca_y'].n_features_in_)
    return data['init_test'][:, None, :] + np.cumsum(pdiff, axis=1), best_a


# =============================================================================
# 诊断
# =============================================================================
def acf_per_node(x, k):
    vals = []
    for j in range(x.shape[1]):
        s = x[:, j]
        if s.std() < 1e-12:
            continue
        vals.append(float(np.corrcoef(s[k:], s[:-k])[0, 1]))
    return np.asarray(vals, dtype=np.float64)


def ljung_box_p(x, m=5):
    """逐节点 Ljung-Box（Q ~ chi2(m)），返回 p 值数组。"""
    from scipy.stats import chi2
    n = x.shape[0]
    ps = []
    for j in range(x.shape[1]):
        s = x[:, j]
        if s.std() < 1e-12:
            continue
        rk = [float(np.corrcoef(s[k:], s[:-k])[0, 1]) for k in range(1, m + 1)]
        q = n * (n + 2) * sum(rk[k - 1] ** 2 / (n - k) for k in range(1, m + 1))
        ps.append(float(1.0 - chi2.cdf(q, m)))
    return np.asarray(ps, dtype=np.float64)


def variance_ratio(x, q):
    """Lo-MacKinlay 重叠方差比 VR(q)（异方差稳健版）与其同方差 z。"""
    y = np.diff(np.asarray(x, dtype=np.float64))
    n = len(y)
    if n <= q + 1:
        return np.nan, np.nan
    mu = y.mean()
    sig_a = float(np.sum((y - mu) ** 2) / (n - 1))
    qs = np.convolve(y, np.ones(q), 'valid') - q * mu
    mm = q * (n - q + 1) * (1.0 - q / n)
    sig_c = float(np.sum(qs ** 2) / mm)
    if sig_a < 1e-15:
        return np.nan, np.nan
    vr = sig_c / sig_a
    sd = np.sqrt(2.0 * (2 * q - 1) * (q - 1) / (3.0 * q * n))
    return vr, (vr - 1.0) / sd


def adf_p_per_node(x):
    try:
        from statsmodels.tsa.stattools import adfuller
    except Exception:
        return None
    ps = []
    for j in range(x.shape[1]):
        s = x[:, j]
        if s.std() < 1e-12:
            continue
        try:
            ps.append(float(adfuller(s, autolag='AIC')[1]))
        except Exception:
            continue
    return np.asarray(ps, dtype=np.float64) if ps else None


def circular_block_idx(n, L, rng):
    nblk = int(np.ceil(n / L))
    starts = rng.integers(0, n, size=nblk)
    idx = np.concatenate([(np.arange(s, s + L) % n) for s in starts])[:n]
    return idx


def bootstrap_ci(se_model, se_zero, metric_key, n_boot=N_BOOT, L=BOOT_L, seed=BOOT_SEED):
    """循环块 bootstrap：重采样窗口，返回 (skill_ci, rmse_ci) 与点值。

    se_model: (S, n_win) 逐种子逐窗口平方误差；se_zero: (n_win,)。
    skill 口径与论文一致：先对种子取 RMSE 均值，再 1−(RMSE/RMSE_zero)²。
    """
    n_win = se_zero.shape[0]
    rng = np.random.default_rng(seed)
    sk, rg, rz = [], [], []
    for _ in range(n_boot):
        idx = circular_block_idx(n_win, L, rng)
        rm = np.sqrt(se_model[:, idx].mean(axis=1)).mean()
        r0 = float(np.sqrt(se_zero[idx].mean()))
        rg.append(rm)
        rz.append(r0)
        sk.append(1.0 - (rm / r0) ** 2)
    sk = np.asarray(sk)
    rg = np.asarray(rg)
    rz = np.asarray(rz)
    return dict(
        skill_pt=float(1.0 - (np.sqrt(se_model.mean(axis=1)).mean()
                              / np.sqrt(se_zero.mean())) ** 2),
        skill_lo=float(np.percentile(sk, 2.5)), skill_hi=float(np.percentile(sk, 97.5)),
        rmse_pt=float(np.sqrt(se_model.mean(axis=1)).mean()),
        rmse_lo=float(np.percentile(rg, 2.5)), rmse_hi=float(np.percentile(rg, 97.5)),
        zero_pt=float(np.sqrt(se_zero.mean())),
        zero_lo=float(np.percentile(rz, 2.5)), zero_hi=float(np.percentile(rz, 97.5)),
    )


# =============================================================================
# 主流程
# =============================================================================
def seeds_of(d):
    out = []
    if not d.is_dir():
        return out
    for r in sorted(d.glob('run_*')):
        if (r / 'predictions.npz').exists() and (r / 'metrics.json').exists():
            try:
                out.append(int(r.name.split('_')[1]))
            except Exception:
                pass
    return sorted(set(out))


def model_root(kind, key, H):
    return (ABL_ROOT / f'H{H}' / key) if kind == 'abl' else (CMP_ROOT / f'H{H}' / key)


def main():
    print('=' * 100)
    print('P04 增量口径重算 —— 数据管线加载')
    print('=' * 100)
    P = load_pipeline()
    te_den = P['te_den']
    print(f"测试集 T={te_den.shape[0]}  N={te_den.shape[1]}  训练增量 {P['y_train'].shape}  "
          f"PCA K={P['pca_y'].n_components_}（解释率 {P['pca_y'].explained_variance_ratio_.sum():.4f}）")

    # ---------- 构造每 H 窗口 ----------
    WIN = {H: build_windows(te_den, H) for H in HZS}
    ZERO = {}
    for H in HZS:
        nw, init_test, v_last, true_inc, true_cum = WIN[H]
        ZERO[H] = zero_rmse(init_test, true_inc, true_cum)
        print(f"  H={H:>2}: n_win={nw:>2}")

    # =====================================================================
    # 硬性锚点自检
    # =====================================================================
    print('\n' + '=' * 100)
    print('锚点自检（不符即停）')
    print('=' * 100)
    ok = True

    # (a) 零增量累计 RMSE_last
    dev_a = max(abs(ZERO[H]['cum_last'] - PERSIST_LAST[H]) for H in HZS)
    for H in HZS:
        flag = 'OK' if abs(ZERO[H]['cum_last'] - PERSIST_LAST[H]) < 1e-4 else 'FAIL'
        print(f"  (a) H={H:>2}  零增量 cum_RMSE_last={ZERO[H]['cum_last']:.6f}  "
              f"参考={PERSIST_LAST[H]:.6f}  [{flag}]")
    print(f"      → 最大偏差 {dev_a:.2e} {'PASS' if dev_a < 1e-4 else 'FAIL'}")
    ok &= dev_a < 1e-4

    # (b) Ours(V7) H=10 种子均值 cum_last_rmse
    d_ours = model_root('abl', OURS_KEY, 10)
    sds = seeds_of(d_ours)
    nw10, init10, v10, ti10, tc10 = WIN[10]
    ours_last = []
    for s in sds:
        pc = np.load(d_ours / f'run_{s}' / 'predictions.npz')['pred_cum']
        ours_last.append(rmse(np.asarray(pc, float)[:, -1, :], tc10[:, -1, :]))
    om, os_ = agg(ours_last)
    print(f"  (b) Ours(V7) H=10  n_seeds={len(sds)}  cum_last_rmse={om:.4f}±{os_:.4f}  "
          f"参考={OURS_H10_MEAN:.4f}±{OURS_H10_STD:.4f}  "
          f"[{'OK' if abs(om-OURS_H10_MEAN)<1e-3 and abs(os_-OURS_H10_STD)<1e-3 else 'FAIL'}]")
    ok &= (abs(om - OURS_H10_MEAN) < 1e-3 and abs(os_ - OURS_H10_STD) < 1e-3)

    # (c) 代表性 run 的真值一致性
    rep = d_ours / 'run_42' / 'predictions.npz'
    if not rep.exists():
        rep = d_ours / f'run_{sds[0]}' / 'predictions.npz'
    drep = np.load(rep)
    truth_npz = np.asarray(drep['true_cum'], dtype=np.float64)
    truth_ref = init10[:, None, :] + np.cumsum(ti10, axis=1)
    dev_c = float(np.max(np.abs(truth_npz - truth_ref)))
    print(f"  (c) {rep.parent.name}: max|true_cum_npz − (init+cumsum(inc))| = {dev_c:.2e}  "
          f"[{'OK' if dev_c < 1e-6 else 'FAIL'}]")
    ok &= dev_c < 1e-6

    if not ok:
        if EXAMPLE_MODE:
            print('\n*** 锚点未通过（EXAMPLE_MODE=1：合成示例数据，跳过硬锚点）***')
        else:
            print('\n*** 锚点未通过：停止，不写任何产物。***')
            sys.exit(1)
    else:
        print('  → 全部锚点 PASS')

    # ---------- 确定性基线的预测（每 H） ----------
    DET = {}          # H -> {key: pred_cum}
    RIDGE_ALPHA = {}
    for H in HZS:
        nw, init_test, v_last, true_inc, true_cum = WIN[H]
        det = {
            BASE_ZERO: pred_zero(init_test, true_cum),
            BASE_LAST: pred_last_inc(init_test, v_last, H),
            BASE_AR1: pred_ar1(init_test, v_last, P['diff1_den'][P['tr']], H),
            BASE_ORACLE: init_test[:, None, :] + np.cumsum(
                pred_oracle(true_inc, P['pca_y']), axis=1),
        }
        data = prepare_data(P['cfg'], P['full_df'], P['wd'], P['wa'], P['meta'],
                            P['tr'], P['va'], P['te'], P['water_feat'], P['geo_fb'],
                            H, disp_raw=P['disp_raw'])
        rp, ra = pred_ridge(data, H)
        det[BASE_RIDGE] = rp
        RIDGE_ALPHA[H] = ra
        DET[H] = det
    print('\n岭回归选中 alpha：' + ', '.join(f'H{H}={RIDGE_ALPHA[H]:g}' for H in HZS))

    # oracle 地板交叉核验（H=10）
    fl = metrics_from_cum(DET[10][BASE_ORACLE], init10, ti10, tc10)['cum_last']
    print(f"oracle(H=10) 累计末步地板 = {fl:.6f}  参考 FLOORS[10]= {FLOOR_LAST_H10:.6f}  "
          f"[{'OK' if abs(fl-FLOOR_LAST_H10) < 1e-4 else 'WARN'}]")

    # =====================================================================
    # 模型聚合
    # =====================================================================
    models = [('abl', v) for v in VARIANTS] + [('ext', m) for m in EXTERNALS] + [('ext', 'GBDT')]
    rows = []
    store = {}                      # (H, model) -> 逐种子 per-window SE（bootstrap 用）
    nseeds_log = []

    for kind, key in models:
        for H in HZS:
            d = model_root(kind, key, H)
            sds = seeds_of(d)
            if not sds:
                continue
            nw, init_test, v_last, true_inc, true_cum = WIN[H]
            mlist, se_cuml, se_cuma, se_incl, se_inca = [], [], [], [], []
            for s in sds:
                pc = np.load(d / f'run_{s}' / 'predictions.npz')['pred_cum']
                m = metrics_from_cum(pc, init_test, true_inc, true_cum)
                se = per_window_se(pc, init_test, true_inc, true_cum)
                mlist.append(m)
                se_cuml.append(se['cum_last'])
                se_cuma.append(se['cum_all'])
                se_incl.append(se['inc_last'])
                se_inca.append(se['inc_all'])
            dfm = pd.DataFrame(mlist)
            nseeds_log.append((key, H, len(sds)))
            z = ZERO[H]

            def emit(caliber, metric, mean, std):
                rows.append(dict(H=H, model=key, caliber=caliber, metric=metric,
                                 mean=float(mean), std=float(std)))

            for caliber, coll, zl, za in (('cum', 'cum', 'cum_last', 'cum_all'),
                                          ('inc', 'inc', 'inc_last', 'inc_all')):
                last_m, last_s = agg(dfm[f'{coll}_last'])
                all_m, all_s = agg(dfm[f'{coll}_all'])
                z_last, z_all = z[zl], z[za]
                emit(caliber, 'RMSE_last', last_m, last_s)
                emit(caliber, 'RMSE_all', all_m, all_s)
                emit(caliber, 'skill_last', skill(last_m, z_last), 0.0)
                emit(caliber, 'skill_all', skill(all_m, z_all), 0.0)
                emit(caliber, 'skill_mse_last',
                     skill(np.sqrt(np.mean(np.asarray(dfm[f'{coll}_last']) ** 2)), z_last), 0.0)
                emit(caliber, 'skill_mse_all',
                     skill(np.sqrt(np.mean(np.asarray(dfm[f'{coll}_all']) ** 2)), z_all), 0.0)
            cm, cs = agg(dfm['corr'])
            emit('inc', 'corr', cm, cs)

            store[(H, key)] = dict(
                cum_last=np.vstack(se_cuml), cum_all=np.vstack(se_cuma),
                inc_last=np.vstack(se_incl), inc_all=np.vstack(se_inca))

    # ---------- 确定性基线行 ----------
    for H in HZS:
        nw, init_test, v_last, true_inc, true_cum = WIN[H]
        z = ZERO[H]
        for key, pcm in DET[H].items():
            m = metrics_from_cum(pcm, init_test, true_inc, true_cum)
            def emitb(caliber, metric, mean, std=0.0):
                rows.append(dict(H=H, model=key, caliber=caliber, metric=metric,
                                 mean=float(mean), std=float(std)))
            for caliber, pref, zl, za in (('cum', 'cum', 'cum_last', 'cum_all'),
                                          ('inc', 'inc', 'inc_last', 'inc_all')):
                mlast, mall = m[f'{pref}_last'], m[f'{pref}_all']
                emitb(caliber, 'RMSE_last', mlast)
                emitb(caliber, 'RMSE_all', mall)
                emitb(caliber, 'skill_last', skill(mlast, z[zl]))
                emitb(caliber, 'skill_all', skill(mall, z[za]))
                emitb(caliber, 'skill_mse_last', skill(mlast, z[zl]))
                emitb(caliber, 'skill_mse_all', skill(mall, z[za]))
            emitb('inc', 'corr', m['corr'])

    tidy = pd.DataFrame(rows, columns=['H', 'model', 'caliber', 'metric', 'mean', 'std'])
    tidy_path = OUT / 'T_p04_increment_caliber.csv'
    tidy.to_csv(tidy_path, index=False, float_format='%.6f')

    # =====================================================================
    # 诊断
    # =====================================================================
    tr_inc = P['diff1_den'][P['tr']][1:]            # 训练增量（去置 0 首行）
    diag = []

    def drow(block, H, item, caliber, metric, value, ci_lo=None, ci_hi=None, n=None, note=''):
        diag.append(dict(block=block, H=H, item=item, caliber=caliber, metric=metric,
                         value=value, ci_lo=ci_lo, ci_hi=ci_hi, n=n, note=note))

    # ACF（H 无关）
    for k in range(1, 6):
        v = acf_per_node(tr_inc, k)
        drow('acf', 10, f'lag{k}', 'inc', 'acf_mean', float(np.nanmean(v)),
             n=len(v), note=f'std={np.nanstd(v):.4f}（训练增量逐节点）')

    # Ljung-Box（训练增量，逐节点，m=5）
    p_lb = ljung_box_p(tr_inc, m=5)
    drow('ljung_box', 10, 'lag5', 'inc', 'p_frac_sig',
         float(np.mean(p_lb < 0.05)), n=len(p_lb), note='逐节点 p<0.05 占比')
    drow('ljung_box', 10, 'lag5', 'inc', 'p_median', float(np.median(p_lb)),
         n=len(p_lb), note='逐节点 p 中位数')

    # VR(q)（测试累计序列，逐节点）
    for q in (2, 5):
        vrs, zs = [], []
        for j in range(te_den.shape[1]):
            vr, zz = variance_ratio(te_den[:, j], q)
            if np.isfinite(vr):
                vrs.append(vr)
                zs.append(zz)
        vrs = np.asarray(vrs)
        zs = np.asarray(zs)
        drow('variance_ratio', 10, f'q{q}', 'cum', 'VR_mean', float(vrs.mean()),
             ci_lo=float(vrs.mean() - 1.96 * vrs.std() / np.sqrt(len(vrs))),
             ci_hi=float(vrs.mean() + 1.96 * vrs.std() / np.sqrt(len(vrs))),
             n=len(vrs), note=f'VR=1 为随机游走；z 均值={zs.mean():.3f}，'
                              f'|z|>1.96 占比={np.mean(np.abs(zs) > 1.96):.3f}')

    # ADF（测试累计序列，逐节点）
    p_adf = adf_p_per_node(te_den)
    if p_adf is None:
        drow('adf', 10, 'test_cum', 'cum', 'p_median', np.nan, note='statsmodels 不可用')
    else:
        drow('adf', 10, 'test_cum', 'cum', 'p_median', float(np.median(p_adf)),
             n=len(p_adf), note=f'拒绝单位根(p<0.05)占比={np.mean(p_adf < 0.05):.3f}'
                                f'（随机游走应不拒绝）')

    # ---------- 循环块 bootstrap（H=10 主，另给全部 H） ----------
    ent_map = [('zero_increment', None), (OURS_KEY, store), (BASE_RIDGE, None)]
    for H in HZS:
        nw, init_test, v_last, true_inc, true_cum = WIN[H]
        zse = zero_per_window_se(init_test, true_inc, true_cum)
        for name, _ in ent_map:
            if name == 'zero_increment':
                continue
            if name == BASE_RIDGE:
                pc = DET[H][BASE_RIDGE]
                se_map = per_window_se(pc, init_test, true_inc, true_cum)
                smap = {k: np.asarray(v)[None, :] for k, v in se_map.items()}
            else:
                smap = store[(H, name)]
            for caliber, sk, zk in (('cum', 'cum_last', 'cum_last'),
                                    ('inc', 'inc_last', 'inc_last')):
                ci = bootstrap_ci(smap[sk], zse[zk], sk)
                drow('bootstrap', H, name, caliber, 'skill_last', ci['skill_pt'],
                     ci['skill_lo'], ci['skill_hi'], n=N_BOOT,
                     note=f'L={BOOT_L},B={N_BOOT},seed={BOOT_SEED},重采样窗口')
                drow('bootstrap', H, name, caliber, 'RMSE_last', ci['rmse_pt'],
                     ci['rmse_lo'], ci['rmse_hi'], n=N_BOOT,
                     note=f'RMSE_zero={ci["zero_pt"]:.4f}'
                          f'[{ci["zero_lo"]:.4f},{ci["zero_hi"]:.4f}]')
        # 零增量的 RMSE CI（skill 恒为 0）
        for caliber, zk in (('cum', 'cum_last'), ('inc', 'inc_last')):
            ci = bootstrap_ci(np.zeros((1, nw)), zse[zk], zk)
            drow('bootstrap', H, 'zero_increment', caliber, 'RMSE_last', ci['zero_pt'],
                 ci['zero_lo'], ci['zero_hi'], n=N_BOOT, note='skill 恒为 0（参考基线）')

    diag_df = pd.DataFrame(diag, columns=['block', 'H', 'item', 'caliber', 'metric',
                                          'value', 'ci_lo', 'ci_hi', 'n', 'note'])
    diag_path = OUT / 'T_p04_diagnostics.csv'
    diag_df.to_csv(diag_path, index=False, float_format='%.6f')

    # =====================================================================
    # stdout 汇总
    # =====================================================================
    print('\n' + '=' * 100)
    print('P04 结果汇总（seed-mean 口径，skill_last = 1−(RMSE_last/RMSE_zero)²）')
    print('=' * 100)

    def cell(H, model, metric):
        r = tidy[(tidy.H == H) & (tidy.model == model) & (tidy.metric == metric)]
        return None if r.empty else r.iloc[0]

    order = [OURS_KEY, 'V0_Pure_TCN', 'V5_Full_Model', 'V1_wo_MeanPool', 'V2_wo_GCN',
             'V3_wo_PhysicsGate', 'V6_wo_CrossNodeAttn', 'V8_wo_VerticalAll',
             'BiLSTM-Attn', 'GraphWaveNet', 'LSTM', 'STGCN', 'Transformer',
             'V-Attention', 'V-MLP', 'GBDT',
             BASE_ZERO, BASE_LAST, BASE_AR1, BASE_ORACLE, BASE_RIDGE]

    print('\n【H=10】累计 / 增量 双口径末步 RMSE 与 skill')
    print(f"{'model':<20}{'cum_RMSE_last':>16}{'cum_skill':>11}{'inc_RMSE_last':>16}"
          f"{'inc_skill':>11}{'inc_corr':>10}")
    print('-' * 84)
    for m in order:
        c = cell(10, m, 'RMSE_last')
        if c is None:
            continue
        cml = cell(10, m, 'RMSE_last'); cms = cell(10, m, 'skill_last')
        cs = tidy[(tidy.H == 10) & (tidy.model == m) & (tidy.caliber == 'cum')
                  & (tidy.metric == 'RMSE_last')]
        csk = tidy[(tidy.H == 10) & (tidy.model == m) & (tidy.caliber == 'cum')
                   & (tidy.metric == 'skill_last')]
        isk = tidy[(tidy.H == 10) & (tidy.model == m) & (tidy.caliber == 'inc')
                   & (tidy.metric == 'skill_last')]
        irl = tidy[(tidy.H == 10) & (tidy.model == m) & (tidy.caliber == 'inc')
                   & (tidy.metric == 'RMSE_last')]
        cor = tidy[(tidy.H == 10) & (tidy.model == m) & (tidy.metric == 'corr')]
        def f1(d, key='mean'):
            return float(d.iloc[0][key]) if not d.empty else float('nan')
        cstd = float(cs.iloc[0]['std']) if not cs.empty else float('nan')
        print(f"{m:<20}{f1(cs):>10.4f}±{cstd:<5.4f}{f1(csk):>+11.4f}"
              f"{f1(irl):>10.4f}{'':>5}{f1(isk):>+11.4f}{f1(cor):>+10.4f}")

    print('\n【全视界】Ours(V7) 双口径')
    print(f"{'H':>4}{'cum_RMSE_last':>16}{'cum_skill':>11}{'cum_skill_mse':>15}"
          f"{'inc_RMSE_last':>16}{'inc_skill':>11}{'inc_skill_mse':>15}")
    print('-' * 78)
    for H in HZS:
        g = lambda cal, met: float(tidy[(tidy.H == H) & (tidy.model == OURS_KEY)
                                        & (tidy.caliber == cal)
                                        & (tidy.metric == met)].iloc[0]['mean'])
        print(f"{H:>4}{g('cum','RMSE_last'):>16.4f}{g('cum','skill_last'):>+11.4f}"
              f"{g('cum','skill_mse_last'):>+15.4f}{g('inc','RMSE_last'):>16.4f}"
              f"{g('inc','skill_last'):>+11.4f}{g('inc','skill_mse_last'):>+15.4f}")

    print('\n【H=10 基线】零增量 / 末增量(匀速) / AR(1) / oracle / 岭回归 双口径 skill_last')
    print(f"{'baseline':<20}{'cum_skill':>11}{'inc_skill':>11}"
          f"{'cum_RMSE_last':>16}{'inc_RMSE_last':>16}")
    print('-' * 74)
    for m in [BASE_ZERO, BASE_LAST, BASE_AR1, BASE_ORACLE, BASE_RIDGE]:
        g = lambda cal, met: float(tidy[(tidy.H == 10) & (tidy.model == m)
                                        & (tidy.caliber == cal)
                                        & (tidy.metric == met)].iloc[0]['mean'])
        print(f"{m:<20}{g('cum','skill_last'):>+11.4f}{g('inc','skill_last'):>+11.4f}"
              f"{g('cum','RMSE_last'):>16.4f}{g('inc','RMSE_last'):>16.4f}")

    print('\n【诊断】')
    for _, r in diag_df[diag_df.block.isin(['acf', 'ljung_box', 'variance_ratio', 'adf'])].iterrows():
        ci = '' if pd.isna(r['ci_lo']) else f"  95%CI[{r['ci_lo']:.3f},{r['ci_hi']:.3f}]"
        print(f"  {r['block']:<15}{r['item']:<8}{r['metric']:<12}{r['value']:>10.4f}{ci}  {r['note']}")

    print('\n【H=10 bootstrap 95% CI】')
    b = diag_df[(diag_df.block == 'bootstrap') & (diag_df.H == 10)]
    print(f"{'entity':<18}{'caliber':>8}{'metric':>13}{'value':>10}{'ci_lo':>10}{'ci_hi':>10}")
    print('-' * 69)
    for _, r in b.iterrows():
        print(f"{r['item']:<18}{r['caliber']:>8}{r['metric']:>13}"
              f"{r['value']:>10.4f}{r['ci_lo']:>10.4f}{r['ci_hi']:>10.4f}")

    print(f"\n已输出：\n  {tidy_path}\n  {diag_path}")
    print('完成。')


if __name__ == '__main__':
    main()