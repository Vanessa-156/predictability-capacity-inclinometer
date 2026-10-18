# -*- coding: utf-8 -*-
"""_p02_timeblock_inference.py — 把统计推断单元从"随机种子"改为"时间块"。

背景：论文表 5/表 4/§4.3 的显著性均以 **30 个随机种子**为独立单元做配对 t 检验。
但同一份测试集只有 **11 个测试窗**（H=10），种子间共享同一批时间窗 ⇒ 种子并非独立
重复，t 检验的自由度被高估。本脚本改用 **window-level 循环块 bootstrap**，把窗口
当作重采样单元（配对：同一次重抽的窗集同时用于两模型、并跨种子共享），输出：
  · 单模型 mean RMSE_last 的 95% CI；
  · 配对 ΔRMSE_last 的 95% CI 与双侧 bootstrap p；
并覆盖表 5 全部模型两两配对、表 4 全部变体 vs Ours、§4.3 压力全部条件。

零训练：只读既有 `predictions.npz`（键 pred_cum/true_cum），不重训、不改既有文件。

同时给出：目标增量/水位的 ACF(φ)、块长敏感性、以及 σ_init vs σ_data 方差分解。
rolling-origin ≥5 折**明确不做**（现有预测锁死单一划分，旧 ckpt 不可复用）。

用法：
  python _p02_timeblock_inference.py
"""
from _env import AUX_ROOT, EXAMPLE_MODE, WORK_ROOT

import os
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

warnings.filterwarnings('ignore')

PROJ = WORK_ROOT
if str(PROJ) not in sys.path:
    sys.path.insert(0, str(PROJ))

from Ablation_Experiment import (          # noqa: E402
    Config, set_seed, load_raw_data_with_meta, load_water_data,
    extract_water_features, build_node_features, denoise_split,
)

ROOT = AUX_ROOT
ABL_ROOT = ROOT / "Ablation_New_5d"
CMP_ROOT = ROOT / "Comparison_Results_w64"
STRESS_ROOT = ROOT / "Stress_Results"
OUT = PROJ / "_paper_tables"
OUT.mkdir(parents=True, exist_ok=True)

HZ = 10                        # 本项主展示步长
LB = 12                        # LOOKBACK
B_BOOT = 2000                  # 主 bootstrap 次数
BOOT_SEED = 42
L_STAR = 3                     # 主块长（依据见选块长小节）
L_SENS = [2, 4]                # 0.5L* 与 1.5L*

OURS = 'V7_wo_VerticalGCN'
VARIANTS = ['V0_Pure_TCN', 'V1_wo_MeanPool', 'V2_wo_GCN', 'V3_wo_PhysicsGate',
            'V5_Full_Model', 'V6_wo_CrossNodeAttn', 'V8_wo_VerticalAll']
# 表 5 对比模型（Ours + 外/内基线；GBDT 计入）
TABLE5_EXT = ['BiLSTM-Attn', 'GraphWaveNet', 'LSTM', 'STGCN', 'Transformer',
              'V-Attention', 'V-MLP', 'GBDT']
# 锚点目标（表 8 / _w64_H10_stats.csv 的种子配对 t 检验 p vs Ours）
ANCHOR_T_P = {'STGCN': 0.0029, 'Transformer': 0.0020, 'V0_Pure_TCN': 0.0142,
              'LSTM': 0.1317, 'BiLSTM-Attn': 0.2657, 'GraphWaveNet': 0.4098}
ZERO_CUM_LAST_H10 = 1.372010
OURS_MEAN_H10, OURS_STD_H10 = 1.5687, 0.0940


# =============================================================================
# 路径解析 / 读取
# =============================================================================
def resolve(kind, key, H):
    if kind == 'abl':
        return ABL_ROOT / f'H{H}' / key
    if kind == 'ext':
        return CMP_ROOT / f'H{H}' / key
    if kind == 'stress':
        cond, m = key
        return STRESS_ROOT / f'H{H}' / cond / m
    raise ValueError(kind)


def run_seeds(d):
    if not d.is_dir():
        return []
    out = []
    for r in sorted(d.glob('run_*')):
        if (r / 'predictions.npz').exists():
            try:
                out.append(int(r.name.split('_')[1]))
            except Exception:
                pass
    return sorted(set(out))


def window_se(pred_cum, true_cum):
    """返回逐窗平方误差（对节点取均值）：累计口径 & 增量口径的末步。"""
    pc = np.asarray(pred_cum, dtype=np.float64)
    tc = np.asarray(true_cum, dtype=np.float64)
    se_cum = np.mean((pc[:, -1, :] - tc[:, -1, :]) ** 2, axis=1)
    pd_last = pc[:, -1, :] - pc[:, -2, :]
    td_last = tc[:, -1, :] - tc[:, -2, :]
    se_inc = np.mean((pd_last - td_last) ** 2, axis=1)
    return se_cum, se_inc


_CACHE = {}


def get_se(kind, key, H=HZ):
    """(seeds, SE_cum(S,nw), SE_inc(S,nw))，带缓存。"""
    tag = (kind, key, H)
    if tag in _CACHE:
        return _CACHE[tag]
    d = resolve(kind, key, H)
    sds = run_seeds(d)
    if not sds:
        _CACHE[tag] = (None, None, None)
        return _CACHE[tag]
    cum, inc = [], []
    for s in sds:
        z = np.load(d / f'run_{s}' / 'predictions.npz')
        a, b = window_se(z['pred_cum'], z['true_cum'])
        cum.append(a)
        inc.append(b)
    _CACHE[tag] = (sds, np.vstack(cum), np.vstack(inc))
    return _CACHE[tag]


def metrics_json_se(d, seeds):
    """从 metrics.json 读 RMSE_last（用于与 T3 文件对照，非主口径）。"""
    out = {}
    for s in seeds:
        p = d / f'run_{s}' / 'metrics.json'
        if p.exists():
            import json
            out[s] = float(json.load(open(p, encoding='utf-8'))['RMSE_last'])
    return out


# =============================================================================
# bootstrap 工具
# =============================================================================
def circular_block_idx(n, L, rng):
    nblk = int(np.ceil(n / L))
    starts = rng.integers(0, n, size=nblk)
    return np.concatenate([(np.arange(s, s + L) % n) for s in starts])[:n]


def pooled_rmse(se, idx):
    """se (S,nw) MSE → 重采样窗集 idx 下的种子均值 RMSE（论文口径：先对种子平均 RMSE）。"""
    return float(np.sqrt(se[:, idx].mean(axis=1)).mean())


def boot_stats(seA, seB, idx_sets):
    """对给定窗集列表，返回 Δ=RMSE_A−RMSE_B 的分布，以及单模型 RMSE 分布。"""
    B = len(idx_sets)
    dA = np.empty(B)
    dB = np.empty(B)
    for i, idx in enumerate(idx_sets):
        dA[i] = pooled_rmse(seA, idx)
        dB[i] = pooled_rmse(seB, idx) if seB is not None else np.nan
    return dA, dB


def ci95(a):
    lo, hi = np.percentile(a, [2.5, 97.5])
    return float(lo), float(hi)


def p_two_sided(delta):
    pos = float(np.mean(delta > 0))
    neg = float(np.mean(delta < 0))
    return float(min(1.0, 2.0 * min(pos, neg)))


# =============================================================================
# 主流程
# =============================================================================
def main():
    print('=' * 100)
    print('时间块推断（window-level circular block bootstrap）')
    print('=' * 100)

    # ---------- 数据管线段（算 ACF 用） ----------
    set_seed(42)
    cfg = Config()
    cfg.LOOKBACK = LB
    full_df = load_raw_data_with_meta(cfg)
    ts = np.sort(full_df['测量时间'].unique())
    wd, wa = load_water_data(ts, cfg)
    water_feat = extract_water_features(wd, wa)
    disp_raw, _, meta = build_node_features(full_df, cfg)
    n = len(ts)
    tr = np.arange(0, int(n * cfg.TRAIN_RATIO))
    va = np.arange(int(n * cfg.TRAIN_RATIO), int(n * (cfg.TRAIN_RATIO + cfg.VAL_RATIO)))
    te = np.arange(int(n * (cfg.TRAIN_RATIO + cfg.VAL_RATIO)), n)
    tr_den, va_den, te_den = denoise_split(disp_raw[tr], disp_raw[va], disp_raw[te], cfg)

    # 窗口（H=10）
    n_win = te_den.shape[0] - LB - HZ + 1
    init_idx = np.arange(LB - 1, LB - 1 + n_win)
    init_test = te_den[init_idx]
    true_last = te_den[init_idx + HZ]                     # (nw,N) 累计末步真值
    zero_cum = float(np.sqrt(np.mean((init_test - true_last) ** 2)))
    print(f'测试 T={te_den.shape[0]} N={te_den.shape[1]} | H={HZ} n_win={n_win} | '
          f'窗口索引 {init_idx[0]}..{init_idx[-1]}')

    # ================= 锚点自检 =================
    print('\n' + '=' * 100)
    print('锚点自检（先打印；失败即停）')
    print('=' * 100)
    ok = True
    sds_o, se_cum_o, se_inc_o = get_se('abl', OURS)
    per_seed = np.sqrt(se_cum_o.mean(axis=1))             # 逐种子 pooled RMSE_last
    m, sd = float(per_seed.mean()), float(per_seed.std(ddof=1))
    print(f'  (a) Ours({OURS}) H=10  逐种子 RMSE_last 均值 = {m:.4f} ± {sd:.4f}  '
          f'（参考 {OURS_MEAN_H10:.4f} ± {OURS_STD_H10:.4f}）  '
          f"[{'OK' if abs(m-OURS_MEAN_H10)<1e-3 and abs(sd-OURS_STD_H10)<1e-3 else 'FAIL'}]")
    ok &= (abs(m - OURS_MEAN_H10) < 1e-3 and abs(sd - OURS_STD_H10) < 1e-3)

    print(f'  (c) 零变化 H=10 累计 RMSE_last = {zero_cum:.6f}  '
          f"（参考 {ZERO_CUM_LAST_H10:.6f}）  "
          f"[{'OK' if abs(zero_cum-ZERO_CUM_LAST_H10)<1e-4 else 'FAIL'}]")
    ok &= abs(zero_cum - ZERO_CUM_LAST_H10) < 1e-4

    # (b) 表 8 种子配对 t 检验 p（由 pred_cum 复算）
    print('  (b) 表 8 种子配对 t 检验 p（由 pred_cum 复算，vs Ours）:')
    anchor_rows = []
    for name, kind, key in ([('TCN-Only', 'abl', 'V0_Pure_TCN')]
                            + [(x, 'ext', x) for x in TABLE5_EXT if x != 'GBDT']):
        sds_x, se_cum_x, _ = get_se(kind, key)
        if sds_x is None:
            print(f'      {name}: 缺数据')
            continue
        common = sorted(set(sds_o) & set(sds_x))
        idx_o = [sds_o.index(s) for s in common]
        idx_x = [sds_x.index(s) for s in common]
        a = np.sqrt(se_cum_o[idx_o].mean(axis=1))
        b = np.sqrt(se_cum_x[idx_x].mean(axis=1))
        t, p = stats.ttest_rel(a, b)
        tgt = ANCHOR_T_P.get('V0_Pure_TCN' if name == 'TCN-Only' else name, None)
        flag = ''
        if tgt is not None:
            flag = 'OK' if abs(p - tgt) < 5e-4 else 'DIFF'
            ok &= abs(p - tgt) < 5e-4
        anchor_rows.append((name, float(t), float(p), tgt))
        print(f'      {name:<14} n={len(common)}  t={t:+.3f}  p={p:.4f}  '
              f'（表 8 参考 {tgt}）[{flag}]')
    # GBDT
    sds_g, se_cum_g, _ = get_se('ext', 'GBDT')
    if sds_g is not None:
        common = sorted(set(sds_o) & set(sds_g))
        a = np.sqrt(se_cum_o[[sds_o.index(s) for s in common]].mean(axis=1))
        b = np.sqrt(se_cum_g[[sds_g.index(s) for s in common]].mean(axis=1))
        t, p = stats.ttest_rel(a, b)
        print(f'      {"GBDT":<14} n={len(common)}  t={t:+.3f}  p={p:.4g}  '
              f'（表 8 参考 <0.0001）[{"OK" if p < 1e-4 else "DIFF"}]')
        ok &= p < 1e-4

    # 与 T3 文件对照（信息性，不参与判定）
    t3 = OUT / 'T3_significance_comparison.csv'
    if t3.exists():
        d3 = pd.read_csv(t3)
        d3 = d3[d3['Horizon'] == HZ]
        print('  与 T3_significance_comparison.csv（metrics.json 口径）对照：')
        for _, r in d3.iterrows():
            print(f"      {r['Comparison']:<28} RMSE_last_p={r['RMSE_last_p']} "
                  f"(Δ%={r['Delta_RMSE_pct']})")

    if not ok:
        if EXAMPLE_MODE:
            print('\n*** 锚点未通过（EXAMPLE_MODE=1：合成示例数据，跳过硬锚点）***')
        else:
            print('\n*** 锚点未通过：停止，不写产物。***')
            sys.exit(1)
    else:
        print('  → 全部锚点 PASS')

    # ================= ACF / φ =================
    print('\n' + '=' * 100)
    print('ACF / φ（lag 1..5；目标增量为去噪一阶差分，逐节点平均）')
    print('=' * 100)

    def acf_mean(x, lag):
        vals = []
        for j in range(x.shape[1]):
            s = x[:, j]
            if s.std() < 1e-12:
                continue
            vals.append(float(np.corrcoef(s[lag:], s[:-lag])[0, 1]))
        return float(np.mean(vals)) if vals else float('nan')

    tr_dif = np.diff(tr_den, axis=0)
    te_dif = np.diff(te_den, axis=0)
    ac_rows = []

    def emit_acf(series, segment, x):
        for lag in range(1, 6):
            v = acf_mean(x, lag)
            ac_rows.append(dict(series=series, segment=segment, lag=lag, acf=v))
        print(f'  {series:<18}[{segment:<5}] ' +
              ' '.join(f'ρ{lag}={acf_mean(x, lag):+.3f}' for lag in range(1, 6)))

    emit_acf('target_increment', 'train', tr_dif)
    emit_acf('target_increment', 'test', te_dif)
    emit_acf('water_depth', 'train', wd[tr][:, None])
    emit_acf('water_depth', 'test', wd[te][:, None])
    emit_acf('water_cumul', 'train', wa[tr][:, None])
    emit_acf('water_cumul', 'test', wa[te][:, None])

    # 窗级 RMSE_last 序列的 ACF（bootstrap 的重采样单元）
    win_rmse = np.sqrt(se_cum_o.mean(axis=0))
    emit_acf('window_rmse_last_ours', 'test', win_rmse[:, None])

    # 沉降 / 锚索：主场地无该序列（跨场地 Liugou/Tan 不属于本目标事件）
    for nm in ('subsidence', 'anchor_cable'):
        ac_rows.append(dict(series=nm, segment='未取得', lag=0, acf=float('nan')))
    print('  subsidence / anchor_cable：主场地无该时间序列，标注"未取得"'
          '（跨场地 Liugou/Tan 数据不属本目标事件，未纳入）')

    ac_df = pd.DataFrame(ac_rows, columns=['series', 'segment', 'lag', 'acf'])
    ac_df.to_csv(OUT / 'T_p02_autocorr.csv', index=False, float_format='%.6f')

    # ---------- 选块长 ----------
    phi1 = acf_mean(tr_dif, 1)
    ncube = n_win ** (1.0 / 3.0)
    print(f'\n  φ(lag1, 训练增量) = {phi1:.4f}；n_win^(1/3) = {ncube:.2f}')
    print(f'  → 主块长 L* = {L_STAR}（取 n_win^(1/3) 上整；φ 仅 0.36、lag2/lag3 转负，'
          f'有效记忆约 2–3 步。n_win={n_win} 下独立块数 ≈ {n_win/L_STAR:.1f}）')

    # ---------- 预生成窗集（跨配对/跨口径共享，保证配对） ----------
    rng = np.random.default_rng(BOOT_SEED)
    idx_sets = {}
    for L in [L_STAR] + L_SENS:
        idx_sets[L] = [circular_block_idx(n_win, L, rng) for _ in range(B_BOOT)]
    print(f'  bootstrap：B={B_BOOT}，seed={BOOT_SEED}，块长 {sorted(idx_sets)}；'
          f'同一次重抽窗集用于两模型且跨种子共享（配对）')

    # ================= 模型清单 =================
    # 表 5：Ours + TCN-Only + 外部/内基线（含 GBDT）
    t5 = [('Ours', 'abl', OURS), ('TCN-Only', 'abl', 'V0_Pure_TCN')]
    for x in TABLE5_EXT:
        t5.append((x, 'ext', x))
    # 去重（GBDT 等）
    seen, t5u = set(), []
    for nm, k, key in t5:
        if nm in seen:
            continue
        seen.add(nm)
        t5u.append((nm, k, key))

    # 表 4：变体 vs Ours
    t4 = [(v, 'abl', v) for v in VARIANTS]

    # §4.3 压力：枚举 cond × model
    stress_conds = sorted([d.name for d in (STRESS_ROOT / f'H{HZ}').iterdir()
                           if d.is_dir() and d.name != 'stress_visualization'])
    stress_models = sorted([d.name for d in (STRESS_ROOT / f'H{HZ}' / stress_conds[0]).iterdir()
                            if d.is_dir()]) if stress_conds else []
    print(f'\n  压力条件({len(stress_conds)})：{stress_conds}')
    print(f'  压力模型({len(stress_models)})：{stress_models}')
    others = [mm for mm in stress_models if mm != OURS]

    # ================= bootstrap 计算 =================
    rows = []
    pairs_t = {}          # (tag_a, tag_b) -> 种子配对 t 检验 p（对照用）

    def tag_of(kind, key, cond=None):
        return (kind, key)

    def paired_t(ta, tb):
        sa, sea, _ = get_se(*ta)
        sb, seb, _ = get_se(*tb)
        if sa is None or sb is None:
            return None
        common = sorted(set(sa) & set(sb))
        a = np.sqrt(sea[[sa.index(s) for s in common]].mean(axis=1))
        b = np.sqrt(seb[[sb.index(s) for s in common]].mean(axis=1))
        t, p = stats.ttest_rel(a, b)
        return float(t), float(p), len(common)

    def emit_pair(scope, cond, name_a, ta, name_b, tb, caliber, do_sens=True):
        sa, sea_c, sea_i = get_se(*ta)
        sb, seb_c, seb_i = get_se(*tb)
        if sa is None or sb is None:
            return
        seA = sea_c if caliber == 'cum' else sea_i
        seB = seb_c if caliber == 'cum' else seb_i
        pt = paired_t(ta, tb)
        # 主块长
        dA, dB = boot_stats(seA, seB, idx_sets[L_STAR])
        delta = dA - dB
        dpt = float(pooled_rmse(seA, np.arange(n_win)) - pooled_rmse(seB, np.arange(n_win)))
        lo, hi = ci95(delta)
        pb = p_two_sided(delta)
        ptt = '' if pt is None else f'{pt[1]:.4f}'
        note = (f'Δ=RMSE({name_a})−RMSE({name_b}); n_win={n_win}; 独立块≈{n_win/L_STAR:.1f}; '
                f'配对t_p={ptt}(种子口径,勿与p_boot混用)')
        rows.append(dict(scope=scope, H=HZ, cond=cond, model_a=name_a, model_b=name_b,
                         caliber=caliber, delta_RMSE=dpt, ci_lo=lo, ci_hi=hi,
                         p_boot=pb, L=L_STAR, n_win=n_win, note=note))
        # 块长敏感性
        if do_sens:
            for L in L_SENS:
                dA2, dB2 = boot_stats(seA, seB, idx_sets[L])
                dl = dA2 - dB2
                lo2, hi2 = ci95(dl)
                rows.append(dict(scope=scope, H=HZ, cond=cond, model_a=name_a,
                                 model_b=name_b, caliber=caliber, delta_RMSE=dpt,
                                 ci_lo=lo2, ci_hi=hi2, p_boot=p_two_sided(dl),
                                 L=L, n_win=n_win,
                                 note='块长敏感性（0.5L* / 1.5L*）'))

    def emit_single(name, ta, caliber):
        sa, sea_c, sea_i = get_se(*ta)
        if sa is None:
            return
        se = sea_c if caliber == 'cum' else sea_i
        dA, _ = boot_stats(se, None, idx_sets[L_STAR])
        pt = float(pooled_rmse(se, np.arange(n_win)))
        lo, hi = ci95(dA)
        rows.append(dict(scope='single_model', H=HZ, cond='', model_a=name,
                         model_b='(single)', caliber=caliber, delta_RMSE=pt,
                         ci_lo=lo, ci_hi=hi, p_boot=float('nan'), L=L_STAR, n_win=n_win,
                         note='单模型 mean RMSE_last：delta_RMSE 列=点值，ci=95%CI'))

    print('\n计算 bootstrap …')
    # 表 5 两两配对
    for i in range(len(t5u)):
        for j in range(i + 1, len(t5u)):
            na, ka, keya = t5u[i]
            nb, kb, keyb = t5u[j]
            for cal in ('cum', 'inc'):
                emit_pair('T5_pairwise', '', na, tag_of(ka, keya), nb, tag_of(kb, keyb), cal)
    # 表 4 变体 vs Ours
    for v, k, key in t4:
        for cal in ('cum', 'inc'):
            emit_pair('T4_ablation_vs_Ours', '', 'Ours', tag_of('abl', OURS),
                      v, tag_of(k, key), cal)
    # §4.3 压力：Ours vs 各基线
    for cond in stress_conds:
        for m2 in others:
            for cal in ('cum', 'inc'):
                emit_pair('S4.3_stress', cond, 'Ours',
                          tag_of('stress', (cond, OURS)),
                          m2, tag_of('stress', (cond, m2)), cal)
    # 单模型 CI（主块长）
    single_tags = [('Ours', tag_of('abl', OURS))]
    single_tags += [(nm, tag_of(k, key)) for nm, k, key in t5u if nm != 'Ours']
    single_tags += [(v, tag_of(k, key)) for v, k, key in t4]
    for cond in stress_conds:
        single_tags.append((f'Ours@{cond}', tag_of('stress', (cond, OURS))))
    for nm, ta in single_tags:
        for cal in ('cum', 'inc'):
            emit_single(nm, ta, cal)

    bb = pd.DataFrame(rows, columns=['scope', 'H', 'cond', 'model_a', 'model_b', 'caliber',
                                     'delta_RMSE', 'ci_lo', 'ci_hi', 'p_boot', 'L',
                                     'n_win', 'note'])
    bb.to_csv(OUT / 'T_p02_block_bootstrap.csv', index=False, float_format='%.6f')
    print(f'  → {OUT / "T_p02_block_bootstrap.csv"}（{len(bb)} 行）')

    # ================= 方差分解（H=10 各模型） =================
    var_rows = []
    for nm, ta in single_tags:
        sa, sea, _ = get_se(*ta)
        if sa is None:
            continue
        Y = sea                                     # (S,nw) MSE 尺度 [mm²]
        S, nw = Y.shape
        mu = Y.mean()
        m_s = Y.mean(axis=1)                        # 先对窗求均值 → 跨种子
        m_w = Y.mean(axis=0)                        # 先对种子求均值 → 跨窗
        sigma_init = float(m_s.std(ddof=1))
        sigma_data = float(m_w.std(ddof=1))
        # (seed×window) 二元加性分解（矩估计）
        ss_seed = nw * float(((m_s - mu) ** 2).sum())
        ss_win = S * float(((m_w - mu) ** 2).sum())
        ss_tot = float(((Y - mu) ** 2).sum())
        ss_res = max(ss_tot - ss_seed - ss_win, 0.0)
        ms_seed = ss_seed / (S - 1)
        ms_win = ss_win / (nw - 1)
        ms_res = ss_res / max((S - 1) * (nw - 1), 1)
        s2_seed = max((ms_seed - ms_res) / nw, 0.0)
        s2_win = max((ms_win - ms_res) / S, 0.0)
        tot = sigma_init ** 2 + sigma_data ** 2
        var_rows.append(dict(
            H=HZ, model=nm, sigma_init=sigma_init, sigma_data=sigma_data,
            share_init=float(sigma_init ** 2 / tot) if tot > 0 else float('nan'),
            share_data=float(sigma_data ** 2 / tot) if tot > 0 else float('nan'),
            note=f'MSE(mm²)尺度; 二分量矩估计 sqrt(σ²_seed)={np.sqrt(s2_seed):.4f},'
                 f'sqrt(σ²_win)={np.sqrt(s2_win):.4f}, resid={np.sqrt(ms_res):.4f},'
                 f'n_seed={S},n_win={nw}'))
    vd = pd.DataFrame(var_rows, columns=['H', 'model', 'sigma_init', 'sigma_data',
                                         'share_init', 'share_data', 'note'])
    vd.to_csv(OUT / 'T_p02_variance_components.csv', index=False, float_format='%.6f')

    # ================= rolling-origin 标记 =================
    ro_note = ('rolling-origin ≥5 折：未做。现有 predictions.npz 锁死单一时间划分'
               '（train70/val15/test15），旧 ckpt 不可复用 ⇒ 必须重训，超出零训练范围。')
    print('\n' + '=' * 100)
    print('rolling-origin：' + ro_note)
    print('=' * 100)

    # ================= stdout 汇总 =================
    print('\n【单模型 mean RMSE_last 95% CI（主块长 L*=3）】')
    print(f"{'model':<24}{'cum_RMSE':>11}{'cum_CI':>22}{'inc_RMSE':>11}{'inc_CI':>22}")
    print('-' * 90)
    for nm, _ in single_tags:
        row = lambda c: bb[(bb.scope == 'single_model') & (bb.model_a == nm)
                           & (bb.caliber == c) & (bb.L == L_STAR)]
        rc, ri = row('cum'), row('inc')
        if rc.empty:
            continue
        rc, ri = rc.iloc[0], ri.iloc[0]
        print(f"{nm:<24}{rc['delta_RMSE']:>11.4f}"
              f"{('[%.4f,%.4f]' % (rc['ci_lo'], rc['ci_hi'])):>22}"
              f"{ri['delta_RMSE']:>11.4f}{('[%.4f,%.4f]' % (ri['ci_lo'], ri['ci_hi'])):>22}")

    # 关键：块 bootstrap vs 种子 t 检验 的结论翻转
    print('\n【口径翻转检测：种子配对 t 检验(p<0.05) vs 时间块 bootstrap(p_boot<0.05)】')
    main = bb[(bb.L == L_STAR) & (bb.model_b != '(single)')].copy()
    flips, n_tot, n_t_sig, n_b_sig = [], 0, 0, 0
    for _, r in main.iterrows():
        note = r['note']
        tp = None
        if '配对t_p=' in note:
            s_ = note.split('配对t_p=')[1].split('(')[0]
            try:
                tp = float(s_)
            except Exception:
                tp = None
        if tp is None:
            continue
        n_tot += 1
        sig_t = tp < 0.05
        sig_b = r['p_boot'] < 0.05
        n_t_sig += int(sig_t)
        n_b_sig += int(sig_b)
        if sig_t != sig_b:
            flips.append((r['scope'], r['cond'], r['model_a'], r['model_b'],
                          r['caliber'], tp, r['p_boot'], sig_t, sig_b))
    ns2sig = [f for f in flips if (not f[7]) and f[8]]
    sig2ns = [f for f in flips if f[7] and (not f[8])]
    print(f'  主块长 L*={L_STAR}：共 {n_tot} 个配对。种子 t 显著 {n_t_sig} 个；'
          f'块 bootstrap 显著 {n_b_sig} 个。')
    print(f'  翻转 {len(flips)} 条 = ns→sig {len(ns2sig)} 条 + sig→ns {len(sig2ns)} 条 '
          f'（后者=真正"结论由显著变不显著"的保守方向）')
    if sig2ns:
        print('  ★ sig→ns（时间块口径下失去显著性）：')
        for f in sig2ns:
            print(f'    [{f[0]}{("/"+f[1]) if f[1] else ""}] {f[2]} vs {f[3]} '
                  f'({f[4]}): t_p={f[5]:.4f} → p_boot={f[6]:.4f}')
    else:
        print('  ★ sig→ns：0 条 ⇒ 时间块口径并未使任何"种子显著"配对失去显著性；'
              '反而是大量 ns→sig（见下）。')
    print(f'  ns→sig（种子不显著、时间块显著）示例（前 25 条，共 {len(ns2sig)} 条）：')
    for f in ns2sig[:25]:
        print(f'    [{f[0]}{("/"+f[1]) if f[1] else ""}] {f[2]} vs {f[3]} '
              f'({f[4]}): t_p={f[5]:.4f} → p_boot={f[6]:.4f}')

    print('\n【σ_init vs σ_data 占比（累计口径 MSE 尺度，H=10）】')
    print(f"{'model':<24}{'σ_init':>10}{'σ_data':>10}{'share_init':>12}{'share_data':>12}")
    print('-' * 68)
    for _, r in vd.iterrows():
        print(f"{r['model']:<24}{r['sigma_init']:>10.4f}{r['sigma_data']:>10.4f}"
              f"{r['share_init']:>12.3f}{r['share_data']:>12.3f}")

    print('\n【块长敏感性（p_boot 随 L=2/3/4 变化）：ns→sig 且 p_boot 接近 .05 的配对】')
    shown = 0
    for f in ns2sig:
        scope, cond, ma, mb, cal = f[0], f[1], f[2], f[3], f[4]
        line = bb[(bb.scope == scope) & (bb.cond == cond) & (bb.model_a == ma)
                  & (bb.model_b == mb) & (bb.caliber == cal)]
        ps = {int(r['L']): r['p_boot'] for _, r in line.iterrows()}
        if max(ps.values()) >= 0.01:      # 只列仍在 L 之间明显变动的
            print(f'    [{scope}{("/"+cond) if cond else ""}] {ma} vs {mb} ({cal}): ' +
                  ' '.join(f'L={L}:{ps[L]:.4f}' for L in sorted(ps)))
            shown += 1
    if shown == 0:
        print('    （无：所有 ns→sig 配对在 L=2/3/4 下 p_boot 均 ≈0，结论不随块长翻转）')

    print(f"\n已输出：\n  {OUT/'T_p02_block_bootstrap.csv'}"
          f"\n  {OUT/'T_p02_variance_components.csv'}"
          f"\n  {OUT/'T_p02_autocorr.csv'}")
    print('完成。')


if __name__ == '__main__':
    main()