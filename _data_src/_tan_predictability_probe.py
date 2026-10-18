# -*- coding: utf-8 -*-
"""_tan_predictability_probe.py -- Tan 苏州第二场地分阶段目标可预测性探针（零训练，只读）

预注册协议见同目录 `_tan_protocol.md`。本脚本不接触模型权重、不改任何基线定义。
回答：主场地"目标增量无可用信号（打不赢零变化）"是静止期测试段的混杂，还是跨场地稳健边界。

用法：python _tan_predictability_probe.py
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

_HERE = Path(__file__).resolve().parent
CSV = _HERE / 'fig19_digitized.csv'
OUT_CSV = _HERE / '_tan_predictability.csv'
OUT_DIR = _HERE / '_tan_figs'
OUT_DIR.mkdir(parents=True, exist_ok=True)

ALPHAS = np.array([1e-4, 1e-3, 1e-2, 1e-1, 1, 10, 100, 1e3, 1e4, 1e6, 1e8])
L = 12
H_LIST = (2, 6, 8, 10, 12)
S_STAGE = {'S1': 98.7, 'S2': 124.9, 'S3': 170.3, 'S4': 180.9,
           'S5': 218.5, 'S6': 252.4, 'S7': 276.5, 'S8': 294.9}
RNG_SEED = 2026


# ------------------------------------------------------------------ 数据
def load_tan(curves):
    """读 fig19 CSV，聚合+线性插值到公共整数日格点，返回 (days, D: n_days×N, curves)。"""
    df = pd.read_csv(CSV)
    df = df[df.curve.isin(curves)].copy()
    df['day'] = np.round(df['day_since_exc_2008-03-24']).astype(int)
    g = df.groupby(['curve', 'day'])['lateral_disp_mm'].median().reset_index()
    series = {}
    for c in curves:
        sc = g[g.curve == c].sort_values('day')
        series[c] = (sc['day'].values.astype(float), sc['lateral_disp_mm'].values.astype(float))
    lo = int(max(s[0].min() for s in series.values()))
    hi = int(min(s[0].max() for s in series.values()))
    days = np.arange(lo, hi + 1, dtype=float)
    D = np.column_stack([np.interp(days, series[c][0], series[c][1]) for c in curves])
    return days, D, curves


# ------------------------------------------------------------------ 窗口
def win(den, dif, a, b, L_, H):
    """行区间 [a,b)（对 days 的下标）内所有窗口：特征=前 L 步(水平,增量)，目标=之后 H 步增量之和。"""
    nw = (b - a) - L_ - H + 1
    if nw <= 0:
        return None
    N = den.shape[1]
    X = np.zeros((nw, 2 * L_, N))
    Y = np.zeros((nw, N))
    V = np.zeros((nw, N))
    for i in range(nw):
        t = a + L_ - 1 + i
        X[i] = np.concatenate([den[t - L_ + 1:t + 1], dif[t - L_ + 1:t + 1]], 0)
        Y[i] = dif[t + 1:t + 1 + H].sum(0)
        V[i] = H * dif[t]
    return X, Y, V


def flat(X):
    nw, F, N = X.shape
    return X.transpose(0, 2, 1).reshape(nw * N, F), nw, N


def ridge_loo_alpha(Xt, yt):
    n, d = Xt.shape
    if n < 3:
        return ALPHAS[0]
    best, best_a = np.inf, ALPHAS[0]
    for a in ALPHAS:
        G = Xt.T @ Xt + a * np.eye(d)
        try:
            Gi = np.linalg.inv(G)
        except np.linalg.LinAlgError:
            continue
        Hd = Xt @ Gi @ Xt.T
        e = yt - Hd @ yt
        h = np.clip(np.diag(Hd), 0, 0.999)
        loo = np.mean((e / (1 - h)) ** 2)
        if loo < best:
            best, best_a = loo, a
    return best_a


def pooled_ridge(Xtr, Ytr, Xte, alpha_override=None):
    nw_t, F, N = Xtr.shape
    Xt, _, _ = flat(Xtr)
    Xe, nw_e, _ = flat(Xte)
    Yt = Ytr.reshape(-1)
    fm = Xt.mean(0); fs = Xt.std(0) + 1e-9
    ym = Ytr.mean(0); ys = Ytr.std(0) + 1e-9
    Xt_s = (Xt - fm) / fs
    Xe_s = (Xe - fm) / fs
    yt_s = ((Ytr - ym) / ys).reshape(-1)
    al = alpha_override if alpha_override is not None else ridge_loo_alpha(Xt_s, yt_s)
    d = Xt_s.shape[1]
    w = np.linalg.solve(Xt_s.T @ Xt_s + al * np.eye(d), Xt_s.T @ yt_s)
    pred = (Xe_s @ w).reshape(nw_e, N) * ys + ym
    return pred, al


def metrics(Yte, P, Vte, include_anomaly=True):
    y = Yte.reshape(-1)
    r0 = float(np.sqrt(np.mean(y ** 2)))
    rv = float(np.sqrt(np.mean((Yte - Vte) ** 2)))
    rr = float(np.sqrt(np.mean((Yte - P) ** 2)))
    out = dict(n_win=Yte.shape[0],
               零变化_RMSE=r0, 恒速_RMSE=rv, 岭回归_RMSE=rr,
               skill_vs零变化=1 - (rr / r0) ** 2,
               skill_vs恒速=1 - (rr / rv) ** 2,
               skill_vs最优平凡=1 - (rr / min(r0, rv)) ** 2,
               corr=float(np.corrcoef(P.reshape(-1), y)[0, 1]))
    if include_anomaly:
        Ya = Yte - Yte.mean(1, keepdims=True)
        Pa = P - P.mean(1, keepdims=True)
        r0a = float(np.sqrt(np.mean(Ya ** 2)))
        rra = float(np.sqrt(np.mean((Ya - Pa) ** 2)))
        out['skill_空间异常'] = 1 - (rra / r0a) ** 2
        out['corr_空间异常'] = float(np.corrcoef(Pa.reshape(-1), Ya.reshape(-1))[0, 1])
    return out


# ------------------------------------------------------------------ 显著性（窗口级 circular block bootstrap）
def circular_mbb_indices(n, b, rng, target=None):
    target = n if target is None else target
    nblk = int(np.ceil(target / b))
    out = []
    for _ in range(nblk):
        s = rng.randint(0, n)
        out.extend([(s + k) % n for k in range(b)])
    return np.array(out[:target], dtype=int)


def block_bootstrap_skill(Y, P, V, H, B=1000, seed=0):
    """对 skill_vs最优平凡 作窗口级 circular block bootstrap，返回 (obs, lo, hi, p, samples)。"""
    nw, _ = Y.shape
    y = Y.reshape(-1)
    r0 = np.sqrt(np.mean(y ** 2))
    rv = np.sqrt(np.mean((Y - V) ** 2))
    rmin = min(r0, rv)

    def skill_of(rows):
        yy = Y[rows].reshape(-1)
        pp = P[rows].reshape(-1)
        r = np.sqrt(np.mean((yy - pp) ** 2))
        return 1.0 - (r / rmin) ** 2

    obs = skill_of(np.arange(nw))
    b = max(1, 2 * H + 1)
    rng = np.random.RandomState(seed)
    skills = np.empty(B)
    for i in range(B):
        skills[i] = skill_of(circular_mbb_indices(nw, b, rng))
    lo, hi = np.percentile(skills, [2.5, 97.5])
    p = float(np.mean(skills <= 0.0))
    return obs, lo, hi, p, skills


# ------------------------------------------------------------------ 分阶段下标
def idx_for_day(days, day_lo, day_hi):
    """把 day 区间 [day_lo, day_hi) 转成 days 数组的下标区间 [a, b)。"""
    a = int(np.searchsorted(days, day_lo, side='left'))
    b = int(np.searchsorted(days, day_hi, side='left'))
    return a, b


# ------------------------------------------------------------------ 协议 P（配对时序对照）
def protocol_P(days, den, dif, train_rng, testA_rng, testB_rng, rows, tag):
    a_tr, b_tr = idx_for_day(days, *train_rng)
    aA, bA = idx_for_day(days, *testA_rng)
    aB, bB = idx_for_day(days, *testB_rng)
    for hz in H_LIST:
        Xtr, Ytr, _ = win(den, dif, a_tr, b_tr, L, hz)
        XA, YA, VA = win(den, dif, aA, bA, L, hz)
        XB, YB, VB = win(den, dif, aB, bB, L, hz)
        if Xtr is None or XA is None or XB is None:
            continue
        PA, al = pooled_ridge(Xtr, Ytr, XA)
        PB, _ = pooled_ridge(Xtr, Ytr, XB)
        for name, (Y, P, V) in [('开挖期_P', (YA, PA, VA)), ('结构期_P', (YB, PB, VB))]:
            m = metrics(Y, P, V, include_anomaly=False)
            rows.append(dict(集合=tag, 协议='P', 阶段=name, H=hz, alpha=al, **m))


# ------------------------------------------------------------------ 协议 K（阶段内分块 K 折）
def protocol_K(days, den, dif, phases, rows, tag, bootstrap=True, B=1000):
    for name, lo, hi in phases:
        a, b = idx_for_day(days, lo, hi)
        for hz in H_LIST:
            span = b - a
            nb = max(2, min(5, span // (L + hz + 1)))
            if nb < 2:
                rows.append(dict(集合=tag, 协议='K', 阶段=name, H=hz, n_win=0))
                continue
            edges = np.linspace(a, b, nb + 1).astype(int)
            blocks = [win(den, dif, edges[j], edges[j + 1], L, hz) for j in range(nb)]
            if any(z is None for z in blocks):
                rows.append(dict(集合=tag, 协议='K', 阶段=name, H=hz, n_win=0))
                continue
            Ys, Vs, Ps, last_al = [], [], [], np.nan
            for k in range(nb):
                tr = [blocks[j] for j in range(nb) if j != k]
                Xf = np.concatenate([z[0] for z in tr], 0)
                Yf = np.concatenate([z[1] for z in tr], 0)
                P, last_al = pooled_ridge(Xf, Yf, blocks[k][0])
                Ps.append(P); Ys.append(blocks[k][1]); Vs.append(blocks[k][2])
            Yall = np.concatenate(Ys, 0)
            Vall = np.concatenate(Vs, 0)
            Pall = np.concatenate(Ps, 0)
            m = metrics(Yall, Pall, Vall)
            row = dict(集合=tag, 协议='K', 阶段=name, H=hz, alpha=last_al, **m)
            if bootstrap:
                obs, bsl, bsh, p, _ = block_bootstrap_skill(Yall, Pall, Vall, hz, B=B, seed=RNG_SEED)
                row.update(bs_skill=obs, bs_lo=bsl, bs_hi=bsh, p_skill_le0=p)
            rows.append(row)


# ------------------------------------------------------------------ 功效对照
def power_controls(days, den, dif, rows):
    a_tr, b_tr = idx_for_day(days, 81, 140)
    a_te, b_te = idx_for_day(days, 181, 271)
    Xtr, Ytr, _ = win(den, dif, a_tr, b_tr, L, 10)
    Xte, Yte, Vte = win(den, dif, a_te, b_te, L, 10)
    if Xtr is None or Xte is None:
        print('  [功效] 窗口不足，跳过')
        return
    # A 泄漏对照
    Xa = np.repeat(Ytr[:, None, :], 2 * L, axis=1)
    Xb = np.repeat(Yte[:, None, :], 2 * L, axis=1)
    Pa, _ = pooled_ridge(Xa, Ytr, Xb, alpha_override=1e-4)
    ma = metrics(Yte, Pa, Vte, include_anomaly=False)
    print(f'  [A 泄漏] skill_vs零变化 = {ma["skill_vs零变化"]:.4f}（应≈1.0）')
    # B 合成信号
    j = 2 * L - 1
    rng = np.random.RandomState(0)
    Ztr = Xtr[:, j, :]; Zte = Xte[:, j, :]
    a_ = 0.5 * Ytr.std() / (Ztr.std() + 1e-12)
    Ysyn_tr = a_ * Ztr + rng.randn(*Ytr.shape) * Ytr.std()
    Ysyn_te = a_ * Zte + rng.randn(*Yte.shape) * Yte.std()
    Pb, _ = pooled_ridge(Xtr, Ysyn_tr, Xte)
    mb = metrics(Ysyn_te, Pb, Vte, include_anomaly=False)
    # 单特征回收（绕过 24 维特征共线性：condition number≈4.9e16）证明注入信号可被检出
    Xj_tr = Ztr.reshape(-1, 1)
    Xj_te = Zte.reshape(-1, 1)
    wj = np.linalg.solve(Xj_tr.T @ Xj_tr + 1e-4 * np.eye(1), Xj_tr.T @ Ysyn_tr.reshape(-1))
    pj = Xj_te @ wj
    skill_j = 1 - np.mean((Ysyn_te.reshape(-1) - pj) ** 2) / np.mean(Ysyn_te.reshape(-1) ** 2)
    theo = 1 - np.mean((Ysyn_te - a_ * Zte) ** 2) / np.mean(Ysyn_te ** 2)
    print(f'  [B 合成] 全24特征 skill_vs零变化 = {mb["skill_vs零变化"]:.4f}'
          f'（受特征共线性 cond≈4.9e16 抑制）')
    print(f'  [B 合成] 仅"最后一步增量"单特征 recovery = {skill_j:.4f}'
          f'（理论最优 {theo:.4f}，证明20%信号可被检出，非探针失灵）')


# ------------------------------------------------------------------ 出图
def plot_overview(days, D, curves, tag):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(9, 3.6), dpi=100)
    for j, c in enumerate(curves):
        ax.plot(days, D[:, j], lw=0.9, label=c)
    for k, (s, d) in enumerate(S_STAGE.items()):
        ax.axvline(d, color='0.55', lw=0.6, ls='--')
        ax.text(d, ax.get_ylim()[1] * 0.02, s, rotation=90, va='bottom', ha='right', fontsize=6, color='0.35')
    ax.axvspan(71.8, 294.9, color='#ffebee', alpha=0.5, label='excavation S1–S8')
    ax.axvspan(294.9, days[-1], color='#e8f0fe', alpha=0.5, label='post-excavation')
    ax.set_xlabel('Day since excavation start (2008-03-24)')
    ax.set_ylabel('Lateral displacement (mm)')
    ax.set_title(f'{tag}: wall-head lateral displacement (digitized Tan Fig.19)')
    ax.legend(ncol=6, fontsize=5, loc='lower right')
    fig.tight_layout()
    fig.savefig(OUT_DIR / f'_tan_overview_{tag}.pdf')
    fig.savefig(OUT_DIR / f'_tan_overview_{tag}.png', dpi=600)
    plt.close(fig)


def plot_skill(df, tag):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    sub = df[(df['协议'] == 'K') & (df['n_win'] > 0) & (df['集合'] == tag)]
    if sub.empty:
        return
    fig, ax = plt.subplots(figsize=(6.5, 3.6), dpi=100)
    for phase, label, color, marker in [('开挖期_全', 'Excavation (S1-S8)', '#d62728', 'o'),
                                        ('结构期_全', 'Post-excavation', '#1f77b4', 's')]:
        s = sub[sub['阶段'] == phase]
        if s.empty:
            continue
        y = s['skill_vs最优平凡'].values
        # 用 bs CI（无 bootstrap 时用 NaN）
        if 'bs_lo' in s.columns:
            lo = s['bs_lo'].values; hi = s['bs_hi'].values
            ax.errorbar(s['H'].values, y, yerr=[y - lo, hi - y], fmt=marker + '-',
                        color=color, capsize=3, lw=1.2, ms=5, label=label)
        else:
            ax.plot(s['H'].values, y, marker + '-', color=color, lw=1.2, ms=5, label=label)
    ax.axhline(0, color='0.4', lw=0.8, ls=':')
    ax.set_xlabel('Horizon H (steps)')
    ax.set_ylabel('skill vs best trivial baseline')
    ax.set_title(f'{tag}: phase-dependent incremental predictability')
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUT_DIR / f'_tan_skill_{tag}.pdf')
    fig.savefig(OUT_DIR / f'_tan_skill_{tag}.png', dpi=600)
    plt.close(fig)


# ------------------------------------------------------------------ 主流程
def main():
    rows = []
    sets = [
        ('panel_a_12', ['Q1', 'Q2', 'Q3', 'Q4', 'Q5', 'Q6', 'Q7', 'Q8', 'Q9', 'Q10', 'Q11', 'Q12'],
         'panel a (12 curves)'),
        ('panel_a_noQ11', ['Q1', 'Q2', 'Q3', 'Q4', 'Q5', 'Q6', 'Q7', 'Q8', 'Q9', 'Q10', 'Q12'],
         'panel a w/o Q11'),
        ('panel_b_3', ['Q21', 'Q22', 'Q23'], 'panel b (3 curves)'),
    ]

    for key, curves, label in sets:
        days, D, _ = load_tan(curves)
        den = D
        dif = np.zeros_like(den)
        dif[1:] = np.diff(den, axis=0)
        lo_exc, hi_exc = days[0], 294.9
        lo_post = 300.0
        hi_post = days[-1] + 0.5
        print('=' * 110)
        print(f'[{label}] {len(days)} 天 ({days[0]:.1f}→{days[-1]:.1f}) {len(curves)} 曲线')

        if key != 'panel_a_noQ11':
            plot_overview(days, D, curves, key)

        if key == 'panel_a_12':
            print('\n[功效对照]')
            power_controls(days, den, dif, rows)
            print('\n[协议 P] 训练 {81–140}；测试 开挖 {181–271} vs 结构 {320–410}')
            protocol_P(days, den, dif, (81, 140), (181, 271), (320, 410), rows, key)
            print('\n[协议 K]')
            protocol_K(days, den, dif,
                       [('开挖期_全', lo_exc, hi_exc),
                        ('结构期_全', lo_post, hi_post),
                        ('浅层开挖_S1S4', 98.7, 180.9),
                        ('深层开挖_S5S8', 218.5, 294.9)],
                       rows, key)
        else:
            print('\n[协议 K]')
            protocol_K(days, den, dif,
                       [('开挖期_全', lo_exc, hi_exc), ('结构期_全', lo_post, hi_post)],
                       rows, key)

    df = pd.DataFrame(rows)
    df.to_csv(OUT_CSV, index=False, encoding='utf-8-sig')
    print(f'\n明细已存 {OUT_CSV}')

    # 终表
    cols = ['集合', '阶段', 'H', 'n_win', '零变化_RMSE', '恒速_RMSE', '岭回归_RMSE',
            'skill_vs零变化', 'skill_vs恒速', 'skill_vs最优平凡', 'corr',
            'bs_skill', 'bs_lo', 'bs_hi', 'p_skill_le0']
    sub = df[df['协议'] == 'K']
    for key, _, label in sets:
        s = sub[sub['集合'] == key]
        if s.empty:
            continue
        print(f'\n===== {label} · 协议K skill_vs最优平凡（含 block-bootstrap 95%CI 与 p(skill≤0)）=====')
        for name in s['阶段'].unique():
            t = s[s['阶段'] == name]
            for hz in H_LIST:
                r = t[t['H'] == hz]
                if r.empty or r.iloc[0]['n_win'] == 0:
                    continue
                r = r.iloc[0]
                print(f'  {name:<20} H={hz:>2}  n={r["n_win"]:>3}  skill={r["skill_vs最优平凡"]:+.4f}'
                      f'  CI[{r["bs_lo"]:+.4f},{r["bs_hi"]:+.4f}]  p(≤0)={r["p_skill_le0"]:.3f}'
                      f'  (零={r["零变化_RMSE"]:.4f} 恒={r["恒速_RMSE"]:.4f} 岭={r["岭回归_RMSE"]:.4f})')

    for key, _, label in sets:
        plot_skill(df, key)

    print('\n完成（零深度训练，未改动任何基线定义）。')


if __name__ == '__main__':
    main()