# -*- coding: utf-8 -*-
"""_p09_skill_states.py -- 状态指标 S 与逐管/逐序列 skill 回归（零训练，只读）。

回答：§4.5"状态依赖"主线能否在**单位观测单元（管/序列）**粒度上成立——
即变形状态越活跃（变形速率越高），增量可预测性 skill 是否越高。

口径（零训练；估算器与 _tan_predictability_probe.py / _liugou_predictability_probe.py 一致，只是
把"跨单元聚合 skill"分解为"池化训练—逐单元归因 skill"）：
- 单位观测单元：Tan 15 管（Q1-Q12 + Q21-Q23）、柳沟 8 序列（x 向）。
- 每个（场地，阶段）窗口内**阶段内对半切**（前半池化训练、后半样本外测试）；
  池化岭回归 + 零变化/恒速双基线，显著性用窗口级 circular block bootstrap（B=500/1000）。
- skill 响应 = skill_vs零变化 = 1 - (RMSE_ridge / RMSE_零)^2 —— 零变化（维持不变）的
  r0 定义良好，不会因恒速基线退化而爆炸；这是逐单元归因的稳定可预测性指标。
- 状态指标 S：S_act = σ(Δ)（活动性幅度，mm/day）；S_rate = 中位|Δ|（mm/day）；
  S_ar1 = Δ 的 lag-1 自相关（平滑度）；S_snrt = 趋势-噪声比。柳沟 8h 步 ×3 归一到日/天。
- 回归：**Spearman ρ（秩）+ 秩 bootstrap 95% CI** 为主判据。不用 log10 最小二乘斜率：
  S_act 双峰（基坑开挖/静止）且单单元 skill 近坍塌，OLS 斜率至 ~10^5 不可解释。
- **否证条件（预注册）**：对任一 S，若 ρ 的 95%CI 含 0（场样本经可检下限 n=49、单侧
  α=0.05 约 |ρ|<0.28 校核），则"状态越活跃增量越可预测"在单位单元粒度不获支持；
  若 ρ 显著为负，则该朴素单调主张被否证。此时正文 §4.5 状态依赖叙事须降级为
  "phase 聚合均值层面的探针级一致性，单位单元粒度无单调律"。

产出：
- _paper_tables/T_p19_skill_states.csv      逐管/逐序列明细（5 视界）
- _paper_tables/T_p19_skill_regression.csv  Spearman ρ + 秩 bootstrap CI 汇总
- _paper_tables/T_p19_digitization_unc.csv  数字化不确定度（S7 标定残差 + 格点化协议敏感性）
- _p09_figs/Fig_Sskill_regression.png/pdf   双联散点（S_act 与 S_ar1；先产出待作者审核，勿自动入稿）

用法：python -u _p09_skill_states.py
"""
from _env import WORK_ROOT

import sys
import math
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.stats as st

ROOT = WORK_ROOT
sys.path.insert(0, str(ROOT / 'Tan_Digitized'))
from _tan_predictability_probe import (win, pooled_ridge, metrics,
                                       block_bootstrap_skill, L, H_LIST,
                                       idx_for_day)

RNG_SEED = 2026
OUT_TABLES = ROOT / '_paper_tables'
OUT_FIGS = ROOT / '_p09_figs'
OUT_TABLES.mkdir(parents=True, exist_ok=True)
OUT_FIGS.mkdir(parents=True, exist_ok=True)

TAN_CSV = ROOT / 'Tan_Digitized' / 'fig19_digitized.csv'
LIOU_CSV = ROOT / 'Liugou_Processed' / 'inclinometer_mm_full.csv'

TAN_CURVES = [f'Q{i}' for i in range(1, 13)] + ['Q21', 'Q22', 'Q23']
LIOU_COLS = ['INC_229_x_d0.55m_mm', 'INC_229_x_d1.35m_mm', 'INC_229_x_d3.05m_mm', 'INC_229_x_d4.75m_mm',
             'INC_230_x_d0.55m_mm', 'INC_230_x_d1.35m_mm', 'INC_230_x_d3.05m_mm', 'INC_230_x_d4.75m_mm']
LIOU_N = 802  # 230 孔末尾 14 时次 NaN，取公共有效 [0, 802)
LIOU_PHASES = [('全时段', 0, LIOU_N), ('前半年', 0, LIOU_N // 2), ('后半年', LIOU_N // 2, LIOU_N)]


# ------------------------------------------------------------------ 阶段内对半切 skill（单位观测单元）
def split_skill(den, dif, a, b, H, B=500):
    """观测窗 [a,b) 内对半切：前半训练、后半测试（样本外）；返回 metrics + block bootstrap。"""
    span = b - a
    split = a + span // 2
    wtr = win(den, dif, a, split, L, H)
    wte = win(den, dif, split, b, L, H)
    if wtr is None or wte is None:
        return None
    Xtr, Ytr, _ = wtr
    Xte, Yte, Vte = wte
    # 目标增量测试段近零（无增量信号）时 skill 无定义，跳过
    if float(np.sqrt(np.mean(Yte ** 2))) < 1e-12:
        return None
    P, al = pooled_ridge(Xtr, Ytr, Xte)
    m = metrics(Yte, P, Vte, include_anomaly=False)
    obs, lo, hi, p, _ = block_bootstrap_skill(Yte, P, Vte, H, B=B, seed=RNG_SEED)
    return dict(alpha=al, n_win=m['n_win'],
                零变化_RMSE=m['零变化_RMSE'], 恒速_RMSE=m['恒速_RMSE'], 岭回归_RMSE=m['岭回归_RMSE'],
                skill_vs零变化=m['skill_vs零变化'], skill_vs恒速=m['skill_vs恒速'],
                skill_vs最优平凡=m['skill_vs最优平凡'], corr=m['corr'],
                bs_skill=obs, bs_lo=lo, bs_hi=hi, p_skill_le0=p)


def state_metrics(dif_win):
    dd = dif_win.reshape(-1).astype(float)
    med = float(np.median(np.abs(dd))) if len(dd) else 0.0
    sig = float(np.std(dd))
    t = np.arange(len(dd), dtype=float)
    if len(dd) >= 3 and sig > 1e-12:
        c, _, _, _ = np.linalg.lstsq(np.c_[t, np.ones(len(t))], dd, rcond=None)
        ar1 = float(np.corrcoef(dd[:-1], dd[1:])[0, 1]) if len(dd) >= 2 else np.nan
        resid_sd = float(np.std(dd - (c[0] * t + c[1])))
        snrt = float(abs(c[0]) / max(resid_sd, 1e-12))
    else:
        ar1, snrt = np.nan, 0.0
    return dict(S_rate=med, S_act=sig, S_ar1=ar1, S_snrt=snrt)


# ------------------------------------------------------------------ Tan 逐管
def tan_common_grid(curves):
    """15 管插值到公共整数日格点，返回 (days, D: n_days×N, curves)。"""
    df = pd.read_csv(TAN_CSV)
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


def pooled_per_unit(D, dif, a_tr, b_tr, a_te, b_te, H, B=500):
    """池化训岭回归（所有单元共用），逐单元测试段归因评估 → 稳定的逐单元 skill。

    与 §4.5 探针同估算器结构（池化平滑），避免单单元孤立拟合产生的退化（
    单单元仅约 20 窗/24 特征，易过拟合且对窗内非平稳敏感）。块 bootstrap CI 逐单元。"""
    wtr = win(D, dif, a_tr, b_tr, L, H)
    wte = win(D, dif, a_te, b_te, L, H)
    if wtr is None or wte is None:
        return [], None
    Xtr, Ytr, _ = wtr
    Xte, Yte, Vte = wte
    P, al = pooled_ridge(Xtr, Ytr, Xte)   # (nw_te, N)
    N = D.shape[1]
    out = []
    for j in range(N):
        r0 = float(np.sqrt(np.mean(Yte[:, j] ** 2)))
        if r0 < 1e-12:
            continue
        Yj = Yte[:, [j]]; Pj = P[:, [j]]; Vj = Vte[:, [j]]
        rv = float(np.sqrt(np.mean((Yj - Vj) ** 2)))
        rr = float(np.sqrt(np.mean((Yj - Pj) ** 2)))
        skill_zero = 1 - (rr / r0) ** 2
        skill_best = 1 - (rr / min(r0, rv)) ** 2
        obs, lo, hi, p, _ = block_bootstrap_skill(Yj, Pj, Vj, H, B=B, seed=RNG_SEED)
        out.append(dict(alpha=al, n_win=int(Yj.shape[0]),
                        零变化_RMSE=r0, 恒速_RMSE=rv, 岭回归_RMSE=rr,
                        skill_vs零变化=skill_zero, skill_vs恒速=1 - (rr / rv) ** 2,
                        skill_vs最优平凡=skill_best,
                        corr=float(np.corrcoef(Pj.reshape(-1), Yj.reshape(-1))[0, 1]),
                        bs_skill=obs, bs_lo=lo, bs_hi=hi, p_skill_le0=p))
    return out, al


def tan_rows():
    rows = []
    days, D, curves = tan_common_grid(TAN_CURVES)
    den, dif = D, np.zeros_like(D)
    dif[1:] = np.diff(D, axis=0)
    for phase, lo, hi in [('开挖期_全', days[0], 294.9), ('结构期_全', 300.0, days[-1] + 0.5)]:
        a, b = idx_for_day(days, lo, hi)
        if b - a < L + H_LIST[0] + 1:
            continue
        split = a + (b - a) // 2
        st_cols = []
        for j in range(D.shape[1]):
            st = state_metrics(dif[a + 1:b, j])
            st_cols.append(dict(S_rate_d=st['S_rate'], S_act_d=st['S_act'],
                                S_ar1=st['S_ar1'], S_snrt=st['S_snrt']))
        for hz in H_LIST:
            per, _ = pooled_per_unit(den, dif, a, split, split, b, hz, B=500)
            for j, r in enumerate(per):
                rows.append(dict(场地='Tan', 序列=curves[j], 阶段=phase, H=hz,
                                 S_rate_mm_day=st_cols[j]['S_rate_d'],
                                 S_act_mm_day=st_cols[j]['S_act_d'],
                                 S_ar1=st_cols[j]['S_ar1'], S_snrt=st_cols[j]['S_snrt'], **r))
    return rows


# ------------------------------------------------------------------ 柳沟逐序列
def liugou_rows():
    df = pd.read_csv(LIOU_CSV)
    D = df[LIOU_COLS].to_numpy(float)[:LIOU_N]
    dif = np.zeros_like(D)
    dif[1:] = np.diff(D, axis=0)
    rows = []
    for phase, a, b in LIOU_PHASES:
        if b - a < L + H_LIST[0] + 1:
            continue
        split = a + (b - a) // 2
        st_cols = []
        for j in range(D.shape[1]):
            st8 = state_metrics(dif[a + 1:b, j])
            st_cols.append(dict(S_rate_d=3.0 * st8['S_rate'], S_act_d=3.0 * st8['S_act'],
                                S_ar1=st8['S_ar1'], S_snrt=st8['S_snrt']))
        for hz in H_LIST:
            per, _ = pooled_per_unit(D, dif, a, split, split, b, hz, B=1000)
            for j, r in enumerate(per):
                rows.append(dict(场地='Liugou', 序列=LIOU_COLS[j], 阶段=phase, H=hz,
                                 S_rate_mm_day=st_cols[j]['S_rate_d'],
                                 S_act_mm_day=st_cols[j]['S_act_d'],
                                 S_ar1=st_cols[j]['S_ar1'], S_snrt=st_cols[j]['S_snrt'], **r))
    return rows


# ------------------------------------------------------------------ 回归（秩统计；Spearman ρ + 秩 bootstrap 95% CI）
def regress(x, y, seed=RNG_SEED):
    """S 与 skill 的秩相关。剔除非有限值；返回 Spearman ρ、p、秩 bootstrap 95% CI。

    不用 log10 最小二乘斜率：S_act 双峰（基坑开挖期大/静止期小）且单单元 skill
    近坍塌，OLS 斜率达到 ~10^5、完全不可解释，故主判据只用秩相关。"""
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    ok = np.isfinite(x) & np.isfinite(y)
    x, y = x[ok], y[ok]
    n = len(x)
    if n < 4:
        return dict(n=n, spearman_rho=np.nan, spearman_p=np.nan, rho_lo=np.nan, rho_hi=np.nan)
    rho, p = st.spearmanr(x, y)
    # 秩 bootstrap 95% CI
    rng = np.random.RandomState(seed)
    B = 10000
    rbs = np.empty(B)
    for i in range(B):
        idx = rng.randint(0, n, size=n)
        rbs[i] = st.spearmanr(x[idx], y[idx]).statistic if hasattr(st.spearmanr(x[idx], y[idx]), 'statistic') else st.spearmanr(x[idx], y[idx])[0]
    lo, hi = np.percentile(rbs, [2.5, 97.5])
    return dict(n=n, spearman_rho=rho, spearman_p=p, rho_lo=lo, rho_hi=hi)


def spearman_power(n, rho, alpha=0.05):
    """Fisher z 近似，单侧检验功效。"""
    z = 0.5 * math.log((1 + rho) / max(1 - rho, 1e-12))
    se = 1.0 / math.sqrt(n - 3)
    zcrit = st.norm.ppf(1 - alpha)
    return float(st.norm.sf(zcrit - z / se))


def detection_floor(n, alpha=0.05):
    """n、单侧 α 下可检出的最小 |ρ|（Fisher z 近似）。"""
    zcrit = st.norm.ppf(1 - alpha)
    return float(math.tanh(zcrit / math.sqrt(n - 3)))


def main():
    trows = tan_rows()
    lrows = liugou_rows()
    rows = trows + lrows
    df = pd.DataFrame(rows)
    df.to_csv(OUT_TABLES / 'T_p19_skill_states.csv', index=False, encoding='utf-8-sig')
    print(f'逐管/逐序列明细 {len(df)} 行 -> T_p19_skill_states.csv')

    H10 = df[df['H'] == 10]

    # ---- 状态指标 X = log10(S)。活动幅度用 S_act（σΔ，mm/day）；平滑度用 S_ar1 ----
    def x10(sub, col):
        return sub[col]

    reg_rows = []

    # ① 场地内状态对比（单位观测单元的"状态依赖"直接检验，与 §4.5 主张同轴）
    tan_all = H10[H10['场地'] == 'Tan']              # 开挖期 + 结构期 → 15×2 单元
    liou_all = H10[H10['场地'] == 'Liugou']          # 全时段 + 前/后半 → 8×3 单元
    reg_rows.append(dict(集合='Tan_场地内_act', S定义='S_act (mm/day)', S列='S_act_mm_day',
                         **regress(x10(tan_all, 'S_act_mm_day'), tan_all['skill_vs零变化'])))
    reg_rows.append(dict(集合='Liugou_场地内_act', S定义='S_act (mm/day)', S列='S_act_mm_day',
                         **regress(x10(liou_all, 'S_act_mm_day'), liou_all['skill_vs零变化'])))

    # ② 跨场地（活动子时段）——活动幅度是否跨场地单调
    tan_ex = H10[(H10['场地'] == 'Tan') & (H10['阶段'] == '开挖期_全')]
    liou_full = H10[(H10['场地'] == 'Liugou') & (H10['阶段'] == '全时段')]
    merged_ex = pd.concat([tan_ex, liou_full], ignore_index=True)
    reg_rows.append(dict(集合='跨场地_开挖vs全时段_act', S定义='S_act (mm/day)', S列='S_act_mm_day',
                         **regress(merged_ex['S_act_mm_day'], merged_ex['skill_vs零变化'])))

    # ③ 跨场地 + 场地内合并（全部 5 阶段，作整体关系）——幅度与平滑度对比
    reg_rows.append(dict(集合='全场_all_act', S定义='S_act (mm/day)', S列='S_act_mm_day',
                         **regress(H10['S_act_mm_day'], H10['skill_vs零变化'])))
    reg_rows.append(dict(集合='全场_all_ar1', S定义='S_ar1 (Δ 平滑度)', S列='S_ar1',
                         **regress(np.maximum(H10['S_ar1'], 1e-4), H10['skill_vs零变化'])))

    # 稳健：S_rate 作 S（剔除阶梯退化 0 值）
    pos = H10[H10['S_rate_mm_day'] > 0]
    if len(pos) >= 8:
        reg_rows.append(dict(集合='全场_all_rate_pos', S定义='S_rate median|Δ| (mm/day)', S列='S_rate_mm_day',
                             **regress(pos['S_rate_mm_day'], pos['skill_vs零变化'])))

    reg = pd.DataFrame(reg_rows)
    # 功效否证参考（全单元样本 n；Fisher z 近似单侧功效）
    n_all = len(H10)
    floor_all = detection_floor(n_all)
    reg['可检下限rho'] = floor_all
    reg['power_rho0.5'] = spearman_power(n_all, 0.5)
    reg['power_rho0.7'] = spearman_power(n_all, 0.7)
    reg.to_csv(OUT_TABLES / 'T_p19_skill_regression.csv', index=False, encoding='utf-8-sig')
    print('回归汇总 -> T_p19_skill_regression.csv')

    # 判定：方向敏感的否证逻辑（主判据 = Spearman ρ 的秩 bootstrap 95% CI）
    print('\n' + '=' * 100)
    for _, r in reg.iterrows():
        ci_excl0 = not (r['rho_lo'] <= 0 <= r['rho_hi'])
        if not ci_excl0 or abs(r['spearman_rho']) < r['可检下限rho']:
            verdict = '无证据（ρ 的 95%CI 含 0 或未达可检下限）'
        elif r['spearman_rho'] > 0:
            verdict = '获支持（S 越大 skill 越高）'
        else:
            verdict = '被否证（S 越大 skill 反而越低）'
        print(f"  [{r['集合']:<26}]  S列={r['S列']:<18} n={int(r['n']):>3}  "
              f"ρ={r['spearman_rho']:+.3f}(p={r['spearman_p']:.3f})  "
              f"ρ95%CI[{r['rho_lo']:+.3f},{r['rho_hi']:+.3f}]  → {verdict}")

    # 场地内逐单元 skill 分布摘要（§4.5 状态依赖的单元粒度事实）
    print('\n逐单元 skill（H=10，池化训练-逐单元归因）分布：')
    for site in ['Tan', 'Liugou']:
        s = H10[H10['场地'] == site]
        for ph in s['阶段'].unique():
            v = s[s['阶段'] == ph]['skill_vs零变化'].values
            print(f'  [{site:<6}] {ph:<10} n={len(v):>2}  '
                  f'中位={np.median(v):+.3f}  IQR=[{np.percentile(v,25):+.3f},{np.percentile(v,75):+.3f}]  '
                  f'范围[{np.min(v):+.3f},{np.max(v):+.3f}]')

    # 数字化不确定度
    digit_rows = digitization_uncertainty()
    pd.DataFrame(digit_rows).to_csv(OUT_TABLES / 'T_p19_digitization_unc.csv',
                                    index=False, encoding='utf-8-sig')
    print(f'数字化不确定度 {len(digit_rows)} 行 -> T_p19_digitization_unc.csv')

    # 图（先产出待审）
    plot_skill_regression(H10)

    print('\n完成（零深度训练）。图先产出待作者审核，勿自动插入正文。')


# ------------------------------------------------------------------ 数字化不确定度
def digitization_uncertainty():
    """两来源：(a) S7 登记的坐标标定残差（分辨率误差界）；(b) 格点化协议敏感性
    （median vs mean vs max 三种日聚合 → 逐曲线两两残差中位/p95），作"提取配置
    不确定性"的下界。真正的独立人工多判读者复读为局限（补材 S28 声明）。"""
    rows = [
        dict(项目='x标定_Fig19', 值='0.37004 pt/天', 残差='≤0.163 pt', 误差界='±0.44 天',
             说明='21 日期刻度回归；文本中心交叉验证斜率差 0.04%'),
        dict(项目='y标定_Fig19', 值='-1.0745 pt/mm', 残差='≤0.017 pt', 误差界='±0.016 mm',
             说明='0–100 mm 标签回归'),
        dict(项目='x标定_Fig7', 值='0.3235 pt/天', 残差='≤0.016 pt', 误差界='±0.05 天',
             说明='日期标签回归'),
        dict(项目='y标定_Fig7', 值='7.95/6.47 pt/m', 残差='未单列', 误差界='≈±0.02 m',
             说明='面板 a/b 分离'),
        dict(项目='x标定_Fig18a', 值='0.33209 pt/天', 残差='≤0.019 pt', 误差界='±0.06 天',
             说明='交叉验证斜率差 0.09%'),
        dict(项目='y标定_Fig18a', 值='-0.62524 pt/mm', 残差='≤0.017 pt', 误差界='±0.027 mm',
             说明='+20~-140 标签回归'),
        dict(项目='同色对墨迹分配', 值='1–2 mm', 残差='—', 误差界='±1–2 mm',
             说明='早期 0–10 天密集纠缠区；Q11 点划线归属靠轨迹距离，平台段差 <1.5 mm'),
        dict(项目='端值锚定', 值='Q1=53.3 ~ Q10/Q11≈91.5 mm', 残差='—', 误差界='与纸面"53–92 mm"一致',
             说明='角部最小/最大值与论文一致'),
    ]
    # (b) 格点化协议敏感性：对 15 管逐曲线，median/mean/max 三种日聚合的两两残差分布
    df = pd.read_csv(TAN_CSV)
    df['day'] = np.round(df['day_since_exc_2008-03-24']).astype(int)
    diffs = []
    for c in TAN_CURVES:
        g = df[df.curve == c].groupby('day')['lateral_disp_mm']
        med = g.median(); mean = g.mean(); mx = g.max()
        idx = med.index.intersection(mean.index).intersection(mx.index)
        a, b2, c3 = med.loc[idx].values, mean.loc[idx].values, mx.loc[idx].values
        diffs.extend(np.abs(a - b2))
        diffs.extend(np.abs(a - c3))
        diffs.extend(np.abs(b2 - c3))
    d = np.array(diffs)
    rows.append(dict(项目='格点化协议敏感性', 值='median/mean/max', 残差=f'中位 {np.median(d):.3f} mm',
                     误差界=f'p95 {np.percentile(d, 95):.3f} mm', 说明=f'15 管全部日点 n={len(d)}'))
    rows.append(dict(项目='多判读者复读', 值='未做独立人工复读', 残差='—', 误差界='局限（补材 S28 声明）',
                     说明='本项目为单判读者自动化提取 + 双源标定交叉验证 + 目检 QA；'
                          '以提取配置敏感性与标定残差复合近似误差界'))
    return rows


# ------------------------------------------------------------------ 图（两联：S_act 与 S_ar1）
def plot_skill_regression(df10):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({
        'font.family': 'serif', 'font.serif': ['Times New Roman'],
        'font.size': 10, 'axes.labelsize': 10, 'axes.titlesize': 11,
        'xtick.labelsize': 9, 'ytick.labelsize': 9, 'legend.fontsize': 8.5,
    })
    tan = df10[df10['场地'] == 'Tan']
    liou = df10[df10['场地'] == 'Liugou']
    # 场地/阶段着色
    colors = {'Tan_开挖期': '#d62728', 'Tan_结构期': '#f5a0a0',
              'Liugou_全时段': '#1f77b4', 'Liugou_前半年': '#6baed6', 'Liugou_后半年': '#9ecae1'}
    labels = {'Tan_开挖期': 'Tan excavation', 'Tan_结构期': 'Tan post-excavation',
              'Liugou_全时段': 'Liugou full span', 'Liugou_前半年': 'Liugou first half',
              'Liugou_后半年': 'Liugou second half'}
    groups = [('Tan_开挖期', tan[tan['阶段'] == '开挖期_全']), ('Tan_结构期', tan[tan['阶段'] == '结构期_全']),
              ('Liugou_全时段', liou[liou['阶段'] == '全时段']), ('Liugou_前半年', liou[liou['阶段'] == '前半年']),
              ('Liugou_后半年', liou[liou['阶段'] == '后半年'])]

    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.3), dpi=100)
    ax0 = axes[0]
    # 左：activity amplitude（对数）。逐单元 bootstrap CI 因退化单元而过度偏斜、不可靠，
    # 故图仅画散点（CI 的不稳定本身是逐单元口径的局限，写入补材 S28）。
    def _clip(sub):
        return np.clip(sub['skill_vs零变化'].values, -1.2, 1.2)

    for name, sub in groups:
        if sub.empty:
            continue
        y = _clip(sub)
        x = np.clip(sub['S_act_mm_day'].values, 1e-2, 20)   # 显示下限裁到 0.01 mm/day（近零静止幅度单元格不再压垮对数轴）
        ax0.scatter(x, y, s=34, color=colors[name],
                    label=labels[name], alpha=0.85, edgecolors='none')
    ax0.set_xscale('log')
    ax0.axhline(0, color='0.4', lw=0.7, ls=':')
    ax0.set_xlabel('S = \u03c3(\u0394y) activity amplitude (mm/day, log)')
    ax0.set_ylabel('skill vs zero-change baseline (H=10)')
    ax0.set_title('(a) Amplitude-based state indicator')
    ax0.legend(loc='lower right', framealpha=0.9, fontsize=8)
    # 右：smoothness（增量 lag-1 自相关）
    ax1 = axes[1]
    for name, sub in groups:
        if sub.empty:
            continue
        y = _clip(sub)
        ax1.scatter(sub['S_ar1'].values, y, s=34, color=colors[name],
                    label=labels[name], alpha=0.85, edgecolors='none')
    ax1.axhline(0, color='0.4', lw=0.7, ls=':')
    ax1.set_xlabel('S = lag-1 autocorrelation of \u0394y (smoothness)')
    ax1.set_ylabel('skill vs zero-change baseline (H=10)')
    ax1.set_title('(b) Smoothness-based state indicator')
    ax1.legend(loc='lower right', framealpha=0.9, fontsize=8)
    fig.tight_layout()
    fig.savefig(OUT_FIGS / 'Fig_Sskill_regression.pdf')
    fig.savefig(OUT_FIGS / 'Fig_Sskill_regression.png', dpi=600)
    plt.close(fig)
    print('图 -> _p09_figs/Fig_Sskill_regression.{pdf,png}（待作者审核）')


if __name__ == '__main__':
    main()
