# -*- coding: utf-8 -*-
"""_loocv_writeup_analysis.py -- 为 P4（LOOCV 写入正文）计算可直接引用的统计量（只读）

产出：
 1) 折内配对检验：每个折模型上"留出断面"与"可见四断面"的末步 RMSE 差
    （逐种子配对 → 每折 n=5，再跨折池化 n=25），给出均值差、%、p。
 2) 跨视界汇总：留出 vs 可见的 RMSE、R²、ΔR²。
 3) 掩码效应对照摘要（Q1 全局 / Q2 留出断面，对比未掩码 V7）：均值、为正折数、显著折数。
 4) 断面难度控制：corr(该断面未掩码 RMSE, LOOCV gap)。
 5) "输出是否断面不可分"诊断：留出断面预测与真值的空间形态相关，vs 可见断面，
    用于判断"gap≈0"是空间泛化造成还是模型近似忽略断面差异造成。
用法：
  python -u _loocv_writeup_analysis.py
"""
from _env import AUX_ROOT, WORK_ROOT

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

ROOT = WORK_ROOT
LOO = ROOT / 'Spatial_LOOCV_Results'
ABL = AUX_ROOT / "Ablation_New_5d"
SECTIONS = ['CX2', 'CX3', 'CX8', 'CX13', 'CX15']
SEEDS5 = [42, 15, 77, 85, 120]
HZ = [2, 6, 8, 10, 12]
VAR = 'V7_wo_VerticalGCN'
OUT = ROOT / '_paper_tables'
OUT.mkdir(exist_ok=True)


def section_mse_last(pred_cum, true_cum, blocks):
    return {s: float(np.mean((pred_cum[:, -1, idx] - true_cum[:, -1, idx]) ** 2))
            for s, idx in blocks.items()}


def node_blocks(nodes):
    b = {}
    for j, n in enumerate(nodes):
        b.setdefault(str(n).split('_')[0], []).append(j)
    return b


# ---------------- 1/2. 折内配对检验 ----------------
rows_fold, rows_pairs = [], []
for hz in HZ:
    nodes = pd.read_excel(ABL / f'H{hz}' / VAR / f'run_{SEEDS5[0]}' / 'node_metrics.xlsx')['Node'] \
        .astype(str).tolist()
    blocks = node_blocks(nodes)
    for fold in SECTIONS:
        d = LOO / f'Fold_{fold}' / f'H{hz}'
        preds = [np.load(d / f'run_{s}' / 'predictions.npz') for s in SEEDS5]
        held, vis = [], []
        for z in preds:
            m = section_mse_last(z['pred_cum'], z['true_cum'], blocks)
            held.append(m[fold])
            vis.append(np.mean([v for k, v in m.items() if k != fold]))
        held, vis = np.array(held), np.array(vis)
        h_r = np.sqrt(held)          # 每次训练独立取 RMSE（种子级配对）
        v_r = np.sqrt(vis)
        t = stats.ttest_rel(h_r, v_r)
        for s, a, b in zip(SEEDS5, h_r, v_r):
            rows_pairs.append(dict(H=hz, Fold=fold, seed=s,
                                   heldout_rmse=a, visible_rmse=b, diff=b - a))
        rows_fold.append(dict(H=hz, Fold=fold,
                              held_rmse=float(np.sqrt(np.mean(held))),
                              vis_rmse=float(np.sqrt(np.mean(vis))),
                              mean_seed_gap_pct=float(np.mean((h_r - v_r) / v_r * 100)),
                              t=t.statistic, p=t.pvalue, n_seed=len(SEEDS5)))

fdf = pd.DataFrame(rows_fold)
pdf = pd.DataFrame(rows_pairs)

print('=' * 104)
print('1) 折内配对检验：留出断面 vs 可见四断面（同一折模型内，逐种子配对；负 = 留出断面更好）')
print('=' * 104)
hdr = f'{"H":>3}{"折":>7}{"留出RMSE":>10}{"可见RMSE":>10}{"逐种子gap%":>13}{"t":>8}{"p":>9}'
print(hdr); print('-' * len(hdr))
for _, r in fdf.iterrows():
    print(f'{int(r.H):>3}{r.Fold:>7}{r.held_rmse:>10.4f}{r.vis_rmse:>10.4f}'
          f'{r.mean_seed_gap_pct:>+13.2f}{r.t:>8.2f}{r.p:>9.3f}')

print('\n跨折池化（每折先算均值，n=5）：')
for hz in HZ:
    s = fdf[fdf.H == hz]
    t = stats.ttest_rel(s.vis_rmse, s.held_rmse)      # 方向：可见 − 留出
    print(f'  H={hz:<3} 留出 {s.held_rmse.mean():.4f}  可见 {s.vis_rmse.mean():.4f}  '
          f'gap {((s.held_rmse.mean() - s.vis_rmse.mean()) / s.vis_rmse.mean() * 100):+.2f}%  '
          f't={t.statistic:+.2f} p={t.pvalue:.3f}（n=5 折）')
tt = stats.ttest_rel(pdf.visible_rmse, pdf.heldout_rmse)
print(f'  全部 25 个 (折×种子) 池化：可见 {pdf.visible_rmse.mean():.4f}  '
      f'留出 {pdf.heldout_rmse.mean():.4f}  t={tt.statistic:+.2f} p={tt.pvalue:.3f}')
print(f'  留出更好的比例 = {int((pdf.heldout_rmse < pdf.visible_rmse).sum())}/{len(pdf)}')

# ---------------- 3. 掩码效应摘要 ----------------
print('\n' + '=' * 104)
print('2) 掩码效应对照（折模型 vs 未掩码 V7，同 5 种子）：正 = 掩码后更差')
print('=' * 104)
mk = pd.read_csv(LOO / '_mask_effect_check.csv')
print(f"  Q1 全局劣化%：均值 {mk.q1_global_pct.mean():+.2f}  为正 {int((mk.q1_global_pct > 0).sum())}/{len(mk)}  "
      f"显著(p<0.05) {int((mk.p_global < 0.05).sum())}/{len(mk)}")
print(f"  Q2 留出断面劣化%：均值 {mk.q2_held_pct.mean():+.2f}  为正 {int((mk.q2_held_pct > 0).sum())}/{len(mk)}  "
      f"显著(p<0.05) {int((mk.p_held < 0.05).sum())}/{len(mk)}")
gp = pd.read_csv(LOO / '_section_gap_control.csv')
print('\n  断面难度控制：corr(该断面未掩码 RMSE, LOOCV gap) = '
      + ', '.join(f'H{int(r.H)} {r.corr_difficulty_vs_loocv_gap:+.3f}' for _, r in gp.iterrows()))

# ---------------- 4. 输出是否"断面不可分" ----------------
print('\n' + '=' * 104)
print('3) 诊断：模型在留出断面上是否仍给出与该断面真值**增量**相关的预测')
print('=' * 104)
print('   为什么用增量而非累计：累计量被窗口起点（=零变化预测）主导，任何预测都会得到高相关，')
print('   该量对"模型是否携带可迁移信息"无判别力。故改看增量场：')
print('   r_inc = corr( pred_cum[:,-1]−pred_cum[:,0] , true_cum[:,-1]−true_cum[:,0] )，'
      '跨 (窗口×节点) 池化。')


def _corr(a, b):
    a = np.asarray(a, float).ravel()
    b = np.asarray(b, float).ravel()
    if a.std() < 1e-12 or b.std() < 1e-12:
        return np.nan
    return float(np.corrcoef(a, b)[0, 1])


rows = []
for hz in HZ:
    nodes = pd.read_excel(ABL / f'H{hz}' / VAR / f'run_{SEEDS5[0]}' / 'node_metrics.xlsx')['Node'] \
        .astype(str).tolist()
    blocks = node_blocks(nodes)
    for fold in SECTIONS:
        d = LOO / f'Fold_{fold}' / f'H{hz}'
        hs, vs = [], []
        rmse_m, rmse_z = [], []
        for s in SEEDS5:
            z = np.load(d / f'run_{s}' / 'predictions.npz')
            for sec, idx in blocks.items():
                pd_inc = z['pred_cum'][:, -1, idx] - z['pred_cum'][:, 0, idx]
                td_inc = z['true_cum'][:, -1, idx] - z['true_cum'][:, 0, idx]
                (hs if sec == fold else vs).append(_corr(pd_inc, td_inc))
                if sec == fold:
                    rmse_m.append(float(np.sqrt(np.mean(
                        (z['pred_cum'][:, -1, idx] - z['true_cum'][:, -1, idx]) ** 2))))
                    rmse_z.append(float(np.sqrt(np.mean(
                        (z['true_cum'][:, 0, idx] - z['true_cum'][:, -1, idx]) ** 2))))
        rows.append(dict(H=hz, Fold=fold,
                         held_inc_r=float(np.nanmean(hs)),
                         visible_inc_r=float(np.nanmean(vs)),
                         rmse_model_held=float(np.mean(rmse_m)),
                         rmse_zeroc_held=float(np.mean(rmse_z))))
p3 = pd.DataFrame(rows)
for hz in HZ:
    s = p3[p3.H == hz]
    print(f'  H={hz:<3} 留出断面增量r {s.held_inc_r.mean():+.3f} | 可见断面 '
          f'{s.visible_inc_r.mean():+.3f}')
t3 = stats.ttest_rel(p3.visible_inc_r.fillna(0), p3.held_inc_r.fillna(0))
print(f'  跨 25 折：留出 {p3.held_inc_r.mean():+.3f} vs 可见 {p3.visible_inc_r.mean():+.3f} '
      f'(t={t3.statistic:+.2f}, p={t3.pvalue:.3f})')
print(f'  留出断面末步 RMSE：折模型 {p3.rmse_model_held.mean():.4f} vs 零变化 '
      f'{p3.rmse_zeroc_held.mean():.4f}（模型落后 '
      f'{(p3.rmse_model_held.mean() / p3.rmse_zeroc_held.mean() - 1) * 100:+.1f}%）')

fdf.to_csv(OUT / '_loocv_fold_paired.csv', index=False, encoding='utf-8-sig')
p3.to_csv(OUT / '_loocv_pattern_corr.csv', index=False, encoding='utf-8-sig')
print(f'\n已写出 {OUT}\\_loocv_fold_paired.csv 与 _loocv_pattern_corr.csv')