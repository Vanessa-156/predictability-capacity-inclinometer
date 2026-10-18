# =============================================================================
# Stress_Test_Experiment.py -- Chapter 4 Stress Testing for MST Journal
# =============================================================================
# Design doc §E:
#   4 models (Ours/V7/TCN-Only/Best-Baseline, the latter auto-detected from the
#             comparison experiment via find_best_baseline)
#   x 5 Horizons (H=2,6,8,10,12)
#   x 3 stress conditions (Noise / Missing / Sparse)
#   x 11 perturbed levels (4 noise + 4 missing + 3 sparse)
#   x 30 seeds = ~1,320 self-trained + ~2,640 perturbed runs (estimate)
#
#   Clean baselines (noise_0.0 / missing_0.0 / sparse_1) are SELF-TRAINED in
#   this script with the exact same architecture as the perturbed runs
#   (no longer reloaded from the ablation results), so that degradation
#   rates compare the SAME model under clean vs perturbed data.
#   Since the three clean conditions produce identical training data, the
#   model is trained once (noise_0.0) and the results are copied to the
#   missing_0.0 / sparse_1 directories.
#
# Architecture note (unified with Ablation_Experiment.py):
#   - Input features exclude the static geology channels (F_in = 4,
#     use_geo=False), geology enters via adjacency / physics gate /
#     cross-node attention only.
#   - 路径 A（2026-09-17）：水文水位不再进入任何模型。VICD 诊断（正文 §3.4）
#     表明水位与位移在增量口径无可用信息，故输入特征由 12 维降为 4 维。
#     water_feat 仅按占位透传以保持 DataLoader / forward 签名不变。
#   - 压力测试的模型集合不再硬编码：全模块版(V5) 恒定参与，对照模型在对比实验
#     跑完后按 H=10 的 RMSE 自动选出（最强外部基线），见 find_best_baseline。
#
# Core metric: DEFENSE = δ_V7 − δ_V5（各模型以自身干净 RMSE 为基准）
#   注意：实现口径与上方旧描述不同。自 §5.2 消融后论文主模型为 V7，
#   故 DEFENSE 度量的是"竖向 GCN 的鲁棒性贡献"（V5 含、V7 不含），
#   而**不是**"我们的模型 vs 最强外部基线"。实测该量在 30 种子下全部
#   落在 ±4.6% 内、多重比较后无稳健显著项，即竖向 GCN 对鲁棒性亦无可测
#   贡献（与 §5.2 精度结论一致）。论文 §4.3 的主线已改为"扰动下模型排序
#   是否保持"（V7 vs V0 / V7 vs 外部基线），DEFENSE 仅作脚注级证据。
#
# SCI visualization（已迁出，2026-09-18）：
#   本脚本只做「数据聚合 + 数据落盘」（all_stress_results.csv /
#   full_seed_data.pkl / cross_horizon_all.csv）。出图见
#   可视化/02代码结果可视化/fig_stress.py，从落盘数据读取后可反复重绘：
#   - 3-panel degradation curves (noise/missing/sparse, H=6)
#   - DEFENSE vs Horizon (3 lines for 3 conditions, core highlight)
#   - DEFENSE bar chart across conditions and horizons
#   - Degradation heatmap (Model x ConditionLevel)
# =============================================================================
# 写盘重试层（须早于 numpy/pandas/torch 导入，见 _robust_io.py）

from _env import AUX_ROOT, DATA_ROOT
import _robust_io  # noqa: F401
import os, json, logging, warnings, time, copy, gc, sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import seaborn as sns
from scipy.spatial.distance import cdist
from scipy.interpolate import interp1d
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, max_error
import pywt
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import random

# Model factory / training helpers from the comparison experiment, so that the
# auto-detected Best-Baseline is trained EXACTLY as in Comparison_Experiment.py
# (same architecture, hyperparameters, optimizer and loss -- issue 3, Plan A).
from Comparison_Experiment import (Config as ComparisonConfig,
                                   get_model as cmp_get_model,
                                   train_model as cmp_train_model,
                                   evaluate as cmp_evaluate,
                                   MODEL_TYPES as CMP_MODEL_TYPES,
                                   OURS_FRAMEWORK_MODELS as CMP_OURS_FRAMEWORK_MODELS)

warnings.filterwarnings('ignore')
logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)


# =============================================================================
# CONFIGURATION
# =============================================================================
class Config:
    BASE_DATA_DIR = DATA_ROOT / "ceshi14/MoXin/data"
    WATER_DIR = DATA_ROOT / "ceshi14/shuiwei/时间统一+插值"
    GEO_PATH = DATA_ROOT / "ceshi14/date/raw/地质参数表.xlsx"
    # 2026-09-21：原指向 `Ablation_New`（旧 12 维含水架构，已作废）——与
    # `Comparison_Experiment.ABLATION_ROOT` 同类陷阱，此处一并修正为当前 4 维架构目录。
    # 注：本常量目前**未被任何代码引用**（V5/V7/V0 均由本脚本按 STRESS_MODELS 自行训练，
    # 见 L1210 起），保留仅为与 Comparison/PCA 的配置命名对齐，避免将来误用旧目录。
    ABLATION_ROOT = AUX_ROOT / "Ablation_New_5d"
    COMPARISON_ROOT = AUX_ROOT / "Comparison_Results"
    OUT_ROOT = AUX_ROOT / "Stress_Results"
    OUT_ROOT.mkdir(parents=True, exist_ok=True)

    POINTS = ['CX2', 'CX3', 'CX8', 'CX13', 'CX15']
    DEPTHS = np.arange(0.5, 16.0, 0.5)
    N_POINTS = len(POINTS)
    N_DEP = len(DEPTHS)
    N_NODES = N_POINTS * N_DEP
    LOOKBACK = 12
    HORIZONS = [2, 6, 8, 10, 12]
    TRAIN_RATIO = 0.70
    VAL_RATIO = 0.15

    BATCH_SIZE = 32
    # 论文口径 400/40；环境变量仅用于示例数据上的缩小规模演示，默认行为不变。
    EPOCHS = int(os.environ.get('ABLATION_EPOCHS', '400'))
    PATIENCE = int(os.environ.get('ABLATION_PATIENCE', '40'))
    LR = 0.001
    LR_GCN = 0.0005
    WD = 1e-3
    GRAD_CLIP = 1.0

    TCN_CHANNELS = [64, 64]
    TCN_KERNEL_SIZE = 7
    TCN_DILATIONS = [1, 2]
    TCN_DROPOUT = 0.3
    GCN_HIDDEN = 64
    ATTENTION_HIDDEN = 32
    DROPOUT = 0.5

    GEO_FEAT_DIM = 6
    # 式(11) 邻接矩阵合成中的两处结构性先验开关（默认 True，数值行为与历史结果等价）。
    SPA_WITHIN_HOLE_PRIOR = True   # 同孔相邻深度边权重 0.8（结构连续性）
    SPA_SHALLOW_PRIOR = True       # 浅层(<3 m)跨断面节点对 0.6，并豁免 3σ 剪枝

    WAVELET = 'db4'
    WAVELET_LEVEL = 1
    THRESH_MODE = 'soft'

    # 30 个种子：与 Ablation/Comparison/PCA 完全一致，见 _gen_seeds.py
    SEEDS = [42, 15, 77, 85, 120, 532, 901, 23, 12, 14, 19, 75, 250, 140, 135,
             1180, 1262, 1284, 605, 1159, 746, 1657, 61, 27, 993, 897, 276,
             1308, 896, 1866]
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # 2026-09-20：`AMP`（自动混合精度）与 `FULL_VIS` 均无消费者，已删除。
    # 2026-09-22：断点续跑，与 Comparison_Experiment 的 COMPARISON_RESUME 同款——
    # 置 1 时跳过已完整产出四件套（metrics.json/best.pth/predictions.npz/losses.npy）
    # 的种子，并读回其 metrics.json 参与汇总。
    # 本脚本规模最大（12 条件 × 4 模型 × 5 视界 × 30 种子 = 7200 runs，估 6~12 h），
    # 无续跑时一旦中断即需从该条件重跑全部 30 个种子。
    RESUME = os.environ.get('STRESS_RESUME', '0') == '1'


# =============================================================================
# STRESS CONDITIONS
# =============================================================================
STRESS_NOISE_LEVELS = [0.0, 0.05, 0.10, 0.20, 0.30]
STRESS_MISSING_RATES = [0.0, 0.1, 0.2, 0.3, 0.5]
STRESS_SPARSE_EVERY = [1, 2, 3, 5]

STRESS_LABELS = {
    'noise_0.0': 'Clean',
    'noise_0.05': 'Noise 5%',
    'noise_0.1': 'Noise 10%',
    'noise_0.2': 'Noise 20%',
    'noise_0.3': 'Noise 30%',
    'missing_0.0': 'Clean',
    'missing_0.1': 'Missing 10%',
    'missing_0.2': 'Missing 20%',
    'missing_0.3': 'Missing 30%',
    'missing_0.5': 'Missing 50%',
    'sparse_1': 'All (0.5m)',
    'sparse_2': 'Every 2 (1.0m)',
    'sparse_3': 'Every 3 (1.5m)',
    'sparse_5': 'Every 5 (2.5m)',
}

STRESS_SHORT = {
    'noise_0.0': '0%', 'noise_0.05': '5%', 'noise_0.1': '10%',
    'noise_0.2': '20%', 'noise_0.3': '30%',
    'missing_0.0': '0%', 'missing_0.1': '10%', 'missing_0.2': '20%',
    'missing_0.3': '30%', 'missing_0.5': '50%',
    'sparse_1': '1x', 'sparse_2': '2x', 'sparse_3': '3x', 'sparse_5': '5x',
}

# 压力测试覆盖 5 个模型：全模块版(V5)、去竖向 GCN 变体(V7=Ours)、纯 TCN(V0) 与
# 固定双外部基线（BiLSTM-Attn + GraphWaveNet，见 STRESS_EXTERNAL_BASELINES 注释）。
# DEFENSE 口径 = δ_V7 − δ_V5（见论文 §4.3）：V5/V7/V0 在本脚本内用 AblationModel +
# 变体开关训练；外部基线由 cmp_get_model/cmp_train_model 按对比实验同款配置训练
# （2026-09-25 起 find_best_baseline 不再参与模型选择，仅作历史追溯保留）。
# 2026-09-22 命名澄清：`FULL_MODEL` 指"含竖向量模块的完整变体 V5"，**不是**论文的主模型
# （主模型自 §5.2 消融后为 V7）。此名仅为与历史代码/绘图键对齐而保留；阅读
# `compute_defense` 时须注意：其中 `ours_*` 变量实际指 V5、`base_*` 指 V7。
FULL_MODEL = 'V5_Ours'
V7_MODEL = 'V7_wo_VerticalGCN'
V0_MODEL = 'V0_Pure_TCN'
STRESS_MODELS = [FULL_MODEL, V7_MODEL, V0_MODEL]

# 2026-09-21：Best-Baseline 必须是**真正的外部方法**。V-Attention / V-MLP 与 Ours
# 共享全部骨干（仅替换竖向量模块），属本文框架变体而非外部基线；若不排除，因二者
# RMSE 低于 LSTM/STGCN，find_best_baseline 会选中它们，使压力测试的"外部基线对照"
# 名不副实（对照与 V5 同源，削弱 DEFENSE 归因）。
OURS_FRAMEWORK_MODELS = {'V-Attention', 'V-MLP'}
# 2026-09-22 修补：Comparison 侧主模型的 ModelKey 是 `V7_Ours`（**不是**本脚本的
# `V7_wo_VerticalGCN`），此前未被排除 ⇒ `find_best_baseline` 会把主模型自己选成
# "最强外部基线"（实测选中 `V7_Ours`，RMSE_last_mean=1.5687 全表最低），导致 §4.3 的
# 外部基线对照名不副实，且 V7 会以两个目录名被重复训练。两个名字都须排除。
NON_EXTERNAL_MODELS = {FULL_MODEL, V7_MODEL, 'V7_Ours', 'V0_TCN-Only', 'GBDT'} | OURS_FRAMEWORK_MODELS

# 2026-09-25 方案 C（用户拍板）：压力矩阵的外部基线从"find_best_baseline 自动选 1 个"
# 改为**固定双基线** = BiLSTM-Attn + GraphWaveNet。原因：w64 容量对齐后最强外部基线
# 翻转为 GraphWaveNet（H=10 末步 1.5910 < BiLSTM-Attn 1.5957），而旧矩阵已有
# BiLSTM-Attn 全套结果（宽度 48 口径，STRESS_RESUME=1 下原样保留、不重训）。
# GraphWaveNet 由本脚本经对比实验同款工厂（cmp_get_model/cmp_train_model，
# ComparisonConfig 现为 w64：宽度 64）补训 12 条件 × 5 视界 × 30 种子 = 1800 次
# （扰动 1650 + clean noise_0.0 150，missing_0.0 / sparse_1 由 clean-copy 复制）。
STRESS_EXTERNAL_BASELINES = ['BiLSTM-Attn', 'GraphWaveNet']

MODEL_LABELS = {
    'V5_Ours': 'Full (V5)',
    'V7_wo_VerticalGCN': 'Ours (V7)',
    'V0_Pure_TCN': 'Pure TCN (V0)',
    'Best_Baseline': 'Best Baseline',
    'BiLSTM-Attn': 'BiLSTM-Attn',
    'GraphWaveNet': 'GraphWNet',
}

# SCI colorblind palette (Wong 2011)
MODEL_COLORS = {
    'V5_Ours': '#0072B2',
    'V7_wo_VerticalGCN': '#D55E00',
    'V0_Pure_TCN': '#999999',
    'Best_Baseline': '#009E73',
    'BiLSTM-Attn': '#009E73',    # 沿用旧矩阵 Best-Baseline 配色，图表连续
    'GraphWaveNet': '#CC79A7',
}

MODEL_MARKERS = {
    'V5_Ours': 'o',
    'V7_wo_VerticalGCN': 's',
    'V0_Pure_TCN': '^',
    'Best_Baseline': 'D',
    'BiLSTM-Attn': 'D',          # 沿用旧 Best-Baseline 标记
    'GraphWaveNet': 'v',
}

CONDITION_COLORS = {
    'noise': '#D55E00',
    'missing': '#0072B2',
    'sparse': '#009E73',
}

plt.rcParams.update({
    'font.family': 'serif', 'font.serif': ['Times New Roman'],
    'font.size': 11, 'axes.labelsize': 13, 'axes.titlesize': 14,
    'legend.fontsize': 10, 'figure.dpi': 150, 'savefig.dpi': 600,
    'savefig.bbox': 'tight', 'axes.grid': True, 'grid.alpha': 0.25
})
sns.set_style("whitegrid")


# =============================================================================
# UTILITIES
# =============================================================================
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def wavelet_denoise(signal, wavelet='db4', level=1, thresh_mode='soft'):
    level = min(level, int(np.log2(len(signal))) - 1)
    if level < 1:
        return signal
    coeffs = pywt.wavedec(signal, wavelet, level=level)
    sigma = np.median(np.abs(coeffs[-1])) / 0.6745
    thr = sigma * np.sqrt(2 * np.log(len(signal)))
    coeffs[1:] = [pywt.threshold(c, thr, thresh_mode) for c in coeffs[1:]]
    rec = pywt.waverec(coeffs, wavelet)
    if len(rec) > len(signal):
        return rec[:len(signal)]
    return np.pad(rec, (0, len(signal) - len(rec)), 'edge')


def cohens_d(a, b):
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    n1, n2 = len(a), len(b)
    s_pooled = np.sqrt(((n1 - 1) * np.var(a, ddof=1) + (n2 - 1) * np.var(b, ddof=1)) / (n1 + n2 - 2))
    if s_pooled < 1e-12:
        return 0.0
    return (np.mean(a) - np.mean(b)) / s_pooled


# =============================================================================
# DATA LOADING
# =============================================================================
def load_raw_data_with_meta(cfg):
    all_dfs = []
    for point in cfg.POINTS:
        fp = cfg.BASE_DATA_DIR / f"{point}.xlsx"
        if not fp.exists():
            continue
        df = pd.read_excel(fp)
        col_map = {}
        for c in df.columns:
            if '测量时间' in str(c): col_map[c] = '测量时间'
            elif '测点' in str(c): col_map[c] = '测点'
            elif '深度_m' in str(c): col_map[c] = '深度_m'
            elif '累计水平位移_mm' in str(c): col_map[c] = '累计水平位移_mm'
        df.rename(columns=col_map, inplace=True)
        all_dfs.append(df)
    full = pd.concat(all_dfs, ignore_index=True)
    full['测量时间'] = pd.to_datetime(full['测量时间'])
    return full.sort_values(['测点', '深度_m', '测量时间']).reset_index(drop=True)


def load_water_data(timestamps, cfg):
    water_files = {'SW1': cfg.WATER_DIR / "SW1.xlsx",
                   'SW4': cfg.WATER_DIR / "SW4.xlsx",
                   'SW7': cfg.WATER_DIR / "SW7.xlsx"}
    all_d, all_a = [], []
    ts_int = timestamps.astype(np.int64)
    for p in water_files.values():
        if not p.exists():
            continue
        df = pd.read_excel(p)
        df['监测时间'] = pd.to_datetime(df['监测时间'])
        df = df.sort_values('监测时间')
        t = df['监测时间'].astype(np.int64)
        interp_d = interp1d(t, df['水位深度_m'], kind='linear', fill_value='extrapolate')
        interp_a = interp1d(t, df['累计水位变化_m'], kind='linear', fill_value='extrapolate')
        all_d.append(interp_d(ts_int))
        all_a.append(interp_a(ts_int))
    if all_d:
        return np.mean(all_d, axis=0).astype(np.float32), np.mean(all_a, axis=0).astype(np.float32)
    return np.zeros(len(timestamps), dtype=np.float32), np.zeros(len(timestamps), dtype=np.float32)


def extract_water_features(wd, wa):
    T = len(wd)
    wf = np.zeros((T, 8), dtype=np.float32)
    wf[:, 0] = wd
    wf[:, 1] = wa
    if T > 1: wf[1:, 2] = np.diff(wd)
    if T > 3: wf[3:, 3] = wd[3:] - wd[:-3]
    if T > 7: wf[7:, 4] = wd[7:] - wd[:-7]
    if T > 15: wf[15:, 5] = wd[15:] - wd[:-15]
    if T > 30: wf[30:, 6] = wd[30:] - wd[:-30]
    if T > 1: wf[1:, 7] = np.sign(np.diff(wd))
    return wf


def load_geology_params(cfg):
    if not cfg.GEO_PATH.exists():
        return None
    df = pd.read_excel(cfg.GEO_PATH)
    col_map = {}
    for c in df.columns:
        if '粘聚力' in str(c): col_map[c] = 'cohesion'
        elif '内摩擦角' in str(c): col_map[c] = 'friction'
        elif '压缩模量' in str(c): col_map[c] = 'modulus'
        elif '重度' in str(c): col_map[c] = 'density'
        elif '泊松比' in str(c): col_map[c] = 'poisson'
        elif '静止侧压力系数' in str(c) or 'K0' in str(c): col_map[c] = 'k0'
        elif '厚度' in str(c): col_map[c] = 'thickness'
    df.rename(columns=col_map, inplace=True)
    records = []
    for _, row in df.iterrows():
        records.append({'thickness': float(row['thickness']),
                        'cohesion': float(row.get('cohesion', 0)),
                        'friction': float(row.get('friction', 0)),
                        'modulus': float(row.get('modulus', 0)),
                        'density': float(row.get('density', 0)),
                        'poisson': float(row.get('poisson', 0.3)),
                        'k0': float(row.get('k0', 0.5))})
    return records


def assign_geo_to_nodes(geo_records, depths, cfg):
    N = len(depths)
    geo_feat = np.zeros((N, cfg.GEO_FEAT_DIM), dtype=np.float32)
    if geo_records is None:
        return geo_feat
    cumulative = 0.0
    for rec in geo_records:
        top = cumulative
        bottom = cumulative + rec['thickness']
        cumulative = bottom
        mask = (depths >= top) & (depths < bottom)
        geo_feat[mask, 0] = rec['cohesion']
        geo_feat[mask, 1] = rec['friction']
        geo_feat[mask, 2] = rec['modulus']
        geo_feat[mask, 3] = rec['density']
        geo_feat[mask, 4] = rec['poisson']
        geo_feat[mask, 5] = rec['k0']
    return geo_feat


def build_node_features(full_df, cfg):
    pivot = full_df.pivot_table(index='测量时间', columns=['测点', '深度_m'],
                                values='累计水平位移_mm', aggfunc='first')
    pivot.columns = [f"{p}_{d:.1f}" for p, d in pivot.columns]
    pivot = pivot.sort_index()
    ts = pivot.index
    nodes = [f"{p}_{d:.1f}" for p in cfg.POINTS for d in cfg.DEPTHS]
    valid_mask = np.ones(len(nodes), dtype=np.float32)
    for i, c in enumerate(nodes):
        if c not in pivot.columns:
            pivot[c] = 0.0
            valid_mask[i] = 0.0
    # 2026-09-21：与 Ablation_Experiment.build_node_features 同步的网格完整性断言。
    # 此前缺列静默填 0，坏数据会一路拖到 PCA.fit 才炸；且下游 node_mask / invalid
    # 置零并非真掩码，反把缺失节点的预测污染成 0 而不报错。改为在此显式终止。
    missing = [n for n, m in zip(nodes, valid_mask) if m == 0.0]
    assert not missing, (
        f"位移网格不完整：{len(missing)}/{len(nodes)} 个测点-深度组合缺列 "
        f"（前 5 个：{missing[:5]}）。请检查原始数据覆盖范围。")
    pivot = pivot[nodes]
    disp_raw = pivot.values.astype(np.float32)
    disp_diff1 = np.zeros_like(disp_raw)
    if len(ts) > 1:
        disp_diff1[1:] = np.diff(disp_raw, axis=0)
    point_names = [p for p in cfg.POINTS for _ in cfg.DEPTHS]
    depths_arr = np.array([d for _ in cfg.POINTS for d in cfg.DEPTHS], dtype=np.float32)
    meta = {'node_names': nodes, 'point_names': point_names, 'depths': depths_arr,
            'timestamps': ts, 'valid_mask': valid_mask}
    coord_path = cfg.BASE_DATA_DIR / "斜测点坐标.xlsx"
    N = len(nodes)
    coords = np.zeros((N, 2), dtype=np.float32)
    if coord_path.exists():
        coord_df = pd.read_excel(coord_path)
        for i, (p, d) in enumerate(zip(point_names, depths_arr)):
            row = coord_df[coord_df['测点'] == p]
            if not row.empty:
                coords[i, 0] = row['位置_X_m'].values[0]
                coords[i, 1] = row['位置_Y_m'].values[0]
            else:
                coords[i, 0] = cfg.POINTS.index(p) * 10.0
    else:
        for i, p in enumerate(point_names):
            coords[i, 0] = cfg.POINTS.index(p) * 10.0
    meta['coords'] = coords
    return disp_raw, disp_diff1, meta


def build_geo_similarity(geo_feat):
    """式(10)：地质相似性矩阵 A^geo（土体参数 Z-score 后欧氏距离的高斯核）。

    用于式(11) 的邻接矩阵合成。
    """
    if geo_feat is None or not np.any(geo_feat):
        return None
    geo_norm = (geo_feat - geo_feat.mean(0)) / (geo_feat.std(0) + 1e-6)
    return np.exp(-cdist(geo_norm, geo_norm, metric='euclidean') ** 2 / 2.0)


def build_adjacency(meta, geo_feat, cfg):
    N = len(meta['node_names'])
    coords3d = np.hstack([meta['coords'], meta['depths'].reshape(-1, 1)])
    dist = cdist(coords3d, coords3d, metric='euclidean')
    sigma = np.median(dist[dist > 0]) * 1.0
    adj_spatial = np.exp(-dist ** 2 / (2 * sigma ** 2))

    # 在式(9) 的纯几何高斯核之上叠加两处结构性先验，均由配置开关控制：
    #   1) 同一测斜孔内相邻深度取 0.8 —— 支护结构沿深度方向的材料连续性；
    #   2) 浅层（< 3 m）跨断面节点对取 max(·, 0.6) —— 悬臂段的空间强耦合。
    n_points, n_dep = len(cfg.POINTS), len(cfg.DEPTHS)
    if getattr(cfg, 'SPA_WITHIN_HOLE_PRIOR', True):
        for p in range(n_points):
            start = p * n_dep
            for k in range(start, start + n_dep - 1):
                adj_spatial[k, k + 1] = 0.8
                adj_spatial[k + 1, k] = 0.8
    shallow = meta['depths'] < 3.0
    idxs = np.where(shallow)[0]
    if getattr(cfg, 'SPA_SHALLOW_PRIOR', True):
        for i in idxs:
            for j in idxs:
                if i != j:
                    adj_spatial[i, j] = max(adj_spatial[i, j], 0.6)

    geo_sim = build_geo_similarity(geo_feat)
    adj = 0.7 * adj_spatial + 0.3 * geo_sim if geo_sim is not None else adj_spatial
    np.fill_diagonal(adj, 1.0)

    # 3σ 距离剪枝；浅层跨断面边被显式豁免——浅层相邻断面间距本身接近 σ，
    # 若一并剪除会抹掉悬臂段的耦合通道（issue 12）。
    adj[dist > 3 * sigma] = 0
    if getattr(cfg, 'SPA_SHALLOW_PRIOR', True):
        shallow_keep = (0.7 * adj_spatial + 0.3 * geo_sim
                        if geo_sim is not None else adj_spatial).copy()
        np.fill_diagonal(shallow_keep, 1.0)
        killed = np.zeros_like(adj, dtype=bool)
        killed[np.ix_(idxs, idxs)] = True
        np.fill_diagonal(killed, False)
        killed &= (dist > 3 * sigma)
        adj[killed] = shallow_keep[killed]

    d = np.sum(adj, axis=1)
    d_inv_sqrt = np.power(np.maximum(d, 1e-6), -0.5)
    return (d_inv_sqrt[:, None] * adj * d_inv_sqrt[None, :]).astype(np.float32)


def build_vertical_adjacency(meta, cfg):
    N = len(meta['node_names'])
    n_dep = len(cfg.DEPTHS)
    vert_adj = np.eye(N, dtype=np.float32)
    for p in range(len(cfg.POINTS)):
        start = p * n_dep
        for k in range(start, start + n_dep - 1):
            vert_adj[k, k + 1] = 1.0
            vert_adj[k + 1, k] = 1.0
    d_v = np.sum(vert_adj, axis=1)
    d_v_inv_sqrt = np.power(np.maximum(d_v, 1e-6), -0.5)
    return (d_v_inv_sqrt[:, None] * vert_adj * d_v_inv_sqrt[None, :]).astype(np.float32)


def denoise_split(train, val, test, cfg):
    N = train.shape[1]
    tr_d = np.zeros_like(train)
    v_d = np.zeros_like(val)
    te_d = np.zeros_like(test)
    for i in range(N):
        tr_d[:, i] = wavelet_denoise(train[:, i], cfg.WAVELET, cfg.WAVELET_LEVEL,
                                     cfg.THRESH_MODE)
        v_d[:, i] = wavelet_denoise(val[:, i], cfg.WAVELET, cfg.WAVELET_LEVEL,
                                    cfg.THRESH_MODE)
        te_d[:, i] = wavelet_denoise(test[:, i], cfg.WAVELET, cfg.WAVELET_LEVEL,
                                     cfg.THRESH_MODE)
    return tr_d, v_d, te_d


def make_features(disp_raw, diff1, meta, wf, geo_static, use_water=True, use_geo=True):
    T, N = disp_raw.shape
    depth_norm = np.broadcast_to(meta['depths'] / 16.0, (T, N)).astype(np.float32)
    shallow = np.tile((meta['depths'] < 3.0).astype(np.float32), (T, 1))
    feat_list = [
        disp_raw[..., np.newaxis],
        diff1[..., np.newaxis],
        depth_norm[..., np.newaxis],
        shallow[..., np.newaxis]
    ]
    # 路径 A（2026-09-17）：彻底移除水文输入（此前 use_water=True 拼入 8 维水位，
    # 使输入达 12 维）。VICD 诊断（正文 §3.4）表明水位在增量口径无可利用信息。
    # wf 入参仅作签名占位以兼容既有调用。
    base = np.concatenate(feat_list, axis=-1)
    if use_geo and geo_static is not None:
        geo_tile = np.tile(geo_static[np.newaxis, :, :], (T, 1, 1))
        return np.concatenate([base, geo_tile], axis=-1).astype(np.float32)
    return base.astype(np.float32)


# =============================================================================
# STRESS PERTURBATION FUNCTIONS
# =============================================================================

def apply_noise_3d(X, noise_level, seed):
    """Gaussian noise injection on (T, N, F) feature arrays."""
    if noise_level <= 0:
        return X
    rng = np.random.RandomState(seed)
    X_noisy = X.copy()
    noise = rng.randn(*X.shape).astype(np.float32) * np.std(X) * noise_level
    return X_noisy + noise


def apply_missing_3d(X, rate, seed):
    """Random node missing on (T, N, F) arrays."""
    if rate <= 0:
        return X
    rng = np.random.RandomState(seed)
    N_nodes = X.shape[1]
    n_miss = max(1, int(N_nodes * rate))
    X_miss = X.copy()
    for t in range(X.shape[0]):
        missing_idx = rng.choice(N_nodes, n_miss, replace=False)
        X_miss[t, missing_idx, :] = 0.0
    return X_miss


def apply_sparse_depth_3d(X, keep_every, cfg):
    """Depth sparse sampling on (T, N, F) arrays."""
    if keep_every <= 1:
        return X
    n_dep = cfg.N_DEP
    n_points = cfg.N_POINTS
    orig_N = n_points * n_dep
    assert X.shape[1] == orig_N, f"Expected {orig_N} nodes, got {X.shape[1]}"
    keep_idx = list(range(0, n_dep, keep_every))
    new_n_dep = len(keep_idx)
    new_N = n_points * new_n_dep
    X_sparse = np.zeros((X.shape[0], new_N, X.shape[2]), dtype=np.float32)
    for p in range(n_points):
        for j, d in enumerate(keep_idx):
            src = p * n_dep + d
            dst = p * new_n_dep + j
            X_sparse[:, dst, :] = X[:, src, :]
    return X_sparse


def prepare_stress_data(cfg, shared, hz, stress_type, stress_level):
    """Prepare training data with stress perturbation applied to raw features."""
    train_feat = shared['train_feat'].copy()
    val_feat = shared['val_feat'].copy()
    test_feat = shared['test_feat'].copy()

    base_seed = 42
    if stress_type == 'noise':
        train_feat = apply_noise_3d(train_feat, stress_level, base_seed)
        val_feat = apply_noise_3d(val_feat, stress_level, base_seed + 1)
        test_feat = apply_noise_3d(test_feat, stress_level, base_seed + 2)
    elif stress_type == 'missing':
        train_feat = apply_missing_3d(train_feat, stress_level, base_seed)
        val_feat = apply_missing_3d(val_feat, stress_level, base_seed + 1)
        test_feat = apply_missing_3d(test_feat, stress_level, base_seed + 2)
    elif stress_type == 'sparse':
        train_feat = apply_sparse_depth_3d(train_feat, stress_level, cfg)
        val_feat = apply_sparse_depth_3d(val_feat, stress_level, cfg)
        test_feat = apply_sparse_depth_3d(test_feat, stress_level, cfg)
    else:
        raise ValueError(f"Unknown stress type: {stress_type}")

    F_in = train_feat.shape[-1]
    N_data = train_feat.shape[1]
    scX = StandardScaler().fit(train_feat.reshape(-1, F_in))
    tr_norm = scX.transform(train_feat.reshape(-1, F_in)).reshape(train_feat.shape)
    va_norm = scX.transform(val_feat.reshape(-1, F_in)).reshape(val_feat.shape)
    te_norm = scX.transform(test_feat.reshape(-1, F_in)).reshape(test_feat.shape)

    y_train = shared['y_train']
    y_val = shared['y_val']
    y_test = shared['y_test']
    test_den = shared['test_den']

    if stress_type == 'sparse' and stress_level > 1:
        n_dep = cfg.N_DEP
        n_points = cfg.N_POINTS
        keep_idx = list(range(0, n_dep, stress_level))
        new_n_dep = len(keep_idx)
        new_N = n_points * new_n_dep
        y_train_sp = np.zeros((y_train.shape[0], new_N), dtype=np.float32)
        y_val_sp = np.zeros((y_val.shape[0], new_N), dtype=np.float32)
        y_test_sp = np.zeros((y_test.shape[0], new_N), dtype=np.float32)
        test_den_sp = np.zeros((test_den.shape[0], new_N), dtype=np.float32)
        for p in range(n_points):
            for j, d in enumerate(keep_idx):
                src = p * n_dep + d
                dst = p * new_n_dep + j
                y_train_sp[:, dst] = y_train[:, src]
                y_val_sp[:, dst] = y_val[:, src]
                y_test_sp[:, dst] = y_test[:, src]
                test_den_sp[:, dst] = test_den[:, src]
        y_train, y_val, y_test = y_train_sp, y_val_sp, y_test_sp
        test_den = test_den_sp
    else:
        new_N = N_data

    pca_y = PCA(0.90).fit(y_train)
    out_dim = pca_y.n_components_
    ytr_pca = pca_y.transform(y_train)
    yva_pca = pca_y.transform(y_val)
    yte_pca = pca_y.transform(y_test)
    scY = StandardScaler().fit(ytr_pca)
    ytr_norm = scY.transform(ytr_pca)
    yva_norm = scY.transform(yva_pca)
    yte_norm = scY.transform(yte_pca)

    def make_samples(feat, tgt, lb, hval):
        X, y = [], []
        for i in range(len(feat) - lb - hval + 1):
            X.append(feat[i:i + lb])
            y.append(tgt[i + lb:i + lb + hval])
        return np.array(X, dtype=np.float32), np.array(y, dtype=np.float32)

    Xtr, ytr = make_samples(tr_norm, ytr_norm, cfg.LOOKBACK, hz)
    Xva, yva = make_samples(va_norm, yva_norm, cfg.LOOKBACK, hz)
    Xte, yte = make_samples(te_norm, yte_norm, cfg.LOOKBACK, hz)
    # 评估口径（2026-09-20）：真值窗口直接切原始一阶差分 y_test（稀疏条件下为
    # 重采样后的 y_test_sp），不经 PCA 投影、不经标准化。此前真值也做 pca_y 往返，
    # 把真值投到 K 维子空间后再比，丢弃的增量方差不计入误差。
    yte_raw = np.stack([y_test[i + cfg.LOOKBACK:i + cfg.LOOKBACK + hz]
                        for i in range(Xte.shape[0])]) if Xte.shape[0] else np.empty(
        (0, hz, y_test.shape[1]), dtype=np.float32)
    init_test = test_den[cfg.LOOKBACK - 1:-hz][:Xte.shape[0]]

    valid_mask = shared['valid_mask']
    if stress_type == 'sparse' and stress_level > 1:
        valid_mask_sp = np.ones(new_N, dtype=np.float32)
        for p in range(n_points):
            for j, d in enumerate(keep_idx):
                valid_mask_sp[p * new_n_dep + j] = valid_mask[p * n_dep + d]
        valid_mask = valid_mask_sp
    invalid_idx = np.where(valid_mask == 0)[0]
    if len(invalid_idx) > 0:
        Xtr[:, :, invalid_idx, :] = 0.0
        Xva[:, :, invalid_idx, :] = 0.0
        Xte[:, :, invalid_idx, :] = 0.0
        init_test[:, invalid_idx] = 0.0

    return {
        'Xtr': Xtr, 'ytr': ytr, 'Xva': Xva, 'yva': yva,
        'Xte': Xte, 'yte': yte, 'yte_raw': yte_raw, 'init_test': init_test,
        'scY': scY, 'pca_y': pca_y, 'out_dim': out_dim, 'F_in': F_in,
        'N': Xtr.shape[2],
        'is_sparse': (stress_type == 'sparse' and stress_level > 1),
        # Perturbed raw (unnormalized) features, needed to rebuild temporal
        # (flattened + PCA) samples for external Best-Baseline models.
        'train_feat_p': train_feat, 'val_feat_p': val_feat, 'test_feat_p': test_feat,
        'valid_mask': valid_mask,
    }


def build_sparse_adj_and_meta(meta, geo_fb, N_full, stress_level, cfg):
    """Rebuild adjacency for sparse-sampled nodes."""
    n_dep = cfg.N_DEP
    n_points = cfg.N_POINTS
    keep_idx = list(range(0, n_dep, stress_level))
    new_n_dep = len(keep_idx)
    new_N = n_points * new_n_dep

    depths_sp = []
    coords_sp = []
    point_names_sp = []
    for p in range(n_points):
        for d in keep_idx:
            idx = p * n_dep + d
            depths_sp.append(meta['depths'][idx])
            coords_sp.append(meta['coords'][idx])
            point_names_sp.append(meta['point_names'][idx])

    depths_arr = np.array(depths_sp, dtype=np.float32)
    coords_arr = np.array(coords_sp, dtype=np.float32)
    meta_sp = {'depths': depths_arr, 'coords': coords_arr,
               'point_names': point_names_sp,
               'node_names': [f"{pn}_{d:.1f}" for pn, d in
                              zip(point_names_sp, depths_sp)]}

    from scipy.spatial.distance import cdist
    coords3d = np.hstack([coords_arr, depths_arr.reshape(-1, 1)])
    dist = cdist(coords3d, coords3d, metric='euclidean')
    sigma = np.median(dist[dist > 0]) * 1.0
    adj_spatial = np.exp(-dist ** 2 / (2 * sigma ** 2))
    for p in range(n_points):
        start = p * new_n_dep
        for k in range(start, start + new_n_dep - 1):
            adj_spatial[k, k + 1] = 0.8
            adj_spatial[k + 1, k] = 0.8
    shallow = depths_arr < 3.0
    idxs = np.where(shallow)[0]
    for i in idxs:
        for j in idxs:
            if i != j:
                adj_spatial[i, j] = max(adj_spatial[i, j], 0.6)

    if geo_fb is not None and np.any(geo_fb):
        # Correct node-wise sparse sampling: node index = p * n_dep + d,
        # so every measurement point keeps its OWN geology parameters at the
        # retained depths (the previous `list(keep_idx) * n_points` duplicated
        # the first point's indices for all points -- issue 5).
        geo_sp = geo_fb[[p * n_dep + d for p in range(n_points) for d in keep_idx]]
        geo_norm = (geo_sp - geo_sp.mean(0)) / (geo_sp.std(0) + 1e-6)
        from scipy.spatial.distance import cdist as cdist2
        geo_sim = np.exp(-cdist2(geo_norm, geo_norm, metric='euclidean') ** 2 / 2.0)
        adj = 0.7 * adj_spatial + 0.3 * geo_sim
    else:
        adj = adj_spatial

    np.fill_diagonal(adj, 1.0)
    # Preserve the designed shallow cross-point edges (0.6) across the 3-sigma
    # distance pruning, otherwise a small median distance sigma can wipe them
    # out entirely (issue 12).
    shallow_keep = adj.copy()
    adj[dist > 3 * sigma] = 0
    shallow_mask = np.zeros_like(adj, dtype=bool)
    shallow_mask[np.ix_(idxs, idxs)] = True
    np.fill_diagonal(shallow_mask, False)
    killed = shallow_mask & (dist > 3 * sigma)
    adj[killed] = shallow_keep[killed]
    d = np.sum(adj, axis=1)
    d_inv_sqrt = np.power(np.maximum(d, 1e-6), -0.5)
    adj_norm = (d_inv_sqrt[:, None] * adj * d_inv_sqrt[None, :]).astype(np.float32)

    vert_adj = np.eye(new_N, dtype=np.float32)
    for p in range(n_points):
        start = p * new_n_dep
        for k in range(start, start + new_n_dep - 1):
            vert_adj[k, k + 1] = 1.0
            vert_adj[k + 1, k] = 1.0
    d_v = np.sum(vert_adj, axis=1)
    d_v_inv_sqrt = np.power(np.maximum(d_v, 1e-6), -0.5)
    vert_adj_norm = (d_v_inv_sqrt[:, None] * vert_adj * d_v_inv_sqrt[None, :]).astype(np.float32)

    return adj_norm, vert_adj_norm, meta_sp


# =============================================================================
# MODEL ARCHITECTURE (reused from Ablation_Experiment.py)
# =============================================================================
class CausalConv1d(nn.Module):
    def __init__(self, ci, co, k, d):
        super().__init__()
        self.pad = (k - 1) * d
        self.conv = nn.Conv1d(ci, co, k, dilation=d, padding=self.pad)

    def forward(self, x):
        out = self.conv(x)
        if self.pad > 0:
            return out[:, :, :-self.pad]
        return out


class TCNBlock(nn.Module):
    def __init__(self, ci, co, k, d, dropout):
        super().__init__()
        self.conv1 = CausalConv1d(ci, co, k, d)
        self.ln1 = nn.LayerNorm(co)
        self.conv2 = CausalConv1d(co, co, k, d)
        self.ln2 = nn.LayerNorm(co)
        self.drop = nn.Dropout(dropout)
        self.downsample = nn.Conv1d(ci, co, 1) if ci != co else None
        self.act = nn.LeakyReLU(0.1)

    def forward(self, x):
        res = x
        out = self.act(self.ln1(self.conv1(x).transpose(1, 2)).transpose(1, 2))
        out = self.drop(out)
        out = self.act(self.ln2(self.conv2(out).transpose(1, 2)).transpose(1, 2))
        out = self.drop(out)
        if self.downsample is not None:
            res = self.downsample(res)
        return self.act(out + res)


class TCNEncoder(nn.Module):
    def __init__(self, in_dim, channels, k, dilations, dropout):
        super().__init__()
        layers = []
        ci = in_dim
        for co, d in zip(channels, dilations):
            layers.append(TCNBlock(ci, co, k, d, dropout))
            ci = co
        self.net = nn.Sequential(*layers)
        self.out_dim = channels[-1]

    def forward(self, x):
        return self.net(x.transpose(1, 2)).transpose(1, 2)


class GatedGCN(nn.Module):
    def __init__(self, ci, co, adj, dropout=0.5):
        super().__init__()
        self.register_buffer('adj', adj)
        self.lin = nn.Linear(ci, co)
        self.gate = nn.Linear(ci + co, co)
        self.drop = nn.Dropout(dropout)

    def forward(self, x, dynamic_adj=None):
        B, N, _ = x.shape
        adj = self.adj if dynamic_adj is None else dynamic_adj
        if adj.dim() == 2:
            adj = adj.unsqueeze(0).expand(B, N, N)
        t = self.lin(x)
        agg = torch.bmm(adj, t)
        gate = torch.sigmoid(self.gate(torch.cat([x, agg], -1)))
        return self.drop(F.relu(gate * t + (1 - gate) * agg))


class PhysicsGate(nn.Module):
    def __init__(self, dim, geo_dim=7):
        super().__init__()
        self.gate = nn.Sequential(nn.Linear(dim + geo_dim, dim), nn.Sigmoid())

    def forward(self, x, geo_feat):
        return x * self.gate(torch.cat([x, geo_feat], dim=-1))


class CrossNodeAttention(nn.Module):
    """式(18)-(20)：跨节点注意力 + 逐节点融合门。

    注：曾试过在打分上叠加地质相似性偏置 γ·A^geo（式(10) 的矩阵）。15 种子 A/B 显示
    γ 训练后仅 ~1e-3 量级、对结果无可测影响（无水位支路的变体逐位不变），故移除；
    地质先验经式(11) 的邻接拓扑与式(21) 的融合门进入模型。
    """

    def __init__(self, dim, geo_dim=7):
        super().__init__()
        self.dim = dim
        self.q = nn.Linear(dim, dim)
        self.k = nn.Linear(dim, dim)
        self.v = nn.Linear(dim, dim)
        self.gate = nn.Linear(dim + geo_dim, 1)

    def forward(self, x, geo_feat):
        B, N, D = x.shape
        q = self.q(x)
        k = self.k(x)
        v = self.v(x)
        attn = torch.softmax(torch.matmul(q, k.transpose(-2, -1)) / np.sqrt(D), dim=-1)
        cross = torch.matmul(attn, v)
        gate = torch.sigmoid(self.gate(torch.cat([x, geo_feat], dim=-1)))
        return gate * x + (1 - gate) * cross


class VerticalGCN(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.vert_proj = nn.Sequential(nn.Linear(dim, dim), nn.ReLU(), nn.Linear(dim, dim))

    def forward(self, x, vert_adj):
        B, N, D = x.shape
        vert_agg = torch.bmm(vert_adj.expand(B, N, N), self.vert_proj(x))
        return x + vert_agg


class AblationModel(nn.Module):
    def __init__(self, cfg, N, in_dim, gcn_hid, tcn_ch, k, dil, attn_hid, hor,
                 adj, out_dim, geo_feat_dim=7, vert_adj=None,
                 use_vert=True, use_gcn=True, use_attn=True,
                 use_physics=True, use_cross=True):
        super().__init__()
        self.use_attn = use_attn
        self.use_gcn = use_gcn
        self.use_cross_attn = use_cross
        self.use_physics_gate = use_physics
        # 路径 A（2026-09-17）：不再使用水文。此前 use_water 控制 water_node_proj
        # 分支，VICD 诊断（正文 §3.4）表明水位在增量口径无可用信息，故彻底移除。
        self.use_water_gcn = False
        self.use_vertical_gcn = use_vert

        self.tcn = TCNEncoder(in_dim, tcn_ch, k, dil, cfg.TCN_DROPOUT)
        tcn_dim = tcn_ch[-1]

        # 架构简化（2026-09-19）：时序注意力改为均值池化（与消融 Ablation_Experiment
        # 同步，见其 __init__ 注释）。删除 attn1/attn_v 模块；USE_ATTN=True→mean,
        # False→last（V1 对照语义不变）。
        # 2026-09-20：`_last_temporal_attn` 死属性已删除（无任何消费者）。

        if self.use_gcn:
            gcn_in_dim = tcn_dim
            self.gcn = GatedGCN(gcn_in_dim, gcn_hid, adj, dropout=cfg.DROPOUT)
            self.gcn_norm = nn.LayerNorm(gcn_hid)
            self.gcn_drop = nn.Dropout(cfg.DROPOUT)
            cur_dim = gcn_hid
        else:
            cur_dim = tcn_dim

        # 2026-09-20：原此处另注册 `static_adj`（与 `self.gcn.adj` 重复），无消费者，已删除。
        if vert_adj is not None:
            self.register_buffer('vert_adj', vert_adj)
        else:
            self.vert_adj = None

        # Module creation order unified with Ablation_Experiment.AblationModel
        # (gcn -> physics -> cross -> vertical_gcn -> head). PyTorch consumes the
        # global RNG for weight init in construction order, so a different order
        # yields different initial weights for the same model -- this was the
        # cause of the V5 clean-baseline discrepancy (stress V5 != ablation V5).
        if self.use_physics_gate:
            self.physics_gate = PhysicsGate(cur_dim, geo_feat_dim)

        if self.use_cross_attn:
            self.cross_attn = CrossNodeAttention(cur_dim, geo_feat_dim)

        if self.use_vertical_gcn:
            vert_in_dim = tcn_dim
            self.vertical_gcn = VerticalGCN(vert_in_dim)

        self.head = nn.Sequential(
            nn.Linear(cur_dim, cur_dim // 2),
            nn.LeakyReLU(0.1),
            nn.Dropout(cfg.DROPOUT),
            nn.Linear(cur_dim // 2, hor * out_dim))
        self.horizon = hor
        self.out_dim = out_dim

    def forward(self, x, water_feat, geo_static, node_mask=None):
        B, seq, Nodes, in_feat = x.shape
        tcn_in = x.permute(0, 2, 1, 3).reshape(B * Nodes, seq, in_feat)
        tcn_out = self.tcn(tcn_in)

        if self.use_attn:
            # 架构简化（2026-09-19）：原单头加性注意力改为显式均值池化（见 __init__ 注释）。
            ctx = tcn_out.mean(dim=1)
        else:
            ctx = tcn_out[:, -1, :]
        ctx = ctx.reshape(B, Nodes, -1)
        # Mask out invalid nodes (missing-data columns) before graph aggregation,
        # same as Ablation_Experiment.py (issue 11).
        if node_mask is not None and ctx.shape[1] == node_mask.shape[0]:
            ctx = ctx * node_mask.view(1, -1, 1).to(ctx.device)

        # 路径 A（2026-09-17）：已移除水文注入（water_node_proj 拼接）。
        # water_feat 入参仅作签名占位以兼容既有 DataLoader。

        if self.use_vertical_gcn:
            vadj = self.vert_adj
            if vadj is not None:
                ctx = self.vertical_gcn(ctx, vadj)

        if self.use_gcn:
            ctx = self.gcn(ctx, dynamic_adj=None)
            ctx = self.gcn_norm(ctx)
            ctx = self.gcn_drop(F.relu(ctx))

        if self.use_physics_gate:
            ctx = self.physics_gate(ctx, geo_static)

        if self.use_cross_attn:
            ctx = self.cross_attn(ctx, geo_static)

        global_feat = ctx.mean(dim=1)
        out = self.head(global_feat)
        return out.reshape(B, self.horizon, self.out_dim), None


# =============================================================================
# TRAINING & EVALUATION
# =============================================================================
class MultiLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.huber = nn.HuberLoss(delta=1.0)
        self.mse = nn.MSELoss()

    def forward(self, pred, target):
        B, hor, dim = pred.shape
        l_hub = self.huber(pred, target)
        l_diff = torch.tensor(0., device=pred.device)
        if hor > 1:
            l_diff = self.mse(pred[:, 1:] - pred[:, :-1], target[:, 1:] - target[:, :-1])
        pred_fft = torch.fft.rfft(pred, dim=1)
        target_fft = torch.fft.rfft(target, dim=1)
        l_freq = self.mse(torch.abs(pred_fft), torch.abs(target_fft))
        return 1.0 * l_hub + 0.5 * l_diff + 0.2 * l_freq


def _graph_lr_param_groups(model):
    """图/物理相关模块参数（低学习率组）与其余参数。

    2026-09-20：改为**按模块对象显式收集**，不再依赖参数名子串匹配
    （'gcn' 会顺带命中 gcn_norm；vertical_gcn 此前靠 'gcn' 巧合命中）。
    变体关闭时模块不存在，用 getattr 容错；返回顺序与原实现逐位等价。
    """
    holders = [getattr(model, n, None) for n in
               ('gcn', 'gcn_norm', 'gcn_drop', 'physics_gate',
                'cross_attn', 'vertical_gcn')]
    gid = {id(p) for m in holders if m is not None for p in m.parameters()}
    gra = [p for _, p in model.named_parameters() if id(p) in gid]
    rest = [p for _, p in model.named_parameters() if id(p) not in gid]
    return gra, rest


def _assert_internal_model(model, caller):
    """护栏（2026-09-22）：本脚本的 train_model / evaluate 只服务**内部变体**
    AblationModel——其 forward 返回 `(out, None)` 且接受 `node_mask`。外部基线
    （Best-Baseline：LSTM / BiLSTM-Attn / STGCN / GraphWaveNet 等）的 forward 返回单个
    tensor、且无 `node_mask` 形参，必须走 Comparison_Experiment 的
    cmp_train_model / cmp_evaluate（分流点：run_stress_single 的 `is_external`）。
    在此提前失败，避免在 `pred, _ = model(...)` 处抛出难以定位的 ValueError/TypeError。
    """
    if not isinstance(model, AblationModel):
        raise TypeError(
            f"{caller}() 只接受内部变体 AblationModel（V5/V7/V0），收到 "
            f"{type(model).__name__}。外部基线请改走 Comparison_Experiment 的 "
            f"cmp_train_model / cmp_evaluate（见 run_stress_single 的 is_external 分流）。")


def train_model(cfg, model, train_ld, val_ld, save_path, model_tag='', seed=0,
                node_mask=None):
    _assert_internal_model(model, 'train_model')
    gra, rest = _graph_lr_param_groups(model)
    opt = torch.optim.AdamW([
        {'params': gra, 'lr': cfg.LR_GCN},
        {'params': rest, 'lr': cfg.LR}
    ], weight_decay=cfg.WD)
    sch = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode='min', factor=0.5, patience=15)
    crit = MultiLoss().to(cfg.DEVICE)
    best_vloss = float('inf')
    wait = 0
    t_start = time.time()
    tr_losses, va_losses = [], []
    log_interval = max(1, cfg.EPOCHS // 20)
    for ep in range(1, cfg.EPOCHS + 1):
        model.train()
        tr_loss = 0
        for batch_data in train_ld:
            Xb = batch_data[0].to(cfg.DEVICE)
            yb = batch_data[1].to(cfg.DEVICE)
            opt.zero_grad()
            pred, _ = model(*([Xb] + [t.to(cfg.DEVICE) for t in batch_data[2:]]),
                           node_mask=node_mask)
            loss = crit(pred, yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.GRAD_CLIP)
            opt.step()
            tr_loss += loss.item() * Xb.size(0)
        tr_loss /= len(train_ld.dataset)
        tr_losses.append(tr_loss)
        model.eval()
        va_loss = 0
        with torch.no_grad():
            for batch_data in val_ld:
                Xb = batch_data[0].to(cfg.DEVICE)
                yb = batch_data[1].to(cfg.DEVICE)
                pred, _ = model(*([Xb] + [t.to(cfg.DEVICE) for t in batch_data[2:]]),
                               node_mask=node_mask)
                loss = crit(pred, yb)
                va_loss += loss.item() * Xb.size(0)
        va_loss /= len(val_ld.dataset)
        va_losses.append(va_loss)
        sch.step(va_loss)
        if ep % log_interval == 0 or ep == 1 or ep == cfg.EPOCHS:
            logger.info(f"    [{model_tag} seed={seed}] Epoch {ep:4d}/{cfg.EPOCHS} | "
                        f"train={tr_loss:.6f} | val={va_loss:.6f}")
        if va_loss < best_vloss:
            best_vloss = va_loss
            wait = 0
            torch.save(model.state_dict(), save_path)
        else:
            wait += 1
            if wait >= cfg.PATIENCE:
                break
    train_time = time.time() - t_start
    model.load_state_dict(torch.load(save_path, map_location=cfg.DEVICE))
    return model, train_time, tr_losses, va_losses


def evaluate(model, test_ld, init_disp, scY, pca_y, hz, device, node_mask=None,
             true_diff_win=None):
    """测试集评估：预测走 PCA 逆变换，真值取原始一阶差分（口径见 2026-09-20 修正）。

    ⚠️ 仅供**内部变体**（V5/V7/V0 = `AblationModel`）调用。外部基线的 forward 返回单个
    tensor 且无 `node_mask` 形参，须改走 `Comparison_Experiment.evaluate`（本文件别名
    `cmp_evaluate`）；两条路径的分流点在 `run_stress_single` 的 `is_external` 判断处。
    """
    _assert_internal_model(model, 'evaluate')
    model.eval()
    preds_norm, trues_norm = [], []
    with torch.no_grad():
        for batch_data in test_ld:
            Xb = batch_data[0].to(device)
            yb = batch_data[1].to(device)
            pred, _ = model(*([Xb] + [t.to(device) for t in batch_data[2:]]),
                           node_mask=node_mask)
            preds_norm.append(pred.cpu().numpy())
            trues_norm.append(yb.cpu().numpy())
    pred_n = np.concatenate(preds_norm, 0)
    out_dim = pred_n.shape[-1]
    ns = pred_n.shape[0]
    pred_pca = scY.inverse_transform(pred_n.reshape(-1, out_dim)).reshape(ns, hz, out_dim)
    N_orig = pca_y.n_features_in_
    pred_diff = pca_y.inverse_transform(pred_pca.reshape(-1, out_dim)).reshape(ns, hz, N_orig)
    # 2026-09-20 口径修正：真值取原始一阶差分切片，不再经 PCA 往返
    # （预测侧仍走 PCA 逆变换）。理由见 Ablation_Experiment.evaluate。
    if true_diff_win is not None:
        true_diff = np.asarray(true_diff_win, dtype=np.float64)[:ns]
    else:
        true_n = np.concatenate(trues_norm, 0)
        true_pca = scY.inverse_transform(true_n.reshape(-1, out_dim)).reshape(ns, hz, out_dim)
        true_diff = pca_y.inverse_transform(true_pca.reshape(-1, out_dim)).reshape(
            ns, hz, N_orig)
    pred_cum = np.zeros_like(pred_diff)
    true_cum = np.zeros_like(true_diff)
    for i in range(ns):
        cp = init_disp[i].copy()
        ct = init_disp[i].copy()
        for t in range(hz):
            cp += pred_diff[i, t, :]
            ct += true_diff[i, t, :]
            pred_cum[i, t, :] = cp
            true_cum[i, t, :] = ct
    return pred_cum, true_cum


def global_metrics(true_cum, pred_cum):
    """全步口径 + 末步口径（`_last` 后缀），口径定义见 Ablation_Experiment.global_metrics。"""
    def _m(a, b):
        return {
            'RMSE': float(np.sqrt(mean_squared_error(a, b))),
            'MAE': float(mean_absolute_error(a, b)),
            'R2': float(r2_score(a, b)),
            'MaxErr': float(max_error(a, b))
        }
    gm = _m(true_cum.ravel(), pred_cum.ravel())
    gm.update({f'{k}_last': v for k, v in
               _m(true_cum[:, -1, :].ravel(), pred_cum[:, -1, :].ravel()).items()})
    return gm


def count_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# =============================================================================
# EXTERNAL (BEST-BASELINE) HELPERS
# =============================================================================
def compute_scaled_laplacian(adj_norm):
    """Scaled Laplacian for STGCN, identical to Comparison_Experiment.py."""
    N = adj_norm.shape[0]
    laplacian = np.eye(N) - adj_norm
    eigvals = np.linalg.eigvalsh(laplacian)
    lambda_max = float(np.max(eigvals))
    L_scaled = 2.0 * laplacian / max(lambda_max, 2.0) - np.eye(N)
    return L_scaled.astype(np.float32)


def make_comparison_cfg(stress_cfg):
    """Comparison-experiment config; EPOCHS/PATIENCE follow the stress cfg so
    that --quick mode works consistently for the Best-Baseline too."""
    cmp_cfg = ComparisonConfig()
    cmp_cfg.EPOCHS = stress_cfg.EPOCHS
    cmp_cfg.PATIENCE = stress_cfg.PATIENCE
    return cmp_cfg


def build_temporal_stress_samples(data, cmp_cfg, hz, lookback):
    """Flatten + PCA the perturbed raw features for temporal Best-Baseline
    models (LSTM / BiLSTM-Attn / Transformer), mirroring
    Comparison_Experiment.build_temporal_samples. Targets / init / scalers are
    reused from the graph pipeline (identical time alignment)."""
    train_feat = data['train_feat_p']
    val_feat = data['val_feat_p']
    test_feat = data['test_feat_p']
    T_tr = train_feat.shape[0]
    N, F_in = train_feat.shape[1], train_feat.shape[2]
    flat_dim = N * F_in
    # 2026-09-25 w64 拆分对齐：时序基线的展平输入 PCA 维固定 48（TEMPORAL_PCA_DIM），
    # 不再跟随 HIDDEN_DIM（拆分后 = 64，是网络宽度）。旧 Stress_Results 中 BiLSTM-Attn
    # 即按输入 48 训练，保持同口径。
    actual_dim = min(getattr(cmp_cfg, 'TEMPORAL_PCA_DIM', cmp_cfg.HIDDEN_DIM),
                     T_tr // 2, flat_dim)
    logger.info(f"    Temporal Best-Baseline: flatten {flat_dim}D -> PCA -> {actual_dim}D")
    pca_in = PCA(n_components=actual_dim).fit(train_feat.reshape(T_tr, flat_dim))
    tr_proj = pca_in.transform(train_feat.reshape(T_tr, flat_dim)).astype(np.float32)
    va_proj = pca_in.transform(val_feat.reshape(-1, flat_dim)).astype(np.float32)
    te_proj = pca_in.transform(test_feat.reshape(-1, flat_dim)).astype(np.float32)
    scX_t = StandardScaler().fit(tr_proj)
    tr_norm = scX_t.transform(tr_proj)
    va_norm = scX_t.transform(va_proj)
    te_norm = scX_t.transform(te_proj)

    def make_x(feat, lb, hval):
        Xs = []
        for i in range(len(feat) - lb - hval + 1):
            Xs.append(feat[i:i + lb])
        return np.array(Xs, dtype=np.float32)

    return {
        'Xtr': make_x(tr_norm, lookback, hz),
        'Xva': make_x(va_norm, lookback, hz),
        'Xte': make_x(te_norm, lookback, hz),
        'in_dim': actual_dim,
    }


# =============================================================================
# RUN STRESS TEST FOR (model, hz, stress_type, stress_level)
# =============================================================================
def run_stress_single(cfg, model_name, shared, hz, stress_type, stress_level):
    tag = f"{model_name}_{stress_type}_{stress_level}"
    logger.info(f"  [{tag}] H={hz} begin")

    data = prepare_stress_data(cfg, shared, hz, stress_type, stress_level)
    Xtr = data['Xtr']; ytr = data['ytr']
    Xva = data['Xva']; yva = data['yva']
    Xte = data['Xte']; yte = data['yte']
    init_test = data['init_test']
    out_dim = data['out_dim']; scY = data['scY']; pca_y = data['pca_y']
    N_data = data['N']; F_in = data['F_in']

    # 内部路径（V5/V7/V0）用本脚本的 AblationModel + 变体开关；其余模型
    # （自动选出的 Best-Baseline）一律走对比实验的模型工厂（is_external）。
    is_external = (model_name not in STRESS_MODELS)
    if model_name == FULL_MODEL:
        # V5：全模块
        use_vert = use_gcn = use_attn = use_physics = use_cross = True
    elif model_name == V7_MODEL:
        # V7：仅关竖向 GCN（$A^{vert}$ 通道），其余全开（与消融 V7 语义一致）
        use_vert = False
        use_gcn = use_attn = use_physics = use_cross = True
    elif model_name == V0_MODEL:
        # V0：纯 TCN（关全部空间模块）。use_attn=True→均值池化，与消融 V0 及 V5
        # 同口径（2026-09-21 修正，理由见 Ablation_Experiment.ABLATION_VARIANTS 注释）
        use_vert = use_gcn = use_physics = use_cross = False
        use_attn = True
    else:
        use_vert = use_gcn = use_attn = use_physics = use_cross = False
    mtype = CMP_MODEL_TYPES.get(model_name, 'graph') if is_external else 'graph'

    adj = build_adjacency(shared['meta'], shared['geo_fb'], cfg)
    vert_adj_np = build_vertical_adjacency(shared['meta'], cfg)
    L_scaled = compute_scaled_laplacian(adj) if is_external else None

    is_sparse = data.get('is_sparse', False)
    if is_sparse:
        adj, vert_adj_np, _ = build_sparse_adj_and_meta(
            shared['meta'], shared['geo_fb'], shared['N'], stress_level, cfg)
        if is_external:
            L_scaled = compute_scaled_laplacian(adj)

    adj_t = torch.FloatTensor(adj).to(cfg.DEVICE)
    vert_adj_t = torch.FloatTensor(vert_adj_np).to(cfg.DEVICE)
    geo_static_full = shared['geo_static']
    if is_sparse:
        n_points = cfg.N_POINTS
        n_dep = cfg.N_DEP
        keep_idx = list(range(0, n_dep, stress_level))
        new_n_dep = len(keep_idx)
        sparse_indices = []
        for p in range(n_points):
            for d in keep_idx:
                sparse_indices.append(p * n_dep + d)
        geo_static = geo_static_full[sparse_indices]
    else:
        geo_static = geo_static_full
    geo_dim = geo_static.shape[1]
    geo_t = torch.FloatTensor(geo_static).to(cfg.DEVICE)

    # 路径 A（2026-09-17）：水位已不再进入模型，以下 water 特征仅作占位透传，
    # 使 DataLoader / forward 签名保持不变（模型内部不再使用 water_feat）。
    water_feat_full = shared['water_feat']
    w_mean = water_feat_full[shared['train_idx']].mean(0)
    w_std = water_feat_full[shared['train_idx']].std(0) + 1e-6
    water_feat_full = (water_feat_full - w_mean) / w_std
    wf_train = water_feat_full[shared['train_idx']][cfg.LOOKBACK - 1:-hz][:Xtr.shape[0]]
    wf_val = water_feat_full[shared['val_idx']][cfg.LOOKBACK - 1:-hz][:Xva.shape[0]]
    wf_test = water_feat_full[shared['test_idx']][cfg.LOOKBACK - 1:-hz][:Xte.shape[0]]
    wf_tr = np.tile(wf_train[:, np.newaxis, :], (1, N_data, 1)).astype(np.float32)
    wf_va = np.tile(wf_val[:, np.newaxis, :], (1, N_data, 1)).astype(np.float32)
    wf_te = np.tile(wf_test[:, np.newaxis, :], (1, N_data, 1)).astype(np.float32)
    geo_tr = np.tile(geo_static[np.newaxis], (Xtr.shape[0], 1, 1))
    geo_va = np.tile(geo_static[np.newaxis], (Xva.shape[0], 1, 1))
    geo_te = np.tile(geo_static[np.newaxis], (Xte.shape[0], 1, 1))

    # Node mask for invalid nodes (issue 11), same as the ablation training.
    valid_mask_t = torch.tensor(data['valid_mask'], dtype=torch.float32,
                                device=cfg.DEVICE)

    out_dir = cfg.OUT_ROOT / f"H{hz}" / f"{stress_type}_{stress_level}" / model_name
    out_dir.mkdir(parents=True, exist_ok=True)

    all_global = []
    for si, seed in enumerate(cfg.SEEDS):
        set_seed(seed)

        # 2026-09-22：断点续跑（STRESS_RESUME=1）——四件套齐全即视为已完成，
        # 读回 metrics.json 参与汇总；跳过时不做数据准备/建模/训练（见 Config.RESUME）。
        _run_dir = out_dir / f"run_{seed}"
        if cfg.RESUME and all((_run_dir / f).exists()
                              for f in ('metrics.json', 'best.pth',
                                        'predictions.npz', 'losses.npy')):
            with open(_run_dir / 'metrics.json', 'r') as f:
                gm = json.load(f)
            gm['seed'] = seed
            all_global.append(gm)
            logger.info(f"    Seed {si + 1}/{len(cfg.SEEDS)} ({seed}): "
                        f"结果已完整，跳过（RESUME=1）")
            continue

        if is_external:
            # Best-Baseline: built and trained exactly as in the comparison
            # experiment (same architecture / hyperparameters / optimizer).
            cmp_cfg = make_comparison_cfg(cfg)
            if mtype == 'temporal':
                tdata = build_temporal_stress_samples(data, cmp_cfg, hz, cfg.LOOKBACK)
                model = cmp_get_model(cmp_cfg, model_name, N_data, tdata['in_dim'],
                                      out_dim, hz, adj, L_scaled)
                train_ds = TensorDataset(torch.FloatTensor(tdata['Xtr']),
                                          torch.FloatTensor(ytr))
                val_ds = TensorDataset(torch.FloatTensor(tdata['Xva']),
                                       torch.FloatTensor(yva))
                test_ds = TensorDataset(torch.FloatTensor(tdata['Xte']),
                                        torch.FloatTensor(yte))
            else:
                n_dep_actual = N_data // cfg.N_POINTS
                model = cmp_get_model(cmp_cfg, model_name, N_data, F_in, out_dim,
                                      hz, adj, L_scaled, geo_dim=geo_dim,
                                      n_points=cfg.N_POINTS, n_dep=n_dep_actual)
                train_ds = TensorDataset(torch.FloatTensor(Xtr), torch.FloatTensor(ytr),
                                         torch.FloatTensor(wf_tr), torch.FloatTensor(geo_tr))
                val_ds = TensorDataset(torch.FloatTensor(Xva), torch.FloatTensor(yva),
                                       torch.FloatTensor(wf_va), torch.FloatTensor(geo_va))
                test_ds = TensorDataset(torch.FloatTensor(Xte), torch.FloatTensor(yte),
                                        torch.FloatTensor(wf_te), torch.FloatTensor(geo_te))
            train_ld = DataLoader(train_ds, batch_size=cmp_cfg.BATCH_SIZE, shuffle=True)
            val_ld = DataLoader(val_ds, batch_size=cmp_cfg.BATCH_SIZE)
            test_ld = DataLoader(test_ds, batch_size=cmp_cfg.BATCH_SIZE)
        else:
            model = AblationModel(
                cfg, N=N_data, in_dim=F_in, gcn_hid=cfg.GCN_HIDDEN,
                tcn_ch=cfg.TCN_CHANNELS, k=cfg.TCN_KERNEL_SIZE,
                dil=cfg.TCN_DILATIONS, attn_hid=cfg.ATTENTION_HIDDEN,
                hor=hz, adj=adj_t, out_dim=out_dim,
                geo_feat_dim=geo_dim, vert_adj=vert_adj_t,
                use_vert=use_vert, use_gcn=use_gcn, use_attn=use_attn,
                use_physics=use_physics, use_cross=use_cross).to(cfg.DEVICE)
            train_ds = TensorDataset(torch.FloatTensor(Xtr), torch.FloatTensor(ytr),
                                     torch.FloatTensor(wf_tr), torch.FloatTensor(geo_tr))
            val_ds = TensorDataset(torch.FloatTensor(Xva), torch.FloatTensor(yva),
                                   torch.FloatTensor(wf_va), torch.FloatTensor(geo_va))
            test_ds = TensorDataset(torch.FloatTensor(Xte), torch.FloatTensor(yte),
                                    torch.FloatTensor(wf_te), torch.FloatTensor(geo_te))
            train_ld = DataLoader(train_ds, batch_size=cfg.BATCH_SIZE, shuffle=True)
            val_ld = DataLoader(val_ds, batch_size=cfg.BATCH_SIZE)
            test_ld = DataLoader(test_ds, batch_size=cfg.BATCH_SIZE)

        n_params = count_params(model)

        save_path = out_dir / f"run_{seed}" / "best.pth"
        save_path.parent.mkdir(parents=True, exist_ok=True)

        try:
            if is_external:
                # 2026-09-20 修复（Bug 2）：cmp_train_model 返回 5 值
                # （model, train_time, tr_losses, va_losses, best_ep），此前用
                # `*_` 把 losses 丢弃，导致下方 np.save(losses.npy) 必然 NameError。
                model, train_time, tr_losses, va_losses, _ = cmp_train_model(
                    cmp_cfg, model, train_ld, val_ld, save_path,
                    model_tag=tag, seed=seed,
                    # Ours 框架基线（V-Attention / V-MLP）沿用对比实验的 LR 分组
                    ours_lr_groups=(model_name in CMP_OURS_FRAMEWORK_MODELS))
                pred_cum, true_cum = cmp_evaluate(
                    model, test_ld, init_test, scY, pca_y, hz, cfg.DEVICE,
                    true_diff_win=data['yte_raw'])
            else:
                model, train_time, tr_losses, va_losses = train_model(
                    cfg, model, train_ld, val_ld, save_path,
                    model_tag=tag, seed=seed, node_mask=valid_mask_t)
                pred_cum, true_cum = evaluate(
                    model, test_ld, init_test, scY, pca_y, hz, cfg.DEVICE,
                    node_mask=valid_mask_t,
                    true_diff_win=data['yte_raw'])
            gm = global_metrics(true_cum, pred_cum)
            gm['seed'] = seed
            gm['Params'] = n_params
            gm['TrainTime'] = train_time
            # g 诊断已在 §6.2 中通过 A/B 对照完成，此处不再逐种子日志记录。
            all_global.append(gm)

            with open(save_path.parent / 'metrics.json', 'w') as f:
                json.dump(gm, f, indent=2)
            np.savez(save_path.parent / 'predictions.npz',
                     pred_cum=pred_cum, true_cum=true_cum)
            # 数据导出层（2026-09-18）
            np.save(save_path.parent / 'losses.npy', {'tr': tr_losses, 'va': va_losses})

            logger.info(f"    Seed {si + 1}/{len(cfg.SEEDS)} ({seed}) "
                        f"RMSE(last)={gm['RMSE_last']:.4f} "
                        f"RMSE(full)={gm['RMSE']:.4f} R2(last)={gm['R2_last']:.4f}")
        except Exception as e:
            import traceback as _tb
            logger.error(f"    Seed {seed} failed: {e}\n{_tb.format_exc()}")

        del model; gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    if not all_global:
        logger.error(f"  [{tag}] H={hz}: ALL seeds failed!")
        return None

    df = pd.DataFrame(all_global)
    summary_dir = out_dir / "summary"
    summary_dir.mkdir(exist_ok=True)
    for old_f in summary_dir.glob('global_*runs.csv'):
        old_f.unlink()
    df.to_csv(summary_dir / f'global_{len(all_global)}runs.csv', index=False)
    logger.info(f"  [{tag}] H={hz}: {len(all_global)} seeds done. "
                f"RMSE(last)={df['RMSE_last'].mean():.4f}+/-{df['RMSE_last'].std():.4f} "
                f"RMSE(full)={df['RMSE'].mean():.4f}+/-{df['RMSE'].std():.4f}")
    return df


# =============================================================================
# CLEAN-CONDITION REUSE (issue 4, Plan A)
# =============================================================================
# The three "clean" conditions (noise_0.0 / missing_0.0 / sparse_1) produce
# IDENTICAL training data (no perturbation is applied at these levels), so the
# model is trained once under noise_0.0 and the results are copied to the
# missing_0.0 / sparse_1 directories instead of being re-trained.
def is_clean_condition(stress_type, stress_level):
    return (stress_type == 'noise' and stress_level == 0.0) or \
           (stress_type == 'missing' and stress_level == 0.0) or \
           (stress_type == 'sparse' and stress_level == 1)


def copy_clean_results(cfg, model_name, hz, stype, level):
    """Copy the noise_0.0 results of (model, hz) to another clean condition."""
    import shutil
    src = cfg.OUT_ROOT / f"H{hz}" / "noise_0.0" / model_name
    dst = cfg.OUT_ROOT / f"H{hz}" / f"{stype}_{level}" / model_name
    if not src.exists() or not list((src / "summary").glob('global_*runs.csv')):
        logger.warning(f"  [Clean-copy] source missing for {model_name} H={hz}, "
                       f"cannot copy to {stype}_{level}")
        return False
    if dst.exists():
        shutil.rmtree(dst)
    shutil.copytree(src, dst)
    logger.info(f"  [Clean-copy] {model_name} H={hz}: noise_0.0 -> {stype}_{level}")
    return True


# =============================================================================
# DATA COLLECTION / EXPORT
# =============================================================================
# 绘图已迁出（2026-09-18）：所有图由 可视化/02代码结果可视化/fig_stress.py
# 从落盘数据（all_stress_results.csv / full_seed_data.pkl / cross_horizon_all.csv）
# 读取后生成，改图/加图无需重跑实验。
def collect_all_stress_results(cfg, hz, valid_models=None):
    """Collect all stress results for a given horizon into a dataframe.
    valid_models: optional set of model names to keep (filters out stale model
    directories left over from earlier runs with a different Best-Baseline)."""
    records = []
    conditions_to_scan = [
        ('noise', STRESS_NOISE_LEVELS),
        ('missing', STRESS_MISSING_RATES),
        ('sparse', STRESS_SPARSE_EVERY),
    ]
    for stype, levels in conditions_to_scan:
        for level in levels:
            cond_dir = cfg.OUT_ROOT / f"H{hz}" / f"{stype}_{level}"
            if not cond_dir.exists():
                continue
            for model_dir in cond_dir.iterdir():
                if not model_dir.is_dir():
                    continue
                model = model_dir.name
                if valid_models is not None and model not in valid_models:
                    continue
                summary_dir = model_dir / "summary"
                glob_files = sorted(summary_dir.glob('global_*runs.csv'))
                if not glob_files:
                    continue
                df = pd.read_csv(glob_files[-1])
                for _, row in df.iterrows():
                    records.append({
                        'Horizon': hz,
                        'StressType': stype,
                        'StressLevel': level,
                        'StressLabel': STRESS_LABELS.get(f'{stype}_{level}', f'{stype}_{level}'),
                        'StressShort': STRESS_SHORT.get(f'{stype}_{level}', str(level)),
                        'Model': model,
                        'ModelLabel': MODEL_LABELS.get(model, model),
                        'RMSE': row['RMSE'],
                        'MAE': row['MAE'],
                        'R2': row['R2'],
                        'MaxErr': row['MaxErr'],
                        # 2026-09-20：末步口径（论文主口径）一并带出
                        'RMSE_last': row['RMSE_last'],
                        'MAE_last': row['MAE_last'],
                        'R2_last': row['R2_last'],
                        'MaxErr_last': row['MaxErr_last'],
                        'seed': row['seed'],
                    })
    return pd.DataFrame(records)


def compute_defense(df_stress, hz):
    """Compute DEFENSE = δ_V7 − δ_V5（各自以自身干净条件为基准，论文 §4.3 口径）。"""
    base_model = get_defense_pair(df_stress)
    if base_model is None:
        return pd.DataFrame()
    ours_clean = get_model_clean_rmse(df_stress, FULL_MODEL, fallback=np.nan)
    base_clean = get_model_clean_rmse(df_stress, base_model, fallback=np.nan)
    if not np.isfinite(ours_clean) or not np.isfinite(base_clean):
        return pd.DataFrame()

    defense_rows = []
    for stype in ['noise', 'missing', 'sparse']:
        for level in df_stress[df_stress['StressType'] == stype]['StressLevel'].unique():
            ours_sub = df_stress[(df_stress['Model'] == FULL_MODEL) &
                                 (df_stress['StressType'] == stype) &
                                 (df_stress['StressLevel'] == level)]
            base_sub = df_stress[(df_stress['Model'] == base_model) &
                                 (df_stress['StressType'] == stype) &
                                 (df_stress['StressLevel'] == level)]
            if ours_sub.empty or base_sub.empty:
                continue
            ours_rmse = ours_sub['RMSE_last'].mean()
            base_rmse = base_sub['RMSE_last'].mean()
            ours_deg = (ours_rmse - ours_clean) / (ours_clean + 1e-8) * 100
            base_deg = (base_rmse - base_clean) / (base_clean + 1e-8) * 100
            defense_rows.append({
                'StressType': stype, 'StressLevel': level,
                'StressLabel': STRESS_LABELS.get(f'{stype}_{level}', f'{stype}_{level}'),
                'Ours_RMSE': ours_rmse, 'Baseline_RMSE': base_rmse,
                'Ours_deg_pct': ours_deg, 'Baseline_deg_pct': base_deg,
                'DEFENSE_pct': base_deg - ours_deg
            })
    return pd.DataFrame(defense_rows)


# ---- Shared helpers for plots (issue 8: per-model clean baseline) ----
def get_model_clean_rmse(df_stress, model, fallback=1.0):
    """Clean (noise_0.0) RMSE of the model itself, used as the reference for
    ITS OWN degradation rate. All clean conditions are trained identically
    (see copy_clean_results), so noise_0.0 is a valid clean reference for
    every stress type."""
    sub = df_stress[(df_stress['Model'] == model) &
                    (df_stress['StressType'] == 'noise') &
                    (df_stress['StressLevel'] == 0.0)]
    return sub['RMSE_last'].mean() if not sub.empty else fallback


def get_defense_pair(df_stress):
    """DEFENSE 的对照模型固定为去竖向 GCN 变体 V7（论文 §4.3 口径 δ_V7−δ_V5）。
    兜底：V7 不在数据中（如旧 2 模型残留数据）时，退回非 Ours 中干净条件
    RMSE 最低者，保证旧图脚本仍可渲染。"""
    if V7_MODEL in set(df_stress['Model'].unique()):
        return V7_MODEL
    others = [m for m in df_stress['Model'].unique() if m != FULL_MODEL]
    if not others:
        return None
    if len(others) > 1:
        others.sort(key=lambda m: get_model_clean_rmse(df_stress, m,
                                                       fallback=float('inf')))
    # 2026-09-22：兜底路径存在语义污染风险——V7 缺失时可能选中外部基线（Best-Baseline
    # 的干净 RMSE 可能低于 V5/V0），DEFENSE 会由"竖向 GCN 贡献"变成"基线与 V5 的差异"。
    # STRESS_MODELS 必含 V7，走到这里即说明数据不完整，须人工确认。
    logger.warning(f"  [DEFENSE] 数据中未找到 {V7_MODEL}，兜底选用 '{others[0]}'"
                   " —— DEFENSE 语义可能已污染，请确认数据完整性。")
    return others[0]


def get_plot_models(df_stress):
    """Models to plot: ours first, then the auto-detected Best-Baseline."""
    present = set(df_stress['Model'].unique())
    models = [m for m in STRESS_MODELS if m in present]
    models += [m for m in sorted(present) if m not in STRESS_MODELS]
    return models


def export_stress_results(df_stress, hz, save_dir):
    """数据落盘层（2026-09-18）：逐种子明细宽表 + CSV 汇总。
    绘图改由 可视化/02代码结果可视化/fig_stress.py 从这些文件读取后生成。"""
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    df_stress.to_csv(save_dir / 'all_stress_results.csv', index=False)
    df_stress.to_pickle(save_dir / 'full_seed_data.pkl')
    logger.info(f"  H={hz}: {len(df_stress)} rows -> "
                f"all_stress_results.csv + full_seed_data.pkl")


# =============================================================================
# MAIN DRIVER
# =============================================================================
def find_best_baseline(cfg):
    """Find the best EXTERNAL baseline from comparison results.

    以论文主展示步长 H=10 的 RMSE 为准，**只在真正的外部方法中挑选**：排除全模块版
    （V5_Ours）、消融复用行（V0_TCN-Only）与本文框架变体（V-Attention / V-MLP，
    与 Ours 共享骨干，见 NON_EXTERNAL_MODELS）。

    2026-09-21 修正：此前仅排除前两者，导致骨干同源的 V-Attention 因 RMSE 低于
    LSTM/STGCN 而被选中，使压力测试的"外部基线对照"名不副实。"""
    summary_f = cfg.COMPARISON_ROOT / "cross_horizon_comparison" / "cross_horizon_summary.csv"
    if not summary_f.exists():
        logger.warning("Comparison summary not found, skipping Best-Baseline detection")
        return None
    df = pd.read_csv(summary_f)
    df_h = df[df['Horizon'] == 10]
    non_ours = df_h[~df_h['Model'].isin(NON_EXTERNAL_MODELS)]
    if non_ours.empty:
        return None
    # 2026-09-20：以末步口径（论文主口径）挑选最强基线；旧 CSV 无该列时回退全步。
    rmse_col = 'RMSE_last_mean' if 'RMSE_last_mean' in df.columns else 'RMSE_mean'
    best = non_ours.loc[non_ours[rmse_col].idxmin()]
    logger.info(f"  Best Baseline detected: {best['ModelLabel']} "
                f"(H=10, {rmse_col}={best[rmse_col]:.4f})")
    return best['Model']


def main(quick_test=False):
    cfg = Config()

    if quick_test:
        cfg.HORIZONS = [6]
        cfg.SEEDS = cfg.SEEDS[:1]
        cfg.EPOCHS = 5
        cfg.PATIENCE = 3
        # 2026-09-22：quick 模式隔离输出目录（与 PCA_Experiment 同款）。否则 1 个种子的
        # run_{seed} 四件套会留在正式 OUT_ROOT 下，被 STRESS_RESUME=1 的种子级 skip 命中，
        # 使该条件永久只有 1 个种子。
        cfg.OUT_ROOT = cfg.OUT_ROOT / "_quick"
        logger.info("=" * 60)
        logger.info("QUICK TEST MODE: H=6, 1 seed, 5 epochs")
        logger.info(f"QUICK TEST OUTPUT (isolated): {cfg.OUT_ROOT}")
        logger.info("=" * 60)
    else:
        logger.info("=" * 60)
        logger.info("FULL STRESS TEST EXPERIMENT")
        logger.info(f"Models: {', '.join(STRESS_MODELS)} + 固定双外部基线 "
                    f"{' & '.join(STRESS_EXTERNAL_BASELINES)}（方案C）")
        logger.info(f"Horizons: {cfg.HORIZONS}")
        logger.info(f"Seeds: {len(cfg.SEEDS)}")
        n_slots = (len(STRESS_NOISE_LEVELS) + len(STRESS_MISSING_RATES) +
                   len(STRESS_SPARSE_EVERY))          # 14 个"类型×档位"槽位
        n_perturbed = n_slots - 3                     # 11 个扰动条件
        # 2026-09-22 修正：实际训练 12 个条件（扰动 11 + clean 的 noise_0.0 训练 1 次，
        # missing_0.0 / sparse_1 由 copy_clean_results 复制）。原式按 14 槽位估算会高估到 8400。
        n_trained = n_perturbed + 1                   # = 12 ⇒ 12×4×5×30 = 7200
        # 2026-09-25 方案 C：n_models 不再依赖 find_best_baseline，固定 3 内部 + 2 外部
        n_models = len(STRESS_MODELS) + len(STRESS_EXTERNAL_BASELINES)
        total = n_models * len(cfg.HORIZONS) * n_trained * len(cfg.SEEDS)
        perturbed_total = n_models * len(cfg.HORIZONS) * n_perturbed * len(cfg.SEEDS)
        logger.info(f"Total training runs: {total} "
                    f"(perturbed: {perturbed_total}; clean trained once per model/horizon, "
                    f"then copied to the other 2 clean slots)")
        logger.info("=" * 60)
        # 2026-09-25 防误触保险（硬性约束：不得覆盖旧结果）：仅在**非 RESUME** 模式下
        # 检查——"直接点运行"（无 STRESS_RESUME=1）会把 V5/V7/V0/BiLSTM-Attn 的全部
        # run_{seed} 原地重训覆盖。检测到既有结果目录即终止：补训新基线（方案C）请设
        # $env:STRESS_RESUME="1" 后重跑；确要全新覆盖请显式设 STRESS_FORCE_NEW=1。
        # quick 模式输出隔离在 _quick，不受影响。
        # （2026-09-25 修复：初版漏判 cfg.RESUME，导致 RESUME=1 正常补训也被拦截）
        if not cfg.RESUME and os.environ.get('STRESS_FORCE_NEW', '0') != '1':
            _existing = sorted(d.name for d in cfg.OUT_ROOT.glob('H*') if d.is_dir())
            if _existing:
                logger.error(
                    f"  检测到既有压力测试结果目录（{', '.join(_existing)}），"
                    "但未设置 STRESS_RESUME=1 —— 直接重跑会原地覆盖旧 run_{seed} 结果，已终止。"
                    "补训新基线（方案C）：$env:STRESS_RESUME=\"1\" 后重跑；"
                    "确要全新覆盖：$env:STRESS_FORCE_NEW=\"1\"。")
                sys.exit(1)

    out_root = cfg.OUT_ROOT
    logger.info(f"Device: {cfg.DEVICE}")
    logger.info(f"Output: {out_root}")

    set_seed(42)
    full_df = load_raw_data_with_meta(cfg)
    # 2026-09-20：`unique()` 顺序继承自原表（按测点排序），不是时间序，与特征矩阵的
    # pivot.sort_index() 不一致会使导出时间戳错位。统一排序。
    ts_arr = np.sort(full_df['测量时间'].unique())
    wd, wa = load_water_data(ts_arr, cfg)
    water_feat = extract_water_features(wd, wa)
    disp_raw, diff1, meta = build_node_features(full_df, cfg)
    valid_mask = meta.get('valid_mask', np.ones(disp_raw.shape[1], dtype=np.float32))
    geo_records = load_geology_params(cfg)
    geo_feat_raw = assign_geo_to_nodes(geo_records, meta['depths'], cfg) if geo_records else None
    n = len(ts_arr)
    tr_end = int(n * cfg.TRAIN_RATIO)
    va_end = int(n * (cfg.TRAIN_RATIO + cfg.VAL_RATIO))
    train_idx = np.arange(tr_end)
    val_idx = np.arange(tr_end, va_end)
    test_idx = np.arange(va_end, n)
    train_disp = disp_raw[train_idx]
    val_disp = disp_raw[val_idx]
    test_disp = disp_raw[test_idx]
    train_den, val_den, test_den = denoise_split(train_disp, val_disp, test_disp, cfg)
    all_den = np.vstack([train_den, val_den, test_den])
    diff1_den = np.zeros_like(all_den)
    if len(ts_arr) > 1:
        diff1_den[1:] = np.diff(all_den, axis=0)
    geo_static_raw = np.concatenate([
        geo_feat_raw if geo_feat_raw is not None else np.zeros((len(meta['depths']), 6), dtype=np.float32),
        (meta['depths'] / 16.0).reshape(-1, 1)], axis=1)
    geo_static = (geo_static_raw - geo_static_raw.mean(0)) / (geo_static_raw.std(0) + 1e-6)
    # Issue 2: geology stays OUT of the input features (use_geo=False, F_in=4),
    # unified with the ablation experiment; geology acts through the adjacency
    # matrix / physics gate / cross-node attention only.
    train_feat = make_features(train_den, diff1_den[train_idx], meta, water_feat[train_idx], geo_static, use_geo=False)
    val_feat = make_features(val_den, diff1_den[val_idx], meta, water_feat[val_idx], geo_static, use_geo=False)
    test_feat = make_features(test_den, diff1_den[test_idx], meta, water_feat[test_idx], geo_static, use_geo=False)
    y_train = diff1_den[train_idx]
    y_val = diff1_den[val_idx]
    y_test = diff1_den[test_idx]
    pca_y = PCA(0.90).fit(y_train)
    out_dim_base = pca_y.n_components_
    scY = StandardScaler().fit(pca_y.transform(y_train))
    ytr_norm = scY.transform(pca_y.transform(y_train))
    yva_norm = scY.transform(pca_y.transform(y_val))
    yte_norm = scY.transform(pca_y.transform(y_test))
    geo_fb = geo_feat_raw if geo_feat_raw is not None else np.zeros((len(meta['depths']), 6), dtype=np.float32)

    shared = {
        'train_feat': train_feat, 'val_feat': val_feat, 'test_feat': test_feat,
        'y_train': y_train, 'y_val': y_val, 'y_test': y_test,
        'train_idx': train_idx, 'val_idx': val_idx, 'test_idx': test_idx,
        'ts_arr': ts_arr, 'meta': meta,
        'geo_static': geo_static, 'geo_fb': geo_fb,
        'F_in': train_feat.shape[-1], 'N': train_feat.shape[1],
        'test_den': test_den, 'valid_mask': valid_mask,
        'pca_y': pca_y, 'scY': scY, 'out_dim': out_dim_base,
        'ytr_norm': ytr_norm, 'yva_norm': yva_norm, 'yte_norm': yte_norm,
        # Raw water features — 路径 A 后仅作占位透传，不再进入模型。
        'water_feat': water_feat,
    }

    # 2026-09-25 方案 C：5 个模型 = V5/V7/V0 内部变体（STRESS_MODELS）+ 固定双外部
    # 基线 BiLSTM-Attn & GraphWaveNet（STRESS_EXTERNAL_BASELINES）。DEFENSE = δ_V7 − δ_V5
    # 不依赖基线；外部基线服务 §4.3 的稳健性对照。BiLSTM-Attn 旧结果（宽度 48 口径）
    # 在 STRESS_RESUME=1 下按种子级四件套校验原样跳过；GraphWaveNet 为 w64 容量对齐
    # 后的最强外部基线，经对比实验同款工厂补训（宽度 64）。基线集合不再依赖
    # find_best_baseline 自动检测（该函数保留仅供历史追溯）。
    active_models = list(STRESS_MODELS)
    for baseline in STRESS_EXTERNAL_BASELINES:
        if baseline not in active_models:
            active_models.append(baseline)
            logger.info(f"  External baseline added to stress run: {baseline}")
        else:
            logger.warning(f"  External baseline {baseline} is already in STRESS_MODELS")

    all_horizon_data = {}
    conditions_to_run = [
        ('noise', STRESS_NOISE_LEVELS),
        ('missing', STRESS_MISSING_RATES),
        ('sparse', STRESS_SPARSE_EVERY),
    ]

    for hz in cfg.HORIZONS:
        logger.info(f"\n{'#' * 60}\n#  HORIZON H={hz}\n{'#' * 60}")

        for model_name in active_models:
            for stype, levels in conditions_to_run:
                for level in levels:
                    # Clean conditions: noise_0.0 is trained; missing_0.0 and
                    # sparse_1 reuse its results (identical training data,
                    # see copy_clean_results -- issue 4 Plan A + L-2).
                    if is_clean_condition(stype, level) and not (stype == 'noise' and level == 0.0):
                        copy_clean_results(cfg, model_name, hz, stype, level)
                        continue
                    df_s = run_stress_single(cfg, model_name, shared, hz, stype, level)
                    if df_s is not None:
                        df_s['Model'] = model_name
                        df_s['StressType'] = stype
                        df_s['StressLevel'] = level

        # valid_models filter avoids collecting stale model directories
        # (e.g. a previous run with a different Best-Baseline).
        df_all = collect_all_stress_results(cfg, hz, valid_models=set(active_models))
        if not df_all.empty:
            df_all['Horizon'] = hz
            all_horizon_data[hz] = df_all
            vis_dir = out_root / f"H{hz}" / "stress_visualization"
            export_stress_results(df_all, hz, vis_dir)
            logger.info(f"  H={hz}: {len(df_all)} result rows collected")

    if len(all_horizon_data) >= 2:
        logger.info(f"\n{'#' * 60}\n#  CROSS-HORIZON ANALYSIS\n{'#' * 60}")
        cross_dir = out_root / "cross_horizon_comparison"
        cross_dir.mkdir(exist_ok=True)

        all_records = []
        for hz, df in all_horizon_data.items():
            all_records.append(df)
        df_all_hz = pd.concat(all_records, ignore_index=True)
        df_all_hz.to_csv(cross_dir / 'cross_horizon_all.csv', index=False)
        logger.info(f"  Cross-horizon: all_data.csv saved")

    logger.info(f"\n{'=' * 60}")
    logger.info(f"ALL DONE! Results saved to: {out_root}")
    logger.info(f"{'=' * 60}")


if __name__ == '__main__':
    if '--regenerate' in sys.argv or '-r' in sys.argv:
        logger.info("REGENERATE MODE: collecting existing results + re-exporting data "
                    "(绘图请运行 可视化/02代码结果可视化/fig_stress.py)")
        cfg = Config()
        out_root = cfg.OUT_ROOT
        all_horizon_data = {}
        for hz in cfg.HORIZONS:
            logger.info(f"Collecting H={hz}...")
            df_all = collect_all_stress_results(cfg, hz)
            if not df_all.empty:
                all_horizon_data[hz] = df_all
                vis_dir = out_root / f"H{hz}" / "stress_visualization"
                export_stress_results(df_all, hz, vis_dir)
                logger.info(f"  H={hz}: {len(df_all)} rows re-exported")
        if len(all_horizon_data) >= 2:
            cross_dir = out_root / "cross_horizon_comparison"
            cross_dir.mkdir(exist_ok=True)
            all_records = []
            for hz, df in all_horizon_data.items():
                all_records.append(df)
            df_all_hz = pd.concat(all_records, ignore_index=True)
            df_all_hz.to_csv(cross_dir / 'cross_horizon_all.csv', index=False)
            logger.info(f"  Cross-horizon: all_data.csv regenerated")
        logger.info("Data re-export complete!")
        sys.exit(0)

    quick = '--quick' in sys.argv or '-q' in sys.argv
    main(quick_test=quick)