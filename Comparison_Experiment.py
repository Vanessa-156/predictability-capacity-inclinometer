# =============================================================================
# Comparison_Experiment.py -- Chapter 4 Baseline Comparison for MST Journal
# 7 externally trained models x 3 Horizons x 15 Seeds = 315 runs
# (+ V0 / V5 loaded from the ablation results, HORIZONS unified to [2,6,12])
#
# Models:
#   1. LSTM             -- classic temporal baseline
#   2. BiLSTM-Attn      -- bidirectional LSTM + attention
#   3. Transformer      -- pure self-attention baseline
#   4. V-Attention      -- Ours framework, self-attn replacing Vertical GCN
#   5. STGCN            -- spatiotemporal graph convolution
#   6. GraphWaveNet     -- adaptive graph + dilated TCN
#   7. V-MLP            -- Ours framework, depth-mixing MLP replacing Vertical GCN
#   8. TCN-Only         -- = V0 from ablation (reuse)
#   9. Ours             -- = V7 from ablation (reuse)
#
# Architecture note (unified with Ablation_Experiment.py):
#   - Graph-model input features exclude the static geology channels
#     (F_in = 4), geology enters via adjacency / physics gate / cross-node
#     attention only.
#   - 路径 A（2026-09-17）：水文水位不再进入任何图模型。VICD 诊断（正文 §3.4）
#     表明水位与位移在增量口径无可用信息，故输入特征由 12 维降为 4 维。
#     water_feat 仍按占位方式透传以保持 DataLoader / forward 签名不变。
#
# SCI Standards:
#   - Colorblind-friendly palette (Wong 2011) + distinct markers/line styles
#   - No information conveyed by color alone
#   - Times New Roman, 300+ DPI, vector PDF output
#   - Complete statistical testing (paired t-test + Cohen's d)
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
from mpl_toolkits.axes_grid1 import make_axes_locatable
import seaborn as sns
from scipy.stats import ttest_rel
from scipy.spatial.distance import cdist
from scipy.interpolate import interp1d
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, max_error
from sklearn.multioutput import MultiOutputRegressor
import pywt
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import random

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
    # 2026-09-20：`Ablation_New` 是旧 12 维（含水）架构目录，已作废；V0/V5 复用行
    # 必须取自当前 4 维架构的 `Ablation_New_5d`，否则对比表里 Ours/TCN-Only 两行会
    # 悄悄变成旧架构数字（且与其它表口径不一致）。
    ABLATION_ROOT = AUX_ROOT / "Ablation_New_5d"
    OUT_ROOT = AUX_ROOT / "Comparison_Results_w64"
    OUT_ROOT.mkdir(parents=True, exist_ok=True)

    POINTS = ['CX2', 'CX3', 'CX8', 'CX13', 'CX15']
    DEPTHS = np.arange(0.5, 16.0, 0.5)
    LOOKBACK = 12
    HORIZONS = [2, 6, 8, 10, 12]  # unified across all four experiments (issue 9)
    TRAIN_RATIO = 0.70
    VAL_RATIO = 0.15

    BATCH_SIZE = 32
    # 论文口径 400/40；环境变量仅用于示例数据上的缩小规模演示，默认行为不变。
    EPOCHS = int(os.environ.get('ABLATION_EPOCHS', '400'))
    PATIENCE = int(os.environ.get('ABLATION_PATIENCE', '40'))
    LR = 0.001
    LR_GCN = 0.0005
    WD = 1e-3          # 与论文 §4.3 及消融/PCA/压力脚本一致（此前误设为 8e-3）
    GRAD_CLIP = 1.0

    # 2026-09-24（P1 容量对齐，交接 §6.2 方案 C）：外部深度基线隐藏宽度 48→64，
    # 与本文骨干（OURS_GCN_HIDDEN=64）对齐，消除 §6.3 声明的容量差异。
    # ⚠️ HIDDEN_DIM 原为双重使用：既是基线网络宽度，又是 L694 纯时序基线展平输入的
    # PCA 降维维数（GBDT 同路）。已拆分：
    #   - TEMPORAL_PCA_DIM=48  —— 纯时序基线（LSTM/BiLSTM-Attn/Transformer）与 GBDT
    #     的展平输入 PCA 维，**保持 48 不变**（输入形式不随宽度对齐改变，750 次重跑中
    #     GBDT 无需重跑的原因）。
    #   - HIDDEN_DIM=64        —— LSTM/BiLSTM-Attn/Transformer 的网络宽度。
    HIDDEN_DIM = 64
    TEMPORAL_PCA_DIM = 48
    N_LAYERS = 1
    N_HEADS = 2
    DROPOUT = 0.5      # 与论文 §4.3 及消融脚本一致（此前误设为 0.6）

    STGCN_CHEB_K = 2
    STGCN_BLOCKS = 1
    STGCN_TEMP_KERNEL = 3

    GWNET_DIFF_STEPS = 1
    GWNET_LAYERS = 2
    GWNET_KERNEL = 3
    GWNET_DILATIONS = [1, 2]
    GWNET_ADP_DIM = 10

    # 外部图基线（STGCN / GraphWaveNet）的隐藏宽度。原 48 为其原始设定；
    # 2026-09-24 P1 容量对齐改为 64（与本文骨干一致），随 750 次重跑生效。
    GRAPH_HIDDEN_DIM = 64
    # Ours 框架变体（V-Attention / V-MLP）的 GCN 隐藏宽度：与 Ablation_Experiment
    # 的 V5 一致，保证"仅替换竖向量模块"是同等容量的公平对比。
    OURS_GCN_HIDDEN = 64
    GEO_FEAT_DIM = 6
    # 式(9) 的纯几何高斯核之上叠加的两处结构性先验（与消融脚本、正文 §2.3 一致）
    SPA_WITHIN_HOLE_PRIOR = True   # 同孔相邻深度边权重 0.8（结构连续性）
    SPA_SHALLOW_PRIOR = True       # 浅层(<3 m)跨断面节点对 0.6，并豁免 3σ 剪枝

    WAVELET = 'db4'
    WAVELET_LEVEL = 1
    THRESH_MODE = 'soft'

    # 以下三项仅用于 Ours 框架变体（V-Attention / V-MLP），取值与消融 V5 一致
    TCN_CHANNELS = [64, 64]
    TCN_KERNEL_SIZE = 7
    TCN_DILATIONS = [1, 2]
    TCN_DROPOUT = 0.3   # 与论文 §4.3 及消融脚本一致（此前误设为 0.5）
    ATTENTION_HIDDEN = 32

    # 30 个种子：与 Ablation/PCA/Stress 完全一致，见 _gen_seeds.py
    SEEDS = [42, 15, 77, 85, 120, 532, 901, 23, 12, 14, 19, 75, 250, 140, 135,
             1180, 1262, 1284, 605, 1159, 746, 1657, 61, 27, 993, 897, 276,
             1308, 896, 1866]
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # 2026-09-20：`AMP`（torch.cuda.amp 自动混合精度）从未接入训练循环，已删除。
    # 2026-09-21：断点续跑，与 Ablation 的 ABLATION_RESUME 同款——置 1 时跳过已完整
    # 产出四件套（metrics.json/best.pth/predictions.npz/losses.npy）的种子，并读回其
    # 结果参与汇总，避免中断后重训已完成种子。
    RESUME = os.environ.get('COMPARISON_RESUME', '0') == '1'


# =============================================================================
# SCI COLORBLIND-FRIENDLY PALETTE + MARKERS + LINE STYLES
# Wong (2011) Nature Methods palette -- 7 colors, extended to 10
# =============================================================================
WONG_PALETTE = [
    '#0072B2',  # blue
    '#D55E00',  # vermillion
    '#009E73',  # green
    '#F0E442',  # yellow
    '#CC79A7',  # purple
    '#56B4E9',  # sky blue
    '#E69F00',  # orange
    '#000000',  # black
    '#882255',  # dark red (extended)
    '#666666',  # gray (extended 2026-09-21: GBDT 非神经网络对照)
]
MARKER_STYLES = ['o', 's', '^', 'D', 'v', 'p', 'h', '*', 'P', 'X']
LINE_STYLES = ['-', '--', '-.', ':', '-', '--', '-.', ':', '-', '-']

COMPARISON_MODELS = [
    'LSTM',
    'BiLSTM-Attn',
    'Transformer',
    'V-Attention',
    'STGCN',
    'GraphWaveNet',
    'V-MLP',
    'V0_TCN-Only',
    'V7_Ours',
    # 2026-09-21：非深度学习的梯度提升树对照（回答"为何不用机器学习"）。
    # 刻意置于列表**末尾**，以保证既有 9 个模型的配色/标记/线型索引完全不变
    # （图 10/11 与已有配色不发生任何位移）。
    'GBDT',
]

# 2026-09-21：`IS_EXTERNAL` 的语义是「该模型是否需要在本脚本中独立训练」，
# 而不是「是否属于外部方法族」：V0/V5 复用消融结果（False），其余全部独立训练（True）。
# 注意 V-Attention / V-MLP 虽取值 True，但它们是 **Ours 框架变体**（骨干与 V5 完全相同，
# 仅替换竖向量模块），方法族归属见 MODEL_FAMILY —— 论文表 4/5 不得把它们与外部基线
# 混排解读，否则会把「Ours 内部消融式替换」误读成「外部方法对比」。
IS_EXTERNAL = {
    'LSTM': True, 'BiLSTM-Attn': True, 'Transformer': True,
    'V-Attention': True, 'STGCN': True, 'GraphWaveNet': True,
    'V-MLP': True, 'V0_TCN-Only': False, 'V7_Ours': False,
    'GBDT': True,
}

MODEL_LABELS = {
    'LSTM': 'LSTM',
    'BiLSTM-Attn': 'BiLSTM-Attn',
    'Transformer': 'Transformer',
    'V-Attention': 'V-Attention',
    'STGCN': 'STGCN',
    'GraphWaveNet': 'GraphWNet',
    'V-MLP': 'V-MLP',
    'V0_TCN-Only': 'TCN-Only',
    'V7_Ours': 'Ours',
    'GBDT': 'GBDT',
}

MODEL_COLORS = {m: WONG_PALETTE[i] for i, m in enumerate(COMPARISON_MODELS)}
MODEL_MARKERS = {m: MARKER_STYLES[i] for i, m in enumerate(COMPARISON_MODELS)}
MODEL_LINESTYLES = {m: LINE_STYLES[i] for i, m in enumerate(COMPARISON_MODELS)}

MODEL_TYPES = {
    'LSTM': 'temporal', 'BiLSTM-Attn': 'temporal', 'Transformer': 'temporal',
    'V-Attention': 'graph', 'STGCN': 'graph', 'GraphWaveNet': 'graph',
    'V-MLP': 'graph', 'V0_TCN-Only': 'graph', 'V7_Ours': 'graph',
    # 2026-09-21：GBDT 走 temporal 路径（展平 + PCA 至 48 维），与 LSTM 系列同一份输入
    'GBDT': 'temporal',
}

# "Ours 框架"变体：骨干（TCN 宽度/GCN 宽度/水位注入/物理门/跨节点注意力）与消融
# V5 完全相同，只替换竖向量模块 —— 因此它们使用与消融一致的骨干尺寸与 LR 分组。
OURS_FRAMEWORK_MODELS = {'V-Attention', 'V-MLP'}

# GBDT 仅在主展示步长上评估：树模型参数量不随隐藏维度缩放，且主展示步长已足以支撑
# "非深度学习方法不适用"这一结论（见论文 4.3 节）。
GBDT_HORIZON = 10

# 方法族标记（2026-09-21）：供对比表分组标注使用，随汇总表落盘（Family 列），
# 使论文表 4/5 可以直接按族分组呈现，避免"共享 Ours 骨干的内部替换"与
# "外部基线"混排造成的公平性误读。
MODEL_FAMILY = {
    'LSTM': 'external_temporal',
    'BiLSTM-Attn': 'external_temporal',
    'Transformer': 'external_temporal',
    'STGCN': 'external_graph',
    'GraphWaveNet': 'external_graph',
    'V-Attention': 'ours_framework',
    'V-MLP': 'ours_framework',
    'V0_TCN-Only': 'ablation',
    'V7_Ours': 'ours',
    'GBDT': 'external_ml',
}
MODEL_FAMILY_LABELS = {
    'ours': '本文模型',
    'ours_framework': 'Ours 框架变体（共享骨干，仅替换竖向量模块）',
    'external_temporal': '外部时序基线',
    'external_graph': '外部图基线（隐藏宽度 64）',
    'ablation': '消融对照（无耦合）',
    'external_ml': '外部 ML 基线',
}

plt.rcParams.update({
    'font.family': 'serif', 'font.serif': ['Times New Roman'],
    'font.size': 15, 'axes.labelsize': 17, 'axes.titlesize': 18,
    'legend.fontsize': 14, 'figure.dpi': 150, 'savefig.dpi': 600,
    'savefig.bbox': 'tight', 'axes.grid': True, 'grid.alpha': 0.25
})
sns.set_style("whitegrid")


# =============================================================================
# UTILITY FUNCTIONS
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

    # 式(9) 之上叠加的两处结构性先验，由配置开关控制（与消融脚本一致）：
    #   1) 同一测斜孔内相邻深度取 0.8 —— 结构沿深度方向的材料连续性；
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

    # 3σ 距离剪枝；浅层跨断面边显式豁免（浅层相邻断面间距本身接近 σ）
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
    adj_norm = (d_inv_sqrt[:, None] * adj * d_inv_sqrt[None, :]).astype(np.float32)
    laplacian = np.eye(N) - adj_norm
    eigvals = np.linalg.eigvalsh(laplacian)
    lambda_max = float(np.max(eigvals))
    L_scaled = 2.0 * laplacian / max(lambda_max, 2.0) - np.eye(N)
    return adj_norm, L_scaled.astype(np.float32)


def denoise_split(train, val, test, cfg):
    N = train.shape[1]
    tr_d = np.zeros_like(train)
    v_d = np.zeros_like(val)
    te_d = np.zeros_like(test)
    for i in range(N):
        tr_d[:, i] = wavelet_denoise(train[:, i], cfg.WAVELET,
                                     cfg.WAVELET_LEVEL, cfg.THRESH_MODE)
        v_d[:, i] = wavelet_denoise(val[:, i], cfg.WAVELET,
                                    cfg.WAVELET_LEVEL, cfg.THRESH_MODE)
        te_d[:, i] = wavelet_denoise(test[:, i], cfg.WAVELET,
                                     cfg.WAVELET_LEVEL, cfg.THRESH_MODE)
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
    # 路径 A（2026-09-17）：彻底移除水文输入。此前 use_water=True 会把 8 维水位
    # 直接拼进 TCN 基准输入，而 VICD 诊断（正文 §3.4）表明水位与位移在增量口径
    # 无可用信息，故不再作为输入。输入特征由 12 维降为 4 维。
    # wf 入参仅作签名占位以兼容既有调用。
    base = np.concatenate(feat_list, axis=-1)
    if use_geo and geo_static is not None:
        geo_tile = np.tile(geo_static[np.newaxis, :, :], (T, 1, 1))
        return np.concatenate([base, geo_tile], axis=-1).astype(np.float32)
    return base.astype(np.float32)


def prepare_shared_data(cfg):
    set_seed(42)
    full_df = load_raw_data_with_meta(cfg)
    # 2026-09-20：`np.sort`。`unique()` 的返回顺序依赖输入行序，一旦原始数据行序
    # 变化，时间轴会与 train/val/test 切分错位（切分按位置索引）。显式排序消除该
    # 隐患（与 Ablation/PCA/Stress 统一）。
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
    if len(meta['timestamps']) > 1:
        diff1_den[1:] = np.diff(all_den, axis=0)
    geo_static_raw = np.concatenate([
        geo_feat_raw if geo_feat_raw is not None else np.zeros((len(meta['depths']), 6), dtype=np.float32),
        (meta['depths'] / 16.0).reshape(-1, 1)], axis=1)
    geo_static = (geo_static_raw - geo_static_raw.mean(0)) / (geo_static_raw.std(0) + 1e-6)
    train_feat = make_features(train_den, diff1_den[train_idx], meta, water_feat[train_idx], geo_static)
    val_feat = make_features(val_den, diff1_den[val_idx], meta, water_feat[val_idx], geo_static)
    test_feat = make_features(test_den, diff1_den[test_idx], meta, water_feat[test_idx], geo_static)

    train_feat_nogeo = make_features(train_den, diff1_den[train_idx], meta, water_feat[train_idx], None, use_geo=False)
    val_feat_nogeo = make_features(val_den, diff1_den[val_idx], meta, water_feat[val_idx], None, use_geo=False)
    test_feat_nogeo = make_features(test_den, diff1_den[test_idx], meta, water_feat[test_idx], None, use_geo=False)

    # 注（2026-09-18 审计）：下面的 10 维 train_feat/F_in（含静态地质通道）为
    # 路径 A 之前的历史遗留，当前**没有任何模型消费**——自训模型与外部基线在
    # build_samples_and_pca / build_temporal_samples / 主循环中一律使用
    # *_nogeo / F_in_nogeo（4 维），与消融口径统一（见文件头 Issue 2 注释）。
    # 保留仅为构建 F_in_nogeo 时同处调用 make_features，不影响任何结果。
    F_in = train_feat.shape[-1]
    F_in_nogeo = train_feat_nogeo.shape[-1]
    N = train_feat.shape[1]
    y_train = diff1_den[train_idx]
    y_val = diff1_den[val_idx]
    y_test = diff1_den[test_idx]
    geo_fb = geo_feat_raw if geo_feat_raw is not None else np.zeros((len(meta['depths']), 6), dtype=np.float32)
    adj_norm, L_scaled = build_adjacency(meta, geo_fb, cfg)
    return {
        'train_feat': train_feat, 'val_feat': val_feat, 'test_feat': test_feat,
        'train_feat_nogeo': train_feat_nogeo, 'val_feat_nogeo': val_feat_nogeo,
        'test_feat_nogeo': test_feat_nogeo, 'F_in_nogeo': F_in_nogeo,
        'y_train': y_train, 'y_val': y_val, 'y_test': y_test,
        'train_idx': train_idx, 'val_idx': val_idx, 'test_idx': test_idx,
        'ts_arr': ts_arr, 'meta': meta,
        'adj_norm': adj_norm, 'L_scaled': L_scaled,
        'geo_static': geo_static, 'geo_fb': geo_fb,
        'F_in': F_in, 'N': N,
        'test_den': test_den,
        'valid_mask': valid_mask,
        # Raw water features — 路径 A 后仅作占位透传，不再进入模型。
        'water_feat': water_feat,
    }


def build_samples_and_pca(shared, cfg, hz):
    # Issue 2: graph models use the NO-GEO feature set (F_in = 4), unified
    # with the ablation experiment; geology acts through the adjacency /
    # physics gate / cross-node attention instead of raw input channels.
    train_feat = shared['train_feat_nogeo']
    val_feat = shared['val_feat_nogeo']
    test_feat = shared['test_feat_nogeo']
    y_train = shared['y_train']
    y_val = shared['y_val']
    y_test = shared['y_test']
    test_den = shared['test_den']
    F_in = shared['F_in_nogeo']
    N = shared['N']

    scX = StandardScaler().fit(train_feat.reshape(-1, F_in))
    tr_norm = scX.transform(train_feat.reshape(-1, F_in)).reshape(train_feat.shape)
    va_norm = scX.transform(val_feat.reshape(-1, F_in)).reshape(val_feat.shape)
    te_norm = scX.transform(test_feat.reshape(-1, F_in)).reshape(test_feat.shape)

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
    init_test = test_den[cfg.LOOKBACK - 1:-hz][:Xte.shape[0]]
    # 评估口径（2026-09-20）：测试集真值窗口直接切原始一阶差分 y_test，不经 PCA
    # 投影、不经标准化。窗口 i 的目标行为 test_idx[LOOKBACK+i : LOOKBACK+i+hz]，
    # 与 init_test（起始绝对位移）累加后正好还原该区间被去噪后的绝对位移。
    # 此前真值也做 PCA 往返，等于把真值投影到 K=13 子空间后再比，丢弃的增量方差
    # 及其随累加步数的放大都不计入误差 → 各表 RMSE/R² 系统性偏乐观。
    yte_raw = np.stack([y_test[i + cfg.LOOKBACK:i + cfg.LOOKBACK + hz]
                        for i in range(Xte.shape[0])]) if Xte.shape[0] else np.empty(
        (0, hz, y_test.shape[1]), dtype=np.float32)

    valid_mask = shared['valid_mask']
    invalid_idx = np.where(valid_mask == 0)[0]
    # 2026-09-21：build_node_features 已加网格完整性断言 → valid_mask 恒为全 1，
    # 本块为可证的恒等操作（保留以兼容签名；它也并非真掩码，见消融脚本同类注释）。
    if len(invalid_idx) > 0:
        Xtr[:, :, invalid_idx, :] = 0.0
        Xva[:, :, invalid_idx, :] = 0.0
        Xte[:, :, invalid_idx, :] = 0.0
        init_test[:, invalid_idx] = 0.0

    geo_static_np = shared['geo_static']
    geo_tr = np.tile(geo_static_np[np.newaxis, :, :], (Xtr.shape[0], 1, 1))
    geo_va = np.tile(geo_static_np[np.newaxis, :, :], (Xva.shape[0], 1, 1))
    geo_te = np.tile(geo_static_np[np.newaxis, :, :], (Xte.shape[0], 1, 1))

    return {
        'Xtr': Xtr, 'ytr': ytr, 'Xva': Xva, 'yva': yva, 'Xte': Xte, 'yte': yte,
        'init_test': init_test, 'scX': scX, 'scY': scY, 'pca_y': pca_y,
        'out_dim': out_dim, 'F_in': F_in, 'N': N,
        'geo_tr': geo_tr, 'geo_va': geo_va, 'geo_te': geo_te,
        'yte_raw': yte_raw
    }


def build_temporal_samples(shared, cfg, hz):
    """时间序列基线专用样本构造（LSTM / BiLSTM-Attn / Transformer）。

    2026-09-21（公平性声明）：此处把 (T, N, F) 展平成 (T, N*F) 再经 PCA 降维，
    目的是把多节点观测压成单条定长序列 —— 这是纯时序模型的固有输入形式，
    代价是**丢失节点拓扑**。故它们与 Ours 的差距中天然包含"无空间耦合"这一
    结构性劣势，不宜作为"空间耦合是否有效"的证据。真正干净的无耦合对照是消融的
    **V0_TCN-Only**（保留 (T, N, F) 张量输入，仅去掉图传播与物理门），论文 §5.1 已据此说明。
    """
    # 2026-09-24 P1 拆分：此处是展平输入的 PCA 维（输入侧），原误用 HIDDEN_DIM；
    # 改用 TEMPORAL_PCA_DIM=48 后，宽度对齐（HIDDEN_DIM 48→64）不再改变时序基线
    # 与 GBDT 的输入形式——这正是 GBDT 无需重跑的原因。
    temporal_dim = cfg.TEMPORAL_PCA_DIM
    train_feat = shared['train_feat_nogeo']
    val_feat = shared['val_feat_nogeo']
    test_feat = shared['test_feat_nogeo']
    y_train = shared['y_train']
    y_val = shared['y_val']
    y_test = shared['y_test']
    test_den = shared['test_den']
    N = shared['N']
    F_in = shared['F_in_nogeo']
    T_tr = train_feat.shape[0]
    train_flat = train_feat.reshape(T_tr, N * F_in)
    val_flat = val_feat.reshape(val_feat.shape[0], N * F_in)
    test_flat = test_feat.reshape(test_feat.shape[0], N * F_in)
    flat_dim = N * F_in
    actual_dim = min(temporal_dim, T_tr // 2, flat_dim)
    logger.info(f"  Temporal model: flatten {flat_dim}D -> PCA -> {actual_dim}D")
    pca_in = PCA(n_components=actual_dim).fit(train_flat)
    train_proj = pca_in.transform(train_flat).astype(np.float32)
    val_proj = pca_in.transform(val_flat).astype(np.float32)
    test_proj = pca_in.transform(test_flat).astype(np.float32)
    scX_t = StandardScaler().fit(train_proj)
    tr_norm = scX_t.transform(train_proj)
    va_norm = scX_t.transform(val_proj)
    te_norm = scX_t.transform(test_proj)
    pca_y = PCA(0.90).fit(y_train)
    out_dim = pca_y.n_components_
    ytr_pca = pca_y.transform(y_train)
    yva_pca = pca_y.transform(y_val)
    yte_pca = pca_y.transform(y_test)
    scY = StandardScaler().fit(ytr_pca)
    ytr_norm = scY.transform(ytr_pca)
    yva_norm = scY.transform(yva_pca)
    yte_norm = scY.transform(yte_pca)

    def make_samples_1d(feat, tgt, lb, hval):
        X, y = [], []
        for i in range(len(feat) - lb - hval + 1):
            X.append(feat[i:i + lb])
            y.append(tgt[i + lb:i + lb + hval])
        return np.array(X, dtype=np.float32), np.array(y, dtype=np.float32)

    Xtr, ytr = make_samples_1d(tr_norm, ytr_norm, cfg.LOOKBACK, hz)
    Xva, yva = make_samples_1d(va_norm, yva_norm, cfg.LOOKBACK, hz)
    Xte, yte = make_samples_1d(te_norm, yte_norm, cfg.LOOKBACK, hz)
    init_test = test_den[cfg.LOOKBACK - 1:-hz][:Xte.shape[0]]
    # 评估口径（2026-09-20）：同 build_samples_and_pca，真值取原始一阶差分切片。
    yte_raw = np.stack([y_test[i + cfg.LOOKBACK:i + cfg.LOOKBACK + hz]
                        for i in range(Xte.shape[0])]) if Xte.shape[0] else np.empty(
        (0, hz, y_test.shape[1]), dtype=np.float32)
    return {
        'Xtr': Xtr, 'ytr': ytr, 'Xva': Xva, 'yva': yva, 'Xte': Xte, 'yte': yte,
        'init_test': init_test, 'scX_t': scX_t, 'scY': scY, 'pca_y': pca_y,
        'pca_in': pca_in, 'out_dim': out_dim, 'temporal_in_dim': actual_dim,
        'yte_raw': yte_raw
    }


# =============================================================================
# EXTERNAL MODEL DEFINITIONS
# =============================================================================

# ---------- 1. LSTM ----------
class LSTMForecaster(nn.Module):
    def __init__(self, input_dim, hidden_dim, out_dim, horizon, n_layers=2, dropout=0.3):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, n_layers, batch_first=True, dropout=dropout)
        self.head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LeakyReLU(0.1),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, horizon * out_dim))
        self.horizon = horizon
        self.out_dim = out_dim

    def forward(self, x, *args, **kwargs):
        out, _ = self.lstm(x)
        pred = self.head(out[:, -1, :])
        return pred.reshape(pred.shape[0], self.horizon, self.out_dim)


# ---------- 2. BiLSTM-Attention ----------
class BiLSTMAttentionForecaster(nn.Module):
    def __init__(self, input_dim, hidden_dim, out_dim, horizon, n_layers=2, dropout=0.3):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim // 2, n_layers,
                            batch_first=True, bidirectional=True, dropout=dropout)
        self.attn = nn.Sequential(nn.Linear(hidden_dim, hidden_dim // 2), nn.Tanh(),
                                  nn.Linear(hidden_dim // 2, 1))
        self.head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LeakyReLU(0.1),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, horizon * out_dim))
        self.horizon = horizon
        self.out_dim = out_dim

    def forward(self, x):
        out, _ = self.lstm(x)
        scores = self.attn(out).squeeze(-1)
        w = F.softmax(scores, dim=1)
        ctx = torch.bmm(w.unsqueeze(1), out).squeeze(1)
        pred = self.head(ctx)
        return pred.reshape(pred.shape[0], self.horizon, self.out_dim)


# ---------- 3. Transformer ----------
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=200):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x):
        return x + self.pe[:, :x.size(1), :]


class TransformerForecaster(nn.Module):
    def __init__(self, input_dim, hidden_dim, out_dim, horizon, n_layers=2, n_heads=4, dropout=0.3):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        self.pos_enc = PositionalEncoding(hidden_dim)
        encoder_layer = nn.TransformerEncoderLayer(d_model=hidden_dim, nhead=n_heads,
                                                    dim_feedforward=hidden_dim * 4,
                                                    dropout=dropout, batch_first=True)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LeakyReLU(0.1),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, horizon * out_dim))
        self.horizon = horizon
        self.out_dim = out_dim

    def forward(self, x):
        x = self.input_proj(x)
        x = self.pos_enc(x)
        out = self.encoder(x)
        pred = self.head(out[:, -1, :])
        return pred.reshape(pred.shape[0], self.horizon, self.out_dim)


# ---------- 5. STGCN ----------
class TemporalGatedConv(nn.Module):
    def __init__(self, in_dim, out_dim, kernel_size):
        super().__init__()
        self.conv = nn.Conv2d(in_dim, out_dim * 2, (1, kernel_size),
                              padding=(0, kernel_size - 1))

    def forward(self, x):
        x_pad = self.conv(x)
        x_pad = x_pad[:, :, :, :x.size(3)]
        P, Q = x_pad.chunk(2, dim=1)
        return P * torch.sigmoid(Q)


class ChebGraphConv(nn.Module):
    def __init__(self, in_dim, out_dim, K):
        super().__init__()
        self.K = K
        self.weight = nn.Parameter(torch.randn(K, in_dim, out_dim) * 0.1)

    def forward(self, x, L_scaled):
        B, N, D = x.shape
        outputs = []
        Tx_prev, Tx_cur = x, torch.bmm(L_scaled.unsqueeze(0).expand(B, N, N), x)
        for k in range(self.K):
            if k == 0:
                Tx = x
            elif k == 1:
                Tx = Tx_cur
            else:
                Tx = 2 * torch.bmm(L_scaled.unsqueeze(0).expand(B, N, N), Tx_cur) - Tx_prev
                Tx_prev, Tx_cur = Tx_cur, Tx
            outputs.append(torch.matmul(Tx, self.weight[k]))
        return sum(outputs)


class STConvBlock(nn.Module):
    def __init__(self, in_dim, hid_dim, out_dim, K, kernel_size):
        super().__init__()
        self.temp1 = TemporalGatedConv(in_dim, hid_dim, kernel_size)
        self.spatial = ChebGraphConv(hid_dim, hid_dim, K)
        self.temp2 = TemporalGatedConv(hid_dim, out_dim, kernel_size)
        self.ln = nn.LayerNorm(out_dim)
        self.downsample = nn.Conv2d(in_dim, out_dim, 1) if in_dim != out_dim else nn.Identity()

    def forward(self, x, L_scaled):
        B, N, D_in, L_in = x.shape
        x = x.permute(0, 2, 1, 3)
        res = self.downsample(x)
        x = F.relu(self.temp1(x))
        x_t = x.mean(dim=-1).permute(0, 2, 1)
        x_s = F.relu(self.spatial(x_t, L_scaled))
        x_s = x_s.permute(0, 2, 1).unsqueeze(-1).expand(B, x_s.size(-1), N, L_in)
        x = x + x_s
        x = F.relu(self.temp2(x))
        out = (x + res).permute(0, 2, 3, 1)
        out = self.ln(out)
        return out.permute(0, 1, 3, 2)


class STGCNForecaster(nn.Module):
    def __init__(self, N, F_in, out_dim, horizon, L_scaled, cfg):
        super().__init__()
        self.register_buffer('L_scaled', L_scaled)
        K = cfg.STGCN_CHEB_K
        kt = cfg.STGCN_TEMP_KERNEL
        ch = getattr(cfg, 'GRAPH_HIDDEN_DIM', cfg.HIDDEN_DIM)
        self.blocks = nn.ModuleList([STConvBlock(F_in, ch, ch, K, kt)])
        self.head = nn.Sequential(
            nn.Linear(ch, ch // 2), nn.LeakyReLU(0.1),
            nn.Dropout(cfg.DROPOUT), nn.Linear(ch // 2, horizon * out_dim))
        self.horizon = horizon
        self.out_dim = out_dim

    def forward(self, x, *args, **kwargs):
        B, L, N, F_in = x.shape
        x = x.permute(0, 2, 3, 1)
        for block in self.blocks:
            x = block(x, self.L_scaled)
        x_g = x.mean(dim=[1, 3])
        pred = self.head(x_g)
        return pred.reshape(pred.shape[0], self.horizon, self.out_dim)


# ---------- 6. GraphWaveNet ----------
class AdaptiveAdjacency(nn.Module):
    def __init__(self, N, embed_dim):
        super().__init__()
        self.E1 = nn.Parameter(torch.randn(N, embed_dim) * 0.1)
        self.E2 = nn.Parameter(torch.randn(embed_dim, N) * 0.1)

    def forward(self):
        return F.softmax(F.relu(torch.matmul(self.E1, self.E2)), dim=-1)


class DiffusionGCN(nn.Module):
    def __init__(self, in_dim, out_dim, diff_steps):
        super().__init__()
        self.diff_steps = diff_steps
        self.fc = nn.Linear(in_dim * (1 + 2 * diff_steps), out_dim)

    def forward(self, x, adj_pre, adj_adp):
        B, N, D = x.shape
        out_list = [x]
        x_in, x_out = x, x
        D_inv_out = adj_pre / (adj_pre.sum(-1, keepdim=True) + 1e-6)
        D_inv_in = adj_pre.T / (adj_pre.T.sum(-1, keepdim=True) + 1e-6)
        for _ in range(self.diff_steps):
            x_in = torch.bmm(D_inv_out.unsqueeze(0).expand(B, N, N), x_in)
            x_out = torch.bmm(D_inv_in.unsqueeze(0).expand(B, N, N), x_out)
            out_list.append(x_in)
            out_list.append(x_out)
        x_agg = torch.cat(out_list, dim=-1)
        return self.fc(x_agg)


class DilatedInceptionTCN(nn.Module):
    def __init__(self, in_dim, out_dim, kernel_size, dilation):
        super().__init__()
        self.pad = (kernel_size - 1) * dilation
        self.filter_conv = nn.Conv2d(in_dim, out_dim, (1, kernel_size), dilation=(1, dilation))
        self.gate_conv = nn.Conv2d(in_dim, out_dim, (1, kernel_size), dilation=(1, dilation))

    def forward(self, x):
        x_pad = F.pad(x, (self.pad, 0))
        f = self.filter_conv(x_pad)
        g = self.gate_conv(x_pad)
        return f * torch.sigmoid(g)


class GraphWaveNetForecaster(nn.Module):
    def __init__(self, N, F_in, out_dim, horizon, adj_norm, cfg):
        super().__init__()
        self.register_buffer('adj_pre', adj_norm)
        self.adp_adj = AdaptiveAdjacency(N, cfg.GWNET_ADP_DIM)
        ch = getattr(cfg, 'GRAPH_HIDDEN_DIM', cfg.HIDDEN_DIM)
        self.start_conv = nn.Conv2d(F_in, ch, (1, 1))
        self.diff_gcn = DiffusionGCN(ch, ch, cfg.GWNET_DIFF_STEPS)
        self.tcn_layers = nn.ModuleList()
        gwn_l = getattr(cfg, 'GRAPH_GWNET_LAYERS', cfg.GWNET_LAYERS)
        gwn_d = getattr(cfg, 'GRAPH_GWNET_DILATIONS', cfg.GWNET_DILATIONS)
        for i in range(gwn_l):
            self.tcn_layers.append(
                DilatedInceptionTCN(ch, ch, cfg.GWNET_KERNEL, gwn_d[i % len(gwn_d)]))
        self.skip_convs = nn.ModuleList([
            nn.Conv2d(ch, ch, (1, 1)) for _ in range(gwn_l)])
        self.head = nn.Sequential(
            nn.Linear(ch, ch // 2), nn.LeakyReLU(0.1),
            nn.Dropout(cfg.DROPOUT), nn.Linear(ch // 2, horizon * out_dim))
        self.horizon = horizon
        self.out_dim = out_dim

    def forward(self, x, *args, **kwargs):
        B, L, N, F_in = x.shape
        x = x.permute(0, 3, 2, 1)
        x = F.relu(self.start_conv(x))
        adp = self.adp_adj().unsqueeze(0).expand(B, N, N)
        x_s = x.permute(0, 2, 3, 1).contiguous().view(B * N, -1, x.size(1)).mean(dim=1).view(B, N, x.size(1))
        x_g = F.relu(self.diff_gcn(x_s, self.adj_pre, adp))
        x = x + x_g.permute(0, 2, 1).unsqueeze(-1).expand(B, x_g.size(-1), N, L)
        skip_sum = 0
        for tcn, skip_conv in zip(self.tcn_layers, self.skip_convs):
            residual = x
            x = tcn(x)
            skip_sum += skip_conv(x[:, :, :, -L:])
            x = x + residual[:, :, :, -x.size(3):]
        x_out = F.relu(skip_sum).mean(dim=[2, 3])
        pred = self.head(x_out)
        return pred.reshape(B, self.horizon, self.out_dim)


# =============================================================================
# OURS FRAMEWORK VARIANTS (V-Attention, V-MLP)
# Same STGNN base as ablation except vertical propagation module
# =============================================================================

# --- CausalConv1d & TCNEncoder ---
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


# --- GCN & Attention & PhysicsGate ---
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


# --- Replacements for Vertical GCN ---

class DepthSelfAttention(nn.Module):
    """Self-attention along depth dimension, replacing VerticalGCN.

    For each monitoring point (5 points x 31 depths = 31 nodes each),
    applies self-attention across the depth sequences.
    """
    def __init__(self, dim, n_dep=31, n_heads=4):
        super().__init__()
        self.dim = dim
        self.n_heads = n_heads
        self.d_k = dim // n_heads
        self.q_proj = nn.Linear(dim, dim)
        self.k_proj = nn.Linear(dim, dim)
        self.v_proj = nn.Linear(dim, dim)
        self.out_proj = nn.Linear(dim, dim)
        self.norm = nn.LayerNorm(dim)

    def forward(self, x, n_points=5, n_dep=31):
        B, N, D = x.shape
        x_reshaped = x.view(B, n_points, n_dep, D)
        x_perm = x_reshaped.permute(0, 1, 3, 2).contiguous().view(B * n_points, D, n_dep).permute(0, 2, 1)
        H = self.n_heads
        d_k = self.d_k
        q = self.q_proj(x_perm).view(B * n_points, n_dep, H, d_k).transpose(1, 2)
        k = self.k_proj(x_perm).view(B * n_points, n_dep, H, d_k).transpose(1, 2)
        v = self.v_proj(x_perm).view(B * n_points, n_dep, H, d_k).transpose(1, 2)
        attn = torch.softmax(torch.matmul(q, k.transpose(-2, -1)) / np.sqrt(d_k), dim=-1)
        out = torch.matmul(attn, v).transpose(1, 2).contiguous().view(B * n_points, n_dep, D)
        out = self.out_proj(out)
        out = self.norm(x_perm + out)
        out = out.view(B, n_points, n_dep, D).permute(0, 1, 3, 2).contiguous().view(B, N, D)
        return out


class DepthMLP(nn.Module):
    """2-layer MLP mixing features ALONG THE DEPTH dimension, replacing
    VerticalGCN (issue 6: the previous version applied a per-node MLP with
    no cross-depth information exchange at all)."""
    def __init__(self, n_dep, hidden_mult=2, dropout=0.1):
        super().__init__()
        self.n_dep = n_dep
        self.mlp = nn.Sequential(
            nn.Linear(n_dep, n_dep * hidden_mult),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(n_dep * hidden_mult, n_dep))
        self.norm = nn.LayerNorm(n_dep)

    def forward(self, x, n_points=5, n_dep=None):
        B, N, D = x.shape
        if n_dep is None:
            n_dep = self.n_dep
        # (B, n_points, n_dep, D) -> (B, n_points, D, n_dep): mix along depth
        xr = x.view(B, n_points, N // n_points, D).permute(0, 1, 3, 2)
        out = self.mlp(xr)
        out = self.norm(xr + out).permute(0, 1, 3, 2)
        return out.reshape(B, N, D)


# --- V-Attention Model (Ours framework + self-attention along depth) ---
class VAttentionModel(nn.Module):
    def __init__(self, cfg, N, in_dim, gcn_hid, tcn_ch, k, dil, attn_hid, hor,
                 adj, out_dim, geo_feat_dim=7, n_points=5, n_dep=None):
        super().__init__()
        self.tcn = TCNEncoder(in_dim, tcn_ch, k, dil, cfg.TCN_DROPOUT)
        tcn_dim = tcn_ch[-1]
        # 路径 A（2026-09-17）：depth_dim 原为 tcn_dim * 2（为容纳并拼入的
        # water_node 投影）；水文移除后深度注意力 / GCN 输入回到 tcn_dim。
        depth_dim = tcn_dim
        # 架构简化（2026-09-19）：时序注意力改为均值池化（与消融 Ablation_Experiment
        # 同步，见其 __init__ 注释）。删除 attn1/attn_v 模块。
        # 数据导出层（2026-09-18）：暂存时间注意力权重（简化后恒 None）
        self._last_temporal_attn = None
        # 路径 A（2026-09-17）：已移除水特征注入（water_node_proj 分支）。此前会把
        # 8 维标准化水位投影到 TCN 隐藏维度后拼入深度注意力/GCN 输入；VICD 诊断
        # （正文 §3.4）表明其无可用信息且引入复杂度惩罚，故彻底移除。
        # 2026-09-20：原本此处另注册 `static_adj` buffer，与 `self.gcn.adj`
        # （GatedGCN 内部）指向同一张量，属重复注册；全仓库无任何消费者，已删除。
        self.depth_attn = DepthSelfAttention(depth_dim, n_heads=4)
        self.gcn = GatedGCN(depth_dim, gcn_hid, adj, dropout=cfg.DROPOUT)
        self.gcn_norm = nn.LayerNorm(gcn_hid)
        self.gcn_drop = nn.Dropout(cfg.DROPOUT)
        cur_dim = gcn_hid
        self.physics_gate = PhysicsGate(cur_dim, geo_feat_dim)
        self.cross_attn = CrossNodeAttention(cur_dim, geo_feat_dim)
        self.head = nn.Sequential(
            nn.Linear(cur_dim, cur_dim // 2),
            nn.LeakyReLU(0.1),
            nn.Dropout(cfg.DROPOUT),
            nn.Linear(cur_dim // 2, hor * out_dim))
        self.horizon = hor
        self.out_dim = out_dim
        self.n_points = n_points
        # Issue 15: derive from cfg.DEPTHS instead of hardcoding 31.
        self.n_dep = n_dep if n_dep is not None else len(cfg.DEPTHS)

    def forward(self, x, water_feat, geo_static):
        B, seq, Nodes, in_feat = x.shape
        tcn_in = x.permute(0, 2, 1, 3).reshape(B * Nodes, seq, in_feat)
        tcn_out = self.tcn(tcn_in)
        # 架构简化（2026-09-19）：时序注意力 → 均值池化（与消融同步）。
        self._last_temporal_attn = None
        ctx = tcn_out.mean(dim=1)
        ctx = ctx.reshape(B, Nodes, -1)
        # 路径 A（2026-09-17）：已移除水文注入（water_node_proj 拼接）。
        # water_feat 入参仅作签名占位以兼容既有 DataLoader。
        # Runtime depth count (robust to reduced node sets, e.g. sparse tests)
        n_dep_rt = Nodes // self.n_points
        ctx = self.depth_attn(ctx, self.n_points, n_dep_rt)
        ctx = self.gcn(ctx, dynamic_adj=None)
        ctx = self.gcn_norm(ctx)
        ctx = self.gcn_drop(F.relu(ctx))
        ctx = self.physics_gate(ctx, geo_static)
        ctx = self.cross_attn(ctx, geo_static)
        global_feat = ctx.mean(dim=1)
        out = self.head(global_feat)
        return out.reshape(B, self.horizon, self.out_dim)


# --- V-MLP Model (Ours framework + MLP along depth) ---
class VMLPModel(nn.Module):
    def __init__(self, cfg, N, in_dim, gcn_hid, tcn_ch, k, dil, attn_hid, hor,
                 adj, out_dim, geo_feat_dim=7, n_points=5, n_dep=None):
        super().__init__()
        self.tcn = TCNEncoder(in_dim, tcn_ch, k, dil, cfg.TCN_DROPOUT)
        tcn_dim = tcn_ch[-1]
        # 路径 A（2026-09-17）：depth_dim 原为 tcn_dim * 2（为容纳并拼入的
        # water_node 投影）；水文移除后 GCN 输入回到 tcn_dim。
        depth_dim = tcn_dim
        # 架构简化（2026-09-19）：时序注意力改为均值池化（与消融 Ablation_Experiment
        # 同步，见其 __init__ 注释）。删除 attn1/attn_v 模块。
        # 数据导出层（2026-09-18）：暂存时间注意力权重（简化后恒 None）
        self._last_temporal_attn = None
        # 路径 A（2026-09-17）：已移除水特征注入（water_node_proj 分支）。
        # 2026-09-20：原本此处另注册 `static_adj` buffer（同 `self.gcn.adj`，重复
        # 注册、无消费者），已删除。
        self.n_points = n_points
        # Issue 15: derive from cfg.DEPTHS instead of hardcoding 31.
        self.n_dep = n_dep if n_dep is not None else len(cfg.DEPTHS)
        self.depth_mlp = DepthMLP(self.n_dep)
        self.gcn = GatedGCN(depth_dim, gcn_hid, adj, dropout=cfg.DROPOUT)
        self.gcn_norm = nn.LayerNorm(gcn_hid)
        self.gcn_drop = nn.Dropout(cfg.DROPOUT)
        cur_dim = gcn_hid
        self.physics_gate = PhysicsGate(cur_dim, geo_feat_dim)
        self.cross_attn = CrossNodeAttention(cur_dim, geo_feat_dim)
        self.head = nn.Sequential(
            nn.Linear(cur_dim, cur_dim // 2),
            nn.LeakyReLU(0.1),
            nn.Dropout(cfg.DROPOUT),
            nn.Linear(cur_dim // 2, hor * out_dim))
        self.horizon = hor
        self.out_dim = out_dim

    def forward(self, x, water_feat, geo_static):
        B, seq, Nodes, in_feat = x.shape
        tcn_in = x.permute(0, 2, 1, 3).reshape(B * Nodes, seq, in_feat)
        tcn_out = self.tcn(tcn_in)
        # 架构简化（2026-09-19）：时序注意力 → 均值池化（与消融同步）。
        self._last_temporal_attn = None
        ctx = tcn_out.mean(dim=1)
        ctx = ctx.reshape(B, Nodes, -1)
        # 路径 A（2026-09-17）：已移除水文注入（water_node_proj 拼接）。
        # water_feat 入参仅作签名占位以兼容既有 DataLoader。
        # Runtime depth count (robust to reduced node sets, e.g. sparse tests)
        n_dep_rt = Nodes // self.n_points
        ctx = self.depth_mlp(ctx, self.n_points, n_dep_rt)
        ctx = self.gcn(ctx, dynamic_adj=None)
        ctx = self.gcn_norm(ctx)
        ctx = self.gcn_drop(F.relu(ctx))
        ctx = self.physics_gate(ctx, geo_static)
        ctx = self.cross_attn(ctx, geo_static)
        global_feat = ctx.mean(dim=1)
        out = self.head(global_feat)
        return out.reshape(B, self.horizon, self.out_dim)


# =============================================================================
# MODEL FACTORY
# =============================================================================
def get_model(cfg, model_name, N, F_in, out_dim, horizon, adj_norm, L_scaled,
              geo_dim=7, n_points=5, n_dep=None):
    device = cfg.DEVICE
    # V-Attention / V-MLP 用 Ours 框架的隐藏宽度（= 消融 V5）；STGCN / GraphWaveNet
    # 在各自 __init__ 里读 GRAPH_HIDDEN_DIM，不受影响。
    ch = cfg.OURS_GCN_HIDDEN
    if model_name == 'LSTM':
        return LSTMForecaster(F_in, cfg.HIDDEN_DIM, out_dim, horizon,
                              cfg.N_LAYERS, cfg.DROPOUT).to(device)
    elif model_name == 'BiLSTM-Attn':
        return BiLSTMAttentionForecaster(F_in, cfg.HIDDEN_DIM, out_dim, horizon,
                                         cfg.N_LAYERS, cfg.DROPOUT).to(device)
    elif model_name == 'Transformer':
        return TransformerForecaster(F_in, cfg.HIDDEN_DIM, out_dim, horizon,
                                     cfg.N_LAYERS, cfg.N_HEADS, cfg.DROPOUT).to(device)
    elif model_name == 'V-Attention':
        adj_t = torch.FloatTensor(adj_norm).to(device)
        return VAttentionModel(cfg, N=N, in_dim=F_in, gcn_hid=ch,
                               tcn_ch=cfg.TCN_CHANNELS, k=cfg.TCN_KERNEL_SIZE,
                               dil=cfg.TCN_DILATIONS, attn_hid=cfg.ATTENTION_HIDDEN,
                               hor=horizon, adj=adj_t, out_dim=out_dim,
                               geo_feat_dim=geo_dim, n_points=n_points,
                               n_dep=n_dep).to(device)
    elif model_name == 'STGCN':
        return STGCNForecaster(N, F_in, out_dim, horizon,
                               torch.FloatTensor(L_scaled).to(device), cfg).to(device)
    elif model_name == 'GraphWaveNet':
        return GraphWaveNetForecaster(N, F_in, out_dim, horizon,
                                      torch.FloatTensor(adj_norm).to(device), cfg).to(device)
    elif model_name == 'V-MLP':
        adj_t = torch.FloatTensor(adj_norm).to(device)
        return VMLPModel(cfg, N=N, in_dim=F_in, gcn_hid=ch,
                         tcn_ch=cfg.TCN_CHANNELS, k=cfg.TCN_KERNEL_SIZE,
                         dil=cfg.TCN_DILATIONS, attn_hid=cfg.ATTENTION_HIDDEN,
                         hor=horizon, adj=adj_t, out_dim=out_dim,
                         geo_feat_dim=geo_dim, n_points=n_points,
                         n_dep=n_dep).to(device)
    else:
        raise ValueError(f"Unknown model: {model_name}")


def count_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def estimate_flops(model, cfg, mtype, X_sample, wf_sample=None, geo_sample=None):
    """FLOPs estimation with per-type dummy inputs matching each model's
    forward signature (issue 13).

    2026-09-21：原先 `except Exception: return 0` 会把 thop 缺失静默吞掉，使全部
    run 的 FLOPs 写成 0（论文复杂度表整列失效）。改为只兜底 `profile` 抛错，
    且失败时告警而非静默。"""
    try:
        from thop import profile
    except ImportError as e:
        logger.warning(f"thop 不可用（{e}），FLOPs 记为 0 —— 请先 pip install thop")
        return 0
    try:
        was_training = model.training
        model.eval()
        x = torch.as_tensor(np.asarray(X_sample), dtype=torch.float32).to(cfg.DEVICE)
        if mtype == 'temporal':
            inputs = (x,)
        else:
            wf = torch.as_tensor(np.asarray(wf_sample), dtype=torch.float32).to(cfg.DEVICE)
            geo = torch.as_tensor(np.asarray(geo_sample), dtype=torch.float32).to(cfg.DEVICE)
            inputs = (x, wf, geo)
        flops, _ = profile(model, inputs=inputs, verbose=False)
        if was_training:
            model.train()
        return flops
    except Exception as e:
        logger.warning(f"FLOPs profile 失败（{type(e).__name__}: {e}），记为 0")
        return 0


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

    2026-09-20：与 Ablation_Experiment 统一为**按模块对象显式收集**，
    不再依赖参数名子串匹配（'gcn' 会顺带命中 gcn_norm；vertical_gcn 此前
    靠 'gcn' 巧合命中）。V-Attention / V-MLP 无 vertical_gcn，getattr 返回
    None 自动跳过；返回顺序与原实现逐位等价。
    """
    holders = [getattr(model, n, None) for n in
               ('gcn', 'gcn_norm', 'gcn_drop', 'physics_gate',
                'cross_attn', 'vertical_gcn')]
    gid = {id(p) for m in holders if m is not None for p in m.parameters()}
    gra = [p for _, p in model.named_parameters() if id(p) in gid]
    rest = [p for _, p in model.named_parameters() if id(p) not in gid]
    return gra, rest


def train_model(cfg, model, train_ld, val_ld, save_path, model_tag='', seed=0,
                ours_lr_groups=False):
    if ours_lr_groups:
        # Ours 框架变体（V-Attention / V-MLP）：参数分组与
        # Ablation_Experiment.train_model 完全相同——图/物理相关模块用 LR_GCN，
        # 其余用 LR，保证与消融 V5 是同一套训练配方。
        graph_params, other_params = _graph_lr_param_groups(model)
        opt = torch.optim.AdamW([
            {'params': graph_params, 'lr': cfg.LR_GCN},
            {'params': other_params, 'lr': cfg.LR}], weight_decay=cfg.WD)
    else:
        opt = torch.optim.AdamW(model.parameters(), lr=cfg.LR, weight_decay=cfg.WD)
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
            pred = model(*([Xb] + [t.to(cfg.DEVICE) for t in batch_data[2:]]))
            loss = crit(pred, yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.GRAD_CLIP)
            opt.step()
            tr_loss += loss.item() * Xb.size(0)
        tr_loss /= len(train_ld.dataset)
        model.eval()
        va_loss = 0
        with torch.no_grad():
            for batch_data in val_ld:
                Xb = batch_data[0].to(cfg.DEVICE)
                yb = batch_data[1].to(cfg.DEVICE)
                pred = model(*([Xb] + [t.to(cfg.DEVICE) for t in batch_data[2:]]))
                loss = crit(pred, yb)
                va_loss += loss.item() * Xb.size(0)
        va_loss /= len(val_ld.dataset)
        tr_losses.append(tr_loss)
        va_losses.append(va_loss)
        sch.step(va_loss)
        if ep % log_interval == 0 or ep == 1 or ep == cfg.EPOCHS:
            lr_now = opt.param_groups[0]['lr']
            logger.info(f"    [{model_tag} seed={seed}] Epoch {ep:4d}/{cfg.EPOCHS} | "
                        f"train_loss={tr_loss:.6f} | val_loss={va_loss:.6f} | "
                        f"lr={lr_now:.2e} | best_val={best_vloss:.6f}")
        if va_loss < best_vloss:
            best_vloss = va_loss
            wait = 0
            best_ep = ep
            torch.save({'state_dict': model.state_dict(), 'epoch': ep,
                        'val_loss': va_loss, 'train_loss': tr_loss}, save_path)
        else:
            wait += 1
            if wait >= cfg.PATIENCE:
                logger.info(f"    [{model_tag} seed={seed}] Early stop at epoch {ep}, "
                            f"best_epoch={best_ep}, best_val_loss={best_vloss:.6f}")
                break
    train_time = time.time() - t_start
    ckpt = torch.load(save_path, map_location=cfg.DEVICE)
    model.load_state_dict(ckpt['state_dict'] if isinstance(ckpt, dict) else ckpt)
    return model, train_time, tr_losses, va_losses, best_ep if 'best_ep' in dir() else ep


def evaluate(model, test_ld, init_disp, scY, pca_y, hz, device, true_diff_win=None):
    """测试集评估：预测走 PCA 逆变换，真值取原始一阶差分（2026-09-20 口径修正）。

    此前真值也做 `pca_y.inverse_transform` 往返，等于把真值投影到 K=13 子空间后再
    比，丢弃的约 10% 增量方差及其随累加步数的放大都不计入误差 → 各表 RMSE/R² 系统
    性偏乐观。现真值直接取 `diff1_den` 对应切片（由调用方经 `yte_raw` 传入）的累加，
    预测侧照旧走 PCA 逆变换（模型本就在该子空间内输出）。
    """
    model.eval()
    preds_norm = []
    # 2026-09-21：`true_diff_win` 已由主路径一律传入，`trues_norm` 仅在回退旧口径
    # （诊断脚本）时才需要收集，避免白占内存。
    need_old_truth = true_diff_win is None
    trues_norm = [] if need_old_truth else None
    with torch.no_grad():
        for batch_data in test_ld:
            Xb = batch_data[0].to(device)
            yb = batch_data[1].to(device)
            pred = model(*([Xb] + [t.to(device) for t in batch_data[2:]]))
            preds_norm.append(pred.cpu().numpy())
            if need_old_truth:
                trues_norm.append(yb.cpu().numpy())
    pred_n = np.concatenate(preds_norm, 0)
    out_dim = pred_n.shape[-1]
    ns = pred_n.shape[0]
    pred_pca = scY.inverse_transform(pred_n.reshape(-1, out_dim)).reshape(ns, hz, out_dim)
    N_orig = pca_y.n_features_in_
    pred_diff = pca_y.inverse_transform(pred_pca.reshape(-1, out_dim)).reshape(ns, hz, N_orig)
    if true_diff_win is not None:
        true_diff = np.asarray(true_diff_win, dtype=np.float64)[:ns]
    else:
        # 兼容路径：调用方未提供原始真值窗口时回退旧口径（诊断脚本可能用到）。
        true_n = np.concatenate(trues_norm, 0)
        true_pca = scY.inverse_transform(true_n.reshape(-1, out_dim)).reshape(ns, hz, out_dim)
        true_diff = pca_y.inverse_transform(true_pca.reshape(-1, out_dim)).reshape(ns, hz, N_orig)
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


def _point_metrics(t, p):
    return {'RMSE': float(np.sqrt(mean_squared_error(t, p))),
            'MAE': float(mean_absolute_error(t, p)),
            'R2': float(r2_score(t, p)),
            'MaxErr': float(max_error(t, p))}


def global_metrics(true_cum, pred_cum):
    """全步口径 + 末步口径（`_last` 后缀），二者一并落盘。

    两个口径都要保留的原因：论文主口径是**末步**（每窗取 t=hz-1，每个测试时刻只
    贡献一行，与视界无关）；全步口径把窗口内每一步都算进去，而测试窗口相互重叠
    （H=2 约 1.9x、H=10 约 5.5x、H=12 约 5.4x），同一时刻会被重复计权多次，导致
    指标随视界漂移、跨视界不可比。`_last` 为主、无后缀为辅。
    """
    gm = _point_metrics(true_cum.ravel(), pred_cum.ravel())
    gm.update({f'{k}_last': v for k, v in
               _point_metrics(true_cum[:, -1, :].ravel(),
                              pred_cum[:, -1, :].ravel()).items()})
    if true_cum.shape[1] > 1:
        true_d = np.diff(true_cum, axis=1)
        pred_d = np.diff(pred_cum, axis=1)
        gm['DA'] = float(np.mean(np.sign(true_d).ravel() == np.sign(pred_d).ravel()))
        gm['DA_last'] = float(np.mean(
            np.sign(true_cum[:, -1, :] - true_cum[:, -2, :]).ravel() ==
            np.sign(pred_cum[:, -1, :] - pred_cum[:, -2, :]).ravel()))
    else:
        gm['DA'] = 0.5
        gm['DA_last'] = np.nan
    return gm


def output_space_floor(yte_raw, init_disp, scY, pca_y, hz, ns=None):
    """K 维 PCA 输出空间导致的**不可达误差地板**（与模型无关）。

    与 `Ablation_Experiment.output_space_floor` 逐行等价，本脚本内联一份以免跨脚本
    导入。原理：真值 y 正交分解为 y = y_S + y_⊥（y_S 为 y 在 K 维主子空间上的投影），
    本输出头下任何模型的预测 p 必落在主子空间内，故 ||y − p||² = ||y_S − p||² + ||y_⊥||²；
    `cumsum` 是线性算子且逐时刻正交，该可加性在末步累加口径下同样成立。

    因此可把主口径 RMSE 无损拆成两项，而**不改动主口径本身**：
        RMSE_full² = RMSE_reach² + RMSE_floor²
    其中 RMSE_floor 只取决于 K 与测试集，是**所有 K 维输出模型的公共下界**。

    Returns
    -------
    dict: MSE_floor_all / RMSE_floor_all（全步累加口径）
          MSE_floor_last / RMSE_floor_last（末步累加口径，与 `RMSE_last` 配套）
    """
    y_T = np.asarray(yte_raw, dtype=np.float64)
    ns = int(y_T.shape[0] if ns is None else min(ns, y_T.shape[0]))
    y_T = y_T[:ns]
    N = int(pca_y.n_features_in_)
    # 真值经同一 PCA 往返 → 主子空间分量 y_S
    y_S = pca_y.inverse_transform(
        scY.inverse_transform(scY.transform(pca_y.transform(y_T.reshape(-1, N))))
    ).reshape(ns, hz, N)
    cum_T = np.zeros((ns, hz, N), dtype=np.float64)
    cum_S = np.zeros((ns, hz, N), dtype=np.float64)
    for i in range(ns):
        cT = np.array(init_disp[i], dtype=np.float64, copy=True)
        cS = cT.copy()
        for t in range(hz):
            cT = cT + y_T[i, t]
            cS = cS + y_S[i, t]
            cum_T[i, t] = cT
            cum_S[i, t] = cS
    d = cum_T - cum_S                      # = cumsum(y_⊥)
    mse_all = float((d ** 2).mean())
    mse_last = float((d[:, -1, :] ** 2).mean())
    return {'MSE_floor_all': mse_all, 'RMSE_floor_all': float(np.sqrt(mse_all)),
            'MSE_floor_last': mse_last, 'RMSE_floor_last': float(np.sqrt(mse_last))}


def node_metrics(true_cum, pred_cum, node_names):
    """逐节点指标：全步口径 + 末步口径（`_last` 后缀，定义见 global_metrics）。"""
    records = []
    for j, name in enumerate(node_names):
        tj = true_cum[:, :, j].ravel()
        pj = pred_cum[:, :, j].ravel()
        tjl = true_cum[:, -1, j]
        pjl = pred_cum[:, -1, j]
        records.append({
            'Node': name,
            'RMSE': np.sqrt(mean_squared_error(tj, pj)),
            'MAE': mean_absolute_error(tj, pj),
            'R2': r2_score(tj, pj) if np.std(tj) > 1e-6 else np.nan,
            'RMSE_last': np.sqrt(mean_squared_error(tjl, pjl)),
            'MAE_last': mean_absolute_error(tjl, pjl),
            'R2_last': r2_score(tjl, pjl) if np.std(tjl) > 1e-6 else np.nan,
        })
    return pd.DataFrame(records)


def _layer_point_summary(df, keys, r2c, rmsec, maec):
    """先按 seed 取组内中位数，再跨种子取均值/标准差（层、测点两级共用）。"""
    per_seed = df.groupby(['seed'] + keys).agg(
        R2_med=(r2c, 'median'), RMSE_med=(rmsec, 'median'),
        MAE_med=(maec, 'median')).reset_index()
    return per_seed.groupby(keys).agg(
        R2_mean=('R2_med', 'mean'), R2_std=('R2_med', 'std'),
        RMSE_mean=('RMSE_med', 'mean'), RMSE_std=('RMSE_med', 'std'),
        MAE_mean=('MAE_med', 'mean'), MAE_std=('MAE_med', 'std')).reset_index()


def _node_summary(df, r2c, rmsec, maec):
    """逐节点汇总（跨种子均值/标准差/中位数）。"""
    return df.groupby('Node').agg(
        R2_mean=(r2c, 'mean'), R2_std=(r2c, 'std'), R2_median=(r2c, 'median'),
        RMSE_mean=(rmsec, 'mean'), RMSE_std=(rmsec, 'std'),
        MAE_mean=(maec, 'mean'), MAE_std=(maec, 'std')).reset_index()


# =============================================================================
# GBDT EXTERNAL BASELINE (2026-09-21)
# =============================================================================
def _make_gbdt(seed):
    """梯度提升树工厂：XGBoost → LightGBM → sklearn HistGBR 依次回退。

    并行度策略：外层 `MultiOutputRegressor` 串行（n_jobs=1），并行交给估计器自身的
    原生线程。原因：joblib 的多进程后端会在子进程中以 ascii 编码注册 memmapping
    临时目录，而本项目 TMP 路径含中文 → UnicodeEncodeError（实测）。
    """
    try:
        from xgboost import XGBRegressor
        return 'XGBoost', XGBRegressor(
            n_estimators=300, max_depth=4, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8,
            random_state=seed, n_jobs=-1, verbosity=0)
    except Exception:
        pass
    try:
        from lightgbm import LGBMRegressor
        return 'LightGBM', LGBMRegressor(
            n_estimators=300, max_depth=4, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8,
            random_state=seed, n_jobs=-1, verbose=-1)
    except Exception:
        pass
    from sklearn.ensemble import HistGradientBoostingRegressor
    return 'GBDT-sklearn', HistGradientBoostingRegressor(
        max_iter=120, max_depth=3, learning_rate=0.06, random_state=seed)


def _decode_predict(pred_n, init_disp, scY, pca_y, hz, true_diff_win):
    """把 (n, hz, out_dim) 标准化 PCA 系数还原为累计位移。

    与 `evaluate` 的内部链路逐行等价（scY 反标准化 → pca_y 逆变换 → 一阶差分累加），
    供**非 nn.Module 的模型**（GBDT）复用完全相同的评估口径。
    """
    out_dim = pred_n.shape[-1]
    ns = pred_n.shape[0]
    pred_pca = scY.inverse_transform(pred_n.reshape(-1, out_dim)).reshape(ns, hz, out_dim)
    n_orig = pca_y.n_features_in_
    pred_diff = pca_y.inverse_transform(pred_pca.reshape(-1, out_dim)).reshape(ns, hz, n_orig)
    true_diff = np.asarray(true_diff_win, dtype=np.float64)[:ns]
    pred_cum = np.zeros_like(pred_diff)
    true_cum = np.zeros_like(true_diff)
    for i in range(ns):
        cp = init_disp[i].copy()
        ct = init_disp[i].copy()
        for t in range(hz):
            cp = cp + pred_diff[i, t, :]
            ct = ct + true_diff[i, t, :]
            pred_cum[i, t, :] = cp
            true_cum[i, t, :] = ct
    return pred_cum, true_cum


def run_gbdt_external(cfg, model_name, shared, hz, model_out_dir):
    """GBDT 外部基线主流程：非神经网络对照，回答"为何不采用机器学习方法"。

    与 `run_external_model` 的关系：**数据构造**（temporal 路径展平 + PCA 至 48 维）、
    **评估链路**（`_decode_predict` → `global_metrics`）与**落盘约定**完全一致，
    唯一区别是"训练"换成 sklearn 的梯度提升树拟合。因此其结果与其余模型严格同口径、
    可直接进入 comparison 汇总与配对检验。

    代价说明：sklearn 的 `MultiOutputRegressor` 需为 hz×out_dim 个输出各拟合一棵树，
    故单种子耗时约 2 分钟（LightGBM），显著慢于神经网络基线。
    """
    data = build_temporal_samples(shared, cfg, hz)
    Xtr = data['Xtr'].reshape(data['Xtr'].shape[0], -1)
    Xva = data['Xva'].reshape(data['Xva'].shape[0], -1)
    Xte = data['Xte'].reshape(data['Xte'].shape[0], -1)
    ytr = data['ytr'].reshape(data['ytr'].shape[0], -1)
    yva = data['yva'].reshape(data['yva'].shape[0], -1)
    out_dim = data['out_dim']

    all_global, all_node = [], []
    failed_seeds = []

    for si, seed in enumerate(cfg.SEEDS):
        set_seed(seed)
        run_dir = model_out_dir / f"run_{seed}"
        # RESUME 判据：GBDT 不产 best.pth / losses.npy（不训练神经网络），
        # 故改用 metrics.json + predictions.npz + node_metrics.xlsx 三件套。
        if cfg.RESUME and all((run_dir / f).exists()
                              for f in ('metrics.json', 'predictions.npz',
                                        'node_metrics.xlsx')):
            with open(run_dir / 'metrics.json', 'r') as f:
                gm = json.load(f)
            gm['seed'] = seed
            all_global.append(gm)
            df_n = pd.read_excel(run_dir / 'node_metrics.xlsx')
            df_n['seed'] = seed
            all_node.append(df_n)
            logger.info(f"    Seed {si + 1}/{len(cfg.SEEDS)} ({seed}): "
                        f"结果已完整，跳过（RESUME=1）")
            continue

        try:
            tag, base = _make_gbdt(seed)
            model = MultiOutputRegressor(base, n_jobs=1)
            t0 = time.time()
            model.fit(Xtr, ytr)
            train_time = time.time() - t0
            val_mse = float(np.mean((model.predict(Xva) - yva) ** 2))
            t1 = time.time()
            pred_n = model.predict(Xte).reshape(-1, hz, out_dim)
            infer_time = (time.time() - t1) / max(len(Xte), 1)

            pred_cum, true_cum = _decode_predict(
                pred_n, data['init_test'], data['scY'], data['pca_y'], hz,
                data['yte_raw'])

            gm = global_metrics(true_cum, pred_cum)
            # 输出空间误差分解（原理见 `output_space_floor`）：地板与模型无关，把主口径
            # RMSE 无损拆为「可达 + 不可达下界」，满足 RMSE_full² = RMSE_reach² + RMSE_floor²。
            _fl = output_space_floor(data['yte_raw'], data['init_test'], data['scY'],
                                     data['pca_y'], hz, ns=pred_cum.shape[0])
            gm['RMSE_floor_last'] = _fl['RMSE_floor_last']
            gm['RMSE_reach_last'] = float(np.sqrt(max(
                gm['RMSE_last'] ** 2 - _fl['MSE_floor_last'], 0.0)))
            gm['RMSE_floor'] = _fl['RMSE_floor_all']
            gm['RMSE_reach'] = float(np.sqrt(max(
                gm['RMSE'] ** 2 - _fl['MSE_floor_all'], 0.0)))
            gm.update({'seed': seed, 'Params': 0, 'FLOPs': 0.0,
                       'TrainTime': train_time, 'InferTime': infer_time,
                       'BestEpoch': -1, 'BestValLoss': val_mse, 'ImplTag': tag})
            all_global.append(gm)

            nodes = [f"Node{i}" for i in range(true_cum.shape[2])]
            df_n = node_metrics(true_cum, pred_cum, nodes)
            df_n['seed'] = seed
            all_node.append(df_n)

            run_dir.mkdir(parents=True, exist_ok=True)
            with open(run_dir / 'metrics.json', 'w') as f:
                json.dump(gm, f, indent=2)
            df_n.to_excel(run_dir / 'node_metrics.xlsx', index=False)
            np.savez(run_dir / 'predictions.npz', pred_cum=pred_cum, true_cum=true_cum)

            logger.info(f"    Seed {si + 1}/{len(cfg.SEEDS)} ({seed}) "
                        f"RMSE(last)={gm['RMSE_last']:.4f} RMSE(full)={gm['RMSE']:.4f} "
                        f"R2(last)={gm['R2_last']:.4f}  [{tag}]")
        except Exception as e:
            logger.error(f"    Seed {seed} failed: {e}")
            failed_seeds.append(seed)
            import traceback
            traceback.print_exc()

    if not all_global:
        logger.error(f"  {model_name} H={hz}: ALL seeds failed!")
        return None, None
    if failed_seeds:
        logger.error(f"  !! {model_name} H={hz}: {len(failed_seeds)}/{len(cfg.SEEDS)} "
                     f"个种子失败 {failed_seeds} —— 汇总将写为 "
                     f"global_{len(all_global)}runs.csv")

    df_global_all = pd.DataFrame(all_global)
    all_node_flat = pd.concat(all_node, ignore_index=True)

    summary_dir = model_out_dir / "summary"
    summary_dir.mkdir(exist_ok=True)
    df_global_all.to_csv(summary_dir / f'global_{len(all_global)}runs.csv', index=False)

    if not all_node_flat.empty:
        meta = shared['meta']
        depths_arr = meta['depths']
        point_names = meta['point_names']
        node_to_depth = dict(zip([f"Node{i}" for i in range(len(depths_arr))], depths_arr))
        node_to_point = dict(zip([f"Node{i}" for i in range(len(point_names))], point_names))
        all_node_flat['Depth'] = all_node_flat['Node'].map(node_to_depth)
        all_node_flat['Point'] = all_node_flat['Node'].map(node_to_point)
        all_node_flat['Layer'] = pd.cut(all_node_flat['Depth'], bins=[0, 3, 8, 16],
                                        labels=['Shallow', 'Middle', 'Deep'], right=False)
        layer_summary = _layer_point_summary(
            all_node_flat, ['Layer'], 'R2', 'RMSE', 'MAE').merge(
            _layer_point_summary(all_node_flat, ['Layer'],
                                 'R2_last', 'RMSE_last', 'MAE_last'),
            on='Layer', suffixes=('', '_last'))
        point_summary = _layer_point_summary(
            all_node_flat, ['Point'], 'R2', 'RMSE', 'MAE').merge(
            _layer_point_summary(all_node_flat, ['Point'],
                                 'R2_last', 'RMSE_last', 'MAE_last'),
            on='Point', suffixes=('', '_last'))
        node_summary = _node_summary(
            all_node_flat, 'R2', 'RMSE', 'MAE').merge(
            _node_summary(all_node_flat, 'R2_last', 'RMSE_last', 'MAE_last'),
            on='Node', suffixes=('', '_last'))
        node_summary['Depth'] = node_summary['Node'].map(node_to_depth)
        node_summary['Point'] = node_summary['Node'].map(node_to_point)
        all_node_flat.to_excel(summary_dir / 'node_summary.xlsx', index=False)
        layer_summary.to_csv(summary_dir / 'layer_summary.csv', index=False)
        point_summary.to_csv(summary_dir / 'point_summary.csv', index=False)
        node_summary.to_excel(summary_dir / 'node_summary_detail.xlsx', index=False)

    logger.info(f"  {model_name} H={hz}: {len(all_global)} seeds done. "
                f"RMSE(last)={df_global_all['RMSE_last'].mean():.4f}"
                f"+/-{df_global_all['RMSE_last'].std():.4f}")
    return df_global_all, all_node_flat


# =============================================================================
# RUN SINGLE EXTERNAL MODEL
# =============================================================================
def run_external_model(cfg, model_name, shared, hz):
    logger.info(f"\n{'=' * 60}\n{model_name}  H={hz}\n{'=' * 60}")
    N = shared['N']
    # Issue 2: graph models use the NO-GEO feature set (F_in = 4), matching
    # the ablation V5 architecture.
    F_in_graph = shared['F_in_nogeo']
    adj_norm = shared['adj_norm']
    L_scaled = shared['L_scaled']
    meta = shared['meta']
    ts_arr = shared['ts_arr']
    test_idx = shared['test_idx']
    valid_mask = shared['valid_mask']

    model_out_dir = cfg.OUT_ROOT / f"H{hz}" / model_name
    model_out_dir.mkdir(parents=True, exist_ok=True)

    # 2026-09-21：GBDT 走独立分支 —— 它不是 nn.Module，无法复用 train_model/evaluate，
    # 但数据构造与评估口径仍与主流程完全一致（temporal 路径 + _decode_predict +
    # global_metrics），故结果可直接进入 comparison 汇总与配对检验。
    if model_name == 'GBDT':
        return run_gbdt_external(cfg, model_name, shared, hz, model_out_dir)

    mtype = MODEL_TYPES.get(model_name, 'temporal')
    if mtype == 'temporal':
        data = build_temporal_samples(shared, cfg, hz)
    else:
        data = build_samples_and_pca(shared, cfg, hz)

    Xtr = data['Xtr']; ytr = data['ytr']
    Xva = data['Xva']; yva = data['yva']
    Xte = data['Xte']; yte = data['yte']
    init_test = data['init_test']
    out_dim = data['out_dim']
    scY = data['scY']
    pca_y = data['pca_y']

    geo_static_np = shared['geo_static']
    n_points = len(cfg.POINTS)
    n_node = init_test.shape[1]

    # 路径 A（2026-09-17）：水位已不再进入模型，以下 water 特征仅作占位透传，
    # 使 DataLoader / forward 签名保持不变（模型内部不再使用 water_feat）。
    water_feat_full = shared['water_feat']
    _w_mean = water_feat_full[shared['train_idx']].mean(0)
    _w_std = water_feat_full[shared['train_idx']].std(0) + 1e-6
    water_feat_full = (water_feat_full - _w_mean) / _w_std
    wf_train = water_feat_full[shared['train_idx']][cfg.LOOKBACK - 1:-hz][:Xtr.shape[0]]
    wf_val = water_feat_full[shared['val_idx']][cfg.LOOKBACK - 1:-hz][:Xva.shape[0]]
    wf_test = water_feat_full[shared['test_idx']][cfg.LOOKBACK - 1:-hz][:Xte.shape[0]]

    all_global = []
    all_node = []
    failed_seeds = []

    for si, seed in enumerate(cfg.SEEDS):
        set_seed(seed)

        run_dir = model_out_dir / f"run_{seed}"
        # 2026-09-21：断点续跑（COMPARISON_RESUME=1），与消融判据一致——四件套齐全即
        # 视为已完成，读回结果参与汇总（metrics.json 无 seed 字段，须补记）。
        if cfg.RESUME and all((run_dir / f).exists()
                              for f in ('metrics.json', 'best.pth',
                                        'predictions.npz', 'losses.npy')):
            with open(run_dir / 'metrics.json', 'r') as f:
                gm = json.load(f)
            gm['seed'] = seed
            all_global.append(gm)
            nf = run_dir / 'node_metrics.xlsx'
            if nf.exists():
                df_n = pd.read_excel(nf)
                df_n['seed'] = seed
                all_node.append(df_n)
            logger.info(f"    Seed {si + 1}/{len(cfg.SEEDS)} ({seed}): "
                        f"结果已完整，跳过（RESUME=1）")
            continue

        if mtype == 'temporal':
            n_in = data['temporal_in_dim']
            model = get_model(cfg, model_name, N, n_in, out_dim, hz, adj_norm, L_scaled)
            train_ds = TensorDataset(torch.FloatTensor(Xtr), torch.FloatTensor(ytr))
            val_ds = TensorDataset(torch.FloatTensor(Xva), torch.FloatTensor(yva))
            test_ds = TensorDataset(torch.FloatTensor(Xte), torch.FloatTensor(yte))
        else:
            n_in = F_in_graph
            geo_dim_actual = geo_static_np.shape[1]
            model = get_model(cfg, model_name, N, n_in, out_dim, hz, adj_norm,
                              L_scaled, geo_dim=geo_dim_actual, n_points=n_points)

            wf_tr = np.tile(wf_train[:, np.newaxis, :], (1, N, 1)).astype(np.float32)
            wf_va = np.tile(wf_val[:, np.newaxis, :], (1, N, 1)).astype(np.float32)
            wf_te = np.tile(wf_test[:, np.newaxis, :], (1, N, 1)).astype(np.float32)

            geo_tr = data.get('geo_tr', np.tile(geo_static_np[np.newaxis], (Xtr.shape[0], 1, 1)))
            geo_va = data.get('geo_va', np.tile(geo_static_np[np.newaxis], (Xva.shape[0], 1, 1)))
            geo_te = data.get('geo_te', np.tile(geo_static_np[np.newaxis], (Xte.shape[0], 1, 1)))

            train_ds = TensorDataset(
                torch.FloatTensor(Xtr), torch.FloatTensor(ytr),
                torch.FloatTensor(wf_tr), torch.FloatTensor(geo_tr))
            val_ds = TensorDataset(
                torch.FloatTensor(Xva), torch.FloatTensor(yva),
                torch.FloatTensor(wf_va), torch.FloatTensor(geo_va))
            test_ds = TensorDataset(
                torch.FloatTensor(Xte), torch.FloatTensor(yte),
                torch.FloatTensor(wf_te), torch.FloatTensor(geo_te))

        train_ld = DataLoader(train_ds, batch_size=cfg.BATCH_SIZE, shuffle=True)
        val_ld = DataLoader(val_ds, batch_size=cfg.BATCH_SIZE)
        test_ld = DataLoader(test_ds, batch_size=cfg.BATCH_SIZE)

        save_path = model_out_dir / f"run_{seed}" / "best.pth"
        save_path.parent.mkdir(parents=True, exist_ok=True)

        try:
            n_params = count_params(model)
            # Issue 13: FLOPs estimation with proper per-type dummy inputs.
            flops = estimate_flops(model, cfg, mtype, Xtr[:1], wf_tr[:1] if mtype != 'temporal' else None,
                                   geo_tr[:1] if mtype != 'temporal' else None)
            model, train_time, tr_losses, va_losses, best_ep = train_model(
                cfg, model, train_ld, val_ld, save_path,
                model_tag=model_name, seed=seed,
                ours_lr_groups=(model_name in OURS_FRAMEWORK_MODELS))

            t_infer_start = time.time()
            pred_cum, true_cum = evaluate(model, test_ld, init_test, scY, pca_y, hz,
                                          cfg.DEVICE, true_diff_win=data['yte_raw'])
            infer_time = (time.time() - t_infer_start) / max(len(test_ld.dataset), 1)

            gm = global_metrics(true_cum, pred_cum)
            # 输出空间误差分解（原理见 `output_space_floor`）：地板与模型无关，把主口径
            # RMSE 无损拆为「可达 + 不可达下界」，满足 RMSE_full² = RMSE_reach² + RMSE_floor²。
            _fl = output_space_floor(data['yte_raw'], init_test, scY, pca_y, hz,
                                     ns=pred_cum.shape[0])
            gm['RMSE_floor_last'] = _fl['RMSE_floor_last']
            gm['RMSE_reach_last'] = float(np.sqrt(max(
                gm['RMSE_last'] ** 2 - _fl['MSE_floor_last'], 0.0)))
            gm['RMSE_floor'] = _fl['RMSE_floor_all']
            gm['RMSE_reach'] = float(np.sqrt(max(
                gm['RMSE'] ** 2 - _fl['MSE_floor_all'], 0.0)))
            gm['seed'] = seed
            gm['Params'] = n_params
            gm['FLOPs'] = flops
            gm['TrainTime'] = train_time
            gm['InferTime'] = infer_time
            gm['BestEpoch'] = best_ep
            gm['FinalTrainLoss'] = tr_losses[-1]
            gm['BestValLoss'] = min(va_losses)

            # g 诊断已移除——现在 g 逻辑已完全删除（见 §6.2 A/B 对照）。

            all_global.append(gm)

            nodes = [f"Node{i}" for i in range(n_node)]
            df_n = node_metrics(true_cum, pred_cum, nodes)
            df_n['seed'] = seed
            all_node.append(df_n)

            with open(save_path.parent / 'metrics.json', 'w') as f:
                json.dump(gm, f, indent=2)
            df_n.to_excel(save_path.parent / 'node_metrics.xlsx', index=False)
            np.savez(save_path.parent / 'predictions.npz',
                     pred_cum=pred_cum, true_cum=true_cum)
            loss_df = pd.DataFrame({'epoch': range(1, len(tr_losses) + 1),
                                    'train_loss': tr_losses,
                                    'val_loss': va_losses})
            loss_df.to_csv(save_path.parent / 'loss_history.csv', index=False)
            # 数据导出层（2026-09-18）
            np.save(save_path.parent / 'losses.npy', {'tr': tr_losses, 'va': va_losses})

            logger.info(f"    Seed {si + 1}/{len(cfg.SEEDS)} ({seed}) "
                        f"RMSE(last)={gm['RMSE_last']:.4f} RMSE(full)={gm['RMSE']:.4f} "
                        f"R2(last)={gm['R2_last']:.4f} "
                        f"BestEp={best_ep} BestValLoss={min(va_losses):.6f}")
        except Exception as e:
            logger.error(f"    Seed {seed} failed: {e}")
            failed_seeds.append(seed)
            import traceback
            traceback.print_exc()

        del model; gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    if not all_global:
        logger.error(f"  {model_name} H={hz}: ALL seeds failed!")
        return None, None

    if failed_seeds:
        logger.error(f"  !! {model_name} H={hz}: {len(failed_seeds)}/{len(cfg.SEEDS)} "
                     f"个种子失败 {failed_seeds} —— 汇总将写为 "
                     f"global_{len(all_global)}runs.csv，不能当作完整的 "
                     f"{len(cfg.SEEDS)} 种子结果使用")

    df_global_all = pd.DataFrame(all_global)
    all_node_flat = pd.concat(all_node, ignore_index=True)

    summary_dir = model_out_dir / "summary"
    summary_dir.mkdir(exist_ok=True)
    df_global_all.to_csv(summary_dir / f'global_{len(all_global)}runs.csv', index=False)

    if not all_node_flat.empty:
        depths_arr = meta['depths']
        point_names = meta['point_names']
        node_names = meta['node_names']
        node_to_depth = dict(zip([f"Node{i}" for i in range(len(depths_arr))], depths_arr))
        node_to_point = dict(zip([f"Node{i}" for i in range(len(point_names))], point_names))
        all_node_flat['Depth'] = all_node_flat['Node'].map(node_to_depth)
        all_node_flat['Point'] = all_node_flat['Node'].map(node_to_point)
        bins = [0, 3, 8, 16]
        labels = ['Shallow', 'Middle', 'Deep']
        all_node_flat['Layer'] = pd.cut(all_node_flat['Depth'], bins=bins,
                                         labels=labels, right=False)
        # 2026-09-20：三级汇总同时给出全步（原列名不变）与末步（`_last` 后缀）
        # 两套口径，口径定义见 global_metrics。
        layer_summary = _layer_point_summary(
            all_node_flat, ['Layer'], 'R2', 'RMSE', 'MAE').merge(
            _layer_point_summary(all_node_flat, ['Layer'],
                                 'R2_last', 'RMSE_last', 'MAE_last'),
            on='Layer', suffixes=('', '_last'))

        point_summary = _layer_point_summary(
            all_node_flat, ['Point'], 'R2', 'RMSE', 'MAE').merge(
            _layer_point_summary(all_node_flat, ['Point'],
                                 'R2_last', 'RMSE_last', 'MAE_last'),
            on='Point', suffixes=('', '_last'))

        node_summary = _node_summary(
            all_node_flat, 'R2', 'RMSE', 'MAE').merge(
            _node_summary(all_node_flat, 'R2_last', 'RMSE_last', 'MAE_last'),
            on='Node', suffixes=('', '_last'))
        node_summary['Depth'] = node_summary['Node'].map(node_to_depth)
        node_summary['Point'] = node_summary['Node'].map(node_to_point)
        all_node_flat.to_excel(summary_dir / 'node_summary.xlsx', index=False)
        layer_summary.to_csv(summary_dir / 'layer_summary.csv', index=False)
        point_summary.to_csv(summary_dir / 'point_summary.csv', index=False)
        node_summary.to_excel(summary_dir / 'node_summary_detail.xlsx', index=False)

    logger.info(f"  {model_name} H={hz}: {len(all_global)} seeds done. "
                f"RMSE(last)={df_global_all['RMSE_last'].mean():.4f}"
                f"+/-{df_global_all['RMSE_last'].std():.4f}  "
                f"RMSE(full)={df_global_all['RMSE'].mean():.4f}"
                f"+/-{df_global_all['RMSE'].std():.4f}")
    return df_global_all, all_node_flat


# =============================================================================
# LOAD ABLATION RESULTS (V0, V5 reuse)
# =============================================================================
def load_ablation_variant_result(ablation_root, variant_name, hz):
    mapping = {'V0_TCN-Only': 'V0_Pure_TCN', 'V7_Ours': 'V7_wo_VerticalGCN'}
    dir_name = mapping.get(variant_name, variant_name)
    vdir = ablation_root / f"H{hz}" / dir_name
    if not vdir.exists():
        logger.warning(f"  No ablation dir for {variant_name} H={hz} at {vdir}")
        return None, None
    run_dirs = sorted([d for d in vdir.iterdir() if d.is_dir() and d.name.startswith('run_')])
    if not run_dirs:
        return None, None
    global_list, node_list = [], []
    for rd in run_dirs:
        mf = rd / 'metrics.json'
        if not mf.exists():
            continue
        with open(mf, 'r') as f:
            gm = json.load(f)
        # 2026-09-21：评估口径修复后消融才写 `_last` 字段；旧格式（无 RMSE_last）与
        # 末步主口径不兼容。此前直接沿用会令 save_statistical_tests 取
        # `RMSE_last` 时 KeyError 崩在 main 循环（无保护）。此处跳过该 run 并告警。
        if 'RMSE_last' not in gm:
            logger.warning(f"  {rd.name}: metrics.json 缺 _last 字段（旧口径），"
                           f"跳过该 run —— 若整目录全是旧格式，V0/V5 两行将缺失")
            continue
        seed = int(rd.name.split('_')[1])
        gm['seed'] = seed
        global_list.append(gm)
        nf = rd / 'node_metrics.xlsx'
        if nf.exists():
            df_node = pd.read_excel(nf)
            df_node['seed'] = seed
            # 2026-09-21（格式契约）：`Node` 列由 Ablation_Experiment 的
            # `nodes = [f"{p}_{d:.1f}" for p in cfg.POINTS for d in cfg.DEPTHS]` 生成，
            # 形如 `CX2_0.5`（断面_深度，分隔符为下划线）。因此 split('_')[0] 取断面、
            # split('_')[1] 取深度；下方 `'_' in n` 分支只是防御性兜底，正常数据不会触发。
            df_node['Depth'] = [float(n.split('_')[1]) if '_' in str(n) else 0
                                for n in df_node['Node']]
            df_node['Point'] = [n.split('_')[0] if '_' in str(n) else 'unknown'
                                for n in df_node['Node']]
            node_list.append(df_node)
    if not global_list:
        return None, None
    df_global = pd.DataFrame(global_list)
    df_nodes = pd.concat(node_list, ignore_index=True) if node_list else pd.DataFrame()
    bins = [0, 3, 8, 16]
    labels = ['Shallow', 'Middle', 'Deep']
    if not df_nodes.empty and 'Depth' in df_nodes.columns:
        df_nodes['Layer'] = pd.cut(df_nodes['Depth'], bins=bins, labels=labels, right=False)
    return df_global, df_nodes


# =============================================================================
# RESULT TABLES（数据落盘；绘图已迁出 -> 可视化/02代码结果可视化/fig_comparison.py）
# =============================================================================

# ---- Statistical significance tables ----
def save_statistical_tests(df_global, hz, save_dir):
    models = [m for m in COMPARISON_MODELS if m in df_global['Model'].values]
    baseline = 'V7_Ours'
    if baseline not in models:
        return
    if 'seed' not in df_global.columns:
        logger.error("  df_global 缺 seed 列，无法进行种子级配对检验")
        return
    base_df = df_global[df_global['Model'] == baseline]
    stat_records = []
    for model in models:
        if model == baseline:
            continue
        sub = df_global[df_global['Model'] == model]
        # 2026-09-20 修复（Bug 1）：配对 t 检验必须按 seed 对齐。此前 V5 来自消融
        # 目录的字符串序、外部基线来自 SEEDS 循环序，直接位置配对全部错位。
        common = sorted(set(base_df['seed']) & set(sub['seed']))
        if len(common) < 5:
            continue
        b = base_df[base_df['seed'].isin(common)].set_index('seed').loc[common]
        c = sub[sub['seed'].isin(common)].set_index('seed').loc[common]
        # 2026-09-20：双口径。末步（`_last`，论文主口径）写出 `*_last_*` 列，
        # 全步沿用原列名；`Delta_RMSE_pct` / `Sig` 由末步口径驱动。
        fam = MODEL_FAMILY.get(model, 'unknown')
        rec = {'Comparison': f'{MODEL_LABELS[model]} vs {MODEL_LABELS[baseline]}',
               'Family': fam,
               'FamilyLabel': MODEL_FAMILY_LABELS.get(fam, fam),
               'N_Seeds': len(common)}
        for r2k, rmk, tag in (('R2_last', 'RMSE_last', '_last'),
                              ('R2', 'RMSE', '')):
            br, cr = b[rmk].values, c[rmk].values
            b2, c2 = b[r2k].values, c[r2k].values
            t_rmse, p_rmse = ttest_rel(br, cr)
            t_r2, p_r2 = ttest_rel(b2, c2)
            rec[f'RMSE{tag}_t'] = round(t_rmse, 3)
            rec[f'RMSE{tag}_p'] = f'{p_rmse:.4f}'
            rec[f'RMSE{tag}_Cohens_d'] = round(cohens_d(br, cr), 3)
            rec[f'R2{tag}_t'] = round(t_r2, 3)
            rec[f'R2{tag}_p'] = f'{p_r2:.4f}'
            rec[f'R2{tag}_Cohens_d'] = round(cohens_d(b2, c2), 3)
            if tag:
                rec['Delta_RMSE_pct'] = round(
                    (np.mean(cr) - np.mean(br)) / np.mean(br) * 100, 2)
                rec['Sig'] = ('***' if p_rmse < 0.001 else
                              ('**' if p_rmse < 0.01 else
                               ('*' if p_rmse < 0.05 else 'ns')))
        # 2026-09-22：可达分量（RMSE_reach_last）上的配对检验。公共地板在同一配对内的
        # 差分中抵消，故此处问的是"剔除不可达下界后，两模型的可达精度是否仍不同"；
        # `Delta_RMSE_reach_pct` 即论文的「可达退化%」。
        if 'RMSE_reach_last' in b.columns and 'RMSE_reach_last' in c.columns:
            brr, crr = b['RMSE_reach_last'].values, c['RMSE_reach_last'].values
            t_rr, p_rr = ttest_rel(brr, crr)
            rec['RMSE_reach_last_t'] = round(t_rr, 3)
            rec['RMSE_reach_last_p'] = f'{p_rr:.4f}'
            rec['RMSE_reach_last_Cohens_d'] = round(cohens_d(brr, crr), 3)
            rec['Delta_RMSE_reach_pct'] = round(
                (np.mean(crr) - np.mean(brr)) / np.mean(brr) * 100, 2)
        stat_records.append(rec)
    pd.DataFrame(stat_records).to_csv(save_dir / 'statistical_tests.csv', index=False)
    pd.DataFrame(stat_records).to_excel(save_dir / 'statistical_tests.xlsx', index=False)


# ---- Comprehensive summary table ----
def save_comprehensive_table(df_global, hz, save_dir):
    models = [m for m in COMPARISON_MODELS if m in df_global['Model'].values]
    metrics_list = ['RMSE', 'MAE', 'R2', 'MaxErr', 'DA', 'Params', 'TrainTime', 'InferTime']
    # 2026-09-20：末步口径（论文主口径）以 `_last` 后缀一并给出，定义见 global_metrics。
    metrics_list = ['RMSE_last', 'MAE_last', 'R2_last', 'MaxErr_last', 'DA_last'] + metrics_list
    # 2026-09-22：输出空间误差分解（定义见 output_space_floor）。RMSE_floor 为公共地板、
    # RMSE_reach 为可达分量（= 旧口径下的误差），满足 RMSE² = RMSE_reach² + RMSE_floor²。
    metrics_list = ['RMSE_floor_last', 'RMSE_reach_last',
                    'RMSE_floor', 'RMSE_reach'] + metrics_list
    metrics_avail = [mt for mt in metrics_list if mt in df_global.columns]
    records = []
    for model in models:
        sub = df_global[df_global['Model'] == model]
        fam = MODEL_FAMILY.get(model, 'unknown')
        row = {'Model': MODEL_LABELS[model], 'ModelKey': model,
               'Family': fam, 'FamilyLabel': MODEL_FAMILY_LABELS.get(fam, fam),
               'N_Seeds': len(sub)}
        for mt in metrics_avail:
            row[f'{mt}_mean'] = sub[mt].mean()
            row[f'{mt}_std'] = sub[mt].std()
        records.append(row)
    df_summary = pd.DataFrame(records)
    df_summary.to_csv(save_dir / 'comprehensive_comparison.csv', index=False)
    df_summary.to_excel(save_dir / 'comprehensive_comparison.xlsx', index=False)
    logger.info(f"  Comprehensive table: {len(records)} models")


def save_cross_horizon_tables(all_horizon_data, out_root):
    """跨视界两级汇总：逐种子明细 `cross_horizon_all.csv` + 均值/标准差 `cross_horizon_summary.csv`。

    2026-09-22：一并带入输出空间误差分解列（`RMSE_floor*` / `RMSE_reach*`，定义见
    `output_space_floor`）；缺列时自动跳过，不影响其余列（旧结果目录可直接复用）。
    """
    logger.info(f"\n{'#' * 60}\n#  CROSS-HORIZON ANALYSIS\n{'#' * 60}")
    cross_dir = out_root / "cross_horizon_comparison"
    cross_dir.mkdir(exist_ok=True)

    optional = ['Params', 'TrainTime', 'InferTime', 'DA', 'seed',
                'RMSE_floor_last', 'RMSE_reach_last', 'RMSE_floor', 'RMSE_reach']
    all_records = []
    for hz, df_g in all_horizon_data.items():
        for _, row in df_g.iterrows():
            rec = {'Horizon': hz, 'Model': row['Model'],
                   'ModelLabel': MODEL_LABELS.get(row['Model'], row['Model']),
                   'RMSE': row['RMSE'], 'R2': row['R2'],
                   'MAE': row['MAE'], 'MaxErr': row['MaxErr']}
            # 2026-09-20：末步口径（论文主口径）一并带入跨视界表。
            for m in ['RMSE', 'R2', 'MAE', 'MaxErr', 'DA']:
                if f'{m}_last' in df_g.columns and f'{m}_last' in row.index:
                    rec[f'{m}_last'] = row[f'{m}_last']
            for col in optional:
                if col in df_g.columns and col in row.index:
                    rec[col] = row[col]
            all_records.append(rec)
    df_cross = pd.DataFrame(all_records)
    df_cross.to_csv(cross_dir / 'cross_horizon_all.csv', index=False)

    agg = {'RMSE_mean': ('RMSE', 'mean'), 'RMSE_std': ('RMSE', 'std'),
           'R2_mean': ('R2', 'mean'), 'R2_std': ('R2', 'std'),
           'MAE_mean': ('MAE', 'mean'), 'MAE_std': ('MAE', 'std'),
           'MaxErr_mean': ('MaxErr', 'mean'), 'MaxErr_std': ('MaxErr', 'std'),
           'RMSE_last_mean': ('RMSE_last', 'mean'),
           'RMSE_last_std': ('RMSE_last', 'std'),
           'R2_last_mean': ('R2_last', 'mean'), 'R2_last_std': ('R2_last', 'std'),
           'MAE_last_mean': ('MAE_last', 'mean'), 'MAE_last_std': ('MAE_last', 'std'),
           'MaxErr_last_mean': ('MaxErr_last', 'mean'),
           'MaxErr_last_std': ('MaxErr_last', 'std')}
    for col in ['RMSE_floor_last', 'RMSE_reach_last', 'RMSE_floor', 'RMSE_reach']:
        if col in df_cross.columns:
            agg[f'{col}_mean'] = (col, 'mean')
            agg[f'{col}_std'] = (col, 'std')
    summary = df_cross.groupby(['Horizon', 'Model', 'ModelLabel']).agg(**agg).reset_index()
    summary.to_csv(cross_dir / 'cross_horizon_summary.csv', index=False)
    summary.to_excel(cross_dir / 'cross_horizon_summary.xlsx', index=False)
    logger.info(f"  Cross-horizon: 2 tables saved")


# =============================================================================
# MAIN DRIVER
# =============================================================================
def main(quick_test=False):
    cfg = Config()
    out_root = cfg.OUT_ROOT

    if quick_test:
        cfg.HORIZONS = [6]
        cfg.SEEDS = cfg.SEEDS[:2]
        cfg.EPOCHS = 5
        cfg.PATIENCE = 3
        logger.info("=" * 60)
        logger.info("QUICK TEST MODE: H=6 only, 2 seeds, 5 epochs")
        logger.info("=" * 60)
    else:
        logger.info("=" * 60)
        logger.info("FULL COMPARISON EXPERIMENT")
        logger.info(f"Models: {len(COMPARISON_MODELS)}")
        logger.info(f"Horizons: {cfg.HORIZONS}")
        logger.info(f"Seeds: {len(cfg.SEEDS)}")
        # GBDT 只在主展示步长上评估，故其视界数与其余模型不同（见 GBDT_HORIZON）。
        n_hz_gbdt = 1 if GBDT_HORIZON in cfg.HORIZONS else 0
        total = sum((n_hz_gbdt if m == 'GBDT' else len(cfg.HORIZONS))
                    for m in COMPARISON_MODELS if IS_EXTERNAL[m]) * len(cfg.SEEDS)
        logger.info(f"Total external runs: {total}")
        logger.info("=" * 60)

    logger.info(f"Device: {cfg.DEVICE}")
    logger.info(f"Output: {out_root}")

    # Shared data
    shared = prepare_shared_data(cfg)
    meta = shared['meta']

    all_horizon_data = {}
    external_models = [m for m in COMPARISON_MODELS if IS_EXTERNAL[m]]
    reuse_models = [m for m in COMPARISON_MODELS if not IS_EXTERNAL[m]]

    for hz in cfg.HORIZONS:
        logger.info(f"\n{'#' * 60}\n#  HORIZON H={hz}\n{'#' * 60}")

        # Run external models
        all_df_global = []
        for model_name in external_models:
            if model_name == 'GBDT' and hz != GBDT_HORIZON:
                logger.info(f"  跳过 {model_name} @ H={hz}（仅评估 H={GBDT_HORIZON}）")
                continue
            df_g, _df_n = run_external_model(cfg, model_name, shared, hz)
            if df_g is not None:
                df_g['Model'] = model_name
                all_df_global.append(df_g)

        # Load reuse models (V0, V7) from ablation
        for model_name in reuse_models:
            df_g, _df_n = load_ablation_variant_result(cfg.ABLATION_ROOT, model_name, hz)
            if df_g is not None:
                df_g['Model'] = model_name
                all_df_global.append(df_g)
                logger.info(f"  Loaded {model_name} H={hz}: "
                            f"RMSE(last)={df_g['RMSE_last'].mean():.4f}"
                            f"+/-{df_g['RMSE_last'].std():.4f}")

        if not all_df_global:
            logger.warning(f"  No results for H={hz}!")
            continue

        df_global_all = pd.concat(all_df_global, ignore_index=True)

        comp_dir = out_root / f"H{hz}" / "comparison"
        comp_dir.mkdir(parents=True, exist_ok=True)
        # 数据导出层（2026-09-18）：逐种子明细宽表（含全部 30 种子，非仅均值）
        df_global_all.to_pickle(comp_dir / 'full_seed_data.pkl')
        # 结果表落盘（绘图已迁出 -> 可视化/02代码结果可视化/fig_comparison.py）
        save_statistical_tests(df_global_all, hz, comp_dir)
        save_comprehensive_table(df_global_all, hz, comp_dir)
        all_horizon_data[hz] = df_global_all

    # Cross-horizon summary tables
    if len(all_horizon_data) >= 2:
        save_cross_horizon_tables(all_horizon_data, out_root)
    else:
        logger.info("Single horizon only -- skipping cross-horizon analysis.")

    logger.info(f"\n{'=' * 60}")
    logger.info("ALL DONE! Results saved to: " + str(out_root))
    logger.info(f"{'=' * 60}")


if __name__ == '__main__':
    quick = '--quick' in sys.argv or '-q' in sys.argv
    main(quick_test=quick)