# =============================================================================
# Ablation_Experiment.py -- Chapter 4 Ablation Study for MST Journal
# 8 Variants (V0-V3,V5-V8) x 5 Horizons x 30 Seeds
#
# Variants:
#   V0  = Pure TCN baseline (all components OFF)
#   V1  = Last-value pooling (vs. mean pooling in V5; 2026-09-20 由 V1_wo_TemporalAttn 更名)
#   V2  = Remove horizontal GCN (2026-09-20 由 V2_wo_GCN_Water 更名)
#   V3  = Remove Physics Gate
#   V5  = Full model (Ours)
#   V6  = Remove Cross-Node Attention
#   V7  = Remove Vertical GCN (A^vert channel only)
#   V8  = Remove ALL vertical coupling (V7 + same-hole edges zeroed, ZERO_SAME_HOLE)
#
# 路径 A（2026-09-17）：彻底移除水文输入（VICD 诊断表明水位在增量口径无可用
# 信息）。故原 V4（无水文增强）不再作为独立消融变体（8->7）。
# 2026-09-18 新增 V8_wo_VerticalAll（干净竖向消融：关竖向 GCN 的同时把 A 中
# 同孔节点对整体置 0，见 §3.9），变体数回到 8（无 V4）。
# 2026-09-20：V1/V2 变体名更新（语义随架构简化/路径 A 变化）。
# All variants: USE_GEO_INJECT = False（Config 全局默认，变体字典不再重复列出）
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
from matplotlib.colors import Normalize
import seaborn as sns
from scipy.stats import ttest_rel
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
    # 2026-09-20：默认输出根由已作废的 `Ablation_New`（旧 12 维含水架构）改为当前
    # 4 维架构的 `Ablation_New_5d`。此前靠 `_rerun_full_5d.py` 覆盖才写对位置，
    # 一旦直接运行本文件就会把新架构结果写进旧目录、与旧结果混在一起。
    # `Ablation_New` 目录本体不动，仍作归档。
    OUT_ROOT = AUX_ROOT / "Ablation_New_5d"
    OUT_ROOT.mkdir(parents=True, exist_ok=True)

    POINTS = ['CX2', 'CX3', 'CX8', 'CX13', 'CX15']
    DEPTHS = np.arange(0.5, 16.0, 0.5)
    LOOKBACK = 12
    HORIZONS = [2, 6, 8, 10, 12]  # unified across all four experiments (issue 9)
    TRAIN_RATIO = 0.70
    VAL_RATIO = 0.15

    TCN_CHANNELS = [64, 64]
    TCN_KERNEL_SIZE = 7
    TCN_DILATIONS = [1, 2]
    TCN_DROPOUT = 0.3
    GCN_HIDDEN = 64
    DROPOUT = 0.5
    ATTENTION_HIDDEN = 32  # 已弃用（2026-09-19 删时序注意力）；保留仅供诊断脚本 import 兼容

    BATCH_SIZE = 32
    # 论文口径为 EPOCHS=400 / 早停 40（默认值，未设环境变量时行为完全不变）。
    # 放开环境变量仅用于在示例数据上做缩小规模的端到端演示（见 README "Quick start"）。
    EPOCHS = int(os.environ.get('ABLATION_EPOCHS', '400'))
    PATIENCE = int(os.environ.get('ABLATION_PATIENCE', '40'))
    LR = 0.001
    LR_GCN = 0.0005
    WD = 1e-3
    GRAD_CLIP = 1.0

    USE_GEO_INJECT = False  # 全局恒 False（地质参数不进 TCN 输入，仅经邻接/物理门/跨节点注意力注入）
    USE_CROSS_NODE_ATTN = True
    USE_PHYSICS_LOSS = True  # 命名沿用历史（语义=是否创建 PhysicsGate 模块，不改变损失）
    USE_GCN = True
    USE_ATTN = True
    USE_VERTICAL_GCN = True
    # 式(9) 的纯几何高斯核之上叠加的两处结构性先验（正文 §2.3 有对应说明）。
    # 二者均为固定常数、不参与训练；置 False 可退化为纯几何邻接用于对照。
    SPA_WITHIN_HOLE_PRIOR = True   # 同孔相邻深度边权重 0.8（结构连续性）
    SPA_SHALLOW_PRIOR = True       # 浅层(<3 m)跨断面节点对 0.6，并豁免 3σ 剪枝
    # 消融 V8 专用开关（2026-09-18）：把同孔节点对（同断面、i≠j）的边在 A^spa 与
    # A^geo 混合后整体置 0，用于彻底切断水平图传播路径上的竖向耦合。
    # 注意：仅置 SPA_WITHIN_HOLE_PRIOR=False 达不到该目的——同孔其余深度对的
    # 纯几何核权重仍≈1.0，且 A^geo 在同一地质层内亦为 1.0。
    ZERO_SAME_HOLE = False

    # 水文相关开关已全部移除（路径 A，2026-09-17）：曾试过 g_logit 门控、逐节点
    # WaterGate 与水位特征注入，均无正贡献（VICD 诊断 §3.4 + 正文 §6.3 方向 B 对照）。
    # water_feat 入参仅作 DataLoader 签名占位，不进入模型。

    GEO_FEAT_DIM = 6
    WAVELET = 'db4'
    WAVELET_LEVEL = 1
    THRESH_MODE = 'soft'
    # 30 个种子：原 15 个保留（与既有实验可比），新增 15 个由
    # numpy.random.RandomState(2026) 在 [1,2000] 无放回抽样、排除已用种子得到，
    # 与实验结果无关，见 _gen_seeds.py。
    SEEDS = [42, 15, 77, 85, 120, 532, 901, 23, 12, 14, 19, 75, 250, 140, 135,
             1180, 1262, 1284, 605, 1159, 746, 1657, 61, 27, 993, 897, 276,
             1308, 896, 1866]
    # 断点续跑：写盘偶发拦截可能中断长跑，置 1 时跳过已完整产出的种子
    RESUME = os.environ.get('ABLATION_RESUME', '0') == '1'
    # 2026-09-20：`AMP`（torch.cuda.amp 自动混合精度）从未接入训练循环，已删除。
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# =============================================================================
# ABLATION VARIANTS
# =============================================================================
ABLATION_VARIANTS = {
    # V0（2026-09-21 修正）：USE_ATTN=True —— 与 V5 同为均值池化，使本变体只差
    # "全部空间模块"，不再混入池化改变。实测该改动影响 +0.45%（p=0.59，H=10×30
    # seeds，见 _v0pool_probe / _v0pool_caliber_check）→ 旧"末值池化"的混杂可忽略，
    # 但改为同口径后消融格子更干净（V0 vs V5 = 纯空间效应，V1 vs V5 = 纯池化效应）。
    "V0_Pure_TCN": {
        "USE_GCN": False, "USE_ATTN": True,
        "USE_CROSS_NODE_ATTN": False,
        "USE_PHYSICS_LOSS": False, "USE_VERTICAL_GCN": False
    },
    # V1（2026-09-20 更名，原 V1_wo_TemporalAttn）：架构简化后主模型为显式均值
    # 池化，本变体用末值池化（USE_ATTN=False→last）作"平滑 vs 末值"对照。
    "V1_wo_MeanPool": {
        "USE_GCN": True, "USE_ATTN": False,
        "USE_CROSS_NODE_ATTN": True,
        "USE_PHYSICS_LOSS": True, "USE_VERTICAL_GCN": True
    },
    # V2（2026-09-20 更名，原 V2_wo_GCN_Water）：仅关水平图传播，Water 后缀已随
    # 路径 A 移除水文而失去语义。
    "V2_wo_GCN": {
        "USE_GCN": False, "USE_ATTN": True,
        "USE_CROSS_NODE_ATTN": True,
        "USE_PHYSICS_LOSS": True, "USE_VERTICAL_GCN": True
    },
    "V3_wo_PhysicsGate": {
        "USE_GCN": True, "USE_ATTN": True,
        "USE_CROSS_NODE_ATTN": True,
        "USE_PHYSICS_LOSS": False, "USE_VERTICAL_GCN": True
    },
    "V5_Full_Model": {
        "USE_GCN": True, "USE_ATTN": True,
        "USE_CROSS_NODE_ATTN": True,
        "USE_PHYSICS_LOSS": True, "USE_VERTICAL_GCN": True
    },
    "V6_wo_CrossNodeAttn": {
        "USE_GCN": True, "USE_ATTN": True,
        "USE_CROSS_NODE_ATTN": False,
        "USE_PHYSICS_LOSS": True, "USE_VERTICAL_GCN": True
    },
    "V7_wo_VerticalGCN": {
        "USE_GCN": True, "USE_ATTN": True,
        "USE_CROSS_NODE_ATTN": True,
        "USE_PHYSICS_LOSS": True, "USE_VERTICAL_GCN": False
    },
    # V8（2026-09-18 新增）：干净竖向消融。V7 只关掉了 A^vert 通道，而水平图传播
    # 所用的 A 中仍含同孔节点间的竖向耦合（归一化后约占 25%，来自 A^spa 的纯几何核
    # 与 A^geo 的同地质层边），故 V7 并非"无竖向"。V8 在 V7 基础上再把同孔节点对
    # 整体置 0，才是真正切断竖向信息传播的对照。
    "V8_wo_VerticalAll": {
        "USE_GCN": True, "USE_ATTN": True,
        "USE_CROSS_NODE_ATTN": True,
        "USE_PHYSICS_LOSS": True, "USE_VERTICAL_GCN": False,
        "ZERO_SAME_HOLE": True
    }
}

VARIANT_SHORT_NAMES = {
    "V0_Pure_TCN": "V0_TCN",
    "V1_wo_MeanPool": "V1_lastPool",
    "V2_wo_GCN": "V2_woGCN",
    "V3_wo_PhysicsGate": "V3_woPhys",
    "V5_Full_Model": "V5_Full",
    "V6_wo_CrossNodeAttn": "V6_woCross",
    "V7_wo_VerticalGCN": "V7_woVert",
    "V8_wo_VerticalAll": "V8_woVertAll"
}

VARIANT_COLORS = {
    "V0_Pure_TCN": "#EE6677",
    "V1_wo_MeanPool": "#CCBB44",
    "V2_wo_GCN": "#BBBBBB",
    "V3_wo_PhysicsGate": "#228833",
    "V5_Full_Model": "#4477AA",
    "V6_wo_CrossNodeAttn": "#AA3377",
    "V7_wo_VerticalGCN": "#EE7733",
    "V8_wo_VerticalAll": "#44AA99"
}

VARIANT_MARKERS = {
    "V0_Pure_TCN": "o",
    "V1_wo_MeanPool": "s",
    "V2_wo_GCN": "^",
    "V3_wo_PhysicsGate": "D",
    "V5_Full_Model": "p",
    "V6_wo_CrossNodeAttn": "*",
    "V7_wo_VerticalGCN": "X",
    "V8_wo_VerticalAll": "v"
}

VARIANT_LINESTYLES = {
    "V0_Pure_TCN": "-",
    "V1_wo_MeanPool": "--",
    "V2_wo_GCN": "-.",
    "V3_wo_PhysicsGate": ":",
    "V5_Full_Model": (0, (5, 2)),
    "V6_wo_CrossNodeAttn": (0, (1, 1)),
    "V7_wo_VerticalGCN": (0, (4, 1, 1, 1, 1, 1)),
    "V8_wo_VerticalAll": (0, (3, 1, 1, 1))
}

VARIANT_HATCHES = {
    "V0_Pure_TCN": "//",
    "V1_wo_MeanPool": "\\\\",
    "V2_wo_GCN": "||",
    "V3_wo_PhysicsGate": "--",
    "V5_Full_Model": "xx",
    "V6_wo_CrossNodeAttn": "oo",
    "V7_wo_VerticalGCN": "**",
    "V8_wo_VerticalAll": "++"
}

# matplotlib / seaborn style for SCI
plt.rcParams.update({
    'font.family': 'serif', 'font.serif': ['Times New Roman'],
    'font.size': 15, 'axes.labelsize': 17, 'axes.titlesize': 18,
    'legend.fontsize': 14, 'figure.dpi': 150, 'savefig.dpi': 600,
    'savefig.bbox': 'tight', 'axes.grid': True, 'grid.alpha': 0.3
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


def wavelet_denoise(signal, cfg):
    level = min(cfg.WAVELET_LEVEL, int(np.log2(len(signal))) - 1)
    if level < 1:  # guard, consistent with the other three scripts (issue 10)
        return signal
    coeffs = pywt.wavedec(signal, cfg.WAVELET, level=level)
    sigma = np.median(np.abs(coeffs[-1])) / 0.6745
    thr = sigma * np.sqrt(2 * np.log(len(signal)))
    coeffs[1:] = [pywt.threshold(c, thr, cfg.THRESH_MODE) for c in coeffs[1:]]
    rec = pywt.waverec(coeffs, cfg.WAVELET)
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
# DATA LOADING (reused from original pipeline)
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
    # 2026-09-20：此前缺列静默填 0，坏数据会一路拖到 PCA.fit 才炸。网格必须完整，
    # 否则下游的 node_mask 只是把 ctx 乘 0（不是真掩码，见 AblationModel.forward），
    # 反而把缺失节点的预测污染成 0 而不报错。改为在此处显式终止。
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


def build_adjacency(meta, geo_feat, cfg, zero_same_hole=False):
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

    # 消融 V8（2026-09-18）：彻底切断竖向耦合。同孔节点之间除 A^spa 的相邻深度先验外，
    # 还有两条通路——同孔其余深度对的纯几何核（权重≈1.0）与 A^geo 的同地质层边（=1.0），
    # 因此必须在二者混合之后整体置 0，只关 SPA_WITHIN_HOLE_PRIOR 是无效的。
    same_hole_mask = None
    if zero_same_hole:
        same_hole_mask = np.zeros((N, N), dtype=bool)
        for p in range(n_points):
            s = p * n_dep
            same_hole_mask[s:s + n_dep, s:s + n_dep] = True
        np.fill_diagonal(same_hole_mask, False)
        adj[same_hole_mask] = 0.0

    # 3σ 距离剪枝；浅层跨断面边被显式豁免——浅层相邻断面间距本身接近 σ，
    # 若一并剪除会抹掉悬臂段的耦合通道。
    adj[dist > 3 * sigma] = 0
    if getattr(cfg, 'SPA_SHALLOW_PRIOR', True):
        shallow_keep = (0.7 * adj_spatial + 0.3 * geo_sim
                        if geo_sim is not None else adj_spatial).copy()
        np.fill_diagonal(shallow_keep, 1.0)
        if same_hole_mask is not None:
            shallow_keep[same_hole_mask] = 0.0   # 豁免恢复时不得把同孔边放回来
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
        tr_d[:, i] = wavelet_denoise(train[:, i], cfg)
        v_d[:, i] = wavelet_denoise(val[:, i], cfg)
        te_d[:, i] = wavelet_denoise(test[:, i], cfg)
    return tr_d, v_d, te_d


def make_features(disp_raw, diff1, meta, wf, geo_static, cfg, use_water=True):
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
    # 直接拼进 TCN 基准输入，而 VICD 诊断（§3.4）表明水位与位移在增量口径无可用
    # 信息（增量相关≈0、归一化互信息≈0.04、符号对齐低于随机），故不再作为输入。
    # 输入特征由 12 维降为 4 维。
    base = np.concatenate(feat_list, axis=-1)
    if cfg.USE_GEO_INJECT and geo_static is not None:
        geo_tile = np.tile(geo_static[np.newaxis, :, :], (T, 1, 1))
        return np.concatenate([base, geo_tile], axis=-1).astype(np.float32)
    return base.astype(np.float32)


# =============================================================================
# MODEL ARCHITECTURE
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
    def __init__(self, dim, geo_dim=6):
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

    def __init__(self, dim, geo_dim=6):
        super().__init__()
        self.dim = dim
        self.q = nn.Linear(dim, dim)
        self.k = nn.Linear(dim, dim)
        self.v = nn.Linear(dim, dim)
        self.gate = nn.Linear(dim + geo_dim, 1)
        self.last_attn = None

    def forward(self, x, geo_feat):
        B, N, D = x.shape
        q = self.q(x)
        k = self.k(x)
        v = self.v(x)
        attn = torch.softmax(torch.matmul(q, k.transpose(-2, -1)) / np.sqrt(D), dim=-1)
        self.last_attn = attn.detach()
        cross = torch.matmul(attn, v)
        gate = torch.sigmoid(self.gate(torch.cat([x, geo_feat], dim=-1)))
        return gate * x + (1 - gate) * cross


class VerticalGCN(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.vert_proj = nn.Sequential(
            nn.Linear(dim, dim),
            nn.ReLU(),
            nn.Linear(dim, dim)
        )

    def forward(self, x, vert_adj):
        B, N, D = x.shape
        vert_out = self.vert_proj(x)
        vert_agg = torch.bmm(vert_adj.expand(B, N, N), vert_out)
        return x + vert_agg


class AblationModel(nn.Module):
    def __init__(self, cfg, N, in_dim, gcn_hid, tcn_ch, k, dil, attn_hid, hor,
                 adj, out_dim, geo_feat_dim=6, vert_adj=None):
        super().__init__()
        self.use_attn = cfg.USE_ATTN
        self.use_gcn = cfg.USE_GCN
        self.use_cross_attn = cfg.USE_CROSS_NODE_ATTN
        self.use_physics_gate = cfg.USE_PHYSICS_LOSS
        # 路径 A（2026-09-17）：不再使用水文。此前 use_water_gcn 控制 water_node_proj
        # 分支，VICD 诊断（§3.4）表明水位与位移在增量口径无可用信息，故彻底移除。
        self.use_water_gcn = False
        self.use_vertical_gcn = getattr(cfg, 'USE_VERTICAL_GCN', False)

        self.tcn = TCNEncoder(in_dim, tcn_ch, k, dil, cfg.TCN_DROPOUT)
        tcn_dim = tcn_ch[-1]

        # 架构简化（2026-09-19）：删除时序注意力模块（attn1/attn_v，约 2.1k 参数）。
        # 实测（_attn_diag/_attn_pool_cmp/_static_weight_probe）表明：TCN 有效感受野
        # 37 步 > 输入窗 12 步，注意力池化的是高度冗余的向量，权重退化为近均匀
        # （cos(ctx_attn, ctx_mean)=0.99975），三种池化（attn/mean/last）下游误差差
        # <0.05%，消融 V1 亦不显著。故 USE_ATTN=True 改为显式均值池化（0 参数），
        # USE_ATTN=False（消融 V1）保持末值池化，保留"平滑 vs 末值"对照语义。
        # 注意：删除模块会改变 PyTorch RNG 消费顺序 → 所有变体必须用本版代码统一重跑。

        if self.use_gcn:
            gcn_in_dim = tcn_dim
            self.gcn = GatedGCN(gcn_in_dim, gcn_hid, adj, dropout=cfg.DROPOUT)
            self.gcn_norm = nn.LayerNorm(gcn_hid)
            self.gcn_drop = nn.Dropout(cfg.DROPOUT)
            cur_dim = gcn_hid
        else:
            cur_dim = tcn_dim

        # 2026-09-20：原此处另注册 `static_adj` buffer，与 `self.gcn.adj`（GatedGCN
        # 内部 L599）指向同一张量，属重复注册；全仓库无任何消费者，已删除。
        if vert_adj is not None:
            self.register_buffer('vert_adj', vert_adj)
        else:
            self.vert_adj = None

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
            nn.Linear(cur_dim // 2, hor * out_dim)
        )
        self.horizon = hor
        self.out_dim = out_dim
        # 2026-09-20：`_last_temporal_attn` 死属性已删除（架构简化后无任何消费者）。

    def forward(self, x, water_feat, geo_static, vert_adj=None, node_mask=None):
        B, seq, Nodes, in_feat = x.shape
        tcn_in = x.permute(0, 2, 1, 3).reshape(B * Nodes, seq, in_feat)
        tcn_out = self.tcn(tcn_in)

        if self.use_attn:
            # 架构简化（2026-09-19）：原单头加性注意力改为显式均值池化（见 __init__ 注释）。
            ctx = tcn_out.mean(dim=1)
        else:
            ctx = tcn_out[:, -1, :]
        ctx = ctx.reshape(B, Nodes, -1)

        # 2026-09-20 说明：这不是真正的掩码——TCN 已消费过该节点的整段序列，且竖向
        # MLP / 物理门 / 跨节点注意力都带偏置，乘 0 会被部分恢复。配合
        # build_node_features 的网格完整性断言，valid_mask 现恒为全 1，本行是
        # 可证的恒等操作，仅为保持 forward 签名兼容而保留。
        if node_mask is not None and ctx.shape[1] == node_mask.shape[0]:
            ctx = ctx * node_mask.view(1, -1, 1).to(ctx.device)

        # 路径 A（2026-09-17）：已移除水文注入（water_node_proj 分支）。此前会把水位
        # 投影拼接进 GCN 输入；VICD 诊断（§3.4）表明其无可利用信息且引入复杂度惩罚，
        # 故彻底移除。water_feat 入参仅作签名占位以兼容既有 DataLoader。

        if self.use_vertical_gcn:
            vadj = vert_adj if vert_adj is not None else self.vert_adj
            if vadj is not None:
                ctx = self.vertical_gcn(ctx, vadj)

        if self.use_gcn:
            ctx = self.gcn(ctx, dynamic_adj=None)
            ctx = self.gcn_norm(ctx)
            # 注意：GatedGCN 内部已做一次 Dropout(0.5)+ReLU（见其 forward），此处
            # 再叠 LayerNorm + Dropout(0.5) + ReLU 属双重正则化（有效保留率约 25%）。
            # 行为已被全部历史实验锁定，勿"修正"——论文 §4.3 的"GCN 为 0.5"据此如实更新。
            ctx = self.gcn_drop(F.relu(ctx))

        if self.use_physics_gate:
            ctx = self.physics_gate(ctx, geo_static)

        if self.use_cross_attn:
            ctx = self.cross_attn(ctx, geo_static)

        global_feat = ctx.mean(dim=1)
        out = self.head(global_feat)
        return out.reshape(B, self.horizon, self.out_dim), None


class MultiLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.huber = nn.HuberLoss(delta=1.0)
        self.mse = nn.MSELoss()

    def forward(self, pred, target, epoch=None):
        B, hor, dim = pred.shape
        l_hub = self.huber(pred, target)
        l_diff = torch.tensor(0., device=pred.device)
        if hor > 1:
            l_diff = self.mse(pred[:, 1:] - pred[:, :-1], target[:, 1:] - target[:, :-1])
        pred_fft = torch.fft.rfft(pred, dim=1)
        target_fft = torch.fft.rfft(target, dim=1)
        l_freq = self.mse(torch.abs(pred_fft), torch.abs(target_fft))
        return 1.0 * l_hub + 0.5 * l_diff + 0.2 * l_freq


# =============================================================================
# DATA PREPARATION
# =============================================================================
def prepare_data(cfg, full_df, wd, wa, meta, train_idx, val_idx, test_idx,
                 water_feat, geo_feat, hz, disp_raw=None):
    # 2026-09-20：disp_raw 由调用方传入（main 已算过一次），避免每个变体重复执行
    # build_node_features 里的 pivot_table。未传入时保持原行为。
    if disp_raw is None:
        disp_raw, _, _ = build_node_features(full_df, cfg)
    valid_mask = meta.get('valid_mask', np.ones(disp_raw.shape[1], dtype=np.float32))
    train_disp = disp_raw[train_idx]
    val_disp = disp_raw[val_idx]
    test_disp = disp_raw[test_idx]
    train_den, val_den, test_den = denoise_split(train_disp, val_disp, test_disp, cfg)
    all_den = np.vstack([train_den, val_den, test_den])
    diff1_den = np.zeros_like(all_den)
    # 首步差分无前值，置 0（左相位一阶差分；该边界行仅在首窗第 1 个时刻出现，
    # 且被 LOOKBACK 窗口覆盖，对训练目标的影响可忽略——如实记录而非隐式默认）。
    if len(meta['timestamps']) > 1:
        diff1_den[1:] = np.diff(all_den, axis=0)
    geo_static_raw = np.concatenate([geo_feat, (meta['depths'] / 16.0).reshape(-1, 1)],
                                    axis=1)
    geo_static = (geo_static_raw - geo_static_raw.mean(0)) / (
            geo_static_raw.std(0) + 1e-6)
    train_feat = make_features(train_den, diff1_den[train_idx], meta,
                               water_feat[train_idx], geo_static, cfg, False)
    val_feat = make_features(val_den, diff1_den[val_idx], meta,
                             water_feat[val_idx], geo_static, cfg, False)
    test_feat = make_features(test_den, diff1_den[test_idx], meta,
                              water_feat[test_idx], geo_static, cfg, False)
    F = train_feat.shape[-1]
    y_train = diff1_den[train_idx]
    y_val = diff1_den[val_idx]
    y_test = diff1_den[test_idx]
    scX = StandardScaler().fit(train_feat.reshape(-1, F))
    tr_norm = scX.transform(train_feat.reshape(-1, F)).reshape(train_feat.shape)
    va_norm = scX.transform(val_feat.reshape(-1, F)).reshape(val_feat.shape)
    te_norm = scX.transform(test_feat.reshape(-1, F)).reshape(test_feat.shape)
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
    # 评估口径（2026-09-20）：测试集真值窗口直接切原始一阶差分 y_test，不经 PCA
    # 投影、不经标准化。窗口 i 的目标行为 test_idx[LOOKBACK+i : LOOKBACK+i+hz]，
    # 与 init_test（起始绝对位移）累加后正好还原该区间被去噪后的绝对位移。
    yte_raw = np.stack([y_test[i + cfg.LOOKBACK:i + cfg.LOOKBACK + hz]
                        for i in range(Xte.shape[0])]) if Xte.shape[0] else np.empty(
        (0, hz, y_test.shape[1]), dtype=np.float32)
    init_test = test_den[cfg.LOOKBACK - 1:-hz][:Xte.shape[0]]
    N = train_disp.shape[1]
    # 式(13) 的 w_t（同时也是 water_node_proj 的输入）按训练集统计量做 Z-score 标准化，
    # 与输入特征同一策略（§4.3：所有标准化参数仅由训练集计算）。水位各通道量纲差异大
    # （米级水位深度 vs 厘米级一阶差分），不标准化会使 sigmoid 迅速饱和。
    w_mean = water_feat[train_idx].mean(0)
    w_std = water_feat[train_idx].std(0) + 1e-6
    water_feat_s = (water_feat - w_mean) / w_std
    wf_train = water_feat_s[train_idx][cfg.LOOKBACK - 1:-hz][:Xtr.shape[0]]
    wf_val = water_feat_s[val_idx][cfg.LOOKBACK - 1:-hz][:Xva.shape[0]]
    wf_test = water_feat_s[test_idx][cfg.LOOKBACK - 1:-hz][:Xte.shape[0]]
    wf_train = np.tile(wf_train[:, np.newaxis, :], (1, N, 1))
    wf_val = np.tile(wf_val[:, np.newaxis, :], (1, N, 1))
    wf_test = np.tile(wf_test[:, np.newaxis, :], (1, N, 1))
    return {'Xtr': Xtr, 'ytr': ytr, 'Xva': Xva, 'yva': yva, 'Xte': Xte, 'yte': yte,
            'yte_raw': yte_raw, 'init_test': init_test, 'scX': scX, 'scY': scY, 'pca_y': pca_y,
            'out_dim': out_dim, 'F_in': F, 'N': N,
            'wf_train': wf_train, 'wf_val': wf_val, 'wf_test': wf_test,
            'geo_static': geo_static, 'geo_dim_actual': geo_static.shape[1],
            'valid_mask': valid_mask}


# =============================================================================
# TRAINING
# =============================================================================
def _graph_lr_param_groups(model):
    """图/物理相关模块参数（低学习率组）与其余参数。

    2026-09-20：改为**按模块对象显式收集**，不再依赖参数名子串匹配。此前
    'gcn' 子串会顺带命中 gcn_norm，而 vertical_gcn 的参数名
    vertical_gcn.vert_proj.* 只能靠 'gcn' 巧合命中——一旦重命名即静默失效。
    变体关闭时对应模块不存在，用 getattr 容错；返回顺序保持
    model.named_parameters() 的原顺序，与原实现逐位等价。
    """
    holders = [getattr(model, n, None) for n in
               ('gcn', 'gcn_norm', 'gcn_drop', 'physics_gate',
                'cross_attn', 'vertical_gcn')]
    gid = {id(p) for m in holders if m is not None for p in m.parameters()}
    gra = [p for _, p in model.named_parameters() if id(p) in gid]
    rest = [p for _, p in model.named_parameters() if id(p) not in gid]
    return gra, rest


def train_model(cfg, model, train_ld, val_ld, save_path, node_mask=None,
                criterion=None):
    """训练循环。

    2026-09-23：新增可选 `criterion`（默认 None → 原生 MultiLoss，行为不变）。
    Spatial_LOOCV 用它注入"节点空间 + 留出断面掩码"的损失，以修复原生
    MultiLoss 在 PCA 系数空间计算、无法按节点掩码导致的真值泄露。
    """
    gra, rest = _graph_lr_param_groups(model)
    opt = torch.optim.AdamW([
        {'params': gra, 'lr': cfg.LR_GCN},
        {'params': rest, 'lr': cfg.LR}
    ], weight_decay=cfg.WD)
    sch = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode='min',
                                                      factor=0.5, patience=15)
    crit = (criterion.to(cfg.DEVICE) if criterion is not None
            else MultiLoss().to(cfg.DEVICE))
    best_vloss = float('inf')
    wait = 0
    tr_losses, va_losses = [], []
    t_start = time.time()
    for ep in range(1, cfg.EPOCHS + 1):
        model.train()
        tr_loss = 0
        for batch in train_ld:
            Xb, yb, wfb, gsb = [t.to(cfg.DEVICE) for t in batch]
            opt.zero_grad()
            pred, _ = model(Xb, wfb, gsb, node_mask=node_mask)
            loss = crit(pred, yb, epoch=ep)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.GRAD_CLIP)
            opt.step()
            tr_loss += loss.item() * Xb.size(0)
        tr_loss /= len(train_ld.dataset)
        model.eval()
        va_loss = 0
        with torch.no_grad():
            for batch in val_ld:
                Xb, yb, wfb, gsb = [t.to(cfg.DEVICE) for t in batch]
                pred, _ = model(Xb, wfb, gsb, node_mask=node_mask)
                loss = crit(pred, yb, epoch=ep)
                va_loss += loss.item() * Xb.size(0)
        va_loss /= len(val_ld.dataset)
        tr_losses.append(tr_loss)
        va_losses.append(va_loss)
        sch.step(va_loss)
        if ep == 1 or va_loss < best_vloss:
            if va_loss < best_vloss:
                best_vloss = va_loss
                wait = 0
            torch.save(model.state_dict(), save_path)
        else:
            wait += 1
            if wait >= cfg.PATIENCE:
                break
    train_time = time.time() - t_start
    # 兜底：若因 NaN 等原因从未保存过 best.pth，则用当前模型
    if not save_path.exists():
        torch.save(model.state_dict(), save_path)
    model.load_state_dict(torch.load(save_path, map_location=cfg.DEVICE))
    return model, tr_losses, va_losses, train_time


# =============================================================================
# EVALUATION & METRICS
# =============================================================================
def evaluate(model, test_ld, init_disp, scY, pca_y, hz, device, node_mask=None,
             true_diff_win=None):
    """测试集评估：预测走 PCA 逆变换，真值取原始一阶差分。

    2026-09-20 口径修正：此前真值也做 `pca_y.inverse_transform` 往返，等于把真值
    投影到 K=13 子空间后再比，丢弃的约 10% 增量方差及其随累加步数的放大都不计入
    误差 → 各表 RMSE/R² 系统性偏乐观，且 PCA 保留率越低越有利（表 9 的
    PCA−NoPCA 差距即被此口径差污染）。现真值直接取 `diff1_den` 对应切片的累加，
    预测侧照旧走 PCA 逆变换（模型本就在该子空间内输出）。
    """
    model.eval()
    preds_norm, trues_norm = [], []
    with torch.no_grad():
        for batch in test_ld:
            Xb, yb, wfb, gsb = [t.to(device) for t in batch]
            pred, _ = model(Xb, wfb, gsb, node_mask=node_mask)
            preds_norm.append(pred.cpu().numpy())
            trues_norm.append(yb.cpu().numpy())
    pred_n = np.concatenate(preds_norm, 0)
    out_dim = pred_n.shape[-1]
    ns = pred_n.shape[0]
    pred_pca = scY.inverse_transform(pred_n.reshape(-1, out_dim)).reshape(ns, hz, out_dim)
    N_orig = pca_y.n_features_in_
    pred_diff = pca_y.inverse_transform(pred_pca.reshape(-1, out_dim)).reshape(
        ns, hz, N_orig)
    if true_diff_win is not None:
        true_diff = np.asarray(true_diff_win, dtype=np.float64)[:ns]
    else:
        # 兼容路径：调用方未提供原始真值窗口时回退到旧口径（诊断脚本可能用到）
        true_n = np.concatenate(trues_norm, 0)
        true_pca = scY.inverse_transform(true_n.reshape(-1, out_dim)).reshape(
            ns, hz, out_dim)
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


def collect_gate_values(model, ld, device):
    """收集式(13) 的 g_t 在各测试窗上的取值，用于诊断其是否真的随水文变化。"""
    model.eval()
    gs = []
    with torch.no_grad():
        for batch in ld:
            Xb, yb, wfb, gsb = [t.to(device) for t in batch]
            model(Xb, wfb, gsb)
            g = getattr(model, '_last_g', None)
            if g is None:
                return None
            gs.append(g.reshape(-1).float().cpu().numpy())
    return np.concatenate(gs) if gs else None


def _point_metrics(t, p):
    return {'RMSE': float(np.sqrt(mean_squared_error(t, p))),
            'MAE': float(mean_absolute_error(t, p)),
            'R2': float(r2_score(t, p)),
            'MaxErr': float(max_error(t, p))}


def global_metrics(true_cum, pred_cum):
    """全步口径 + 末步口径，二者一并落盘。

    2026-09-20：原实现把 (ns, hz, N) 一并 ravel，同一真实时点被窗口重叠反复计入，
    且重复倍数随 hz 增大（H=2 约 1.9x、H=10 约 5.5x、H=12 约 5.4x）→ 长视界的
    RMSE/R² 天然比短视界好看，跨视界比较被污染。新增**末步口径**（后缀 `_last`）：
    每窗只取 t=hz-1，ns 个窗口对应 ns 个互不重复的时点，是预测文献的标准 h 步
    提前口径，论文主表用它。原列名保持不变，作为全步辅助口径。
    """
    gm = _point_metrics(true_cum.ravel(), pred_cum.ravel())
    gm.update({f'{k}_last': v for k, v in
               _point_metrics(true_cum[:, -1, :].ravel(),
                              pred_cum[:, -1, :].ravel()).items()})
    if true_cum.shape[1] > 1:
        true_d = np.diff(true_cum, axis=1)
        pred_d = np.diff(pred_cum, axis=1)
        gm['DA'] = float(np.mean(np.sign(true_d).ravel() == np.sign(pred_d).ravel()))
        # 末步口径的 DA 只取最后一步增量（与全步 DA 同一判定式，样本为 ns 个窗口）
        gm['DA_last'] = float(np.mean(
            np.sign(true_cum[:, -1, :] - true_cum[:, -2, :]).ravel() ==
            np.sign(pred_cum[:, -1, :] - pred_cum[:, -2, :]).ravel()))
    else:
        gm['DA'] = 0.5
        gm['DA_last'] = np.nan
    return gm


def output_space_floor(yte_raw, init_disp, scY, pca_y, hz, ns=None):
    """K 维 PCA 输出空间导致的**不可达误差地板**（与模型无关）。

    原理（见任务交接 §3.19.7）：真值 y 正交分解为 y = y_S + y_⊥，其中 y_S 为 y 在
    K 维主子空间上的投影、y_⊥ ⊥ 主子空间；任何本架构模型的预测 p 必落在主子空间内，
    故逐元素 ⟨y_S − p, y_⊥⟩ = 0，于是
        ||y − p||² = ||y_S − p||² + ||y_⊥||²
    `cumsum` 是线性算子且逐时刻正交，故该可加性在**末步累加口径**下依然成立
    （实测交叉项 ~1e-7，相对残差 ~1e-7）。

    因此可把主口径 RMSE 无损拆成两项，而**不改动主口径本身**：
        RMSE_full² = RMSE_reach² + RMSE_floor²
    其中 RMSE_floor 只取决于 K 与测试集，是**所有 K 维输出模型的公共下界**，
    与具体模型无关（同一 K 下逐种子相同）。

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


def count_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def estimate_flops(model, cfg, N, F_in, hz):
    # 2026-09-21：原先 `except Exception: return 0` 会把 thop 缺失静默吞掉，使全部
    # run 的 FLOPs 写成 0（论文复杂度表整列失效）。改为只兜底 `profile` 抛错，
    # 且失败时告警而非静默。
    try:
        from thop import profile
    except ImportError as e:
        logger.warning(f"thop 不可用（{e}），FLOPs 记为 0 —— 请先 pip install thop")
        return 0
    try:
        dummy_x = torch.randn(1, cfg.LOOKBACK, N, F_in).to(cfg.DEVICE)
        dummy_w = torch.randn(1, N, 8).to(cfg.DEVICE)
        dummy_g = torch.randn(1, N, cfg.GEO_FEAT_DIM + 1).to(cfg.DEVICE)
        flops, _ = profile(model, inputs=(dummy_x, dummy_w, dummy_g, None, None),
                           verbose=False)
        return flops
    except Exception as e:
        logger.warning(f"FLOPs profile 失败（{type(e).__name__}: {e}），记为 0")
        return 0


# =============================================================================
# RUN SINGLE SEED
# =============================================================================
def run_single_seed(cfg, seed, data, meta, adj_t, geo_static_t,
                    train_idx, val_idx, test_idx, ts, hz, vert_adj_t=None):
    run_dir = cfg.OUT_ROOT / f"run_{seed}"
    # 2026-09-18 审计修复：判据补 losses.npy（数据导出层），否则种子在三件套写出
    # 后、losses/attn 写出前崩溃时会被 RESUME 永久跳过，可视化层静默缺数据。
    # 2026-09-19 架构简化：时序注意力已改为均值池化，attn_weights.npy 不再生成，
    # RESUME 判据不再检查该文件。
    done_files = ('metrics.json', 'best.pth', 'predictions.npz', 'losses.npy')
    if cfg.RESUME and all((run_dir / f).exists() for f in done_files):
        logger.info(f"    Seed {seed}: 结果已完整，跳过（RESUME=1）")
        return None
    set_seed(seed)
    Xtr, ytr = data['Xtr'], data['ytr']
    Xva, yva = data['Xva'], data['yva']
    Xte, yte = data['Xte'], data['yte']
    init_test = data['init_test']
    out_dim = data['out_dim']
    N = data['N']
    F_in = data['F_in']
    wf_train, wf_val, wf_test = data['wf_train'], data['wf_val'], data['wf_test']
    valid_mask_t = torch.FloatTensor(data['valid_mask']).to(cfg.DEVICE)

    train_ds = TensorDataset(
        torch.FloatTensor(Xtr), torch.FloatTensor(ytr),
        torch.FloatTensor(wf_train),
        geo_static_t.unsqueeze(0).expand(Xtr.shape[0], N, -1))
    val_ds = TensorDataset(
        torch.FloatTensor(Xva), torch.FloatTensor(yva),
        torch.FloatTensor(wf_val),
        geo_static_t.unsqueeze(0).expand(Xva.shape[0], N, -1))
    test_ds = TensorDataset(
        torch.FloatTensor(Xte), torch.FloatTensor(yte),
        torch.FloatTensor(wf_test),
        geo_static_t.unsqueeze(0).expand(Xte.shape[0], N, -1))
    train_ld = DataLoader(train_ds, batch_size=cfg.BATCH_SIZE, shuffle=True)
    val_ld = DataLoader(val_ds, batch_size=cfg.BATCH_SIZE)
    test_ld = DataLoader(test_ds, batch_size=cfg.BATCH_SIZE)

    geo_dim_actual = data['geo_dim_actual']
    model = AblationModel(cfg, N=N, in_dim=F_in, gcn_hid=cfg.GCN_HIDDEN,
                          tcn_ch=cfg.TCN_CHANNELS, k=cfg.TCN_KERNEL_SIZE,
                          dil=cfg.TCN_DILATIONS, attn_hid=cfg.ATTENTION_HIDDEN,
                          hor=hz, adj=adj_t, out_dim=out_dim,
                          geo_feat_dim=geo_dim_actual, vert_adj=vert_adj_t).to(cfg.DEVICE)

    n_params = count_params(model)
    flops = estimate_flops(model, cfg, N, F_in, hz)

    save_path = cfg.OUT_ROOT / f"run_{seed}" / "best.pth"
    save_path.parent.mkdir(parents=True, exist_ok=True)
    model, tr_losses, va_losses, train_time = train_model(
        cfg, model, train_ld, val_ld, save_path, node_mask=valid_mask_t)

    pred_cum, true_cum = evaluate(model, test_ld, init_test, data['scY'],
                                   data['pca_y'], hz, cfg.DEVICE,
                                   node_mask=valid_mask_t,
                                   true_diff_win=data['yte_raw'])

    test_start = test_idx[0] + cfg.LOOKBACK
    test_ts_dt = ts[test_start:test_start + pred_cum.shape[0]]

    gm = global_metrics(true_cum, pred_cum)
    # 2026-09-21 输出空间误差分解（原理见 `output_space_floor` docstring 与交接 §3.19.7）：
    # 地板与模型无关，把主口径 RMSE 无损拆为「可达 + 不可达下界」，满足
    # RMSE_full² = RMSE_reach² + RMSE_floor²。**不改动任何既有键**，仅新增 4 个。
    _fl = output_space_floor(data['yte_raw'], init_test, data['scY'], data['pca_y'],
                             hz, ns=pred_cum.shape[0])
    gm['RMSE_floor_last'] = _fl['RMSE_floor_last']
    gm['RMSE_reach_last'] = float(np.sqrt(max(
        gm['RMSE_last'] ** 2 - _fl['MSE_floor_last'], 0.0)))
    gm['RMSE_floor'] = _fl['RMSE_floor_all']
    gm['RMSE_reach'] = float(np.sqrt(max(gm['RMSE'] ** 2 - _fl['MSE_floor_all'], 0.0)))
    gm['Params'] = n_params
    gm['FLOPs'] = flops
    gm['TrainTime'] = train_time

    # 水特征门控诊断已移除（见 __init__ 注释：门控恒为常数或引入方差）。
    # collect_gate_values 依赖从未赋值的 _last_g，恒返回 None，仅保留给历史脚本
    # _test_watergate.py 导入兼容，勿在生产路径调用。论文 §3.3/§6.3 的"门开度≈0.5"
    # 来自旧 A/B 对照实验（§6.3），重跑后需重新验证，不由本函数复现。

    df_node = node_metrics(true_cum, pred_cum, meta['node_names'])

    with open(save_path.parent / 'metrics.json', 'w') as f:
        json.dump(gm, f, indent=2)
    df_node.to_excel(save_path.parent / 'node_metrics.xlsx', index=False)
    np.savez(save_path.parent / 'predictions.npz',
             pred_cum=pred_cum, true_cum=true_cum,
             test_ts_dt=test_ts_dt.astype(str))
    # 数据导出层（2026-09-18）：训练/验证 loss。2026-09-19 架构简化：时序注意力
    # 已改为均值池化，不再收集/落盘 attn_weights.npy。
    np.save(save_path.parent / 'losses.npy', {'tr': tr_losses, 'va': va_losses})

    # 2026-09-20：原此处另算一份 layer_r2s（按深度分层的 R² 中位数）随返回值传出，
    # 但上层丢弃了 run_single_seed 的整个返回值，该变量从未被消费，已删除。
    # 分层指标由 aggregate_variant → layer_summary.csv 统一产出。
    return gm, pred_cum, true_cum, test_ts_dt, model


# =============================================================================
# AGGREGATE VARIANT RESULTS
# =============================================================================
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


def aggregate_variant(variant_dir):
    # 2026-09-20：排序。此前 iterdir() 的枚举顺序依赖 NTFS 目录项顺序，各变体目录
    # 一旦不一致，下游 ttest_rel 按行位置"配对"就会把种子配错（不报错、静默错位）。
    run_dirs = sorted((d for d in variant_dir.iterdir()
                       if d.is_dir() and d.name.startswith('run_')),
                      key=lambda d: int(d.name.split('_')[1]))
    global_list, node_list = [], []
    for rd in run_dirs:
        metrics_file = rd / 'metrics.json'
        if not metrics_file.exists():
            continue
        with open(metrics_file, 'r') as f:
            gm = json.load(f)
        seed = int(rd.name.split('_')[1])
        gm['seed'] = seed
        global_list.append(gm)
        node_file = rd / 'node_metrics.xlsx'
        if node_file.exists():
            df_node = pd.read_excel(node_file)
            df_node['seed'] = seed
            df_node['Depth'] = [float(n.split('_')[1]) for n in df_node['Node']]
            df_node['Point'] = [n.split('_')[0] for n in df_node['Node']]
            node_list.append(df_node)

    df_global = pd.DataFrame(global_list)
    summary_dir = variant_dir / "summary"
    summary_dir.mkdir(exist_ok=True)

    if node_list:
        all_nodes = pd.concat(node_list, ignore_index=True)
        bins = [0, 3, 8, 16]
        labels = ['Shallow', 'Middle', 'Deep']
        all_nodes['Layer'] = pd.cut(all_nodes['Depth'], bins=bins, labels=labels,
                                     right=False)

        # 2026-09-20：三级汇总同时给出全步（原列名不变）与末步（`_last` 后缀）
        # 两套口径，口径定义见 global_metrics。
        layer_summary = _layer_point_summary(
            all_nodes, ['Layer'], 'R2', 'RMSE', 'MAE').merge(
            _layer_point_summary(all_nodes, ['Layer'],
                                 'R2_last', 'RMSE_last', 'MAE_last'),
            on='Layer', suffixes=('', '_last'))

        point_summary = _layer_point_summary(
            all_nodes, ['Point'], 'R2', 'RMSE', 'MAE').merge(
            _layer_point_summary(all_nodes, ['Point'],
                                 'R2_last', 'RMSE_last', 'MAE_last'),
            on='Point', suffixes=('', '_last'))

        node_summary = _node_summary(all_nodes, 'R2', 'RMSE', 'MAE').merge(
            _node_summary(all_nodes, 'R2_last', 'RMSE_last', 'MAE_last'),
            on='Node', suffixes=('', '_last'))
        node_summary['Depth'] = [float(n.split('_')[1]) for n in node_summary['Node']]
        node_summary['Point'] = [n.split('_')[0] for n in node_summary['Node']]

        layer_summary.to_csv(summary_dir / 'layer_summary.csv', index=False)
        point_summary.to_csv(summary_dir / 'point_summary.csv', index=False)
        node_summary.to_excel(summary_dir / 'node_summary.xlsx', index=False)

    # 清掉可能残留的旧汇总（如试跑时留下的 global_3runs.csv），
    # 否则 compare_ablation_variants 的 glob(...)[0] 会读到旧文件、造成新老数据混编
    for old_f in summary_dir.glob('global_*runs.csv'):
        old_f.unlink()
    df_global.to_csv(summary_dir / f'global_{len(run_dirs)}runs.csv', index=False)
    return df_global


# =============================================================================
# PER-HORIZON COMPARISON & VISUALIZATION
# =============================================================================
def compare_ablation_variants(base_dir, hz, variant_keys, meta):
    comp_dir = base_dir / "comparison"
    comp_dir.mkdir(exist_ok=True)
    global_dfs, layer_dfs, point_dfs, node_dfs = [], [], [], []

    for v in variant_keys:
        vdir = base_dir / v
        summary_dir = vdir / "summary"
        glob_files = list(summary_dir.glob('global_*runs.csv'))
        if not glob_files:
            continue
        df_global = pd.read_csv(glob_files[0])
        df_global['Variant'] = v
        global_dfs.append(df_global)

        layer_file = summary_dir / 'layer_summary.csv'
        if layer_file.exists():
            df_layer = pd.read_csv(layer_file)
            df_layer['Variant'] = v
            layer_dfs.append(df_layer)

        point_file = summary_dir / 'point_summary.csv'
        if point_file.exists():
            df_point = pd.read_csv(point_file)
            df_point['Variant'] = v
            point_dfs.append(df_point)

        node_file = summary_dir / 'node_summary.xlsx'
        if node_file.exists():
            df_node = pd.read_excel(node_file)
            df_node['Variant'] = v
            node_dfs.append(df_node)

    if not global_dfs:
        logger.warning(f"No results found for H={hz}")
        return None, None, None, None

    df_global_all = pd.concat(global_dfs, ignore_index=True)
    # 数据导出层（2026-09-18）：逐种子明细宽表（含全部 30 种子，非仅均值）
    df_global_all.to_pickle(comp_dir / 'full_seed_data.pkl')
    df_layer_all = pd.concat(layer_dfs, ignore_index=True) if layer_dfs else None
    df_point_all = pd.concat(point_dfs, ignore_index=True) if point_dfs else None
    df_node_all = pd.concat(node_dfs, ignore_index=True) if node_dfs else None

    # 注：Fig_a..Fig_i 的绘图代码已迁至 可视化\02代码结果可视化\fig_ablation.py，
    # 本函数只负责数据聚合与落盘（不再绘图）。

    # ---- Save comprehensive summary tables ----
    # 2026-09-20：全步口径沿用原列名（RMSE_mean 等），末步口径加 `_last` 后缀，
    # 两套一并落盘；口径定义见 global_metrics。
    agg_spec = {'Params_mean': ('Params', 'mean'),
                'FLOPs_mean': ('FLOPs', 'mean'),
                'TrainTime_mean': ('TrainTime', 'mean')}
    for suf in ('', '_last'):
        for m in ('RMSE', 'MAE', 'R2', 'MaxErr', 'DA'):
            agg_spec[f'{m}{suf}_mean'] = (f'{m}{suf}', 'mean')
            agg_spec[f'{m}{suf}_std'] = (f'{m}{suf}', 'std')
    # 2026-09-21：补误差分解列（RMSE² = RMSE_reach² + RMSE_floor²，见 output_space_floor）。
    # 此前它们只存在于逐 run 的 metrics.json / global_*runs.csv，本聚合的显式白名单漏了，
    # 导致组合比较表无法直接读分解。列缺失时跳过，避免 groupby.agg KeyError。
    for m in ('RMSE_reach', 'RMSE_floor'):
        for suf in ('', '_last'):
            col = f'{m}{suf}'
            if col in df_global_all.columns:
                agg_spec[f'{col}_mean'] = (col, 'mean')
                agg_spec[f'{col}_std'] = (col, 'std')
    global_summary = df_global_all.groupby('Variant').agg(**agg_spec).reset_index()
    global_summary['Variant_Short'] = global_summary['Variant'].map(VARIANT_SHORT_NAMES)
    global_summary.to_csv(comp_dir / 'global_comparison.csv', index=False)
    global_summary.to_excel(comp_dir / 'global_comparison.xlsx', index=False)

    # ---- Statistical tests table (V5 vs each) ----
    # 2026-09-20 修复：配对 t 检验必须按 seed 对齐。此前直接取 `['RMSE'].values`
    # 按行位置配对，而行序继承自 variant 目录的枚举顺序；一旦各目录顺序不一致，
    # "配对"就把种子配错，且不报错（静默错位）。现按 seed 列取交集后对齐，
    # 并把参与配对的种子数写入 N_Seeds。主口径为末步（`_last`，论文表 7 用），
    # 全步口径一并给出以保留原列名语义。
    stat_records = []
    baseline = 'V5_Full_Model'
    if baseline in variant_keys:
        base_df = df_global_all[df_global_all['Variant'] == baseline]
        for v in variant_keys:
            if v == baseline:
                continue
            sub = df_global_all[df_global_all['Variant'] == v]
            common = sorted(set(base_df['seed']) & set(sub['seed']))
            if len(common) < 5:
                continue
            b = base_df.set_index('seed').loc[common]
            c = sub.set_index('seed').loc[common]
            rec = {'Comparison': f'{VARIANT_SHORT_NAMES[v]} vs '
                                 f'{VARIANT_SHORT_NAMES[baseline]}',
                   'N_Seeds': len(common)}
            for r2k, rmk, tag in (('R2_last', 'RMSE_last', '_last'),
                                  ('R2', 'RMSE', '')):
                br, cr = b[rmk].values, c[rmk].values
                b2, c2 = b[r2k].values, c[r2k].values
                t_rmse, p_rmse = ttest_rel(br, cr)
                t_r2, p_r2 = ttest_rel(b2, c2)
                rec[f'RMSE{tag}_t'] = round(t_rmse, 3)
                rec[f'RMSE{tag}_p'] = round(p_rmse, 4)
                rec[f'RMSE{tag}_Cohens_d'] = round(cohens_d(br, cr), 3)
                rec[f'R2{tag}_t'] = round(t_r2, 3)
                rec[f'R2{tag}_p'] = round(p_r2, 4)
                rec[f'R2{tag}_Cohens_d'] = round(cohens_d(b2, c2), 3)
                if tag:
                    rec['Delta_RMSE_pct'] = round(
                        (np.mean(cr) - np.mean(br)) / np.mean(br) * 100, 2)
                    rec['Sig'] = ('***' if p_rmse < 0.001 else
                                  ('**' if p_rmse < 0.01 else
                                   ('*' if p_rmse < 0.05 else 'ns')))
            stat_records.append(rec)
    pd.DataFrame(stat_records).to_csv(comp_dir / 'statistical_tests.csv', index=False)
    pd.DataFrame(stat_records).to_excel(comp_dir / 'statistical_tests.xlsx', index=False)

    logger.info(f"  H={hz}: {len(global_dfs)} variants, 3 tables saved "
                f"(绘图见 可视化\\02代码结果可视化\\fig_ablation.py)")
    return df_global_all, df_node_all, df_layer_all, df_point_all


# =============================================================================
# CROSS-HORIZON VISUALIZATION
# =============================================================================
def cross_horizon_visualizations(all_horizon_data, variant_keys, out_root):
    cross_dir = out_root / "cross_horizon_comparison"
    cross_dir.mkdir(exist_ok=True)
    # 注：CH_Fig1..CH_Fig6 的绘图代码已迁至 可视化\02代码结果可视化\fig_ablation.py，
    # 本函数只负责数据聚合与落盘（不再绘图）。

    # Collect all data
    all_records = []
    for hz, df_global in all_horizon_data.items():
        df_global['Horizon'] = hz
        all_records.append(df_global)
    df_all_hz = pd.concat(all_records, ignore_index=True)
    horizons = sorted(df_all_hz['Horizon'].unique())
    df_all_hz.to_csv(cross_dir / 'cross_horizon_raw.csv', index=False)

    # Compute summary per variant per horizon
    summary_rows = []
    for hz in horizons:
        sub_hz = df_all_hz[df_all_hz['Horizon'] == hz]
        for v in variant_keys:
            sub_v = sub_hz[sub_hz['Variant'] == v]
            if len(sub_v) == 0:
                continue
            # 2026-09-21 修复：跨视界汇总此前只落**全步**口径，而全步口径的窗口重叠
            # 倍数随 hz 漂移（H=2 约1.9x、H=10 约5.5x），长视界天然好看、跨视界比较
            # 被污染（见 global_metrics docstring）——与论文主口径（末步）冲突。
            # 现补 `_last` 列，与 Comparison_Experiment 的 cross_horizon_summary 对齐。
            row = {'Horizon': hz, 'Variant': v}
            for m in ('RMSE', 'R2', 'MAE', 'MaxErr'):
                row[f'{m}_mean'] = sub_v[m].mean()
                row[f'{m}_std'] = sub_v[m].std()
                if f'{m}_last' in sub_v.columns:
                    row[f'{m}_last_mean'] = sub_v[f'{m}_last'].mean()
                    row[f'{m}_last_std'] = sub_v[f'{m}_last'].std()
            summary_rows.append(row)
    df_summary = pd.DataFrame(summary_rows)
    df_summary.to_csv(cross_dir / 'cross_horizon_summary.csv', index=False)
    df_summary.to_excel(cross_dir / 'cross_horizon_summary.xlsx', index=False)

    # ---- CH-Fig6 的 decay 数据落盘（CH_Fig1..CH_Fig6 绘图已迁至
    #      可视化\02代码结果可视化\fig_ablation.py，本函数不再绘图）----
    decay_rows = []
    # 2026-09-21 修复：衰减率改用**末步**口径（`R2_last_mean`，论文主口径）；
    # 若旧 CSV 无该列则回退全步 `R2_mean`。
    r2_key = 'R2_last_mean' if 'R2_last_mean' in df_summary.columns else 'R2_mean'
    for v in variant_keys:
        sub = df_summary[df_summary['Variant'] == v]
        sub_sorted = sub.sort_values('Horizon')
        if len(sub_sorted) >= 2:
            first = sub_sorted.iloc[0]
            last = sub_sorted.iloc[-1]
            decay = (first[r2_key] - last[r2_key]) / (first[r2_key] + 1e-6) * 100
            decay_rows.append({'Variant': v, 'DecayRate_pct': decay,
                               'H_first_R2': first[r2_key],
                               'H_last_R2': last[r2_key]})
    if decay_rows:
        df_decay = pd.DataFrame(decay_rows)
        df_decay = df_decay.sort_values('DecayRate_pct')
        df_decay.to_csv(cross_dir / 'horizon_decay_rates.csv', index=False)

    logger.info("  Cross-horizon: 2 tables saved "
                "(绘图见 可视化\\02代码结果可视化\\fig_ablation.py)")
    return df_summary


# =============================================================================
# RUN ONE ABLATION VARIANT
# =============================================================================
def run_ablation_variant(variant_name, switch_dict, base_cfg, hz,
                         shared_data_cache=None):
    t_start = time.time()
    cfg = copy.deepcopy(base_cfg)
    for k, v in switch_dict.items():
        setattr(cfg, k, v)
    cfg.OUT_ROOT = base_cfg.OUT_ROOT / variant_name
    cfg.OUT_ROOT.mkdir(parents=True, exist_ok=True)

    logger.info(f"  [{variant_name}] H={hz} - Loading data...")

    if shared_data_cache is not None and 'full_df' in shared_data_cache:
        full_df = shared_data_cache['full_df']
        ts = shared_data_cache['ts']
        wd = shared_data_cache['wd']
        wa = shared_data_cache['wa']
        water_feat = shared_data_cache['water_feat']
        meta = shared_data_cache['meta']
        geo_feat_raw = shared_data_cache['geo_feat_raw']
        disp_raw = shared_data_cache.get('disp_raw')  # 诊断脚本可能构造精简缓存
        train_idx = shared_data_cache['train_idx']
        val_idx = shared_data_cache['val_idx']
        test_idx = shared_data_cache['test_idx']
    else:
        set_seed(42)
        full_df = load_raw_data_with_meta(cfg)
        ts = np.sort(full_df['测量时间'].unique())
        wd, wa = load_water_data(ts, cfg)
        water_feat = extract_water_features(wd, wa)
        disp_raw, _, meta = build_node_features(full_df, cfg)
        geo_records = load_geology_params(cfg)
        geo_feat_raw = assign_geo_to_nodes(geo_records, meta['depths'], cfg) \
            if geo_records else None
        n = len(ts)
        tr_end = int(n * cfg.TRAIN_RATIO)
        va_end = int(n * (cfg.TRAIN_RATIO + cfg.VAL_RATIO))
        train_idx = np.arange(tr_end)
        val_idx = np.arange(tr_end, va_end)
        test_idx = np.arange(va_end, n)

    geo_fb = geo_feat_raw if geo_feat_raw is not None else np.zeros(
        (len(meta['depths']), cfg.GEO_FEAT_DIM), dtype=np.float32)

    data = prepare_data(cfg, full_df, wd, wa, meta, train_idx, val_idx, test_idx,
                        water_feat, geo_fb, hz, disp_raw=disp_raw)
    geo_fb_use = geo_feat_raw if geo_feat_raw is not None and np.any(geo_feat_raw) else None
    # V8（ZERO_SAME_HOLE=True）会把 A 中同孔节点对整体置 0，其余变体保持默认图。
    adj = build_adjacency(meta, geo_fb_use, cfg,
                          zero_same_hole=getattr(cfg, 'ZERO_SAME_HOLE', False))
    adj_t = torch.FloatTensor(adj).to(cfg.DEVICE)
    vert_adj = build_vertical_adjacency(meta, cfg)
    vert_adj_t = torch.FloatTensor(vert_adj).to(cfg.DEVICE) if vert_adj is not None else None
    geo_static_t = torch.FloatTensor(data['geo_static']).to(cfg.DEVICE)

    N_SEEDS = len(cfg.SEEDS)
    for si, seed in enumerate(cfg.SEEDS):
        logger.info(f"    Seed {si + 1}/{N_SEEDS} ({seed})")
        run_single_seed(cfg, seed, data, meta, adj_t, geo_static_t,
                        train_idx, val_idx, test_idx, ts, hz,
                        vert_adj_t=vert_adj_t)

    df_global = aggregate_variant(cfg.OUT_ROOT)
    elapsed = time.time() - t_start
    logger.info(f"  [{variant_name}] H={hz} done in {elapsed:.1f}s "
                f"({len(cfg.SEEDS)} seeds, RMSE(last)="
                f"{df_global['RMSE_last'].mean():.4f}, "
                f"RMSE(full)={df_global['RMSE'].mean():.4f})")
    return df_global


# =============================================================================
# MAIN DRIVER
# =============================================================================
def main(quick_test=False):
    base_cfg = Config()
    out_root = base_cfg.OUT_ROOT

    if quick_test:
        base_cfg.HORIZONS = [6]
        base_cfg.SEEDS = base_cfg.SEEDS[:2]
        base_cfg.EPOCHS = 5
        base_cfg.PATIENCE = 3
        logger.info("=" * 60)
        logger.info("QUICK TEST MODE: H=6 only, 2 seeds, 5 epochs")
        logger.info("=" * 60)
    else:
        logger.info("=" * 60)
        logger.info("FULL ABLATION EXPERIMENT")
        logger.info(f"Variants: {len(ABLATION_VARIANTS)}")
        logger.info(f"Horizons: {base_cfg.HORIZONS}")
        logger.info(f"Seeds: {len(base_cfg.SEEDS)}")
        logger.info(f"Total runs: {len(ABLATION_VARIANTS) * len(base_cfg.HORIZONS) * len(base_cfg.SEEDS)}")
        logger.info("=" * 60)

    logger.info(f"Device: {base_cfg.DEVICE}")
    logger.info(f"Output: {out_root}")

    variant_keys = list(ABLATION_VARIANTS.keys())

    # Pre-load shared data for all variants in this horizon
    set_seed(42)
    full_df = load_raw_data_with_meta(base_cfg)
    # 2026-09-20：`unique()` 的返回顺序继承自按"测点"排序的原表，不是时间序；
    # 而特征矩阵走 `pivot.sort_index()` 是时间序。两者不一致会让导出的时间戳
    # （`test_ts_dt`）错位。统一改为排序后的时间轴。
    ts = np.sort(full_df['测量时间'].unique())
    wd, wa = load_water_data(ts, base_cfg)
    water_feat = extract_water_features(wd, wa)
    disp_raw, _, meta = build_node_features(full_df, base_cfg)
    geo_records = load_geology_params(base_cfg)
    geo_feat_raw = assign_geo_to_nodes(geo_records, meta['depths'], base_cfg) \
        if geo_records else None
    n = len(ts)
    tr_end = int(n * base_cfg.TRAIN_RATIO)
    va_end = int(n * (base_cfg.TRAIN_RATIO + base_cfg.VAL_RATIO))
    train_idx = np.arange(tr_end)
    val_idx = np.arange(tr_end, va_end)
    test_idx = np.arange(va_end, n)

    shared_cache = {
        'full_df': full_df, 'ts': ts, 'wd': wd, 'wa': wa,
        'water_feat': water_feat, 'meta': meta, 'geo_feat_raw': geo_feat_raw,
        'disp_raw': disp_raw,
        'train_idx': train_idx, 'val_idx': val_idx, 'test_idx': test_idx
    }

    all_horizon_data = {}
    for hz in base_cfg.HORIZONS:
        logger.info(f"\n{'#' * 60}\n#  HORIZON H={hz}\n{'#' * 60}")
        base_cfg.OUT_ROOT = out_root / f"H{hz}"
        base_cfg.OUT_ROOT.mkdir(parents=True, exist_ok=True)

        for v_name, switch_dict in ABLATION_VARIANTS.items():
            logger.info(f"\n--- {v_name} (H={hz}) ---")
            df_g = run_ablation_variant(v_name, switch_dict, base_cfg, hz,
                                        shared_data_cache=shared_cache)

        logger.info(f"\n--- Generating H={hz} comparison visualizations ---")
        df_global_all, df_node_all, df_layer_all, df_point_all = \
            compare_ablation_variants(base_cfg.OUT_ROOT, hz, variant_keys, meta)

        if df_global_all is not None:
            all_horizon_data[hz] = df_global_all
            logger.info(f"  H={hz} comparison complete "
                        f"({df_global_all['Variant'].nunique()} variants)")

    # Cross-horizon visualizations
    if len(all_horizon_data) >= 2:
        logger.info(f"\n{'#' * 60}\n#  CROSS-HORIZON ANALYSIS\n{'#' * 60}")
        base_cfg.OUT_ROOT = out_root
        df_summary = cross_horizon_visualizations(all_horizon_data, variant_keys,
                                                  out_root)
    else:
        logger.info("Single horizon only -- skipping cross-horizon analysis.")

    logger.info(f"\n{'=' * 60}")
    logger.info(f"ALL DONE! Results saved to: {out_root}")
    logger.info(f"Summaries: {out_root}/cross_horizon_comparison/")
    for hz in base_cfg.HORIZONS:
        logger.info(f"  H{hz} comparison: {out_root}/H{hz}/comparison/")
    logger.info(f"{'=' * 60}")


if __name__ == '__main__':
    quick = '--quick' in sys.argv or '-q' in sys.argv
    main(quick_test=quick)