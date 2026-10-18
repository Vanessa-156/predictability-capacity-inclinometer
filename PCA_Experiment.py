# =============================================================================
# PCA_Experiment.py -- Chapter 4 PCA Ablation for MST Journal
# =============================================================================
# Design doc F:
#   1 model (full architecture) x 5 Horizons (H=2,6,8,10,12)
#   x 4 PCA retention rates (80%, 90%, 95%, None/NoPCA=155d)
#   x 30 seeds = 600 runs
#
# Architecture note (unified with Ablation_Experiment.py):
#   - 路径 A（2026-09-17）：水文水位不再进入模型。VICD 诊断（正文 §3.4）表明
#     水位与位移在增量口径无可用信息，输入特征由 12 维降为 4 维，PCA 条件结果
#     仍与消融全模块架构直接可比。water_feat 仅按占位透传以保持 DataLoader 签名。
#   - Input features exclude static geology (USE_GEO_INJECT = False), as in
#     the ablation.
#   - 2026-09-22：对齐论文主模型。§5.2 消融后正文 Ours = V7（去竖向 GCN 通路），
#     故 PCA 容量实验改在 V7 上做（USE_VERTICAL_GCN = False）。结果写入
#     PCA_Results_V7，不覆盖此前 V5 全模块的 PCA_Results。
#
# Hypothesis:
#   "Horizon越长，PCA压缩价值越大"
#   H=2: NoPCA ~ PCA90% (narrow gap)
#   H=12: NoPCA >> PCA90% (wide gap, full-dim overfits under small sample)
#
# Visualizations（2026-09-18 迁出为本目录外独立脚本）:
#   绘图代码移至 可视化\02代码结果可视化\fig_pca.py，从落盘数据出图，无需重跑实验：
#   Fig4-7: RMSE + Params vs PCA retention rate (H=10, see KEY_HORIZON)
#   Fig4-8: RMSE gap (NoPCA - PCA90%) vs Horizon
# 本脚本只负责模型训练、指标计算与数据落盘（含 comparison\pca_summary.csv/xlsx）。
# =============================================================================
# 写盘重试层（须早于 numpy/pandas/torch 导入，见 _robust_io.py）

from _env import AUX_ROOT, DATA_ROOT
import _robust_io  # noqa: F401
import os, json, logging, warnings, time, copy, gc, sys
from pathlib import Path
import numpy as np
import pandas as pd
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
    # 2026-09-22：对齐论文主模型——§5.2 消融后正文 Ours = V7（去竖向 GCN）。
    # False 时 PCAModel 不创建 vertical_gcn，模块创建顺序与
    # Ablation_Experiment.AblationModel(use_vertical_gcn=False) 一致。
    USE_VERTICAL_GCN = False
    VARIANT_TAG = 'V7' if not USE_VERTICAL_GCN else 'V5'

    # 结果目录按变体隔离，保留此前 V5 全模块的 PCA_Results 不动。
    OUT_ROOT = AUX_ROOT / "PCA_Results_V7"
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
    WAVELET = 'db4'
    WAVELET_LEVEL = 1
    THRESH_MODE = 'soft'
    USE_GEO_INJECT = False

    SPA_WITHIN_HOLE_PRIOR = True   # 同孔相邻深度边权重 0.8（结构连续性）
    SPA_SHALLOW_PRIOR = True       # 浅层(<3 m)跨断面节点对 0.6，并豁免 3σ 剪枝

    # 30 个种子：与 Ablation/Comparison/Stress 完全一致，见 _gen_seeds.py
    SEEDS = [42, 15, 77, 85, 120, 532, 901, 23, 12, 14, 19, 75, 250, 140, 135,
             1180, 1262, 1284, 605, 1159, 746, 1657, 61, 27, 993, 897, 276,
             1308, 896, 1866]
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # 2026-09-20：`AMP`（自动混合精度）与 `FULL_VIS` 均无消费者，已删除。
    # 2026-09-22：断点续跑，与 Comparison_Experiment 的 COMPARISON_RESUME 同款——
    # 置 1 时跳过已完整产出四件套（metrics.json/best.pth/predictions.npz/losses.npy）
    # 的种子，并读回其 metrics.json 参与汇总，避免中断后重训已完成种子。
    # 全量规模为 4 条件 × 5 视界 × 30 种子 = 600 runs，中断代价较高。
    RESUME = os.environ.get('PCA_RESUME', '0') == '1'


# =============================================================================
# PCA CONDITIONS
# =============================================================================
PCA_CONDITIONS = {
    'pca80': {'pca_variance': 0.80, 'label': 'PCA80%'},
    'pca90': {'pca_variance': 0.90, 'label': 'PCA90%'},
    'pca95': {'pca_variance': 0.95, 'label': 'PCA95%'},
    'nopca': {'pca_variance': None, 'label': 'No PCA (155d)'},
}

# 论文主展示步长 H=10（此前误用 H=6，2026-09-17 修正）。
# 回退：若 HORIZONS 中无 H=10（例如 quick_test），取 HORIZONS[0]。
KEY_HORIZON = 10

CONDITION_COLORS = {
    'pca80': '#D55E00',
    'pca90': '#0072B2',
    'pca95': '#009E73',
    'nopca': '#882255',
}

CONDITION_MARKERS = {'pca80': 's', 'pca90': 'o', 'pca95': '^', 'nopca': 'D'}


# =============================================================================
# UTILITIES
# =============================================================================
def set_seed(seed=42):
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def wavelet_denoise(signal, wavelet='db4', level=1, thresh_mode='soft'):
    level = min(level, int(np.log2(len(signal))) - 1)
    if level < 1: return signal
    coeffs = pywt.wavedec(signal, wavelet, level=level)
    sigma = np.median(np.abs(coeffs[-1])) / 0.6745
    thr = sigma * np.sqrt(2 * np.log(len(signal)))
    coeffs[1:] = [pywt.threshold(c, thr, thresh_mode) for c in coeffs[1:]]
    rec = pywt.waverec(coeffs, wavelet)
    if len(rec) > len(signal): return rec[:len(signal)]
    return np.pad(rec, (0, len(signal) - len(rec)), 'edge')


# =============================================================================
# DATA LOADING
# =============================================================================
def load_raw_data_with_meta(cfg):
    all_dfs = []
    for point in cfg.POINTS:
        fp = cfg.BASE_DATA_DIR / f"{point}.xlsx"
        if not fp.exists(): continue
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
        if not p.exists(): continue
        df = pd.read_excel(p)
        df['监测时间'] = pd.to_datetime(df['监测时间'])
        df = df.sort_values('监测时间')
        t = df['监测时间'].astype(np.int64)
        interp_d = interp1d(t, df['水位深度_m'], kind='linear', fill_value='extrapolate')
        interp_a = interp1d(t, df['累计水位变化_m'], kind='linear', fill_value='extrapolate')
        all_d.append(interp_d(ts_int))
        all_a.append(interp_a(ts_int))
    if all_d: return np.mean(all_d, axis=0).astype(np.float32), np.mean(all_a, axis=0).astype(np.float32)
    return np.zeros(len(timestamps), dtype=np.float32), np.zeros(len(timestamps), dtype=np.float32)


def extract_water_features(wd, wa):
    T = len(wd)
    wf = np.zeros((T, 8), dtype=np.float32)
    wf[:, 0] = wd; wf[:, 1] = wa
    if T > 1: wf[1:, 2] = np.diff(wd)
    if T > 3: wf[3:, 3] = wd[3:] - wd[:-3]
    if T > 7: wf[7:, 4] = wd[7:] - wd[:-7]
    if T > 15: wf[15:, 5] = wd[15:] - wd[:-15]
    if T > 30: wf[30:, 6] = wd[30:] - wd[:-30]
    if T > 1: wf[1:, 7] = np.sign(np.diff(wd))
    return wf


def load_geology_params(cfg):
    if not cfg.GEO_PATH.exists(): return None
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
    if geo_records is None: return geo_feat
    cumulative = 0.0
    for rec in geo_records:
        top = cumulative; bottom = cumulative + rec['thickness']; cumulative = bottom
        mask = (depths >= top) & (depths < bottom)
        geo_feat[mask, 0] = rec['cohesion']; geo_feat[mask, 1] = rec['friction']
        geo_feat[mask, 2] = rec['modulus']; geo_feat[mask, 3] = rec['density']
        geo_feat[mask, 4] = rec['poisson']; geo_feat[mask, 5] = rec['k0']
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
            pivot[c] = 0.0; valid_mask[i] = 0.0
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
    if len(ts) > 1: disp_diff1[1:] = np.diff(disp_raw, axis=0)
    point_names = [p for p in cfg.POINTS for _ in cfg.DEPTHS]
    depths_arr = np.array([d for _ in cfg.POINTS for d in cfg.DEPTHS], dtype=np.float32)
    meta = {'node_names': nodes, 'point_names': point_names, 'depths': depths_arr,
            'timestamps': ts, 'valid_mask': valid_mask}
    coord_path = cfg.BASE_DATA_DIR / "斜测点坐标.xlsx"
    N = len(nodes); coords = np.zeros((N, 2), dtype=np.float32)
    if coord_path.exists():
        coord_df = pd.read_excel(coord_path)
        for i, (p, d) in enumerate(zip(point_names, depths_arr)):
            row = coord_df[coord_df['测点'] == p]
            if not row.empty: coords[i, 0] = row['位置_X_m'].values[0]; coords[i, 1] = row['位置_Y_m'].values[0]
            else: coords[i, 0] = cfg.POINTS.index(p) * 10.0
    else:
        for i, p in enumerate(point_names): coords[i, 0] = cfg.POINTS.index(p) * 10.0
    meta['coords'] = coords
    return disp_raw, disp_diff1, meta


def build_geo_similarity(geo_feat):
    """式(10)：地质相似性矩阵 A^geo（土体参数 Z-score 后欧氏距离的高斯核）。

    用于式(11) 的邻接矩阵合成。
    """
    if geo_feat is None or not np.any(geo_feat): return None
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
                adj_spatial[k, k + 1] = 0.8; adj_spatial[k + 1, k] = 0.8
    shallow = meta['depths'] < 3.0
    idxs = np.where(shallow)[0]
    if getattr(cfg, 'SPA_SHALLOW_PRIOR', True):
        for i in idxs:
            for j in idxs:
                if i != j: adj_spatial[i, j] = max(adj_spatial[i, j], 0.6)

    geo_sim = build_geo_similarity(geo_feat)
    adj = 0.7 * adj_spatial + 0.3 * geo_sim if geo_sim is not None else adj_spatial
    np.fill_diagonal(adj, 1.0)

    # 3σ 距离剪枝；浅层跨断面边被显式豁免——浅层相邻断面间距本身接近 σ，
    # 若一并剪除会抹掉悬臂段的耦合通道。
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

    d = np.sum(adj, axis=1); d_inv_sqrt = np.power(np.maximum(d, 1e-6), -0.5)
    return (d_inv_sqrt[:, None] * adj * d_inv_sqrt[None, :]).astype(np.float32)


def build_vertical_adjacency(meta, cfg):
    N = len(meta['node_names']); n_dep = len(cfg.DEPTHS)
    vert_adj = np.eye(N, dtype=np.float32)
    for p in range(len(cfg.POINTS)):
        start = p * n_dep
        for k in range(start, start + n_dep - 1):
            vert_adj[k, k + 1] = 1.0; vert_adj[k + 1, k] = 1.0
    d_v = np.sum(vert_adj, axis=1); d_v_inv_sqrt = np.power(np.maximum(d_v, 1e-6), -0.5)
    return (d_v_inv_sqrt[:, None] * vert_adj * d_v_inv_sqrt[None, :]).astype(np.float32)


def denoise_split(train, val, test, cfg):
    N = train.shape[1]
    tr_d = np.zeros_like(train); v_d = np.zeros_like(val); te_d = np.zeros_like(test)
    for i in range(N):
        tr_d[:, i] = wavelet_denoise(train[:, i], cfg.WAVELET, cfg.WAVELET_LEVEL, cfg.THRESH_MODE)
        v_d[:, i] = wavelet_denoise(val[:, i], cfg.WAVELET, cfg.WAVELET_LEVEL, cfg.THRESH_MODE)
        te_d[:, i] = wavelet_denoise(test[:, i], cfg.WAVELET, cfg.WAVELET_LEVEL, cfg.THRESH_MODE)
    return tr_d, v_d, te_d


def make_features(disp_raw, diff1, meta, wf, geo_static, use_water=True, use_geo=True):
    T, N = disp_raw.shape
    depth_norm = np.broadcast_to(meta['depths'] / 16.0, (T, N)).astype(np.float32)
    shallow = np.tile((meta['depths'] < 3.0).astype(np.float32), (T, 1))
    feat_list = [
        disp_raw[..., np.newaxis], diff1[..., np.newaxis],
        depth_norm[..., np.newaxis], shallow[..., np.newaxis]
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
# DATA PREPARATION (PCA rate is the variable)
# =============================================================================
def prepare_data_with_pca(cfg, full_df, wd, wa, meta, train_idx, val_idx, test_idx,
                          water_feat, geo_feat, hz, pca_variance):
    """Prepare data with a specific PCA retention rate. pca_variance=None means No PCA."""
    disp_raw, diff1_all, _ = build_node_features(full_df, cfg)
    train_disp = disp_raw[train_idx]; val_disp = disp_raw[val_idx]; test_disp = disp_raw[test_idx]
    train_den, val_den, test_den = denoise_split(train_disp, val_disp, test_disp, cfg)
    all_den = np.vstack([train_den, val_den, test_den])
    diff1_den = np.zeros_like(all_den)
    if len(meta['timestamps']) > 1: diff1_den[1:] = np.diff(all_den, axis=0)

    geo_static_raw = np.concatenate([
        geo_feat if geo_feat is not None else np.zeros((len(meta['depths']), cfg.GEO_FEAT_DIM), dtype=np.float32),
        (meta['depths'] / 16.0).reshape(-1, 1)], axis=1)
    geo_static = (geo_static_raw - geo_static_raw.mean(0)) / (geo_static_raw.std(0) + 1e-6)

    train_feat = make_features(train_den, diff1_den[train_idx], meta, water_feat[train_idx], geo_static, use_geo=cfg.USE_GEO_INJECT)
    val_feat = make_features(val_den, diff1_den[val_idx], meta, water_feat[val_idx], geo_static, use_geo=cfg.USE_GEO_INJECT)
    test_feat = make_features(test_den, diff1_den[test_idx], meta, water_feat[test_idx], geo_static, use_geo=cfg.USE_GEO_INJECT)
    F = train_feat.shape[-1]
    y_train = diff1_den[train_idx]; y_val = diff1_den[val_idx]; y_test = diff1_den[test_idx]
    N_orig = y_train.shape[1]

    scX = StandardScaler().fit(train_feat.reshape(-1, F))
    tr_norm = scX.transform(train_feat.reshape(-1, F)).reshape(train_feat.shape)
    va_norm = scX.transform(val_feat.reshape(-1, F)).reshape(val_feat.shape)
    te_norm = scX.transform(test_feat.reshape(-1, F)).reshape(test_feat.shape)

    if pca_variance is not None:
        pca_y = PCA(pca_variance).fit(y_train)
        out_dim = pca_y.n_components_
        ytr_pca = pca_y.transform(y_train); yva_pca = pca_y.transform(y_val); yte_pca = pca_y.transform(y_test)
    else:
        pca_y = None; out_dim = N_orig
        ytr_pca = y_train.copy(); yva_pca = y_val.copy(); yte_pca = y_test.copy()

    scY = StandardScaler().fit(ytr_pca)
    ytr_norm = scY.transform(ytr_pca); yva_norm = scY.transform(yva_pca); yte_norm = scY.transform(yte_pca)

    def make_samples(feat, tgt, lb, hval):
        X, y = [], []
        for i in range(len(feat) - lb - hval + 1):
            X.append(feat[i:i + lb]); y.append(tgt[i + lb:i + lb + hval])
        return np.array(X, dtype=np.float32), np.array(y, dtype=np.float32)

    Xtr, ytr = make_samples(tr_norm, ytr_norm, cfg.LOOKBACK, hz)
    Xva, yva = make_samples(va_norm, yva_norm, cfg.LOOKBACK, hz)
    Xte, yte = make_samples(te_norm, yte_norm, cfg.LOOKBACK, hz)
    # 评估口径（2026-09-20）：真值窗口直接切原始一阶差分 y_test，不经 PCA 投影、
    # 不经标准化。此前真值也做 pca_y 往返，等于把真值投到 K 维子空间后再比，
    # 丢掉的增量方差不计入误差 → 保留率越低越"好看"，表 9 的 PCA−NoPCA 差距
    # 即被此口径差污染（实测累计位移偏差均值 0.72 mm、相对 15.6%）。
    yte_raw = np.stack([y_test[i + cfg.LOOKBACK:i + cfg.LOOKBACK + hz]
                        for i in range(Xte.shape[0])]) if Xte.shape[0] else np.empty(
        (0, hz, y_test.shape[1]), dtype=np.float32)
    init_test = test_den[cfg.LOOKBACK - 1:-hz][:Xte.shape[0]]

    return {
        'Xtr': Xtr, 'ytr': ytr, 'Xva': Xva, 'yva': yva, 'Xte': Xte, 'yte': yte,
        'yte_raw': yte_raw,
        'init_test': init_test, 'scX': scX, 'scY': scY, 'pca_y': pca_y,
        'out_dim': out_dim, 'F_in': F, 'N': train_disp.shape[1],
        'n_components': out_dim,
    }


# =============================================================================
# MODEL ARCHITECTURE (full architecture, same as ablation full model)
# =============================================================================
class CausalConv1d(nn.Module):
    def __init__(self, ci, co, k, d):
        super().__init__()
        self.pad = (k - 1) * d; self.conv = nn.Conv1d(ci, co, k, dilation=d, padding=self.pad)
    def forward(self, x):
        out = self.conv(x)
        if self.pad > 0: return out[:, :, :-self.pad]
        return out


class TCNBlock(nn.Module):
    def __init__(self, ci, co, k, d, dropout):
        super().__init__()
        self.conv1 = CausalConv1d(ci, co, k, d); self.ln1 = nn.LayerNorm(co)
        self.conv2 = CausalConv1d(co, co, k, d); self.ln2 = nn.LayerNorm(co)
        self.drop = nn.Dropout(dropout)
        self.downsample = nn.Conv1d(ci, co, 1) if ci != co else None
        self.act = nn.LeakyReLU(0.1)
    def forward(self, x):
        res = x
        out = self.act(self.ln1(self.conv1(x).transpose(1, 2)).transpose(1, 2))
        out = self.drop(out)
        out = self.act(self.ln2(self.conv2(out).transpose(1, 2)).transpose(1, 2))
        out = self.drop(out)
        if self.downsample is not None: res = self.downsample(res)
        return self.act(out + res)


class TCNEncoder(nn.Module):
    def __init__(self, in_dim, channels, k, dilations, dropout):
        super().__init__()
        layers = []; ci = in_dim
        for co, d in zip(channels, dilations):
            layers.append(TCNBlock(ci, co, k, d, dropout)); ci = co
        self.net = nn.Sequential(*layers); self.out_dim = channels[-1]
    def forward(self, x): return self.net(x.transpose(1, 2)).transpose(1, 2)


class GatedGCN(nn.Module):
    def __init__(self, ci, co, adj, dropout=0.5):
        super().__init__()
        self.register_buffer('adj', adj)
        self.lin = nn.Linear(ci, co); self.gate = nn.Linear(ci + co, co)
        self.drop = nn.Dropout(dropout)
    def forward(self, x, dynamic_adj=None):
        B, N, _ = x.shape
        adj = self.adj if dynamic_adj is None else dynamic_adj
        if adj.dim() == 2: adj = adj.unsqueeze(0).expand(B, N, N)
        t = self.lin(x); agg = torch.bmm(adj, t)
        gate = torch.sigmoid(self.gate(torch.cat([x, agg], -1)))
        return self.drop(F.relu(gate * t + (1 - gate) * agg))


class PhysicsGate(nn.Module):
    def __init__(self, dim, geo_dim=7):
        super().__init__()
        self.gate = nn.Sequential(nn.Linear(dim + geo_dim, dim), nn.Sigmoid())
    def forward(self, x, geo_feat): return x * self.gate(torch.cat([x, geo_feat], dim=-1))


class CrossNodeAttention(nn.Module):
    """式(18)-(20)：跨节点注意力 + 逐节点融合门。

    注：曾试过在打分上叠加地质相似性偏置 γ·A^geo（式(10) 的矩阵）。15 种子 A/B 显示
    γ 训练后仅 ~1e-3 量级、对结果无可测影响（无水位支路的变体逐位不变），故移除；
    地质先验经式(11) 的邻接拓扑与式(21) 的融合门进入模型。
    """
    def __init__(self, dim, geo_dim=7):
        super().__init__()
        self.dim = dim; self.q = nn.Linear(dim, dim); self.k = nn.Linear(dim, dim)
        self.v = nn.Linear(dim, dim); self.gate = nn.Linear(dim + geo_dim, 1)
    def forward(self, x, geo_feat):
        B, N, D = x.shape
        q = self.q(x); k = self.k(x); v = self.v(x)
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
        return x + torch.bmm(vert_adj.expand(B, N, N), self.vert_proj(x))


class PCAModel(nn.Module):
    """Full model with configurable output dimension (PCA retention rate)."""
    def __init__(self, cfg, N, in_dim, gcn_hid, tcn_ch, k, dil, attn_hid, hor,
                 adj, out_dim, geo_feat_dim=7, vert_adj=None,
                 use_vertical_gcn=True):
        super().__init__()
        self.use_vertical_gcn = use_vertical_gcn
        self.tcn = TCNEncoder(in_dim, tcn_ch, k, dil, cfg.TCN_DROPOUT)
        tcn_dim = tcn_ch[-1]

        # 架构简化（2026-09-19）：时序注意力改为均值池化（与消融 Ablation_Experiment
        # 同步，见其 __init__ 注释）。删除 attn1/attn_v 模块。
        # 数据导出层（2026-09-18）：暂存时间注意力权重（简化后恒 None）
        self._last_temporal_attn = None

        # 路径 A（2026-09-17）：PCAModel 与消融全模块架构同步，水文已彻底移除。
        self.use_water_gcn = False

        # 2026-09-20：原此处另注册 `static_adj`（与 `self.gcn.adj` 重复），无消费者，已删除。

        # Module creation order unified with Ablation_Experiment.AblationModel
        # (tcn -> gcn -> physics -> cross -> vertical_gcn -> head).
        # PyTorch consumes the global RNG for weight init in construction order;
        # matching the ablation's order keeps PCA's "full model" initial weights
        # identical to ablation full model (issue: PCA vs ablation absolute mismatch).
        # 2026-09-22：V7（use_vertical_gcn=False）不创建 vertical_gcn，
        # 与 AblationModel(use_vertical_gcn=False) 的 RNG 消费顺序一致。
        self.gcn = GatedGCN(tcn_dim, gcn_hid, adj, dropout=cfg.DROPOUT)
        self.gcn_norm = nn.LayerNorm(gcn_hid); self.gcn_drop = nn.Dropout(cfg.DROPOUT)

        if vert_adj is not None: self.register_buffer('vert_adj', vert_adj)
        else: self.vert_adj = None

        self.physics_gate = PhysicsGate(gcn_hid, geo_feat_dim)
        self.cross_attn = CrossNodeAttention(gcn_hid, geo_feat_dim)

        self.vertical_gcn = VerticalGCN(tcn_dim) if self.use_vertical_gcn else None

        self.head = nn.Sequential(
            nn.Linear(gcn_hid, gcn_hid // 2), nn.LeakyReLU(0.1),
            nn.Dropout(cfg.DROPOUT), nn.Linear(gcn_hid // 2, hor * out_dim))
        self.horizon = hor; self.out_dim = out_dim

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
        ctx = self.vertical_gcn(ctx, self.vert_adj) if self.use_vertical_gcn else ctx
        ctx = self.gcn(ctx, dynamic_adj=None)
        ctx = self.gcn_norm(ctx); ctx = self.gcn_drop(F.relu(ctx))

        ctx = self.physics_gate(ctx, geo_static)
        ctx = self.cross_attn(ctx, geo_static)

        global_feat = ctx.mean(dim=1)
        out = self.head(global_feat)
        return out.reshape(B, self.horizon, self.out_dim)


# =============================================================================
# TRAINING & EVALUATION
# =============================================================================
class MultiLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.huber = nn.HuberLoss(delta=1.0); self.mse = nn.MSELoss()
    def forward(self, pred, target):
        B, hor, dim = pred.shape
        l_hub = self.huber(pred, target)
        l_diff = torch.tensor(0., device=pred.device)
        if hor > 1: l_diff = self.mse(pred[:, 1:] - pred[:, :-1], target[:, 1:] - target[:, :-1])
        pred_fft = torch.fft.rfft(pred, dim=1); target_fft = torch.fft.rfft(target, dim=1)
        l_freq = self.mse(torch.abs(pred_fft), torch.abs(target_fft))
        return 1.0 * l_hub + 0.5 * l_diff + 0.2 * l_freq


def _graph_lr_param_groups(model):
    """图/物理相关模块参数（低学习率组）与其余参数。

    2026-09-20：改为**按模块对象显式收集**，不再依赖参数名子串匹配
    （原 'vert' 前缀过宽、'gcn' 会顺带命中 gcn_norm）。变体关闭时模块
    不存在，用 getattr 容错；返回顺序与原实现逐位等价。
    """
    holders = [getattr(model, n, None) for n in
               ('gcn', 'gcn_norm', 'gcn_drop', 'physics_gate',
                'cross_attn', 'vertical_gcn')]
    gid = {id(p) for m in holders if m is not None for p in m.parameters()}
    gra = [p for _, p in model.named_parameters() if id(p) in gid]
    rest = [p for _, p in model.named_parameters() if id(p) not in gid]
    return gra, rest


def train_model(cfg, model, train_ld, val_ld, save_path, tag='', seed=0):
    gra, rest = _graph_lr_param_groups(model)
    opt = torch.optim.AdamW([
        {'params': gra, 'lr': cfg.LR_GCN},
        {'params': rest, 'lr': cfg.LR}
    ], weight_decay=cfg.WD)
    sch = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode='min', factor=0.5, patience=15)
    crit = MultiLoss().to(cfg.DEVICE)
    best_vloss = float('inf'); wait = 0; t_start = time.time()
    tr_losses, va_losses = [], []
    log_interval = max(1, cfg.EPOCHS // 20)
    for ep in range(1, cfg.EPOCHS + 1):
        model.train(); tr_loss = 0
        for batch_data in train_ld:
            Xb = batch_data[0].to(cfg.DEVICE); yb = batch_data[1].to(cfg.DEVICE)
            opt.zero_grad()
            pred = model(*([Xb] + [t.to(cfg.DEVICE) for t in batch_data[2:]]))
            loss = crit(pred, yb); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.GRAD_CLIP)
            opt.step(); tr_loss += loss.item() * Xb.size(0)
        tr_loss /= len(train_ld.dataset)
        tr_losses.append(tr_loss)
        model.eval(); va_loss = 0
        with torch.no_grad():
            for batch_data in val_ld:
                Xb = batch_data[0].to(cfg.DEVICE); yb = batch_data[1].to(cfg.DEVICE)
                pred = model(*([Xb] + [t.to(cfg.DEVICE) for t in batch_data[2:]]))
                loss = crit(pred, yb); va_loss += loss.item() * Xb.size(0)
        va_loss /= len(val_ld.dataset); va_losses.append(va_loss); sch.step(va_loss)
        if ep % log_interval == 0 or ep == 1 or ep == cfg.EPOCHS:
            logger.info(f"    [{tag} seed={seed}] Epoch {ep:4d}/{cfg.EPOCHS} | "
                        f"train={tr_loss:.6f} | val={va_loss:.6f}")
        if va_loss < best_vloss: best_vloss = va_loss; wait = 0; torch.save(model.state_dict(), save_path)
        else: wait += 1
        if wait >= cfg.PATIENCE: break
    train_time = time.time() - t_start
    model.load_state_dict(torch.load(save_path, map_location=cfg.DEVICE))
    return model, train_time, tr_losses, va_losses


def evaluate(model, test_ld, init_disp, scY, pca_y, hz, device, N_orig,
             true_diff_win=None):
    """测试集评估：预测走 PCA 逆变换，真值取原始一阶差分。

    2026-09-20 口径修正：此前真值也做 pca_y.inverse_transform 往返，把真值投影到
    K 维子空间后再比，丢弃的增量方差（NoPCA 与非 NoPCA 之间尤其明显）不计入误差，
    使 PCA 保留率之间的比较失真。现真值统一取原始 diff1 切片的累加。
    """
    model.eval()
    preds_norm, trues_norm = [], []
    with torch.no_grad():
        for batch_data in test_ld:
            Xb = batch_data[0].to(device); yb = batch_data[1].to(device)
            pred = model(*([Xb] + [t.to(device) for t in batch_data[2:]]))
            preds_norm.append(pred.cpu().numpy()); trues_norm.append(yb.cpu().numpy())
    pred_n = np.concatenate(preds_norm, 0)
    out_dim = pred_n.shape[-1]; ns = pred_n.shape[0]
    pred_un = scY.inverse_transform(pred_n.reshape(-1, out_dim)).reshape(ns, hz, out_dim)
    pred_diff = pca_y.inverse_transform(pred_un.reshape(-1, out_dim)).reshape(
        ns, hz, N_orig) if pca_y is not None else pred_un
    if true_diff_win is not None:
        true_diff = np.asarray(true_diff_win, dtype=np.float64)[:ns]
    else:
        # 兼容路径：未提供原始真值窗口时回退旧口径
        true_n = np.concatenate(trues_norm, 0)
        true_un = scY.inverse_transform(true_n.reshape(-1, out_dim)).reshape(ns, hz, out_dim)
        true_diff = pca_y.inverse_transform(true_un.reshape(-1, out_dim)).reshape(
            ns, hz, N_orig) if pca_y is not None else true_un
    pred_cum = np.zeros_like(pred_diff); true_cum = np.zeros_like(true_diff)
    for i in range(ns):
        cp = init_disp[i].copy(); ct = init_disp[i].copy()
        for t in range(hz): cp += pred_diff[i, t, :]; ct += true_diff[i, t, :]; pred_cum[i, t, :] = cp; true_cum[i, t, :] = ct
    return pred_cum, true_cum


def global_metrics(true_cum, pred_cum):
    """全步口径 + 末步口径（`_last` 后缀）。

    2026-09-20：原实现把 (ns, hz, N) 一并 ravel，同一真实时点被窗口重叠反复计入，
    且重复倍数随 hz 增大（H=2 约 1.9x、H=12 约 5.4x）→ 长视界的 RMSE/R² 天然更好
    看，跨视界比较被污染。末步口径每窗只取 t=hz-1，ns 个窗口对应 ns 个互不重复的
    时点，为论文主口径；原列名为全步辅助口径，语义不变。
    """
    def _m(a, b):
        return {'RMSE': float(np.sqrt(mean_squared_error(a, b))),
                'MAE': float(mean_absolute_error(a, b)),
                'R2': float(r2_score(a, b)),
                'MaxErr': float(max_error(a, b))}
    gm = _m(true_cum.ravel(), pred_cum.ravel())
    gm.update({f'{k}_last': v for k, v in
               _m(true_cum[:, -1, :].ravel(), pred_cum[:, -1, :].ravel()).items()})
    return gm


def collect_gate_values(model, ld, device):
    """收集式(13) 的 g_t 在各测试窗上的取值，用于诊断其是否真的随水文变化。"""
    model.eval()
    gs = []
    with torch.no_grad():
        for batch_data in ld:
            Xb = batch_data[0].to(device)
            model(*([Xb] + [t.to(device) for t in batch_data[2:]]))
            g = getattr(model, '_last_g', None)
            if g is None: return None
            gs.append(g.reshape(-1).float().cpu().numpy())
    return np.concatenate(gs) if gs else None


def count_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# =============================================================================
# RUN SINGLE PCA CONDITION
# =============================================================================
def run_pca_condition(cfg, shared_cache, hz, pca_key, pca_variance):
    tag = f"PCA_{pca_key}_H{hz}"
    summary_dir = cfg.OUT_ROOT / f"H{hz}" / pca_key / "summary"
    if summary_dir.exists() and list(summary_dir.glob('global_*runs.csv')):
        logger.info(f"  [{tag}] already done, skipping")
        return pd.read_csv(sorted(summary_dir.glob('global_*runs.csv'))[-1])

    logger.info(f"  [{tag}] begin")

    full_df = shared_cache['full_df']; ts = shared_cache['ts']
    wd = shared_cache['wd']; wa = shared_cache['wa']; water_feat = shared_cache['water_feat']
    meta = shared_cache['meta']; geo_feat_raw = shared_cache['geo_feat_raw']
    train_idx = shared_cache['train_idx']; val_idx = shared_cache['val_idx']; test_idx = shared_cache['test_idx']

    geo_fb = geo_feat_raw if geo_feat_raw is not None else np.zeros((len(meta['depths']), cfg.GEO_FEAT_DIM), dtype=np.float32)
    data = prepare_data_with_pca(cfg, full_df, wd, wa, meta, train_idx, val_idx, test_idx,
                                  water_feat, geo_fb, hz, pca_variance)

    out_dim = data['out_dim']; n_components_actual = data['n_components']
    N_data = data['N']; F_in = data['F_in']

    geo_fb_use = geo_feat_raw if geo_feat_raw is not None and np.any(geo_feat_raw) else None
    adj = build_adjacency(meta, geo_fb_use, cfg)
    adj_t = torch.FloatTensor(adj).to(cfg.DEVICE)
    vert_adj_np = build_vertical_adjacency(meta, cfg)
    vert_adj_t = torch.FloatTensor(vert_adj_np).to(cfg.DEVICE)

    geo_static_raw = np.concatenate([
        geo_fb, (meta['depths'] / 16.0).reshape(-1, 1)], axis=1)
    geo_static = (geo_static_raw - geo_static_raw.mean(0)) / (geo_static_raw.std(0) + 1e-6)
    geo_dim = geo_static.shape[1]

    # 路径 A（2026-09-17）：水位已不再进入模型，以下 water 特征仅作占位透传，
    # 使 DataLoader / forward 签名保持不变（模型内部不再使用 water_feat）。
    w_mean = water_feat[train_idx].mean(0)
    w_std = water_feat[train_idx].std(0) + 1e-6
    water_feat_s = (water_feat - w_mean) / w_std
    wf_train = water_feat_s[train_idx][cfg.LOOKBACK - 1:-hz][:data['Xtr'].shape[0]]
    wf_val = water_feat_s[val_idx][cfg.LOOKBACK - 1:-hz][:data['Xva'].shape[0]]
    wf_test = water_feat_s[test_idx][cfg.LOOKBACK - 1:-hz][:data['Xte'].shape[0]]
    wf_tr = np.tile(wf_train[:, np.newaxis, :], (1, N_data, 1)).astype(np.float32)
    wf_va = np.tile(wf_val[:, np.newaxis, :], (1, N_data, 1)).astype(np.float32)
    wf_te = np.tile(wf_test[:, np.newaxis, :], (1, N_data, 1)).astype(np.float32)
    geo_tr_np = np.tile(geo_static[np.newaxis], (data['Xtr'].shape[0], 1, 1))
    geo_va_np = np.tile(geo_static[np.newaxis], (data['Xva'].shape[0], 1, 1))
    geo_te_np = np.tile(geo_static[np.newaxis], (data['Xte'].shape[0], 1, 1))

    out_dir = cfg.OUT_ROOT / f"H{hz}" / pca_key
    out_dir.mkdir(parents=True, exist_ok=True)
    # 数据导出层（2026-09-18）：保存 PCA 主成分载荷，供特征空间/模态重构可视化
    if data['pca_y'] is not None:
        np.savez(out_dir / 'pca_components.npz',
                 components=data['pca_y'].components_,
                 explained_variance_ratio=data['pca_y'].explained_variance_ratio_,
                 mean=data['pca_y'].mean_)

    all_global = []
    for si, seed in enumerate(cfg.SEEDS):
        set_seed(seed)

        # 2026-09-22：断点续跑（PCA_RESUME=1）——四件套齐全即视为已完成，
        # 读回 metrics.json 参与汇总，不重建模型、不重训（见 Config.RESUME 注释）。
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

        model = PCAModel(cfg, N=N_data, in_dim=F_in, gcn_hid=cfg.GCN_HIDDEN,
                         tcn_ch=cfg.TCN_CHANNELS, k=cfg.TCN_KERNEL_SIZE,
                         dil=cfg.TCN_DILATIONS, attn_hid=cfg.ATTENTION_HIDDEN,
                         hor=hz, adj=adj_t, out_dim=out_dim,
                         geo_feat_dim=geo_dim, vert_adj=vert_adj_t,
                         use_vertical_gcn=cfg.USE_VERTICAL_GCN).to(cfg.DEVICE)

        n_params = count_params(model)
        train_ds = TensorDataset(torch.FloatTensor(data['Xtr']), torch.FloatTensor(data['ytr']),
                                  torch.FloatTensor(wf_tr), torch.FloatTensor(geo_tr_np))
        val_ds = TensorDataset(torch.FloatTensor(data['Xva']), torch.FloatTensor(data['yva']),
                                torch.FloatTensor(wf_va), torch.FloatTensor(geo_va_np))
        test_ds = TensorDataset(torch.FloatTensor(data['Xte']), torch.FloatTensor(data['yte']),
                                 torch.FloatTensor(wf_te), torch.FloatTensor(geo_te_np))
        train_ld = DataLoader(train_ds, batch_size=cfg.BATCH_SIZE, shuffle=True)
        val_ld = DataLoader(val_ds, batch_size=cfg.BATCH_SIZE)
        test_ld = DataLoader(test_ds, batch_size=cfg.BATCH_SIZE)

        save_path = out_dir / f"run_{seed}" / "best.pth"
        save_path.parent.mkdir(parents=True, exist_ok=True)

        try:
            model, train_time, tr_losses, va_losses = train_model(cfg, model, train_ld, val_ld, save_path, tag=tag, seed=seed)
            pred_cum, true_cum = evaluate(model, test_ld, data['init_test'], data['scY'],
                                           data['pca_y'], hz, cfg.DEVICE, N_data,
                                           true_diff_win=data['yte_raw'])
            gm = global_metrics(true_cum, pred_cum)
            gm['seed'] = seed; gm['Params'] = n_params; gm['TrainTime'] = train_time
            gm['n_components'] = int(n_components_actual)
            # 水特征门控诊断已移除（见 __init__ 注释：门控恒为常数或引入方差）
            all_global.append(gm)

            with open(save_path.parent / 'metrics.json', 'w') as f: json.dump(gm, f, indent=2)
            np.savez(save_path.parent / 'predictions.npz', pred_cum=pred_cum, true_cum=true_cum)
            # 数据导出层（2026-09-18）
            np.save(save_path.parent / 'losses.npy', {'tr': tr_losses, 'va': va_losses})
            logger.info(f"    Seed {si + 1}/{len(cfg.SEEDS)} ({seed}) "
                        f"RMSE(last)={gm['RMSE_last']:.4f} "
                        f"RMSE(full)={gm['RMSE']:.4f} "
                        f"R2(last)={gm['R2_last']:.4f} Params={n_params}")
        except Exception as e:
            logger.error(f"    Seed {seed} failed: {e}"); import traceback; traceback.print_exc()

        del model; gc.collect()
        if torch.cuda.is_available(): torch.cuda.empty_cache()

    if not all_global:
        logger.error(f"  [{tag}] ALL seeds failed!"); return None

    df = pd.DataFrame(all_global)
    summary_dir = out_dir / "summary"; summary_dir.mkdir(exist_ok=True)
    for old_f in summary_dir.glob('global_*runs.csv'): old_f.unlink()
    df.to_csv(summary_dir / f'global_{len(all_global)}runs.csv', index=False)
    logger.info(f"  [{tag}] {len(all_global)} seeds done. "
                f"RMSE(last)={df['RMSE_last'].mean():.4f}+/-{df['RMSE_last'].std():.4f} "
                f"RMSE(full)={df['RMSE'].mean():.4f}+/-{df['RMSE'].std():.4f}")
    return df


# =============================================================================
# RESULT AGGREGATION & EXPORT (no plotting)
# =============================================================================
# 绘图代码已迁出至 可视化\02代码结果可视化\fig_pca.py（2026-09-18）。
# 实验脚本只负责数据聚合与落盘；今后改图/加图无需重跑实验。
def collect_all_pca_results(cfg, hz):
    records = []
    for pca_key in PCA_CONDITIONS:
        summary_dir = cfg.OUT_ROOT / f"H{hz}" / pca_key / "summary"
        glob_files = sorted(summary_dir.glob('global_*runs.csv'))
        if not glob_files: continue
        df = pd.read_csv(glob_files[-1])
        for _, row in df.iterrows():
            rec = {
                'Horizon': hz, 'Condition': pca_key,
                'Label': PCA_CONDITIONS[pca_key]['label'],
                'RMSE': row['RMSE'], 'MAE': row['MAE'],
                'R2': row['R2'], 'MaxErr': row['MaxErr'],
                'Params': row['Params'],
                'n_components': row.get('n_components', 0),
                'seed': row['seed']}
            # 2026-09-22：末步口径为论文主口径（见 global_metrics 注释），此处补齐
            # 与 Ablation/Comparison 汇总层对齐；旧 CSV 若无 `_last` 列则跳过。
            for m in ('RMSE_last', 'MAE_last', 'R2_last', 'MaxErr_last'):
                if m in row:
                    rec[m] = row[m]
            records.append(rec)
    return pd.DataFrame(records)


def summarize_pca_results(all_horizon_data, save_dir):
    """数据汇总（无绘图）：各视界逐条件的均值/标准差落盘为 CSV/XLSX。

    绘图代码已迁出至 可视化\02代码结果可视化\fig_pca.py（2026-09-18）。
    """
    records = []
    for hz, df in all_horizon_data.items():
        for cond in ['pca80', 'pca90', 'pca95', 'nopca']:
            sub = df[(df['Condition'] == cond)]
            if sub.empty: continue
            rec = {
                'Horizon': int(hz),
                'Condition': PCA_CONDITIONS[cond]['label'],
                'RMSE': f"{sub['RMSE'].mean():.4f}+/-{sub['RMSE'].std():.4f}",
                'R2': f"{sub['R2'].mean():.4f}",
                'Params': f"{int(sub['Params'].mean()):,}",
                'n_K': int(sub['n_components'].mean()) if cond != 'nopca' else 155
            }
            # 2026-09-22：追加末步口径（论文主口径，见 global_metrics 注释），与
            # Ablation/Comparison 汇总层对齐。原列名与列序不变，`_last` 列追加于
            # 末尾，以兼容下游 fig_pca.py 的按名取列。
            if 'RMSE_last' in sub.columns:
                rec['RMSE_last'] = (f"{sub['RMSE_last'].mean():.4f}"
                                    f"+/-{sub['RMSE_last'].std():.4f}")
            if 'R2_last' in sub.columns:
                rec['R2_last'] = f"{sub['R2_last'].mean():.4f}"
            if 'MAE_last' in sub.columns:
                rec['MAE_last'] = f"{sub['MAE_last'].mean():.4f}"
            records.append(rec)
    if not records: return
    save_dir = Path(save_dir); save_dir.mkdir(parents=True, exist_ok=True)
    df_tab = pd.DataFrame(records)
    df_tab.to_csv(save_dir / 'pca_summary.csv', index=False)
    df_tab.to_excel(save_dir / 'pca_summary.xlsx', index=False)
    logger.info(f"  PCA summary exported: {save_dir / 'pca_summary.csv'}")


# =============================================================================
# MAIN DRIVER
# =============================================================================
def main(quick_test=False):
    cfg = Config()

    if quick_test:
        cfg.HORIZONS = [6]; cfg.SEEDS = cfg.SEEDS[:1]; cfg.EPOCHS = 5; cfg.PATIENCE = 3
        # 2026-09-22：quick 模式隔离输出目录。否则 1 个种子留下的 global_1runs.csv
        # 会被 run_pca_condition 的条件级 skip（只看 summary 目录有无该文件）命中，
        # 导致后续全量跑把该视界 4 个条件整体跳过、永久只有 1 个种子。
        cfg.OUT_ROOT = cfg.OUT_ROOT / "_quick"
        logger.info("=" * 60)
        logger.info("QUICK TEST MODE: H=6 only, 1 seed, 5 epochs")
        logger.info(f"QUICK TEST OUTPUT (isolated): {cfg.OUT_ROOT}")
        logger.info("=" * 60)
    else:
        logger.info("=" * 60)
        logger.info("FULL PCA EXPERIMENT")
        logger.info(f"Variant: {cfg.VARIANT_TAG} "
                    f"(USE_VERTICAL_GCN={cfg.USE_VERTICAL_GCN})")
        logger.info(f"Horizons: {cfg.HORIZONS}")
        logger.info(f"PCA conditions: {list(PCA_CONDITIONS.keys())}")
        logger.info(f"Seeds: {len(cfg.SEEDS)}")
        logger.info(f"Total runs: {len(cfg.HORIZONS)} x {len(PCA_CONDITIONS)} x {len(cfg.SEEDS)} = "
                    f"{len(cfg.HORIZONS) * len(PCA_CONDITIONS) * len(cfg.SEEDS)}")
        logger.info("=" * 60)

    out_root = cfg.OUT_ROOT
    logger.info(f"Device: {cfg.DEVICE}"); logger.info(f"Output: {out_root}")

    # Shared data
    set_seed(42)
    full_df = load_raw_data_with_meta(cfg)
    ts = np.sort(full_df['测量时间'].unique())
    wd, wa = load_water_data(ts, cfg)
    water_feat = extract_water_features(wd, wa)
    _, _, meta = build_node_features(full_df, cfg)
    geo_records = load_geology_params(cfg)
    geo_feat_raw = assign_geo_to_nodes(geo_records, meta['depths'], cfg) if geo_records else None
    n = len(ts)
    tr_end = int(n * cfg.TRAIN_RATIO); va_end = int(n * (cfg.TRAIN_RATIO + cfg.VAL_RATIO))
    train_idx = np.arange(tr_end); val_idx = np.arange(tr_end, va_end); test_idx = np.arange(va_end, n)

    shared_cache = {
        'full_df': full_df, 'ts': ts, 'wd': wd, 'wa': wa, 'water_feat': water_feat,
        'meta': meta, 'geo_feat_raw': geo_feat_raw,
        'train_idx': train_idx, 'val_idx': val_idx, 'test_idx': test_idx
    }

    all_horizon_data = {}
    for hz in cfg.HORIZONS:
        logger.info(f"\n{'#' * 60}\n#  HORIZON H={hz}\n{'#' * 60}")
        for pca_key, pca_info in PCA_CONDITIONS.items():
            df = run_pca_condition(cfg, shared_cache, hz, pca_key, pca_info['pca_variance'])
            if df is not None:
                df['Horizon'] = hz; df['Condition'] = pca_key

        df_all = collect_all_pca_results(cfg, hz)
        if not df_all.empty:
            all_horizon_data[hz] = df_all
            logger.info(f"  H={hz}: collected {len(df_all)} result rows")

    comp_dir = out_root / "comparison"
    comp_dir.mkdir(parents=True, exist_ok=True)
    # 数据导出层（2026-09-18）：逐种子明细宽表（含全部 30 种子，非仅均值）
    if all_horizon_data:
        pd.concat([d.assign(Horizon=h) for h, d in all_horizon_data.items()],
                  ignore_index=True).to_pickle(comp_dir / 'full_seed_data.pkl')
    summarize_pca_results(all_horizon_data, comp_dir)
    logger.info(f"\n{'=' * 60}\nALL DONE! Results saved to: {out_root}\n{'=' * 60}")


if __name__ == '__main__':
    quick = '--quick' in sys.argv or '-q' in sys.argv
    main(quick_test=quick)