# =============================================================================
# Spatial_LOOCV.py — Leave-One-Section-Out 空间交叉验证
#
# 对 5 个断面（CX2, CX3, CX8, CX13, CX15）逐一留一，检验 V7 (Ours) 的**空间泛化**：
# 在完全看不到某断面当前读数的情况下，能否由其余 4 个断面重建该断面。
#
# 每折掩码三条**输入**路径，使留出断面在推理时不可见：
#   1. ctx（TCN 输出）→ 乘 node_mask，阻止留出断面信号进入空间模块
#   2. geo_static（地质静态特征）→ 乘 node_mask，阻止 PhysicsGate/CrossNodeAttn
#      通过地质特征绕过掩码
#   3. vert_adj（竖向邻接矩阵）→ 断开留出节点所有入边
#      （V7 无 VerticalGCN 通路，此项对当前变体为空操作，保留以兼容 V5）
#
# 2026-09-23（口径修复，重要）：此前**留出断面的预测目标仍参与训练**——原生
# MultiLoss 在 PCA 系数空间（K=13）计算，而每个系数是全部 155 个节点的全局混合，
# 因此无法按节点掩码。后果：模型在训练时就"见过"该断面的变形历史（被监督去复现
# 它），故旧口径实际测的是「由可见断面**插补**该断面」，而非脚本与论文所声称的
# 「预测一个**从未见过**的断面」。现改用 `MaskedFieldLoss`：先把 pred/target 由
# 系数空间逆变换回 155 节点场空间，再乘 node_mask 后计算同形式的损失，从而在训练时
# **完全排除**留出断面 —— 这才是名实相符的"留一断面"泛化检验。
#
# 残余（须在论文中声明的）局限 —— 不构成目标泄露，但使本实验仍是**偏乐观**估计：
#   a) PCA 基在**全部 155 节点**的训练期数据上拟合，故基中已含断面间协方差结构；
#      由于"全局 PCA 基 + 单一全场输出头"框架下，留出断面的预测量本身就是基的线性
#      组合，无法在数学上同时做到"基不含留出断面"与"能预测留出断面"——这是该框架的
#      结构性限制，而非实现缺陷。故本实验回答的是："在已知该墙体变形模态（由训练期
#      数据估计）的前提下，能否由可见断面预测未见断面"。
#   b) 架构为"全局节点均值池化 + 单一预测头"，训练期留出断面的输入已被掩码，故
#      池化只汇总可见断面（此项不构成泄露，仅说明输出为全场单一预测）。
#
# 每折 5 种子 × 5 折 × 5 视界 = 125 次训练。
# 输出目录：Spatial_LOOCV_Results/
#
# 评估口径：
#   主口径 = 末步 RMSE_last（论文主表口径，与消融/对比一致）
#   辅助口径 = 全步 RMSE（保留作为参照）
#   泛化间隙 = (留出断面 RMSE − 训练断面 RMSE) / 训练断面 RMSE × 100%
#   2026-09-23：RMSE/R² 一律改为 **MSE 聚合**口径（逐种子先算 MSE/Var，再跨种子与
#   跨断面平均）。此前用"逐节点 R² 取中位数"，在小变形节点上产生 −20 量级的伪值，
#   且 gap_R2 除以 |train_R2| 会给出 ±100% 的无意义数，已废弃。
#
# 用法：
#   python Spatial_LOOCV.py              # 完整运行
#   python Spatial_LOOCV.py --quick      # 快速测试（2 种子，10 epochs）
# =============================================================================

import _robust_io  # noqa: F401

import os
import sys
import json
import logging
import warnings
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

# =============================================================================
# 复用消融实验的全部基础设施（导入安全：Ablation_Experiment.py 有 __main__ guard）
# =============================================================================
from Ablation_Experiment import (
    Config as BaseConfig,
    set_seed,
    load_raw_data_with_meta,
    load_water_data,
    extract_water_features,
    build_node_features,
    load_geology_params,
    assign_geo_to_nodes,
    build_adjacency,
    build_vertical_adjacency,
    denoise_split,
    make_features,
    prepare_data,
    AblationModel,
    MultiLoss,
    _graph_lr_param_groups,
    train_model,
    evaluate,
    global_metrics,
    node_metrics,
    count_params,
    output_space_floor,
    cohens_d,
    VARIANT_SHORT_NAMES,
    VARIANT_COLORS,
)

warnings.filterwarnings('ignore')
logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)


# =============================================================================
# CONFIGURATION — 继承 BaseConfig，仅覆盖 LOOCV 特有项
# =============================================================================
class SpatialConfig(BaseConfig):
    OUT_ROOT = Path(__file__).parent / "Spatial_LOOCV_Results"
    OUT_ROOT.mkdir(parents=True, exist_ok=True)

    SEEDS = BaseConfig.SEEDS[:5]  # 每折 5 种子（留一验证主要方差来自折间，非种子间）
    HORIZONS = [2, 6, 8, 10, 12]  # 全视界，与消融/对比一致

    # 5 个断面名（与 POINTS 列表对应，每个断面包含 POINTS 的全部 CDEPTHS 深度）
    SECTION_NAMES = BaseConfig.POINTS  # ['CX2', 'CX3', 'CX8', 'CX13', 'CX15']

    # 断点续跑：ABLATION_RESUME 或 SPATIAL_RESUME 任一个为 '1' 即开启
    RESUME = (BaseConfig.RESUME
              or os.environ.get('SPATIAL_RESUME', '0') == '1')

    # 快速测试开关
    QUICK = '--quick' in sys.argv or '-q' in sys.argv

    @property
    def N_DEPTHS(self):
        return len(self.DEPTHS)  # 31


# =============================================================================
# HELPERS
# =============================================================================
def build_section_masks(cfg):
    """为 5 个断面各生成 [N_nodes] 的 0/1 掩码。

    Returns
    -------
    masks : dict[str, np.ndarray]
        key = 断面名（如 'CX2'），value = (N_nodes,) 的 float32 掩码，
        被留断面对应索引 = 0，其余 = 1。
    indices : dict[str, np.ndarray]
        key = 断面名，value = 该断面在 155 节点中的索引数组。
    """
    n_dep = cfg.N_DEPTHS
    masks = {}
    indices = {}
    for si, section in enumerate(cfg.SECTION_NAMES):
        mask = np.ones(len(cfg.SECTION_NAMES) * n_dep, dtype=np.float32)
        start = si * n_dep
        end = start + n_dep
        mask[start:end] = 0.0
        masks[section] = mask
        indices[section] = np.arange(start, end)
    return masks, indices


def per_section_metrics(df_node, meta, cfg):
    """按断面拆分逐节点指标，每断面取中位数。

    Parameters
    ----------
    df_node : pd.DataFrame
        node_metrics 输出，含 Node / RMSE / RMSE_last / R2 / R2_last / MAE / MAE_last 列。
    meta : dict
        含 'point_names' 列表（每个节点所属断面名）。
    cfg : SpatialConfig

    Returns
    -------
    dict[str, dict[str, float]]
        key = 断面名，value = {RMSE_last, RMSE, R2_last, R2, MAE_last, MAE}
    """
    if 'Point' not in df_node.columns:
        df_node = df_node.copy()
        df_node['Point'] = [n.split('_')[0] if '_' in str(n) else 'unknown'
                            for n in df_node['Node']]

    results = {}
    for sec in cfg.SECTION_NAMES:
        sub = df_node[df_node['Point'] == sec]
        if len(sub) == 0:
            continue
        results[sec] = {}
        for col in ('RMSE', 'RMSE_last', 'R2', 'R2_last', 'MAE', 'MAE_last'):
            if col in sub.columns:
                results[sec][col] = float(sub[col].median())
    return results


class MaskedFieldLoss(nn.Module):
    """LOOCV 专用损失：在**节点空间**计算，并把留出断面从损失中剔除。

    为什么需要它：原生 `MultiLoss` 在 PCA 系数空间（K=13）计算，而每个系数是
    全部 155 个节点的全局线性混合，因此**无法按节点掩码** —— 留出断面的真值会
    经系数进入训练损失（旧口径的漏洞）。本类先把 pred / target 由"标准化系数空间"
    逆变换回 155 节点场空间，再乘 node_mask（留出节点置零），最后按与 `MultiLoss`
    完全相同的三项形式（Huber + 一阶差分 MSE + 频域 MSE）计算，从而在训练时
    完全排除留出断面。

    逆变换链与 `Ablation_Experiment.evaluate` 一致：
        coeff(标准化) --scY.inverse_transform--> coeff(PCA) --pca.inverse_transform--> 节点场
    """

    def __init__(self, scY, pca_y, node_mask):
        super().__init__()
        self.register_buffer('scY_mean',
                             torch.tensor(scY.mean_, dtype=torch.float32))
        self.register_buffer('scY_scale',
                             torch.tensor(scY.scale_, dtype=torch.float32))
        self.register_buffer('pca_mean',
                             torch.tensor(pca_y.mean_, dtype=torch.float32))
        self.register_buffer('pca_comp',
                             torch.tensor(pca_y.components_, dtype=torch.float32))
        self.register_buffer('mask',
                             torch.tensor(node_mask, dtype=torch.float32))
        self.huber = nn.HuberLoss(delta=1.0)
        self.mse = nn.MSELoss()

    def _to_field(self, coeff):
        """(B, hor, K) 标准化系数 → (B, hor, N) 节点空间一阶差分。"""
        z = coeff * self.scY_scale + self.scY_mean      # 反标准化
        return z @ self.pca_comp + self.pca_mean        # PCA 逆变换

    def forward(self, pred, target, epoch=None):
        m = self.mask.view(1, 1, -1)
        pf = self._to_field(pred) * m
        tf = self._to_field(target) * m
        l_hub = self.huber(pf, tf)
        l_diff = torch.tensor(0., device=pf.device)
        if pf.shape[1] > 1:
            l_diff = self.mse(pf[:, 1:] - pf[:, :-1], tf[:, 1:] - tf[:, :-1])
        pred_fft = torch.fft.rfft(pf, dim=1)
        target_fft = torch.fft.rfft(tf, dim=1)
        l_freq = self.mse(torch.abs(pred_fft), torch.abs(target_fft))
        return 1.0 * l_hub + 0.5 * l_diff + 0.2 * l_freq


def section_metrics_from_cum(pred_cum, true_cum, node_names):
    """按断面计算 MSE / Var（节点空间），供后续 MSE 聚合出 RMSE 与 R²。

    逐种子先算 MSE 与 Var，聚合时再取比值，避免"逐节点 R² 取中位数"在小变形
    节点上产生的 −20 量级伪值（旧口径缺陷）。
    """
    secs = {}
    for j, n in enumerate(node_names):
        secs.setdefault(str(n).split('_')[0], []).append(j)

    rows = []
    for sec, idx in secs.items():
        if not idx:
            continue
        pl, tl = pred_cum[:, -1, idx], true_cum[:, -1, idx]
        pa, ta = pred_cum[:, :, idx], true_cum[:, :, idx]
        rows.append({
            'Section': sec,
            'MSE_last': float(np.mean((pl - tl) ** 2)),
            'Var_last': float(np.mean((tl - tl.mean()) ** 2)),
            'MSE_all': float(np.mean((pa - ta) ** 2)),
            'Var_all': float(np.mean((ta - ta.mean()) ** 2)),
        })
    return rows


def _agg_rmse_r2(df):
    """MSE 聚合口径：RMSE = sqrt(mean(MSE))，R² = 1 − mean(MSE)/mean(Var)。"""
    if df is None or len(df) == 0:
        return (np.nan,) * 4
    mse = df['MSE_last'].to_numpy(dtype=float)
    var = df['Var_last'].to_numpy(dtype=float)
    mse_a = df['MSE_all'].to_numpy(dtype=float)
    var_a = df['Var_all'].to_numpy(dtype=float)
    return (float(np.sqrt(mse.mean())),
            float(1.0 - mse.mean() / (var.mean() + 1e-12)),
            float(np.sqrt(mse_a.mean())),
            float(1.0 - mse_a.mean() / (var_a.mean() + 1e-12)))


def train_and_eval_one_seed(cfg, seed, data, meta, adj_t, geo_static_t,
                            vert_adj_t, node_mask, hz, save_dir):
    """单种子训练 → 评估。geo_static_t 传入时可能已在 CUDA，此处取 CPU 副本用于
    TensorDataset（避免 CUDA tensor 混入 CPU 数据集），训练循环内统一 .to(device)。"""
    set_seed(seed)
    Xtr, ytr = data['Xtr'], data['ytr']
    Xva, yva = data['Xva'], data['yva']
    Xte, yte = data['Xte'], data['yte']
    init_test = data['init_test']
    out_dim = data['out_dim']
    N = data['N']
    F_in = data['F_in']
    wf_train, wf_val, wf_test = data['wf_train'], data['wf_val'], data['wf_test']
    valid_mask_t = torch.FloatTensor(node_mask).to(cfg.DEVICE)

    geo_static_cpu = geo_static_t.cpu()
    train_ds = TensorDataset(
        torch.FloatTensor(Xtr), torch.FloatTensor(ytr),
        torch.FloatTensor(wf_train),
        geo_static_cpu.unsqueeze(0).expand(Xtr.shape[0], N, -1))
    val_ds = TensorDataset(
        torch.FloatTensor(Xva), torch.FloatTensor(yva),
        torch.FloatTensor(wf_val),
        geo_static_cpu.unsqueeze(0).expand(Xva.shape[0], N, -1))
    test_ds = TensorDataset(
        torch.FloatTensor(Xte), torch.FloatTensor(yte),
        torch.FloatTensor(wf_test),
        geo_static_cpu.unsqueeze(0).expand(Xte.shape[0], N, -1))
    train_ld = DataLoader(train_ds, batch_size=cfg.BATCH_SIZE, shuffle=True)
    val_ld = DataLoader(val_ds, batch_size=cfg.BATCH_SIZE)
    test_ld = DataLoader(test_ds, batch_size=cfg.BATCH_SIZE)

    geo_dim_actual = data['geo_dim_actual']
    model = AblationModel(cfg, N=N, in_dim=F_in, gcn_hid=cfg.GCN_HIDDEN,
                          tcn_ch=cfg.TCN_CHANNELS, k=cfg.TCN_KERNEL_SIZE,
                          dil=cfg.TCN_DILATIONS, attn_hid=cfg.ATTENTION_HIDDEN,
                          hor=hz, adj=adj_t, out_dim=out_dim,
                          geo_feat_dim=geo_dim_actual,
                          vert_adj=vert_adj_t).to(cfg.DEVICE)

    n_params = count_params(model)

    save_path = save_dir / "best.pth"
    save_path.parent.mkdir(parents=True, exist_ok=True)
    # 2026-09-23：损失改在**节点空间**计算并掩码留出断面，修复"留出断面真值仍
    # 进入训练损失"的口径漏洞（见类文档）。
    criterion = MaskedFieldLoss(data['scY'], data['pca_y'], node_mask)
    model, tr_losses, va_losses, train_time = train_model(
        cfg, model, train_ld, val_ld, save_path, node_mask=valid_mask_t,
        criterion=criterion)

    pred_cum, true_cum = evaluate(model, test_ld, init_test, data['scY'],
                                   data['pca_y'], hz, cfg.DEVICE,
                                   node_mask=valid_mask_t,
                                   true_diff_win=data['yte_raw'])

    gm = global_metrics(true_cum, pred_cum)
    _fl = output_space_floor(data['yte_raw'], init_test, data['scY'],
                             data['pca_y'], hz, ns=pred_cum.shape[0])
    gm['RMSE_floor_last'] = _fl['RMSE_floor_last']
    gm['RMSE_reach_last'] = float(np.sqrt(max(
        gm['RMSE_last'] ** 2 - _fl['MSE_floor_last'], 0.0)))
    gm['RMSE_floor'] = _fl['RMSE_floor_all']
    gm['RMSE_reach'] = float(np.sqrt(max(gm['RMSE'] ** 2 - _fl['MSE_floor_all'], 0.0)))
    gm['Params'] = n_params
    gm['TrainTime'] = train_time
    gm['seed'] = seed

    df_node = node_metrics(true_cum, pred_cum, meta['node_names'])
    sec_rows = section_metrics_from_cum(pred_cum, true_cum, meta['node_names'])

    with open(save_path.parent / 'metrics.json', 'w') as f:
        json.dump(gm, f, indent=2)
    df_node.to_excel(save_path.parent / 'node_metrics.xlsx', index=False)
    np.savez(save_path.parent / 'predictions.npz',
             pred_cum=pred_cum, true_cum=true_cum)
    np.savez(save_path.parent / 'losses.npz', tr=tr_losses, va=va_losses)

    return gm, df_node, sec_rows


# =============================================================================
# RUN ONE FOLD（一个断面作为留出对象）
# =============================================================================
def run_one_fold(cfg, holdout_section, node_mask, shared_data_cache, hz):
    """对单个断面做留一验证（三条掩码路径）。

    掩码策略：
      a) ctx（TCN 输出）→ AblationModel.forward 内乘 node_mask
      b) geo_static  → 乘 node_mask（阻止 PhysicsGate/CrossNodeAttn 泄露）
      c) vert_adj    → 断开留出节点入边（阻止 VerticalGCN 泄露）
    注：TCN 权重全节点训练 + PCA 逆变换全局 155→155 → 此为泛化上界。

    Parameters
    ----------
    cfg : SpatialConfig
    holdout_section : str
        被留出的断面名（如 'CX2'）
    node_mask : np.ndarray
        (155,) 的 0/1 掩码，0 = 留出断面节点
    shared_data_cache : dict
        来自 _build_shared_cache 的预计算数据
    hz : int
        预测视界

    Returns
    -------
    fold_global : list[dict]
        该折所有种子的 global_metrics 列表
    fold_section : dict[str, pd.DataFrame]
        该折按断面拆分的逐节点指标（含所有种子）
    """
    logger.info(f"  [{holdout_section}] H={hz} — 开始训练 ({len(cfg.SEEDS)} seeds)")

    fold_out = cfg.OUT_ROOT / f"Fold_{holdout_section}" / f"H{hz}"
    fold_out.mkdir(parents=True, exist_ok=True)

    geo_fb = shared_data_cache['geo_feat_raw']
    if geo_fb is not None and not np.any(geo_fb):
        geo_fb = None

    # 图结构与 Ablation V7 一致，随后掩码留出断面
    adj = build_adjacency(shared_data_cache['meta'], geo_fb, cfg,
                          zero_same_hole=False)
    adj_t = torch.FloatTensor(adj).to(cfg.DEVICE)

    vert_adj = build_vertical_adjacency(shared_data_cache['meta'], cfg)
    vert_adj_t = torch.FloatTensor(vert_adj)

    geo_static_raw = shared_data_cache['data']['geo_static']
    geo_static_t = torch.FloatTensor(geo_static_raw)

    # ── 掩码三条泄露路径（均在 CPU 上操作，最后统一 .to(DEVICE)）──
    node_mask_t = torch.FloatTensor(node_mask)
    # 路径 2: 地质静态特征 → 阻止 PhysicsGate/CrossNodeAttn 绕过 ctx 掩码
    geo_static_t = geo_static_t * node_mask_t.view(-1, 1)
    # 路径 3: 竖向邻接矩阵 → 断开留出节点所有入边
    mask_2d_v = node_mask_t.view(-1, 1) * node_mask_t.view(1, -1)
    vert_adj_t = vert_adj_t * mask_2d_v
    # 路径 1: ctx（TCN 输出）在 AblationModel.forward 内乘 node_mask，此处不重复
    # ── 掩码结束 ──

    geo_static_t = geo_static_t.to(cfg.DEVICE)
    vert_adj_t = vert_adj_t.to(cfg.DEVICE)

    # 保存全量配置快照以便复现
    cfg_vars = {k: str(v) for k, v in vars(cfg).items()
                if not k.startswith('_') and not callable(v)}
    cfg_vars['holdout_section'] = holdout_section
    cfg_vars['n_masked_nodes'] = int((node_mask == 0).sum())
    with open(fold_out / 'config.json', 'w') as f:
        json.dump(cfg_vars, f, indent=2, default=str)

    fold_global = []
    all_node_dfs = []
    all_sec_rows = []

    for si, seed in enumerate(cfg.SEEDS):
        seed_dir = fold_out / f"run_{seed}"

        # 断点续跑
        done_files = ('metrics.json', 'best.pth', 'predictions.npz', 'losses.npz')
        if cfg.RESUME and all((seed_dir / f).exists() for f in done_files):
            logger.info(f"    Seed {seed}: 已完成，跳过")
            with open(seed_dir / 'metrics.json', 'r') as f:
                gm = json.load(f)
            gm['seed'] = seed
            fold_global.append(gm)
            nf = seed_dir / 'node_metrics.xlsx'
            if nf.exists():
                all_node_dfs.append(pd.read_excel(nf))
            # 断面级指标由落盘的 pred/true 现算，避免依赖已废弃的节点中位数口径
            pf = seed_dir / 'predictions.npz'
            if pf.exists():
                z = np.load(pf)
                for r in section_metrics_from_cum(
                        z['pred_cum'], z['true_cum'],
                        shared_data_cache['meta']['node_names']):
                    r['seed'] = seed
                    all_sec_rows.append(r)
            continue

        t0 = time.time()
        gm, df_node, sec_rows = train_and_eval_one_seed(
            cfg, seed, shared_data_cache['data'], shared_data_cache['meta'],
            adj_t, geo_static_t, vert_adj_t, node_mask, hz, seed_dir)
        for r in sec_rows:
            r['seed'] = seed
            all_sec_rows.append(r)
        elapsed = time.time() - t0

        gm['seed'] = seed
        df_node['seed'] = seed
        fold_global.append(gm)
        all_node_dfs.append(df_node)
        logger.info(f"    Seed {seed} ({si + 1}/{len(cfg.SEEDS)}): "
                     f"RMSE_last={gm['RMSE_last']:.4f}, "
                     f"R2_last={gm['R2_last']:.4f}, "
                     f"time={elapsed:.0f}s")

    # 折内汇总
    df_g = pd.DataFrame(fold_global)
    df_g.to_csv(fold_out / 'global_seeds.csv', index=False)

    df_nodes_all = pd.concat(all_node_dfs, ignore_index=True)
    df_nodes_all.to_excel(fold_out / 'node_metrics_all.xlsx', index=False)

    # 断面级 MSE/Var 汇总（2026-09-23 新增，供 MSE 聚合出 RMSE 与 R²）
    df_sec = pd.DataFrame(all_sec_rows)
    if len(df_sec):
        df_sec.to_csv(fold_out / 'section_metrics_all.csv', index=False)

    # 按断面拆分
    fold_section = {}
    df_nodes_all['Point'] = [n.split('_')[0] if '_' in str(n) else 'unknown'
                              for n in df_nodes_all['Node']]
    for sec in cfg.SECTION_NAMES:
        sub = df_nodes_all[df_nodes_all['Point'] == sec].copy()
        if len(sub):
            fold_section[sec] = sub

    return fold_global, fold_section, df_sec


# =============================================================================
# BUILD SHARED DATA
# =============================================================================
def _build_shared_cache(cfg):
    """构建各折共享的数据缓存（只执行一次数据加载+预处理）。

    输出与 Ablation_Experiment 的 prepare_data 完全兼容（V7 变体开关，见
    _build_shared_cache 中的 USE_VERTICAL_GCN=False），
    同时额外保留未经 PCA 的时间切分索引与原始矩阵，供各折复用。
    """
    logger.info("=" * 60)
    logger.info("构建共享数据缓存（仅加载一次）...")
    t0 = time.time()

    set_seed(42)
    full_df = load_raw_data_with_meta(cfg)
    ts = np.sort(full_df['测量时间'].unique())
    wd, wa = load_water_data(ts, cfg)
    water_feat = extract_water_features(wd, wa)
    disp_raw, _, meta = build_node_features(full_df, cfg)
    geo_records = load_geology_params(cfg)
    geo_feat_raw = assign_geo_to_nodes(geo_records, meta['depths'], cfg) \
        if geo_records else np.zeros((len(meta['depths']), cfg.GEO_FEAT_DIM),
                                     dtype=np.float32)

    n = len(ts)
    tr_end = int(n * cfg.TRAIN_RATIO)
    va_end = int(n * (cfg.TRAIN_RATIO + cfg.VAL_RATIO))
    train_idx = np.arange(tr_end)
    val_idx = np.arange(tr_end, va_end)
    test_idx = np.arange(va_end, n)

    # 2026-09-22：对齐论文主模型——§5.2 消融后正文 Ours = V7（去竖向 GCN）。
    # V5 全量开关（含 USE_VERTICAL_GCN=True）已改为 V7：仅关竖向通路。
    cfg.USE_GCN = True
    cfg.USE_ATTN = True
    cfg.USE_CROSS_NODE_ATTN = True
    cfg.USE_PHYSICS_LOSS = True
    cfg.USE_VERTICAL_GCN = False
    cfg.USE_GEO_INJECT = False
    cfg.ZERO_SAME_HOLE = False

    logger.info(f"  时间轴: {len(ts)} 步 → train {len(train_idx)} / "
                f"val {len(val_idx)} / test {len(test_idx)}")
    logger.info(f"  断面: {cfg.SECTION_NAMES} × {cfg.N_DEPTHS} 深度 = "
                f"{len(cfg.SECTION_NAMES) * cfg.N_DEPTHS} 节点")

    elapsed = time.time() - t0
    logger.info(f"  共享缓存就绪 ({elapsed:.0f}s)")

    return {
        'full_df': full_df, 'ts': ts, 'wd': wd, 'wa': wa,
        'water_feat': water_feat, 'meta': meta,
        'geo_feat_raw': geo_feat_raw, 'disp_raw': disp_raw,
        'train_idx': train_idx, 'val_idx': val_idx, 'test_idx': test_idx,
    }


# =============================================================================
# AGGREGATION & REPORTING
# =============================================================================
def compute_loocv_summary(all_fold_sec, cfg, hz):
    """跨折汇总：留出断面 vs 训练断面的指标差异（MSE 聚合口径）。

    2026-09-23 重写：改用断面级 MSE/Var 聚合（`section_metrics_from_cum` 落盘），
    不再使用"逐节点 R² 取中位数"（会在小变形节点上给出 −20 量级伪值）。R² 的
    断面差异改为**原始差**（R² 单位），不再除以 |train_R²|（旧式在 R²≈−20 时
    输出 ±100% 的无意义值）。
    """
    records = []
    held_pool, train_pool = [], []

    for holdout in cfg.SECTION_NAMES:
        sec_df = all_fold_sec.get(holdout)
        if sec_df is None or len(sec_df) == 0:
            continue
        held = sec_df[sec_df['Section'] == holdout]
        train = sec_df[sec_df['Section'] != holdout]
        if len(held) == 0 or len(train) == 0:
            continue

        h_rmse_last, h_r2_last, h_rmse, h_r2 = _agg_rmse_r2(held)
        t_rmse_last, t_r2_last, t_rmse, t_r2 = _agg_rmse_r2(train)

        records.append({
            'Fold': holdout,
            'Heldout_RMSE_last': round(h_rmse_last, 4),
            'Train_RMSE_last': round(t_rmse_last, 4),
            'Gap_RMSE_last_pct': round((h_rmse_last - t_rmse_last) / (t_rmse_last + 1e-12) * 100, 2),
            'Heldout_R2_last': round(h_r2_last, 4),
            'Train_R2_last': round(t_r2_last, 4),
            'Diff_R2_last': round(t_r2_last - h_r2_last, 4),
            'Heldout_RMSE': round(h_rmse, 4),
            'Train_RMSE': round(t_rmse, 4),
            'Gap_RMSE_pct': round((h_rmse - t_rmse) / (t_rmse + 1e-12) * 100, 2),
        })
        held_pool.append(held)
        train_pool.append(train)

    df = pd.DataFrame(records)
    if len(df) == 0:
        return df

    # 均值行：跨折**池化**后聚合（每断面各作留出一次），优于对各折百分比取平均
    h_rmse_last, h_r2_last, h_rmse, h_r2 = _agg_rmse_r2(pd.concat(held_pool, ignore_index=True))
    t_rmse_last, t_r2_last, t_rmse, t_r2 = _agg_rmse_r2(pd.concat(train_pool, ignore_index=True))
    mean_row = {
        'Fold': 'Mean',
        'Heldout_RMSE_last': round(h_rmse_last, 4),
        'Train_RMSE_last': round(t_rmse_last, 4),
        'Gap_RMSE_last_pct': round((h_rmse_last - t_rmse_last) / (t_rmse_last + 1e-12) * 100, 2),
        'Heldout_R2_last': round(h_r2_last, 4),
        'Train_R2_last': round(t_r2_last, 4),
        'Diff_R2_last': round(t_r2_last - h_r2_last, 4),
        'Heldout_RMSE': round(h_rmse, 4),
        'Train_RMSE': round(t_rmse, 4),
        'Gap_RMSE_pct': round((h_rmse - t_rmse) / (t_rmse + 1e-12) * 100, 2),
    }
    df = pd.concat([df, pd.DataFrame([mean_row])], ignore_index=True)

    logger.info(f"\n{'=' * 60}")
    logger.info(f"LOOCV 汇总 — H={hz}（MSE 聚合口径）")
    logger.info(f"{'=' * 60}")
    for _, row in df.iterrows():
        if row['Fold'] == 'Mean':
            logger.info(f"  [池化] gap_RMSE_last = {row['Gap_RMSE_last_pct']:+.2f}%, "
                         f"ΔR²_last = {row['Diff_R2_last']:+.4f}")
        else:
            logger.info(f"  [{row['Fold']}] heldout_RMSE_last={row['Heldout_RMSE_last']:.4f}, "
                         f"train_RMSE_last={row['Train_RMSE_last']:.4f}, "
                         f"gap={row['Gap_RMSE_last_pct']:+.2f}%")
    logger.info(f"{'=' * 60}")

    return df


# =============================================================================
# MAIN
# =============================================================================
def main():
    cfg = SpatialConfig()

    logger.info("=" * 60)
    logger.info("SPATIAL LEAVE-ONE-SECTION-OUT CROSS-VALIDATION")
    logger.info(f"Model: V7 (Ours, 去竖向 GCN; USE_VERTICAL_GCN={cfg.USE_VERTICAL_GCN})")
    logger.info(f"Sections: {cfg.SECTION_NAMES}")
    logger.info(f"Seeds per fold: {len(cfg.SEEDS)}")
    logger.info(f"Horizons: {cfg.HORIZONS}")
    logger.info(f"Total runs: {len(cfg.SECTION_NAMES)} folds × "
                f"{len(cfg.HORIZONS)} horizons × "
                f"{len(cfg.SEEDS)} seeds = "
                f"{len(cfg.SECTION_NAMES) * len(cfg.HORIZONS) * len(cfg.SEEDS)}")
    logger.info(f"Device: {cfg.DEVICE}")
    logger.info(f"Output: {cfg.OUT_ROOT}")
    logger.info("=" * 60)

    # 快速测试模式
    if cfg.QUICK:
        cfg.HORIZONS = [6]
        cfg.SEEDS = cfg.SEEDS[:2]
        cfg.EPOCHS = 10
        cfg.PATIENCE = 5
        logger.info("⚠ QUICK MODE: H=6 only, 2 seeds, 10 epochs")

    # ---- Step 1: 构建共享缓存（只加载一次原始数据） ----
    shared = _build_shared_cache(cfg)

    # ---- Step 2: 生成断面掩码 ----
    masks, section_indices = build_section_masks(cfg)
    for sec, m in masks.items():
        logger.info(f"  断面 {sec}: 节点 {len(section_indices[sec])} 个, "
                     f"掩码中置零 = {int((m == 0).sum())}")

    # ---- Step 3: 按视界遍历 ----
    all_loocv_summaries = {}

    for hz in cfg.HORIZONS:
        logger.info(f"\n{'#' * 60}")
        logger.info(f"#  HORIZON H={hz}")
        logger.info(f"{'#' * 60}")

        # 为该视界准备一份数据（prepare_data 依赖 hz → yte_raw 窗口不同）
        data = prepare_data(cfg, shared['full_df'], shared['wd'], shared['wa'],
                            shared['meta'],
                            shared['train_idx'], shared['val_idx'],
                            shared['test_idx'],
                            shared['water_feat'],
                            shared['geo_feat_raw'], hz,
                            disp_raw=shared['disp_raw'])
        shared['data'] = data

        all_fold_sec = {}

        for holdout_section in cfg.SECTION_NAMES:
            logger.info(f"\n--- Fold: 留出 [{holdout_section}] ---")
            node_mask = masks[holdout_section]

            fold_global, fold_section, fold_sec = run_one_fold(
                cfg, holdout_section, node_mask, shared, hz)

            all_fold_sec[holdout_section] = fold_sec

        # ---- Step 4: 该视界的跨折汇总 ----
        hz_out = cfg.OUT_ROOT / f"H{hz}"
        hz_out.mkdir(parents=True, exist_ok=True)

        df_summary = compute_loocv_summary(all_fold_sec, cfg, hz)
        df_summary.to_csv(hz_out / 'loocv_summary.csv', index=False)
        df_summary.to_excel(hz_out / 'loocv_summary.xlsx', index=False)

        all_loocv_summaries[hz] = df_summary

    # ---- Step 5: 跨视界汇总 ----
    if len(all_loocv_summaries) >= 2:
        logger.info(f"\n{'#' * 60}")
        logger.info("#  CROSS-HORIZON LOOCV SUMMARY")
        logger.info(f"{'#' * 60}")

        cross_rows = []
        for hz, df_sum in all_loocv_summaries.items():
            mean_row = df_sum[df_sum['Fold'] == 'Mean']
            if len(mean_row) == 0:
                continue
            r = mean_row.iloc[0].to_dict()
            r['Horizon'] = hz
            cross_rows.append(r)

        if cross_rows:
            df_cross = pd.DataFrame(cross_rows)
            df_cross = df_cross.sort_values('Horizon')
            df_cross.to_csv(cfg.OUT_ROOT / 'cross_horizon_loocv.csv', index=False)
            df_cross.to_excel(cfg.OUT_ROOT / 'cross_horizon_loocv.xlsx', index=False)

            logger.info("\n跨视界 LOOCV 汇总:")
            for _, row in df_cross.iterrows():
                logger.info(f"  H={int(row['Horizon']):2d}  "
                         f"gap_RMSE_last={row['Gap_RMSE_last_pct']:+.2f}%  "
                         f"ΔR²_last={row['Diff_R2_last']:+.4f}")

    logger.info(f"\n{'=' * 60}")
    logger.info(f"ALL DONE! 结果保存至: {cfg.OUT_ROOT}")
    logger.info(f"{'=' * 60}")


if __name__ == '__main__':
    main()