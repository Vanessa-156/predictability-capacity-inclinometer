# -*- coding: utf-8 -*-
r"""_p03_config_capacity.py —— 分离"配置侧"与"可达分量可学性侧"。

本脚本做什么
------------
**固定输出表示容量 K = 13，只变测量配置（深度布点密度）**，逐配置重估
floor（与模型无关的配置侧下界）与 reach（模型可达分量，反映可学性）：

    keep_every = 1 / 2 / 3 / 5   →   N = 155 / 80 / 55 / 35 个节点
    （15.5 m 内对应 0.5 / 1.0 / 1.5 / 2.5 m，与 §4.3 压力实验的稀疏档一一对应）

判读（脚本末尾自动打印）：
  · floor 随配置变、reach 不随配置变  ⇒ 约束在**配置侧**
  · floor 基本不变、reach 随配置变    ⇒ 约束主要在学习侧
  · 逐配置 skill 上限 1 - floor²/zero² 与实际 skill 的差距 = **可学性缺口**

为什么必须固定 K（本实验的关键设计）
----------------------------------
`AB.prepare_data` 用 `PCA(0.90)` 方差阈值选 K，**K 会随配置自动变化**，于是
"配置变 → floor 变"里混着"K 变"（K 同样决定 floor），归因不干净。
故本脚本把 `AB.PCA` 打补丁为固定 `n_components=13`；同时另用**未打补丁**的原始
PCA 记录各配置"90% 方差所需 K"，作为该混淆真实存在的留档证据（写入汇总表）。

铁律（沿用项目约定）
------------------
· **不覆盖旧结果**：预测/指标全部写入新建目录
  `<WORK_ROOT>\Stage_Capacity_Config\`，汇总表写入
  `_paper_tables\T_p03_config_capacity.csv`（逐种子）与
  `_paper_tables\T_p03_config_summary.csv`（逐配置）。
· **不改训练配方**：损失 / 优化器 / 调度 / 早停 / 数据划分 / LOOKBACK 全部沿用
  锁定脚本 AB，只改两处：(i) 输出表示容量固定为 13；(ii) 深度节点子集。
· **锚点自检**：keep_every=1、K=13 下 Ours(V7) H=10 末步 RMSE 应**逐种子**复现
  权威表 `Ablation_New_5d_mid\H10\V7_wo_VerticalGCN\summary\global_30runs.csv`
  （容差 1e-6），零变化 = 1.372010（打补丁在 K=13 处应为恒等变换）。
  注意不要用"本批 n 种子均值"对标"权威 30 种子均值"：种子 42/15/77 本身偏高
  （1.8154/1.7005/1.6057），均值不可比。
· **恒等式自检**：逐种子校验 RMSE² = reach² + floor²（相对残差应 < 1e-12）。

运行（PowerShell）
-----------------
  $py = "python"

  & $py -u _p03_config_capacity.py --smoke     # 冒烟：1 配置 / 2 种子 / 5 epoch
  & $py -u _p03_config_capacity.py --probe     # 锚点：1 配置 / 3 种子（全 epoch）
  & $py -u _p03_config_capacity.py             # 正式：4 配置 × 30 种子 × 1 模型
  & $py -u _p03_config_capacity.py --models ours,v0    # 加跑 TCN-Only 对照

  续跑（断点续跑，跳过已完成的 run_<seed>）：$env:P03_RESUME="1"
  只跑部分配置： --configs 1,3     指定种子： --seeds 42,15,77

产物规模
--------
正式一次 = 4 配置 × 30 种子 × 1 模型 = 120 次训练（H=10，EPOCHS=400、早停 40）。
"""
from __future__ import annotations

from _env import AUX_ROOT, WORK_ROOT

import argparse
import copy
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy import stats
from torch.utils.data import DataLoader, TensorDataset

PROJ = WORK_ROOT
sys.path.insert(0, str(PROJ))

import Ablation_Experiment as AB          # noqa: E402

# -----------------------------------------------------------------------------
# 常量
# -----------------------------------------------------------------------------
HZ = 10                                   # 主视界（与主表一致）
K_FIXED = 13                              # 固定输出表示容量（本实验的自变量控制点）
OUT_ROOT = PROJ / "Stage_Capacity_Config"
TABLES = PROJ / "_paper_tables"
TABLES.mkdir(parents=True, exist_ok=True)

ANCHOR_MEAN, ANCHOR_STD = 1.5687, 0.0940  # Ours(V7) H=10 末步 RMSE 的 30 种子均值
ZERO_H10 = 1.372010                       # 零变化基线锚点
ANCHOR_TOL = 0.010                        # 均值容差（mm，仅在种子数 >= 15 时使用）
PER_SEED_TOL = 1e-6                       # 逐种子复现容差（应与锁定脚本逐位一致）
# 权威逐种子表（H=10 · V7_wo_VerticalGCN · 30 种子）
def _resolve_summary_csv(d):
    """取该 summary 目录下种子数最多的 global_*runs.csv（真实=30，示例可能更少）。"""
    cand = sorted(Path(d).glob("global_*runs.csv"))
    if not cand:
        return Path(d) / "global_30runs.csv"
    return max(cand, key=lambda p: len(p.read_text(encoding="utf-8-sig").splitlines()))


AUTH_TABLE = _resolve_summary_csv(
    AUX_ROOT / "Ablation_New_5d_mid/H10/V7_wo_VerticalGCN/summary")
SPARSE_LEVELS = [1, 2, 3, 5]
VARIANT_OF = {"ours": "V7_wo_VerticalGCN", "v0": "V0_Pure_TCN"}
MODEL_LABEL = {"ours": "Ours(V7)", "v0": "TCN-Only(V0)"}
VAR_LABEL = {v: MODEL_LABEL[k] for k, v in VARIANT_OF.items()}   # 变体名 → 展示名

RESUME = os.environ.get("P03_RESUME", "0") == "1"

# -----------------------------------------------------------------------------
# 固定 K 补丁（必须在任何 AB.prepare_data 调用之前生效）
# -----------------------------------------------------------------------------
_ORIG_PCA = AB.PCA                        # 原始 sklearn PCA 类（记录 90% 方差所需 K）


def _fixed_pca(n_components=None, **kw):
    """把 `PCA(0.90)` 这类方差阈值写法替换为固定成分数 K_FIXED。

    返回**真正的** sklearn PCA 实例（非子类），以避开 sklearn 对 __init__ 签名的校验。
    在 K=13 处本补丁为恒等变换（数据本身 90% 方差即落在 13 个成分上），故 keep=1
    的锚点应逐位复现锁定脚本。
    """
    if isinstance(n_components, float) and 0.0 < n_components < 1.0:
        n_components = K_FIXED
    return _ORIG_PCA(n_components=n_components, **kw)


# -----------------------------------------------------------------------------
# 数据缓存 / 划分 / 稀疏化
# -----------------------------------------------------------------------------
def load_cache():
    """与锁定脚本一致的共享数据（AB 支路）。"""
    cfg = AB.Config()
    AB.set_seed(42)
    full_df = AB.load_raw_data_with_meta(cfg)
    ts = np.sort(full_df["测量时间"].unique())
    wd, wa = AB.load_water_data(ts, cfg)
    disp_raw, _, meta = AB.build_node_features(full_df, cfg)
    geo = AB.load_geology_params(cfg)
    geo_feat_raw = AB.assign_geo_to_nodes(geo, meta["depths"], cfg) if geo else None
    return dict(cfg=cfg, full_df=full_df, ts=ts, wd=wd, wa=wa,
                water_feat=AB.extract_water_features(wd, wa),
                disp_raw=disp_raw, meta=meta, geo_feat_raw=geo_feat_raw)


def make_split(n):
    """锁定划分：train [0,.70n) / val [.70n,.85n) / test [.85n,n)，必须是连续区间。

    `AB.prepare_data` 先按 `disp_raw[idx]` 重排、随后仍用**原索引**索回差分
    （`diff1_den[train_idx]`），故三段必须是原时间轴上的连续区间，否则目标错位。
    """
    t0, t1 = int(n * AB.Config.TRAIN_RATIO), int(n * (AB.Config.TRAIN_RATIO
                                                       + AB.Config.VAL_RATIO))
    return np.arange(0, t0), np.arange(t0, t1), np.arange(t1, n)


def sparsify(cache, keep_every):
    """按 (point-major, depth-minor) 顺序每隔 keep_every 个深度保留一个节点。

    节点序与 `AB.build_node_features` 一致：index = p * n_dep + d。
    返回 (cfg_sp, meta_sp, disp_sp, geo_sp)：cfg_sp.DEPTHS 同步缩短，
    以便 `AB.build_adjacency / build_vertical_adjacency` 中的 `len(cfg.DEPTHS)`
    与同孔相邻深度先验仍然正确。
    """
    cfg0, meta, disp = cache["cfg"], cache["meta"], cache["disp_raw"]
    n_dep, n_pts = len(cfg0.DEPTHS), len(cfg0.POINTS)
    keep_d = list(range(0, n_dep, keep_every))
    keep_idx = np.array([p * n_dep + d for p in range(n_pts) for d in keep_d])

    cfg = copy.deepcopy(cfg0)
    cfg.DEPTHS = np.asarray(cfg0.DEPTHS, dtype=float)[keep_d]
    meta_sp = {
        "node_names": [meta["node_names"][i] for i in keep_idx],
        "point_names": [meta["point_names"][i] for i in keep_idx],
        "depths": np.asarray(meta["depths"])[keep_idx],
        "coords": np.asarray(meta["coords"])[keep_idx],
        "timestamps": meta["timestamps"],
        "valid_mask": np.asarray(meta["valid_mask"])[keep_idx],
    }
    disp_sp = disp[:, keep_idx]
    geo = cache["geo_feat_raw"]
    geo_sp = geo[keep_idx] if geo is not None else None
    return cfg, meta_sp, disp_sp, geo_sp


def k_var90(cfg_sp, disp_sp, tr, va, te, hz):
    """该配置下 `PCA(0.90)` 会选出的 K（**未打补丁**的原始 PCA）。

    用途：留档证明"不固定 K 则 K 随配置漂移"这一混淆确实存在。
    """
    tr_d, va_d, te_d = AB.denoise_split(disp_sp[tr], disp_sp[va], disp_sp[te], cfg_sp)
    all_den = np.vstack([tr_d, va_d, te_d])
    d1 = np.zeros_like(all_den)
    if all_den.shape[0] > 1:
        d1[1:] = np.diff(all_den, axis=0)
    return int(_ORIG_PCA(0.90).fit(d1[tr]).n_components_)


# -----------------------------------------------------------------------------
# 单次训练 + 评估（地板—可达分解）
# -----------------------------------------------------------------------------
def run_seed(cfg_sp, variant, keep_every, cache, meta_sp, disp_sp, geo_sp, tr, va, te,
             hz, seed, out_dir, epochs=None):
    """训练一个 (配置 × 模型 × 种子)，返回逐种子指标字典。"""
    cfg = copy.deepcopy(cfg_sp)
    for k, v in AB.ABLATION_VARIANTS[variant].items():
        setattr(cfg, k, v)
    if epochs is not None:
        cfg.EPOCHS = int(epochs)

    n_nodes = len(meta_sp["depths"])
    run_dir = out_dir / f"run_{seed}"
    run_dir.mkdir(parents=True, exist_ok=True)
    mfile = run_dir / "metrics.json"
    if RESUME and mfile.exists():
        print(f"    [skip] {out_dir.name}/run_{seed}（已存在）")
        return json.load(open(mfile, encoding="utf-8"))

    geo_fb = geo_sp if geo_sp is not None else np.zeros(
        (n_nodes, cfg.GEO_FEAT_DIM), dtype=np.float32)
    geo_use = geo_fb if np.any(geo_fb) else None

    # 训练/验证/测试特征与目标（K 已被补丁固定为 13）
    data = AB.prepare_data(cfg, cache["full_df"], cache["wd"], cache["wa"], meta_sp,
                           tr, va, te, cache["water_feat"], geo_fb, hz,
                           disp_raw=disp_sp)
    N, F_in = data["N"], data["F_in"]

    # 邻接：cfg.DEPTHS 已同步缩短 ⇒ 同孔先验/剪枝/地质核与稀疏节点集一致
    adj = AB.build_adjacency(meta_sp, geo_use, cfg,
                             zero_same_hole=getattr(cfg, "ZERO_SAME_HOLE", False))
    vert = AB.build_vertical_adjacency(meta_sp, cfg)
    adj_t = torch.FloatTensor(adj).to(cfg.DEVICE)
    vert_t = torch.FloatTensor(vert).to(cfg.DEVICE)
    geo_t = torch.FloatTensor(data["geo_static"]).to(cfg.DEVICE)
    mask_t = torch.FloatTensor(data["valid_mask"]).to(cfg.DEVICE)

    def ld(X, y, wf, sh):
        return DataLoader(TensorDataset(
            torch.FloatTensor(X), torch.FloatTensor(y), torch.FloatTensor(wf),
            geo_t.unsqueeze(0).expand(X.shape[0], N, -1)),
            batch_size=cfg.BATCH_SIZE, shuffle=sh)

    ld_tr = ld(data["Xtr"], data["ytr"], data["wf_train"], True)
    ld_va = ld(data["Xva"], data["yva"], data["wf_val"], False)
    ld_te = ld(data["Xte"], data["yte"], data["wf_test"], False)

    AB.set_seed(seed)
    model = AB.AblationModel(cfg, N=N, in_dim=F_in, gcn_hid=cfg.GCN_HIDDEN,
                             tcn_ch=cfg.TCN_CHANNELS, k=cfg.TCN_KERNEL_SIZE,
                             dil=cfg.TCN_DILATIONS, attn_hid=cfg.ATTENTION_HIDDEN,
                             hor=hz, adj=adj_t, out_dim=data["out_dim"],
                             geo_feat_dim=data["geo_dim_actual"],
                             vert_adj=vert_t).to(cfg.DEVICE)
    # 复现锁定脚本 `AB.run_single_seed` 的 RNG 消耗：FLOPs 估计会跑一次 train 态
    # forward（含 dropout）并消费全局随机数；不调用则训练轨迹与锚点不一致。
    AB.count_params(model)
    AB.estimate_flops(model, cfg, N, F_in, hz)

    model, _, _, _ = AB.train_model(cfg, model, ld_tr, ld_va,
                                    run_dir / "best.pth", node_mask=mask_t)
    pred, true = AB.evaluate(model, ld_te, data["init_test"], data["scY"],
                             data["pca_y"], hz, cfg.DEVICE, node_mask=mask_t,
                             true_diff_win=data["yte_raw"])
    gm = AB.global_metrics(true, pred)
    fl = AB.output_space_floor(data["yte_raw"], data["init_test"], data["scY"],
                               data["pca_y"], hz, ns=pred.shape[0])

    rmse = float(gm["RMSE_last"])
    floor = float(fl["RMSE_floor_last"])
    reach = float(np.sqrt(max(rmse ** 2 - floor ** 2, 0.0)))
    zero = float(np.sqrt(np.mean((data["init_test"] - true[:, -1, :]) ** 2)))
    resid = abs(rmse ** 2 - reach ** 2 - floor ** 2) / max(rmse ** 2, 1e-12)

    rec = dict(variant=variant, model=VAR_LABEL.get(variant, variant),
               keep_every=int(keep_every), n_nodes=int(n_nodes),
               K=int(data["out_dim"]), seed=int(seed),
               n_win=int(pred.shape[0]), F_in=int(F_in),
               RMSE_last=rmse, R2_last=float(gm["R2_last"]),
               RMSE_floor_last=floor, RMSE_reach_last=reach,
               RMSE_zero_last=zero,
               skill=1.0 - (rmse / zero) ** 2,
               skill_cap=1.0 - (floor / zero) ** 2,
               floor_share=floor ** 2 / rmse ** 2,
               identity_resid=resid)
    json.dump(rec, open(mfile, "w", encoding="utf-8"), indent=2)
    np.savez(run_dir / "predictions.npz", pred_cum=pred, true_cum=true)
    try:
        (run_dir / "best.pth").unlink()       # 只留 summary，不落中间权重
    except OSError:
        pass
    return rec


# -----------------------------------------------------------------------------
# 汇总与判读
# -----------------------------------------------------------------------------
def summarize(df, k90_map):
    rows = []
    for (v, ke), g in df.groupby(["variant", "keep_every"], sort=True):
        # 配对基准 = 最密配置（keep_every 最小）；仅同模型内比较
        base = df[(df.variant == v) & (df.keep_every == df.keep_every.min())]
        bmap = base.set_index("seed")["RMSE_last"]
        cur = g.set_index("seed")["RMSE_last"]
        common = bmap.index.intersection(cur.index)
        if ke == df.keep_every.min() or len(common) < 2:
            pct, pval = 0.0, np.nan
        else:
            pct = float(cur[common].mean() / bmap[common].mean() - 1.0) * 100.0
            pval = float(stats.ttest_rel(bmap[common], cur[common]).pvalue)
        rows.append(dict(
            variant=v, model=VAR_LABEL.get(v, v), keep_every=int(ke),
            N_nodes=int(g.n_nodes.iloc[0]), K_fixed=K_FIXED,
            K_var90=int(k90_map.get((v, ke), -1)), n_seeds=int(len(g)),
            RMSE_last_mean=float(g.RMSE_last.mean()),
            RMSE_last_std=float(g.RMSE_last.std(ddof=1)),
            RMSE_floor_mean=float(g.RMSE_floor_last.mean()),
            RMSE_reach_mean=float(g.RMSE_reach_last.mean()),
            RMSE_zero_mean=float(g.RMSE_zero_last.mean()),
            skill_mean=float(g.skill.mean()),
            skill_cap_mean=float(g.skill_cap.mean()),
            learn_gap_mean=float((g.skill_cap - g.skill).mean()),
            floor_share_mean=float(g.floor_share.mean()),
            identity_resid_max=float(g.identity_resid.max()),
            RMSE_pct_vs_base=pct, p_paired_vs_base=pval))
    return pd.DataFrame(rows)


def print_verdict(sm):
    print("\n" + "=" * 100)
    print("判读（固定 K=13，仅变深度布点配置）")
    print("=" * 100)
    for v, g in sm.groupby("variant", sort=True):
        g = g.sort_values("keep_every")
        print(f"\n【{VAR_LABEL.get(v, v)}】")
        print(f"{'稀疏档':>7}{'N':>6}{'K(0.9)':>8}{'RMSE_last':>16}"
              f"{'floor':>9}{'reach':>9}{'skill':>9}{'上限':>9}{'缺口':>9}{'Δ%':>8}{'p':>8}")
        for _, r in g.iterrows():
            pstr = "  —  " if not np.isfinite(r.p_paired_vs_base) else f"{r.p_paired_vs_base:.3f}"
            print(f"{int(r.keep_every):>7}{int(r.N_nodes):>6}{int(r.K_var90):>8}"
                  f"{r.RMSE_last_mean:>10.4f}±{r.RMSE_last_std:.4f}"
                  f"{r.RMSE_floor_mean:>9.4f}{r.RMSE_reach_mean:>9.4f}"
                  f"{r.skill_mean:>9.4f}{r.skill_cap_mean:>9.4f}"
                  f"{r.learn_gap_mean:>9.4f}{r.RMSE_pct_vs_base:>+8.2f}{pstr:>8}")
        if len(g) >= 2:
            d_floor = g.RMSE_floor_mean.iloc[-1] - g.RMSE_floor_mean.iloc[0]
            d_reach = g.RMSE_reach_mean.iloc[-1] - g.RMSE_reach_mean.iloc[0]
            rel_f = d_floor / max(g.RMSE_floor_mean.iloc[0], 1e-12)
            rel_r = d_reach / max(g.RMSE_reach_mean.iloc[0], 1e-12)
            print(f"  从最密到最稀：floor {rel_f:+.2%}、reach {rel_r:+.2%}"
                  f"  => 约束{'偏向配置侧' if abs(rel_f) > abs(rel_r) else '偏向可达可学性侧'}")
            print(f"  K(0.9) 随配置取值：{list(g.K_var90)}"
                  f"  => {'K 会随配置漂移，固定 K 是必要控制' if g.K_var90.nunique() > 1 else 'K 在本场景不漂移'}")
        print(f"  恒等式 RMSE²=reach²+floor² 最大相对残差 = {g.identity_resid_max.max():.2e}")


def check_anchor(df):
    """锚点自检：keep_every=1 应**逐种子**复现锁定脚本，零变化应逐位一致。

    注意：不能用"本批 n 种子均值"去比"权威 30 种子均值"——种子 42/15/77 本身
    就是该分布里偏高的几个（1.8154/1.7005/1.6057），3 种子均值 1.7072 与 30 种子
    均值 1.5687 不可比。故此处以**逐种子对照权威表**为准（容差 1e-6）。
    """
    a = df[(df.variant == VARIANT_OF["ours"]) & (df.keep_every == 1)]
    if a.empty:
        return
    print("\n" + "-" * 100)
    print(f"[锚点自检] keep_every=1 · Ours(V7) · H=10 · 本批 n={len(a)} 种子")
    z = a.RMSE_zero_last.mean()
    z_max = (a.RMSE_zero_last - ZERO_H10).abs().max()
    print(f"  零变化 = {z:.6f}（与期望 {ZERO_H10:.6f} 的最大偏差 {z_max:.2e}）"
          f"   {'[OK]' if z_max <= 1e-3 else '[WARN]'}")

    if not AUTH_TABLE.exists():
        print(f"  [WARN] 未找到权威逐种子表：{AUTH_TABLE}")
        if len(a) >= 15:
            ok_m = abs(a.RMSE_last.mean() - ANCHOR_MEAN) <= ANCHOR_TOL
            print(f"         退化为均值对照：本批 {a.RMSE_last.mean():.4f} ± "
                  f"{a.RMSE_last.std(ddof=1):.4f} 对标 {ANCHOR_MEAN} ± {ANCHOR_STD}"
                  f"   {'[OK]' if ok_m else '[WARN]'}")
        else:
            print(f"         种子数 {len(a)} < 15，均值不可比 ⇒ 无法核验锚点")
        print("-" * 100)
        return

    ref = pd.read_csv(AUTH_TABLE).set_index("seed")["RMSE_last"]
    m = a.set_index("seed")["RMSE_last"]
    common = m.index.intersection(ref.index)
    diff = (m[common] - ref[common]).abs()
    print(f"  逐种子对照权威表：{AUTH_TABLE}")
    print(f"    （全表 {len(ref)} 种子，mean={ref.mean():.4f} ± {ref.std(ddof=1):.4f}）")
    for s in common:
        print(f"    seed={s:<5} 本批 {m[s]:.6f}  权威 {ref[s]:.6f}  "
              f"Δ={m[s] - ref[s]:+.2e}")
    ok = (len(common) == len(m)) and float(diff.max()) <= PER_SEED_TOL
    print(f"  匹配 {len(common)}/{len(m)} 种子；最大绝对偏差 = {float(diff.max()):.2e}"
          f"  => {'[OK] 逐位复现锁定脚本，锚点通过' if ok else '[WARN] 存在偏差，本批结论不可用'}")
    print("-" * 100)


# -----------------------------------------------------------------------------
# main
# -----------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hz", type=int, default=HZ)
    ap.add_argument("--configs", type=str, default=",".join(map(str, SPARSE_LEVELS)))
    ap.add_argument("--models", type=str, default="ours",
                    help="ours | v0 | ours,v0")
    ap.add_argument("--seeds", type=str, default=None)
    ap.add_argument("--smoke", action="store_true", help="1 配置 / 2 种子 / 5 epoch")
    ap.add_argument("--probe", action="store_true", help="1 配置 / 3 种子（锚点自检）")
    args = ap.parse_args()

    keep_list = [int(x) for x in args.configs.split(",") if x.strip()]
    model_keys = [m.strip() for m in args.models.split(",") if m.strip()]
    seeds = ([int(s) for s in args.seeds.split(",")] if args.seeds
             else list(AB.Config.SEEDS))
    epochs = None
    if args.smoke:
        keep_list, seeds, epochs = [1], seeds[:2], 5
    if args.probe:
        keep_list, seeds = [1], seeds[:3]
    for m in model_keys:
        if m not in VARIANT_OF:
            raise SystemExit(f"未知模型键 {m}（可选：{list(VARIANT_OF)}）")

    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    print("=" * 100)
    print("固定 K=%d，变深度布点配置（分离配置侧 vs 可达可学性侧）" % K_FIXED)
    print(f"  视界 H={args.hz} | 配置 keep_every={keep_list} | 模型={model_keys} | "
          f"种子数={len(seeds)} | epoch={epochs or AB.Config.EPOCHS}")
    print(f"  输出目录：{OUT_ROOT}（新建，不覆盖旧结果）")
    print("=" * 100)

    try:
        import thop  # noqa: F401
        thop_ok = True
    except ImportError:
        thop_ok = False
    print(f"  thop 可用：{thop_ok}"
          + ("" if thop_ok else
             "  <- 不可用！AB.estimate_flops 会跳过 train 态 forward，"
             "RNG 轨迹与锚点不一致，锚点自检大概率不通过（请先 pip install thop）"))

    cache = load_cache()
    n = len(cache["ts"])
    tr, va, te = make_split(n)
    print(f"  时间戳 n={n}；划分 train/val/test = {len(tr)}/{len(va)}/{len(te)}"
          f"（test 段 [{te[0]},{te[-1]}]）")

    # 关键控制点：把输出表示容量固定为 K_FIXED
    AB.PCA = _fixed_pca
    print(f"  [OK] 已固定 AB.PCA -> n_components={K_FIXED}（方差阈值写法一律映射为 {K_FIXED}）")

    # 各配置的数据驱动 K（未打补丁的原始 PCA），证明混淆真实存在
    k90_map, prep = {}, {}
    print("\n[准备] 各配置的节点数与 90% 方差所需 K（未固定 K 时会随配置漂移的量）")
    print(f"{'keep_every':>12}{'N_nodes':>10}{'K_var90':>10}")
    for keep in keep_list:
        cfg_sp, meta_sp, disp_sp, geo_sp = sparsify(cache, keep)
        k90 = k_var90(cfg_sp, disp_sp, tr, va, te, args.hz)
        n_nodes = len(meta_sp["depths"])
        for m in model_keys:
            k90_map[(VARIANT_OF[m], keep)] = k90
        prep[keep] = (cfg_sp, meta_sp, disp_sp, geo_sp)
        print(f"{keep:>12}{n_nodes:>10}{k90:>10}")

    rows = []
    for m in model_keys:
        variant = VARIANT_OF[m]
        for keep in keep_list:
            cfg_sp, meta_sp, disp_sp, geo_sp = prep[keep]
            out_dir = OUT_ROOT / f"{m}_keep{keep}"
            out_dir.mkdir(parents=True, exist_ok=True)
            print("\n" + "=" * 100)
            print(f"[{MODEL_LABEL.get(m, m)}] keep_every={keep} · "
                  f"N={len(meta_sp['depths'])} · {len(seeds)} 种子 → {out_dir}")
            print("=" * 100)
            for si, seed in enumerate(seeds, 1):
                t0 = time.time()
                rec = run_seed(cfg_sp, variant, keep, cache, meta_sp, disp_sp, geo_sp,
                               tr, va, te, args.hz, seed, out_dir, epochs=epochs)
                rows.append(rec)
                print(f"  [{si:>2}/{len(seeds)}] seed={seed:<5} "
                      f"RMSE_last={rec['RMSE_last']:.4f} floor={rec['RMSE_floor_last']:.4f} "
                      f"reach={rec['RMSE_reach_last']:.4f} skill={rec['skill']:+.4f} "
                      f"({time.time() - t0:.1f}s)")

    df = pd.DataFrame(rows)
    df_out = TABLES / "T_p03_config_capacity.csv"
    df.round(8).to_csv(df_out, index=False, encoding="utf-8-sig")
    sm = summarize(df, k90_map)
    sm_out = TABLES / "T_p03_config_summary.csv"
    sm.round(8).to_csv(sm_out, index=False, encoding="utf-8-sig")

    if len(seeds) >= 3 and not args.smoke:
        print_verdict(sm)
        check_anchor(df)
    print(f"\n[已写出] {df_out}（逐种子）\n[已写出] {sm_out}（逐配置）")


if __name__ == "__main__":
    main()