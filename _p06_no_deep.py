# -*- coding: utf-8 -*-
r"""_p06_no_deep.py —— "含/不含深部节点"两套模型结果。

深部（8–15.5 m，80 节点）承载幅值可观的真实负位移（|min| 中位 4.3 mm、75% 超
2.0 mm 漂移带，见 `_p06_deep_neg.py`）。为回答"深部节点是否影响容量结论"，
本脚本在**固定 K=13、剔除全部深部节点（保留 0.5–7.5 m 共 75 节点）**下重训
Ours(V7)（H=10，30 种子），与全节点权威（1.5687）配对比较。

实现：完全复用 `_p03_config_capacity.py` 的锁定管线副本（prepare_data 节点子集 +
固定 K=13 补丁 + run_seed + 地板—可达分解），仅把节点子集从"隔 N 深度稀疏"改为
"截断深部"。锚点：本脚本 `--anchor` 模式跑 1 个种子的**全节点**对照，须逐位复现
权威表（证明子集机制本身无偏）。

运行（训练类；~30 次，约 10–20 min）
------------------------------------------------
  $env:P06ND_RESUME="1"   # 续跑
  & $py -u _p06_no_deep.py --anchor     # 1 种子全节点锚点自检
  & $py -u _p06_no_deep.py              # 正式：不含深部 30 种子
输出 `Stage_NoDeep\` + `_paper_tables\T_p06_nodeep.csv`；补材 **S27**。
"""
from _env import AUX_ROOT, WORK_ROOT

import copy
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch

PROJ = WORK_ROOT
sys.path.insert(0, str(PROJ))
import Ablation_Experiment as AB            # noqa: E402
import _p03_config_capacity as P03          # noqa: E402

OUT_ROOT = PROJ / "Stage_NoDeep"
TABLES = PROJ / "_paper_tables"
TABLES.mkdir(parents=True, exist_ok=True)
RESUME = os.environ.get("P06ND_RESUME", "0") == "1"
K_FIXED = 13
HZ = 10
SEEDS = list(AB.Config.SEEDS)
def _resolve_summary_csv(d):
    """取该 summary 目录下种子数最多的 global_*runs.csv（真实=30，示例可能更少）。"""
    cand = sorted(Path(d).glob("global_*runs.csv"))
    if not cand:
        return Path(d) / "global_30runs.csv"
    return max(cand, key=lambda p: len(p.read_text(encoding="utf-8-sig").splitlines()))


AUTH = _resolve_summary_csv(
    AUX_ROOT / "Ablation_New_5d_mid/H10/V7_wo_VerticalGCN/summary")
def sparsify_deep(cache, keep_shallow_m=8.0):
    """保留深度 < keep_shallow_m 的节点（剔除深部）。索引 = p*n_dep + d，d 升序。"""
    cfg0, meta, disp = cache["cfg"], cache["meta"], cache["disp_raw"]
    n_dep, n_pts = len(cfg0.DEPTHS), len(cfg0.POINTS)
    keep_d = [d for d in range(n_dep) if cfg0.DEPTHS[d] < keep_shallow_m]
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
    geo = cache["geo_feat_raw"]
    return cfg, meta_sp, disp[:, keep_idx], (geo[keep_idx] if geo is not None else None)


def patch_fixed_k():
    P03._ORIG_PCA = AB.PCA
    AB.PCA = P03._fixed_pca          # 已有固定 K 补丁（_p03 L114-123）


def run_seed(cfg_sp, cache, meta_sp, disp_sp, geo_sp, tr, va, te, seed, out_dir):
    return P03.run_seed(cfg_sp, "V7_wo_VerticalGCN", 1, cache, meta_sp, disp_sp,
                        geo_sp, tr, va, te, HZ, seed, out_dir)


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--anchor", action="store_true", help="1 种子全节点锚点自检")
    ap.add_argument("--seeds", type=int, default=30)
    args = ap.parse_args()

    patch_fixed_k()
    cache = P03.load_cache()
    n = len(cache["ts"])
    tr, va, te = P03.make_split(n)
    out_root = OUT_ROOT / ("_anchor" if args.anchor else "")
    out_root.mkdir(parents=True, exist_ok=True)

    if args.anchor:
        # 全节点 = keep_d=range(31)，即 P03 的 keep_every=1 —— 必须逐位复现权威
        keep_d = list(range(len(cache["cfg"].DEPTHS)))
        keep_idx = np.array([p * len(cache["cfg"].DEPTHS) + d
                             for p in range(len(cache["cfg"].POINTS)) for d in keep_d])
        cfg0, meta, disp = cache["cfg"], cache["meta"], cache["disp_raw"]
        cfg = copy.deepcopy(cfg0)
        meta_sp = {k: (np.asarray(v)[keep_idx] if isinstance(v, np.ndarray)
                       and v.shape[0] == len(meta["depths"]) else v)
                   for k, v in meta.items()}
        disp_sp = disp[:, keep_idx]
        geo = cache["geo_feat_raw"]
        geo_sp = geo[keep_idx] if geo is not None else None
        rec = run_seed(cfg, cache, meta_sp, disp_sp, geo_sp, tr, va, te,
                       SEEDS[0], out_root)
        ref = {r["seed"]: r["RMSE_last"] for _, r in
               __import__("pandas").read_csv(AUTH).iterrows()}
        diff = abs(rec["RMSE_last"] - ref[SEEDS[0]])
        print(f"[锚点] seed={SEEDS[0]} 本批 {rec['RMSE_last']:.6f} vs 权威 "
              f"{ref[SEEDS[0]]:.6f}  Δ={diff:.2e}  "
              f"{'[OK]' if diff <= 1e-6 else '[WARN]'}")
        return

    cfg_sp, meta_sp, disp_sp, geo_sp = sparsify_deep(cache)
    print(f"[准备] 不含深部节点：N = {len(meta_sp['depths'])}"
          f"（保留 {cfg_sp.DEPTHS.min()}-{cfg_sp.DEPTHS.max()} m）| K 固定 {K_FIXED}"
          f"| 视界 {HZ} | 种子 {args.seeds}")
    rows = []
    for seed in SEEDS[:args.seeds]:
        rec = run_seed(cfg_sp, cache, meta_sp, disp_sp, geo_sp, tr, va, te,
                       seed, out_root)
        rows.append(rec)
        print(f"  seed={seed:<5} RMSE_last={rec['RMSE_last']:.4f} "
              f"skill={rec['skill']:.4f} K={rec['K']}")
    import pandas as pd
    df = pd.DataFrame(rows)
    df.to_csv(TABLES / "T_p06_nodeep.csv", index=False, encoding="utf-8-sig")
    ours = df["RMSE_last"].mean()
    print(f"\n不含深部 Ours(V7)：RMSE_last = {ours:.4f} ± {df['RMSE_last'].std():.4f}"
          f" | 权威全节点 = 1.5687 ± 0.0940"
          f" | Δ = {(ours-1.5687)/1.5687*100:+.2f}%"
          f" | skill = {df['skill'].mean():+.4f}"
          f" | 零变化 = {df['RMSE_zero_last'].iloc[0]:.4f}")
    print(f"\n[已写出] {TABLES / 'T_p06_nodeep.csv'}（不含深部 30 种子）")


if __name__ == "__main__":
    main()
