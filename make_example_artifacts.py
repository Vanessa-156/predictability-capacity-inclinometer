# -*- coding: utf-8 -*-
"""make_example_artifacts.py — 用发布的管线在合成数据上"真跑"出示例中间产物。

为什么这样做：审稿人要求 "Archived code runs with the archived data"。与其手写假的
predictions.npz，本脚本直接调用 `Ablation_Experiment.run_ablation_variant`，在
`make_example_data.py` 生成的合成数据上跑一遍**缩小规模**的训练（默认 2 个种子、
8 个 epoch），产出与真实实验完全同 schema 的
`Ablation_New_5d/H{hz}/{variant}/run_{seed}/{predictions.npz,metrics.json,...}`
以及 `summary/global_*runs.csv`，供其余 20 余个零训练分析脚本读取。

规模说明：论文结果来自 30 个种子 × 400 epoch 的保密真实数据；此处缩小规模仅为
演示整条分析链路可运行，数值不可与论文表格对照。

运行（PowerShell，在仓库根目录）：
  & "<PYENV_ROOT>\python.exe" -u make_example_artifacts.py
"""
import copy
import os
import shutil
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd

import Ablation_Experiment as AE
from _env import AUX_ROOT

EPOCHS = int(os.environ.get("EXAMPLE_EPOCHS", "8"))
SEEDS = [int(s) for s in os.environ.get("EXAMPLE_SEEDS", "42,15").split(",")]
HORIZONS = [2, 6, 8, 10, 12]
ABL_ROOT = AUX_ROOT / "Ablation_New_5d"


def build_cache(cfg):
    AE.set_seed(42)
    full_df = AE.load_raw_data_with_meta(cfg)
    ts = pd.to_datetime(np.sort(full_df["测量时间"].unique()))
    wd, wa = AE.load_water_data(ts, cfg)
    disp_raw, _, meta = AE.build_node_features(full_df, cfg)
    geo = AE.load_geology_params(cfg)
    geo_feat_raw = AE.assign_geo_to_nodes(geo, meta["depths"], cfg) if geo else None
    n = len(ts)
    tr = int(n * cfg.TRAIN_RATIO)
    va = int(n * (cfg.TRAIN_RATIO + cfg.VAL_RATIO))
    return dict(full_df=full_df, ts=ts, wd=wd, wa=wa,
                water_feat=AE.extract_water_features(wd, wa), meta=meta,
                geo_feat_raw=geo_feat_raw, disp_raw=disp_raw,
                train_idx=np.arange(tr), val_idx=np.arange(tr, va),
                test_idx=np.arange(va, n))


def main():
    base = AE.Config()
    base.EPOCHS = EPOCHS
    base.PATIENCE = max(2, EPOCHS // 2)
    base.SEEDS = SEEDS
    cache = build_cache(base)
    print("[cache] disp=%s  test_idx=%d" % (cache["disp_raw"].shape, len(cache["test_idx"])))

    t_all = time.time()
    for hz in HORIZONS:
        cfg = copy.deepcopy(base)
        cfg.OUT_ROOT = ABL_ROOT / f"H{hz}"
        for i, (name, sw) in enumerate(AE.ABLATION_VARIANTS.items(), 1):
            t0 = time.time()
            AE.run_ablation_variant(name, sw, cfg, hz, shared_data_cache=cache)
            print("  H%-2d %-2d/%-2d %-22s %.1fs"
                  % (hz, i, len(AE.ABLATION_VARIANTS), name, time.time() - t0), flush=True)
    print("[ablation] %.1f min" % ((time.time() - t_all) / 60))

    # 论文另有 Ablation_New_5d_mid（H6/H10 的权威表）：从同批结果复制
    mid = AUX_ROOT / "Ablation_New_5d_mid"
    if mid.exists():
        shutil.rmtree(mid)
    for hz in (6, 10):
        shutil.copytree(ABL_ROOT / f"H{hz}", mid / f"H{hz}")
    print("[mid] Ablation_New_5d_mid <- H6/H10")

    # 归档瘦身：零训练分析脚本只读 predictions.npz / metrics.json / summary，
    # 不需要逐轮 loss。但 _p07 需重载 V7 权重、_p06_selection_check 需重载 V0/V7 权重，
    # 故仅保留这两个变体的 best.pth。
    KEEP_WEIGHTS = ("V7_wo_VerticalGCN", "V0_Pure_TCN")
    n_del = 0
    for pat in ("best.pth", "losses.npy"):
        for f in AUX_ROOT.rglob(pat):
            if f.name == "best.pth" and any(k in str(f) for k in KEEP_WEIGHTS):
                continue
            f.unlink()
            n_del += 1
    print("[prune] 删除 %d 个权重/损失文件（保留 V0/V7 的 best.pth）" % n_del)

    print("完成。AUX_ROOT=%s" % AUX_ROOT)


if __name__ == "__main__":
    main()
