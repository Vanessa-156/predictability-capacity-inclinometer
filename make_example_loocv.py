# -*- coding: utf-8 -*-
"""make_example_loocv.py — 用 Spatial_LOOCV 的正式流程生成留一断面示例产物。

产物：WORK_ROOT/Spatial_LOOCV_Results/Fold_{section}/H{hz}/run_{seed}/predictions.npz
规模：5 折 × 5 视界 × 5 种子，但每折仅少量 epoch（演示链路，不与论文表对照）。

运行（PowerShell，在仓库根目录）：
  & "<PYENV_ROOT>\python.exe" -u make_example_loocv.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import Spatial_LOOCV as SL

EPOCHS = 4
SEEDS = [42, 15, 77, 85, 120]
HORIZONS = [2, 6, 8, 10, 12]


def main():
    cfg = SL.SpatialConfig()
    cfg.EPOCHS = EPOCHS
    cfg.PATIENCE = max(2, EPOCHS // 2)
    cfg.SEEDS = SEEDS
    cfg.HORIZONS = HORIZONS
    print("[cfg] epochs=%d seeds=%d horizons=%s" % (EPOCHS, len(SEEDS), HORIZONS))

    shared = SL._build_shared_cache(cfg)
    masks, _ = SL.build_section_masks(cfg)

    for hz in HORIZONS:
        data = SL.prepare_data(cfg, shared["full_df"], shared["wd"], shared["wa"],
                               shared["meta"], shared["train_idx"], shared["val_idx"],
                               shared["test_idx"], shared["water_feat"],
                               shared["geo_feat_raw"], hz, disp_raw=shared["disp_raw"])
        shared["data"] = data
        for sec in cfg.SECTION_NAMES:
            SL.run_one_fold(cfg, sec, masks[sec], shared, hz)
            print("  H%-2d Fold_%s done" % (hz, sec), flush=True)
    print("完成。->", cfg.OUT_ROOT)


if __name__ == "__main__":
    main()
