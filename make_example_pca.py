# -*- coding: utf-8 -*-
"""make_example_pca.py — 用 PCA_Experiment 在合成数据上生成输出表示容量示例产物。

产物：AUX_ROOT/PCA_Results_V7/H{hz}/{K候选}/run_{seed}/...
规模：全部 K 候选 × 5 视界 × 2 种子 × 5 epoch（演示链路，不与论文表对照）。

运行（PowerShell，在仓库根目录）：
  & "<PYENV_ROOT>\python.exe" -u make_example_pca.py
"""
import os
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import PCA_Experiment as PE

if __name__ == "__main__":
    PE.Config.HORIZONS = [2, 6, 8, 10, 12]
    PE.Config.SEEDS = PE.Config.SEEDS[:2]
    PE.Config.EPOCHS = 5
    PE.Config.PATIENCE = 3
    # 干净重建：PCA_Experiment 以 summary/global_*runs.csv 判定"已完成"，
    # 若上一轮只删了权重，会整批跳过、留下缺 best.pth 的目录 ⇒ 先清空再跑。
    out = PE.Config().OUT_ROOT
    if out.exists():
        print("[clean] 删除旧示例产物 %s" % out)
        shutil.rmtree(out)
    PE.main(quick_test=False)
    print("完成。->", PE.Config().OUT_ROOT)
