# -*- coding: utf-8 -*-
"""make_example_comparison.py — 用 Comparison_Experiment 在合成数据上生成外部基线示例产物。

产物：AUX_ROOT/Comparison_Results_w64/H{hz}/{model}/run_{seed}/...
规模：8 个模型 × 5 视界 × 2 种子 × 5 epoch（演示链路，不与论文表对照）。

运行（PowerShell，在仓库根目录）：
  & "<PYENV_ROOT>\python.exe" -u make_example_comparison.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import Comparison_Experiment as CE

if __name__ == "__main__":
    CE.Config.HORIZONS = [2, 6, 8, 10, 12]
    CE.Config.SEEDS = CE.Config.SEEDS[:2]
    CE.Config.EPOCHS = 5
    CE.Config.PATIENCE = 3
    CE.main()
    print("完成。->", CE.Config().OUT_ROOT)
