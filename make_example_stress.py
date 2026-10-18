# -*- coding: utf-8 -*-
"""make_example_stress.py — 用 Stress_Test_Experiment 在合成数据上生成压力条件示例产物。

产物：AUX_ROOT/Stress_Results/H{hz}/{condition}/...
规模：全部压力条件 × 5 视界 × 2 种子 × 5 epoch（演示链路，不与论文表对照）。

运行（PowerShell，在仓库根目录）：
  & "<PYENV_ROOT>\python.exe" -u make_example_stress.py
"""
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

# 示例产物可随时重建：允许覆盖上一次的示例结果
os.environ.setdefault("STRESS_FORCE_NEW", "1")

import Stress_Test_Experiment as SE

if __name__ == "__main__":
    SE.Config.HORIZONS = [2, 6, 8, 10, 12]
    SE.Config.SEEDS = SE.Config.SEEDS[:2]
    SE.Config.EPOCHS = 5
    SE.Config.PATIENCE = 3
    SE.main(quick_test=False)
    print("完成。->", SE.Config().OUT_ROOT)
