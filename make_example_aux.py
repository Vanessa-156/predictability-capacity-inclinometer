# -*- coding: utf-8 -*-
"""make_example_aux.py — 生成跨场地/跨场景辅助输入与两侧查表的示例版本。

生成内容（写入 WORK_ROOT）：
  Tan_Digitized/fig19_digitized.csv                 —— 由 Tan et al. (2018) 曲线数字化而来，CC BY
  Tan_Digitized/_tan_predictability_probe.py        —— 跨场地探针模块（随代码发布）
  Liugou_Processed/inclinometer_mm_full.csv         —— 柳沟边坡样例（结构一致、数值虚构）
  _vicd_swapped/vicd_swapped_records.csv            —— 掉转目标记录（结构一致、数值虚构）
  _paper_tables/T_p02_walkforward.csv               —— 滚动起点汇总（结构一致、数值虚构）

Tan 曲线为 CC BY 授权，随代码分发；柳沟为公开数据集（doi:10.12072/ncdc.nieer.db2828.2024），
此处仅放结构一致的虚构样例，正式数据请按 README 从数据中心下载。

运行（PowerShell，在仓库根目录）：
  & "<PYENV_ROOT>\python.exe" -u make_example_aux.py
"""
import shutil
from pathlib import Path

import numpy as np
import pandas as pd

from _env import WORK_ROOT

RNG = np.random.default_rng(20261017)
SRC_TAN = Path(__file__).resolve().parent / "_data_src" / "fig19_digitized.csv"
SRC_PROBE = Path(__file__).resolve().parent / "_data_src" / "_tan_predictability_probe.py"

LIOU_COLS = []
for hole in (229, 230):
    for d in ("0.55m", "1.35m", "3.05m", "4.75m"):
        LIOU_COLS += [f"INC_{hole}_x_d{d}_mm", f"INC_{hole}_y_d{d}_mm"]


def tan():
    d = WORK_ROOT / "Tan_Digitized"
    d.mkdir(parents=True, exist_ok=True)
    shutil.copy2(SRC_TAN, d / "fig19_digitized.csv")
    shutil.copy2(SRC_PROBE, d / "_tan_predictability_probe.py")
    print("  Tan_Digitized: fig19_digitized.csv + _tan_predictability_probe.py")


def liugou():
    d = WORK_ROOT / "Liugou_Processed"
    d.mkdir(parents=True, exist_ok=True)
    n = 816
    t = pd.date_range("2019-08-17 16:30", periods=n, freq="6h")
    data = {"time": t}
    for c in LIOU_COLS:
        base = RNG.normal(0, 0.05, n).cumsum()
        data[c] = np.round(base, 4)
    pd.DataFrame(data).to_csv(d / "inclinometer_mm_full.csv", index=False)
    print("  Liugou_Processed: inclinometer_mm_full.csv (816 x 17)")


def swapped():
    d = WORK_ROOT / "_vicd_swapped"
    d.mkdir(parents=True, exist_ok=True)
    rows = []
    for target in ("settlement", "anchor"):
        for inp in ("disp", "water"):
            for _ in range(40):
                layer1 = RNG.choice(["可用", "越界→需再检验", "不可用"], p=[0.3, 0.25, 0.45])
                rows.append(dict(target=target, input=inp, layer1=layer1))
    pd.DataFrame(rows).to_csv(d / "vicd_swapped_records.csv", index=False)
    print("  _vicd_swapped: vicd_swapped_records.csv")


def walkforward():
    d = WORK_ROOT / "_paper_tables"
    d.mkdir(parents=True, exist_ok=True)
    rows = []
    for fold in range(1, 7):
        for hz in (2, 6, 8, 10, 12):
            for model, base in (("Ours", 0.95), ("V0", 1.05)):
                for _ in range(3):
                    rows.append(dict(fold=fold, H=hz, model=model,
                                     RMSE_last=float(base * (0.9 + 0.2 * RNG.random()))))
    pd.DataFrame(rows).to_csv(d / "T_p02_walkforward.csv", index=False)
    print("  _paper_tables: T_p02_walkforward.csv")


if __name__ == "__main__":
    print("生成示例辅助输入 -> WORK_ROOT=%s" % WORK_ROOT)
    tan()
    liugou()
    swapped()
    walkforward()
    print("完成。")
