# -*- coding: utf-8 -*-
"""_p11_swap_test.py -- 掉转目标不对称的正式检验（零训练、只读）。

数据：_vicd_swapped/vicd_swapped_records.csv（_multitarget_vicd.py 输出，
     层 1 max-ratio 判定 T_obs<=T_thr95 -> "不可用"，否则 "越界->需再检验"）。
对每个目标（settlement / anchor）构造 2×2 表：
    行 = 输入（位移 disp / 水位 water）；列 = 层 1 判定（不可用 / 越界）。
用 Fisher 精确检验（双侧）检验"输入类别 × 判定"是否独立；
另对"目标=沉降"给单侧说明（DA-6：'水位→沉降 vs 位移→沉降'）。
用法：python -u _p11_swap_test.py
"""
from _env import WORK_ROOT

import sys
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import stats

ROOT = WORK_ROOT
OUT = ROOT / "_paper_tables"
OUT.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(ROOT / "_vicd_swapped" / "vicd_swapped_records.csv")
df["越界"] = (df["layer1"] == "越界→需再检验").astype(int)
df["不可用"] = (df["layer1"] == "不可用").astype(int)

rows = []
print("=" * 78)
print("掉转目标不对称的正式检验（Fisher 2×2，双侧）")
print("=" * 78)
for target in ["settlement", "anchor"]:
    sub = df[df["target"] == target]
    print(f"\n### 目标 = {target}")
    for inp in ["disp", "water"]:
        g = sub[sub["input"] == inp]
        n_bad = int(g["不可用"].sum())
        n_over = int(g["越界"].sum())
        print(f"  输入={inp:<6} 记录数={len(g):>3}  不可用={n_bad:>3} ({n_bad/len(g):.0%})"
              f"  越界={n_over:>3} ({n_over/len(g):.0%})")

    a = sub[sub["input"] == "disp"]
    b = sub[sub["input"] == "water"]
    # 2×2：[[输入=disp 不可用, 输入=disp 越界],
    #      [输入=water 不可用, 输入=water 越界]]
    table = [[int(a["不可用"].sum()), int(a["越界"].sum())],
             [int(b["不可用"].sum()), int(b["越界"].sum())]]
    odds, p_two = stats.fisher_exact(table, alternative="two-sided")
    _, p_less = stats.fisher_exact(table, alternative="less")   # 位移不可用占比 <= 水位
    _, p_greater = stats.fisher_exact(table, alternative="greater")
    print(f"  2×2 表(不可用,越界): disp={table[0]}  water={table[1]}")
    print(f"  Fisher 双侧 p={p_two:.4f}  OR={odds:.3f}  "
          f"(单侧 p_less={p_less:.4f} / p_greater={p_greater:.4f})")
    rows.append(dict(target=target,
                     n_disp=len(a), n_disp_bad=int(a["不可用"].sum()),
                     n_water=len(b), n_water_bad=int(b["不可用"].sum()),
                     p_fisher_two=p_two, odds_ratio=odds,
                     p_less=p_less, p_greater=p_greater))

res = pd.DataFrame(rows)
res.round(4).to_csv(OUT / "T_p11_swap_test.csv", index=False, encoding="utf-8-sig")
print(f"\n[已写出] {OUT / 'T_p11_swap_test.csv'}")
