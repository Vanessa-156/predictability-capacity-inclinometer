# -*- coding: utf-8 -*-
r"""_p06_deep_neg.py —— 深部负读数的幅值分布（非频次）+ 2.0 mm 漂移带对比。

本脚本做什么（零训练、只读）
----------------------------
1) 逐断面 × 逐深度的负读数**幅值**（累计位移的负向幅度 |min|、负值时长占比、末值），
   按深度带（浅 <3 / 中 3–8 / 深 8–16 m）聚合；
2) 与 **2.0 mm 漂移带**对比：负读数峰值 ≤2 mm（漂移带内，可归因于基准点漂移）的
   节点比例 vs >2 mm（超出漂移带，须另作解释）的比例；
3) 判读：深部负读数是否"幅值上"超出漂移带、是否集中在深部（支撑"监测深度覆盖"
   作为配置变量，§4.3/§5.2）。
输出 `_paper_tables\T_p06_deep_neg.csv`（逐节点 + 逐带）+ 判读文字。

铁律
----
· 零训练、只读数据，不改锁定脚本；只写 `_paper_tables\`。
· 补材编号预留 **S27**（S26=现代时序基线）。
"""
from _env import DATA_ROOT, WORK_ROOT

import numpy as np
import pandas as pd
from pathlib import Path

PROJ = WORK_ROOT
TABLES = PROJ / "_paper_tables"
TABLES.mkdir(parents=True, exist_ok=True)

GRID_DIR = DATA_ROOT / "ceshi14/MoXin/data"
POINTS = ["CX2", "CX3", "CX8", "CX13", "CX15"]
DEPTHS = np.arange(0.5, 16.0, 0.5)
DRIFT_BAND = 2.0          # mm：基准点漂移带（正文 §2.2 既有定义）
BANDS = [("Shallow", 0.5, 3.0), ("Middle", 3.0, 8.0), ("Deep", 8.0, 16.0)]


def load_grid(section):
    df = pd.read_excel(GRID_DIR / f"{section}.xlsx")
    df["测量时间"] = pd.to_datetime(df["测量时间"])
    piv = df.pivot_table(index="测量时间", columns="深度_m",
                         values="累计水平位移_mm", aggfunc="first")
    return piv.reindex(columns=DEPTHS).sort_index().values.astype(np.float64)


def main():
    print("=" * 100)
    print("深部负读数幅值分布 | 2.0 mm 漂移带对比 | 零训练只读")
    print("=" * 100)
    records = []
    for sec in POINTS:
        disp = load_grid(sec)
        for j, d in enumerate(DEPTHS):
            s = disp[:, j]
            neg_mask = s < 0
            rec = {"section": sec, "depth": float(d),
                   "min_cum": float(s.min()), "last_cum": float(s[-1]),
                   "neg_frac": float(neg_mask.mean()),
                   "neg_peak": float(-s[neg_mask].min()) if neg_mask.any() else 0.0,
                   "within_drift": bool((not neg_mask.any()) or (-s[neg_mask].min() <= DRIFT_BAND))}
            records.append(rec)
    df = pd.DataFrame(records)
    df.to_csv(TABLES / "T_p06_deep_neg.csv", index=False, encoding="utf-8-sig")

    # ---- 逐带汇总 ----
    print("\n逐带汇总（155 节点；累计位移，栅格 213 时点）")
    print(f"{'带':<10}{'深度':<10}{'节点数':>6}{'负cell占比':>10}{'|min|中位':>10}"
          f"{'|min|p95':>10}{'|min|最大':>10}{'末值中位':>10}{'≤2mm漂移带占比':>12}")
    for name, lo, hi in BANDS:
        sub = df[(df.depth >= lo) & (df.depth < hi)]
        peak = sub.neg_peak
        print(f"{name:<10}{f'{lo}-{hi}':<10}{len(sub):>6}{sub.neg_frac.mean():>10.1%}"
              f"{peak.median():>10.2f}{peak.quantile(0.95):>10.2f}{peak.max():>10.2f}"
              f"{sub.last_cum.median():>10.2f}{sub.within_drift.mean():>12.1%}")

    # ---- 判读 ----
    deep = df[df.depth >= 8]
    deep_peak = deep.neg_peak
    print("\n" + "=" * 100)
    print("判读")
    print("=" * 100)
    print(f"① 深部（8–16 m）节点 {len(deep)} 个：|min| 中位 {deep_peak.median():.2f} mm、"
          f"p95 {deep_peak.quantile(0.95):.2f} mm、最大 {deep_peak.max():.2f} mm；"
          f"≤2.0 mm 漂移带内占比 {deep.within_drift.mean():.0%}。")
    print(f"   深部负读数中 {deep.within_drift.mean():.0%} 在漂移带内可归因基准漂移，"
          f"其余（|min|>2 mm）超出漂移带，须另作解释（非基准漂移）。")
    for name, lo, hi in BANDS:
        sub = df[(df.depth >= lo) & (df.depth < hi)]
        print(f"② {name} 层：负 cell 占比 {sub.neg_frac.mean():.0%}、|min| p95 "
              f"{sub.neg_peak.quantile(0.95):.2f} mm —— 负读数{'集中在深部' if name=='Deep' and sub.neg_frac.mean()>0.6 else '不限于深部'}。")
    print("\n说明：幅值口径按逐节点累计位移最小值的负向幅度计（非频次）；")
    print("'含/不含深部节点'两套模型结果 = 重训实验（脚本 _p06_no_deep.py，待作者跑）。")
    print(f"\n[已写出] {TABLES / 'T_p06_deep_neg.csv'}（155 节点逐节点）")


if __name__ == "__main__":
    main()
