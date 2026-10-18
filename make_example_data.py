# -*- coding: utf-8 -*-
"""make_example_data.py — 生成 schema 一致的合成示例数据（仅结构/变量名/格式与真实一致，数值虚构）。

用途：让审稿人在无真实监测数据的干净环境下，也能按 README 步骤"无错误运行"全部
分析脚本。合成数据不包含任何真实工程数值，论文中报告的结果全部基于保密真实数据。

生成内容（写到 DATA_ROOT / WORK_ROOT，见 _env.py）：
  1. DATA_ROOT/ceshi14/MoXin/data/CX{2,3,8,13,15}.xlsx   —— 测斜位移（5 测点 × 31 深度 × 213 时点）
  2. DATA_ROOT/ceshi14/shuiwei/时间统一+插值/SW{1,4,7}.xlsx —— 水位（3 口井）
  3. DATA_ROOT/ceshi14/date/raw/地质参数表.xlsx             —— 地质分层参数
  4. DATA_ROOT/ceshi14/MoXin/data/斜测点坐标.xlsx           —— 测点平面坐标
  5. DATA_ROOT/shuju/沉降/{6,18,31}.xlsx                    —— 地表沉降（S6/S18/S31）
  6. WORK_ROOT/ML数据/ML{2,3,8,13,15}-1.xlsx               —— 支撑轴力（锚索测力）

运行（PowerShell，在仓库根目录）：
  & "<PYENV_ROOT>\python.exe" -u make_example_data.py
"""
import os
from pathlib import Path

import numpy as np
import pandas as pd

from _env import DATA_ROOT, WORK_ROOT

# 每个生成器用独立种子流，避免改动某一类文件时连带改变其他文件的数值。
# 注意：不能直接用内置 hash()（进程间随机化），故用固定映射。
_TAG_SEEDS = {"points": 101, "raw": 202, "water": 303, "geo": 404,
              "coords": 505, "settle": 606, "anchor": 707}


def _rng(tag: str):
    return np.random.default_rng(_TAG_SEEDS[tag])

POINTS = ["CX2", "CX3", "CX8", "CX13", "CX15"]
DEPTHS = np.arange(0.5, 16.0, 0.5)          # 31 个深度
N_TIME = 213                                 # 与真实栅格时点数一致（2 天间隔）
TIMESTAMPS = pd.date_range("2022-01-01", periods=N_TIME, freq="2D")


def _bulge(depth_m, t_frac):
    """构造一个随深度/时间光滑变化的墙侧移剖面（仅用于合成，无物理含义）。"""
    depth_norm = depth_m / 16.0
    # 中间深度鼓出最大，随开挖进度(t_frac)增大
    bulge = 30.0 * np.sin(np.pi * np.clip(depth_norm, 0, 1)) * t_frac
    return bulge


def gen_point_files():
    rng = _rng("points")
    d = DATA_ROOT / "ceshi14" / "MoXin" / "data"
    d.mkdir(parents=True, exist_ok=True)
    t_frac = np.linspace(0.05, 1.0, N_TIME)
    # 施工进度标签：开挖期标注到米（供 _p05 解析开挖深度锚点），到底后进入结构期
    split = int(N_TIME * 0.6)
    exc_depth = np.linspace(0.0, 16.0, split)
    stage = np.array(["开挖至%.0f米" % d for d in exc_depth]
                     + ["结构施工"] * (N_TIME - split), dtype=object)
    for p in POINTS:
        rows = []
        for di, depth in enumerate(DEPTHS):
            base = _bulge(depth, t_frac) + 0.02 * di
            noise = rng.normal(0, 0.15, N_TIME).cumsum() * 0.1
            vals = base + noise
            for ti in range(N_TIME):
                rows.append((TIMESTAMPS[ti], p, depth, vals[ti], stage[ti]))
        df = pd.DataFrame(rows, columns=["测量时间", "测点", "深度_m", "累计水平位移_mm", "施工进度"])
        df.to_excel(d / f"{p}.xlsx", index=False)
    print("  测点位移: %d 文件 × %d 行（含 施工进度 列）" % (len(POINTS), len(rows)))


def gen_raw_files():
    """原始非规则时点（周报口径），供 _p10_noise_floor 的保守对照使用。

    与栅格文件的差别：长表按「每次观测一行」堆叠，并带本次增量、测量序号、
    施工进度与累计天数，列结构与真实周报完全一致（8 列）。
    """
    rng = _rng("raw")
    d = DATA_ROOT / "ceshi14" / "date" / "raw"
    d.mkdir(parents=True, exist_ok=True)
    n_raw = 60
    raw_ts = pd.date_range("2022-01-03", periods=n_raw, freq="7D")
    t_frac = np.linspace(0.05, 1.0, n_raw)
    # 施工进度：与 gen_point_files 同一规则（前 60% 开挖并标注到米，其后结构施工）
    split = int(n_raw * 0.6)
    exc_depth = np.linspace(0.0, 16.0, split)
    stage = np.array(["开挖至%.0f米" % x for x in exc_depth]
                     + ["结构施工"] * (n_raw - split), dtype=object)
    days = [(t - raw_ts[0]).days for t in raw_ts]
    for p in POINTS:
        rows = []
        for di, depth in enumerate(DEPTHS):
            cum = _bulge(depth, t_frac) + 0.02 * di + rng.normal(0, 0.2, n_raw)
            inc = np.diff(np.concatenate([[0.0], cum]))
            inc = np.round(inc / 0.01) * 0.01          # 0.01 mm 量化
            for ti in range(n_raw):
                rows.append((raw_ts[ti], p, depth, ti + 1, inc[ti], cum[ti],
                             stage[ti], days[ti]))
        pd.DataFrame(rows, columns=["测量时间", "测点", "深度_m", "第几次测量",
                                    "本次水平位移_mm", "累计水平位移_mm",
                                    "施工进度", "累计天数_天"]) \
            .to_excel(d / f"{p}.xlsx", index=False)
    print("  原始周报栅格: %d 文件 × %d 行（8 列）" % (len(POINTS), len(rows)))


def gen_water_files():
    rng = _rng("water")
    d = DATA_ROOT / "ceshi14" / "shuiwei" / "时间统一+插值"
    d.mkdir(parents=True, exist_ok=True)
    for well in ["SW1", "SW4", "SW7"]:
        depth = 8.0 - 2.0 * np.linspace(0, 1, N_TIME) + rng.normal(0, 0.1, N_TIME)
        change = depth - depth[0]
        df = pd.DataFrame({"监测时间": TIMESTAMPS, "水位深度_m": depth, "累计水位变化_m": change})
        df.to_excel(d / f"{well}.xlsx", index=False)
    print("  水位: SW1/SW4/SW7")


def gen_geology():
    p = DATA_ROOT / "ceshi14" / "date" / "raw" / "地质参数表.xlsx"
    p.parent.mkdir(parents=True, exist_ok=True)
    # 4 层 × 4 m，覆盖 0–16 m
    df = pd.DataFrame([
        {"厚度": 4.0, "粘聚力": 18.0, "内摩擦角": 15.0, "压缩模量": 6.0,
         "重度": 18.5, "泊松比": 0.33, "静止侧压力系数": 0.55},
        {"厚度": 4.0, "粘聚力": 15.0, "内摩擦角": 12.0, "压缩模量": 5.0,
         "重度": 18.0, "泊松比": 0.35, "静止侧压力系数": 0.60},
        {"厚度": 4.0, "粘聚力": 25.0, "内摩擦角": 20.0, "压缩模量": 12.0,
         "重度": 19.0, "泊松比": 0.30, "静止侧压力系数": 0.45},
        {"厚度": 4.0, "粘聚力": 30.0, "内摩擦角": 25.0, "压缩模量": 20.0,
         "重度": 19.5, "泊松比": 0.28, "静止侧压力系数": 0.40},
    ])
    df.to_excel(p, index=False)
    print("  地质参数: 4 层")


def gen_coords():
    p = DATA_ROOT / "ceshi14" / "MoXin" / "data" / "斜测点坐标.xlsx"
    p.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame([
        {"测点": "CX2", "位置_X_m": 0.0, "位置_Y_m": 0.0},
        {"测点": "CX3", "位置_X_m": 8.0, "位置_Y_m": 1.0},
        {"测点": "CX8", "位置_X_m": 16.0, "位置_Y_m": 0.5},
        {"测点": "CX13", "位置_X_m": 24.0, "位置_Y_m": -0.5},
        {"测点": "CX15", "位置_X_m": 32.0, "位置_Y_m": 0.0},
    ])
    df.to_excel(p, index=False)
    print("  测点坐标: 5 点")


def gen_settlement():
    rng = _rng("settle")
    d = DATA_ROOT / "shuju" / "沉降"
    d.mkdir(parents=True, exist_ok=True)
    for name in ["6", "18", "31"]:           # S6/S18/S31 → 文件名去掉 'S'
        vals = -8.0 * np.linspace(0, 1, N_TIME) + rng.normal(0, 0.2, N_TIME)
        df = pd.DataFrame({"监测时间": TIMESTAMPS, "累计沉降量_mm": vals})
        df.to_excel(d / f"{name}.xlsx", index=False)
    print("  沉降: 6/18/31.xlsx")


def gen_anchor():
    """支撑轴力（锚索测力）：列结构与真实 ML*.xlsx 一致（7 列）。"""
    rng = _rng("anchor")
    d = WORK_ROOT / "ML数据"
    d.mkdir(parents=True, exist_ok=True)
    design_lock = {"ML2": 290.0, "ML3": 290.0, "ML8": 405.0,
                   "ML13": 405.0, "ML15": 290.0}
    for ml, lock in design_lock.items():
        vals = 600.0 + 300.0 * np.linspace(0, 1, N_TIME) + rng.normal(0, 5.0, N_TIME)
        inc = np.concatenate([[0.0], np.diff(vals)])   # 首次测量无前值，记 0
        df = pd.DataFrame({
            "测点": f"{ml}-1",
            "设计锁定力_KN": lock,
            "监测次数": np.arange(1, N_TIME + 1),
            "测量时间": TIMESTAMPS,
            "测试值_KN": vals,
            "本次变化量_KN": inc,
            "累计变化量_KN": vals - vals[0],
        })
        df.to_excel(d / f"{ml}-1.xlsx", index=False)
    print("  支撑轴力: ML{2,3,8,13,15}-1.xlsx（7 列）")


def main():
    print("生成合成示例数据 -> DATA_ROOT=%s" % DATA_ROOT)
    gen_point_files()
    gen_raw_files()
    gen_water_files()
    gen_geology()
    gen_coords()
    gen_settlement()
    gen_anchor()
    print("完成。")


if __name__ == "__main__":
    main()
