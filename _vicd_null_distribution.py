# -*- coding: utf-8 -*-
r"""VICD 判据升级：把"四个口径的点估计"升为"带零假设与阈值的判据"。

背景
----
论文 §3.4.1 已定义 VICD 四口径（增量相关 / 归一化互信息 / 符号对齐率 / 滞后-先导），
但三案例（水文、锚索、沉降）目前只有点估计，没有回答审稿人必问的两件事：
  (a) "相关很小" 的阈值是多少？凭什么说 0.018 就是小？
  (b) 这个阈值本身的不确定度多大？随样本长度、自相关强度怎么变？

本脚本把判据实现为**两层**：

  层 1（输入资格，强判据）—— 增量口径
    M1 |ρ(Δy, Δu)|、M2 归一化互信息(Δ 口径)、M3 |符号对齐率−0.5|、M4 max_τ|ρ(Δy_{t+τ}, Δu_t)|
    四项各配"相位随机化代理"零分布；用 **max-ratio 统计量** T = max_k(m_k/thr_k) 做统一判定
    （避免"四项都要过 ⇒ 假阳率 ≈ 1−0.95⁴ ≈ 18%"的多重比较膨胀）。
    T_obs ≤ T_null 的 95% 分位 ⇒ 判"不可作预测输入"。

  层 2（验证资格，定位判据）—— 累计口径
    报告 |ρ(y_cum, u_cum)| 与其零分布阈值作参考，但**不做显著性主张**，只做定位：
      累计强 + 增量弱 ⇒ "共享趋势型耦合" ⇒ 可作独立一致性验证量
      累计弱 + 增量弱 ⇒ 无任何耦合       ⇒ 连验证都不可用
    理由（实跑后发现）：累计序列的功率谱本身就被趋势主导，任何保谱代理都会同样"趋势相关"，
    故累计口径不可能有一个既能检出真耦合、又不被共同趋势击穿的谱匹配阈值。这一点必须写进论文。

零假设 H0：u 对 y 无可利用信息，但**保留 u 自身的自相关/功率谱结构**。
  → 相位随机化代理（保功率谱）+ 循环时移代理（稳健性对照）；另设 iid 代理作**反例**，
    用于说明"若用白噪声零分布，阈值会被系统性压小"。

三项检验：
  (n)     子样本 n×{1.0,0.75,0.5}：阈值随样本长度漂移；
  (cal)   零分布校准：真零假设下三种代理构造的假阳率应 ≈5%（检验判据本身是否被正确标定）；
  (power) 合成正对照 y 含 τ=3 先导注入：判据应检出"可用"，证明判据不恒输出"不可用"。

口径说明：全部指标在**标量序列**上计算（断面内层均值），不把 155 个节点 ravel 成 155·T 个
"样本"——节点间高度相关，ravel 会把有效样本量虚增两个量级。

运行：
  python -u _vicd_null_distribution.py                # 全量 B=1000
  python -u _vicd_null_distribution.py --quick        # B=200 冒烟
  python -u _vicd_null_distribution.py --cases water,settlement,anchor
输出：终端 + _vicd_null/*.csv
"""
from __future__ import annotations

from _env import DATA_ROOT, WORK_ROOT

import argparse
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).parent
OUT = ROOT / "_vicd_null"
OUT.mkdir(parents=True, exist_ok=True)

sys.path.insert(0, str(ROOT))

SEED = 42
LAGS = (0, 3, 6, 12)      # 滞后-先导的 τ（与 §3.4.1 一致）
BINS = 20                 # 互信息离散化箱数（与既有脚本一致）
TOL_DAYS = 1.5            # 时间最近邻对齐容差（天）
BAD_CAP = 1500.0          # 锚索坏点剔除上限（kN）
DOT = {"\u2219", "\u00b7", "\u2022"}

INCREMENT_KEYS = ("M1_inc_corr", "M2_inc_mi", "M3_sign_gain", "M4_lead_max")


# ---------------------------------------------------------------- 基础统计量
def safe_corr(a, b) -> float:
    a = np.asarray(a, float)
    b = np.asarray(b, float)
    if len(a) < 10 or a.std() < 1e-12 or b.std() < 1e-12:
        return np.nan
    return float(np.corrcoef(a, b)[0, 1])


def norm_mi(x, y, bins: int = BINS) -> float:
    """离散化归一化互信息 MI(x,y)/H(y)。"""
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    x = x[np.isfinite(x)]
    y = y[np.isfinite(y)]
    n = min(len(x), len(y))
    if n < 20:
        return np.nan
    x, y = x[:n], y[:n]
    if x.std() < 1e-12 or y.std() < 1e-12:
        return 0.0

    def digitize(v):
        edges = np.quantile(v, np.linspace(0, 1, bins + 1)[1:-1])
        return np.clip(np.digitize(v, edges), 0, bins - 1)

    xb, yb = digitize(x), digitize(y)
    c = np.zeros((bins, bins), float)
    np.add.at(c, (xb, yb), 1.0)
    pxy = c / c.sum()
    px = pxy.sum(1, keepdims=True)
    py = pxy.sum(0, keepdims=True)
    mi = float((pxy * np.log(pxy / (px * py + 1e-12) + 1e-12)).sum())
    hy = float(-(py * np.log(py + 1e-12)).sum())
    return mi / (hy + 1e-12)


def sign_align_gain(dy, du) -> float:
    """|符号对齐率 − 0.5|，0.5 为随机基线。"""
    dy = np.asarray(dy, float)
    du = np.asarray(du, float)
    n = min(len(dy), len(du))
    if n < 10:
        return np.nan
    sdy, sdu = np.sign(dy[:n]), np.sign(du[:n])
    ok = (sdy != 0) & (sdu != 0)
    if ok.sum() < 10:
        return np.nan
    return float(abs((sdy[ok] == sdu[ok]).mean() - 0.5))


def lead_corr_max(dy, du, lags=LAGS) -> float:
    """max_τ |ρ(Δy_{t+τ}, Δu_t)|。"""
    dy = np.asarray(dy, float)
    du = np.asarray(du, float)
    vals = []
    for tau in lags:
        n = min(len(dy), len(du)) - tau
        if n < 10:
            continue
        vals.append(abs(safe_corr(dy[tau:tau + n], du[:n])))
    vals = [v for v in vals if np.isfinite(v)]
    return float(max(vals)) if vals else np.nan


def metrics(y_cum, u_cum) -> dict:
    """四口径（增量）+ 水平相关（定位用）。全部取"越大越可疑"的绝对值方向。"""
    y_cum = np.asarray(y_cum, float)
    u_cum = np.asarray(u_cum, float)
    n = min(len(y_cum), len(u_cum))
    y_cum, u_cum = y_cum[:n], u_cum[:n]
    dy, du = np.diff(y_cum), np.diff(u_cum)
    return {
        "M1_inc_corr": abs(safe_corr(dy, du)),
        "M2_inc_mi": norm_mi(dy, du),
        "M3_sign_gain": sign_align_gain(dy, du),
        "M4_lead_max": lead_corr_max(dy, du),
        "MLEV_corr": abs(safe_corr(y_cum, u_cum)),
    }


# ---------------------------------------------------------------- 代理序列
def phase_surrogate(x, rng) -> np.ndarray:
    """相位随机化：保功率谱（⇒ 保自相关），打乱相位。DC/Nyquist 相位不动。"""
    x = np.asarray(x, float)
    n = len(x)
    x = x - x.mean()
    F = np.fft.rfft(x)
    mag, ph = np.abs(F), np.angle(F)
    new_ph = rng.uniform(-np.pi, np.pi, len(F))
    new_ph[0] = ph[0]
    if n % 2 == 0:
        new_ph[-1] = ph[-1]
    return np.fft.irfft(mag * np.exp(1j * new_ph), n=n)


def shift_surrogate(x, rng) -> np.ndarray:
    """循环时移：严格保全部二阶结构，只用"时间错位"破坏同期关联。"""
    n = len(x)
    k = int(rng.integers(1, n))
    return np.roll(np.asarray(x, float), k)


def iid_surrogate(x, rng) -> np.ndarray:
    """白噪声代理（标准差匹配）：**反例**，故意破坏自相关，用于说明 iid 零分布会压小阈值。"""
    x = np.asarray(x, float)
    return rng.normal(x.mean(), x.std(), size=len(x)) if x.std() > 1e-12 else x.copy()


SURROGATES = {"phase": phase_surrogate, "shift": shift_surrogate, "iid": iid_surrogate}


# ---------------------------------------------------------------- 阈值与判定
def null_analysis(y_cum, u_cum, B: int, rng, make_surrogate=phase_surrogate,
                  alpha: float = 0.95, boot: int = 200,
                  keys=INCREMENT_KEYS + ("MLEV_corr",)) -> dict:
    """一次遍历代理，给出：各指标零分布阈值(+bootstrap CI) 与 max-ratio 统计量的零分布。

    T = max_k (m_k / thr_k)（仅对增量口径 INCREMENT_KEYS）；T_obs ≤ T_null 的 alpha 分位
    ⇒ 判"不可用"。这样多重比较被自动吸收，真零假设下假阳率 ≈ 1−alpha（而非 1−alpha⁴）。
    MLEV_corr 只报阈值作定位参考，不进入 T。
    """
    obs = metrics(y_cum, u_cum)
    null = {k: [] for k in keys}
    for _ in range(B):
        m = metrics(y_cum, make_surrogate(u_cum, rng))
        for k in keys:
            null[k].append(m[k])
    arr = {k: np.asarray(null[k], float) for k in keys}

    # 阈值与 T_null 分用两半代理（held-out），消除"同批数据估阈值又估 T 分布"的乐观偏差
    half = max(B // 2, 20)
    thr, ci = {}, {}
    for k in keys:
        v = arr[k][:half][np.isfinite(arr[k][:half])]
        if len(v) < 20:
            thr[k], ci[k] = np.nan, (np.nan, np.nan)
            continue
        thr[k] = float(np.percentile(v, alpha * 100))
        if boot > 0:
            bs = [np.percentile(rng.choice(v, size=len(v), replace=True), alpha * 100)
                  for _ in range(boot)]
            ci[k] = tuple(float(x) for x in np.percentile(bs, [2.5, 97.5]))
        else:
            ci[k] = (np.nan, np.nan)

    T_null = []
    for b in range(half, B):
        ratios = [arr[k][b] / thr[k] for k in INCREMENT_KEYS
                  if np.isfinite(arr[k][b]) and np.isfinite(thr[k]) and thr[k] > 0]
        if ratios:
            T_null.append(max(ratios))
    T_null = np.asarray(T_null, float)
    T_thr = float(np.percentile(T_null, alpha * 100)) if len(T_null) >= 20 else np.nan
    ratios_obs = [(k, obs[k] / thr[k]) for k in INCREMENT_KEYS
                  if np.isfinite(obs.get(k, np.nan)) and np.isfinite(thr[k]) and thr[k] > 0]
    T_obs = max((r for _, r in ratios_obs), default=np.nan)

    return {"obs": obs, "thr": thr, "ci": ci,
            "T_obs": T_obs, "T_thr": T_thr,
            "argmax": max(ratios_obs, key=lambda t: t[1])[0] if ratios_obs else None}


def one_shot_verdict(y_cum, u_cum, B: int, rng, make_surrogate, alpha) -> str:
    """校准用：只回判定标签。"""
    r = null_analysis(y_cum, u_cum, B, rng, make_surrogate, alpha=alpha, boot=0)
    if not np.isfinite(r["T_obs"]) or not np.isfinite(r["T_thr"]):
        return "无法判定"
    return "不可用" if r["T_obs"] <= r["T_thr"] else "越界→需再检验"


def verification_position(mlev_obs: float, mlev_thr: float, m1_obs: float,
                          same_quantity: bool) -> str:
    """层 2 定位（**不做显著性主张**）。

    实跑后发现：累计序列的功率谱被趋势主导 ⇒ 保谱代理也"趋势相关" ⇒ 谱匹配零分布阈值
    几乎不可达（沉降 MLEV 0.79 vs 阈 0.82；水位深层 0.68 vs 阈 0.72）。故累计口径不能
    给出显著性判定，只能按**物理角色**定位：
      同量异器（沉降↔位移） → 累计耦合本身即验证依据（映射一致即可用）
      异质变量（水位、锚索）→ 累计耦合只反映共同趋势，不构成验证资格
    """
    cum_over = (np.isfinite(mlev_obs) and np.isfinite(mlev_thr) and mlev_obs >= mlev_thr)
    if not same_quantity:
        return "异质变量 → 不作独立验证量（仅层1输入资格 + 健康诊断旁证）"
    if cum_over:
        return "同量异器 + 累计耦合越零分布 → 可作独立一致性验证"
    return "同量异器；累计耦合未越零分布(共同趋势混淆) → 仅定性验证，不作显著性主张"


# ---------------------------------------------------------------- 数据装配
def build_records(case: str, cfg, disp_raw, meta, ts, depths, point_names):
    """返回 [(name, section, layer, y_cum, u_cum), ...]（标量序列对）。"""
    recs = []
    LAYERS = {
        "全深度": np.ones(len(depths), bool),
        "浅(<3m)": depths < 3.0,
        "中(3-8m)": (depths >= 3.0) & (depths < 8.0),
        "深(8-16m)": depths >= 8.0,
    }

    def layer_series(section, t_map):
        pidx = np.where(np.asarray(point_names) == section)[0]
        for lname, lmask in LAYERS.items():
            lsel = pidx[lmask[pidx]]
            if len(lsel) == 0:
                continue
            yield lname, disp_raw[t_map][:, lsel].mean(1)

    if case == "water":
        import Ablation_Experiment as AE
        wd, _wa = AE.load_water_data(ts, cfg)
        u_full = np.asarray(wd, float)
        t_map = np.arange(len(ts))
        for lname, y in layer_series("CX2", t_map):   # 水位为场地级，任一断面代表即可
            recs.append(("水位(wd)", "全场", lname, y, u_full))

    elif case == "settlement":
        PAIRS = {"S6": "CX2", "S18": "CX8", "S31": "CX15"}
        for sname, cx in PAIRS.items():
            fp = DATA_ROOT / f"shuju/沉降/{sname[1:]}.xlsx"
            if not fp.exists():
                print(f"[warn] 缺失 {fp}，跳过")
                continue
            sd = pd.read_excel(fp)
            sd["监测时间"] = pd.to_datetime(sd["监测时间"])
            sd = sd.sort_values("监测时间").reset_index(drop=True)
            dt_days = (sd["监测时间"].values[:, None] - ts.values[None, :]) \
                .astype("timedelta64[h]").astype(float) / 24.0
            idx = np.abs(dt_days).argmin(1)
            gap = np.abs(dt_days).min(1)
            keep = (gap <= TOL_DAYS) & sd["累计沉降量_mm"].notna().values
            if keep.sum() < 20:
                print(f"[warn] {sname} 对齐后样本 {keep.sum()} < 20，跳过")
                continue
            t_map = idx[keep]
            u = sd["累计沉降量_mm"].values[keep].astype(float)
            for lname, y in layer_series(cx, t_map):
                recs.append((f"{sname}↔{cx}", cx, lname, y, u))

    elif case == "anchor":
        ML = WORK_ROOT / "ML数据"
        PAIRS = {"ML2": "CX2", "ML3": "CX3", "ML8": "CX8",
                 "ML13": "CX13", "ML15": "CX15"}

        def to_float(x):
            if isinstance(x, (int, float)):
                return float(x)
            if isinstance(x, str):
                s = x.strip()
                for c in DOT:
                    s = s.replace(c, ".")
                s = re.sub(r"(?<=\d)\s+(?=\d)", ".", s).replace(" ", "")
                try:
                    return float(s)
                except ValueError:
                    return np.nan
            return np.nan

        for mlpref, cx in sorted(PAIRS.items()):
            for fk in sorted(ML.glob(f"{mlpref}-*.xlsx")):
                df = pd.read_excel(fk)
                df["日期"] = pd.to_datetime(df["测量时间"], errors="coerce").dt.normalize()
                df["v"] = df["测试值_KN"].apply(to_float)
                df = df[df["v"].notna() & (df["v"].abs() <= BAD_CAP)]
                g = df.groupby("日期", as_index=False)["v"].mean().sort_values("日期")
                g = g.dropna(subset=["日期"]).reset_index(drop=True)
                if len(g) < 20:
                    continue
                a_t = pd.to_datetime(g["日期"])
                a_v = g["v"].to_numpy(float)
                dt_days = (a_t.values[:, None] - ts.values[None, :]) \
                    .astype("timedelta64[h]").astype(float) / 24.0
                idx = np.abs(dt_days).argmin(1)
                gap = np.abs(dt_days).min(1)
                keep = gap <= TOL_DAYS
                if keep.sum() < 20:
                    continue
                t_map, u = idx[keep], a_v[keep]
                for lname, y in layer_series(cx, t_map):
                    recs.append((f"{fk.stem}↔{cx}", cx, lname, y, u))
    else:
        raise ValueError(case)
    return recs


def ar1(n, phi, rng):
    e = rng.normal(size=n)
    x = np.zeros(n)
    for i in range(1, n):
        x[i] = phi * x[i - 1] + e[i]
    return x


# ---------------------------------------------------------------- 主流程
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true", help="B=200 冒烟")
    ap.add_argument("--cases", default="water,settlement,anchor")
    ap.add_argument("--alpha", type=float, default=0.95)
    args = ap.parse_args()
    B = 200 if args.quick else 1000
    alpha = args.alpha

    import Ablation_Experiment as AE
    cfg = AE.Config()
    cfg.RESUME = 0
    AE.set_seed(SEED)
    rng = np.random.default_rng(SEED)

    full_df = AE.load_raw_data_with_meta(cfg)
    disp_raw, _, meta = AE.build_node_features(full_df, cfg)
    ts = pd.to_datetime(meta["timestamps"])
    depths = np.asarray(meta["depths"], float)
    point_names = np.asarray(meta["point_names"])

    print("=" * 108)
    print(f"VICD 零分布判据  |  B={B}  alpha={alpha:.2f}  seed={SEED}")
    print(f"位移场 T={disp_raw.shape[0]} 节点={disp_raw.shape[1]}  对齐容差={TOL_DAYS}天  "
          f"互信息箱数={BINS}  先导 τ={LAGS}")
    print("层1（输入资格）=增量口径 M1–M4，max-ratio 统一判定；层2（验证资格）=累计口径定位（不做显著性主张）")
    print("=" * 108)

    rows = []
    cases = [c.strip() for c in args.cases.split(",") if c.strip()]

    for case in cases:
        recs = build_records(case, cfg, disp_raw, meta, ts, depths, point_names)
        if not recs:
            print(f"\n[{case}] 无可用记录")
            continue
        print(f"\n{'#' * 108}\n# 案例：{case}   记录数={len(recs)}\n{'#' * 108}")
        header = (f"{'记录':<16}{'层':<10}{'n':>4} | "
                  f"{'M1增量ρ':>8}{'阈':>7} | {'M2增量MI':>9}{'阈':>7} | "
                  f"{'M3符号':>8}{'阈':>7} | {'M4先导':>8}{'阈':>7} | "
                  f"{'T_obs':>7}{'T阈':>7} | {'层1':<14}| {'MLEV':>7}{'阈':>7} | 层2")
        print(header)
        print("-" * len(header))

        for name, section, lname, y_cum, u_cum in recs:
            y = np.asarray(y_cum, float)
            u = np.asarray(u_cum, float)
            n = min(len(y), len(u))
            y, u = y[:n], u[:n]
            ok = np.isfinite(y) & np.isfinite(u)
            y, u = y[ok], u[ok]
            if len(y) < 20:
                continue
            r = null_analysis(y, u, B, rng, phase_surrogate, alpha=alpha)
            o, t = r["obs"], r["thr"]
            v1 = ("不可用" if (np.isfinite(r["T_obs"]) and np.isfinite(r["T_thr"])
                           and r["T_obs"] <= r["T_thr"]) else "越界→需再检验")
            v2 = verification_position(o["MLEV_corr"], t["MLEV_corr"], o["M1_inc_corr"],
                                       same_quantity=(case == "settlement"))

            def g(d, k):
                return f"{d[k]:.3f}" if np.isfinite(d.get(k, np.nan)) else "nan"

            print(f"{name:<16}{lname:<10}{len(y):>4} | "
                  f"{g(o,'M1_inc_corr'):>8}{g(t,'M1_inc_corr'):>7} | "
                  f"{g(o,'M2_inc_mi'):>9}{g(t,'M2_inc_mi'):>7} | "
                  f"{g(o,'M3_sign_gain'):>8}{g(t,'M3_sign_gain'):>7} | "
                  f"{g(o,'M4_lead_max'):>8}{g(t,'M4_lead_max'):>7} | "
                  f"{r['T_obs']:>7.3f}{r['T_thr']:>7.3f} | {v1:<14}| "
                  f"{g(o,'MLEV_corr'):>7}{g(t,'MLEV_corr'):>7} | {v2}")

            rows.append({
                "case": case, "record": name, "section": section, "layer": lname,
                "n": len(y), **{k: o[k] for k in o},
                **{f"{k}_thr_95": t[k] for k in t},
                **{f"{k}_thr_ci_lo": r["ci"][k][0] for k in r["ci"]},
                **{f"{k}_thr_ci_hi": r["ci"][k][1] for k in r["ci"]},
                "T_obs": r["T_obs"], "T_thr_95": r["T_thr"], "argmax": r["argmax"],
                "layer1_input_verdict": v1, "layer2_verification_position": v2,
            })

    df = pd.DataFrame(rows)
    if not df.empty:
        df.round(4).to_csv(OUT / "vicd_thresholds.csv", index=False, encoding="utf-8-sig")
        print(f"\n[已写出] {OUT / 'vicd_thresholds.csv'}")

        print("\n" + "=" * 108)
        print("层 1 判定汇总（按案例 × 层）：'不可用' 占比 = 判为不可作预测输入的比例")
        print("=" * 108)
        g = df.copy()
        g["inc"] = (g["layer1_input_verdict"] == "不可用").astype(float)
        piv = g.groupby(["case", "layer"], as_index=False).agg(
            层1_不可用占比=("inc", "mean"), n_中位=("n", "median"),
            M1中位=("M1_inc_corr", "median"), MLEV中位=("MLEV_corr", "median"))
        piv["层1_不可用占比"] = piv["层1_不可用占比"].map("{:.0%}".format)
        piv["n_中位"] = piv["n_中位"].astype(int)
        print(piv.round(3).to_string(index=False))

        print("\n" + "=" * 108)
        print("层 2 定位汇总（按案例）：三案例应分别落入三种定位")
        print("=" * 108)
        print(df.groupby(["case", "layer2_verification_position"]).size()
              .rename("记录数").reset_index().to_string(index=False))

    # ---------------------------------------------------- 敏感性 (n)
    print("\n" + "=" * 108)
    print("敏感性 n：阈值随样本长度漂移（取第一条记录，截断前 100%/75%/50%）")
    print("=" * 108)
    sens_rows = []
    if rows:
        rec0 = build_records(cases[0], cfg, disp_raw, meta, ts, depths, point_names)
        if rec0:
            y0 = np.asarray(rec0[0][3], float)
            u0 = np.asarray(rec0[0][4], float)
            n_full = min(len(y0), len(u0))
            for frac in (1.0, 0.75, 0.5):
                k = int(n_full * frac)
                r = null_analysis(y0[:k], u0[:k], max(B // 2, 100), rng, phase_surrogate,
                                  alpha=alpha, boot=0)
                sens_rows.append({"frac": frac, "n": k, "T_thr_95": r["T_thr"],
                                  **{f"{key}_thr_95": r["thr"][key] for key in INCREMENT_KEYS}})
                print(f"  n={k:>4} ({frac:.0%})  " + "  ".join(
                    f"{key}={r['thr'][key]:.3f}" for key in INCREMENT_KEYS)
                    + f"  T阈值={r['T_thr']:.3f}")
            pd.DataFrame(sens_rows).round(4).to_csv(
                OUT / "vicd_threshold_sensitivity_n.csv", index=False, encoding="utf-8-sig")

    # ---------------------------------------------------- 校准 (cal)：假阳率
    print("\n" + "=" * 108)
    print("零分布校准：真零假设（y、u 各自 AR(1)，相互独立）下的假阳率，应 ≈ 1−alpha")
    print("=" * 108)
    cal_rows = []
    n_cal, reps = 150, 40
    for phi in (0.0, 0.5, 0.9):
        line = [f"  phi={phi:.1f} n={n_cal} reps={reps} |"]
        for si, (sname, sfun) in enumerate(SURROGATES.items()):
            fp = 0
            a = np.random.default_rng(SEED + int(phi * 100) + si)
            for _ in range(reps):
                u = ar1(n_cal, phi, a) if phi > 0 else a.normal(size=n_cal)
                y = ar1(n_cal, phi, a) if phi > 0 else a.normal(size=n_cal)
                if one_shot_verdict(y, u, max(B // 4, 100), rng, sfun, alpha) != "不可用":
                    fp += 1
            rate = fp / reps
            cal_rows.append({"phi": phi, "surrogate": sname, "fpr": rate, "reps": reps})
            line.append(f" {sname}: FPR={rate:.0%}")
        line.append(f" | 名义 FPR={1 - alpha:.0%}")
        print("".join(line))
    pd.DataFrame(cal_rows).round(4).to_csv(
        OUT / "vicd_null_calibration.csv", index=False, encoding="utf-8-sig")

    # ---------------------------------------------------- 判别力 (power) 正对照
    print("\n" + "=" * 108)
    print("判别力正对照：合成 y 含 τ=3 先导注入 y_t = a·u_{t−3} + 噪声（u 为 AR(1),phi=0.8）")
    print("=" * 108)
    n_pc, reps_pc = 150, 20
    power_rows = []
    for a_eff in (0.0, 0.15, 0.3, 0.6):
        det = 0
        for r_i in range(reps_pc):
            a = np.random.default_rng(SEED + 1000 + r_i)
            u = ar1(n_pc, 0.8, a)
            y = np.zeros(n_pc)
            y[3:] = a_eff * u[:-3] + 0.5 * a.normal(size=n_pc - 3)
            if one_shot_verdict(y, u, max(B // 3, 100), rng, phase_surrogate, alpha) != "不可用":
                det += 1
        power_rows.append({"effect": a_eff, "detect_rate": det / reps_pc, "reps": reps_pc})
        print(f"  注入强度 a={a_eff:.2f}  检出率={det / reps_pc:.0%}")
    pd.DataFrame(power_rows).round(4).to_csv(
        OUT / "vicd_power_positive_control.csv", index=False, encoding="utf-8-sig")

    print("\nDone ->", OUT)


if __name__ == "__main__":
    main()