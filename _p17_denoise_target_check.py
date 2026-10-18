# _p17_denoise_target_check.py - 去噪目标口径核查（零训练）。
# 代码核查结论：prepare_data 的目标 y = diff1_den（去噪序列一阶差分），评估也用去噪真值。
# 本脚本量化：① 去噪对增量目标的实际改变量级；② "打不赢零变化"对目标口径是否稳健
#   （测试段 AR(1) 一步 skill 在原始 vs 去噪增量上的对照）。

from _env import WORK_ROOT
import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
sys.path.insert(0, str(WORK_ROOT))
import numpy as np
import Ablation_Experiment as AE
from Ablation_Experiment import Config, build_node_features, denoise_split

cfg = Config()
full_df = AE.load_raw_data_with_meta(cfg)
disp_raw, _, meta = build_node_features(full_df, cfg)
n = len(meta['timestamps'])
tr_end = int(n * cfg.TRAIN_RATIO); va_end = int(n * (cfg.TRAIN_RATIO + cfg.VAL_RATIO))
train_idx = np.arange(tr_end); val_idx = np.arange(tr_end, va_end); test_idx = np.arange(va_end, n)

tr_d, va_d, te_d = denoise_split(disp_raw[train_idx], disp_raw[val_idx], disp_raw[test_idx], cfg)
all_den = np.vstack([tr_d, va_d, te_d])
raw_diff = np.zeros_like(disp_raw); raw_diff[1:] = np.diff(disp_raw, axis=0)
den_diff = np.zeros_like(all_den); den_diff[1:] = np.diff(all_den, axis=0)

# 1) 去噪对增量目标的改变量级
d = den_diff - raw_diff
print("=" * 70)
print("[1] 去噪 vs 原始增量 差异量级（全节点×时点）")
print("=" * 70)
print(f"  原始增量 std      = {raw_diff.std():.4f} mm")
print(f"  去噪增量 std      = {den_diff.std():.4f} mm")
print(f"  相对幅度变化      = {(den_diff.std()/raw_diff.std()-1)*100:+.1f}%")
print(f"  差异 mean/std/max = {d.mean():.4f} / {d.std():.4f} / {np.abs(d).max():.4f} mm")
corr = np.mean([np.corrcoef(raw_diff[:, j], den_diff[:, j])[0, 1] for j in range(raw_diff.shape[1])])
print(f"  逐节点原始-去噪增量相关均值 = {corr:.4f}")
print(f"  去噪改动幅度 > 0.01mm(分辨率) 的 (节点,时点) 占比 = "
      f"{(np.abs(d) > 0.01).mean():.1%}")

# 2) 目标口径稳健性：测试段 AR(1) 一步预测 skill（相对零变化）
print("=" * 70)
print("[2] 测试段 AR(1) 一步 skill（相对零变化）· 原始 vs 去噪目标")
print("=" * 70)
for name, series in [("原始增量", raw_diff), ("去噪增量", den_diff)]:
    te = series[test_idx]
    err_ar, err_zero, nz = [], [], 0
    for j in range(te.shape[1]):
        x = te[:, j]
        if np.all(np.abs(x) < 1e-9):
            continue
        nz += 1
        x1, x2 = x[:-1], x[1:]
        m = np.polyfit(x1, x2, 1)
        e_ar = np.mean((x2 - (x1 * m[0] + m[1])) ** 2)
        e_zero = np.mean(x2 ** 2)
        err_ar.append(e_ar); err_zero.append(e_zero)
    skill = 1 - np.mean(err_ar) / np.mean(err_zero)
    print(f"  {name:<6} 有效节点 {nz} | AR(1) 一步 skill = {skill:+.4f}")

print("=" * 70)
print("解读：若 [1] 去噪改动远小于增量幅度且相关≈1，且 [2] 两口径 AR(1) 均≤0，")
print("  则'打不赢零变化'对目标口径稳健、不可归因于去噪；仅需正文订正目标口径声明。")
