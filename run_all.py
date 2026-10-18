# -*- coding: utf-8 -*-
"""run_all.py — 依次运行全部发布脚本（示例数据 / 缩小规模），逐一报告通过/失败。

用途：对应审稿要求 "Archived code runs with the archived data"。
在干净工作目录里执行 `python run_all.py`，应全部返回 exit code 0。

说明：
  · 先用 make_example_data.py / make_example_aux.py / make_example_artifacts.py 生成示例输入；
  · 训练类脚本一律用 --smoke/--probe（缩小规模），零训练脚本直接跑；
  · 每个脚本独立子进程运行，失败不中断，最后汇总。
"""
import os
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
PY = sys.executable

# (脚本, 参数, 单脚本超时秒)
JOBS = [
    ("_gen_seeds.py", [], 60),
    ("_p02_timeblock_inference.py", [], 600),
    ("_p04_increment_caliber.py", [], 900),
    ("_uq_verify.py", [], 600),
    ("_p07_block_conformal.py", ["--smoke"], 900),
    ("_p09_skill_states.py", [], 900),
    ("_p10_noise_floor.py", [], 300),
    ("_p11_swap_test.py", [], 120),
    ("_p11_water_inject.py", [], 900),
    ("_p16_vicd_denoise_check.py", [], 600),
    ("_p17_denoise_target_check.py", [], 600),
    ("_p17_rollorig_zero.py", [], 600),
    ("_k_scan_probe.py", [], 600),
    ("_loocv_mask_effect_check.py", [], 300),
    ("_loocv_section_gap_control.py", [], 300),
    ("_loocv_writeup_analysis.py", [], 300),
    ("_multitarget_vicd.py", ["--quick"], 600),
    ("_persist_baseline.py", [], 600),
    ("_p03_config_capacity.py", ["--smoke"], 900),
    ("_p05_stage_input.py", ["--probe"], 600),
    ("_p06_deep_neg.py", [], 300),
    ("_p06_no_deep.py", ["--anchor"], 900),
    ("_p06_selection_check.py", [], 1200),
    ("_p06_stage2_nestedcv.py", ["--smoke"], 1200),
    ("_p08_learning_curve.py", ["--smoke"], 900),
    ("_p12_p15_baseline_budget.py", ["--smoke"], 1200),
    ("_p02_rollorig.py", ["--stage", "single", "--skip-anchor"], 1800),
    ("_vicd_layer2_criteria.py", ["--quick"], 900),
    ("_vicd_null_distribution.py", ["--quick"], 900),
    ("_vicd_phase_recheck.py", ["--quick"], 900),
]


def main():
    env = dict(os.environ)
    env.setdefault("PYTHONIOENCODING", "utf-8")
    env.setdefault("MPLBACKEND", "Agg")
    # 缩小规模跑通（论文口径为 400/40；此处的默认值只影响本次演示）
    env.setdefault("ABLATION_EPOCHS", "5")
    env.setdefault("ABLATION_PATIENCE", "3")
    # 示例模式：合成数据不可能复现论文数值，硬锚点只告警不终止
    env.setdefault("EXAMPLE_MODE", "1")
    print("ABLATION_EPOCHS=%s (论文口径 400，此处缩小规模)" % env["ABLATION_EPOCHS"])
    results = []
    t_all = time.time()
    for name, args, timeout in JOBS:
        script = HERE / name
        if not script.exists():
            results.append((name, "MISSING", 0.0))
            print("[MISS] %s" % name, flush=True)
            continue
        t0 = time.time()
        try:
            r = subprocess.run([PY, "-u", str(script)] + args, cwd=str(HERE),
                               capture_output=True, text=True, timeout=timeout, env=env)
            dt = time.time() - t0
            if r.returncode == 0:
                results.append((name, "PASS", dt))
                print("[PASS] %-34s %6.1fs" % (name, dt), flush=True)
            else:
                results.append((name, "FAIL", dt))
                tail = (r.stderr or r.stdout or "").strip().splitlines()[-6:]
                print("[FAIL] %-34s %6.1fs" % (name, dt), flush=True)
                for ln in tail:
                    print("       " + ln, flush=True)
        except subprocess.TimeoutExpired:
            results.append((name, "TIMEOUT", time.time() - t0))
            print("[TIME] %-34s > %ds" % (name, timeout), flush=True)

    print("\n" + "=" * 60)
    ok = sum(1 for _, s, _ in results if s == "PASS")
    print("总计 %d 个脚本：PASS %d / FAIL %d / TIMEOUT %d / MISSING %d（%.1f min）"
          % (len(results), ok,
             sum(1 for _, s, _ in results if s == "FAIL"),
             sum(1 for _, s, _ in results if s == "TIMEOUT"),
             sum(1 for _, s, _ in results if s == "MISSING"),
             (time.time() - t_all) / 60))
    bad = [r for r in results if r[1] != "PASS"]
    if bad:
        print("未通过：")
        for name, st, _ in bad:
            print("  %-34s %s" % (name, st))
    return 0 if not bad else 1


if __name__ == "__main__":
    sys.exit(main())
