# -*- coding: utf-8 -*-
"""生成统一的 30 个随机种子（供四个实验脚本共用）。

规则（可写入论文"实验设置"）：
  保留原有 15 个种子（保证与已完成实验可比），
  追加的 15 个由固定随机源 numpy.random.RandomState(2026)
  在 [1, 2000] 内无放回抽样得到，且排除已用种子。
  该过程与实验结果无关，不存在"挑种子"余地。
"""
import numpy as np

OLD = [42, 15, 77, 85, 120, 532, 901, 23, 12, 14, 19, 75, 250, 140, 135]

rs = np.random.RandomState(2026)
pool = [int(s) for s in rs.choice(np.arange(1, 2001), size=2000, replace=False)]
new = [s for s in pool if s not in OLD][:15]

seeds = OLD + new
assert len(seeds) == 30, len(seeds)
assert len(set(seeds)) == 30

print("NEW 15 :", new)
print("FULL 30:", seeds)
print("SEEDS = [%s]" % ", ".join(str(s) for s in seeds))