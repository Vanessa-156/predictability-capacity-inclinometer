# -*- coding: utf-8 -*-
"""Path configuration for the released analysis code.

All machine-specific absolute paths in the original project were neutralized to
four placeholders — <DATA_ROOT>, <WORK_ROOT>, <AUX_ROOT>, <PYENV_ROOT> — and are
resolved here from environment variables. When the variables are unset, the code
falls back to the bundled synthetic example data and example artifacts, so every
script runs out of the box on the demo data (see README.md, "Reproducing").

Override (optional) to point at your own data / artifacts:

    PowerShell:
      $env:DATA_ROOT = "<absolute path to the monitoring data>"
      $env:AUX_ROOT  = "<absolute path to the saved results>"
      $env:WORK_ROOT = "<absolute path to this repository>"

    bash:
      export DATA_ROOT=/path/to/monitoring/data
      export AUX_ROOT=/path/to/saved/artifacts
      export WORK_ROOT=/path/to/this/repository

The private monitoring data and the trained-model artifacts are NOT distributed
(see the manuscript's Data availability statement). The bundled example data are
synthetic and reproduce only the schema of the real files, not the real values.
"""
import os
from pathlib import Path

_HERE = Path(__file__).resolve().parent

DATA_ROOT = Path(os.environ.get("DATA_ROOT", str(_HERE / "example_data")))
WORK_ROOT = Path(os.environ.get("WORK_ROOT", str(_HERE)))
AUX_ROOT = Path(os.environ.get("AUX_ROOT", str(_HERE / "example_artifacts")))
PYENV_ROOT = Path(os.environ.get("PYENV_ROOT", ""))

# 示例模式：在合成示例数据上演示时置 1，跳过"必须与论文数值逐位一致"的硬锚点自检。
# 锚点是针对保密真实数据的回归保护；示例数据不可能复现论文数字，故只告警不终止。
EXAMPLE_MODE = os.environ.get("EXAMPLE_MODE", "0") == "1"
