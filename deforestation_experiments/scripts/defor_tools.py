"""The environment of the existing tools (riskmapjnr, forestatrisk), which run in their own venv.

tool_comparison.py runs them as subprocesses of .venv_tools (GDAL, forestatrisk and riskmapjnr apart
from the main environment) with this interpreter and environment.
"""

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import defor_data as DD                                                     # noqa: E402

TOOLS_PYTHON = DD.ROOT / ".venv_tools" / "Scripts" / "python.exe"
_OSGEO = DD.ROOT / ".venv_tools" / "Lib" / "site-packages" / "osgeo" / "data"
# The machine-wide PROJ_LIB (PostGIS) points to a missing folder; use the GDAL wheel's own data.
TOOLS_ENV = {**{k: v for k, v in os.environ.items() if k not in ("PROJ_LIB", "PROJ_DATA", "GDAL_DATA")},
             "PROJ_DATA": str(_OSGEO / "proj"), "PROJ_LIB": str(_OSGEO / "proj"), "GDAL_DATA": str(_OSGEO / "gdal"),
             "PYTHONIOENCODING": "utf-8", "MPLBACKEND": "Agg"}   # not the notebook kernel's inline backend
