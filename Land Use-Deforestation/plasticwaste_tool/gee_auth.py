"""
Earth Engine initialisation, following this project's established convention.

WHY THIS EXISTS
---------------
`landuse_tool.config` reads the service-account key from the `GEE_SA_KEY`
environment variable and expects it to hold the **contents** of the key JSON
(config.py:50 says so explicitly). This project's `.env` instead sets
`GEE_SA_KEY` to a *file path*. The result is a silent failure:
`initialize_gee()` cannot parse a path as PEM, falls through to a bare
`ee.Initialize()`, and dies with the unhelpful "no project found".

Every working notebook in this repo therefore does the same thing first —
reads the key file and overwrites `GEE_SA_KEY` with its contents before
calling `initialize_gee()`. See `landuse_analysis_improved.ipynb` cell 2,
which carries the note:

    do NOT rely on a hand-pasted GEE_SA_KEY value in .env
    (that breaks PEM parsing; confirmed root cause ...)

This module is that same shim, in one place, so the plastic-characterisation
scripts do not each re-implement it and so nothing in `landuse_tool/` or the
existing `plasticwaste_tool/` FDI code has to be edited.

ORDER MATTERS
-------------
`landuse_tool.config.PROCESSING_CONFIG` is built from `os.environ` at IMPORT
time and then frozen. So `GEE_SA_KEY` must be set **before** `landuse_tool` is
imported. That is why the import sits inside the function rather than at module
scope — importing it at the top would freeze the config before the key is
loaded and reintroduce exactly the bug this module exists to avoid.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_KEY = PROJECT_ROOT / "landuse_tool" / "credentials" / \
    "tribal-mapper-275801-7b57ce84f3bd.json"


def _resolve_key_path() -> Optional[Path]:
    """Find the service-account JSON, preferring whatever .env points at."""
    for var in ("GEE_SA_KEY_FILE", "GEE_SA_KEY"):
        raw = os.environ.get(var)
        if not raw or "PRIVATE KEY" in raw:  # already contents, not a path
            continue
        p = Path(raw)
        if not p.is_absolute():
            p = PROJECT_ROOT / raw.lstrip("./").lstrip(".\\")
        if p.exists():
            return p
    return DEFAULT_KEY if DEFAULT_KEY.exists() else None


def init_gee_project_convention(verbose: bool = True) -> bool:
    """
    Initialise Earth Engine the way this project's notebooks do.

    Returns True on success. Safe to call more than once.
    """
    from dotenv import load_dotenv
    load_dotenv(dotenv_path=PROJECT_ROOT / ".env", override=True)

    key_path = _resolve_key_path()
    if key_path is None:
        if verbose:
            print("  ! no service-account key found (GEE_SA_KEY_FILE / GEE_SA_KEY)")
        return False

    # The load-bearing line: contents, not a path.
    os.environ["GEE_SA_KEY"] = key_path.read_text(encoding="utf-8")

    import sys
    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT))

    from landuse_tool.azure_utils import load_secrets_into_env
    load_secrets_into_env()

    from landuse_tool import api_connector
    ok = bool(api_connector.initialize_gee())
    if verbose:
        print(f"  GEE initialised via landuse_tool.api_connector: {ok} "
              f"(key: {key_path.name})")
    return ok
