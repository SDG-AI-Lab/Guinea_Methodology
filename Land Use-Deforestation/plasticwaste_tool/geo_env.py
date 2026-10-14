"""
Repair the PROJ data path before rasterio is used.

THE PROBLEM
-----------
This machine has a system-wide `PROJ_LIB` pointing at PostgreSQL's PostGIS
install:

    PROJ_LIB=C:\\Program Files\\PostgreSQL\\13\\share\\contrib\\postgis-3.2\\proj

That directory belongs to a different (older) PROJ build. rasterio picks the
environment variable up in preference to its own bundled `proj_data`, fails to
find the EPSG tables in it, and dies on the first CRS operation with:

    CRSError: The EPSG code is unknown.
              PROJ: proj_create_from_database: Cannot find proj.db

It is confusing because `pyproj` keeps working throughout — pyproj resolves its
data directory internally rather than through `PROJ_LIB`, so a quick
`Transformer.from_crs(...)` test passes and suggests the environment is fine.

THE FIX
-------
Point `PROJ_LIB`/`PROJ_DATA` at rasterio's own bundled copy, for this process
only. The user's system variable is left alone — PostGIS presumably needs it,
and changing a machine-wide setting to suit one script would be the wrong
trade.

Call `fix_proj()` BEFORE the first rasterio CRS operation. Importing rasterio
first is fine; it reads these variables lazily.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional


def fix_proj(verbose: bool = False) -> Optional[str]:
    """Point PROJ at rasterio's bundled database for this process. Returns the path used."""
    try:
        import rasterio
    except ImportError:
        return None

    bundled = Path(rasterio.__file__).parent / "proj_data"
    if not (bundled / "proj.db").exists():
        # Fall back to pyproj's copy, which is equally valid.
        try:
            import pyproj
            bundled = Path(pyproj.datadir.get_data_dir())
        except Exception:
            return None
        if not (bundled / "proj.db").exists():
            return None

    previous = os.environ.get("PROJ_LIB")
    os.environ["PROJ_LIB"] = str(bundled)
    os.environ["PROJ_DATA"] = str(bundled)

    # rasterio caches the PROJ search path inside its GDAL environment, so a
    # session that already touched rasterio needs the path pushed in directly.
    try:
        from rasterio._env import set_proj_data_search_path
        set_proj_data_search_path(str(bundled))
    except Exception:
        pass

    if verbose:
        print(f"PROJ_LIB: {previous} -> {bundled}")
    return str(bundled)
