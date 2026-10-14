"""
Deforestation Analysis Module.

A forest-loss / deforestation extension of the Land Use platform. Reuses the same
data connection (api_connector), grid alignment (data_loader), raster utilities
(utils) and configuration (config), and adds:

  config      - forest datasets (Hansen GFC, JRC TMF, MODIS NDVI/fire) + params
  forest      - forest baseline & masks (Hansen canopy or LULC forest classes)
  detection   - annual loss, LULC-transition deforestation, VI degradation, K-means
  stats       - areas, forest trajectory, rates, conversion matrix, VI trend, zonal
  drivers     - driver stack + deforestation-risk model + importances + risk map
  scenario    - future loss projection + risk-weighted allocation + policy levers
  pipeline    - orchestrator (run_from_arrays / run_from_files) + self-test

Call config.register() once at startup to add the annual datasets to the main
catalog so the existing fetch/align/cache pipeline handles them.
"""

from . import config
from . import forest
from . import detection
from . import stats
from . import drivers
from . import scenario
from . import pipeline

__all__ = ["config", "forest", "detection", "stats", "drivers", "scenario", "pipeline"]
