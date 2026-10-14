# landuse_tool/__init__.py
"""
Land Use Monitoring and Prediction Tool

A framework for fetching multitemporal geospatial data, analysing land-cover
change, and simulating future land use with machine learning. Azure-ready.

Modules:
- config: Configuration, dataset catalog, and credential specs
- azure_utils: Key Vault secret loading + Blob Storage cache
- api_connector: External data APIs (GEE, OpenTopography, WorldPop, OSM, ...)
- utils: Raster processing utilities (resampling, masking, terrain derivatives)
- data_loader: Data loading, grid alignment, vector rasterization, sampling
- change_analysis: Markov chain and trend analysis
- training: Model training with stratified sampling and growth modes
- scenarios: Scenario building and modification
- prediction: Two-stage suitability generation and allocation
- visualization: Interactive maps and data export
"""

from . import config
from . import azure_utils
from . import api_connector
from . import utils
from . import data_loader
from . import change_analysis
from . import training
from . import scenarios
from . import prediction
from . import visualization

__version__ = '2.0.0'
__author__ = 'SDG AI Lab'

__all__ = [
    'config',
    'azure_utils',
    'api_connector',
    'utils',
    'data_loader',
    'change_analysis',
    'training',
    'scenarios',
    'prediction',
    'visualization',
]
