"""
Land-cover class areas of one published land-cover map (feature 005;
specs/005-land-use-deforestation-views/contracts/toolchain.md §5).

    class_areas(cog_path, classes) -> {class_code: area_ha, ..., "no_data": area_ha}

The map is read window by window, so a national COG never has to fit in
memory. Areas are latitude-corrected pixel areas (the deforestation
toolchain's `stats.pixel_area_ha`), accepted as equal-area for this purpose
(doc 13 GIS-ADM-012). Nodata pixels are left out. A pixel whose value is not
one of `classes` (the classes the legend draws; clouds, 10, are not one) is
summed into `no_data`: it is not a land-cover class, and a chart must not
show it as one (spec FR-036b). A class of `classes` with no pixel is 0.0.
"""

from __future__ import annotations

import numpy as np

from .deforestation_tool.stats import pixel_area_ha


def class_areas(cog_path, classes: dict) -> dict:
    import rasterio
    from rasterio.windows import transform as window_transform

    codes = [int(code) for code in classes]
    totals = {code: 0.0 for code in codes}
    no_data = 0.0
    with rasterio.open(cog_path) as dataset:
        profile = {"transform": dataset.transform, "crs": dataset.crs,
                   "height": dataset.height, "width": dataset.width}
        nodata = dataset.nodata
        for _, window in dataset.block_windows(1):
            values = dataset.read(1, window=window)
            # The window's own rows: pixel area varies with latitude.
            area = pixel_area_ha({**profile, "transform": window_transform(window, dataset.transform)},
                                 shape=values.shape)
            area = np.broadcast_to(np.asarray(area, dtype="float64"), values.shape)
            valid = np.ones(values.shape, bool) if nodata is None else values != nodata
            counted = np.zeros(values.shape, bool)
            for code in codes:
                mask = valid & (values == code)
                totals[code] += float(area[mask].sum())
                counted |= mask
            no_data += float(area[valid & ~counted].sum())
    return {**totals, "no_data": no_data}
