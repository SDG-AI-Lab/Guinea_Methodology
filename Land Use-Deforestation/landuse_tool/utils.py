import rasterio
import numpy as np
from rasterio.warp import calculate_default_transform, reproject, Resampling
from rasterio.features import rasterize
from scipy.ndimage import distance_transform_edt
from scipy.interpolate import griddata
import geopandas as gpd
import os
import logging

logger = logging.getLogger(__name__)

# ============================================================================
# RASTER PROJECTION & RESAMPLING
# ============================================================================

def reproject_raster(src_path, dst_crs, resampling=Resampling.bilinear):
    """
    Reproject a raster to a target CRS.
    
    Args:
        src_path (str): Path to source raster.
        dst_crs (str or dict): Target CRS (e.g., 'EPSG:4326').
        resampling: rasterio.warp.Resampling method.
    
    Returns:
        tuple: (np.ndarray reprojected data, dict updated profile)
    """
    with rasterio.open(src_path) as src:
        transform, width, height = calculate_default_transform(
            src.crs, dst_crs, src.width, src.height, *src.bounds
        )
        kwargs = src.meta.copy()
        kwargs.update({
            "crs": dst_crs,
            "transform": transform,
            "width": width,
            "height": height
        })

        dst = np.empty((height, width), dtype=src.meta["dtype"])

        reproject(
            source=rasterio.band(src, 1),
            destination=dst,
            src_transform=src.transform,
            src_crs=src.crs,
            dst_transform=transform,
            dst_crs=dst_crs,
            resampling=resampling
        )
    return dst, kwargs


def resample_raster(array, src_profile, ref_profile, resampling=Resampling.bilinear):
    """
    Resample a raster array to match a reference profile.
    
    Args:
        array (np.ndarray): Input raster array.
        src_profile (dict): Source raster profile.
        ref_profile (dict): Reference raster profile.
        resampling: rasterio.warp.Resampling method.
    
    Returns:
        np.ndarray: Resampled array.
    """
    dst = np.empty(
        (ref_profile["height"], ref_profile["width"]), 
        dtype=array.dtype
    )

    reproject(
        source=array,
        destination=dst,
        src_transform=src_profile["transform"],
        src_crs=src_profile["crs"],
        dst_transform=ref_profile["transform"],
        dst_crs=ref_profile["crs"],
        resampling=resampling
    )
    return dst


def align_rasters(raster_list, resampling=Resampling.bilinear):
    """
    Align multiple rasters to the same grid (CRS, resolution, extent).
    
    Args:
        raster_list: list of (array, profile) tuples
        resampling: resampling method
    
    Returns:
        list of (aligned_array, ref_profile) tuples
    """
    if not raster_list:
        raise ValueError("raster_list cannot be empty")
    
    # Use the first raster as reference
    ref_array, ref_profile = raster_list[0]
    aligned = [(ref_array, ref_profile)]

    for arr, prof in raster_list[1:]:
        if prof["crs"] != ref_profile["crs"]:
            logger.warning(
                f"CRS mismatch: {prof['crs']} vs {ref_profile['crs']}. Reprojecting..."
            )
        aligned_arr = resample_raster(arr, prof, ref_profile, resampling=resampling)
        aligned.append((aligned_arr, ref_profile))
    
    return aligned


# ============================================================================
# MISSING VALUE HANDLING
# ============================================================================

def identify_missing_values(array, nodata=None, invalid_values=None):
    """
    Identify all missing/invalid values in a raster.
    
    Args:
        array (np.ndarray): Input raster
        nodata: Nodata value(s) from raster metadata
        invalid_values (list): Additional invalid values (e.g., [254, 255])
    
    Returns:
        np.ndarray: Boolean mask (True = missing/invalid)
    """
    if invalid_values is None:
        invalid_values = [254, 255]
    
    missing_mask = np.zeros_like(array, dtype=bool)
    
    # Mark nodata values
    if nodata is not None:
        if isinstance(nodata, (list, tuple)):
            for nd in nodata:
                missing_mask |= (array == nd)
        else:
            missing_mask |= (array == nodata)
    
    # Mark invalid values
    for val in invalid_values:
        missing_mask |= (array == val)
    
    # Mark NaN values
    missing_mask |= np.isnan(array)
    
    return missing_mask


def handle_missing_values(array, method='mask', fill_value=np.nan, 
                         nodata=None, invalid_values=None):
    """
    Handle missing values in a raster according to specified method.
    
    Args:
        array (np.ndarray): Input raster
        method (str): Handling method
                     - 'mask': Replace with fill_value (default: NaN)
                     - 'mean': Replace with local mean (window-based)
                     - 'median': Replace with local median
                     - 'nearest': Interpolate from nearest valid value
                     - 'keep': Keep as-is (no processing)
        fill_value: Value to use for 'mask' method (default: NaN)
        nodata: Nodata value from raster metadata
        invalid_values (list): Additional invalid values to mark as missing
    
    Returns:
        tuple: (processed_array, missing_mask)
        - processed_array: Array with missing values handled
        - missing_mask: Boolean mask of original missing locations
    """
    
    # Identify missing values
    missing_mask = identify_missing_values(array, nodata, invalid_values)
    n_missing = np.sum(missing_mask)
    
    if n_missing == 0:
        logger.info("No missing values detected")
        return array.copy(), missing_mask
    
    missing_percent = 100 * n_missing / array.size
    logger.info(f"Detected {n_missing} missing values ({missing_percent:.2f}%)")
    
    result = array.copy().astype(np.float32)
    
    if method == 'keep':
        logger.info("Keeping missing values as-is (may contain NaN/nodata)")
        return result, missing_mask
    
    elif method == 'mask':
        result[missing_mask] = fill_value
        logger.info(f"Replaced {n_missing} missing values with {fill_value}")
        return result, missing_mask
    
    elif method == 'mean':
        # Local mean imputation using 3x3 window
        from scipy.ndimage import generic_filter
        def local_mean(x):
            valid = x[~np.isnan(x)]
            return np.mean(valid) if len(valid) > 0 else np.nan
        
        # Replace missing with NaN first
        temp = result.copy()
        temp[missing_mask] = np.nan
        
        # Apply local mean filter
        filled = generic_filter(temp, local_mean, size=3, mode='constant', 
                               cval=np.nan)
        result[missing_mask] = filled[missing_mask]
        logger.info(f"Applied local mean imputation for {n_missing} pixels")
        return result, missing_mask
    
    elif method == 'median':
        # Local median imputation
        from scipy.ndimage import generic_filter
        def local_median(x):
            valid = x[~np.isnan(x)]
            return np.median(valid) if len(valid) > 0 else np.nan
        
        temp = result.copy()
        temp[missing_mask] = np.nan
        filled = generic_filter(temp, local_median, size=3, mode='constant', 
                               cval=np.nan)
        result[missing_mask] = filled[missing_mask]
        logger.info(f"Applied local median imputation for {n_missing} pixels")
        return result, missing_mask
    
    elif method == 'nearest':
        # Interpolate from nearest valid values
        rows, cols = np.meshgrid(np.arange(array.shape[0]), 
                                 np.arange(array.shape[1]), indexing='ij')
        valid_mask = ~missing_mask
        
        if np.sum(valid_mask) == 0:
            logger.warning("No valid values to interpolate from!")
            result[missing_mask] = fill_value
            return result, missing_mask
        
        # Get coordinates of valid pixels
        valid_points = np.column_stack((rows[valid_mask], cols[valid_mask]))
        valid_values = result[valid_mask]
        
        # Get coordinates of missing pixels
        missing_points = np.column_stack((rows[missing_mask], cols[missing_mask]))
        
        # Interpolate
        filled_values = griddata(valid_points, valid_values, missing_points, 
                                method='nearest')
        result[missing_mask] = filled_values
        logger.info(f"Applied nearest-neighbor interpolation for {n_missing} pixels")
        return result, missing_mask
    
    else:
        logger.warning(f"Unknown method: {method}. Using 'mask' instead")
        result[missing_mask] = fill_value
        return result, missing_mask


def get_missing_value_stats(array, nodata=None, invalid_values=None):
    """
    Get statistics about missing values in a raster.
    
    Args:
        array (np.ndarray): Input raster
        nodata: Nodata value
        invalid_values (list): Additional invalid values
    
    Returns:
        dict: Missing value statistics
    """
    missing_mask = identify_missing_values(array, nodata, invalid_values)
    n_missing = np.sum(missing_mask)
    total = array.size
    
    stats = {
        'n_missing': int(n_missing),
        'n_valid': int(total - n_missing),
        'total_pixels': int(total),
        'missing_percent': float(100 * n_missing / total),
        'valid_percent': float(100 * (total - n_missing) / total),
        'is_empty': n_missing == total,
        'has_valid_data': n_missing < total
    }
    
    return stats


# ============================================================================
# MASKING & VALIDITY
# ============================================================================

def create_mask(array, nodata=None, invalid_values=None):
    """
    Create a boolean mask for valid pixels.
    
    Args:
        array: np.ndarray input raster
        nodata: nodata value(s) to exclude
        invalid_values: list of additional values to exclude (e.g., [254, 255])
    
    Returns:
        np.ndarray: Boolean mask (True = valid, False = invalid)
    """
    if invalid_values is None:
        invalid_values = [254, 255]
    
    # Start with all valid
    mask = np.ones_like(array, dtype=bool)
    
    # Mark nodata
    if nodata is not None:
        if isinstance(nodata, (list, tuple)):
            for nd in nodata:
                mask &= array != nd
        else:
            mask &= array != nodata
    
    # Mark invalid values
    for val in invalid_values:
        mask &= array != val
    
    # Mark NaN
    mask &= ~np.isnan(array)
    
    return mask


def apply_boundary_mask(raster_array, boundary_mask):
    """
    Apply an administrative boundary mask to a raster.
    Sets pixels outside the boundary to NaN.
    
    Args:
        raster_array (np.ndarray): Input raster data
        boundary_mask (np.ndarray): Boolean mask (True = inside, False = outside)
    
    Returns:
        np.ndarray: Masked array with NaN outside boundaries
    """
    result = raster_array.astype(np.float32).copy()
    result[~boundary_mask] = np.nan
    return result


# ============================================================================
# VECTOR PROCESSING
# ============================================================================

def process_vector_layer(vector_path, profile, mask, mode='distance', 
                         attribute=None, max_distance=None,
                         missing_value_handling='mask'):
    """
    Robust vector processor supporting multiple rasterization modes.
    
    Args:
        vector_path (str): Path to vector file (.shp, .geojson, .gpkg)
        profile (dict): Rasterio profile (width, height, transform, crs)
        mask (np.ndarray): Valid data mask (True = inside region)
        mode (str): 'distance', 'heatmap', or 'zoning'
        attribute (str): Attribute name for 'zoning' mode
        max_distance (float): Maximum distance for distance mode (None = unlimited)
        missing_value_handling (str): How to handle missing values in output
                                     ('mask', 'mean', 'nearest', 'keep')
    
    Returns:
        np.ndarray: Processed raster with NaN outside mask
    """
    if not os.path.exists(vector_path):
        logger.warning(f"Vector file not found: {vector_path}")
        return np.full((profile['height'], profile['width']), np.nan, dtype=np.float32)

    try:
        # 1. Load vector data
        gdf = gpd.read_file(vector_path)
        
        if gdf.empty:
            logger.warning(f"Vector file is empty: {vector_path}")
            return np.full((profile['height'], profile['width']), np.nan, dtype=np.float32)

        # 2. Fix invalid geometries
        gdf['geometry'] = gdf.geometry.make_valid()
        gdf = gdf[gdf.geometry.is_valid]
        
        if gdf.empty:
            logger.warning(f"No valid geometries in: {vector_path}")
            return np.full((profile['height'], profile['width']), np.nan, dtype=np.float32)

        # 3. Align projection
        if gdf.crs != profile['crs']:
            gdf = gdf.to_crs(profile['crs'])

        # 4. Rasterize
        if mode == 'zoning' and attribute and attribute in gdf.columns:
            shapes = ((geom, val) for geom, val in zip(gdf.geometry, gdf[attribute]))
        else:
            shapes = ((geom, 1) for geom in gdf.geometry)

        rasterized = rasterize(
            shapes=shapes,
            out_shape=(profile['height'], profile['width']),
            transform=profile['transform'],
            fill=0,
            all_touched=True,
            dtype='float32'
        )

        # 5. Spatial operations
        if mode == 'distance':
            # Distance from polygons/lines
            result = distance_transform_edt(rasterized == 0).astype(np.float32)
            
            # Clip to max_distance if specified
            if max_distance is not None:
                result = np.minimum(result, max_distance)
                
        elif mode == 'heatmap':
            # Exponential decay heatmap
            dist = distance_transform_edt(rasterized == 0).astype(np.float32)
            max_dist = dist.max() if dist.max() > 0 else 1.0
            decay_factor = max_dist * 0.1 if max_dist > 0 else 1.0
            result = np.exp(-dist / decay_factor)
            
        elif mode == 'zoning':
            # Direct rasterization with attribute values
            result = rasterized
            
        else:
            raise ValueError(f"Unknown mode: {mode}")

        # 6. Handle missing values in result
        if missing_value_handling != 'keep':
            result, _ = handle_missing_values(
                result, method=missing_value_handling, 
                nodata=0, invalid_values=[]
            )
        
        # 7. Apply boundary mask
        result = np.where(mask, result, np.nan)
        
        return result

    except Exception as e:
        logger.error(f"Error processing vector layer {vector_path}: {e}")
        return np.full((profile['height'], profile['width']), np.nan, dtype=np.float32)


def rasterize_points_as_heatmap(point_gdf, profile, mask, decay_factor=0.1,
                                missing_value_handling='mask'):
    """
    Convert point features to a heatmap using distance decay.
    
    Args:
        point_gdf (GeoDataFrame): Points to rasterize
        profile (dict): Rasterio profile
        mask (np.ndarray): Valid data mask
        decay_factor (float): Decay parameter for exponential function
        missing_value_handling (str): How to handle missing values
    
    Returns:
        np.ndarray: Heatmap raster
    """
    if point_gdf.crs != profile['crs']:
        point_gdf = point_gdf.to_crs(profile['crs'])
    
    # Rasterize points
    shapes = ((geom, 1) for geom in point_gdf.geometry)
    rasterized = rasterize(
        shapes=shapes,
        out_shape=(profile['height'], profile['width']),
        transform=profile['transform'],
        fill=0,
        dtype='float32'
    )
    
    # Create distance-based heatmap
    dist = distance_transform_edt(rasterized == 0).astype(np.float32)
    max_dist = dist.max() if dist.max() > 0 else 1.0
    heatmap = np.exp(-dist / (max_dist * decay_factor))
    
    # Handle missing values
    if missing_value_handling != 'keep':
        heatmap, _ = handle_missing_values(
            heatmap, method=missing_value_handling,
            nodata=0, invalid_values=[]
        )
    
    return np.where(mask, heatmap, np.nan)


# ============================================================================
# TERRAIN DERIVATIVES
# ============================================================================

def calculate_slope(dem_array, profile, missing_value_handling='mean'):
    """
    Calculate slope from DEM using Zevenbergen & Thorne (1987) method.
    
    Args:
        dem_array (np.ndarray): Digital elevation model
        profile (dict): Rasterio profile (contains transform)
        missing_value_handling (str): How to handle missing values in DEM
    
    Returns:
        np.ndarray: Slope in degrees
    """
    # Handle missing values first
    dem_filled, _ = handle_missing_values(
        dem_array, method=missing_value_handling,
        nodata=profile.get('nodata')
    )
    
    # Get pixel size (assuming square pixels)
    pixel_size_x = abs(profile['transform'].a)
    pixel_size_y = abs(profile['transform'].e)
    
    # Pad DEM to handle edges
    dem_padded = np.pad(dem_filled, 1, mode='edge')
    
    # Zevenbergen & Thorne method
    x = (dem_padded[2:, 0:-2] - dem_padded[2:, 2:]) / (2 * pixel_size_x)
    y = (dem_padded[0:-2, 1:-1] - dem_padded[2:, 1:-1]) / (2 * pixel_size_y)
    
    slope = np.arctan(np.sqrt(x**2 + y**2))
    slope_degrees = np.degrees(slope)
    
    logger.info(f"Calculated slope: min={np.nanmin(slope_degrees):.2f}°, "
               f"max={np.nanmax(slope_degrees):.2f}°")
    
    return slope_degrees


def calculate_aspect(dem_array, profile, missing_value_handling='mean'):
    """
    Calculate aspect from DEM.
    
    Args:
        dem_array (np.ndarray): Digital elevation model
        profile (dict): Rasterio profile
        missing_value_handling (str): How to handle missing values in DEM
    
    Returns:
        np.ndarray: Aspect in degrees (0-360, -1 for flat areas)
    """
    # Handle missing values
    dem_filled, _ = handle_missing_values(
        dem_array, method=missing_value_handling,
        nodata=profile.get('nodata')
    )
    
    pixel_size_x = abs(profile['transform'].a)
    pixel_size_y = abs(profile['transform'].e)
    
    # Pad DEM
    dem_padded = np.pad(dem_filled, 1, mode='edge')
    
    # Calculate gradients
    x = (dem_padded[2:, 0:-2] - dem_padded[2:, 2:]) / (2 * pixel_size_x)
    y = (dem_padded[0:-2, 1:-1] - dem_padded[2:, 1:-1]) / (2 * pixel_size_y)
    
    # Calculate aspect
    aspect = np.arctan2(x, -y)
    aspect = np.degrees(aspect)
    aspect = (aspect + 360) % 360
    
    # Set flat areas to -1
    slope_check = np.sqrt(x**2 + y**2)
    aspect[slope_check < 0.00001] = -1
    
    logger.info(f"Calculated aspect: flat areas={np.sum(aspect == -1)}")
    
    return aspect


# ============================================================================
# VALIDATION & REPORTING
# ============================================================================

def validate_raster_alignment(rasters_dict, profile_key=None):
    """
    Validate that all rasters have matching dimensions and CRS.
    
    Args:
        rasters_dict (dict): Dictionary of {name: (array, profile)}
        profile_key (str): Reference profile key (default: first key)
    
    Returns:
        tuple: (is_valid, error_messages)
    """
    if not rasters_dict:
        return False, ["Empty rasters dictionary"]
    
    errors = []
    keys = list(rasters_dict.keys())
    
    if profile_key is None:
        profile_key = keys[0]
    
    ref_array, ref_profile = rasters_dict[profile_key]
    ref_shape = ref_array.shape
    ref_crs = ref_profile.get('crs')
    ref_transform = ref_profile.get('transform')
    
    for key in keys:
        if key == profile_key:
            continue
        
        arr, prof = rasters_dict[key]
        
        if arr.shape != ref_shape:
            errors.append(
                f"{key}: Shape mismatch {arr.shape} vs {ref_shape}"
            )
        
        if prof.get('crs') != ref_crs:
            errors.append(
                f"{key}: CRS mismatch {prof.get('crs')} vs {ref_crs}"
            )
    
    return len(errors) == 0, errors


def get_raster_statistics(array, mask=None, nodata=None, invalid_values=None):
    """
    Get statistics for a raster, optionally masked.
    
    Args:
        array (np.ndarray): Raster data
        mask (np.ndarray): Valid data mask
        nodata: Nodata value(s)
        invalid_values (list): Additional invalid values
    
    Returns:
        dict: Statistics (min, max, mean, std, valid_count, valid_percent, 
              missing_count, missing_percent)
    """
    valid_mask = create_mask(array, nodata=nodata, invalid_values=invalid_values)
    
    if mask is not None:
        valid_mask &= mask
    
    valid_data = array[valid_mask]
    missing_count = array.size - np.sum(valid_mask)
    
    stats = {
        'min': float(np.nanmin(valid_data)) if len(valid_data) > 0 else np.nan,
        'max': float(np.nanmax(valid_data)) if len(valid_data) > 0 else np.nan,
        'mean': float(np.nanmean(valid_data)) if len(valid_data) > 0 else np.nan,
        'std': float(np.nanstd(valid_data)) if len(valid_data) > 0 else np.nan,
        'median': float(np.nanmedian(valid_data)) if len(valid_data) > 0 else np.nan,
        'valid_count': int(np.sum(valid_mask)),
        'total_count': array.size,
        'valid_percent': float(100 * np.sum(valid_mask) / array.size),
        'missing_count': int(missing_count),
        'missing_percent': float(100 * missing_count / array.size)
    }
    
    return stats
