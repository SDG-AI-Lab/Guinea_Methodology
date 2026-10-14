"""
Scenarios Module

Handles scenario building for future land cover simulations:
1. Modify predictor layers (arithmetic operations)
2. Upload/add new future predictor layers
3. Override policy demand (transition pixel counts)
4. Enable stochastic simulation for uncertainty analysis
"""

import rasterio
import numpy as np
import os
import tempfile
from rasterio.windows import Window
from scipy.ndimage import uniform_filter
import logging

from .data_loader import _open_as_raster
from .utils import handle_missing_values, get_raster_statistics
from . import config

logger = logging.getLogger(__name__)

# ============================================================================
# PREDICTOR LAYER MODIFICATION
# ============================================================================

def modify_predictor_raster(input_raster_file, operator, value, temp_dir, progress_callback=None):
    if progress_callback is None:
        def progress_callback(frac, text): pass

    """
    Applies a simple arithmetic operation to a raster and saves it as a new
    temporary file. This is done in a memory-safe, windowed manner.
    
    Useful for scenario analysis:
    - "Multiply by 1.5" for increased development pressure
    - "Add 100" for new infrastructure
    - "Subtract 50" for conservation areas
    
    Args:
        input_raster_file: Streamlit UploadedFile or file path
        operator (str): 'Multiply', 'Add', 'Subtract', or 'Divide'
        value (float): Operand value
        temp_dir (str): Directory to save output
    
    Returns:
        str: Path to modified raster, or None if failed
    """
    try:
        # Create a unique name for the new temporary file
        file_name = input_raster_file.name if hasattr(input_raster_file, 'name') else os.path.basename(input_raster_file)
        output_name = f"scenario_{operator.lower()}_{value}_{file_name}"
        output_path = os.path.join(temp_dir, output_name)
        
        logger.info(f"Modifying raster: {file_name} with {operator} {value}")

        with _open_as_raster(input_raster_file) as src:
            profile = src.profile
            profile.update(dtype='float32')  # Update to float to handle decimals
            original_nodata = src.nodata
            
            with rasterio.open(output_path, 'w', **profile) as dst:
                # Process the raster in chunks (windows)
                total_blocks = sum(1 for _ in src.block_windows(1))
                
                for block_idx, (block_id, window) in enumerate(src.block_windows(1)):
                    window_data = src.read(1, window=window).astype('float32')
                    
                    # Apply the selected operation
                    if operator == 'Multiply':
                        result_data = window_data * value
                    elif operator == 'Add':
                        result_data = window_data + value
                    elif operator == 'Subtract':
                        result_data = window_data - value
                    elif operator == 'Divide':
                        if value != 0:
                            result_data = np.where(window_data != original_nodata, 
                                                 window_data / value, 
                                                 original_nodata)
                        else:
                            logger.warning("⚠️ Division by zero attempted. Using original data.")
                            result_data = window_data
                    else:
                        result_data = window_data
                    
                    dst.write(result_data, 1, window=window)
                    progress_callback((block_idx + 1) / total_blocks, f"Modifying raster {operator} {value}...")
        
        logger.info(f"✅ Created modified raster: {output_path}")
        return output_path
    
    except Exception as e:
        logger.error(f"Predictor modification error: {e}", exc_info=True)
        return None


def upload_future_predictors(uploaded_files, temp_dir, progress_callback=None):
    if progress_callback is None:
        def progress_callback(frac, text): pass
    """
    Copies uploaded future predictor files to the working directory.
    
    Args:
        uploaded_files (list): List of Streamlit UploadedFile objects
        temp_dir (str): Directory to save files
    
    Returns:
        list: List of saved file paths
    """
    saved_paths = []
    
    for f in uploaded_files:
        try:
            output_path = os.path.join(temp_dir, f"future_{f.name}")
            
            with open(output_path, 'wb') as out_f:
                f.seek(0)
                out_f.write(f.read())
            
            logger.info(f"✅ Saved future predictor: {output_path}")
            saved_paths.append(output_path)
        
        except Exception as e:
            logger.error(f"Error saving future predictor: {e}")
    
    return saved_paths


# ============================================================================
# NEIGHBORHOOD PREDICTORS (Shared with training)
# ============================================================================

def generate_neighborhood_predictors(base_raster_file, temp_dir, radius_pixels=5, output_prefix='neighborhood', progress_callback=None):
    if progress_callback is None:
        def progress_callback(frac, text): pass

    """
    Generates neighborhood-based predictor rasters (e.g., class density).
    
    Used for:
    - Training: Spatially-Aware AI feature
    - Scenarios: Apply same neighborhood logic to future predictors
    
    Features:
    - Fast uniform filter for efficient neighborhood analysis
    - Generates one predictor per land cover class
    - Memory-safe windowed processing
    
    Args:
        base_raster_file: Land cover raster file (or Streamlit UploadedFile)
        temp_dir (str): Directory to save output files
        radius_pixels (int): Neighborhood radius in pixels
        output_prefix (str): Prefix for output filenames ('neighborhood' or 'scenario_neighborhood')
    
    Returns:
        list: List of paths to generated neighborhood predictor rasters
        
    Example:
        >>> predictor_paths = generate_neighborhood_predictors(lc_file, 'temp/', radius_pixels=5)
        >>> # Returns: ['temp/neighborhood_class_1.tif', 'temp/neighborhood_class_2.tif', ...]
    """
    new_predictor_paths = []
    
    try:
        with _open_as_raster(base_raster_file) as src:
            profile = src.profile
            profile.update(dtype='float32', count=1)
            lc_arr = src.read(1)
            nodata = src.nodata
            
            # Get unique classes (excluding nodata)
            if nodata is not None:
                valid_mask = (lc_arr != nodata) & np.isfinite(lc_arr)
                unique_classes = np.unique(lc_arr[valid_mask])
            else:
                unique_classes = np.unique(lc_arr)
            
            logger.info(f"Generating neighborhood predictors for {len(unique_classes)} classes, "
                       f"radius={radius_pixels} pixels")
        
        # The size of the filter window is (2 * radius + 1)
        filter_size = 2 * radius_pixels + 1
        
        for class_idx, cls in enumerate(unique_classes):
            # Create a binary map for the current class
            binary_map = (lc_arr == cls).astype('float32')
            
            # Use a uniform filter to calculate the density (proportion) in the neighborhood
            # This is a fast and efficient way to perform a moving window analysis
            density_map = uniform_filter(binary_map, size=filter_size, mode='constant', cval=0)
            
            # Handle missing values in output
            if nodata is not None:
                density_map[~valid_mask] = np.nan
            
            # Save the new raster
            output_path = os.path.join(temp_dir, f'{output_prefix}_class_{int(cls)}.tif')
            with rasterio.open(output_path, 'w', **profile) as dst:
                dst.write(density_map, 1)
            
            logger.info(f"Generated: {output_path}")
            new_predictor_paths.append(output_path)
            progress_callback((class_idx + 1) / len(unique_classes),
                              f"Generated density map for Class {int(cls)}")
        
        logger.info(f"✅ Generated {len(new_predictor_paths)} neighborhood predictors")
        return new_predictor_paths
    
    except Exception as e:
        logger.error(f"Neighborhood generation error: {e}", exc_info=True)
        return []


# ============================================================================
# POLICY DEMAND OVERRIDE
# ============================================================================

def override_transition_demand(transition_counts_df, policy_edits_dict):
    """
    Override historical transition demand with policy-driven targets.
    
    Allows users to define desired land use change allocations that differ
    from historical trends (e.g., increase forest, reduce urban sprawl).
    
    Args:
        transition_counts_df (pd.DataFrame): Original transition pixel counts
        policy_edits_dict (dict): {(from_cls, to_cls): new_pixel_count}
    
    Returns:
        pd.DataFrame: Modified transition counts with policy overrides applied
    """
    modified_counts = transition_counts_df.copy()
    
    logger.info(f"Applying policy overrides to {len(policy_edits_dict)} transitions")
    
    for (from_cls, to_cls), new_count in policy_edits_dict.items():
        try:
            old_count = modified_counts.loc[from_cls, to_cls]
            modified_counts.loc[from_cls, to_cls] = new_count
            logger.info(f"Policy override: {from_cls}→{to_cls}: {old_count} → {new_count} pixels")
        except KeyError:
            logger.warning(f"Transition {from_cls}→{to_cls} not found in transition matrix")
    
    return modified_counts


def validate_policy_demand(transition_counts_df, max_change_percent=100):
    """
    Validate that policy-driven demands are realistic.
    
    Args:
        transition_counts_df (pd.DataFrame): Transition counts to validate
        max_change_percent (float): Maximum % change from historical (0-1000)
    
    Returns:
        tuple: (is_valid, warnings_list)
    """
    warnings = []
    
    # Check if any transition demand is extremely high
    max_single_transition = transition_counts_df.max().max()
    if max_single_transition > 1e6:
        warnings.append(f"⚠️ Very high transition demand detected: {max_single_transition} pixels")
    
    # Check if total change is physically possible
    total_change = transition_counts_df.sum().sum() - np.diag(transition_counts_df).sum()
    if total_change < 0:
        warnings.append("❌ Negative total change detected (physically impossible)")
        return False, warnings
    
    return True, warnings


# ============================================================================
# STOCHASTIC SIMULATION SETUP
# ============================================================================

def configure_stochastic_parameters(n_simulations=5, randomness_factor=0.1):
    """
    Configure parameters for stochastic (probabilistic) simulation.
    
    Args:
        n_simulations (int): Number of independent simulations to run
        randomness_factor (float): Scale of stochastic variation (0-1)
                                  Higher = more variation between runs
    
    Returns:
        dict: Stochastic configuration
    """
    config = {
        'n_simulations': max(1, min(n_simulations, 100)),  # Clamp to 1-100
        'randomness_factor': np.clip(randomness_factor, 0.0, 1.0),
        'allocation_method': 'probabilistic',  # vs. deterministic
        'uncertainty_threshold': 0.3,  # Pixels with suitability between 0.3-0.7 are uncertain
        'seed': np.random.randint(0, 2**31)  # Random seed for reproducibility
    }
    
    logger.info(f"Configured stochastic simulation: {n_simulations} runs, "
               f"randomness={randomness_factor}")
    
    return config


# ============================================================================
# SCENARIO PACKAGE CREATION
# ============================================================================

def create_scenario_package(scenario_name, transition_counts, predictor_files,
                           stochastic_config=None, policy_edits=None, temp_dir=None):
    """
    Package all scenario settings into a structured format for simulation.
    
    Args:
        scenario_name (str): Name of the scenario (e.g., "Green Growth 2030")
        transition_counts (pd.DataFrame): Transition pixel counts
        predictor_files (list): List of predictor file paths to use
        stochastic_config (dict): Stochastic parameters (optional)
        policy_edits (dict): Policy override edits (optional)
        temp_dir (str): Directory to save scenario package (default: config output dir)
    
    Returns:
        dict: Complete scenario configuration
    """
    import json
    from datetime import datetime

    # Azure-safe default under the configured output dir (/tmp/...).
    if temp_dir is None:
        temp_dir = os.path.join(config.PROCESSING_CONFIG['output_dir'], 'scenarios')
    os.makedirs(temp_dir, exist_ok=True)
    
    scenario = {
        'metadata': {
            'name': scenario_name,
            'timestamp': datetime.now().isoformat(),
            'version': '1.0'
        },
        'transition_demand': {
            'type': 'policy_override' if policy_edits else 'historical',
            'data': transition_counts.to_dict() if hasattr(transition_counts, 'to_dict') else transition_counts,
            'policy_edits': policy_edits or {}
        },
        'predictors': {
            'files': predictor_files,
            'count': len(predictor_files)
        },
        'stochastic': stochastic_config or {},
        'simulation_type': 'stochastic' if stochastic_config else 'deterministic'
    }
    
    # Save scenario configuration
    os.makedirs(temp_dir, exist_ok=True)
    scenario_path = os.path.join(temp_dir, f"scenario_{scenario_name.replace(' ', '_')}.json")
    
    try:
        with open(scenario_path, 'w') as f:
            json.dump(scenario, f, indent=4, default=str)
        logger.info(f"✅ Saved scenario package: {scenario_path}")
    except Exception as e:
        logger.error(f"Failed to save scenario package: {e}")
    
    return scenario


# ============================================================================
# SCENARIO VALIDATION & REPORTING
# ============================================================================

def validate_scenario(scenario_dict):
    """
    Validate scenario configuration for simulation readiness.
    
    Args:
        scenario_dict (dict): Scenario configuration
    
    Returns:
        tuple: (is_valid, error_messages, warning_messages)
    """
    errors = []
    warnings = []
    
    # Check transition demand
    if not scenario_dict.get('transition_demand'):
        errors.append("Missing transition demand configuration")
    
    # Check predictors
    predictors = scenario_dict.get('predictors', {}).get('files', [])
    if len(predictors) == 0:
        warnings.append("No predictor files specified")
    
    # Check for file existence
    for pred_file in predictors:
        if isinstance(pred_file, str) and not os.path.exists(pred_file):
            errors.append(f"Predictor file not found: {pred_file}")
    
    # Check stochastic config if enabled
    if scenario_dict.get('simulation_type') == 'stochastic':
        stoch = scenario_dict.get('stochastic', {})
        if stoch.get('n_simulations', 0) < 1:
            errors.append("Invalid number of stochastic simulations")
    
    is_valid = len(errors) == 0
    return is_valid, errors, warnings


def generate_scenario_report(scenario_dict, output_path=None):
    """
    Generate a human-readable report of the scenario configuration.
    
    Args:
        scenario_dict (dict): Scenario configuration
        output_path (str): Optional path to save report as text file
    
    Returns:
        str: Formatted scenario report
    """
    report = []
    report.append("=" * 60)
    report.append(f"SCENARIO REPORT: {scenario_dict['metadata']['name']}")
    report.append(f"Created: {scenario_dict['metadata']['timestamp']}")
    report.append("=" * 60)
    
    report.append("\n📊 TRANSITION DEMAND")
    report.append(f"  Type: {scenario_dict['transition_demand']['type']}")
    if scenario_dict['transition_demand']['policy_edits']:
        report.append(f"  Policy Edits: {len(scenario_dict['transition_demand']['policy_edits'])}")
    
    report.append("\n🗂️  PREDICTORS")
    report.append(f"  Count: {scenario_dict['predictors']['count']}")
    for i, pred in enumerate(scenario_dict['predictors']['files'], 1):
        report.append(f"    {i}. {os.path.basename(pred) if isinstance(pred, str) else pred}")
    
    if scenario_dict.get('stochastic'):
        report.append("\n🎲 STOCHASTIC CONFIGURATION")
        stoch = scenario_dict['stochastic']
        report.append(f"  Simulations: {stoch.get('n_simulations', 'N/A')}")
        report.append(f"  Randomness: {stoch.get('randomness_factor', 'N/A')}")
        report.append(f"  Seed: {stoch.get('seed', 'N/A')}")
    
    report_text = "\n".join(report)
    
    if output_path:
        with open(output_path, 'w') as f:
            f.write(report_text)
        logger.info(f"Saved scenario report: {output_path}")
    
    return report_text
