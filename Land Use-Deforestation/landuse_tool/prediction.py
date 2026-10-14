"""
Prediction & Simulation Module

Handles two-stage prediction process:
1. Stage 2: Generate suitability atlas (pre-compute suitability maps)
2. Stage 3: Run allocation simulation (allocate projected change to suitable pixels)

This two-stage approach:
- Separates computationally intensive suitability generation from allocation
- Allows users to inspect/download suitability maps
- Enables multiple allocation runs with same suitability atlas
- Provides better progress feedback and recovery options
"""

import rasterio
import numpy as np
import tempfile
import os
import joblib
from contextlib import ExitStack
from rasterio.windows import Window
import gc
from scipy.ndimage import binary_dilation
import json
import logging

from .data_loader import _open_as_raster
from .utils import handle_missing_values, get_raster_statistics
from . import config

logger = logging.getLogger(__name__)

# ============================================================================
# STAGE 2: SUITABILITY GENERATION
# ============================================================================

def _generate_single_suitability_map(model_path, predictor_files, lc_end_file, 
                                    temp_dir, required_features, model_type='',
                                    missing_value_handling='mean'):
    """
    Generate a single suitability map for a transition using a memory-efficient,
    chunk-based processing strategy.
    
    Args:
        model_path (str): Path to trained Random Forest model (.joblib)
        predictor_files (list): List of predictor raster files
        lc_end_file: Reference land cover raster (defines extent)
        temp_dir (str): Directory to save output suitability map
        required_features (list): Feature names required by the model
        model_type (str): 'expander', 'patcher', or '' (standard)
        missing_value_handling (str): How to handle missing values in predictors
    
    Returns:
        str: Path to generated suitability map (GeoTIFF)
    """
    try:
        logger.info(f"Generating suitability map: {model_type or 'standard'} "
                   f"(model: {os.path.basename(model_path)})")
        
        # Load trained model
        model = joblib.load(model_path)
        logger.info(f"✅ Loaded model: {model_path}")
        
        # --- Align predictors with model requirements ---
        available_predictors = {
            os.path.basename(f.name if hasattr(f, 'name') else f): f 
            for f in predictor_files
        }
        
        aligned_predictor_files = []
        for feature_name in required_features:
            source = available_predictors.get(feature_name)
            if source:
                aligned_predictor_files.append(source)
            else:
                raise FileNotFoundError(
                    f"CRITICAL: Model requires predictor '{feature_name}', "
                    f"but not found in {list(available_predictors.keys())}"
                )
        
        logger.info(f"Aligned {len(aligned_predictor_files)} predictors with model requirements")
        
        # --- Extract class information from model path ---
        basename = os.path.basename(model_path).replace('.joblib', '')
        parts = basename.split('_')
        from_class = parts[-2]
        to_cls_str = parts[-1]
        
        if model_type:
            filename = f'suitability_{model_type}_{from_class}_to_{to_cls_str}.tif'
        else:
            filename = f'suitability_{from_class}_to_{to_cls_str}.tif'
        
        temp_filepath = os.path.join(temp_dir, filename)
        
        # --- Memory-efficient chunk-based processing ---
        with ExitStack() as stack:
            # Open all required predictor files
            predictors = [
                stack.enter_context(_open_as_raster(f)) 
                for f in aligned_predictor_files
            ]
            ref_src = stack.enter_context(_open_as_raster(lc_end_file))

            # GUARD: every predictor must share the reference land-cover grid.
            # Window reads below assume identical grids; a misaligned predictor
            # would either crash opaquely or silently pair wrong pixels with
            # wrong features. Fail with an explicit message instead (mirrors the
            # same guard in training.create_transition_dataset).
            for _pf, _psrc in zip(aligned_predictor_files, predictors):
                if (_psrc.height, _psrc.width) != (ref_src.height, ref_src.width):
                    raise ValueError(
                        f"Predictor '{_pf}' has grid "
                        f"{_psrc.height}x{_psrc.width}, but the reference land "
                        f"cover is {ref_src.height}x{ref_src.width}. Predictor "
                        f"files must be pre-aligned to the reference grid before "
                        f"suitability generation."
                    )

            # Get raster profile
            profile = ref_src.profile
            profile.update(dtype='float32', count=1, nodata=-1.0)
            
            # Initialize suitability map with -1 (no data)
            suitability_map = np.full(
                (profile['height'], profile['width']), 
                -1.0, dtype='float32'
            )
            
            logger.info(f"Processing {profile['width']}x{profile['height']} raster...")
            
            # --- Process raster in windows (chunks) ---
            total_blocks = sum(1 for _ in ref_src.block_windows(1))
            processed_pixels = 0
            
            for block_idx, (block_id, window) in enumerate(ref_src.block_windows(1)):
                # 1. Read land cover chunk
                lc_chunk = ref_src.read(1, window=window)
                
                # 2. Find pixels of interest (from_class)
                from_mask_chunk = (lc_chunk == int(from_class))
                
                # 3. Skip if no relevant pixels in this chunk
                if not np.any(from_mask_chunk):
                    continue
                
                # 4. Read predictor data for this chunk
                predictor_chunks = [p.read(1, window=window) for p in predictors]
                
                # 5. Create feature matrix
                # Stack: (n_predictors, height, width) -> (height, width, n_predictors)
                X_chunk = np.stack(predictor_chunks).transpose(1, 2, 0)
                
                # Extract only pixels of interest
                X_chunk = X_chunk[from_mask_chunk]  # Shape: (n_pixels, n_predictors)
                
                # 6. Handle missing values in features
                if missing_value_handling != 'keep':
                    for feat_idx in range(X_chunk.shape[1]):
                        feat_col = X_chunk[:, feat_idx]
                        missing_mask = np.isnan(feat_col) | ~np.isfinite(feat_col)
                        
                        if np.any(missing_mask):
                            if missing_value_handling == 'mask':
                                X_chunk[missing_mask, feat_idx] = 0.0
                            elif missing_value_handling == 'mean':
                                valid_mean = np.nanmean(feat_col)
                                if np.isnan(valid_mean):
                                    valid_mean = 0.0
                                X_chunk[missing_mask, feat_idx] = valid_mean
                
                # 7. Run prediction on chunk
                if X_chunk.shape[0] > 0:
                    try:
                        # predict_proba returns probabilities for each class
                        # [:, 1] gives probability of transition (class 1)
                        probs = model.predict_proba(X_chunk)[:, 1]
                        
                        # 8. Place results in suitability map
                        chunk_prob_map = np.full(lc_chunk.shape, -1.0, dtype='float32')
                        chunk_prob_map[from_mask_chunk] = probs
                        
                        # Update main suitability map
                        row_start = window.row_off
                        col_start = window.col_off
                        row_stop = row_start + window.height
                        col_stop = col_start + window.width
                        
                        map_slice = suitability_map[row_start:row_stop, col_start:col_stop]
                        np.copyto(map_slice, chunk_prob_map, where=(from_mask_chunk))
                        
                        processed_pixels += np.sum(from_mask_chunk)
                    
                    except Exception as e:
                        logger.error(f"Error predicting on chunk {block_idx}: {e}")
                        continue
                
                # 9. Clear memory between chunks
                del X_chunk, predictor_chunks, lc_chunk, from_mask_chunk
                gc.collect()
                
                if (block_idx + 1) % 10 == 0:
                    logger.debug(f"Processed block {block_idx + 1}/{total_blocks}")
            
            logger.info(f"Suitability map complete: {processed_pixels} pixels evaluated")
        
        # --- Write suitability map to file ---
        with rasterio.open(temp_filepath, 'w', **profile) as dst:
            dst.write(suitability_map, 1)
        
        logger.info(f"✅ Saved suitability map: {temp_filepath}")
        return temp_filepath
    
    except Exception as e:
        logger.error(f"Error generating suitability map: {e}", exc_info=True)
        raise


def generate_suitability_atlas(predictor_files, lc_end_file, trained_model_paths, 
                              temp_dir, missing_value_handling='mean',
                              progress_callback=None):
    """
    STAGE 2: Generate complete suitability atlas.
    
    Creates suitability maps for all trained transitions. This is the most
    computationally intensive step, but only runs once per scenario.
    
    Args:
        predictor_files (list): List of predictor raster files
        lc_end_file: Reference land cover raster
        trained_model_paths (dict): {(from, to) or (from, to, mode): model_path}
        temp_dir (str): Directory to save suitability maps
        missing_value_handling (str): 'mean', 'mask', or 'keep'
        progress_callback (callable): Function(fraction, message) for progress updates
    
    Returns:
        dict: {key: suitability_map_path} matching trained_model_paths keys
    """
    if progress_callback is None:
        def progress_callback(p, t):
            pass
    
    suitability_paths = {}
    total_maps = len(trained_model_paths)
    
    logger.info(f"Generating suitability atlas: {total_maps} maps")
    
    for idx, (key, model_path) in enumerate(trained_model_paths.items(), 1):
        try:
            # Load feature schema
            features_path = model_path.replace(".joblib", "_features.json")
            if not os.path.exists(features_path):
                raise FileNotFoundError(
                    f"Feature schema not found: {features_path}"
                )
            
            with open(features_path, 'r') as f:
                required_features = json.load(f)
            
            logger.info(f"Feature schema: {required_features}")
            
            # Determine model type from key
            if len(key) == 3:  # Growth mode
                from_cls, to_cls, mode = key
                msg = f"Generating {mode} suitability map {idx}/{total_maps}: {from_cls}→{to_cls}"
                logger.info(msg)
                progress_callback(idx / total_maps, msg)
                
                suitability_paths[key] = _generate_single_suitability_map(
                    model_path, predictor_files, lc_end_file, temp_dir,
                    required_features, model_type=mode,
                    missing_value_handling=missing_value_handling
                )
            else:  # Standard mode
                from_cls, to_cls = key
                msg = f"Generating suitability map {idx}/{total_maps}: {from_cls}→{to_cls}"
                logger.info(msg)
                progress_callback(idx / total_maps, msg)
                
                suitability_paths[key] = _generate_single_suitability_map(
                    model_path, predictor_files, lc_end_file, temp_dir,
                    required_features, model_type='',
                    missing_value_handling=missing_value_handling
                )
        
        except Exception as e:
            logger.error(f"Failed to generate suitability map for {key}: {e}")
            progress_callback(idx / total_maps, f"❌ Failed: {key}")
            continue
    
    progress_callback(1.0, f"✅ Suitability atlas complete! ({len(suitability_paths)}/{total_maps} maps)")
    logger.info(f"Suitability atlas generation complete: {len(suitability_paths)} maps created")
    
    return suitability_paths


# ============================================================================
# STAGE 3: ALLOCATION SIMULATION
# ============================================================================

def run_allocation_simulation(lc_end_file, transition_counts, suitability_paths,
                             temp_dir, stochastic=False, stochastic_config=None,
                             progress_callback=None, constraint_mask=None,
                             allocation_report=None):
    """
    STAGE 3: Run final allocation simulation.

    Allocates projected land use change to the most suitable pixels based on
    pre-generated suitability maps. This stage is much faster than Stage 2.

    Args:
        lc_end_file: Baseline land cover raster (starting point for simulation)
        transition_counts (pd.DataFrame): Transition demand (pixels to change)
        suitability_paths (dict): {key: suitability_map_path} from Stage 2
        temp_dir (str): Directory to save output
        stochastic (bool): If True, use probabilistic allocation
        stochastic_config (dict): Stochastic parameters (n_simulations, etc.)
        progress_callback (callable): Function(fraction, message) for progress updates
        constraint_mask (np.ndarray): Optional boolean scenario zone. True means
            "no change may be allocated here". Removed from the candidate pool,
            not merely down-weighted -- see the note below.
        allocation_report (dict): Optional dict filled in with per-transition
            demand, allocated and unmet pixel counts. Passed in rather than
            returned so the return type stays a path and existing callers keep
            working.

    Returns:
        str: Path to predicted land cover map (or list if stochastic with n>1)

    Why `constraint_mask` is a HARD exclusion, not a suitability penalty
    -------------------------------------------------------------------
    Zeroing suitability inside the zone does NOT keep it unbuilt. Deterministic
    selection here is `np.argpartition(available_scores, -num_to_change)`, which
    takes the top N scores whatever they are: once demand exceeds the number of
    positive-suitability pixels, zero-scored pixels are selected, and a larger
    zeroed area makes that MORE likely, not less. The companion deforestation
    module hit exactly this in the Guinea pilot -- zeroing risk inside protected
    areas allocated MORE clearing inside them (808,182 ha) than business-as-usual
    (720,398 ha), the reverse of what protection means. The fix there and here is
    the same: remove the pixels from `available_mask` before any scoring, so they
    can never be ranked at all.

    When demand cannot fit outside the zone, the shortfall is recorded as unmet
    rather than quietly spilling into it. Unmet demand is a real, reportable
    finding -- "the projected change does not fit outside the protected zone" --
    and must not be hidden.
    """
    if progress_callback is None:
        def progress_callback(p, t):
            pass
    
    logger.info(f"Starting allocation simulation (stochastic={stochastic})")

    # GUARD: with an empty suitability atlas every transition is skipped and the
    # "prediction" returned is the unchanged baseline map -- which previously
    # slipped through into validation and produced meaningless (persistence)
    # accuracy/FoM numbers that looked plausible. Refuse outright; any working
    # run always has at least one suitability map.
    if not suitability_paths:
        raise RuntimeError(
            "run_allocation_simulation received an empty suitability atlas -- "
            "the result would be the unchanged baseline map masquerading as a "
            "prediction. Train transition models and generate the suitability "
            "atlas first (see training.train_transition_models / "
            "prediction.generate_suitability_atlas logs for why it is empty)."
        )

    # Check if using growth modes
    growth_mode = any(len(k) == 3 for k in suitability_paths.keys())
    logger.info(f"Growth mode: {growth_mode}")
    
    # Load baseline land cover
    with _open_as_raster(lc_end_file) as src:
        future_lc = src.read(1).copy()
        profile = src.profile

    logger.info(f"Loaded baseline LC: {future_lc.shape}")

    blocked = None
    if constraint_mask is not None:
        blocked = np.asarray(constraint_mask, dtype=bool)
        if blocked.shape != future_lc.shape:
            raise ValueError(
                f"constraint_mask shape {blocked.shape} does not match the "
                f"land-cover grid {future_lc.shape}; rasterize it against the "
                f"same reference profile.")
        logger.info("Spatial constraint active: %d pixels excluded from allocation.",
                    int(blocked.sum()))

    report_rows = []
    
    # Get sorted transitions by demand (largest first)
    sorted_transitions = transition_counts.stack().sort_values(ascending=False).index.tolist()
    
    logger.info(f"Processing {len(sorted_transitions)} transitions in priority order")
    
    # --- Allocate changes for each transition ---
    for trans_idx, (from_cls, to_cls) in enumerate(sorted_transitions):
        if from_cls == to_cls:
            continue
        
        demand = int(transition_counts.loc[from_cls, to_cls])
        if demand <= 0:
            continue
        
        progress_callback(trans_idx / len(sorted_transitions),
                         f"Allocating {from_cls}→{to_cls} ({demand} pixels)")
        logger.info(f"Transition {from_cls}→{to_cls}: demand={demand} pixels")
        
        # --- Load suitability map(s) ---
        if growth_mode:
            # Use expander and patcher models
            exp_path = suitability_paths.get((from_cls, to_cls, 'expander'))
            pat_path = suitability_paths.get((from_cls, to_cls, 'patcher'))
            
            if not exp_path or not pat_path:
                logger.warning(f"Missing growth mode maps for {from_cls}→{to_cls}")
                continue
            
            with rasterio.open(exp_path) as exp_src:
                exp_suit = exp_src.read(1)
            with rasterio.open(pat_path) as pat_src:
                pat_suit = pat_src.read(1)
            
            # Combine: expander for boundary, patcher for new patches
            to_class_mask = (future_lc == to_cls)
            dilated_mask = binary_dilation(to_class_mask, structure=np.ones((3, 3)))
            suitability_map = np.where(dilated_mask, exp_suit, pat_suit)
            
            logger.debug(f"Combined expander/patcher maps for {from_cls}→{to_cls}")
        
        else:
            # Standard single model
            suitability_path = suitability_paths.get((from_cls, to_cls))
            
            if not suitability_path:
                logger.warning(f"Missing suitability map for {from_cls}→{to_cls}")
                continue
            
            with rasterio.open(suitability_path) as src:
                suitability_map = src.read(1)
        
        # --- Find available pixels (current land cover matches from_cls) ---
        # The constraint is applied HERE, to the candidate pool itself, so
        # excluded pixels are never scored or ranked. See the docstring for why
        # a suitability penalty is not equivalent.
        available_mask = (future_lc == from_cls)
        if blocked is not None:
            available_mask &= ~blocked
        available_count = np.sum(available_mask)

        if available_count == 0:
            logger.debug(f"No available pixels for {from_cls}→{to_cls}")
            report_rows.append({'from_class': int(from_cls), 'to_class': int(to_cls),
                                'demand_px': demand, 'allocated_px': 0,
                                'unmet_px': demand})
            del suitability_map
            gc.collect()
            continue
        
        available_scores = suitability_map[available_mask]
        available_coords = np.argwhere(available_mask)
        
        num_to_change = min(demand, available_count)
        logger.debug(f"Available: {available_count}, To change: {num_to_change}")
        if num_to_change < demand:
            logger.warning(
                "Transition %s->%s: demand %d px exceeds available capacity %d px%s. "
                "Allocating what fits and reporting the shortfall rather than "
                "placing change where it is not allowed.",
                from_cls, to_cls, demand, available_count,
                " outside the spatial constraint" if blocked is not None else "")
        
        # --- Select pixels for change ---
        if stochastic:
            # Probabilistic allocation
            non_zero_indices = np.where(available_scores > 0)[0]
            size_for_choice = min(num_to_change, len(non_zero_indices))
            
            if size_for_choice > 0:
                scores_for_choice = available_scores[non_zero_indices]
                probs_for_choice = scores_for_choice / np.sum(scores_for_choice)
                
                chosen_indices = np.random.choice(
                    len(non_zero_indices),
                    size=size_for_choice,
                    replace=False,
                    p=probs_for_choice
                )
                top_indices = non_zero_indices[chosen_indices]
                logger.debug(f"Stochastic selection: {len(top_indices)} pixels")
            else:
                top_indices = []
        
        else:
            # Deterministic allocation (highest suitability)
            if num_to_change > 0:
                top_indices = np.argpartition(available_scores, -num_to_change)[-num_to_change:]
                logger.debug(f"Deterministic selection: {len(top_indices)} pixels")
            else:
                top_indices = []
        
        # --- Apply changes to map ---
        if len(top_indices) > 0:
            coords_to_change = available_coords[top_indices]
            rows, cols = coords_to_change.T
            future_lc[rows, cols] = to_cls
            logger.info(f"Changed {len(top_indices)} pixels: {from_cls}→{to_cls}")

        report_rows.append({'from_class': int(from_cls), 'to_class': int(to_cls),
                            'demand_px': demand, 'allocated_px': int(len(top_indices)),
                            'unmet_px': int(demand - len(top_indices))})

        # Clear memory
        del suitability_map
        gc.collect()

    if allocation_report is not None:
        total_unmet = sum(r['unmet_px'] for r in report_rows)
        allocation_report['transitions'] = report_rows
        allocation_report['total_demand_px'] = sum(r['demand_px'] for r in report_rows)
        allocation_report['total_allocated_px'] = sum(r['allocated_px'] for r in report_rows)
        allocation_report['total_unmet_px'] = total_unmet
        allocation_report['constrained'] = blocked is not None
        allocation_report['constrained_px'] = int(blocked.sum()) if blocked is not None else 0
        if total_unmet:
            allocation_report['note'] = (
                "Some projected change could not be placed. This is a result, not "
                "an error: the demand does not fit in the locations the scenario "
                "allows. Report the unmet share rather than presenting the map as "
                "though all demand was satisfied.")

    # --- Save predicted map ---
    output_path = os.path.join(temp_dir, "predicted_land_cover.tif")
    with rasterio.open(output_path, 'w', **profile) as dst:
        dst.write(future_lc, 1)
    
    logger.info(f"✅ Saved predicted map: {output_path}")
    progress_callback(1.0, "✅ Allocation complete!")
    
    return output_path


# ============================================================================
# VALIDATION & REPORTING
# ============================================================================

def validate_suitability_maps(suitability_paths):
    """
    Validate suitability maps for quality.
    
    Args:
        suitability_paths (dict): Suitability map paths from Stage 2
    
    Returns:
        tuple: (is_valid, stats_dict, warnings_list)
    """
    stats = {}
    warnings = []
    
    for key, path in suitability_paths.items():
        try:
            with rasterio.open(path) as src:
                data = src.read(1)
            
            valid_data = data[data != -1.0]  # Exclude nodata
            
            if len(valid_data) == 0:
                warnings.append(f"{key}: No valid suitability values")
                continue
            
            stats[str(key)] = {
                'min': float(np.min(valid_data)),
                'max': float(np.max(valid_data)),
                'mean': float(np.mean(valid_data)),
                'std': float(np.std(valid_data)),
                'valid_pixels': int(len(valid_data))
            }
        
        except Exception as e:
            warnings.append(f"{key}: Error reading map - {e}")
    
    is_valid = len(warnings) == 0
    return is_valid, stats, warnings


def compare_baseline_prediction(baseline_path, predicted_path):
    """
    Compare baseline and predicted land cover maps.
    
    Args:
        baseline_path (str): Path to baseline LC map
        predicted_path (str): Path to predicted LC map
    
    Returns:
        dict: Comparison statistics
    """
    with rasterio.open(baseline_path) as src:
        baseline = src.read(1)
    with rasterio.open(predicted_path) as src:
        predicted = src.read(1)
    
    # Changes
    changed_mask = baseline != predicted
    num_changed = np.sum(changed_mask)
    
    # Class transitions
    classes = np.unique(baseline)
    transition_summary = {}
    
    for cls in classes:
        from_count = np.sum(baseline == cls)
        to_count = np.sum(predicted == cls)
        transition_summary[f'Class {cls}'] = {
            'baseline_pixels': int(from_count),
            'predicted_pixels': int(to_count),
            'change': int(to_count - from_count)
        }
    
    summary = {
        'total_pixels': baseline.size,
        'changed_pixels': int(num_changed),
        'change_percent': float(100 * num_changed / baseline.size),
        'class_summary': transition_summary
    }
    
    logger.info(f"Prediction summary: {num_changed} pixels changed "
               f"({100*num_changed/baseline.size:.1f}%)")
    
    return summary


# ============================================================================
# MONTE CARLO ENSEMBLE ALLOCATION
# ============================================================================
# run_allocation_simulation(stochastic=True) already does probability-weighted
# pixel selection for ONE draw. The methodology document describes "running
# multiple independent Monte Carlo-style simulations" to "generate a range of
# plausible spatial futures" -- that part was never built; only a single
# stochastic draw was ever produced. This wrapper closes that specific gap
# with the lightest addition that actually delivers what's already claimed:
# run the existing single-draw allocation N times with different random
# seeds, and aggregate into ONE new, easy-to-explain output -- a per-pixel
# change-frequency map -- rather than exposing every individual draw to
# users. "This pixel changed in 18 of 25 simulated futures" is intuitive on
# its own; it does not need a bigger uncertainty-quantification framework
# around it to be useful.

def run_monte_carlo_allocation(lc_end_file, transition_counts, suitability_paths,
                                temp_dir, n_simulations=25, base_seed=42,
                                progress_callback=None, constraint_mask=None):
    """
    Run stochastic allocation n_simulations times and aggregate into a
    change-frequency map plus a per-class area-range summary.

    Args:
        lc_end_file, transition_counts, suitability_paths: same as
            run_allocation_simulation.
        temp_dir (str): directory to write per-draw maps and the aggregate.
        n_simulations (int): number of independent draws. 25 is a modest
            default -- enough to see a stable frequency pattern emerge
            without the runtime cost of hundreds of draws; increase if the
            change-frequency map still looks noisy at 25.
        base_seed (int): draws use base_seed, base_seed+1, ... base_seed+N-1
            so a run is exactly reproducible.

    Returns:
        dict: {
            'frequency_map_path': str,  # GeoTIFF, values 0..1 = fraction of
                                          # draws in which that pixel changed
                                          # from its baseline class
            'draw_paths': [str, ...],    # each individual simulated map
            'area_summary': pd.DataFrame,  # per-class mean/std/min/max pixel
                                             # count across the N draws
        }
    """
    import pandas as pd

    if progress_callback is None:
        def progress_callback(p, t):
            pass

    with _open_as_raster(lc_end_file) as src:
        baseline = src.read(1).copy()
        profile = src.profile

    draw_paths = []
    change_count = np.zeros(baseline.shape, dtype=np.int32)
    class_counts_per_draw = []

    draw_dir = os.path.join(temp_dir, "monte_carlo_draws")
    os.makedirs(draw_dir, exist_ok=True)

    for i in range(n_simulations):
        progress_callback(i / n_simulations, f"Monte Carlo draw {i + 1}/{n_simulations}")
        np.random.seed(base_seed + i)

        draw_path = run_allocation_simulation(
            lc_end_file, transition_counts, suitability_paths,
            temp_dir=draw_dir, stochastic=True,
            constraint_mask=constraint_mask,
        )
        # run_allocation_simulation always writes to a fixed filename inside
        # temp_dir; rename per-draw so all N survive.
        renamed_path = os.path.join(draw_dir, f"draw_{i:03d}.tif")
        if os.path.exists(renamed_path):
            os.remove(renamed_path)
        os.rename(draw_path, renamed_path)
        draw_paths.append(renamed_path)

        with rasterio.open(renamed_path) as src:
            draw_map = src.read(1)

        change_count += (draw_map != baseline).astype(np.int32)

        classes, counts = np.unique(draw_map, return_counts=True)
        class_counts_per_draw.append(dict(zip(classes.tolist(), counts.tolist())))

        del draw_map
        gc.collect()

    frequency_map = (change_count.astype(np.float32) / n_simulations)

    freq_profile = profile.copy()
    freq_profile.update(dtype="float32", count=1)
    frequency_map_path = os.path.join(temp_dir, "change_frequency_map.tif")
    with rasterio.open(frequency_map_path, "w", **freq_profile) as dst:
        dst.write(frequency_map, 1)

    all_classes = sorted({c for d in class_counts_per_draw for c in d.keys()})
    rows = []
    for cls in all_classes:
        vals = np.array([d.get(cls, 0) for d in class_counts_per_draw])
        rows.append({
            "class": cls, "mean_pixels": float(vals.mean()),
            "std_pixels": float(vals.std()), "min_pixels": int(vals.min()),
            "max_pixels": int(vals.max()),
        })
    area_summary = pd.DataFrame(rows)

    logger.info(f"✅ Monte Carlo ensemble complete: {n_simulations} draws, "
               f"frequency map saved to {frequency_map_path}")
    progress_callback(1.0, "✅ Monte Carlo ensemble complete!")

    return {
        "frequency_map_path": frequency_map_path,
        "draw_paths": draw_paths,
        "area_summary": area_summary,
    }