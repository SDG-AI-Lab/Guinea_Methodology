"""
Model Training Module

Handles:
1. Transition-specific training dataset creation
2. Random Forest model training with validation metrics
3. Model persistence and feature schema management
4. Support for advanced growth modes (expander/patcher)
5. Missing value handling in training data
"""

import rasterio
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from contextlib import ExitStack
from rasterio.windows import Window
from scipy.ndimage import binary_dilation
import joblib
import json
import os
import logging
from typing import Optional, Tuple, Dict, List

from .data_loader import _open_as_raster
from .utils import handle_missing_values, get_missing_value_stats
from . import config

logger = logging.getLogger(__name__)

# ============================================================================
# TRAINING DATA CREATION
# ============================================================================

def create_transition_dataset(from_cls, to_cls, lc_start_file, lc_end_file, 
                             predictor_files, mode='standard', max_samples=50000,
                             missing_value_handling='exclude'):
    """
    Creates a balanced training dataset for a specific land cover transition.
    
    Features:
    - Balanced positive/negative samples
    - Support for growth modes (expander/patcher)
    - Memory-efficient pixel extraction
    - Handles missing values in predictors
    
    Args:
        from_cls (int): Source land cover class
        to_cls (int): Target land cover class
        lc_start_file: Initial land cover raster
        lc_end_file: Final land cover raster
        predictor_files (list): List of predictor raster files
        mode (str): 'standard', 'expander', or 'patcher'
                   - standard: all transition pixels
                   - expander: transition pixels adjacent to existing target class
                   - patcher: isolated new patches of target class
        max_samples (int): Maximum samples per class (prevents memory overload)
        missing_value_handling (str): 'exclude' = skip samples with any NaN,
                                     'mask' = replace NaN with 0
    
    Returns:
        tuple: (X_samples, y_samples) numpy arrays or (None, None) if failed
        - X_samples: shape (n_samples, n_predictors)
        - y_samples: shape (n_samples,) with values [0=no_change, 1=changed]
    """
    try:
        # Load land cover maps
        with _open_as_raster(lc_start_file) as src_start, _open_as_raster(lc_end_file) as src_end:
            lc_start = src_start.read(1)
            lc_end = src_end.read(1)
            logger.info(f"Loaded LC maps: start shape {lc_start.shape}, end shape {lc_end.shape}")
        
        # Identify positive samples (pixels that transitioned from->to)
        positive_mask = (lc_start == from_cls) & (lc_end == to_cls)
        positive_coords = np.argwhere(positive_mask)
        
        logger.info(f"Transition {from_cls}→{to_cls}: {len(positive_coords)} positive pixels")
        
        # Apply growth mode filtering
        if mode in ['expander', 'patcher']:
            # Identify pixels that were already the target class
            initial_to_class_mask = (lc_start == to_cls)
            
            # Dilate to find adjacent pixels
            dilated_mask = binary_dilation(initial_to_class_mask, structure=np.ones((3, 3)))
            
            if mode == 'expander':
                # Only keep transitions that are adjacent to existing target class
                positive_coords = np.argwhere(dilated_mask & positive_mask)
                logger.info(f"Expander mode: {len(positive_coords)} boundary pixels")
            
            elif mode == 'patcher':
                # Only keep transitions that are NOT adjacent (new patches)
                patcher_mask = ~dilated_mask & positive_mask
                positive_coords = np.argwhere(patcher_mask)
                logger.info(f"Patcher mode: {len(positive_coords)} new patch pixels")
        
        # Identify negative samples (pixels that stayed in from_cls)
        negative_mask = (lc_start == from_cls) & (lc_end == from_cls)
        negative_coords = np.argwhere(negative_mask)
        
        logger.info(f"Negative samples: {len(negative_coords)} pixels")
        
        # Balance dataset
        n_samples = min(len(positive_coords), len(negative_coords))
        
        if n_samples < 10:
            logger.warning(f"Insufficient samples for transition {from_cls}→{to_cls}: "
                          f"pos={len(positive_coords)}, neg={len(negative_coords)}")
            return None, None
        
        # Cap at max_samples to prevent memory issues
        n_samples = min(n_samples, max_samples)
        logger.info(f"Using {n_samples} samples per class (balanced)")
        
        # Randomly sample
        positive_sample_coords = positive_coords[
            np.random.choice(len(positive_coords), n_samples, replace=False)
        ]
        negative_sample_coords = negative_coords[
            np.random.choice(len(negative_coords), n_samples, replace=False)
        ]
        
        all_coords = np.vstack([positive_sample_coords, negative_sample_coords])
        y = np.array([1] * n_samples + [0] * n_samples)
        
        logger.info(f"Sampling {len(all_coords)} pixels from {len(predictor_files)} predictors")
        
        # Extract predictor values at sampled locations
        X = []
        valid_indices = []
        skipped_count = 0
        
        logger.info(f"Loading {len(predictor_files)} predictors into memory for vectorized extraction...")
        
        # Generate spatial block IDs (e.g., 100x100 pixel blocks) to prevent data leakage in cross-validation
        block_size = 100
        block_ids = (all_coords[:, 0] // block_size) * 1000 + (all_coords[:, 1] // block_size)
        
        X_list = []
        for p_file in predictor_files:
            with rasterio.open(p_file) as src:
                arr = src.read(1)
                # GUARD: every predictor must already be on the same grid as the
                # land-cover maps. Indexing LC-grid coordinates into a raster on
                # its native (unaligned) grid previously failed here with an
                # opaque IndexError, was caught upstream, and silently skipped
                # EVERY transition model (0 models trained) -- which then let an
                # unchanged persistence map masquerade as a validated prediction.
                # Fail with an explicit, actionable message instead.
                if arr.shape != lc_start.shape:
                    raise ValueError(
                        f"Predictor '{p_file}' has grid {arr.shape}, but the "
                        f"land-cover maps are {lc_start.shape}. Predictor files "
                        f"must be pre-aligned to the reference grid (e.g. written "
                        f"from the aligned in-memory predictor stack) before "
                        f"training. Refusing to sample from a misaligned raster."
                    )
                # Extract all sampled points at once via advanced indexing
                vals = arr[all_coords[:, 0], all_coords[:, 1]]
                X_list.append(vals)
        
        # Transpose to shape (n_samples, n_predictors)
        X = np.column_stack(X_list).astype(np.float32)
        
        # Align NaN handling directly with the configured strategy
        if missing_value_handling == 'mask':
            X[np.isnan(X)] = 0.0
        elif missing_value_handling == 'mean':
            col_means = np.nanmean(X, axis=0)
            inds = np.where(np.isnan(X))
            X[inds] = np.take(col_means, inds[1])
        elif missing_value_handling == 'exclude':
            valid_mask = ~np.isnan(X).any(axis=1)
            X = X[valid_mask]
            y = y[valid_mask]
            block_ids = block_ids[valid_mask]
            skipped_count = len(all_coords) - len(y)
            if skipped_count > 0:
                logger.info(f"Skipped {skipped_count} samples with missing values")
        
        if len(X) < 10:
            logger.warning(f"Final dataset too small: {len(X)} samples")
            return None, None, None
        
        logger.info(f"Successfully created dataset: {X.shape}")
        return X, y, block_ids
    
    except Exception as e:
        logger.error(f"Error creating transition dataset: {e}", exc_info=True)
        return None, None


# ============================================================================
# MODEL TRAINING
# ============================================================================

def train_rf_model(X, y, block_ids, model_path, predictor_schema, 
                  test_size=0.3, n_estimators=100, random_state=42,
                  cv_folds=5):
    """
    Trains a Random Forest classifier for a specific transition.
    
    Features:
    - Train/test split validation
    - Cross-validation score
    - Feature importance ranking
    - Model and schema persistence
    - Reports missing value handling
    
    Args:
        X (np.ndarray): Training features (n_samples, n_predictors)
        y (np.ndarray): Binary labels (1=transition, 0=no_transition)
        model_path (str): Path to save trained model (.joblib)
        predictor_schema (list): List of predictor names (for feature mapping)
        test_size (float): Test set fraction
        n_estimators (int): Number of trees in forest
        random_state (int): Random seed
        cv_folds (int): Cross-validation folds
    
    Returns:
        tuple: (overall_accuracy, metrics_dict) or (0.0, {}) if training fails
    """
    
    if X is None or y is None or len(X) == 0:
        logger.warning("Invalid training data")
        return 0.0, {}
    
    try:
        logger.info(f"Training model with {len(X)} samples, {X.shape[1]} predictors")
        
        # Check for remaining NaN values (shouldn't happen after sampling, but be safe)
        nan_mask = np.isnan(X).any(axis=1)
        if np.any(nan_mask):
            logger.warning(f"Found {np.sum(nan_mask)} samples with NaN, removing them")
            X = X[~nan_mask]
            y = y[~nan_mask]
        
        if len(X) < 10:
            logger.error(f"Insufficient valid training samples: {len(X)}")
            return 0.0, {}
        
        # Split data using Spatial Block Cross-Validation
        gss = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=random_state)
        train_idx, test_idx = next(gss.split(X, y, groups=block_ids))
        
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        
        logger.info(f"Train set: {len(X_train)}, Test set: {len(X_test)} (Spatially isolated)")
        
        # Train model (hyperparameters from config.MODEL_CONFIG)
        model = RandomForestClassifier(
            n_estimators=n_estimators,
            random_state=random_state,
            n_jobs=config.MODEL_CONFIG.get('n_jobs', -1),
            max_depth=config.MODEL_CONFIG.get('max_depth', 20),
            min_samples_split=config.MODEL_CONFIG.get('min_samples_split', 5),
            min_samples_leaf=config.MODEL_CONFIG.get('min_samples_leaf', 2)
        )
        model.fit(X_train, y_train)
        
        # Evaluate
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        
        logger.info(f"Test accuracy: {accuracy:.4f}")
        
        # Cross-validation score
        cv_scores = cross_val_score(model, X_train, y_train, cv=cv_folds, n_jobs=-1)
        mean_cv_score = cv_scores.mean()
        std_cv_score = cv_scores.std()
        
        logger.info(f"CV scores: {mean_cv_score:.4f} ± {std_cv_score:.4f}")
        
        # Classification metrics
        report = classification_report(y_test, y_pred, output_dict=True)
        cm = confusion_matrix(y_test, y_pred)
        
        # Feature importance
        feature_importance = model.feature_importances_
        
        # Create metrics dictionary
        metrics = {
            'accuracy': float(accuracy),
            'cv_mean': float(mean_cv_score),
            'cv_std': float(std_cv_score),
            'confusion_matrix': cm.tolist(),
            'classification_report': report,
            'feature_importance': feature_importance.tolist(),
            'feature_names': predictor_schema,
            'n_train_samples': len(X_train),
            'n_test_samples': len(X_test),
            'n_predictors': X.shape[1],
            'missing_value_handling': config.MISSING_VALUE_CONFIG['handling_strategies']['training']['method']
        }
        
        # Save model
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        joblib.dump(model, model_path)
        logger.info(f"✅ Model saved to {model_path}")
        
        # Save feature schema
        schema_path = model_path.replace('.joblib', '_features.json')
        with open(schema_path, 'w') as f:
            json.dump(predictor_schema, f, indent=4)
        logger.info(f"✅ Feature schema saved to {schema_path}")
        
        # Save metrics
        metrics_path = model_path.replace('.joblib', '_metrics.json')
        with open(metrics_path, 'w') as f:
            json.dump(metrics, f, indent=4)
        logger.info(f"✅ Metrics saved to {metrics_path}")
        
        return accuracy, metrics
    
    except Exception as e:
        logger.error(f"Error training model: {e}", exc_info=True)
        return 0.0, {}


def load_trained_model(model_path):
    """
    Load a trained model and its associated metadata.
    
    Args:
        model_path (str): Path to .joblib model file
    
    Returns:
        tuple: (model, predictor_schema) or (None, None) if loading fails
    """
    try:
        model = joblib.load(model_path)
        
        # Load feature schema
        schema_path = model_path.replace('.joblib', '_features.json')
        if os.path.exists(schema_path):
            with open(schema_path, 'r') as f:
                schema = json.load(f)
        else:
            logger.warning(f"Schema file not found: {schema_path}")
            schema = None
        
        logger.info(f"✅ Loaded model from {model_path}")
        return model, schema
    
    except Exception as e:
        logger.error(f"Error loading model {model_path}: {e}")
        return None, None


def get_model_metrics(model_path):
    """
    Load metrics for a trained model.
    
    Args:
        model_path (str): Path to .joblib model file
    
    Returns:
        dict: Metrics dictionary or {} if not found
    """
    metrics_path = model_path.replace('.joblib', '_metrics.json')
    
    if os.path.exists(metrics_path):
        with open(metrics_path, 'r') as f:
            return json.load(f)
    
    logger.warning(f"Metrics file not found: {metrics_path}")
    return {}


# ============================================================================
# BATCH TRAINING ORCHESTRATION
# ============================================================================

def train_transition_models(significant_transitions, target_files, predictor_files,
                           use_growth_modes=False, temp_dir=None,
                           missing_value_handling='exclude',
                           progress_callback=None):
    """
    Train Random Forest models for all significant transitions.
    
    Args:
        significant_transitions (list): List of (from_cls, to_cls) tuples
        target_files (list): List of target raster files
        predictor_files (list): List of predictor raster files
        use_growth_modes (bool): If True, train expander and patcher models
        temp_dir (str): Directory to save models
        missing_value_handling (str): Strategy for handling missing values
        progress_callback (callable): Function(fraction, message) for progress updates
    
    Returns:
        tuple: (model_paths_dict, accuracies_dict)
        - model_paths_dict: {(from, to) or (from, to, mode): path}
        - accuracies_dict: {key: accuracy_score}
    """
    
    if progress_callback is None:
        def progress_callback(frac, msg):
            pass

    # Azure-safe default: write models under the configured output dir (/tmp/...).
    if temp_dir is None:
        temp_dir = os.path.join(config.PROCESSING_CONFIG['output_dir'], 'models')
    os.makedirs(temp_dir, exist_ok=True)
    
    model_paths = {}
    accuracies = {}
    
    # Calculate total models to train
    if use_growth_modes:
        total_models = len(significant_transitions) * 2
        modes = ['expander', 'patcher']
    else:
        total_models = len(significant_transitions)
        modes = ['standard']
    
    logger.info(f"Training {total_models} models ({len(significant_transitions)} transitions × "
               f"{len(modes)} mode(s))")
    logger.info(f"Missing value handling: {missing_value_handling}")
    
    model_counter = 0
    
    for from_cls, to_cls in significant_transitions:
        for mode in modes:
            model_counter += 1
            
            msg = f"Training {mode} model {model_counter}/{total_models}: {from_cls}→{to_cls}"
            progress_callback(model_counter / total_models, msg)
            logger.info(msg)
            
            # Create training data
            dataset = create_transition_dataset(
                from_cls, to_cls,
                target_files[0]['file'],
                target_files[-1]['file'],
                predictor_files,
                mode=mode,
                missing_value_handling=missing_value_handling
            )
            
            if dataset[0] is None:
                logger.warning(f"Skipped {mode} model for {from_cls}→{to_cls}")
                continue
                
            X, y, block_ids = dataset
            
            # Train model
            predictor_schema = [
                f.name if hasattr(f, 'name') else os.path.basename(f) 
                for f in predictor_files
            ]
            
            if mode == 'standard':
                model_path = os.path.join(temp_dir, f"model_{from_cls}_{to_cls}.joblib")
                key = (from_cls, to_cls)
            else:
                model_path = os.path.join(temp_dir, f"model_{mode}_{from_cls}_{to_cls}.joblib")
                key = (from_cls, to_cls, mode)
            
            accuracy, metrics = train_rf_model(
                                    X,
                                    y,
                                    block_ids,
                                    model_path,
                                    predictor_schema
                                )
            
            if accuracy > 0:
                model_paths[key] = model_path
                accuracies[str(key)] = accuracy
                logger.info(f"✅ Saved model: {model_path} (accuracy: {accuracy:.2%})")
    
    # GUARD: if transitions were requested but not a single model trained, every
    # one of them failed (most likely misaligned predictor rasters -- see the
    # shape guard in create_transition_dataset). Returning an empty dict here
    # previously propagated into an empty suitability atlas and an allocation
    # step that returned the baseline map unchanged, producing plausible-looking
    # but meaningless validation numbers. A working run is never affected by
    # this check.
    if significant_transitions and not model_paths:
        raise RuntimeError(
            f"0 of {total_models} transition models trained -- every transition "
            f"failed. Check the ERROR log lines above (typical cause: predictor "
            f"files not aligned to the land-cover reference grid)."
        )

    logger.info(f"✅ Training complete: {len(model_paths)} models trained")
    return model_paths, accuracies
