"""
Change Analysis Module

Handles:
1. Markov Chain analysis - transition probability matrices
2. Linear trend analysis (2 time periods)
3. Non-linear trend analysis (3+ time periods)
4. Visualization of historical trends
"""

import rasterio
import numpy as np
import pandas as pd
import matplotlib
# Headless backend: this module runs inside Celery workers (no display), where
# matplotlib's default backend auto-detection is platform/library dependent.
# Must be set before pyplot is imported anywhere in the process.
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import confusion_matrix
from scipy.optimize import curve_fit
import logging

from .data_loader import _open_as_raster

logger = logging.getLogger(__name__)

# ============================================================================
# MARKOV CHAIN ANALYSIS
# ============================================================================

def calculate_transition_matrix(file1, file2, nodata_value=None, progress_callback=None):
    if progress_callback is None:
        def progress_callback(frac, text): pass
    """
    Calculates transition probability matrix and pixel counts between two land cover maps.
    
    Uses Markov Chain analysis: transition_probability = count(t->t+1) / count(all from t)
    
    Args:
        file1: Initial land cover raster file (time t)
        file2: Final land cover raster file (time t+1)
        nodata_value: Value to treat as nodata (auto-detected if None)
    
    Returns:
        tuple: (transition_matrix_df, transition_counts_df, []) 
        - transition_matrix_df: Probability matrix (rows sum to 1)
        - transition_counts_df: Raw pixel counts
        - []: Empty list for consistency with non-linear function
    """
    try:
        with _open_as_raster(file1) as src1, _open_as_raster(file2) as src2:
            arr1 = src1.read(1)
            arr2 = src2.read(1)
            
            # Validate dimensions
            if arr1.shape != arr2.shape:
                logger.error(f"Dimension mismatch: {arr1.shape} vs {arr2.shape}")
                return None, None, []
            
            # Determine nodata value
            nodata = nodata_value if nodata_value is not None else src1.nodata
            if nodata is None:
                unique_vals = np.unique(arr1)
                nodata = -9999 if -9999 not in unique_vals else np.min(unique_vals) - 1
            
            # Create valid mask (exclude nodata)
            valid_mask = (arr1 != nodata) & (arr2 != nodata)
            arr1_flat = arr1[valid_mask]
            arr2_flat = arr2[valid_mask]
            
            logger.info(f"Valid pixels: {np.sum(valid_mask)} / {arr1.size}")
            
            # Get all unique classes
            classes = sorted(list(np.unique(np.concatenate([arr1_flat, arr2_flat]))))
            logger.info(f"Found {len(classes)} land cover classes: {classes}")
            
            # Build confusion matrix (transition counts)
            counts = confusion_matrix(arr1_flat, arr2_flat, labels=classes)
        
        # Calculate transition probabilities (rows sum to 1)
        row_sums = counts.sum(axis=1, keepdims=True)
        with np.errstate(divide='ignore', invalid='ignore'):
            matrix = np.where(row_sums > 0, counts / row_sums, 0)
        
        # Create DataFrames
        matrix_df = pd.DataFrame(matrix, index=classes, columns=classes)
        counts_df = pd.DataFrame(counts, index=classes, columns=classes)
        
        logger.info(f"Transition matrix:\n{matrix_df}")
        
        return matrix_df, counts_df, []
    
    except Exception as e:
        logger.error(f"Transition matrix error: {e}", exc_info=True)
        return None, None, []


def get_transition_statistics(transition_counts_df):
    """
    Extract key statistics from transition matrix.
    
    Args:
        transition_counts_df (pd.DataFrame): Transition counts from calculate_transition_matrix
    
    Returns:
        dict: Statistics including persistence, change rates, etc.
    """
    stats = {}
    total_pixels = transition_counts_df.sum().sum()
    
    # Persistence (diagonal) and change (off-diagonal)
    persistence = np.diag(transition_counts_df.values).sum()
    change = total_pixels - persistence
    
    stats['total_pixels'] = total_pixels
    stats['persistence_pixels'] = persistence
    stats['persistence_percent'] = 100 * persistence / total_pixels if total_pixels > 0 else 0
    stats['change_pixels'] = change
    stats['change_percent'] = 100 * change / total_pixels if total_pixels > 0 else 0
    
    # Per-class statistics
    class_stats = {}
    for cls in transition_counts_df.index:
        from_count = transition_counts_df.loc[cls].sum()
        to_count = transition_counts_df[cls].sum()
        persistence_count = transition_counts_df.loc[cls, cls]
        
        class_stats[cls] = {
            'pixels_from': from_count,
            'pixels_to': to_count,
            'persistence': persistence_count,
            'persistence_percent': 100 * persistence_count / from_count if from_count > 0 else 0
        }
    
    stats['class_stats'] = class_stats
    return stats


# ============================================================================
# TREND ANALYSIS
# ============================================================================

def analyze_linear_trends(file1, file2, period_years=None, progress_callback=None):
    if progress_callback is None:
        def progress_callback(frac, text): pass
    """
    Simple linear trend analysis for 2 time periods.
    
    Args:
        file1: Initial land cover raster
        file2: Final land cover raster
        period_years (int): Number of years between maps (for annualized rates)
    
    Returns:
        tuple: (transition_matrix_df, projected_counts_df, [])
    """
    matrix_df, counts_df, _ = calculate_transition_matrix(file1, file2)
    
    if matrix_df is None:
        return None, None, []
    
    # If period_years is provided, annualize the rates
    if period_years and period_years > 0:
        annual_counts = counts_df / period_years
        logger.info(f"📊 Annualized rates computed over {period_years}-year period")
        return matrix_df, annual_counts.round(), []
    
    return matrix_df, counts_df, []


# ============================================================================
# CANDIDATE TREND MODELS (for genuine non-linear extrapolation)
# ============================================================================
# analyze_non_linear_trends() below was, despite its name, fitting only a
# straight line to each transition's annualized rate across intervals. That
# captures acceleration/deceleration BETWEEN intervals (an improvement over a
# single-step Markov projection) but is not actually non-linear curve fitting.
# These are the genuine candidates, compared by corrected AIC (AICc, which
# penalizes extra parameters more heavily at small sample sizes -- appropriate
# here since there are typically only ~4-8 historical epochs per transition):
#   - linear:      a*x + b                    (2 params; always fit-able)
#   - exponential: a*exp(b*x)                 (2 params; unbounded growth)
#   - logistic:    L / (1 + exp(-k*(x - x0))) (3 params; saturating S-curve,
#                  appropriate for a class approaching some ceiling, e.g.
#                  Mining/Urban area bounded by total land area or demand)

def _model_linear(x, a, b):
    return a * x + b


def _model_exponential(x, a, b):
    return a * np.exp(b * x)


def _model_logistic(x, L, k, x0):
    return L / (1.0 + np.exp(-k * (x - x0)))


def _aicc(y_true, y_pred, n_params):
    n = len(y_true)
    rss = float(np.sum((np.asarray(y_true) - np.asarray(y_pred)) ** 2))
    rss = max(rss, 1e-9)  # avoid log(0) for a perfect fit
    aic = n * np.log(rss / n) + 2 * n_params
    denom = n - n_params - 1
    if denom > 0:
        aic += (2 * n_params * (n_params + 1)) / denom
    return aic


def fit_transition_curve(years, rates, future_x, min_points_exponential=3,
                          min_points_logistic=5, delta_aicc_threshold=2.0):
    """
    Fit linear / exponential / logistic candidates to one transition's
    (years, annualized-rate) time series and pick the best by AICc, requiring
    a more complex model to beat a simpler one by >= delta_aicc_threshold
    (standard AIC practice -- otherwise prefer the simpler model).

    Returns dict: {'model': 'linear'|'exponential'|'logistic', 'params': ...,
                    'predict': callable, 'predicted_rate': float,
                    'candidates': {name: {'aicc':.., 'params':..}}}
    or None if even the linear fit fails (e.g. all-zero series).
    """
    years = np.asarray(years, dtype=float)
    rates = np.asarray(rates, dtype=float)
    if np.allclose(rates, 0):
        return None

    candidates = {}

    # Linear -- always attempted, the fallback floor.
    try:
        lin_model = LinearRegression().fit(years.reshape(-1, 1), rates)
        pred = lin_model.predict(years.reshape(-1, 1))
        candidates["linear"] = {
            "aicc": _aicc(rates, pred, 2),
            "predict": lambda x, m=lin_model: m.predict(np.asarray(x).reshape(-1, 1)),
            "params": {"a": float(lin_model.coef_[0]), "b": float(lin_model.intercept_)},
        }
    except Exception as e:
        logger.debug(f"Linear fit failed: {e}")

    # Exponential -- needs a few points and only makes sense for non-negative
    # rates (pixel counts / duration are non-negative by construction).
    if len(years) >= min_points_exponential and np.all(rates >= 0):
        try:
            a0 = max(np.mean(rates), 1e-3)
            popt, _ = curve_fit(_model_exponential, years - years.min(), rates,
                                 p0=[a0, 0.01], maxfev=5000)
            pred = _model_exponential(years - years.min(), *popt)
            candidates["exponential"] = {
                "aicc": _aicc(rates, pred, 2),
                "predict": lambda x, p=popt, x0=years.min(): _model_exponential(np.asarray(x) - x0, *p),
                "params": {"a": float(popt[0]), "b": float(popt[1])},
            }
        except Exception as e:
            logger.debug(f"Exponential fit failed: {e}")

    # Logistic -- needs enough points to identify 3 parameters without
    # trivially overfitting; bound the asymptote to a plausible multiple of
    # the observed range rather than letting it float freely.
    if len(years) >= min_points_logistic and np.all(rates >= 0) and np.max(rates) > 0:
        try:
            L0 = np.max(rates) * 1.5
            x0_0 = np.median(years)
            bounds = ([np.max(rates), -10, years.min() - 20],
                      [np.max(rates) * 5, 10, years.max() + 20])
            popt, _ = curve_fit(_model_logistic, years, rates,
                                 p0=[L0, 0.5, x0_0], bounds=bounds, maxfev=10000)
            pred = _model_logistic(years, *popt)
            candidates["logistic"] = {
                "aicc": _aicc(rates, pred, 3),
                "predict": lambda x, p=popt: _model_logistic(np.asarray(x), *p),
                "params": {"L": float(popt[0]), "k": float(popt[1]), "x0": float(popt[2])},
            }
        except Exception as e:
            logger.debug(f"Logistic fit failed: {e}")

    if not candidates:
        return None

    # Pick lowest AICc, but only prefer a more complex model if it beats the
    # simplest available one by more than the threshold (Occam's razor).
    order = [m for m in ("linear", "exponential", "logistic") if m in candidates]
    best = order[0]
    for name in order[1:]:
        if candidates[name]["aicc"] < candidates[best]["aicc"] - delta_aicc_threshold:
            best = name

    predicted_rate = float(max(0.0, candidates[best]["predict"]([future_x])[0]))

    return {
        "model": best,
        "params": candidates[best]["params"],
        "predict": candidates[best]["predict"],
        "predicted_rate": predicted_rate,
        "candidates": {k: {"aicc": v["aicc"], "params": v["params"]} for k, v in candidates.items()},
    }


def analyze_non_linear_trends(target_files_with_years, progress_callback=None):
    if progress_callback is None:
        def progress_callback(frac, text): pass
    
    """
    Analyzes change over multiple time periods, fitting the best-supported
    trend model (linear / exponential / logistic, selected by AICc -- see
    fit_transition_curve) to each transition's annualized rate and
    extrapolating to the next period.

    Features:
    - Fits linear, exponential and logistic candidates per transition and
      keeps whichever is best-supported given the amount of historical data
    - Extrapolates to predict future change
    - Generates plots for significant transitions, showing all fitted
      candidates so the choice of model is visible/auditable

    Args:
        target_files_with_years (list): List of dicts with 'file' and 'year' keys
                                       Must have at least 3 time points

    Returns:
        tuple: (final_transition_matrix_df, projected_counts_df, generated_plots)
    """
    logger.info("📈 Non-linear mode: fitting linear/exponential/logistic candidates per transition...")
    
    # Sort by year
    sorted_targets = sorted(target_files_with_years, key=lambda x: x['year'])
    
    logger.info(f"Analyzing {len(sorted_targets)} time periods: "
               f"{[t['year'] for t in sorted_targets]}")
    
    # Calculate annual rates for each period
    periods = []
    for i in range(len(sorted_targets) - 1):
        start = sorted_targets[i]
        end = sorted_targets[i + 1]
        period_duration = end['year'] - start['year']
        
        if period_duration <= 0:
            logger.warning(f"Invalid period duration: {period_duration}")
            continue
        
        midpoint_year = start['year'] + period_duration / 2
        
        # Calculate transition matrix for this period
        _, counts, _ = calculate_transition_matrix(start['file'], end['file'])
        
        if counts is not None:
            # Annualize the counts
            annual_rate = counts / period_duration
            periods.append({
                'start_year': start['year'],
                'end_year': end['year'],
                'midpoint': midpoint_year,
                'duration': period_duration,
                'rates': pd.DataFrame(annual_rate, index=counts.index, columns=counts.columns)
            })
            logger.info(f"Period {start['year']}-{end['year']}: {period_duration} years")
    
    # Need at least 3 periods for trend fitting
    if len(periods) < 3:
        logger.error(f"Insufficient periods: {len(periods)} (need ≥2)")
        return None, None, []
    
    logger.info(f"Loaded {len(periods)} time periods for trend analysis")
    
    # Prepare data for trend fitting
    all_classes = sorted(list(set(
        cls for p in periods for cls in p['rates'].index
    )))
    
    # Initialize projected rates
    final_rates_template = periods[-1]['rates'].reindex(
        index=all_classes, columns=all_classes, fill_value=0.0
    )
    projected_annual_rates = pd.DataFrame(
        0.0, index=final_rates_template.index, columns=final_rates_template.columns
    )
    
    # Extract midpoint years for regression
    years = np.array([p['midpoint'] for p in periods])
    last_period_duration = periods[-1]['duration']
    future_midpoint = periods[-1]['midpoint'] + last_period_duration
    
    generated_plots = []
    plot_counter = 0
    model_choices = []
    
    progress_callback(0.0, f"📊 Fitting candidate trend models for {len(all_classes)}² transitions...")
    
    # Fit trend for each transition
    for idx, (from_cls, to_cls) in enumerate(
        [(f, t) for f in all_classes for t in all_classes if f != t]
    ):
        # Extract rate timeseries for this transition
        rate_timeseries = np.array([
            p['rates'].loc[from_cls, to_cls] 
            if (from_cls in p['rates'].index and to_cls in p['rates'].columns) 
            else 0.0
            for p in periods
        ])
        
        # Skip if no change
        if np.sum(rate_timeseries) == 0:
            continue
        
        try:
            fit = fit_transition_curve(years, rate_timeseries, future_midpoint)
            if fit is None:
                continue

            projected_annual_rates.loc[from_cls, to_cls] = fit["predicted_rate"]
            model_choices.append({
                "from": from_cls, "to": to_cls, "model": fit["model"],
                "predicted_rate": fit["predicted_rate"],
                **{f"aicc_{k}": v["aicc"] for k, v in fit["candidates"].items()},
            })

            # Generate plot for significant transitions (>100 px/yr average)
            if np.mean(rate_timeseries) > 100 and plot_counter < 20:  # Limit plots
                fig, ax = plt.subplots(figsize=(8, 5))

                ax.scatter(years, rate_timeseries, s=100, alpha=0.7, zorder=5,
                           label='Historical Annual Rate', color='black')

                x_smooth = np.linspace(years.min(), future_midpoint, 100)
                for name in ("linear", "exponential", "logistic"):
                    if name not in fit["candidates"]:
                        continue
                    # Re-derive a predict callable is awkward post-hoc; refit
                    # just for plotting continuity isn't needed -- reuse the
                    # winning model's predict function, and only overlay
                    # others as markers at the historical fit if it's the
                    # chosen one, to keep this readable rather than re-storing
                    # every candidate's full callable.
                    if name == fit["model"]:
                        ax.plot(x_smooth, fit["predict"](x_smooth), 'r--', linewidth=2,
                                label=f'Selected: {name}')

                ax.scatter([future_midpoint], [fit["predicted_rate"]], c='green', s=150,
                          zorder=6, label='Predicted Rate', marker='*')

                ax.set_title(f'Trend Analysis: Class {from_cls} → Class {to_cls} '
                             f'(model: {fit["model"]})')
                ax.set_xlabel('Year')
                ax.set_ylabel('Annual Pixel Change Rate')
                ax.legend(loc='best')
                ax.grid(True, alpha=0.3)

                generated_plots.append(fig)
                plot_counter += 1
                logger.info(f"Generated plot for transition {from_cls}→{to_cls} (model={fit['model']})")

        except Exception as e:
            logger.warning(f"Failed to fit trend for {from_cls}→{to_cls}: {e}")
            continue
        
        progress_callback((idx + 1) / (len(all_classes) ** 2), f"Fitting transition {from_cls}→{to_cls}")
    
    # Calculate total projected change over historical period
    total_historical_period = sorted_targets[-1]['year'] - sorted_targets[0]['year']
    projected_total_counts = (projected_annual_rates * total_historical_period).round().astype(int)
    
    # Get reference matrix from last period
    final_matrix, _, _ = calculate_transition_matrix(
        sorted_targets[-2]['file'], 
        sorted_targets[-1]['file']
    )

    if model_choices:
        n_logistic = sum(1 for m in model_choices if m["model"] == "logistic")
        n_exp = sum(1 for m in model_choices if m["model"] == "exponential")
        n_lin = sum(1 for m in model_choices if m["model"] == "linear")
        logger.info(f"Model selection across {len(model_choices)} transitions: "
                   f"{n_lin} linear, {n_exp} exponential, {n_logistic} logistic")
    
    logger.info(f"Non-linear analysis generated {len(generated_plots)} plots")
    
    return final_matrix, projected_total_counts, generated_plots


# ============================================================================
# INTENSITY ANALYSIS (Aldwaik & Pontius, 2012)
# ============================================================================
# Referenced in the Land Use methodology but never implemented. Answers three
# questions the Markov/RF/allocation pipeline doesn't: (1) is change faster or
# slower than the long-run average in a given interval? (2) which categories
# are gaining/losing faster than the study-area-wide average? (3) for a given
# gaining category, does it draw disproportionately from ("target") a
# particular source category, or avoid it, relative to what uniform random
# reallocation would produce? All intensities are annualized percentages so
# intervals of different length are directly comparable.
#
# Reference: Aldwaik, S.Z. & Pontius Jr, R.G. (2012). Intensity analysis to
# unify measurements of size and stationarity of land changes by interval,
# category and transition. Landscape and Urban Planning, 106(1), 103-114.

def interval_level_intensity(transition_matrices, durations, class_names=None):
    """
    Interval-level intensity analysis.

    Args:
        transition_matrices (list[pd.DataFrame]): one from->to pixel-count
            matrix per interval (as returned by calculate_transition_matrix's
            `counts` output), in chronological order.
        durations (list[float]): number of years in each interval, same order
            and length as transition_matrices.

    Returns:
        pd.DataFrame indexed by interval (0-based), columns:
            'duration_years', 'total_pixels', 'changed_pixels',
            'interval_intensity_pct_per_yr', 'faster_than_uniform'
        plus the scalar uniform annual intensity U (attached as `.attrs['U']`).
    """
    if len(transition_matrices) != len(durations):
        raise ValueError("transition_matrices and durations must be the same length")

    rows = []
    total_changed, total_duration_area = 0.0, 0.0
    total_pixels_ref = None

    for t, (m, dur) in enumerate(zip(transition_matrices, durations)):
        total_pixels = float(m.values.sum())
        changed = float(m.values.sum() - np.trace(m.values))
        s_t = 100.0 * changed / (dur * total_pixels) if dur > 0 and total_pixels > 0 else 0.0
        rows.append({
            "duration_years": dur, "total_pixels": total_pixels,
            "changed_pixels": changed, "interval_intensity_pct_per_yr": s_t,
        })
        total_changed += changed
        total_duration_area += dur * total_pixels
        total_pixels_ref = total_pixels  # assume constant study-area extent

    U = 100.0 * total_changed / total_duration_area if total_duration_area > 0 else 0.0
    df = pd.DataFrame(rows)
    df["faster_than_uniform"] = df["interval_intensity_pct_per_yr"] > U
    df.attrs["uniform_annual_intensity_pct_per_yr"] = U
    logger.info(f"📐 Intensity Analysis (interval level): uniform annual rate U = {U:.3f}%/yr")
    return df


def category_level_intensity(transition_matrices, durations):
    """
    Category-level intensity analysis: per-interval loss and gain intensity
    for every category, benchmarked against that interval's uniform rate S_t.

    Returns:
        pd.DataFrame with one row per (interval, category), columns:
            'interval', 'category', 'loss_intensity_pct_per_yr',
            'gain_intensity_pct_per_yr', 'interval_uniform_pct_per_yr',
            'actively_losing', 'actively_gaining'
        ('actively_x' = intensity exceeds that interval's uniform line, i.e.
        this category changes faster than the study-area average.)
    """
    interval_df = interval_level_intensity(transition_matrices, durations)
    rows = []

    for t, (m, dur) in enumerate(zip(transition_matrices, durations)):
        s_t = interval_df.loc[t, "interval_intensity_pct_per_yr"]
        classes = sorted(set(m.index) | set(m.columns))
        m = m.reindex(index=classes, columns=classes, fill_value=0)

        for cls in classes:
            initial_area = float(m.loc[cls, :].sum())     # Y_ti•
            final_area = float(m.loc[:, cls].sum())        # Y_t•i
            loss = float(m.loc[cls, :].sum() - m.loc[cls, cls])
            gain = float(m.loc[:, cls].sum() - m.loc[cls, cls])

            l_ti = 100.0 * loss / (dur * initial_area) if dur > 0 and initial_area > 0 else 0.0
            g_ti = 100.0 * gain / (dur * final_area) if dur > 0 and final_area > 0 else 0.0

            rows.append({
                "interval": t, "category": cls,
                "loss_intensity_pct_per_yr": l_ti,
                "gain_intensity_pct_per_yr": g_ti,
                "interval_uniform_pct_per_yr": s_t,
                "actively_losing": l_ti > s_t,
                "actively_gaining": g_ti > s_t,
            })

    return pd.DataFrame(rows)


def transition_level_intensity(transition_matrices, durations, target_categories=None):
    """
    Transition-level intensity analysis (gain-oriented): for each interval and
    each gaining category n, tests whether n draws disproportionately from a
    given source category m ("targets" m) or disproportionately little
    ("avoids" m), relative to the uniform rate that would result if n's gain
    were spread randomly over all categories other than n, weighted by their
    available area.

    Args:
        target_categories: optional list of category codes to restrict the
            (usually numerous) m->n pairs to the transitions actually of
            interest (e.g. the ones already selected for suitability modelling
            in Table 2), since the full m x n x interval table can be large.

    Returns:
        pd.DataFrame with columns:
            'interval', 'from', 'to', 'transition_intensity_pct_per_yr',
            'uniform_gain_intensity_pct_per_yr', 'targeted'
        ('targeted' = True means category `to` draws from `from` faster than
        the uniform/random-reallocation rate, i.e. this is a systematic,
        non-random transition rather than incidental churn.)
    """
    rows = []
    for t, (m, dur) in enumerate(zip(transition_matrices, durations)):
        classes = sorted(set(m.index) | set(m.columns))
        m = m.reindex(index=classes, columns=classes, fill_value=0)
        total_pixels = float(m.values.sum())

        for to_cls in classes:
            if target_categories and to_cls not in target_categories:
                continue
            final_area_n = float(m.loc[:, to_cls].sum())      # Y_t•n
            persistence_n = float(m.loc[to_cls, to_cls])
            # Uniform expectation: n's gain spread proportionally over every
            # category's area other than n itself.
            denom = dur * (total_pixels - final_area_n)
            u_tn = 100.0 * (final_area_n - persistence_n) / denom if denom > 0 else 0.0

            for from_cls in classes:
                if from_cls == to_cls:
                    continue
                if target_categories and from_cls not in target_categories:
                    continue
                y_tmn = float(m.loc[from_cls, to_cls])
                initial_area_m = float(m.loc[from_cls, :].sum())
                r_tmn = 100.0 * y_tmn / (dur * initial_area_m) if dur > 0 and initial_area_m > 0 else 0.0

                rows.append({
                    "interval": t, "from": from_cls, "to": to_cls,
                    "transition_intensity_pct_per_yr": r_tmn,
                    "uniform_gain_intensity_pct_per_yr": u_tn,
                    "targeted": r_tmn > u_tn,
                })

    return pd.DataFrame(rows)


def run_intensity_analysis(target_files_with_years, target_categories=None, progress_callback=None):
    """
    Convenience wrapper: builds one transition matrix per consecutive interval
    from a list of {'file', 'year'} targets, then runs all three Intensity
    Analysis levels.

    Returns:
        dict with keys 'interval', 'category', 'transition' (each a
        DataFrame) and 'uniform_annual_intensity_pct_per_yr' (float).
    """
    if progress_callback is None:
        def progress_callback(frac, text): pass

    sorted_targets = sorted(target_files_with_years, key=lambda x: x["year"])
    if len(sorted_targets) < 2:
        raise ValueError("Need at least 2 time points for Intensity Analysis")

    matrices, durations = [], []
    for i in range(len(sorted_targets) - 1):
        progress_callback(i / (len(sorted_targets) - 1), "Computing interval transition matrix")
        _, counts, _ = calculate_transition_matrix(sorted_targets[i]["file"], sorted_targets[i + 1]["file"])
        matrices.append(counts)
        durations.append(sorted_targets[i + 1]["year"] - sorted_targets[i]["year"])

    interval_df = interval_level_intensity(matrices, durations)
    category_df = category_level_intensity(matrices, durations)
    transition_df = transition_level_intensity(matrices, durations, target_categories)

    return {
        "interval": interval_df,
        "category": category_df,
        "transition": transition_df,
        "uniform_annual_intensity_pct_per_yr": interval_df.attrs["uniform_annual_intensity_pct_per_yr"],
    }


# ============================================================================
# CHANGE ANALYSIS DISPATCHER
# ============================================================================

def run_change_analysis(target_files_with_years, mode='auto', progress_callback=None):
    if len(target_files_with_years) < 2:
        logger.error("❌ Need at least 2 time periods")
        return None, None, [], "None"

    """
    Main dispatcher for change analysis.

    Automatically selects between linear and non-linear analysis based on
    number of time periods, unless overridden via `mode`.

    Args:
        target_files_with_years (list): List of dicts with 'file' and 'year'
        mode (str): 'auto' (default), 'linear', or 'nonlinear'. 'auto' picks
            Non-Linear only when 4+ maps are available (matches the
            ChangeAnalysisRequestSerializer contract in landuse_api).

    Returns:
        tuple: (transition_matrix_df, transition_counts_df, plots_list, analysis_mode)
    """

    sorted_targets = sorted(target_files_with_years, key=lambda x: x['year'])
    mode = (mode or 'auto').lower()
    if mode == 'linear':
        analysis_mode = "Linear Trend"
    elif mode == 'nonlinear':
        if len(sorted_targets) < 4:
            logger.warning("Non-linear mode requested with only %d time period(s); "
                            "curve fitting is less reliable below 4.", len(sorted_targets))
        analysis_mode = "Non-Linear Trend"
    else:
        # Trigger Non-Linear only if we have 4 or more maps (3 periods)
        analysis_mode = "Non-Linear Trend" if len(sorted_targets) >= 4 else "Linear Trend"

    logger.info(f"🔍 Running {analysis_mode} analysis with {len(sorted_targets)} time periods")
    
    if analysis_mode == "Linear Trend":
        period_years = sorted_targets[-1]['year'] - sorted_targets[0]['year']
        matrix, counts, plots = analyze_linear_trends(
            sorted_targets[0]['file'],
            sorted_targets[-1]['file'],
            period_years=period_years
        )
    else:  # Non-linear
        matrix, counts, plots = analyze_non_linear_trends(target_files_with_years, progress_callback)
    
    if matrix is None:
        logger.error("❌ Change analysis failed")
        return None, None, [], analysis_mode
    
    return matrix, counts, plots, analysis_mode