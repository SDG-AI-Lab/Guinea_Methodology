"""
Visualization Module

Handles interactive map displays and static export of land use analysis results:
1. Interactive folium maps with togglable layers (Categorical & Continuous)
2. Static map images for download (PNG, PDF)
3. Data export (GeoTIFF, CSV, JSON)
4. Comparison visualizations (baseline vs. prediction)
5. Legend generation and styling
"""

import rasterio
import numpy as np
import folium
from rasterio.enums import Resampling
from PIL import Image
import io
import base64
import matplotlib as mpl
# Headless backend: this module runs inside Celery workers (no display).
# Must be set before pyplot is imported anywhere in the process; change_analysis
# also sets this, but each module needing pyplot must not assume import order.
mpl.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import ListedColormap
import json
import os
import logging
import pandas as pd

from .data_loader import _open_as_raster
from .utils import get_raster_statistics
from . import config

logger = logging.getLogger(__name__)

# ============================================================================
# HELPER FUNCTIONS: RASTER CONVERSION
# ============================================================================

def _raster_to_png_overlay(raster_file, is_categorical=True, cmap_name='tab20', nodata_val=None):
    """
    Convert a raster file to a PNG image overlay for folium maps.
    Handles both categorical (LULC) and continuous (Population, Distance) data.
    """
    try:
        with _open_as_raster(raster_file) as src:
            bounds = [[src.bounds.bottom, src.bounds.left], 
                     [src.bounds.top, src.bounds.right]]
            
            if nodata_val is None:
                nodata_val = src.nodata if src.nodata is not None else -9999
            
            # Downscaling for memory efficiency
            max_dim = 1024
            scale_factor = max(src.width / max_dim, src.height / max_dim, 1)
            
            if scale_factor > 1:
                out_shape = (1, int(src.height / scale_factor), int(src.width / scale_factor))
                data = src.read(out_shape=out_shape, resampling=Resampling.nearest)[0]
            else:
                data = src.read(1)
            
            valid_mask = (data != nodata_val) & np.isfinite(data)
            valid_data = data[valid_mask]
            
            if len(valid_data) == 0:
                return None, bounds, [], {}, {}

            # Get colormap
            cmap = mpl.colormaps[cmap_name]
            
            colored_data = np.zeros((*data.shape, 4), dtype=np.uint8)
            
            if is_categorical:
                # CATEGORICAL LOGIC (Land Cover, Zoning)
                unique_vals = np.unique(valid_data)
                
                # Failsafe: if too many unique values, force continuous
                if len(unique_vals) > 50:
                    logger.warning("Too many unique values. Treating as continuous.")
                    is_categorical = False
                else:
                    cmap_resampled = cmap.resampled(len(unique_vals))
                    colors = {i: cmap_resampled(i / len(unique_vals)) for i in range(len(unique_vals))}
                    val_map = {val: i for i, val in enumerate(unique_vals)}
                    
                    for val in unique_vals:
                        mask = data == val
                        color_idx = val_map[val]
                        colored_data[mask] = (np.array(colors[color_idx][:4]) * 255).astype(np.uint8)

            if not is_categorical:
                # CONTINUOUS LOGIC (Population, Distance, Heatmaps)
                # Use percentiles to ignore extreme outliers
                vmin, vmax = np.nanpercentile(valid_data, 2), np.nanpercentile(valid_data, 98)
                
                # Prevent division by zero if all values are identical
                if vmax == vmin:
                    vmax = vmin + 1e-5
                    
                # Normalize data between 0 and 1
                norm_data = (data - vmin) / (vmax - vmin)
                norm_data = np.clip(norm_data, 0, 1)
                
                # Apply colormap directly to normalized array
                colored_rgba = cmap(norm_data)
                colored_data = (colored_rgba * 255).astype(np.uint8)
                
                # Placeholder returns for continuous legends
                unique_vals, colors, val_map = [], {}, {}
            
            # Set nodata to transparent
            colored_data[~valid_mask] = [0, 0, 0, 0]
            
            # Convert to PNG
            img = Image.fromarray(colored_data, 'RGBA')
            buf = io.BytesIO()
            img.save(buf, format='PNG')
            img_str = base64.b64encode(buf.getvalue()).decode()
            png_uri = f'data:image/png;base64,{img_str}'
            
            return png_uri, bounds, unique_vals, colors, val_map
            
    except Exception as e:
        logger.error(f"Error converting raster to PNG: {e}")
        raise


def _create_legend_html(class_legends, unique_vals, colors, val_map, max_classes=20):
    """
    Create HTML legend for categorical map display.
    """
    legend_html = f'''
    <div style="position: fixed; bottom: 50px; left: 50px; width: auto; 
    max-width: 300px; max-height: {min(max_classes * 25, 400)}px; 
    overflow-y: auto; padding: 10px; border: 2px solid grey; 
    z-index: 9999; font-size: 14px; background-color: white; 
    color: black; border-radius: 5px;">
    <b>🗺️ Land Cover Classes</b><br>
    '''
    
    sorted_vals = sorted(unique_vals)
    
    for val in sorted_vals:
        class_id = int(val)
        class_name = class_legends.get(class_id, f"Class {class_id}") if class_legends else f"Class {class_id}"
        
        color_idx = val_map.get(val, 0)
        color = colors.get(color_idx, (0, 0, 0, 1))
        color_hex = '#%02x%02x%02x' % tuple((np.array(color[:3]) * 255).astype(int))
        
        legend_html += f'<br><i class="fa fa-square" style="color:{color_hex}"></i>&nbsp;{class_name}'
    
    legend_html += '<br></div>'
    return legend_html


# ============================================================================
# INTERACTIVE MAP CREATION
# ============================================================================

def create_interactive_map(target_files_with_years=None, prediction_filepath=None, 
                          predictor_files=None, class_legends=None, initial_layer='prediction'):
    """
    Create an interactive folium map with multiple togglable layers.
    
    Args:
        target_files_with_years (list): Historical LC maps with years (Categorical)
        prediction_filepath (str): Path to predicted LC map (Categorical)
        predictor_files (list): Optional list of dicts with continuous predictor metadata
                                e.g., [{'file': path, 'name': 'Population', 'is_categorical': False}]
        class_legends (dict): {class_id: class_name}
        initial_layer (str): Layer name to show initially
    """
    try:
        # Determine reference file for map bounds
        ref_file = prediction_filepath or (target_files_with_years[0]['file'] 
                                          if target_files_with_years else None)
        
        if not ref_file:
            logger.warning("No raster files provided, creating blank map")
            return folium.Map(location=[0, 0], zoom_start=2)
        
        # Get map bounds and center
        with _open_as_raster(ref_file) as src:
            center_lat = (src.bounds.bottom + src.bounds.top) / 2
            center_lon = (src.bounds.left + src.bounds.right) / 2
            nodata_val = src.nodata
        
        logger.info(f"Map center: ({center_lat}, {center_lon})")
        
        # Create base map
        m = folium.Map(
            location=[center_lat, center_lon],
            zoom_start=10,
            tiles="CartoDB positron"
        )
        
        all_unique_vals = set()
        all_colors = {}
        all_val_maps = {}
        layer_info = []
        
        # 1. Add prediction layer (Categorical)
        if prediction_filepath:
            layer_info.append({
                'file': prediction_filepath,
                'name': 'Simulated Future',
                'show': initial_layer == 'prediction',
                'is_categorical': True,
                'cmap': 'tab20'
            })
        
        # 2. Add historical layers (Categorical)
        if target_files_with_years:
            for item in reversed(target_files_with_years):
                layer_info.append({
                    'file': item['file'],
                    'name': f"Historical {item['year']}",
                    'show': (initial_layer == str(item['year']) or initial_layer == item['year']),
                    'is_categorical': True,
                    'cmap': 'tab20'
                })

        # 3. Add predictor layers (Continuous/Various)
        if predictor_files:
            for pred in predictor_files:
                if isinstance(pred, dict):
                    p_file = pred.get('file')
                    p_name = pred.get('name', os.path.basename(p_file))
                    is_cat = pred.get('is_categorical', False)
                    cmap = pred.get('cmap', 'viridis')
                else:
                    p_file = pred
                    p_name = os.path.basename(pred)
                    is_cat = False
                    cmap = 'viridis'

                layer_info.append({
                    'file': p_file,
                    'name': f"Predictor: {p_name}",
                    'show': False, # Hide by default to prevent map clutter
                    'is_categorical': is_cat,
                    'cmap': cmap
                })
        
        logger.info(f"Adding {len(layer_info)} layers to map")
        
        # Add all compiled layers to the map
        for layer in layer_info:
            try:
                png_uri, bounds, unique_vals, colors, val_map = _raster_to_png_overlay(
                    layer['file'], 
                    is_categorical=layer['is_categorical'],
                    cmap_name=layer['cmap'],
                    nodata_val=nodata_val
                )
                
                # Only aggregate legends for categorical maps
                if layer['is_categorical']:
                    all_unique_vals.update(unique_vals)
                    all_colors.update(colors)
                    all_val_maps.update(val_map)
                
                folium.raster_layers.ImageOverlay(
                    image=png_uri,
                    bounds=bounds,
                    opacity=0.8,
                    name=layer['name'],
                    show=layer['show']
                ).add_to(m)
                
                logger.info(f"Added layer: {layer['name']}")
            
            except Exception as e:
                logger.error(f"Failed to add layer {layer['name']}: {e}")
                continue
        
        # Add legend strictly for categorical LULC classes
        if all_unique_vals:
            legend_html = _create_legend_html(class_legends, 
                                            np.array(sorted(all_unique_vals)),
                                            all_colors, all_val_maps)
            m.get_root().html.add_child(folium.Element(legend_html))
            logger.info("Added categorical legend to map")
        
        # Add layer control
        folium.LayerControl(position='topright', collapsed=False).add_to(m)
        
        logger.info("Map creation complete")
        return m
    
    except Exception as e:
        logger.error(f"Error creating interactive map: {e}", exc_info=True)
        return folium.Map(location=[0, 0], zoom_start=2)


# ============================================================================
# STATIC MAP EXPORT
# ============================================================================

def create_downloadable_map_image(filepath, class_legends=None, title="Map Export", 
                                  is_categorical=True, cmap_name='tab20', dpi=150):
    """
    Create a high-quality static map image for download.
    """
    try:
        with _open_as_raster(filepath) as src:
            data = src.read(1)
            nodata = src.nodata if src.nodata is not None else -9999
        
        logger.info(f"Creating map image: {data.shape}, DPI={dpi}")
        
        valid_mask = (data != nodata) & np.isfinite(data)
        valid_data = data[valid_mask]
        
        fig, ax = plt.subplots(figsize=(12, 10), dpi=dpi)
        
        if is_categorical:
            unique_vals = np.unique(valid_data)
            n_cls = max(len(unique_vals), 1)
            cmap = mpl.colormaps[cmap_name].resampled(n_cls)
            
            im = ax.imshow(data, cmap=cmap, interpolation='nearest')
            
            # Create categorical legend
            if class_legends or len(unique_vals) > 0:
                patches = []
                for idx, val in enumerate(unique_vals):
                    class_id = int(val)
                    class_name = class_legends.get(class_id, f'Class {class_id}') if class_legends else f'Class {class_id}'
                    patches.append(mpatches.Patch(
                        color=cmap(idx / len(unique_vals)),
                        label=class_name
                    ))
                ax.legend(handles=patches, bbox_to_anchor=(1.05, 1), 
                         loc='upper left', fontsize=10, framealpha=0.9)
                         
        else:
            # Continuous logic rendering for exports
            vmin, vmax = np.nanpercentile(valid_data, 2), np.nanpercentile(valid_data, 98)
            cmap = mpl.colormaps[cmap_name]
            im = ax.imshow(data, cmap=cmap, vmin=vmin, vmax=vmax)
            plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="Value")

        ax.set_title(title, fontsize=18, fontweight='bold', pad=20)
        ax.set_xlabel("Longitude (pixels)", fontsize=12)
        ax.set_ylabel("Latitude (pixels)", fontsize=12)
        
        fig.tight_layout()
        
        buf = io.BytesIO()
        fig.savefig(buf, format='png', bbox_inches='tight', dpi=dpi)
        buf.seek(0)
        plt.close(fig)
        
        logger.info(f"Generated PNG: {len(buf.getvalue())} bytes")
        return buf.getvalue()
    
    except Exception as e:
        logger.error(f"Error creating map image: {e}", exc_info=True)
        raise


def create_pdf_report(prediction_filepath, transition_counts=None, 
                     model_accuracies=None, class_legends=None, 
                     title="Land Use Change Simulation Report"):
    """
    Create a comprehensive PDF report with maps and statistics.
    """
    try:
        from matplotlib.backends.backend_pdf import PdfPages
        
        logger.info("Creating PDF report")
        
        buf = io.BytesIO()
        
        with PdfPages(buf) as pdf:
            # Page 1: Title and summary
            fig = plt.figure(figsize=(8.5, 11))
            fig.text(0.5, 0.9, title, ha='center', fontsize=20, fontweight='bold')
            fig.text(0.5, 0.85, "Simulation Results Summary", ha='center', fontsize=14)
            
            summary_text = f"""
            Prediction File: {os.path.basename(prediction_filepath)}
            
            Report Generated: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}
            """
            fig.text(0.1, 0.7, summary_text, fontsize=10, family='monospace',
                    verticalalignment='top')
            
            if model_accuracies:
                acc_text = "Model Accuracies:\n"
                for transition, acc in sorted(model_accuracies.items()):
                    acc_text += f"  {transition}: {acc:.2%}\n"
                fig.text(0.1, 0.5, acc_text, fontsize=9, family='monospace',
                        verticalalignment='top')
            
            pdf.savefig(fig, bbox_inches='tight')
            plt.close(fig)
            
            # Page 2: Map
            try:
                map_img = create_downloadable_map_image(
                    prediction_filepath, class_legends, title, is_categorical=True
                )
                map_pil = Image.open(io.BytesIO(map_img))
                
                fig, ax = plt.subplots(figsize=(8.5, 11))
                ax.imshow(map_pil)
                ax.axis('off')
                
                pdf.savefig(fig, bbox_inches='tight')
                plt.close(fig)
            except Exception as e:
                logger.warning(f"Could not add map to PDF: {e}")
            
            # Page 3: Transition table (if available)
            if transition_counts is not None:
                fig, ax = plt.subplots(figsize=(8.5, 11))
                ax.axis('off')
                
                # Create table
                table_data = transition_counts.values
                colLabels = [f"→{c}" for c in transition_counts.columns]
                rowLabels = [f"{r}→" for r in transition_counts.index]
                
                table = ax.table(cellText=table_data, colLabels=colLabels,
                               rowLabels=rowLabels, cellLoc='center',
                               loc='center', bbox=[0, 0, 1, 1])
                table.auto_set_font_size(False)
                table.set_fontsize(8)
                
                ax.set_title("Transition Counts (pixels)", fontsize=14, pad=20)
                
                pdf.savefig(fig, bbox_inches='tight')
                plt.close(fig)
        
        buf.seek(0)
        logger.info(f"Generated PDF: {len(buf.getvalue())} bytes")
        return buf.getvalue()
    
    except Exception as e:
        logger.error(f"Error creating PDF: {e}", exc_info=True)
        raise


# ============================================================================
# DATA EXPORT
# ============================================================================

def export_raster_geotiff(raster_filepath, output_path):
    """
    Copy raster to GeoTIFF format (lossless).
    """
    try:
        with _open_as_raster(raster_filepath) as src:
            data = src.read()
            profile = src.profile
        
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        with rasterio.open(output_path, 'w', **profile) as dst:
            dst.write(data)
        
        logger.info(f"Saved GeoTIFF: {output_path}")
        return output_path
    
    except Exception as e:
        logger.error(f"Error exporting GeoTIFF: {e}")
        raise


def export_transition_table_csv(transition_counts, output_path):
    """
    Export transition counts as CSV.
    """
    try:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        transition_counts.to_csv(output_path)
        logger.info(f"Saved CSV: {output_path}")
        return output_path
    except Exception as e:
        logger.error(f"Error exporting CSV: {e}")
        raise


def export_model_metrics_json(model_accuracies, output_path):
    """
    Export model metrics as JSON.
    """
    try:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # Convert numpy types to native Python types for JSON serialization
        metrics = {k: float(v) if isinstance(v, (np.floating, np.integer)) else v 
                  for k, v in model_accuracies.items()}
        
        with open(output_path, 'w') as f:
            json.dump(metrics, f, indent=4)
        
        logger.info(f"Saved metrics JSON: {output_path}")
        return output_path
    except Exception as e:
        logger.error(f"Error exporting JSON: {e}")
        raise


def export_scenario_summary(scenario_name, transition_counts, predictions_path,
                           class_legends, output_dir):
    """
    Export complete scenario summary (all formats).
    """
    try:
        os.makedirs(output_dir, exist_ok=True)
        
        outputs = {}
        
        # GeoTIFF
        tif_path = os.path.join(output_dir, f"{scenario_name}_prediction.tif")
        outputs['geotiff'] = export_raster_geotiff(predictions_path, tif_path)
        
        # CSV
        csv_path = os.path.join(output_dir, f"{scenario_name}_transitions.csv")
        outputs['csv'] = export_transition_table_csv(transition_counts, csv_path)
        
        # PNG
        png_path = os.path.join(output_dir, f"{scenario_name}_map.png")
        png_data = create_downloadable_map_image(
            predictions_path, class_legends, title=scenario_name, is_categorical=True
        )
        with open(png_path, 'wb') as f:
            f.write(png_data)
        outputs['png'] = png_path
        
        # PDF
        try:
            pdf_path = os.path.join(output_dir, f"{scenario_name}_report.pdf")
            pdf_data = create_pdf_report(
                predictions_path, transition_counts,
                class_legends=class_legends, title=scenario_name
            )
            with open(pdf_path, 'wb') as f:
                f.write(pdf_data)
            outputs['pdf'] = pdf_path
        except Exception as e:
            logger.warning(f"Could not generate PDF: {e}")
        
        logger.info(f"Exported scenario summary: {len(outputs)} formats")
        return outputs
    
    except Exception as e:
        logger.error(f"Error exporting scenario summary: {e}", exc_info=True)
        raise


# ============================================================================
# COMPARISON VISUALIZATIONS
# ============================================================================

def create_change_map(baseline_filepath, prediction_filepath, class_legends=None):
    """
    Create a map showing only pixels that changed.
    """
    try:
        with _open_as_raster(baseline_filepath) as src:
            baseline = src.read(1)
        with _open_as_raster(prediction_filepath) as src:
            prediction = src.read(1)
        
        # Create change map
        changed_mask = baseline != prediction
        change_data = np.zeros_like(baseline, dtype=np.int32)
        change_data[changed_mask] = 1
        
        # Plot
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        
        # Baseline
        axes[0].imshow(baseline, cmap='tab20')
        axes[0].set_title('Baseline Land Cover')
        axes[0].axis('off')
        
        # Changes only
        cmap = ListedColormap(['white', 'red'])
        axes[1].imshow(change_data, cmap=cmap)
        axes[1].set_title(f'Changes ({np.sum(changed_mask)} pixels)')
        axes[1].axis('off')
        
        fig.tight_layout()
        
        buf = io.BytesIO()
        fig.savefig(buf, format='png', bbox_inches='tight', dpi=150)
        buf.seek(0)
        plt.close(fig)
        
        return buf.getvalue()
    
    except Exception as e:
        logger.error(f"Error creating change map: {e}")
        raise