import streamlit as st
import geopandas as gpd
from shapely.geometry import box
import os

# Import your modules
from landuse_tool import config, api_connector, data_loader, change_analysis, training, prediction

st.title("🧪 Backend Pipeline Test Runner")

# Ensure your actual Earth Engine enabled Google Cloud Project ID is set here
config.PROCESSING_CONFIG["gee_project_id"] = "tribal-mapper-275801" 

if st.button("Run Full Pipeline"):
    
    # 1. Initialize Directories and GEE using your Project ID
    config.ensure_directories()
    
    st.info("🔄 Initializing Google Earth Engine...")
    api_connector.initialize_gee(project_id=config.PROCESSING_CONFIG["gee_project_id"])
    
    # 2. Setup a reusable UI progress callback for your refactored modules
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    def ui_progress_callback(fraction, message):
        progress_bar.progress(float(fraction))
        status_text.text(message)
    
    # 3. Create a Dummy ROI (Bounding Box for testing)
    minx, miny, maxx, maxy = -13.7, 9.5, -13.5, 9.7 
    test_bbox = box(minx, miny, maxx, maxy)
    roi_gdf = gpd.GeoDataFrame({'geometry': [test_bbox]}, crs="EPSG:4326")
    
    # 4. Fetch Targets (Passing the progress callback)
    st.header("1. Fetching Data")
    t1_path = api_connector.fetch_gee_raster(
        config.TARGET_RASTERS_CONFIG["DYNAMIC_WORLD"]["collection"], 
        roi_gdf, "lc_2015.tif", "2015-01-01", "2015-12-31", reducer='mode'
    )
    t2_path = api_connector.fetch_gee_raster(
        config.TARGET_RASTERS_CONFIG["DYNAMIC_WORLD"]["collection"], 
        roi_gdf, "lc_2020.tif", "2020-01-01", "2020-12-31", reducer='mode'
    )
    
    # 5. Run Data Loader to validate targets (Using progress callback)
    # This calls your newly refactored data_loader function!
    target_files = [{'file': t1_path, 'year': 2015}, {'file': t2_path, 'year': 2020}]
    profiles_dict, masks_dict, boundary_mask = data_loader.load_targets(
        [t1_path, t2_path], 
        align_boundary=True, 
        progress_callback=ui_progress_callback
    )
    
    # 6. Change Analysis (Using progress callback)
    st.header("2. Change Analysis")
    matrix, counts, plots, mode = change_analysis.run_change_analysis(
        target_files, 
        progress_callback=ui_progress_callback
    )
    st.dataframe(counts)
    
    st.success("Pipeline test complete!")