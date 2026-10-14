// =============================================================================
// DPWMS — Module 2: Mesoscale Plastic Hotspot Mapping
// Floating Debris Index (FDI) via Sentinel-2 + GEE App Interface
//
// Reference: Biermann et al. (2020), Sci. Rep. 10, 5364
// SRS: DPWMS-FR-0201 to DPWMS-FR-0207
// CRS: WGS84 / EPSG:4326
// =============================================================================

// =============================================================================
// 0. CONFIGURATION — edit these values before running
// =============================================================================
var CONFIG = {

  // --- Study area: Guinea coastal + riverine zone ---
  // Bounding box covers Conakry peninsula, Konkouré/Fatala river mouths,
  // mangrove fringe, and Atlantic coastal accumulation zones
  AOI: ee.Geometry.BBox(-15.5, 8.8, -13.0, 10.8),

  // --- Sentinel-2 collection ---
  S2_COLLECTION: 'COPERNICUS/S2_SR_HARMONIZED',

  // --- Default date window (can be changed via UI sliders) ---
  START_DATE: '2024-01-01',
  END_DATE:   '2024-01-31',

  // --- Cloud masking ---
  CLOUD_THRESHOLD: 70,   // tiles with > this % cloud cover → SAR alert (FR-0207)

  // --- Spectral thresholds ---
  // Otsu is computed automatically, but a manual fallback is available
  FDI_MANUAL_THRESHOLD: 0.005,  // fallback if Otsu fails on sparse scenes

  // --- Mask thresholds ---
  NDVI_THRESHOLD: 0.15,  // pixels above → vegetation → excluded
  NDWI_THRESHOLD: 0.0,   // pixels above → open water → excluded

  // --- Minimum patch area for vectorization (sq metres) ---
  MIN_PATCH_AREA_M2: 500,

  // --- Band names (S2 SR Harmonized) ---
  BANDS: {
    B3:  'B3',   // Green   (~560 nm)
    B4:  'B4',   // Red     (~665 nm)
    B8:  'B8',   // NIR     (~842 nm)
    B8A: 'B8A',  // Red-edge (~865 nm)
    B11: 'B11',  // SWIR-1  (~1610 nm)
    B12: 'B12'   // SWIR-2  (~2190 nm)
  },

  // --- Central wavelengths (nm) — used in FDI formula ---
  LAMBDA: {
    B8:  832.8,
    B8A: 864.7,
    B11: 1613.7
  },

  // --- Visualization palettes ---
  VIZ: {
    FDI:      {min: -0.05, max: 0.05, palette: ['#042C53','#185FA5','#85B7EB','#E6F1FB','#FAEEDA','#BA7517','#633806']},
    NDVI:     {min: -0.3,  max: 0.7,  palette: ['#4A1B0C','#E8593C','#FAEEDA','#639922','#173404']},
    NDWI:     {min: -0.5,  max: 0.5,  palette: ['#4A1B0C','#E8593C','#E6F1FB','#185FA5','#042C53']},
    HOTSPOT:  {palette: ['#E24B4A']},
    RGB:      {bands: ['B4','B3','B2'], min: 0, max: 3000, gamma: 1.4},
    FALSE_CLR:{bands: ['B8','B4','B3'], min: 0, max: 5000, gamma: 1.2}
  }
};


// =============================================================================
// 1. HELPER FUNCTIONS
// =============================================================================

/**
 * Applies Sentinel-2 cloud/shadow masking using SCL band
 * (Scene Classification Layer — available in S2_SR_HARMONIZED)
 */
function maskS2clouds(image) {
  var scl = image.select('SCL');
  // SCL classes to keep: 4=Vegetation, 5=Not-vegetated, 6=Water, 11=Snow
  // Exclude: 2=Dark area, 3=Cloud shadow, 7=Unclassified, 8=Cloud med, 9=Cloud high, 10=Thin cirrus
  var clearMask = scl.neq(2).and(scl.neq(3)).and(scl.neq(7))
                             .and(scl.neq(8)).and(scl.neq(9)).and(scl.neq(10));
  return image.updateMask(clearMask)
              .divide(10000)  // scale SR to 0–1 reflectance
              .copyProperties(image, ['system:time_start']);
}

/**
 * Computes FDI per image
 * FDI = B8 − [B8A + (B11 − B8A) × (λ_B8 − λ_B8A)/(λ_B11 − λ_B8A) × 10]
 * Biermann et al. (2020), Sci. Rep.
 */
function computeFDI(image) {
  var B8  = image.select(CONFIG.BANDS.B8);
  var B8A = image.select(CONFIG.BANDS.B8A);
  var B11 = image.select(CONFIG.BANDS.B11);

  // Wavelength interpolation factor
  var wl_factor = (CONFIG.LAMBDA.B8 - CONFIG.LAMBDA.B8A) /
                  (CONFIG.LAMBDA.B11 - CONFIG.LAMBDA.B8A);

  // FDI formula
  var fdi = B8.subtract(
    B8A.add(
      B11.subtract(B8A).multiply(wl_factor).multiply(10)
    )
  ).rename('FDI');

  return image.addBands(fdi);
}

/**
 * Computes NDVI: (B8 - B4) / (B8 + B4)
 */
function computeNDVI(image) {
  var ndvi = image.normalizedDifference([CONFIG.BANDS.B8, CONFIG.BANDS.B4])
                  .rename('NDVI');
  return image.addBands(ndvi);
}

/**
 * Computes NDWI: (B3 - B8) / (B3 + B8)
 * Modified NDWI to isolate water; high values = open water
 */
function computeNDWI(image) {
  var ndwi = image.normalizedDifference([CONFIG.BANDS.B3, CONFIG.BANDS.B8])
                  .rename('NDWI');
  return image.addBands(ndwi);
}

/**
 * Applies NDVI + NDWI masks to isolate plastic spectral signature:
 *   - Exclude vegetation (NDVI > threshold)
 *   - Exclude open water (NDWI > threshold)
 * Residual = floating debris / coastal plastic accumulation
 */

function applySpectralMasks(image, ndviThresh, ndwiThresh) {
  var ndvi = image.select('NDVI');
  var ndwi = image.select('NDWI');

  var vegMask = ndvi.lt(ndviThresh);

  // Keep transitional + weak water (better for debris)
  var waterMask = ndwi.lt(0.1);

  return image.updateMask(vegMask.and(waterMask));
}

/**
 * Otsu thresholding on a single-band image within an AOI
 * Returns the optimal threshold value
 */
function otsuThreshold(image, band, aoi, scale) {
  scale = scale || 100;  // default 100m for efficiency
  var histogram = image.select(band).reduceRegion({
    reducer: ee.Reducer.autoHistogram({maxBuckets: 256, minBucketWidth: 0.0001}),
    geometry: aoi,
    scale: scale,
    maxPixels: 1e9,
    bestEffort: true
  });

  var histArray = ee.Array(ee.List(histogram.get(band)));
  var counts = histArray.slice(1, 0, 1).project([0]);   // bin counts
  var means  = histArray.slice(1, 1, 2).project([0]);   // bin centers

  var total  = counts.reduce(ee.Reducer.sum(), [0]).get([0]);

  // Otsu criterion: maximize inter-class variance
  var indexes = ee.List.sequence(1, counts.length().get([0]).subtract(1));
  var initValues = ee.Dictionary({threshold: -1, maxVar: -1});

  var result = indexes.iterate(function(i, prev) {
    i = ee.Number(i);
    var dict = ee.Dictionary(prev);

    var w0 = counts.slice(0, 0, i).reduce(ee.Reducer.sum(), [0]).get([0])
                   .divide(total);
    var w1 = ee.Number(1).subtract(w0);

    var m0 = means.slice(0, 0, i).multiply(counts.slice(0, 0, i))
                  .reduce(ee.Reducer.sum(), [0]).get([0])
                  .divide(counts.slice(0, 0, i).reduce(ee.Reducer.sum(), [0]).get([0]));
    var m1 = means.slice(0, i).multiply(counts.slice(0, i))
                  .reduce(ee.Reducer.sum(), [0]).get([0])
                  .divide(counts.slice(0, i).reduce(ee.Reducer.sum(), [0]).get([0]));

    var interClassVar = w0.multiply(w1).multiply(m0.subtract(m1).pow(2));
    var betterThreshold = interClassVar.gt(dict.get('maxVar'));

    return ee.Dictionary({
      threshold: betterThreshold.multiply(means.get([i])).add(
        ee.Number(1).subtract(betterThreshold).multiply(dict.get('threshold'))
      ),
      maxVar: betterThreshold.multiply(interClassVar).add(
        ee.Number(1).subtract(betterThreshold).multiply(dict.get('maxVar'))
      )
    });
  }, initValues);

  return ee.Number(ee.Dictionary(result).get('threshold'));
}

/**
 * Vectorizes a binary raster into polygons and computes attributes
 */
function vectorizeHotspots(binaryImage, aoi, acquDate, cloudPct) {
  // Convert binary mask to vectors
  var vectors = binaryImage.selfMask()
    .reduceToVectors({
      reducer: ee.Reducer.countEvery(),
      geometry: aoi,
      scale: 20,           // S2 native 10m → 20m for efficiency
      maxPixels: 1e9,
      bestEffort: true,
      geometryType: 'polygon',
      eightConnected: false
    });

  // Filter by minimum patch area (FR-0204)
  vectors = vectors.filter(
    ee.Filter.gte('count', CONFIG.MIN_PATCH_AREA_M2 / 400) // 20m pixel = 400m²
  );

  // Add metadata attributes (FR-0204)
  vectors = vectors.map(function(feat) {
    var area_m2 = feat.geometry().area(1);
    // Plastic concentration score: normalized by max expected patch (1 km²)
    var score = area_m2.divide(1e6).min(1.0);
    return feat
      .set('plastic_concentration_score', score)
      .set('acquisition_date', acquDate)
      .set('cloud_cover_pct', cloudPct)
      .set('confidence_level', ee.String('MEDIUM'))  // upgradeable with validation
      .set('area_m2', area_m2)
      .set('module', 'M2_FDI')
      .set('source', 'Sentinel-2_SR_Harmonized');
  });

  return vectors;
}

/**
 * Check average cloud cover for AOI and raise SAR alert flag (FR-0207)
 */
function checkCloudCover(collection, aoi) {
  var meanCloud = collection.aggregate_mean('CLOUDY_PIXEL_PERCENTAGE');
  return meanCloud;
}


// =============================================================================
// 2. MAIN ANALYSIS PIPELINE
// =============================================================================

/**
 * Runs the full M2 pipeline for a given date range
 * Returns: { composite, fdi, ndvi, ndwi, maskedFDI, hotspotBinary, hotspotVectors, threshold }
 */
function runM2Pipeline(startDate, endDate, ndviThresh, ndwiThresh, aoi) {
  aoi = aoi || CONFIG.AOI;

  // --- 2.1 Load & filter Sentinel-2 collection ---
  var s2Raw = ee.ImageCollection(CONFIG.S2_COLLECTION)
    .filterDate(startDate, endDate)
    .filterBounds(aoi)
    .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', CONFIG.CLOUD_THRESHOLD));

  var collectionSize = s2Raw.size();

  // --- 2.2 Cloud mask + scale ---
  var s2Masked = s2Raw;//.map(maskS2clouds);

  // --- 2.3 Monthly median composite (reduces shadow/cloud residuals) ---
  var composite = s2Masked.median().clip(aoi);

  // --- 2.4 Compute spectral indices ---
  composite = computeFDI(composite);
  composite = computeNDVI(composite);
  composite = computeNDWI(composite);

  // --- 2.5 Apply vegetation + water masks ---
  var maskedFDI = applySpectralMasks(composite, ndviThresh, ndwiThresh)
                    .select('FDI');

  // --- 2.6 Otsu thresholding on masked FDI ---
  var otsuT = otsuThreshold(maskedFDI, 'FDI', aoi, 100);
  // Use manual fallback if Otsu produces no valid threshold
  var threshold = ee.Algorithms.If(
  otsuT,
  otsuT,
  CONFIG.FDI_MANUAL_THRESHOLD
  );
  threshold = ee.Number(threshold);

  // --- 2.7 Binary hotspot classification ---
  var hotspotBinary = maskedFDI.gt(threshold).rename('hotspot');

  // --- 2.8 Vectorize patches ---
  var acquDate   = startDate;
  var cloudPct   = s2Raw.aggregate_mean('CLOUDY_PIXEL_PERCENTAGE');
  var hotspotVec = vectorizeHotspots(hotspotBinary, aoi, acquDate, cloudPct);

  return {
    composite:       composite,
    fdi:             composite.select('FDI'),
    ndvi:            composite.select('NDVI'),
    ndwi:            composite.select('NDWI'),
    maskedFDI:       maskedFDI,
    hotspotBinary:   hotspotBinary,
    hotspotVectors:  hotspotVec,
    threshold:       threshold,
    collectionSize:  collectionSize,
    cloudPct:        cloudPct
  };
}


// =============================================================================
// 3. GEE APP USER INTERFACE
// =============================================================================

// --- UI State ---
var state = {
  currentLayers: [],
  results: null,
  isRunning: false
};

// --- Color Palette (DPWMS brand) ---
var UI = {
  DARK_BG:    '#0D1B2A',
  PANEL_BG:   '#112233',
  ACCENT:     '#1D9E75',   // teal-600 from design system
  ACCENT_LITE:'#5DCAA5',   // teal-400
  WARN:       '#E24B4A',   // red-400
  AMBER:      '#EF9F27',   // amber-400
  TEXT:       '#E8EDF2',
  MUTED:      '#8FA6BC',
  BORDER:     '#1E3A52',
  WHITE:      '#FFFFFF'
};

// -----------------------------------------------------------------------
// 3.1 — Map Setup
// -----------------------------------------------------------------------
Map.setCenter(-14.6, 9.6, 9);   // Conakry, Guinea
Map.setOptions('HYBRID');

// -----------------------------------------------------------------------
// 3.2 — Side Panel: Main Container
// -----------------------------------------------------------------------
var panel = ui.Panel({
  layout: ui.Panel.Layout.flow('vertical'),
  style: {
    width: '320px',
    padding: '0px',
    backgroundColor: '#FFFFFF',
    position: 'top-left'
  }
});

// --- Header ---
var header = ui.Panel({
  style: {
  backgroundColor: '#FFFFFF',
  padding: '14px 16px',
  margin: '0px 0px 8px 0px'
  }
});

var titleLabel = ui.Label({
  value: '◉  Digital Plastic Waste Management System',
  style: {
    fontSize: '13px',
    fontWeight: 'bold',
    color: '#000000',
    margin: '0 0 2px 0'
  }
});

var subtitleLabel = ui.Label({
  value: 'Mesoscale Plastic Hotspot Mapping',
  style: {
    fontSize: '11px',
    color: '#000000',
    margin: '0 0 6px 0'
  }
});

var tagRow = ui.Panel({
  layout: ui.Panel.Layout.flow('horizontal'),
  style: {margin: '0px', padding: '0px'}
});
tagRow.add(makeTag('Sentinel-2 FDI', UI.ACCENT));
tagRow.add(makeTag('Otsu Thresh.', UI.AMBER));
tagRow.add(makeTag('Guinea', UI.MUTED));

header.add(titleLabel);
header.add(subtitleLabel);
header.add(tagRow);
panel.add(header);

// --- Section: Date Range ---
panel.add(makeSectionLabel('Date Range'));

var startYearSlider = makeSlider('Year', 2020, 2024, 2024, 1);
var startMonthSlider = makeSlider('Month', 1, 12, 1, 1);
var durationSlider  = makeSlider('Duration (months)', 1, 3, 1, 1);
panel.add(startYearSlider.panel);
panel.add(startMonthSlider.panel);
panel.add(durationSlider.panel);

// --- Section: Spectral Parameters ---
panel.add(makeSectionLabel('Spectral Parameters'));

var ndviSlider = makeSlider('NDVI threshold', 0.0, 0.4, 0.15, 0.01);
var ndwiSlider = makeSlider('NDWI threshold', -0.2, 0.3, 0.0, 0.01);
var cloudSlider = makeSlider('Max cloud cover %', 10, 90, 70, 5);
panel.add(ndviSlider.panel);
panel.add(ndwiSlider.panel);
panel.add(cloudSlider.panel);

// --- Section: Area of Interest ---
panel.add(makeSectionLabel('Area of Interest'));

var aoiSelect = ui.Select({
  items: [
    {label: 'Guinea — Full coastal zone', value: 'full'},
    {label: 'Conakry peninsula', value: 'conakry'},
    {label: 'Konkouré river mouth', value: 'konkouré'},
    {label: 'Fatala river mouth', value: 'fatala'},
    {label: 'Mangrove fringe (south)', value: 'mangrove'}
  ],
  value: 'full',
  style: {
    width: '100%',
    color: 'black',
    margin: '0 16px 10px 16px',
    border: '1px solid #2A4A66'
  }
});
panel.add(aoiSelect);

// --- Section: Display Layers ---
panel.add(makeSectionLabel('Display Layers'));

var showRGB       = makeCheckbox('True color (RGB)', true);
var showFalseClr  = makeCheckbox('False color (NIR)', false);
var showFDI       = makeCheckbox('FDI raster', true);
var showNDVI      = makeCheckbox('NDVI mask', false);
var showNDWI      = makeCheckbox('NDWI mask', false);
var showMasked    = makeCheckbox('Masked FDI', false);
var showHotspot   = makeCheckbox('Hotspot binary', true);
var showVectors   = makeCheckbox('Hotspot polygons', true);

panel.add(showRGB.panel);
panel.add(showFalseClr.panel);
panel.add(showFDI.panel);
panel.add(showNDVI.panel);
panel.add(showNDWI.panel);
panel.add(showMasked.panel);
panel.add(showHotspot.panel);
panel.add(showVectors.panel);

// --- Run Button ---
var runButton = ui.Button({
  label: '▶  Run M2 Analysis',
  style: {
    width: '288px',
    height: '40px',
    backgroundColor: UI.ACCENT,
    color: 'black',
    border: 'none',
    margin: '10px 16px 8px 16px',
    fontWeight: 'bold',
    fontSize: '13px'
  },
  onClick: onRunClicked
});
panel.add(runButton);

// --- Clear Button ---
var clearButton = ui.Button({
  label: '✕  Clear layers',
  style: {
    width: '288px',
    height: '32px',
    color: UI.MUTED,
    border: '1px solid ' + UI.BORDER,
    margin: '0px 16px 8px 16px',
    fontSize: '12px'
  },
  onClick: function() {
    Map.layers().reset();
    Map.layers().add(ui.Map.Layer(
      ee.Geometry(getAOI(aoiSelect.getValue())),
      {color: UI.ACCENT, fillColor: '00000000', width: 1.5},
      'AOI boundary'
    ));
    statusPanel.style().set('shown', false);
    resultsPanel.style().set('shown', false);
  }
});
panel.add(clearButton);

// --- Status Panel ---
var statusPanel = ui.Panel({
  style: {
    padding: '8px 16px',
    margin: '0 0 0 0',
    border: '1px solid ' + UI.BORDER,
    shown: false
  }
});
var statusLabel = ui.Label({
  value: '',
  style: {fontSize: '11px', color: '#000000', margin: '0px'}
});
statusPanel.add(statusLabel);
panel.add(statusPanel);

// --- Results Panel ---
var resultsPanel = ui.Panel({
  style: {
    padding: '10px 16px 12px 16px',
    margin: '0 0 0 0',
    shown: false
  }
});
var resultsTitle = ui.Label({
  value: 'Analysis Results',
  style: {fontSize: '12px', fontWeight: 'bold', color: '#000000', margin: '0 0 8px 0'}
});
var resultThreshold = ui.Label({value: '', style: {fontSize: '11px', color: '#000000', margin: '2px 0'}});
var resultPatches   = ui.Label({value: '', style: {fontSize: '11px', color: '#000000', margin: '2px 0'}});
var resultImages    = ui.Label({value: '', style: {fontSize: '11px', color: '#000000', margin: '2px 0'}});
var resultCloud     = ui.Label({value: '', style: {fontSize: '11px', color: '#000000', margin: '2px 0'}});
var sarAlert        = ui.Label({
  value: '⚠  SAR FALLBACK RECOMMENDED — cloud cover exceeds threshold',
  style: {fontSize: '11px', color: '#000000', fontWeight: 'bold', margin: '6px 0 0 0', shown: false}
});
resultsPanel.add(resultsTitle);
resultsPanel.add(resultThreshold);
resultsPanel.add(resultPatches);
resultsPanel.add(resultImages);
resultsPanel.add(resultCloud);
resultsPanel.add(sarAlert);
panel.add(resultsPanel);

// --- Legend ---
panel.add(makeSectionLabel('FDI Legend'));
panel.add(makeLegend([
  {label: 'High plastic signal', color: '#FF0000'},
  {label: 'Moderate signal',     color: '#FFA500'},
  {label: 'Background / water',  color: '#0000FF'},
  {label: 'Hotspot patch',       color: '#000000'}
]));

// --- Footer ---
panel.add(ui.Label({
  value: 'UNDP SDG AI Lab · DPWMS Guinea',
  style: {
    fontSize: '10px',
    color: '#000000',
    textAlign: 'center',
    margin: '10px 0 8px 0'
  }
}));

ui.root.insert(0, panel);

// -----------------------------------------------------------------------
// 3.3 — Helper functions for UI
// -----------------------------------------------------------------------

function makeSectionLabel(text) {
  return ui.Label({
    value: text.toUpperCase(),
    style: {
      fontSize: '10px',
      color: '#000000',
      fontWeight: 'bold',
      margin: '12px 16px 4px 16px'
    }
  });
}

function makeSlider(label, min, max, defaultVal, step) {
  var sliderPanel = ui.Panel({
    layout: ui.Panel.Layout.flow('horizontal'),
    style: {margin: '0 16px 6px 16px', padding: '0px'}
  });
  var lbl = ui.Label({
    value: label,
    style: {fontSize: '11px', color: '#000000', width: '140px', margin: '6px 0 0 0'}
  });
  var slider = ui.Slider({
    min: min, max: max, value: defaultVal, step: step,
    style: {width: '90px', margin: '0px', color: UI.ACCENT}
  });
  var valLbl = ui.Label({
    value: String(defaultVal),
    style: {fontSize: '11px', color: '#000000', margin: '6px 0 0 4px', width: '36px'}
  });
  slider.onChange(function(v) { valLbl.setValue(String(v)); });
  sliderPanel.add(lbl);
  sliderPanel.add(slider);
  sliderPanel.add(valLbl);
  return {panel: sliderPanel, slider: slider};
}

function makeCheckbox(label, checked) {
  var row = ui.Panel({
    layout: ui.Panel.Layout.flow('horizontal'),
    style: {margin: '2px 16px', padding: '0px'}
  });
  var cb = ui.Checkbox({
    label: label,
    value: checked,
    style: {fontSize: '11px', color: 'black', margin: '0px'}
  });
  row.add(cb);
  return {panel: row, checkbox: cb};
}

function makeTag(text, color) {
  return ui.Label({
    value: text,
    style: {
      fontSize: '9px',
      color: '#000000',
      border: '1px solid ' + color,
      padding: '1px 5px',
      margin: '0 4px 0 0',
      borderRadius: '3px'
    }
  });
}

function makeLegend(items) {
  var legendPanel = ui.Panel({
    layout: ui.Panel.Layout.flow('vertical'),
    style: {margin: '0 16px 10px 16px', padding: '0px'}
  });
  items.forEach(function(item) {
    var row = ui.Panel({
      layout: ui.Panel.Layout.flow('horizontal'),
      style: {margin: '1px 0', padding: '0px'}
    });
    var swatch = ui.Label({
      value: '  ',
      style: {
        backgroundColor: item.color,
        width: '14px',
        height: '12px',
        margin: '3px 6px 0 0',
        padding: '0px'
      }
    });
    var label = ui.Label({
      value: item.label,
      style: {fontSize: '11px', color: '#000000', margin: '0px'}
    });
    row.add(swatch);
    row.add(label);
    legendPanel.add(row);
  });
  return legendPanel;
}

// -----------------------------------------------------------------------
// 3.4 — AOI definitions
// -----------------------------------------------------------------------
function getAOI(key) {
  var aois = {
    full:     ee.Geometry.BBox(-15.5, 8.8, -13.0, 10.8),
    conakry:  ee.Geometry.BBox(-13.9, 9.3, -13.4, 9.7),
    konkouré: ee.Geometry.BBox(-14.8, 9.7, -14.2, 10.3),
    fatala:   ee.Geometry.BBox(-14.2, 10.1, -13.6, 10.6),
    mangrove: ee.Geometry.BBox(-14.5, 9.0, -13.5, 9.5)
  };
  return aois[key] || aois.full;
}

// -----------------------------------------------------------------------
// 3.5 — Run Analysis Handler
// -----------------------------------------------------------------------
function onRunClicked() {
  if (state.isRunning) return;
  state.isRunning = true;
  runButton.setDisabled(true);
  runButton.setLabel('⟳  Running...');

  // Show status
  statusLabel.setValue('⟳  Building Sentinel-2 collection...');
  statusPanel.style().set('shown', true);
  resultsPanel.style().set('shown', false);

  // --- Build date range ---
  var year   = startYearSlider.slider.getValue();
  var month  = startMonthSlider.slider.getValue();
  var dur    = durationSlider.slider.getValue();
  var startDate = ee.Date.fromYMD(year, month, 1);
  var endDate   = startDate.advance(dur, 'month');

  var ndviT  = ndviSlider.slider.getValue();
  var ndwiT  = ndwiSlider.slider.getValue();
  var cloudT = cloudSlider.slider.getValue();
  CONFIG.CLOUD_THRESHOLD = cloudT;

  var aoi = getAOI(aoiSelect.getValue());

  // --- Clear existing layers ---
  Map.layers().reset();

  // --- Show AOI outline ---
  Map.addLayer(aoi, {color: UI.ACCENT, fillColor: '#00000000', width: 1.5}, 'AOI boundary');
  Map.centerObject(aoi, 9);

  // --- Run pipeline ---
  statusLabel.setValue('⟳  Computing FDI + spectral masks...');

  var results = runM2Pipeline(
    startDate.format('YYYY-MM-dd'),
    endDate.format('YYYY-MM-dd'),
    ndviT, ndwiT, aoi
  );

  // --- Add layers conditionally ---
  if (showRGB.checkbox.getValue()) {
    Map.addLayer(results.composite, CONFIG.VIZ.RGB, 'True color (RGB)', true, 0.85);
  }
  if (showFalseClr.checkbox.getValue()) {
    Map.addLayer(results.composite, CONFIG.VIZ.FALSE_CLR, 'False color (NIR/Red/Green)', false, 0.85);
  }
  if (showFDI.checkbox.getValue()) {
    Map.addLayer(results.fdi, CONFIG.VIZ.FDI, 'FDI raster', true, 0.75);
  }
  if (showNDVI.checkbox.getValue()) {
    Map.addLayer(results.ndvi, CONFIG.VIZ.NDVI, 'NDVI', false, 0.6);
  }
  if (showNDWI.checkbox.getValue()) {
    Map.addLayer(results.ndwi, CONFIG.VIZ.NDWI, 'NDWI', false, 0.6);
  }
  if (showMasked.checkbox.getValue()) {
    Map.addLayer(results.maskedFDI, CONFIG.VIZ.FDI, 'Masked FDI (post-NDVI/NDWI)', true, 0.8);
  }
  if (showHotspot.checkbox.getValue()) {
    Map.addLayer(
      results.hotspotBinary.selfMask(),
      {palette: [UI.WARN], opacity: 0.7},
      'Hotspot binary'
    );
  }
  if (showVectors.checkbox.getValue()) {
    Map.addLayer(
      results.hotspotVectors,
      {color: UI.WARN, fillColor: UI.WARN + '55', width: 1.5},
      'Hotspot polygons (vectorized)'
    );
  }

  // --- Evaluate results for display ---
  results.threshold.evaluate(function(threshVal) {
    resultThreshold.setValue(
      '  Otsu threshold : ' + (threshVal !== null ? threshVal.toFixed(5) : 'N/A (fallback)')
    );
  });

  results.hotspotVectors.size().evaluate(function(n) {
    resultPatches.setValue('  Plastic patches  : ' + (n || '0') + ' detected');
  });

  results.collectionSize.evaluate(function(n) {
    resultImages.setValue('  S2 images used  : ' + (n || '0') + ' scenes');
  });

  results.cloudPct.evaluate(function(pct) {
    var pctStr = pct !== null ? pct.toFixed(1) + '%' : 'N/A';
    resultCloud.setValue('  Mean cloud cover: ' + pctStr);

    // SAR alert (FR-0207)
    if (pct !== null && pct > cloudSlider.slider.getValue()) {
      sarAlert.style().set('shown', true);
      sarAlert.setStyle({color: UI.WARN, fontWeight: 'bold', shown: true});
    } else {
      sarAlert.style().set('shown', false);
    }

    // Finalize UI
    resultsPanel.style().set('shown', true);
    statusLabel.setValue('✓  Analysis complete.');
    statusLabel.style({color: UI.ACCENT});
    runButton.setDisabled(false);
    runButton.setLabel('▶  Run M2 Analysis');
    state.isRunning = false;
  });

  state.results = results;
}


// =============================================================================
// 4. MAP CLICK INSPECTOR
// =============================================================================
// Clicking on the map shows per-pixel spectral values for the current composite
Map.onClick(function(coords) {
  if (!state.results) return;

  var point = ee.Geometry.Point([coords.lon, coords.lat]);
  var composite = state.results.composite;

  composite.select(['FDI','NDVI','NDWI','B4','B8','B11'])
    .reduceRegion({
      reducer: ee.Reducer.first(),
      geometry: point,
      scale: 20
    })
    .evaluate(function(vals) {
      if (!vals) return;
      var lines = [
        '— Click Inspector —',
        'Lat: '  + coords.lat.toFixed(5),
        'Lon: '  + coords.lon.toFixed(5),
        'FDI:  ' + (vals.FDI  !== null ? vals.FDI.toFixed(5)  : 'masked'),
        'NDVI: ' + (vals.NDVI !== null ? vals.NDVI.toFixed(3) : 'masked'),
        'NDWI: ' + (vals.NDWI !== null ? vals.NDWI.toFixed(3) : 'masked'),
        'B4:   ' + (vals.B4  !== null ? vals.B4.toFixed(4)   : 'masked'),
        'B8:   ' + (vals.B8  !== null ? vals.B8.toFixed(4)   : 'masked'),
        'B11:  ' + (vals.B11 !== null ? vals.B11.toFixed(4)  : 'masked')
      ];
      statusLabel.setValue(lines.join('\n'));
      statusLabel.style({color: UI.ACCENT_LITE});
      statusPanel.style().set('shown', true);
    });
});


// =============================================================================
// 5. INITIAL MAP SETUP — show AOI on load
// =============================================================================
(function initMap() {
  Map.addLayer(
    ee.Geometry(getAOI('full')),
    {color: UI.ACCENT, fillColor: '00000000', width: 1.5},
    'AOI boundary — Guinea coastal zone'
  );

  // Informational label
  Map.add(ui.Label({
    value: 'Select parameters and click ▶ Run M2 Analysis',
    style: {
      position: 'bottom-center',
      fontSize: '12px',
      color: 'black',
      backgroundColor: 'rgba(255,255,255,0.85)',
      padding: '6px 14px',
      border: '1px solid #ccc',
      borderRadius: '4px'
    }
  }));
})();

// =============================================================================
// END OF SCRIPT
// ====================== copy to code.earthengine.google.com =================