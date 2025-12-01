// --- Configuration ---
// Define the two regions of interest using standard bounding boxes
var REGIONS = {
  CONUS: ee.Geometry.Rectangle([-125, 25, -66, 50]),
  HAWAII: ee.Geometry.Rectangle([-160, 18, -154, 23]),
};

// Start and end years for annual averaging (e.g., 2010-2020)
var START_YEAR = 2010;
var END_YEAR = 2020;

// FIX: Define a fixed, coarser scale (e.g., 10km) for statistical sampling.
var SAMPLING_SCALE = 10000; // 10,000 meters = 10 km

// --- EXPORT FUNCTION (New) ---
var exportDistributionData = function(image, regionName, scale, description) {
  var sampledPoints = image.sample({
    region: REGIONS[regionName],
    scale: scale,
    // Use the count from the printStatistics output (e.g., 145243 for CONUS Temp) 
    // or a large number to ensure all pixels are sampled at the defined scale.
    numPixels: 1e8, 
    geometries: false // Export only the value, not the geometry (faster/smaller file)
  });

  Export.table.toDrive({
    collection: sampledPoints,
    description: description + '_' + regionName + '_Export',
    fileNamePrefix: description + '_' + regionName + '_data',
    folder: 'GEE_Export_Climate_Distributions',
    fileFormat: 'CSV'
  });
  
  print('✅ Export task started for:', description + ' (' + regionName + ')');
  print('   Check the "Tasks" tab in GEE to run the export.');
};


// --- PLOTTING FUNCTION ---
var plotDistribution = function(image, regionName, scale, title) {
  var bandName = image.bandNames().get(0);
  
  // Determine plot range dynamically based on the image statistics (optional, but cleaner)
  var rangeStats = image.reduceRegion({
    reducer: ee.Reducer.min().combine({reducer2: ee.Reducer.max(), sharedInputs: true}),
    geometry: REGIONS[regionName],
    scale: scale,
    maxPixels: 1e13
  });
  
  var minVal = rangeStats.get(ee.String(bandName).cat('_min'));
  var maxVal = rangeStats.get(ee.String(bandName).cat('_max'));
  
  var chart = ui.Chart.image.histogram({
    image: image, 
    region: REGIONS[regionName], 
    scale: scale, 
    maxBuckets: 100 // Maximum number of histogram bins
  })
  .setOptions({
    title: regionName + ' - ' + title + ' Distribution',
    legend: {position: 'none'},
    hAxis: {title: 'Value (' + image.bandNames().get(0).getInfo() + ')', minValue: minVal, maxValue: maxVal},
    vAxis: {title: 'Pixel Count'},
    // Force standard colors for consistency
    series: [{ color: regionName === 'CONUS' ? '#1f77b4' : '#ff7f0e' }]
  });
  print(chart);
};

// Function to calculate and print descriptive statistics
var printStatistics = function(image, regionName, statName) {
  // Get the band name of the image (e.g., 'annual_prcp' or 'annual_temp_C')
  var bandName = image.bandNames().get(0);

  // Use the fixed SAMPLING_SCALE for reduceRegion
  var stats = image.reduceRegion({
    // Combining reducers. GEE automatically prefixes output keys with the band name 
    // when using combine on a single-band image. We rely on this behavior now.
    reducer: ee.Reducer.count()
             .combine({reducer2: ee.Reducer.mean(), sharedInputs: true})
             .combine({reducer2: ee.Reducer.stdDev(), sharedInputs: true})
             .combine({reducer2: ee.Reducer.min(), sharedInputs: true})
             .combine({reducer2: ee.Reducer.max(), sharedInputs: true})
             // Percentile outputs will be: bandName_p25, bandName_p50, bandName_p75
             .combine({reducer2: ee.Reducer.percentile([25, 50, 75]), sharedInputs: true}),
    geometry: REGIONS[regionName],
    scale: SAMPLING_SCALE, // Use the fixed sampling scale
    maxPixels: 1e13
  });

  // Helper function to safely retrieve the prefixed key (e.g., 'annual_prcp_mean')
  var getStat = function(key) {
    var prefixedKey = ee.String(bandName).cat('_').cat(key);
    return stats.get(prefixedKey);
  };
  
  print('--- ' + statName + ' Spatial Distribution for ' + regionName + ' ---');
  print('Sampling Scale:', SAMPLING_SCALE + 'm');
  print('Count (Sampled Pixels):', getStat('count'));
  print('Min:', getStat('min'));
  print('Max:', getStat('max'));
  print('Mean:', getStat('mean'));
  print('Median (p50):', getStat('p50'));
  print('StdDev:', getStat('stdDev'));
  print('Q1 (p25):', getStat('p25'));
  print('Q3 (p75):', getStat('p75'));
};

// --- 1. Annual Precipitation (CHIRPS) ---
// CHIRPS is daily precipitation, summed monthly to get annual totals.
// Units: mm/day -> mm/year
var chirps = ee.ImageCollection("UCSB-CHG/CHIRPS/DAILY")
    .filterDate(START_YEAR + '-01-01', END_YEAR + '-12-31')
    .select('precipitation');

// Calculate annual precipitation for each year in the period
var annualPrecip = ee.ImageCollection(
  ee.List.sequence(START_YEAR, END_YEAR).map(function(year) {
    var start = ee.Date.fromYMD(year, 1, 1);
    var end = start.advance(1, 'year');
    
    // Sum precipitation across the year
    var sum = chirps.filterDate(start, end).sum();
    
    // Add a year property for metadata, and return
    return sum.set('year', year).rename('annual_prcp');
  })
);

// Calculate the long-term (multi-year) mean annual precipitation
var meanAnnualPrecip = annualPrecip.mean();

// Define visualization parameters for precipitation (blue palette)
var prcpVis = {min: 0, max: 2500, palette: ['white', 'cyan', 'blue', 'darkblue']};

// --- Analyze and Map Precipitation ---
Map.addLayer(meanAnnualPrecip.clip(REGIONS.CONUS), prcpVis, 'CONUS Mean Annual PRCP');
Map.addLayer(meanAnnualPrecip.clip(REGIONS.HAWAII), prcpVis, 'Hawaii Mean Annual PRCP');
Map.centerObject(REGIONS.HAWAII, 6); // Start centered on Hawaii

print('Precipitation Nominal Scale (CHIRPS):', meanAnnualPrecip.projection().nominalScale());

// --- Print Stats & Plot ---
printStatistics(meanAnnualPrecip, 'CONUS', 'Annual Precipitation (mm/yr)');
plotDistribution(meanAnnualPrecip, 'CONUS', SAMPLING_SCALE, 'Annual Precipitation');
exportDistributionData(meanAnnualPrecip, 'CONUS', SAMPLING_SCALE, 'Annual_Precipitation'); // <-- Export
printStatistics(meanAnnualPrecip, 'HAWAII', 'Annual Precipitation (mm/yr)');
plotDistribution(meanAnnualPrecip, 'HAWAII', SAMPLING_SCALE, 'Annual Precipitation');
exportDistributionData(meanAnnualPrecip, 'HAWAII', SAMPLING_SCALE, 'Annual_Precipitation'); // <-- Export


// --- 2. Annual Average Temperature (ERA5-Land) ---
// ERA5-Land provides 2m air temperature (k) monthly.
// Units: Kelvin (K). Convert to Celsius (C) for easier interpretation.
var era5 = ee.ImageCollection('ECMWF/ERA5_LAND/MONTHLY')
    .filterDate(START_YEAR + '-01-01', END_YEAR + '-12-31')
    .select('temperature_2m'); // Band is Kelvin

// Calculate the long-term (multi-year) mean temperature
var meanTempK = era5.mean();
var meanTempC = meanTempK.subtract(273.15).rename('annual_temp_C');

// Define visualization parameters for temperature (red palette)
var tempVis = {min: 0, max: 30, palette: ['lightblue', 'yellow', 'red', 'darkred']};

// --- Analyze and Map Temperature ---
Map.addLayer(meanTempC.clip(REGIONS.CONUS), tempVis, 'CONUS Mean Annual TEMP (C)');
Map.addLayer(meanTempC.clip(REGIONS.HAWAII), tempVis, 'Hawaii Mean Annual TEMP (C)');

print('Temperature Nominal Scale (ERA5-Land):', meanTempC.projection().nominalScale());

// --- Print Stats & Plot ---
printStatistics(meanTempC, 'CONUS', 'Annual Average Temperature (C)');
plotDistribution(meanTempC, 'CONUS', SAMPLING_SCALE, 'Annual Average Temperature');
exportDistributionData(meanTempC, 'CONUS', SAMPLING_SCALE, 'Annual_Temperature'); // <-- Export
printStatistics(meanTempC, 'HAWAII', 'Annual Average Temperature (C)');
plotDistribution(meanTempC, 'HAWAII', SAMPLING_SCALE, 'Annual Average Temperature');
exportDistributionData(meanTempC, 'HAWAII', SAMPLING_SCALE, 'Annual_Temperature'); // <-- Export


// --- 3. (Placeholder for SWDOWN_ANNUAL) ---
print('\n--- Note on Shortwave Radiation ---');
print('SWDOWN_ANNUAL analysis requires loading a high-frequency product and aggregating (similar to PRCP).');
print('The analysis above focuses on the two most hydrologically significant drivers: Precipitation and Temperature.');