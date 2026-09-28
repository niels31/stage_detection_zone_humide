/*
Copyright (©) CEREMA/DTerOCC/DT/OSECC All rights reserved
auteur : Niels Saura
date de création : Septembre 2026

Script pour l'API javascript de Google Earth Engine.
Ce script sert à exporter le terme temporel du filtre de Quegan de toute l'année 2019 pour les GRD S1.
Le terme mensuel du filtre de Quegan est la moyenne du rapport entre l'intensité d'un pixel et la moyenne de l'intensité de son voisinnage.
Ce script permet de faire tous les calculs sur serveur et de ne télécharger qu'une seule image par orbite et polarisation.
*/

// géometrie de la zone d'étude
roi = roi.geometry();

// Chargement GRD S1 pour toute l'année 2019 VV et VH
var s1 = ee.ImageCollection('COPERNICUS/S1_GRD')
  .filterDate('2019-01-01', '2019-12-31')
  .filterBounds(roi)
  .filter(ee.Filter.listContains('transmitterReceiverPolarisation', 'VV'))
  .filter(ee.Filter.listContains('transmitterReceiverPolarisation', 'VH'))
  .select(['VV', 'VH']);

print('Nombre total d\'images S1 GRD (VV+VH) en 2019 sur le ROI :', s1.size());

// Séparation par direction d'orbite
var s1Asc = s1.filter(ee.Filter.eq('orbitProperties_pass', 'ASCENDING'));
var s1Desc = s1.filter(ee.Filter.eq('orbitProperties_pass', 'DESCENDING'));

print('Nombre d\'images ASCENDING :', s1Asc.size());
print('Nombre d\'images DESCENDING :', s1Desc.size());


// Fonction calculant le ratio entre intensité d'un pixel et intensité moyenne de son voisinnage
// le voisinnage correspond au kernel, ici radius : 2 --> kernel=5x5
var kernel = ee.Kernel.square({radius: 2, units: 'pixels', normalize: true});

var processImage = function(img) {
  var bands = ['VV', 'VH'];
  var imgSAR = img.select(bands);

  // Reconversion dB -> intensité linéaire : I = 10^(dB/10)
  var intensity = ee.Image(10).pow(imgSAR.divide(10)).rename(bands);

  // 2) Convolution moyenne 5x5
  var meanNeighborhood = intensity.convolve(kernel);

  // 3) Rapport pixel / moyenne du voisinage
  var ratio = intensity.divide(meanNeighborhood)
    .rename(bands.map(function(b) { return ee.String(b).cat('_ratio'); }));

  return ratio.copyProperties(img, img.propertyNames());
};


// Application + moyenne temporelle, séparément pour ASC et DESC
var ratioAsc  = s1Asc.map(processImage);
var ratioDesc = s1Desc.map(processImage);

var annualMeanRatioAsc  = ratioAsc.mean().clip(roi);
var annualMeanRatioDesc = ratioDesc.mean().clip(roi);

print('Moyenne annuelle du ratio - ASCENDING :', annualMeanRatioAsc);
print('Moyenne annuelle du ratio - DESCENDING :', annualMeanRatioDesc);


// Visualisation
Map.centerObject(roi, 10);
Map.addLayer(roi, {color: 'yellow'}, 'ROI', true, 0.3);

var visParams = {min: 0.5, max: 1.5, palette: ['blue', 'white', 'red']};

Map.addLayer(annualMeanRatioAsc.select('VV_ratio'),  visParams, 'ASC - VV ratio');
Map.addLayer(annualMeanRatioAsc.select('VH_ratio'),  visParams, 'ASC - VH ratio');
Map.addLayer(annualMeanRatioDesc.select('VV_ratio'), visParams, 'DESC - VV ratio');
Map.addLayer(annualMeanRatioDesc.select('VH_ratio'), visParams, 'DESC - VH ratio');

// Export vers Drive

Export.image.toDrive({
  image: annualMeanRatioAsc,
  description: 'S1_2019_ratio_ASCENDING',
  region: roi,
  scale: 10,
  maxPixels: 1e13
});

Export.image.toDrive({
  image: annualMeanRatioDesc,
  description: 'S1_2019_ratio_DESCENDING',
  region: roi,
  scale: 10,
  maxPixels: 1e13
});