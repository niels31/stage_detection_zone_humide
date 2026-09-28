/*
Copyright (©) CEREMA/DTerOCC/DT/OSECC All rights reserved
auteur : Niels Saura
date de création : Septembre 2026

Script pour l'API javascript de Google Earth Engine.
Ce script sert à exporter la moyenne par mois, polarisation et direction d'orbite
des images Sentinel-1 GRD de l'année 2019. Permet de n'exporter que 12 images délimitées au roi par
mois, polarisation orbite.
*/


// géometrie de la zone d'étude
var roi = roi.geometry();

// Chargement S1 GRD, VV+VH, 2019, sur le ROI
var s1 = ee.ImageCollection('COPERNICUS/S1_GRD')
  .filterDate('2019-01-01', '2019-12-31')
  .filterBounds(roi)
  .filter(ee.Filter.listContains('transmitterReceiverPolarisation', 'VV'))
  .filter(ee.Filter.listContains('transmitterReceiverPolarisation', 'VH'))
  .select(['VV', 'VH']);

print('Nombre total d\'images S1 GRD (VV+VH) en 2019 sur le ROI :', s1.size());

// Conversion dB -> intensité linéaire
var toIntensity = function(img) {
  var intensity = ee.Image(10).pow(img.select(['VV', 'VH']).divide(10))
                    .rename(['VV', 'VH']);
  return intensity.copyProperties(img, img.propertyNames());
};

var s1Intensity = s1.map(toIntensity);

// Séparation par orbite ET par polarisation -> 4 groupes
var s1Asc  = s1Intensity.filter(ee.Filter.eq('orbitProperties_pass', 'ASCENDING'));
var s1Desc = s1Intensity.filter(ee.Filter.eq('orbitProperties_pass', 'DESCENDING'));

var groups = {
  'A_VV': s1Asc.select(['VV']),
  'A_VH': s1Asc.select(['VH']),
  'D_VV': s1Desc.select(['VV']),
  'D_VH': s1Desc.select(['VH'])
};

// Moyenne mensuelle pour chaque groupe -> image multi-bandes (12 bandes)
var months = ee.List.sequence(1, 12);

// Valeur "no-data" à utiliser si un mois n'a aucune image
// (évite un échec d'export ; à adapter selon vos besoins)
var buildMonthlyStack = function(collection, bandName) {
  var monthlyImages = months.map(function(m) {
    m = ee.Number(m);
    var start = ee.Date.fromYMD(2019, 1, 1).advance(m.subtract(1), 'month');
    var end = start.advance(1, 'month');

    var monthColl = collection.filterDate(start, end);
    var monthCount = monthColl.size();

    var meanImg = ee.Image(ee.Algorithms.If(
      monthCount.gt(0),
      monthColl.mean(),
      ee.Image.constant(0).updateMask(ee.Image(0))  // image entièrement masquée si pas d'image ce mois-ci
    )).rename(ee.String(bandName).cat('_m')
                .cat(ee.Number(m).format('%02d')));

    return meanImg.set('month', m).set('image_count', monthCount);
  });

  return ee.ImageCollection.fromImages(monthlyImages);
};

var stacks = {};
var stackedImages = {};

Object.keys(groups).forEach(function(key) {
  var bandName = key; // ex: 'A_VV'
  var monthlyColl = buildMonthlyStack(groups[key], bandName);
  stacks[key] = monthlyColl;

  // Empile les 12 images mensuelles en une seule image à 12 bandes
  var stackedImg = monthlyColl.toBands().clip(roi);
  // Renomme proprement les bandes (toBands() préfixe avec l'index système)
  var newNames = ee.List.sequence(1, 12).map(function(m) {
    return ee.String(bandName).cat('_m').cat(ee.Number(m).format('%02d'));
  });
  stackedImg = stackedImg.rename(newNames);

  stackedImages[key] = stackedImg;

  print('Stack ' + key + ' :', stackedImg);
});

// Visualisation rapide (bande du mois 1 de chaque groupe)
Map.centerObject(roi, 10);
Map.addLayer(roi, {color: 'yellow'}, 'ROI', true, 0.3);

var visParams = {min: 0, max: 1, palette: ['black', 'white']};
Map.addLayer(stackedImages['A_VV'].select('A_VV_m01'), visParams, 'A_VV - Janvier');

// Export des 4 images (12 bandes chacune) vers Drive
Object.keys(stackedImages).forEach(function(key) {
  Export.image.toDrive({
    image: stackedImages[key],
    description: 'S1_2019_monthly_' + key,
    folder: 'GEE_exports',
    fileNamePrefix: 'S1_2019_monthly_' + key,
    region: roi,
    scale: 10,
    maxPixels: 1e13,
    fileFormat: 'GeoTIFF'
  });
});