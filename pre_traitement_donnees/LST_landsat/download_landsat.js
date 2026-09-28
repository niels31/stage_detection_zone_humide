// Copyright (©) CEREMA/DTerOCC/DT/OSECC All rights reserved
// auteur : Niels Saura
// date de création : Septembre 2026

// Script pour l'API javascript de Google Earth Engine.

// Export de toutes les acquisitions valides sur la zone
// Années couvertes : 2019, 2021, 2022 (configurable, non contiguës OK)
//
// Aucun groupement par heure ni découpage spatial : toutes les
// acquisitions valides sont triées chronologiquement et empilées
// en bandes, par lots de CHUNK_SIZE pour limiter la taille des fichiers.
//
// Chaque bande est nommée avec : satellite, date, heure d'acquisition
// précise (UTC), et numéro d'orbite (WRS_PATH/WRS_ROW).
//
// Masque nuages : QA_PIXEL (CFMask, Collection 2)
// ============================================================


//géometrie de la zone d'étude
var ROI = ROI.geometry();

// Années à traiter
var ANNEES = [2019, 2020, 2021, 2022];

// Nombre max d'acquisitions (bandes) par fichier GeoTIFF exporté
var CHUNK_SIZE = 400;

// Résolution d'export en mètres (30 = résolution native Landsat)
var RESOLUTION = 30;

// Dossier Google Drive de destination
var DRIVE_FOLDER = 'LST_Landsat';


// application masque nuage

function masqueNuages(image) {
  var qa = image.select('QA_PIXEL');
  // Bit 1 : cloud dilaté ; Bit 3 : cloud ; Bit 4 : cloud shadow
  var masque = qa.bitwiseAnd(1 << 1).eq(0)
    .and(qa.bitwiseAnd(1 << 3).eq(0))
    .and(qa.bitwiseAnd(1 << 4).eq(0));
  return image.updateMask(masque);
}


// Calcul de la LST en degrés Celcius

// Même formule pour tous les satellites Collection 2 :
//   LST_K = DN * 0.00341802 + 149.0
// L7 : bande ST_B6  |  L8 & L9 : bande ST_B10
function calcLST(image) {
  var spacecraft = image.get('SPACECRAFT_ID');
  var lst = ee.Algorithms.If(
    ee.String(spacecraft).equals('LANDSAT_7'),
    image.select('ST_B6').multiply(0.00341802).add(149.0).subtract(273.15),
    image.select('ST_B10').multiply(0.00341802).add(149.0).subtract(273.15)
  );
  return image
    .addBands(ee.Image(lst).rename('LST_C'))
    .select('LST_C')
    .copyProperties(image, image.propertyNames());
}


// ── 4. CONSTRUCTION DU NOM DE BANDE (métadonnées précises) ───

/**
 * Construit un identifiant de bande contenant :
 *   - le satellite (L7, L8, L9)
 *   - la date d'acquisition (YYYY-MM-dd)
 *   - l'heure UTC précise d'acquisition (HHhMMmSS, issue du timestamp réel)
 *   - le numéro d'orbite WRS (PATH-ROW, identifiant officiel Landsat
 *     de la trace au sol — équivalent au "numéro d'orbite")
 *
 * Exemple de nom généré : LST_L8_2021-06-12_10h14m32_P196R029
 */
function ajouterIdentifiant(image) {
  var ts = ee.Date(image.get('system:time_start'));
  var dateStr = ts.format('YYYY-MM-dd');
  var heureStr = ts.format('HH').cat('h')
    .cat(ts.format('mm')).cat('m')
    .cat(ts.format('ss'));

  var sat = ee.String(image.get('SPACECRAFT_ID')).slice(8, 9); // '7', '8' ou '9'

  var path = ee.Number(image.get('WRS_PATH')).format('%03d');
  var row  = ee.Number(image.get('WRS_ROW')).format('%03d');
  var orbite = ee.String('P').cat(path).cat('R').cat(row);

  var nomBande = ee.String('LST_L').cat(sat).cat('_')
    .cat(dateStr).cat('_')
    .cat(heureStr).cat('_')
    .cat(orbite);

  return image.set('nom_bande', nomBande);
}


/**
 * Charge et fusionne toutes les années de ANNEES en UNE SEULE collection
 * triée chronologiquement, en appliquant pour chaque année le bon jeu
 * de satellites
 */
function chargerCollectionMultiAnnees(annees) {
  var collectionsAnnuelles = annees.map(function(annee) {
    var debut = annee + '-01-01';
    var fin   = annee + '-12-31';

    function charger(collection_id) {
      return ee.ImageCollection(collection_id)
        .filterBounds(ROI)
        .filterDate(debut, fin)
        .map(masqueNuages)
        .map(calcLST);
    }

    var useL7 = (annee === 2019 || annee === 2021);
    var useL9 = (annee === 2021 || annee === 2022);

    var merged = charger('LANDSAT/LC08/C02/T1_L2'); // L8 toujours inclus
    if (useL7) merged = merged.merge(charger('LANDSAT/LE07/C02/T1_L2'));
    if (useL9) merged = merged.merge(charger('LANDSAT/LC09/C02/T1_L2'));

    return merged;
  });

  var collectionGlobale = collectionsAnnuelles.reduce(function(acc, col) {
    return acc.merge(col);
  });

  return collectionGlobale.sort('system:time_start').map(ajouterIdentifiant);
}

var LST_COLLECTION = chargerCollectionMultiAnnees(ANNEES);
print('Acquisitions totales (toutes années confondues) :', LST_COLLECTION.size());


/**
 * Toutes les acquisitions valides, triées chronologiquement, sont
 * découpées en lots de CHUNK_SIZE et empilées en bandes multiples.
 * Chaque bande porte son identifiant complet (satellite, date, heure,
 * orbite). Un même fichier peut contenir des acquisitions de plusieurs
 * années et de plusieurs orbites/heures, sans distinction.
 */
var n = LST_COLLECTION.size().getInfo();
print('Nombre total d\'acquisitions à exporter :', n);

var imgList = LST_COLLECTION.toList(n);

// Renommage de chaque image avec son identifiant complet
var imgRenommees = imgList.map(function(img) {
  img = ee.Image(img);
  var nom = ee.String(img.get('nom_bande'));
  return img.select(['LST_C'], [nom]);
});

var nChunks = Math.ceil(n / CHUNK_SIZE);
var totalExports = 0;

for (var c = 0; c < nChunks; c++) {
  var debut = c * CHUNK_SIZE;
  var fin   = Math.min(debut + CHUNK_SIZE, n);

  var chunk = ee.List.sequence(debut, fin - 1).map(function(i) {
    return imgRenommees.get(i);
  });

  var stack = ee.ImageCollection(chunk).toBands().clip(ROI);

  // Dates première/dernière du chunk pour le nom de fichier
  var premiereDate = ee.Image(imgList.get(debut)).date().format('YYYY-MM-dd').getInfo();
  var derniereDate = ee.Image(imgList.get(fin - 1)).date().format('YYYY-MM-dd').getInfo();

  var nomFichier = 'LST_Landsat_' + premiereDate + '_a_' + derniereDate +
                   '_chunk' + (c + 1) + 'of' + nChunks;

  Export.image.toDrive({
    image          : stack,
    description    : nomFichier,
    folder         : DRIVE_FOLDER,
    fileNamePrefix : nomFichier,
    region         : ROI,
    scale          : RESOLUTION,
    crs            : 'EPSG:4326',
    maxPixels      : 1e13,
    fileFormat     : 'GeoTIFF'
  });

  totalExports++;
  print('Export : chunk ' + (c + 1) + '/' + nChunks +
        ' [' + premiereDate + ' → ' + derniereDate + ']' +
        ' (' + (fin - debut) + ' bandes)');
}

print('════════════════════════════════════');
print('Total exports planifiés :', totalExports);
print('→ Onglet Tasks pour démarrer les exports.');


// affichage

var lstViz = {min: 0, max: 45, palette: [
  '040274','0502a3','0602ff','235cb1','307ef3',
  '30c8e2','3be285','3ae237','b5e22e','d6e21f',
  'fff705','ffd611','ffb613','ff8b13','ff6e08',
  'ff500d','ff0000','de0101','c21301'
]};

Map.centerObject(ROI, 8);

ANNEES.forEach(function(annee, idx) {
  var debut = annee + '-01-01';
  var fin   = annee + '-12-31';
  var col = LST_COLLECTION.filterDate(debut, fin).select('LST_C');
  var median = col.median().clip(ROI);
  Map.addLayer(median, lstViz, 'LST médiane ' + annee + ' (°C)', idx === 0);
});

// Légende
var legend = ui.Panel({style: {position: 'bottom-left', padding: '8px'}});
legend.add(ui.Label('LST (°C)', {fontWeight: 'bold'}));
legend.add(ui.Thumbnail({
  image: ee.Image.pixelLonLat().select(0),
  params: {bbox: [0,0,1,0.1], dimensions: '200x20',
           palette: lstViz.palette, min: lstViz.min, max: lstViz.max},
  style: {stretch: 'horizontal', margin: '0 8px'}
}));
legend.add(ui.Panel([
  ui.Label('0 °C',  {margin: '0 auto 0 0'}),
  ui.Label('45 °C', {margin: '0 0 0 auto'})
], ui.Panel.Layout.flow('horizontal'), {stretch: 'horizontal'}));
Map.add(legend);