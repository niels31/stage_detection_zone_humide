# %%
"""
Copyright (©) CEREMA/DTerOCC/DT/OSECC All rights reserved
auteur : Niels Saura
date de création : Septembre 2026

Script pour constituer les jeux de données de référence nécessaires à l'entraînement du modèle implémenté dans le script 2_training.py.
entrée : chemin du fichier de config (cf 1_create_datasets.yaml)
sorties :
- zh_indivs.pkl : dataframe pandas des individus classés zones humides
- znh_indivs.pkl : dataframe pandas des individus classés non zones humides
"""

# %%
#Imports
import sys
import os
from pathlib import Path

project_path = Path(__file__).resolve().parent.parent / "lib"
sys.path.append(str(project_path))

import shreklib
import numpy as np
import geopandas as gpd
import psycopg2
from shapely import box
import rasterio
import pandas as pd
from os.path import join
from functools import partial
from rasterio import windows
from rasterio.transform import array_bounds
from rasterio.warp import transform_bounds
from rasterio.windows import from_bounds
from rasterio.windows import intersection
from rasterio.windows import Window

# %%
import argparse

parser = argparse.ArgumentParser()

parser.add_argument("yaml_config", help="Chemin du fichier de config.")

args = parser.parse_args()

yaml_config=args.yaml_config

# %%
#Fonctions

#fonction qui retourne la liste des polygones des cellules d'une grille de référence passée avec transform, qui sont inclus dans un polygone passé avec zh (pour zone humide)
#Note : Pour beaucoup optimiser le calcul des géometries, on pourrait ne renvoyer que les coordonnées matricielles et calculer les coordonnées géographiques sur les 3 millions de coordonnées à la fois
#au lieu de calculer zone humide par zone humide.
#zh: liste de polygones shapely
#transform: matrice 3x3 de géoréférencement de raster
def subgrid(zh: list, transform: np.ndarray):
    xmin, ymin, xmax, ymax=zh.bounds
    #on calcule les coordonnées de la zone humide dans la grille de référence
    c_pmin, l_pmin, _=np.linalg.inv(transform)@np.array([xmin,ymin,1]) #attention, la transformation est colonne, ligne et non ligne colonne, car les points géographiques sont en lon, lat alors que la lon correspond aux colonnes et non aux lignes des matrices.
    c_pmax, l_pmax, _=np.linalg.inv(transform)@np.array([xmax,ymax,1])

    lmin, lmax=sorted([l_pmin, l_pmax])
    cmin, cmax=sorted([c_pmin, c_pmax])

    #on prend les cellules inclues dans la bbox de la zone humide, car on souhaite de toute façon ne garder que celles totalement dans la zone humide
    lmin, cmin=np.ceil([lmin, cmin]).astype(int)
    lmax, cmax=np.floor([lmax, cmax]).astype(int)
    ll, cc=np.meshgrid(np.arange(lmin, lmax+1), np.arange(cmin, cmax+1))
    
    coords_mat=np.stack([cc.ravel(), ll.ravel(), np.ones(ll.size)]) #attention la transformation est bien en colonnes, lignes et pas en lignes, colonnes
    coords_geo=transform@coords_mat
    next_coords_geo=transform@(coords_mat+np.array([1,1,0])[:, None])
    x, y=coords_geo[:2]
    x_next, y_next=next_coords_geo[:2]

    cells_boxes=box(np.minimum(x,x_next), np.minimum(y, y_next), np.maximum(x, x_next), np.maximum(y,y_next)) #ceci est une manière pour créer une box, sans présupposer de l'ordre des points, ce qui dépend de la transformation
    return cells_boxes, coords_mat.T


# %%
# fonction pour ne lire un geotif qu'à une emprise correspondant à un autre raster
# ref_crs: crs du raster dont l'emprise sert de référence pour lire un autre raster
# ref_transform: objet affine.Affine représentant la matrice de transformation du raster de référence
# ref_shape: tuple (nombre_lignes, nombre_colonnes) du raster de référence
#img_fp : chemin du .tif à lire à une emprise réduite
def read_to_window(ref_crs, ref_transform, ref_shape, img_fp):
    ref_height, ref_width=ref_shape

    ref_bounds = array_bounds(
        ref_height,
        ref_width,
        ref_transform,
    )

    with rasterio.open(img_fp) as src:
        transformed_bounds = transform_bounds(
            ref_crs,
            src.crs,
            *ref_bounds,
        )

        window = from_bounds(
            *transformed_bounds,
            transform=src.transform,
        )

        window=window.round_offsets().round_lengths()

        window=intersection(window, Window(0, 0, src.width, src.height))

        transform=windows.transform(window, src.transform)

        return src.read(window=window), transform


# %%
import yaml
with open(yaml_config, "r", encoding="utf-8") as f:
    config = yaml.safe_load(f)

# %%
mensual_imgs_fps=list(config["mensual_imgs_fps"].values())
annual_imgs_fps=list(config["annual_imgs_fps"].values())

# %%
# parsing des input

with rasterio.open(mensual_imgs_fps[0], "r") as src:
    months_count=src.count

rng=np.random.default_rng(config["seed"])

with rasterio.open(annual_imgs_fps[0], "r") as src:
    ref_transform=src.transform
    ref_crs=src.crs
    ref_shape=src.height, src.width

if config["valid_mask_fp"]:
    with rasterio.open(config["valid_mask_fp"], "r") as src:
        valid_mask=src.read(1).astype(bool)
else:
    valid_mask=np.full(ref_shape, True)


# %%
#Tirage des coordonnées zones humides
print("tirage coordonnées zones humides")
# récupération des cellules
zh=gpd.read_file(config["zh_fp"]).to_crs(ref_crs)

subgrid_in_ref=partial(subgrid, transform=np.array(ref_transform).reshape((3,3)))

#on récupère les cellules des toutes les zh
zh_grid_cells=shreklib.batch_map(subgrid_in_ref, zh.geometry.to_list())

zh_grid_cells_boxes=[zh_coords[0] for zh_coords in zh_grid_cells] #coordonnées en ref_crs
zh_grid_cells_mat=[zh_coords[1] for zh_coords in zh_grid_cells] #coordonnées des cellules dans la grille de référence

zh_grid_cells_boxes=[grid_cell for zh in zh_grid_cells_boxes for grid_cell in zh] #décompression
zh_grid_cells_mat=[grid_cell for zh in zh_grid_cells_mat for grid_cell in zh]

grid_cells_gdf=gpd.GeoDataFrame({"geometry":zh_grid_cells_boxes, "mat_coords":zh_grid_cells_mat}, crs=ref_crs)

#on filtre les cellules qui sont contenues dans les zh
inside_zh=grid_cells_gdf.sjoin(zh, predicate="within")


# %%
# suppression des intersections avec le masque bdtopo
conn=psycopg2.connect(
    dbname=config["pgsql"]["db"],
    user=config["pgsql"]["user"],
    password=config["pgsql"]["password"],
    host=config["pgsql"]["host"],
    port=config["pgsql"]["port"]
)

bdtopo_mask=gpd.read_postgis(
    "SELECT * FROM bdtopo_mask",
    conn
)

bdtopo_mask=bdtopo_mask.to_crs(ref_crs)
masked_cells=inside_zh[["geometry"]].sjoin(bdtopo_mask)
valid_zh_cells=inside_zh.loc[~inside_zh.index.isin(masked_cells.index)]

# %%
valid_zh_cells.loc[:,"col"]=np.stack(valid_zh_cells.mat_coords.to_list())[:,0]
valid_zh_cells.loc[:,"line"]=np.stack(valid_zh_cells.mat_coords.to_list())[:,1]

# %%
ref_height, ref_width=ref_shape
valid_zh_cells_ref=valid_zh_cells.loc[(valid_zh_cells.col>=0) & (valid_zh_cells.col<ref_width) & (valid_zh_cells.line>=0) & (valid_zh_cells.line<ref_height)]

# %%
zh_dataset_coords=valid_zh_cells_ref.sample(config["zh_size"], random_state=config["seed"])

# %%
# Tirage des coordonnées non zones humides
print("tirage des coordonnées non zones humides")

# lecture cnzh
read_to_ref_grid=partial(read_to_window, ref_crs=ref_crs, ref_transform=ref_transform, ref_shape=ref_shape)

patrinat, patrinat_transform=read_to_ref_grid(img_fp=config["cnzh_fp"])

# %%
# dans cette méthode, on est obligé de mettre les valeurs du masque à zéro et cela vant la sous-résolution, car on va avoir besoin de toutes les probas et pas seulement le masque et les probas de 0.
# Or une valeur de plus de 100 pour les zones humides entraîne des moyennes avec les pixels adjacents qui augmente leur proba d'être des zones humides, alors que ça devrait la baisser.
# on donne donc dès le début une proba de 0 d'être une zone humide aux pixels du masque, ce qui donnera des moyennes de proba qui reflètent mieux les probas d'être une zone humide
patrinat=np.where(patrinat>100, 0, patrinat)

# %%
with rasterio.open(config["cnzh_fp"], "r") as src:
    patrinat_crs=src.crs

# %%
# reprojection de patrinat sur la grille de référence

# %%
from rasterio.warp import reproject
from affine import Affine
from rasterio.enums import Resampling

# %%
patrinat_ref_grid=np.zeros((ref_height, ref_width), dtype=patrinat.dtype)

reproject(
    source=patrinat,
    destination=patrinat_ref_grid,

    src_crs=patrinat_crs,
    src_transform=patrinat_transform,

    dst_crs=ref_crs,
    dst_transform=ref_transform,

    resampling=Resampling.bilinear
)

# %%
img_zh=np.full(ref_shape, False)
img_zh[valid_zh_cells_ref.line.astype(int), valid_zh_cells_ref.col.astype(int)]=True

# %%
# masquage des zones humides et du valid mask
patrinat_ref_masked=np.where(valid_mask & ~img_zh, patrinat_ref_grid, 100)

# %%
proba_nzh=100-patrinat_ref_masked

# %%
import time
from tqdm import tqdm
# on veut maintenant tirer des cellules de sorte à ce que l'histogramme des probabilités du jeu tiré soit le plus linéaire possible.
# à noter que les faibles probas sont très peu présentes même si elles doivent être peu tirées
# si un intervalle n'a pas un effectif suffisant, on tire dans l'intervalle suivant.
# à noter aussi que pour une proba p allant de 0 à 100, on tire dans l'intervalle ]p, p+1] et non dans l'intervalle [p, p+1[ de sorte à ne pas tirer de probas 0 et à tirer des probas de 100%
# attention ! on donne à un intervalle p, p+1, l'effectif de p+1 et non de p. Cela signifie que l'on approxime ce qui devrait être les pixels de proba 100 par les pixels de 99 exclu à 100
retenue=0
picked=[]
img=proba_nzh
total_sample_size=config["znh_size"]
for p in tqdm(range(100)):
    base_size=100*101/2 # taille d'un échantillon dont l'effectif pour un interval ]p,p+1] est p
    interval_size=np.round((total_sample_size/base_size)*(p+1))+retenue# on déduit la taille de l'échantillon pour atteindre la total_sample_size avec un produit en croix
    interval_img=(img>p) & (img<=p+1)
    to_pick=int(np.min([interval_img.sum(), interval_size])) #on tire soit tout ce qu'il nous faut pour compléter l'échantillon, soit si il n'y a pas assez, tout ce que l'on peut tirer
    picked_idx=rng.choice(interval_img.sum(), size=to_pick, replace=False) #on tire le nombre d'index souhaités
    picked.append(np.stack(np.where(interval_img))[:,picked_idx]) #on récupère les coordonnées matricielles correspondantes)
    retenue=interval_size-to_pick
znh_dataset_coords=np.concat(picked, axis=1)

# %%
# construction des individus

# chargement des images annuelles
print("chargement des images annuelles")
parse_band_name=lambda fp:os.path.basename(fp).split(".")[0]
bands_names=list(map(parse_band_name,annual_imgs_fps))
annual_stack=[]
for fp in annual_imgs_fps:
    with rasterio.open(fp, "r") as src:
        annual_stack.append(src.read(1))

get_band_id=lambda fp:bands_names.index(parse_band_name(fp))
annual_stack.append(annual_stack[get_band_id(config["annual_imgs_fps"]["s1vv_a"])]-annual_stack[get_band_id(config["annual_imgs_fps"]["s1vh_a"])])
annual_stack.append(annual_stack[get_band_id(config["annual_imgs_fps"]["s1vv_d"])]-annual_stack[get_band_id(config["annual_imgs_fps"]["s1vh_d"])])
annual_stack.append(annual_stack[get_band_id(config["annual_imgs_fps"]["mns"])]-annual_stack[get_band_id(config["annual_imgs_fps"]["mnt"])])
annual_stack=np.stack(annual_stack)
bands_names+=["s1vv-s1vh_a_annuel", "s1vv-s1vh_d_annuel", "mnh"]
# construction individus
bands_names+=list(map(parse_band_name, mensual_imgs_fps))

zh_indivs=[]
znh_indivs=[]

print("chargement des images mensuelles")
for month in tqdm(range(months_count)):
    month_stack=[]
    for fp in mensual_imgs_fps:
        with rasterio.open(fp, "r") as src:
            month_stack.append(src.read(month+1))
    month_stack=np.stack(month_stack)

    month_zh_indivs=pd.DataFrame(np.concat([annual_stack[:,zh_dataset_coords.line.astype(int), zh_dataset_coords.col.astype(int)],
                month_stack[:, zh_dataset_coords.line.astype(int), zh_dataset_coords.col.astype(int)]]).T,
                columns=bands_names)

    month_znh_indivs=pd.DataFrame(np.concat([annual_stack[:,znh_dataset_coords[0], znh_dataset_coords[1]],
                month_stack[:, znh_dataset_coords[0], znh_dataset_coords[1]]]).T,
                columns=bands_names)  

    #rapport s1
    s1vv_a_name=parse_band_name(config["mensual_imgs_fps"]["s1vv_a"])
    s1vh_a_name=parse_band_name(config["mensual_imgs_fps"]["s1vh_a"])
    s1vv_d_name=parse_band_name(config["mensual_imgs_fps"]["s1vv_d"])
    s1vh_d_name=parse_band_name(config["mensual_imgs_fps"]["s1vh_d"])
    month_zh_indivs["s1vv-s1vh_a_mensuel"]=month_zh_indivs[s1vv_a_name]-month_zh_indivs[s1vh_a_name]
    month_znh_indivs["s1vv-s1vh_a_mensuel"]=month_znh_indivs[s1vv_a_name]-month_znh_indivs[s1vh_a_name]
    month_zh_indivs["s1vv-s1vh_d_mensuel"]=month_zh_indivs[s1vv_d_name]-month_zh_indivs[s1vh_d_name]
    month_znh_indivs["s1vv-s1vh_d_mensuel"]=month_znh_indivs[s1vv_d_name]-month_znh_indivs[s1vh_d_name]

    # indices optiques
    optic_keys=["b2", "b3", "b4", "b8", "b11", "b12"]
    optic_names=[parse_band_name(config["mensual_imgs_fps"][key]) for key in optic_keys]

    b2, b3, b4, b8, b11, b12=month_zh_indivs[optic_names].values.T

    month_zh_indivs["ndvi"]=(b8-b4)/(b8+b4)
    month_zh_indivs["evi"]=2.5*(b8-b4)/(b8+6*b4-7.5*b2+1)
    month_zh_indivs["ndwi"]=(b8-b12)/(b8+b12)
    month_zh_indivs["ndwi2"]=(b3-b8)/(b3+b8)
    month_zh_indivs["mndwi"]=(b3-b12)/(b3+b12)
    month_zh_indivs["ndmi"]=(b8-b11)/(b8+b11)
    month_zh_indivs["msi"]=b11/b8
    month_zh_indivs["bi"]=np.sqrt((b4**2+b3**2)/2)
    month_zh_indivs["bi2"]=np.sqrt((b4**2+b3**2+b8**2)/3)
    month_zh_indivs["bsi"]=(b11+b4-b8-b2)/(b11+b4+b8+b2)

    b2, b3, b4, b8, b11, b12=month_znh_indivs[optic_names].values.T
    
    month_znh_indivs["ndvi"]=(b8-b4)/(b8+b4)
    month_znh_indivs["evi"]=2.5*(b8-b4)/(b8+6*b4-7.5*b2+1)
    month_znh_indivs["ndwi"]=(b8-b12)/(b8+b12)
    month_znh_indivs["ndwi2"]=(b3-b8)/(b3+b8)
    month_znh_indivs["mndwi"]=(b3-b12)/(b3+b12)
    month_znh_indivs["ndmi"]=(b8-b11)/(b8+b11)
    month_znh_indivs["msi"]=b11/b8
    month_znh_indivs["bi"]=np.sqrt((b4**2+b3**2)/2)
    month_znh_indivs["bi2"]=np.sqrt((b4**2+b3**2+b8**2)/3)
    month_znh_indivs["bsi"]=(b11+b4-b8-b2)/(b11+b4+b8+b2)   

    #bandes temporelles
    sin_temporel=np.sin((month/12)*2*np.pi)
    cos_temporel=np.cos((month/12)*2*np.pi)
    month_zh_indivs["sin_annuel"]=sin_temporel
    month_zh_indivs["cos_annuel"]=cos_temporel
    month_znh_indivs["sin_annuel"]=sin_temporel
    month_znh_indivs["cos_annuel"]=cos_temporel
    month_zh_indivs["month"]=month
    month_znh_indivs["month"]=month
    

    #ajout colonnes dfs coords
    month_znh_indivs["line"]=znh_dataset_coords[0]
    month_znh_indivs["col"]=znh_dataset_coords[1]

    month_zh_indivs=month_zh_indivs.dropna().join(zh_dataset_coords.reset_index(drop=True))
    
    zh_indivs.append(month_zh_indivs.copy())
    znh_indivs.append(month_znh_indivs.copy())

print("export")
# %%
zh_indivs=pd.concat(zh_indivs)
znh_indivs=pd.concat(znh_indivs)

# %%
zh_indivs.to_pickle(join(config["out_folder"], "zh_indivs.pkl"))
znh_indivs.dropna().to_pickle(join(config["out_folder"], "znh_indivs.pkl"))


