# %%
"""
Copyright (©) CEREMA/DTerOCC/DT/OSECC All rights reserved
auteur : Niels Saura
date de création : Septembre 2026

Script pour réaliser l'inférence d'un modèle obtenu via 2_training.py à partir des images fournies en entrée de 1_create_datasets.py.
entrée : chemin du fichier de config (cf 3_inference.yaml)
sortie :
- carte_zones_humides_potentielles : geotiff de la cartographie des zones humides potentielles
- probabilités mensuelles : geotiff des probabilités pour chacun des mois fournis en entrée. Les probabilités sont quantifiées sur les entiers de 0 à 254. 255 est réservé aux nodata.
"""

# %%
#Imports
import sys
import os
from pathlib import Path
import joblib
import gc
from tqdm import tqdm

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

import torch
torch.no_grad()

# %%
yaml_config="/home/scgsi/Documents/scripts_prod/pipeline/3_inference.yaml"

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
valid_line, valid_col=np.where(valid_mask)

# %%
def monthly_scale(df, months, scalers, column_orders):
    df["month"]=months
    df=df[column_orders+["month"]]
    df_cr=[]
    for month, group in df.groupby("month"):
        cred_tens=scalers[month].transform(torch.from_numpy(group.values))
        df_cr.append(pd.DataFrame(cred_tens, columns=df.columns))
    df_cr=pd.concat(df_cr).drop(columns=["month"])
    return df_cr.reset_index(drop=True)

# %%
model_dict=joblib.load(config["model_dict_fp"])

# %%
# construction des individus

# chargement des images annuelles
print("inférence du modèle")
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

preds=[]
lines=[]
col=[]


# %%

for month in tqdm(range(months_count)):
    if month==0:
        continue
    month_stack=[]
    for fp in mensual_imgs_fps:
        with rasterio.open(fp, "r") as src:
            month_stack.append(src.read(month+1))
    month_stack=np.stack(month_stack)

    month_indivs=pd.DataFrame(np.concat([annual_stack[:,valid_line, valid_col],
                month_stack[:, valid_line, valid_col]]).T,
                columns=bands_names)

    #rapport s1
    s1vv_a_name=parse_band_name(config["mensual_imgs_fps"]["s1vv_a"])
    s1vh_a_name=parse_band_name(config["mensual_imgs_fps"]["s1vh_a"])
    s1vv_d_name=parse_band_name(config["mensual_imgs_fps"]["s1vv_d"])
    s1vh_d_name=parse_band_name(config["mensual_imgs_fps"]["s1vh_d"])
    month_indivs["s1vv-s1vh_a_mensuel"]=month_indivs[s1vv_a_name]-month_indivs[s1vh_a_name]
    month_indivs["s1vv-s1vh_d_mensuel"]=month_indivs[s1vv_d_name]-month_indivs[s1vh_d_name]


    # indices optiques
    optic_keys=["b2", "b3", "b4", "b8", "b11", "b12"]
    optic_names=[parse_band_name(config["mensual_imgs_fps"][key]) for key in optic_keys]

    b2, b3, b4, b8, b11, b12=month_indivs[optic_names].values.T

    month_indivs["ndvi"]=(b8-b4)/(b8+b4)
    month_indivs["evi"]=2.5*(b8-b4)/(b8+6*b4-7.5*b2+1)
    month_indivs["ndwi"]=(b8-b12)/(b8+b12)
    month_indivs["ndwi2"]=(b3-b8)/(b3+b8)
    month_indivs["mndwi"]=(b3-b12)/(b3+b12)
    month_indivs["ndmi"]=(b8-b11)/(b8+b11)
    month_indivs["msi"]=b11/b8
    month_indivs["bi"]=np.sqrt((b4**2+b3**2)/2)
    month_indivs["bi2"]=np.sqrt((b4**2+b3**2+b8**2)/3)
    month_indivs["bsi"]=(b11+b4-b8-b2)/(b11+b4+b8+b2)

    #bandes temporelles
    sin_temporel=np.sin((month/12)*2*np.pi)
    cos_temporel=np.cos((month/12)*2*np.pi)
    month_indivs["sin_annuel"]=sin_temporel
    month_indivs["cos_annuel"]=cos_temporel
    nan_idx=np.any(~np.isfinite(month_indivs.values), dim=1)
    month_indivs=month_indivs.loc[~nan_idx]

    month_line, month_col=valid_line[~nan_idx], valid_col[~nan_idx]
    month_indivs_cr=monthly_scale(month_indivs, month, model_dict["scalers"], model_dict["column_orders"].tolist())
    month_indivs_pcaed=pd.DataFrame(model_dict["pca"].transform(month_indivs_cr), columns=[f"CP{i}" for i in range(model_dict["pca"].n_components_)])
    month_indivs_pcaed[["sin_annuel", "cos_annuel"]]=month_indivs[["sin_annuel", "cos_annuel"]]
    pred=np.round(254*model_dict["rf"].predict_proba(month_indivs_pcaed)[:,1]).astype(np.uint8)

    preds.append(pred)
    lines.append(month_line)
    col.append(month_col)

    del pred, month_line, month_col, month_indivs, sin_temporel, cos_temporel, nan_idx, b2, b3, b4, b8, b11, b12, month_stack
    gc.collect()


# %%
month_indivs_cr=monthly_scale(month_indivs, month, model_dict["scalers"], model_dict["column_orders"].tolist())
month_indivs_pcaed=pd.DataFrame(model_dict["pca"].transform(month_indivs_cr), columns=[f"CP{i}" for i in range(model_dict["pca"].n_components_)])
month_indivs_pcaed[["sin_annuel", "cos_annuel"]]=month_indivs[["sin_annuel", "cos_annuel"]]
pred=np.round(254*model_dict["rf"].predict_proba(month_indivs_pcaed)[:,1]).astype(np.uint8)

preds.append(pred)
lines.append(month_line)
col.append(month_col)

del pred, month_line, month_col, month_indivs, sin_temporel, cos_temporel, nan_idx, b2, b3, b4, b8, b11, b12, month_stack
gc.collect()


# %%
test=torch.full((ref_shape[0], ref_shape[1]), 255, dtype=torch.uint8)
test[lines[0], col[0]]=torch.from_numpy(preds[0])

# %%
import matplotlib.pyplot as plt

# %%
shreklib.tif_export(test.numpy(), shreklib.ref_grid["transform"], 32631, "/mnt/RAM_disk/test.tif", "uint8")

# %%
probas_imgs=torch.full((months_count, *ref_shape), 255, dtype=torch.uint8)
for month,(pred, month_line, month_col) in enumerate(zip(preds, lines, col)):
    probas_imgs[month, month_line, month_col]=torch.from_numpy(pred)

# %%
carte_zones_humides=((probas_imgs>=np.round(2.54*model_dict["proba_seuil_optim"])).sum(dim=0)>=6).type(torch.uint8).numpy()

# %%
shreklib.tif_export(carte_zones_humides, np.array(ref_transform).reshape((3,3)), ref_crs, join(config["out_folder"], "carte_zones_humides_potentielles.tif"), "uint8")
shreklib.tif_export(probas_imgs, np.array(ref_transform).reshape((3,3)), ref_crs, join(config["out_folder"], "probabilites_mensuelles.tif"), "uint8")


