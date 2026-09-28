
# Notebook pour calculer les hauteurs par rapport au voisinnage.
# on prend des kernels circulaires de 100m, 500m et 1000m de diamètre.


# %%
import numpy as np
import matplotlib.pyplot as plt
import rasterio
from dask.distributed import Client
from functools import partial
import time
from scipy.signal import fftconvolve
from os.path import join
import sys
from pathlib import Path
import os

project_path=str(Path(__file__).resolve().parent.parent.parent)
sys.path.append(os.path.abspath(project_path))
import libs.shreklib as shreklib


import argparse

parser = argparse.ArgumentParser()

parser.add_argument("rge_fp", help="Chemin de fichier du MNT.")
parser.add_argument("out_folder", help="Dossier d'export des ITP")

args = parser.parse_args()

rge_fp=args.rge_fp
out_folder=args.out_folder


# %%
def circular_kernel(r):
    xx, yy=np.meshgrid(range(-r,r+1), range(-r, r+1))
    circle=(np.floor(xx**2+yy**2)<=r**2).astype(float)
    circle/=circle.sum()
    return circle

# %%
with rasterio.open(rge_fp, "r") as src:
    mnt=src.read(1)

# %%
mnt_nanned=np.where((np.isnan(mnt)) |( mnt==-99999), 0, mnt)

# %%
mnt_mean_100m=fftconvolve(mnt_nanned, circular_kernel(5), mode="same")

# %%
mnt_mean_500m=fftconvolve(mnt_nanned, circular_kernel(25), mode="same")

# %%
mnt_mean_1000m=fftconvolve(mnt_nanned, circular_kernel(50), mode="same")

# %%
cuvette100m=mnt-mnt_mean_100m
cuvette500m=mnt-mnt_mean_500m
cuvette1000m=mnt-mnt_mean_1000m

# %%

# %%
for cuvette, size in zip([cuvette100m, cuvette500m, cuvette1000m], [100,500,1000]):
    shreklib.tif_export(cuvette, shreklib.ref_grid["transform"], shreklib.ref_grid["crs"].item(), join(out_folder, f"cuvette{size}m.tif"))


