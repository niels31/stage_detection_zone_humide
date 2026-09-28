import rasterio
from scipy.signal import convolve2d
import numpy as np
from os.path import join
import sys
from pathlib import Path
import os

project_path=str(Path(__file__).resolve().parent.parent.parent)
sys.path.append(os.path.abspath(project_path))
import libs.shreklib as shreklib
import argparse

parser = argparse.ArgumentParser()

parser.add_argument("mnt_fp", help="Chemin de fichier du MNT.")
parser.add_argument("out_folder", help="Dossier d'export des ITP")

args = parser.parse_args()

mnt_fp=args.mnt_fp
out_folder=args.out_folder

with rasterio.open(mnt_fp, "r") as src:
    mnt=src.read(1)
    mnt_transform=src.transform
r=np.array(mnt_transform)[0]
dx_kernel=np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
dy_kernel=dx_kernel.T
dz_dx=convolve2d(mnt, dx_kernel, mode="same")
dz_dy=convolve2d(mnt, dy_kernel, mode="same")
dz_dx/=8*r
dz_dy/=8*r
pente=np.sqrt(dz_dx**2, dz_dy**2)
with rasterio.open(mnt_fp, "r") as src:
    transform=src.transform
transform=np.array(transform).reshape((3,3))
shreklib.tif_export(pente, transform, 32631, join(out_folder,"pente_test.tif"), "float64")