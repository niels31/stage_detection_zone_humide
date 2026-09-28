"""
Copyright (©) CEREMA/DTerOCC/DT/OSECC All rights reserved
auteur : Niels Saura
date de création : Septembre 2026

librairie regroupant des fonctions, attributs et classes utilitaires dans différents scripts du dépôt
"""

import numpy as np
import rasterio
from os.path import join
from affine import Affine
from concurrent.futures import ProcessPoolExecutor
from functools import partial
import torch

#fonction pour diminuer la résolution d'une image pour accélérer son affichage avec plt.imshow
#img doit être un array numpy 2D. Agg est le facteur de diminution de résolution
def downsample_img(img, agg=10, how="mean"):
    img=img[np.newaxis, ...] if img.ndim==2 else img
    
    b, l, c=img.shape
    nl=agg*(l//agg)
    nc=agg*(c//agg)
    img=img[:, :nl,:nc]
    img=img.reshape((b, int(nl/agg), agg, int(nc/agg), agg))
    match how:
        case "mean":
            return img.mean(axis=(2,4)).squeeze()
        case "median":
            return np.median(img,axis=(2,4)).squeeze()
        case "max":
            return np.max(img,axis=(2,4)).squeeze()


#fonction d'export d'un array numpy vers un geotiff
#img doit être au format c, h, w ou si un seul canal h, w suffit.
#transform est une matrice de transformation en coordonnées homogènes numpy.
def tif_export(img, transform, crs, filepath, dtype):
    transform = Affine(*transform[:2, :].ravel())

    count, height, width = (
        img.shape if len(img.shape) == 3
        else (1, *img.shape)
    )

    img = img.reshape((count, height, width))

    with rasterio.open(
        filepath,
        "w",
        driver="GTiff",
        height=height,
        width=width,
        count=count,
        dtype=dtype,
        crs=crs,
        transform=transform,

        # Organisation du fichier
        interleave="band",
        tiled=True,
        blockxsize=512,
        blockysize=512

    ) as export:

        for band in range(count):
            export.write(img[band], band + 1)

#Fonction pour supprimer les valeurs extérieures aux percentiles spécifiés
def adjust_hist(img, percentiles=[1,99]):
    img=img[np.newaxis, ...] if img.ndim==2 else img
    #on s'assure d'avoir une image en C, H, W
    
    low, high=np.percentile(img, percentiles, axis=(1,2))
    low=low[:, None, None]
    high=high[:, None, None]
    
    img=np.clip(img, low, high)
    img=img-low
    img=img/(high-low)
    return img.squeeze()

#prend une image c, h, w, lui applique downsample, adjust_hist
#et transpose ses dimensions en h, w, c
#la fonction suppose que les trois premières bandes sont b, g, r et renvoie uniquement r, g, b
def rgb_visualisable(img, agg=10, percentiles=[1,99], downsample=True):
    img=img[2::-1]
    if downsample:
        img=downsample_img(img, agg=agg)
    img=adjust_hist(img, percentiles=percentiles)
    return np.transpose(img, (1,2,0))

#Fonction pour charger les fichiers customs géoréférencés npz au format c,h,w.
#Il s'agit du format de pentes et de l3 processed.
def load_custom_npz(fp, bands="ALL", known_crs=False):
    file=np.load(fp)
    if bands=="ALL":
        bands=[b for b in file.keys() if b not in ["crs", "transform"]]
    img=np.stack([file[b] for b in bands])
    crs=file["crs"] if not known_crs else known_crs
    return img, file["transform"], crs.item()


def circle_kernel(r):
    ll, cc=np.meshgrid(range(-r, r+1), range(-r, r+1))
    kernel=np.zeros((2*r+1, 2*r+1))
    test_rayon=ll**2+cc**2-r**2
    test_rayon=np.floor(test_rayon)
    kernel[test_rayon<=0]=1
    return kernel

def percentile_clip(img, percentiles=(1,99)):
    img=img[np.newaxis, ...] if img.ndim==2 else img

    notnan=~np.isnan(img)
    img_nanned=img[notnan]
    high, low=np.percentile(img_nanned, percentiles)
    return np.clip(img, high, low).squeeze()
    
#fonction utilisée par batch_map
def process_batch(batch, function):
    return [function(img) for img in batch]

#fonction qui applique une fonction fournie en argument à un array d'images au format c, h, w.
#La fonction fournie est appliquée séparément à chaque image avec ProcessPoolExecutor.
#L'intérêt de cette fonction est de gérer dynamiquement la taille des batchs à fournir à chaque processus pour optimiser l'usage de chaque cpu.
# j'ai en effet remarqué que lorsqu'on fournissait beaucoup d'éléments à process pool executor, il semblait passer beaucoup de temps à créer les jobs.
def batch_map(function, elements, num_workers=32):
    n=len(elements)

    num_big_batches=n%num_workers
    size_small_batches=n//num_workers

    big_batches=[elements[batch_start:batch_start+size_small_batches+1] for batch_start in range(0,num_big_batches*(size_small_batches+1), size_small_batches+1)]
    small_batches=[elements[batch_start:batch_start+size_small_batches] for batch_start in range(num_big_batches*(size_small_batches+1), n, size_small_batches)]

    batches=big_batches+small_batches

    partial_batch_function=partial(process_batch, function=function)

    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        full_processed=list(executor.map(partial_batch_function, batches))
    
    processed_elements=[element for sublist in full_processed for element in sublist]
    return processed_elements

ref_grid=np.load("/mnt/Data/30_Stages_Encours/2026/2026_SHREQ_Niels/02_Donnees/input/ref_grid.npz")

class torchScaler():
    def __init__(self, tens):
        self.stds=tens.std(dim=0)
        self.means=tens.mean(dim=0)
    
    def transform(self, tens):
        return (tens-self.means[None, :])/self.stds[None,:]

def gaussian(arr_square, sig): # sig est l'écart type de la gaussienne
    return (1/(sig*np.sqrt(2*np.pi)))*np.exp((-1/2)*arr_square/sig**2)

#kernel gaussien d'écart-type r/2 et tronqué avec le cercle de rayon r, pour en faire un kernel gaussien circulaire, qui ne sur-représente pas les angles.
def gaussian_kernel(r=10):
    ll, cc=np.meshgrid(range(-r, r+1), range(-r, r+1))
    kernel=np.zeros((2*r+1, 2*r+1))
    test_rayon=ll**2+cc**2-r**2
    test_rayon_int=np.floor(test_rayon)
    kernel[test_rayon_int<=0]=1
    gaussian_img=gaussian(ll**2+cc**2, r/2)
    gaussian_img=np.where(kernel, gaussian_img,0)
    return gaussian_img