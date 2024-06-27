# -*- coding: utf-8 -*-

"""
Script complementary of script analyse_traj and script_model for deduct some swimming parameters of Scapholeberis mucronata.

Author: Renaud Serre
Creation: 30.04.2024


History of modifications
30.04.2024 : - creation of the script
             - export data .npy of x,y for each frames 

Infos Trackpy at https://soft-matter.github.io/trackpy/dev/tutorial/walkthrough.html
"""



# =============================================================================
# Packages
# =============================================================================
from __future__ import division, unicode_literals, print_function  # for compatibility with Python 2 and 3

import sys

# connection cluster à distance
connection_cluster = True

import matplotlib as mpl
if connection_cluster:
    mpl.use('agg')
import matplotlib.pyplot as plt

import numpy as np
import pandas as pd
from pandas import DataFrame, Series  # for convenience

import pims
import trackpy as tp

# =============================================================================
# Initialisation
# =============================================================================
mpl.rc('figure',  figsize=(10, 5))
mpl.rc('image', cmap='gray')

test_param = False # if we need tp.locate

# =============================================================================
# Import photos
# =============================================================================

# indiquez ici le nom du dossier à traiter (AAMMJJ)

# file_photos = "240514_expe_1"
# file_photos = "240516_expe_2"
# file_photos = "240523_expe_3"
# file_photos = "240524_expe_4_1"
file_photos = "240528_expe_5"

# parametres de traitement d'images
var_file = 100 # 
lim = 4000
lim_med = 300 #100
display_label = False

file_photos_scale_bar = "240514"

if not connection_cluster:
    path_photos = "D:/CODE_stage/photos/" + file_photos
    path_photo_scale = "D:/CODE_stage/photos/scales_bars/" 
    output_trajectories = "D:/CODE_stage/fichiers"
    output_figures = "D:/CODE_stage/figures"
else:
    path_photos = "/home/reserre/photos/" + file_photos
    # path_photo_scale = "/home/reserre/"
    output_trajectories = "/home/reserre/output_fichiers/"
    output_figures = "/home/reserre/output_figures/"
    

frames = pims.open(path_photos + '/*.tiff')
scale = pims.open(path_photo_scale + file_photos_scale_bar + ".tiff")

pixel_size_mm = 1/25
micron_par_pixels = (10**(-3))/pixel_size_mm
frames_par_sec = 27

# =============================================================================
# Test img
# =============================================================================

fig, axes = plt.subplots(1, 2, figsize=(15, 5))
axes[0].imshow(frames[1]) # first image
axes[1].imshow(frames[-1]) # last image
axes[2].imshow(scale[0]) # scale
plt.tight_layout()

plt.savefig(output_figures + f'verif_img_{file_photos}.png', dpi=300, pad_inches=0.1)


# =============================================================================
# Normalisation img par mediane
# =============================================================================


median_value = np.median(frames[0:lim_med],axis = 0)

frames_flatfield = np.divide(frames[0:lim], median_value)

fig, axes = plt.subplots(1, 2, figsize=(15, 5))
axes[0].imshow(frames_flatfield[1]) # first image
axes[1].imshow(frames_flatfield[-1]) # last image
plt.tight_layout()


plt.savefig(output_figures + f'verif_img_normalisee_{file_photos}.png', dpi=300, pad_inches=0.1)

# =============================================================================
# Parameters particles
# =============================================================================

diameter = 9 # diameter of particle
minmass = 1 # masse min en terme de luminosité => somme des valeurs des px
noise_size = 1 # 1 by default. Gaussian blurring kernel => filtre les impureté du signal, plus il augmente et plus on filtre des gros signaux
smoothing_size = 11 # diameter by default, enlève les biais de grande échelle (gradient de lum par ex)
separation = 15 # distance en dessous de laquelle on regroupe la même particle

# locate renvoie x,y et mass pour chaque frame => pour regler parametre
# look for dark feature with invert = True

if test_param:
    f = tp.locate(frames_flatfield[0], 
                  diameter = diameter, 
                  invert=True 
                  ,minmass = minmass, 
                  noise_size = noise_size, 
                  smoothing_size = smoothing_size,
                  separation=separation) 
    print("locate ok")
    tp.annotate(f, frames_flatfield[0])
    
    # histogram différentes mass de chaque particules
    fig, ax = plt.subplots()
    ax.hist(f['mass'], bins=100)
    ax.set(xlabel='mass', ylabel='count')

# batch = tp.locate mais sur chaques frames. Associe
f = tp.batch(frames_flatfield, diameter, 
              invert=True ,minmass = minmass, noise_size = noise_size, 
              smoothing_size = smoothing_size,
              separation=separation)
print("batch ok")

# =============================================================================
# Analyse trajectoires
# =============================================================================

memory = 0 # nbr de frame où on garde en mémoire un ind s'il disparait
max_deplacement = 80 # nbr de pixels max entre chaque deplacement

t = tp.link(f, max_deplacement, memory=memory)
print("link ok")

# filtre les particulesqui ont une trajectoire très petite (poussières ou autres)
t_filter = tp.filter_stubs(t, 0)
print('Before:', t_filter['particle'].nunique())
print('After:', t_filter['particle'].nunique())

# filtre les impureté avec une grosse ocillation
x_var_per_particle = t_filter.groupby('particle')['x'].var()
y_var_per_particle = t_filter.groupby('particle')['y'].var()
var_tot = 0
var_tot = x_var_per_particle + y_var_per_particle

print(var_tot)

particles_ok = []

for particle,var in var_tot.items():
    if var > var_file :
        particles_ok.append(particle)
print(len(particles_ok))

# filtration avec un masque
mask = t_filter['particle'].isin(particles_ok)
t_filter2 = t_filter[mask]

# conversion en mm

t_filter2['x'] = t_filter2['x'] * pixel_size_mm
t_filter2['y'] = t_filter2['y'] * pixel_size_mm

# faie variance pour éliminer petites trajectoires

# plot trajectoires
plt.close('all')

fig, ax = plt.subplots()

# Tracer les trajectoires sur la nouvelle figure
tp.plot_traj(t_filter2, ax=ax,label=display_label)

fig.savefig(output_figures + f'trajectoires_{file_photos}.png', dpi=300, pad_inches=0.1)

# save en csv pour pouvoir l'utiliser dans analyse_traj
t_filter2.to_csv(output_trajectories + f"infos_particles_{file_photos}.csv", index=False)

# np.save(output_trajectories + f"infos_particles_{file_photos}",t_filter)
