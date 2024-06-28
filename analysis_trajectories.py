# -*- coding: utf-8 -*-

"""
Script complementary of script analyse_img and script_model for deduct some swimming parameters of Scapholeberis mucronata.

Author: Renaud Serre
Creation: 30.04.2024


History of modifications
30.04.2024 : creation of the script
24/06/2024 : mise en forme majeur des figures
25/06/2024 : ajout du calcul de la tortusité et de l'uptake des trajectoires'

Infos Trackpy at https://soft-matter.github.io/trackpy/dev/tutorial/walkthrough.html

"""



# =============================================================================
# Packages
# =============================================================================
from __future__ import division, unicode_literals, print_function  # for compatibility with Python 2 and 3

import sys

import math as mt
import matplotlib as mpl


# connection cluster à distance
connection_cluster = False

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


# file = "240514_expe_1"
file = "240516_expe_2"
# file = "240523_expe_3"
# file = "240524_expe_4_1"
# file = "240528_expe_5"

if not connection_cluster:
    data_tot = pd.read_csv('C:/Users/serre/OneDrive/Bureau/STAGE/CODE/fichiers/infos_particles_'+ file +".csv")
    output_trajectories = "C:/Users/serre/OneDrive/Bureau/STAGE/CODE/fichiers/"
    output_figures =  "C:/Users/serre/OneDrive/Bureau/STAGE/CODE/figures/"
else:
    data_tot = pd.read_csv("/home/reserre/output_fichiers/infos_particles_" + file +".csv")
    output_trajectories = "/home/reserre/output_fichiers/"
    output_figures = "/home/reserre/output_figures/"

# =============================================================================
# Parameters
# =============================================================================
mass_ind_heavy = 3.8 # treshold for diference between light and heavy individus

pixel_size_mm = 1/25
micron_par_pixels = (10**(-3))/pixel_size_mm
frames_par_sec = 30

# =============================================================================
# Séparation en fonction de la masse
# =============================================================================
mean_mass_par_part = data_tot.groupby('particle')['mass'].mean()
print(mean_mass_par_part)

particles_heavy = []
particles_light = []


# separation individus adulte et juvéniles en fonction d'un seuil
for particle, mean_mass in mean_mass_par_part.items():
    if mean_mass > mass_ind_heavy:
        particles_heavy.append(particle)
        print("particle ", particle, "is heavy : ",round(mean_mass,2))

    else :
        particles_light.append(particle)
        print("particle ", particle, "is light : ",round(mean_mass,2))


# isoler x,y et frames pour particles heavy and ligth
data_heavy = data_tot[data_tot['particle'].isin(particles_heavy)][['x', 'y', 'frame','particle']]
data_light = data_tot[data_tot['particle'].isin(particles_light)][['x', 'y', 'frame','particle']]

# save en csv
data_heavy.to_csv(output_trajectories + "traj_heavy_" + f"{file}" + ".csv", index=False)
data_light.to_csv(output_trajectories + "traj_light_" + f"{file}" + ".csv", index=False)



# =============================================================================
# Calculation of MSD
# =============================================================================
#### MSD for each mass
# heavy
plt.close('all')
im = tp.imsd(data_heavy, micron_par_pixels , frames_par_sec)
em = tp.emsd(data_heavy, micron_par_pixels, frames_par_sec)

fig, ax = plt.subplots()
ax.plot(im.index, im, 'k-', alpha=0.1)
ax.set(ylabel=r'MSD (mm²)',
       xlabel='$t$ (s)')
ax.set_xscale('log')
ax.set_yscale('log')
param_MSD = tp.utils.fit_powerlaw(em,color="black",plot=False)    
n_MSD = float(param_MSD['n'])
A_MSD = float(param_MSD['A'])
n_round = str(round(n_MSD,2))
A_round = round(A_MSD,4)
ax.text(0.05, 0.01, fr'$MSD = {A_round} \cdot t^{{{n_round}}}$', color='black', fontsize=20)
tp.utils.fit_powerlaw(em,color="black",plot=True)   


fig.savefig(output_figures + f'msd_heavy_{file}.png', dpi=300, pad_inches=0.1)

# light
plt.close('all')
im = tp.imsd(data_light, micron_par_pixels , frames_par_sec)
em = tp.emsd(data_light, micron_par_pixels, frames_par_sec)

fig, ax = plt.subplots()
ax.plot(im.index, im, 'k-', alpha=0.1)
ax.set(ylabel=r'MSD (mm²)',
       xlabel='$t$ (s)')
ax.set_xscale('log')
ax.set_yscale('log')
param_MSD = tp.utils.fit_powerlaw(em,color="black",plot=False)    
n_MSD = float(param_MSD['n'])
A_MSD = float(param_MSD['A'])
n_round = str(round(n_MSD,2))
A_round = round(A_MSD,4)
ax.text(0.05, 0.01, fr'$MSD = {A_round} \cdot t^{{{n_round}}}$', color='black', fontsize=20)
tp.utils.fit_powerlaw(em,color="black",plot=True)   

fig.savefig(output_figures + f'msd_light_{file}.png', dpi=300, pad_inches=0.1)

# all
plt.close('all')
im = tp.imsd(data_tot, micron_par_pixels , frames_par_sec)
em = tp.emsd(data_tot, micron_par_pixels, frames_par_sec)

fig, ax = plt.subplots()
ax.plot(im.index, im, 'k-', alpha=0.1)
ax.set(ylabel=r'MSD (mm²)',
       xlabel='$t$ (s)')
ax.set_xscale('log')
ax.set_yscale('log')
param_MSD = tp.utils.fit_powerlaw(em,color="black",plot=False)    
n_MSD = float(param_MSD['n'])
A_MSD = float(param_MSD['A'])
n_round = str(round(n_MSD,2))
A_round = round(A_MSD,4)
ax.text(0.05, 0.01, fr'$MSD = {A_round} \cdot t^{{{n_round}}}$', color='black', fontsize=20)
tp.utils.fit_powerlaw(em,color="black",plot=True)   

fig.savefig(output_figures + f'msd_tot_{file}.png', dpi=300, pad_inches=0.1)

# =============================================================================
# Display instantaneous speed on graph
# =============================================================================

# Calculate particle velocities
# Calculate particle velocities manually
def calculate_velocities(data, micron_par_pixels, frames_par_sec):
    
    data_sorted = data.sort_values(by=['particle', 'frame'])
    data_sorted['dx'] = data_sorted.groupby('particle')['x'].diff()
    data_sorted['dy'] = data_sorted.groupby('particle')['y'].diff()
    data_sorted['dt'] = data_sorted['frame'].diff() / frames_par_sec
    data_sorted['velocity'] = np.sqrt(data_sorted['dx']**2 + data_sorted['dy']**2) / data_sorted['dt']
    
    return data_sorted['velocity']

# Calculate velocities for heavy and light particles
velocity_tot = calculate_velocities(data_tot, micron_par_pixels, frames_par_sec)
velocities_heavy = calculate_velocities(data_heavy, micron_par_pixels, frames_par_sec)
velocities_light = calculate_velocities(data_light, micron_par_pixels, frames_par_sec)


# =============================================================================
# Calculation of instantaneous parameters
# =============================================================================

def angle_trig(a, b, c):
    cos_C = (a**2 + b**2 - c**2) / (2 * a * b)
    angle_C = mt.acos(cos_C) # en radians

    return angle_C

def orientation(p1, p2, p3):
    # p1 = x[-3],y[-3], p2 = x[-2],y[-2], p3 = x[-1],y[-1] 
    # Produit vectoriel entre les vecteurs p1p2 et p2p3
    value = (p2[1] - p1[1]) * (p3[0] - p2[0]) - (p2[0] - p1[0]) * (p3[1] - p2[1])
    
    if value == 0:
        orientation = "Colinéaires"
    elif value > 0:
        orientation = "right"
    else:
        orientation = "left"
    
    return orientation


def calc_angles(data):
    particles = np.unique(data['particle']).tolist()
    angles_per_particle = {}
    
    for particle in particles:
        liste_angles = []
        
        particle_data = data[data['particle'] == particle]
        x = particle_data['x'].tolist()
        y = particle_data['y'].tolist()
        
        for i in range(2, len(x)):
            p1 = [x[i-2], y[i-2]]
            p2 = [x[i-1], y[i-1]]
            p3 = [x[i], y[i]]
            
            segment_1 = np.sqrt((x[i-1] - x[i-2])**2 + (y[i-1] - y[i-2])**2)
            segment_2 = np.sqrt((x[i] - x[i-1])**2 + (y[i] - y[i-1])**2)
            segment_3 = np.sqrt((x[i] - x[i-2])**2 + (y[i] - y[i-2])**2)
            
            angle = np.pi - angle_trig(segment_1, segment_2, segment_3)
            orientat = orientation(p1, p2, p3)
            
            if orientat == "left":
                angle = -angle
                
            liste_angles.append(angle)
        
        angles_per_particle[particle] = liste_angles
        
    return angles_per_particle

def calc_l(data):
    
    particles = np.unique(data['particle']).tolist()
    length_per_particle = {}
    
    for particle in particles:
        liste_length = []
        
        particle_data = data[data['particle'] == particle]
        x = particle_data['x'].tolist()
        y = particle_data['y'].tolist()
        
        for i in range(len(x)):
            length = np.sqrt((x[i] - x[i-1])**2 + (y[i] - y[i-1])**2)
            
            liste_length.append(length)
        
        length_per_particle[particle] = liste_length
    
    return length_per_particle

length_per_particle = calc_l(data_tot)
angles_per_particle = calc_angles(data_tot)

all_length = [length for liste_length in length_per_particle.values() for length in liste_length]
all_angles = [angle for liste_angles in angles_per_particle.values() for angle in liste_angles]

#pd.DataFrame(all_length, columns=['valeur']).to_csv("C:/Users/serre/OneDrive/Bureau/STAGE/CODE/fichiers/all_length.csv", index=False)
# pd.DataFrame(all_angles, columns=['valeur']).to_csv("C:/Users/serre/OneDrive/Bureau/STAGE/CODE/fichiers/all_angles.csv", index=False)

# =============================================================================
# Graphs
# =============================================================================
def densite_expo(x, param):
    y = (param) * np.exp(-x*param)
    return y

def calc_densite_lognormal(x, esp, var):

    sigma2 = mt.log(1 + (var/esp**2))
    mu = mt.log(esp) - (1/2)*sigma2
    
    y = (np.exp(-(np.log(x) - mu)**2 / (2 * sigma2)) / (x * np.sqrt(sigma2) * np.sqrt(2 * np.pi)))
    
    return y

def plot_distrib_l_alpha(data_tot, data_light, data_heavy):
    
    angles_per_particle_all = calc_angles(data_tot)
    angles_per_particle_light = calc_angles(data_light)
    angles_per_particle_heavy = calc_angles(data_heavy)
    
    velocities_all = calculate_velocities(data_tot, micron_par_pixels, frames_par_sec)
    velocities_light = calculate_velocities(data_light, micron_par_pixels, frames_par_sec)
    velocities_heavy = calculate_velocities(data_heavy, micron_par_pixels, frames_par_sec)
    
    # Distributions totales
    
    all_angles_all = [angle for liste_angles in angles_per_particle_all.values() for angle in liste_angles]
    all_angles_light = [angle for liste_angles in angles_per_particle_light.values() for angle in liste_angles]
    all_angles_heavy = [angle for liste_angles in angles_per_particle_heavy.values() for angle in liste_angles]
    
    all_angles_all_abs = [abs(x) for x in all_angles_all]
    all_angles_light_abs = [abs(x) for x in all_angles_light]
    all_angles_heavy_abs = [abs(x) for x in all_angles_heavy]
    
    opac = 0.5 # définition de l'opacité
    bins = 300 # nbr barres
    
    ### alpha ----------------------------------------------------------------------------
    plt.close('all')

    var_alpha_light = 1/np.mean(all_angles_light_abs)
    var_alpha_heavy = 1/np.mean(all_angles_heavy_abs)
    
    x_values_light = np.linspace(0, max(all_angles_light), 1000)
    y_values_light = densite_expo(x_values_light, var_alpha_light)

    x_values_heavy = np.linspace(0, max(all_angles_heavy), 1000)
    y_values_heavy = densite_expo(x_values_heavy, var_alpha_heavy)

    plt.subplot(1,4,1)
    plt.xlabel('Radians',fontsize=15)
    plt.ylabel('Densité',fontsize=15)
    
    plt.hist(all_angles_heavy, density=True, alpha=opac, color="grey", label="Adultes",bins=bins)
    plt.hist(all_angles_light, density=True, alpha=opac-0.25, color="grey", label="Juvéniles",bins=bins)
    
    plt.plot(x_values_light, y_values_light/2, color='grey',alpha=opac)
    plt.plot(-x_values_light, y_values_light/2, color='grey',alpha=opac)
    
    plt.plot(x_values_heavy, y_values_heavy/2, color='grey')
    plt.plot(-x_values_heavy, y_values_heavy/2, color='grey')
  
    var_alpha_round_light = round(var_alpha_light,3)
    var_alpha_round_heavy = round(var_alpha_heavy,3)
    
    # plt.text(-2.5,0.5, f'βi = {var_alpha_round_light}', color='black', fontsize=14)
    # plt.text(-2.5,1, f'βi = {var_alpha_round_heavy}', color='black', fontsize=14)
             
    plt.legend(loc="best")

    ### vit instant ------------------------------------------------------------------------------
    
    plt.subplot(1,4,2)
    
    plt.xlim([0,6.5])
    plt.xlabel('mm/s',fontsize=15)

    plt.hist(velocities_heavy, density=True, alpha=opac, color="grey", label="Adultes",bins=bins*3)
    plt.hist(velocities_light, density=True, alpha=opac-0.25, color="grey", label="Juvéniles",bins=bins*3)

    
    plt.legend()
    
    ### All ------------------------------------------------------------------------------
    
    plt.subplot(1,4,3)
    
    plt.xlabel('Radians',fontsize=15)
    
    var_alpha_all = 1/np.mean(all_angles_all_abs)
    plt.hist(all_angles_all, density=True, alpha=opac-0.1, color="black", label="Total",bins=bins)
    
    x_values_all = np.linspace(0, max(all_angles_all), 1000)
    y_values_all = densite_expo(x_values_all, var_alpha_all)
    
    plt.plot(x_values_all, y_values_all/2, color='black')
    plt.plot(-x_values_all, y_values_all/2, color='black')

    var_alpha_round_all = round(var_alpha_all,3)
    
    # plt.text(-2.5,1, f'βi = {var_alpha_round_all}', color='black', fontsize=20)
    
    plt.legend()

         
    plt.subplot(1,4,4)
    
    plt.xlim([0,6.5])
    plt.xlabel('mm/s',fontsize=15)
    # plt.ylabel('Densité',fontsize=20)
    
    plt.hist(velocities_all, density=True, alpha=opac-0.1, color="black", label="Total",bins=bins*3)
    
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(output_figures + f'NEW_____param_all_particles_{file}.png', dpi=300, pad_inches=0.1)

plot_distrib_l_alpha(data_tot, data_light, data_tot)
plot_distrib_l_alpha(data_tot, data_light, data_light)
plot_distrib_l_alpha(data_tot, data_light, data_heavy)
plt.close('all')
print("END")

# =============================================================================
# Calcul de la tortuosité et de l'uptake des particules
# =============================================================================



def calc_Tortuosity(data):
    particles = np.unique(data['particle']).tolist()
    tortuosities = {}
    
    for particle in particles:
        particle_data = data[data['particle'] == particle]
        x = particle_data['x'].tolist()
        y = particle_data['y'].tolist()
        
        # Calculate the shortest length
        shorter_length = np.sqrt((x[-1] - x[0])**2 + (y[-1] - y[0])**2)
        
        # Calculate the total length
        total_length = np.sum(np.sqrt(np.diff(x)**2 + np.diff(y)**2))
        
        tortuosity = total_length / shorter_length

        tortuosities[particle] = tortuosity
    
    return tortuosities,particles

tortuosities, particles = calc_Tortuosity(data_tot)
print(tortuosities)
tortuosities_list = list(tortuosities.values())

plt.close("all")
plt.hist(tortuosities_list,bins=len(particles)*5)
plt.savefig(output_figures + 'distrib_T.png', dpi=300, pad_inches=0.1)
plt.show()
med_T_Sm = np.median(tortuosities_list)
max_frames = max(data_tot['frame'])
print("médiane tortuosité :", med_T_Sm, f"sur {max_frames} frames")





def calc_uptake(data, plot_uptake,range_radius_Sm):
    particles = np.unique(data['particle']).tolist()
    uptakes = {}
    max_frames = max(data['frame'])
    
    for particle in particles:
        particle_data = data[data['particle'] == particle]
        x = particle_data['x'].tolist()
        y = particle_data['y'].tolist()
        
        
        # calcul precision uptake en fonction de la taille des particules 
        grid_precision = 3
        range_radius = np.mean(particle_data['size'])*range_radius_Sm
        
        min_x, max_x = min(x), max(x)
        min_y, max_y = min(y), max(y)
        
        taille_x_grid = int((max_x - min_x) * grid_precision) + 1
        taille_y_grid = int((max_y - min_y) * grid_precision) + 1
        
        grid = np.ones((taille_y_grid, taille_x_grid), dtype=bool)
        grid = grid.astype(int) # transformation boolean into 0 and 1
        
        for pos_x, pos_y in zip(x, y):
            grid_x = int((pos_x - min_x) * grid_precision)
            grid_y = int((max_y - pos_y) * grid_precision)
            
            # Mark the cells within the specified range
            for radius_x in range(-range_radius, range_radius):
                for radius_y in range(-range_radius, range_radius):
                    
                    pos_x = grid_x + radius_x
                    pos_y = grid_y + radius_y
                    
                    if 0 <= pos_x < taille_x_grid and 0 <= pos_y < taille_y_grid:
                        # grid shape = grid[y,x]
                        grid[pos_y, pos_x] = False
        
        # count number of sqaure reached
        U = np.count_nonzero(grid == 0) # number of 0
        uptakes[particle] = U
    
        print("\n","\n")
        print(f"Equivalent matrice particule {particle} :")
        print(grid)
        print(f"Nombre de case atteintes particule {particle} : ",U, f" sur {max_frames} frames.")
        
        
        if plot_uptake:
            plt.close("all")
            plt.imshow(grid, cmap='gray', origin='upper', extent=[min_x, max_x, min_y, max_y], alpha=0.5)
                
            plt.plot(x, y, linestyle = '-', color = 'black')
        
            plt.title('Uptake')
            plt.xlabel('x (mm)')
            plt.ylabel('y (mm)')
            plt.gca().set_aspect('equal')
            plt.tight_layout()
            plt.savefig(output_figures + "uptakes_all_particles/" + f'uptake_{particle}.png', dpi=300, pad_inches=0.1)
            plt.show()
            
    return uptakes, particles

plot_uptake = True
range_radius_Sm = 1

uptakes, particles = calc_uptake(data_tot, plot_uptake, range_radius_Sm)
print(uptakes)
uptakes_list = list(tortuosities.values())
max_U = max(uptakes_list)
print("max_U = ", max_U)
uptakes_list = uptakes_list/max_U


plt.close("all")

plt.hist(uptakes_list,bins=len(particles)*5)

plt.savefig(output_figures + 'distrib_U.png', dpi=300, pad_inches=0.1)
plt.show()



med_U_Sm = np.median(uptakes_list)
max_frames = max(data_tot['frame'])
print("médiane uptake :", med_U_Sm, f"sur {max_frames} frames")
