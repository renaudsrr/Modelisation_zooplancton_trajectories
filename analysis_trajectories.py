# -*- coding: utf-8 -*-
"""
Script complementary of script analyse_img and script_model for deduct some swimming parameters of Scapholeberis mucronata.

Author: Renaud Serre
Creation: 30.04.2024


History of modifications
30.04.2024 : - creation of the script
02.05.2024 : 

Infos Trackpy at https://soft-matter.github.io/trackpy/dev/tutorial/walkthrough.html

"""
# =============================================================================
# Packages
# =============================================================================
from __future__ import division, unicode_literals, print_function  # for compatibility with Python 2 and 3

import sys

import math as mt
import matplotlib as mpl
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

# connection cluster à distance
connection_cluster = False

file = "240516_expe_2"

if not connection_cluster:
    data_tot = pd.read_csv('D:/CODE_stage/fichiers/infos_particles_'+ file +".csv")
    output_trajectories = "D:/CODE_stage/fichiers/"
    output_figures = "D:/CODE_stage/figures/"
else:
    data_tot = pd.read_csv("/home/reserre/output_fichiers/" + file +".csv")
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
im = tp.imsd(data_heavy, micron_par_pixels , frames_par_sec)
em = tp.emsd(data_heavy, micron_par_pixels, frames_par_sec)

fig, ax = plt.subplots()
ax.plot(im.index, im, 'k-', alpha=0.1)
ax.set(ylabel=r'distance (mm²)',
       xlabel='lag time $t$ (s)')
ax.set_xscale('log')
ax.set_yscale('log')
param_MSD = tp.utils.fit_powerlaw(em,color="black",plot=False)    
n_MSD = float(param_MSD['n'])
A_MSD = float(param_MSD['A'])
n_round = str(round(n_MSD,2))
A_round = round(A_MSD,4)
ax.text(0.05, 0.01, fr'$y = {A_round} \cdot x^{{{n_round}}}$', color='black', fontsize=15)
tp.utils.fit_powerlaw(em,color="black",plot=True)   

fig.savefig(output_figures + 'msd_heavy.png', dpi=300, pad_inches=0.1)

# light
im = tp.imsd(data_light, micron_par_pixels , frames_par_sec)
em = tp.emsd(data_light, micron_par_pixels, frames_par_sec)

fig, ax = plt.subplots()
ax.plot(im.index, im, 'k-', alpha=0.1)
ax.set(ylabel=r'distance (mm²)',
       xlabel='lag time $t$ (s)')
ax.set_xscale('log')
ax.set_yscale('log')
param_MSD = tp.utils.fit_powerlaw(em,color="black",plot=False)    
n_MSD = float(param_MSD['n'])
A_MSD = float(param_MSD['A'])
n_round = str(round(n_MSD,2))
A_round = round(A_MSD,4)
ax.text(0.05, 0.01, fr'$y = {A_round} \cdot x^{{{n_round}}}$', color='black', fontsize=15)
tp.utils.fit_powerlaw(em,color="black",plot=True)   

fig.savefig(output_figures + 'msd_light.png', dpi=300, pad_inches=0.1)

# all
im = tp.imsd(data_tot, micron_par_pixels , frames_par_sec)
em = tp.emsd(data_tot, micron_par_pixels, frames_par_sec)

fig, ax = plt.subplots()
ax.plot(im.index, im, 'k-', alpha=0.1)
ax.set(ylabel=r'distance (mm²)',
       xlabel='lag time $t$ (s)')
ax.set_xscale('log')
ax.set_yscale('log')
param_MSD = tp.utils.fit_powerlaw(em,color="black",plot=False)    
n_MSD = float(param_MSD['n'])
A_MSD = float(param_MSD['A'])
n_round = str(round(n_MSD,2))
A_round = round(A_MSD,4)
ax.text(0.05, 0.01, fr'$y = {A_round} \cdot x^{{{n_round}}}$', color='black', fontsize=15)
tp.utils.fit_powerlaw(em,color="black",plot=True)   

fig.savefig(output_figures + 'msd_tot.png', dpi=300, pad_inches=0.1)



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
velocities_heavy = calculate_velocities(data_heavy, micron_par_pixels, frames_par_sec)
velocities_light = calculate_velocities(data_light, micron_par_pixels, frames_par_sec)

# =============================================================================
# Definir jump
# =============================================================================



# =============================================================================
# Calculation of alpha and l
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


# =============================================================================
# all param + export
# =============================================================================

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

def plot_distrib_l_alpha(data, data_name, by_particule):
    
    angles_per_particle = calc_angles(data)
    length_per_particle = calc_l(data)

    nb_particules = len(np.unique(data['particle']))
    indice = 0
    
    # Distributions totales
    all_length = [length for liste_length in length_per_particle.values() for length in liste_length]
    all_angles = [angle for liste_angles in angles_per_particle.values() for angle in liste_angles]
    all_angles_abs = [abs(x) for x in all_angles]
    
    opac = 0.7 # définition de l'opacité
    bins = 300 # nbr barres
    
    if by_particule:
        fig, axs = plt.subplots(nb_particules, 4, figsize=(15, 6*nb_particules))  # Création de la figure avec plusieurs sous-graphiques
        
        particles = np.unique(data["particle"]).tolist()    
    
        for particle in particles:
            data_particle = data[data['particle'] == particle]
            
            velocities_particle = calculate_velocities(data_particle, micron_par_pixels, frames_par_sec)
            
            ax = axs[indice, 0]
            sc = ax.scatter(data_particle['x'], data_particle['y'], c=velocities_particle, cmap='viridis')
            # plt.plot(data_particle['x'], data_particle['y'],color="black",alpha=0.2)
            plt.colorbar(sc, label='Velocity (mm/s)')
            ax.set_xlabel('X position (mm)')
            ax.set_ylabel('Y position (mm)')
            # ax.set_title(f'velocities particle : {particle}  ({data_name})')
            
            indice += 1
    
        indice = 0    
    
        # Distributions d'angles
        for particle, liste_angles in angles_per_particle.items():
            print(particle, len(liste_angles))
            
            ax = axs[indice, 1]
            liste_angles = angles_per_particle[particle]
            
            # ax.set_title(f'Distribution particule : {particle} ({data_name})')
            ax.set_xlabel('radians')
            ax.set_ylabel('density')
            ax.hist(liste_angles, density=True, alpha=opac, color="red", label="Alpha Distribution",bins=bins)
            
            indice += 1
        
        indice = 0
        
        # Distributions de longueurs
        for particle, liste_length in length_per_particle.items():
            print(particle, len(liste_length))
            
            ax = axs[indice, 2]  
            liste_length = length_per_particle[particle]
            
            ax.set_title(f'Distribution particule : {particle} ({data_name})')
            ax.set_xlabel('mm')
            ax.set_ylabel('density')
            ax.hist(liste_length, density=True, alpha=opac, color="blue", label="Length Distribution",bins=bins)
            
            indice += 1
            

    # distribution totale 
    if by_particule:
        ### alpha ----------------------------------------------------------------------------
        var_alpha = 1/np.mean(all_angles_abs)
        
        x_values = np.linspace(0, max(all_angles), 1000)
        y_values = densite_expo(x_values, var_alpha)
        
        ax_alpha_total = axs[0, 3] 
        ax_alpha_total.set_title(f'Distribution totale d\'angles ({data_name})')
        ax_alpha_total.set_xlabel('radians')
        ax_alpha_total.set_ylabel('density')
        ax_alpha_total.hist(all_angles, density=True, alpha=opac+0.15, color="red", label="Alpha Distribution",bins=bins)
        ax_alpha_total.plot(x_values/2, y_values, color='red', label='Densité exponentielle ajustée')
        ax_alpha_total.plot(-x_values/2, y_values, color='red')
        plt.legend()
        
        ### l -------------------------------------------------------------------------------
        l_mean = np.mean(all_length)
        l_var = np.var(all_length)
        
        x_values = np.linspace(0, max(all_length), 1000)
        y_values = calc_densite_lognormal(x_values,l_mean,l_var)
        
        ax_length_total = axs[1, 3]
        ax_length_total.set_title(f'Distribution totale de longueurs ({data_name})')
        ax_length_total.set_xlabel('mm')
        ax_length_total.set_ylabel('density')
        ax_length_total.hist(all_length, density=True, alpha=opac+0.15, color="blue", label="Length Distribution",bins=bins*2)
        ax_length_total.plot(x_values, y_values, color='blue', label='Densité log-normale ajustée')
        ax_length_total.set_xlim([0, 3])
        plt.legend()
        
        plt.savefig(output_figures + 'param_particles_separates.png', dpi=300, pad_inches=0.1)

        
    else:
        ### alpha ----------------------------------------------------------------------------
        var_alpha = 1/np.mean(all_angles_abs)
        print(var_alpha)
        
        x_values = np.linspace(0, max(all_angles), 1000)
        y_values = densite_expo(x_values, var_alpha)
        
        plt.subplot(1,2,1)
        # plt.title(f'Distrib alpha ({data_name})')
        plt.xlabel('radians')
        plt.ylabel('density')
        plt.hist(all_angles, density=True, alpha=opac-0.15, color="grey", label="Alpha Distribution",bins=bins)
        plt.plot(x_values, y_values/2, color='grey', label='Densité exponentielle ajustée')
        plt.plot(-x_values, y_values/2, color='grey')
        
        var_alpha_round = round(var_alpha,2)
        plt.text(-2.5,0.6, f'β = {var_alpha_round}',
                 color='black', fontsize=15)
        #plt.legend()
        plt.tight_layout()

        ### l ------------------------------------------------------------------------------
        l_mean = np.mean(all_length)
        l_var = np.var(all_length)
        print(l_mean,l_var)
        
        x_values = np.linspace(0, max(all_length), 1000)
        y_values = calc_densite_lognormal(x_values,l_mean,l_var)
        
        plt.subplot(1,2,2)
        # plt.title(f'Distrib l ({data_name})')
        plt.xlabel('mm')
        plt.ylabel('density')
        plt.hist(all_length, density=True, alpha=opac-0.15, color="grey", label="Length Distribution",bins=bins*3)
        plt.plot(x_values, y_values, color='grey', label='Densité log-normale ajustée')
        plt.xlim([0, 3])
        # plt.legend()
        plt.tight_layout()
        
        l_mean_round = round(l_mean,2)
        l_var_round = round(l_var,2)

        plt.text(0.5,9, f'σ = {l_var_round}\nµ={l_mean_round}',
                 color='black', fontsize=15)
        
        plt.savefig(output_figures + 'param_all_particles.png', dpi=300, pad_inches=0.1)
    
    plt.tight_layout()
    plt.show()


by_particule = False
plot_distrib_l_alpha(data_tot, "all", by_particule)
plot_distrib_l_alpha(data_light, "data light",by_particule) 
plot_distrib_l_alpha(data_heavy, "data heavy", by_particule)




# =============================================================================
# Calculation of Tp
# =============================================================================



# =============================================================================
# Calculation of Th
# =============================================================================







































