# -*- coding: utf-8 -*-
""" Script building a model, from biological data, the trajectory of Scapholeberis mucronata, a moving cladoceran of the Daphniidae family
in a 2-dimensional space at the interface between air and water, by implementing different possible strategies such as self-avoidance.

Author: Renaud Serre
Creation: 15.04.2024


History of modifications
25.04.2024 : - mise à jour cosmétique du code 
             - ajout d'une probabilité d'auto-évitement
             - delete the function create_tracks whithout avoidance
26.04.2024 : - ajout d'une animation video en mp4
02/05/2024 : - ajout d'une fonction permettant de calculer le MSD
             - ajout d'une heatmap représentant les variation de MSD en fonction de la persistance (var_alpha) et de l'autoévitement (proba_sa)
             - ajout du fait que si l'auto-evitement dans create_tracks_sa == 1 et que l'organisme est bloqué trop longtemps alors la trajectoire s'arrête (pour palier au problème d'affichage de la heatmap)
03/05/2024 : - ajout d'une fonction permettant de calculer la tortuositée 
             - ajout de la heatmap de la tortuositée
06/05/2024 : - amélioration et affinage de la totalité du code + en particulier de la fonction heatmap
17/05/2024 : - mise à jour de petites fonctionnalitées + ajout de l'uptake
18/05/2024-27/05/24 : - réglages multiples bug et incohérence dans calc_MSD et Heatmaps 
29/05/2024 : réparation majeure du MSD due à un problème dans time_discretisation l.633
"""

# =============================================================================
# Packages 
# =============================================================================
import warnings
import pprint as pp
import sys

import math as mt
import random as rd
import csv

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import numpy as np
import pandas as pd
from scipy import stats
import trackpy as tp

# =============================================================================
# Working environment 
# =============================================================================

# disable msg warnings
warnings.filterwarnings("ignore")

# cluster remote connection
connection_cluster = False

if not connection_cluster:
    path_output = "D:/CODE_stage/fichiers/positions.csv"
    path_ani = "D:/CODE_stage/animation/animation.mp4"
    output_trajectories = "D:/CODE_stage/fichiers/"
    output_data_heatmap = "D:/CODE_stage/fichiers/heatmap/"
    output_img_heatmap = "D:/CODE_stage/figures/"
    output_img = output_img_heatmap

else : 
    path_output = "/home/reserre/output_fichiers/positions.csv"
    path_ani = "/home/reserre/output_figures/animation.mp4"
    output_trajectories = "/home/reserre/output_fichiers/"
    output_data_heatmap = "/home/reserre/output_fichiers/"
    output_img_heatmap = "/home/reserre/output_figures/"
    
# =============================================================================
# Parameters
# =============================================================================

# time parameters
t = []
t.append(0)
dt = 0.2 # step of time (fraction of one hop)

m = 26 # number of hops
proba_sa = 0.92 # probability self-avoidance

# initial state
x0 = 0 
y0 = 0

# distribution parameters
vit = 3 # swimming speed in mm/s

var_alpha = 1.2 # parameter for distribution of angles, corresponding to mean value of the associated exponential distribution

l_mean = 1 # mean length of a hop for distribution hop lengths, in mm
l_var = 6 # variance of the hop distribution, in mm
l_fix = True # fixation length: if true, the length of a hop is fixed with value l_mean. If false, a lognormal distribution of mean l_mean and variance var is used instead.

Tp_mean = 1  # mean duration of a pause, in s. Used in lognormal distribution of pause time
Tp_var = 0.05 # variance of a pause duration, in s2. Used in lognormal distribution of pause time

# graphic settings
lim_graph = 10 # if you need it, please modify the code
nb_traj = 1 # number of trajectory on a graph, stay at 1
draw_pauses = False # just an aesthetic parameter

# video settings for animation
fps_video = 10

# Calculation of MSD
micron_par_pixels = 1000 # = 1 mm since here the unit of measure of trajectories is mm
frames_par_sec = 1/dt # number of frames for calculation

# parameters of uptake
plot_uptake = True # display all graph of uptake
grid_precision = 8 # precision of the matrix representing the trajectory

# parameter of heatmap : see below

# Choice of exectution part
graph_distrib = False
graph_trajectory = False
all_trajectories_plot = False
plot_MSD = True
plot_heatmaps = False
convergence_test = False
create_animation = False

# =============================================================================
# Parameters sampling
# =============================================================================

def sample_params(vit,var_alpha,l_mean,l_var,Tp_mean,Tp_var,l_fix,nb_tir=1):
    '''
    
    Parameters
    ----------
    vit : integer or float
        swimming speed.
        
    var_alpha : integer or float
        parameter lamda for exponential distribution of angles .
        
    l_mean : integer or float
        parameter mu for normal distribution of lengths.
    l_var : integer or float
        parameter sigma² for normal distribution of lengths.
        
    Tp_mean : integer or float
        parameters mu for normal distribution of break times.
    Tp_var : integer or float
        parameter sigma² for normal distribution of break times.
        
    l_fix : boolean, optional
        fixation lenght to l_mean. The default is True.
        
    nb_tir : TYPE, optional
        desired number of jumps. The default is 1.

    Returns
    -------
    alpha, l, Tp, Th

    '''
    
    ## alpha ------------------------------------------------------------------
    
    # draw
    alpha = np.array([])
    tirage_alpha = [0,0]; probabilites = [0.5, 0.5]
    
    # choice between positive of negative sampling
    for i in range (0,nb_tir):
        tirage_alpha[0] = (np.random.exponential(scale=var_alpha, size=1))
        tirage_alpha[1] = - tirage_alpha[0]
        choix_alpha = rd.choices(tirage_alpha, weights=probabilites, k=1)
        
        # correction of -pi > angles > pi
        alpha = np.append(alpha,((choix_alpha[0] + mt.pi)%(2*mt.pi))-mt.pi)
    
    ## l ----------------------------------------------------------------------
    
    # calculation of log-normal parameters whith normal parameters
    sigma2_l = mt.log(1 + (l_var/l_mean**2))
    mu_l = mt.log(l_mean) - (1/2)*sigma2_l
    
    # draw
    l = np.array([])
    l = np.random.lognormal(mu_l, np.sqrt(sigma2_l), nb_tir)
    
    # fixation lenght
    if l_fix:
        for i in range(len(l)):
            l[i] = l_mean
    
    ## Tp ---------------------------------------------------------------------
    
    # calculation of log-normal parameters whith normal parameters
    sigma2_Tp = mt.log(1 + (Tp_var/Tp_mean**2))
    mu_Tp = mt.log(Tp_mean) - (1/2)*sigma2_Tp
    
    # draw 
    Tp = np.array([])
    Tp = np.random.lognormal(mu_Tp, np.sqrt(sigma2_Tp), nb_tir)
    # Tp = np.random.exponential(scale=Tp_mean, size=nb_tir)
  
    ## Th ---------------------------------------------------------------------
    
    # calculation
    Th = np.array([0]*nb_tir)
    Th = l/vit
    
    ## Only for creat_tracks to transform arrays in float ---------------------
    if nb_tir == 1:
        alpha = float(alpha[0])
        l = float(l[0])
        Tp =  float(Tp[0])
        Th = float(Th[0])
    
    return(alpha, l, Tp, Th)

# =============================================================================
# Calculation of density curves for all distributions
# =============================================================================

def calc_densite_expo(distrib, param):
    '''
    
    Parameters
    ----------
    distrib : array
        type of distribution.
    param : float
        lambda parameter of exponential distribution.

    Returns
    -------
    x,y.

    '''
    x = np.linspace(0, max(distrib), 1000)
    y = (1 / param) * np.exp(-x/param)    
    
    return(x,y)

def calc_densite_normal(distrib, mu, sigma):
    '''

    Parameters
    ----------
    distrib : array
        type of distribution.
    mu : float
        mean of normal distribution.
    sigma : float
        variance of normal distribution.

    Returns
    -------
    x,y.

    '''
    x = np.linspace(0, max(distrib), 1000)
    y = 1/(sigma * np.sqrt(2 * np.pi)) * np.exp( - (x - mu)**2 / (2 * sigma**2))
    
    return(x,y)

def calc_densite_lognormal(distrib, esp, var):
    '''

    Parameters
    ----------
    distrib : array
        type of distribution.
    esp : float
        mean of normal distribution.
    var : float
        variance of normal distribution.

    Returns
    -------
    x,y.

    '''
    sigma2 = mt.log(1 + (var/esp**2))
    mu = mt.log(esp) - (1/2)*sigma2
    
    x = np.linspace(0, max(distrib), 1000)
    y = (np.exp(-(np.log(x) - mu)**2 / (2 * sigma2)) / (x * np.sqrt(sigma2) * np.sqrt(2 * np.pi)))
    
    return(x,y)

def calc_densite_lognormal_Th(distrib, esp, var, vit):
    '''
    Same as calc_densite_lognormal but only for Th.

    Parameters
    ----------
    distrib : array
        type of distribution.
    esp : float
        mean of normal distribution.
    var : float
        variance of normal distribution.
    vit : float
        swimming speed.

    Returns
    -------
    x,y.

    '''
    sigma2_l = np.log(1 + (var/esp**2))
    mu_l = np.log(esp) - (1/2)*sigma2_l
    
    mu_Th = mu_l - np.log(vit)
    
    x = np.linspace(0, max(distrib), 1000)
    y =  (1 / (np.sqrt(sigma2_l) * x * np.sqrt(2 * np.pi))) * ((np.exp((-1/(2*sigma2_l)) * (np.log(x) - mu_Th )**2))) 
    
    return(x,y)

def print_graphs(vit,var_alpha,l_mean,l_var,Tp_mean,Tp_var,l_fix,nb_tir):
    ''' 
    
    Parameters
    ----------
    vit : integer or float
        swimming speed.
        
    var_alpha : integer or float
        parameter lamda for exponential distribution of angles .
        
    l_mean : integer or float
        parameter mu for normal distribution of lengths.
    l_var : integer or float
        parameter sigma² for normal distribution of lengths.
        
    Tp_mean : integer or float
        parameters mu for normal distribution of break times.
    Tp_var : integer or float
        parameter sigma² for normal distribution of break times.
        
    l_fix : boolean, optional
        fixation lenght to l_mean. The default is True.
        
    nb_tir : TYPE, optional
        desired number of jumps. The default is 1.

    Returns
    -------
    None.

    '''
    
    alpha, l, Tp, Th = sample_params(vit,var_alpha,l_mean,l_var,Tp_mean,Tp_var,l_fix, nb_tir)
        
    opac = 0.55 # def opacity
    bin_hist = 200 # def number of bins of hist
    
    # subplot alpha 
    x1, y1 = calc_densite_expo(alpha,var_alpha)
    plt.subplot(1, 2, 1)
    plt.title(r'Distribution des angles $\alpha$')
    plt.xlabel('radians')
    plt.ylabel('density')
    plt.hist(alpha,density=True,alpha=opac,color="grey",bins=bin_hist, label="Alpha Distribution")
    plt.plot(x1, y1/2, color='grey',label = "loi expo")
    plt.plot(-x1, y1/2, color='grey')
    # plt.legend()
    
    # subplot lenght
    x2, y2 = calc_densite_lognormal(l,l_mean,l_var)
    plt.subplot(1, 2, 2)
    plt.title(r'Distribution des longueurs $l$')
    plt.xlabel('mm')
    plt.ylabel('density')
    plt.hist(l,density=True,alpha=opac,color="grey",bins=bin_hist*3, label = "l distribution")
    plt.plot(x2, y2, color='grey', label = "loi log-normale")
    # plt.legend()
    plt.xlim((0, 5))
    
    plt.savefig(output_img + 'distrib_1.png', dpi=300, pad_inches=0.1)
    plt.tight_layout()

    plt.show()
    
    # plot for Th et Tp
    plt.title('Distribution des Temps de pause et de saut')
    plt.xlabel('sec')
    plt.ylabel('density')
    
    #x3, y3 = calc_densite_expo(Tp,Tp_mean) OR  :
    x3, y3 = calc_densite_lognormal(Tp,Tp_mean,Tp_var)
    x4, y4 = calc_densite_lognormal_Th(Th,l_mean,l_var, vit)
    
    plt.hist(Tp,density=True,alpha=opac,color="grey",bins=bin_hist, label = r"$Tp$")
    plt.hist(Th,density=True,alpha=opac,color="black",bins=bin_hist*9,label = r"$Th$")
    plt.plot(x3, y3, color='grey') # a changer en fonction loi choisie
    plt.plot(x4, y4, color='black')
    plt.xlim((0, 2))
    plt.legend()
    
    plt.savefig(output_img + 'distrib_2.png', dpi=300, pad_inches=0.1)
    plt.tight_layout()

    plt.show()

if graph_distrib:    
    print_graphs(vit,var_alpha,l_mean,l_var,Tp_mean,Tp_var,l_fix,10000)


# =============================================================================
# Create trajectories with self-avoidance ("sa")
# =============================================================================

def create_tracks_sa(m, x0, y0,proba_sa, vit, var_alpha,l_mean,l_var,Tp_mean,Tp_var,l_fix):
    """
    
    This function return a trajectory x,y with a parameter proba_sa which control the self-avoiding of the particle
    All parameters below refined this trajectory
    
    Parameters
    ----------
    m : interer
        number of hops.
    x0 : float
        position x[0].
    y0 : float
        position y[0].
    proba_sa : float
        probability of self-avoidance.
    vit : float
        swimming speed in mm/s
    var_alpha : float
        mean value of exponential distribution giving turning angle, in radians
    l_mean : float
        mean value of hop length, in mm
    l_var : float
        variance of hop length, in mm2
    Tp_mean : float
        mean duration of pause, in s
    Tp_var : float
        variance of pause time, in s2
    l_fix : boolean
        if true, the hop length is fixed and equal to l_mean

    Returns
    -------
    x : array
        all x positions.
    y : array
        all y positions.
    t_end_hop_ind : array
        all hops durations in s
    t_end_pause_ind : array
        all pause durations in s
    """
    
    # intialisation finals lists ----------------------------------------------
    x = [x0]
    y = [y0]
    traj_alpha_i = [rd.uniform(-np.pi,np.pi)] # random between -pi and pi
    t_end_hop_ind = [0]
    t_end_pause_ind = [0]
    
    # dictionnary for intersections
    intersect = {}
    
    for i in range(m):
        
        Jump_ok = False
        segments_coupe = 1

        while not Jump_ok: 
            
            # draw parameters -------------------------------------------------
            alpha,l,Tp,Th = sample_params(vit,var_alpha,l_mean,l_var,Tp_mean,Tp_var,l_fix)
            
            # we define the new trajectory angle in relation to the global reference
            alpha_i = traj_alpha_i[-1] + alpha
            
            # calculation of next position after hop
            x.append(x[-1] + l * mt.cos(alpha_i))
            y.append(y[-1] + l * mt.sin(alpha_i))
                        
            # segment name [i,i+1]
            segment_cree_i = "[" + str(i) + ", " + str(i+1) + "]"   
            # dictionnary for segments [k,k+1]
            segments_traj = {}
            
            # Boucle pour iterer sur chaque segment formés sauf le dernier ----
            for k in range(i-1):
                S_dividende = (x[i] - x[k]) * (y[i+1] - y[i]) - (y[i] - y[k]) * (x[i+1] - x[i])
                S_diviseur = (x[k+1] - x[k]) * (y[i+1] - y[i]) - (y[k+1] - y[k]) * (x[i+1] - x[i])
                S = round(S_dividende/S_diviseur,2)
                
                T_dividende = (x[i] - x[k]) * (y[k+1] - y[k]) - (y[i] - y[k]) * (x[k+1] - x[k])
                T_diviseur = (x[k+1] - x[k]) * (y[i+1] - y[i]) - (y[k+1] - y[k]) * (x[i+1] - x[i])
                T = round(T_dividende/T_diviseur,2)
                
                # segment name [k,k+1]
                segment_traj_k = "[" + str(k) + ", " + str(k+1) + "]" 
                segments_traj[segment_traj_k] = (S, T)
            
            intersect[segment_cree_i] = segments_traj
            
            # Check presence intersections -----------------------------------
            intersection_found = False
            for segment_traj_k, values in segments_traj.items():
                if 0 <= values[0] <= 1 and 0 <= values[1] <= 1:
                    
                    # pp.pprint(intersect)
                    # print("Le segment ", segment_cree_i, "coupe le ", segment_traj_k,"c'est sa",segments_coupe,"-ème intersection")
                    # pour éviter boucle infinie lorsque l'organisme s'enferme quand proba_sa >= 1. On laisse une porte de sortie
                    
                    segments_coupe += 1
                    # # print(segments_coupe)
                    # if segments_coupe >= 6 :
                    #     proba_sa = 0.9
                    
                    # Intersection found, decide whether to avoid or not based on the proba "proba_sa"
                    draw = rd.random()
                    # print('tirage =',draw)
                    # print('proba =',proba_sa)
                    
                    if draw < proba_sa:
                        # Avoid intersection => remove last position
                        x.pop(-1)
                        y.pop(-1)

                        Jump_ok = False
                        intersection_found = True 
                        break
                    
                    else:                                                    
                        Jump_ok = True
                        intersection_found = True
                        
            if intersection_found and Jump_ok:
                # Continue without avoiding the intersection, add last position
                t_end_hop = l / vit
                t_end_pause = Tp
                traj_alpha_i.append(alpha_i)
                t_end_hop_ind.append(t_end_hop)
                t_end_pause_ind.append(t_end_pause)
            
            if not intersection_found:
                # No intersection found, add last position
                t_end_hop = l / vit
                t_end_pause = Tp
                traj_alpha_i.append(alpha_i)
                t_end_hop_ind.append(t_end_hop)
                t_end_pause_ind.append(t_end_pause)
                            
                Jump_ok = True
                
    return x, y, t_end_hop_ind, t_end_pause_ind

# Display trajectories 

def plot_graph_traj(traj_x,traj_y,lim_graph,draw_pauses,nb_traj):
    '''
    
    Parameters
    ----------
    traj_x : Array
        trajectory of x axis
    traj_y : Array
        trajectory of y axis
    lim_graph : integer of float
        limits of graph for -x,x,y,-y.
    draw_pauses : boolean
        display point for pauses.
    nb_traj : integer
        number of trajectories.

    Returns
    -------
    None.

    '''
    
    for i in range(nb_traj):
        
        plt.plot(traj_x, traj_y, linestyle = '-', color = 'black', label='hop')
        plt.scatter(x0,y0,color='red',label='start')

        if draw_pauses:
            plt.scatter(traj_x,traj_y,color='black',label='pauses')

    plt.title('trajectoires')
    plt.xlabel('x (mm)')
    plt.ylabel('y (mm)')
    plt.legend()
    
    # plt.xlim((-lim_graph,lim_graph))
    # plt.ylim((-lim_graph,lim_graph))
    plt.gca().set_aspect('equal')
    
    plt.show()

traj_x,traj_y,t_end_hop_ind,t_end_pause_ind = create_tracks_sa(m, x0, y0,proba_sa, vit,var_alpha,l_mean,l_var,Tp_mean,Tp_var,l_fix)

if graph_trajectory:
    plot_graph_traj(traj_x,traj_y,lim_graph,draw_pauses,nb_traj)


# matrix of some trajectories
if all_trajectories_plot:
    m = 100

    vit = 3 

    l_mean = 1 
    l_var = 6 
    l_fix = True 
    
    Tp_mean = 3  # mean duration of a pause, in s. Used in lognormal distribution of pause time
    Tp_var = 2 # variance of a pause duration, in s2. Used in lognormal distribution of pause time

    
    proba_sa_min = 0
    proba_sa_max = 0.9
    var_alpha_min = 0.01
    var_alpha_moy = 1.2
    var_alpha_max = 20
    alpha = 0.6
    
    traj_x_1,traj_y_1,t_end_hop_ind_1,t_end_pause_ind_1 = create_tracks_sa(m//5, x0, y0,proba_sa_min, vit,var_alpha_min,l_mean,l_var,Tp_mean,Tp_var,l_fix)
    traj_x_2,traj_y_2,t_end_hop_ind_2,t_end_pause_ind_2 = create_tracks_sa(m, x0, y0,proba_sa_min, vit,var_alpha_max,l_mean,l_var,Tp_mean,Tp_var,l_fix)
    traj_x_3,traj_y_3,t_end_hop_ind_3,t_end_pause_ind_3 = create_tracks_sa(m, x0, y0,proba_sa_max, vit,var_alpha_moy,l_mean,l_var,Tp_mean,Tp_var,l_fix)
    traj_x_4,traj_y_4,t_end_hop_ind_4,t_end_pause_ind_4 = create_tracks_sa(m, x0, y0,proba_sa_max, vit,var_alpha_max,l_mean,l_var,Tp_mean,Tp_var,l_fix)
    
    plt.scatter(x0,y0,color='black',label='start',s = 20)
    
    plt.plot(traj_x_1, traj_y_1, linestyle = '-', color = 'red', label=f'Pa = {proba_sa_min}; β = {var_alpha_min}',alpha=alpha)
    plt.plot(traj_x_2, traj_y_2, linestyle = '-', color = 'blue', label=f'Pa = {proba_sa_min}; β = {var_alpha_max}',alpha=alpha)
    plt.plot(traj_x_3, traj_y_3, linestyle = '-', color = 'green', label=f'Pa = {proba_sa_max}; β = {var_alpha_moy}',alpha=alpha)
    plt.plot(traj_x_4, traj_y_4, linestyle = '-', color = 'orange', label=f'Pa = {proba_sa_max}; β = {var_alpha_max}',alpha=alpha)
               
    plt.xlabel('x (mm)')
    plt.ylabel('y (mm)')
    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5),fontsize=8)
    plt.gca().set_aspect('equal')
    
    plt.tight_layout(pad=2.0)
    plt.savefig(output_img + 'all_trajectories.png', dpi=300, pad_inches=0.1)
    plt.show()

# =============================================================================
# Time discretisation
# =============================================================================

def time_discretisation(traj_x,traj_y,t_end_hop_ind,t_end_pause_ind,dt):
    """

    Parameters
    ----------
    traj_x : Array
        trajectory of x axis
    traj_y : Array
        trajectory of y axis
    t_end_hop_ind : Array
        list of all break times
    t_end_pause_ind : Array
        list of all hop times
    dt : float
        Divide one jump into small fraction of length dt.

    Returns
    -------
    traj_x_steps : array
        trajectory of x axis discretized.
    traj_y_steps : array
        trajectory of y axis discretized.
    t : array
        step of time.

    """

    traj_x = np.array(traj_x) 
    traj_y = np.array(traj_y) 
    
    L_k = np.sqrt((traj_x[1:]-traj_x[:-1])**2+(traj_y[1:]-traj_y[:-1])**2)
    
    t_end_pause_ind = np.array(t_end_pause_ind)
    t_end_hop_ind = np.array(t_end_hop_ind)
    thp = t_end_pause_ind + t_end_hop_ind 
    tcum = np.cumsum(thp) 

    t_max = np.amax(tcum) 
    nframes_max = int(np.floor(t_max/dt))+1
    t = np.linspace(0,dt*(nframes_max-1), nframes_max)
    
    t = t[:, np.newaxis] 

    collec_heaviside_x = (
                            ( 
                        traj_x[:-1] + vit*(t-tcum[:-1])*(traj_x[1:]-traj_x[:-1])/L_k
                            )*np.heaviside(tcum[:-1]+L_k/vit-t,0)*np.heaviside(t-tcum[:-1],0)
                        + 
                        traj_x[1:]*np.heaviside(t-(tcum[:-1]+L_k/vit),0)*np.heaviside(tcum[:-1]+L_k/vit+t_end_pause_ind[1:]-t,0)
                        )
    
    traj_x_steps = np.sum(collec_heaviside_x,axis=1)

    collec_heaviside_y = (
                            ( 
                        traj_y[:-1] + vit*(t-tcum[:-1])*(traj_y[1:]-traj_y[:-1])/L_k
                            )*np.heaviside(tcum[:-1]+L_k/vit-t,0)*np.heaviside(t-tcum[:-1],0)
                        + 
                        traj_y[1:]*np.heaviside(t-(tcum[:-1]+L_k/vit),0)*np.heaviside(tcum[:-1]+L_k/vit+t_end_pause_ind[1:]-t,0)
                        )
    
    traj_y_steps = np.sum(collec_heaviside_y,axis=1)

    # plt.figure()
    # plt.plot(traj_x, traj_y, linestyle = '-', color = 'black', label='hop')
    
    # #plt.plot(traj_x_steps, traj_y_steps, linestyle = '--', color = 'green', label='hop')
    
    # plt.scatter(traj_x,traj_y,color='black',label='pauses')
    # plt.scatter(traj_x_steps,traj_y_steps,color='green',label='pauses', alpha = 0.2)
    # plt.gca().set_aspect('equal')
    # plt.show()
    
    return traj_x_steps,traj_y_steps, t

x, y, t = time_discretisation(traj_x,traj_y,t_end_hop_ind,t_end_pause_ind,dt)

# =============================================================================
# Calculation of MSD and tortuosity
# =============================================================================

def fit_powerlaw(data, min_lagtime, plot=True, **kwargs):
    """
    Fit a powerlaw by doing a linear regression in log space.
    
    Parameters
    ----------
    data : pd.DataFrame or pd.Series
        Data to fit. The index should represent the x-values.
    plot : bool, optional
        If True, plots the data and the fit. The default is True.
    **kwargs : Additional keyword arguments for the plotting function.
    
    Returns
    -------
    values : pd.DataFrame
        DataFrame containing the slope (n) and intercept (A) for each column in the input data.
    """
    
    ys = pd.DataFrame(data)
    x = pd.Series(data.index.values, index=data.index, dtype=np.float64)
    values = pd.DataFrame(index=['n', 'A'])
    fits = {}
        
    for col in ys:
        y = ys[col].dropna()
        slope, intercept, r, p, stderr = \
            stats.linregress(np.log(x[min_lagtime:]), np.log(y[min_lagtime:]))
        values[col] = [slope, np.exp(intercept)]
        fits[col] = x.apply(lambda x: np.exp(intercept)*x**slope)
        
    values = values.T
    fits = pd.concat(fits, axis=1)
    
    if plot:
        from trackpy import plots
        plots.fit(data, fits, logx=True, logy=True, legend=False, **kwargs)
        
    return values


def calc_MSD(x_list, y_list, micron_par_pixels, frames_par_sec, display_graph_MSD, var_alpha, proba_sa, m, nb_traj_square, max_lagtime, min_lagtime):
    """
    
    Parameters
    ----------
    x_list : list of multiples array
        one array = calculation of IMSD one one particle and here multiple array (multiples particles)
        make calculation of EMSD possible.
    y_list : list of multiples array
        same.
    micron_par_pixels : float
        for calculation of MSD.
    frames_par_sec : integer
        1/dt.
    display_graph_MSD : boolean
        display graph of MSD if True.
    var_alpha : integer or float
        parameter lamda for exponential distribution of angles .
    proba_sa : float
        probability of self-avoidance.
    m : interer
        number of hops.
    nb_traj_square : integer
        replicates for eachs square.
    

    Returns
    -------
    n_MSD : float
        power-law exponent.
    A_MSD : float
        A = 4D, where D is the particles' diffusivity.

    """
    
    all_data = pd.DataFrame()
    
    for particle_idx, (x, y) in enumerate(zip(x_list, y_list)):
        frame = np.arange(1, len(x) + 1)
        particle = np.full_like(frame, particle_idx)  # Assigning particle index
        
        # tranformation en liste python
        x = x.tolist()
        y = y.tolist()
        
        data_temp = {
            'x': x,
            'y': y,
            'frame': frame,
            'particle': particle
        }

        particle_data = pd.DataFrame(data_temp)
        all_data = pd.concat([all_data, particle_data], ignore_index=True)
    
    # Calculation of MSD
    im = tp.imsd(all_data, micron_par_pixels, frames_par_sec, max_lagtime = max_lagtime)
    em = tp.emsd(all_data, micron_par_pixels, frames_par_sec, max_lagtime = max_lagtime)

    # data.to_csv(output_trajectories + "traj_particles_model.csv", index=False)
    
    param_MSD = fit_powerlaw(em, min_lagtime, plot=display_graph_MSD,color="black")
    
    n_MSD = float(param_MSD['n'])
    A_MSD = float(param_MSD['A'])
    n_round = round(n_MSD,2)
    
    # Desactivate display of figures
    if display_graph_MSD:
        fig, ax = plt.subplots(1, 2)
        ax[0].plot(im.index, im, '-', color="green",alpha=0.1)
        ax[0].set_xscale('log')
        ax[0].set_yscale('log')
        ax[1].set(ylabel=r'MSD (mm²)',
                  xlabel='$t$ (s)')
        # ax[0].set_title("IMSD model")
        # ax[0].legend([f'proba self avoid = {proba_sa}\n β = {var_alpha}\n nb sauts = {m}\n nb particules = {nb_traj_square}'], loc='best',fontsize=8)
        
        ax[1].scatter(em.index, em, color="black",alpha=0.05,s = 6)
        
        # Perform linear regression
        log_em_index = np.log(em.index[min_lagtime:])
        log_em_values = np.log(em.values[min_lagtime:])
        slope, intercept, r_value, p_value, std_err = stats.linregress(log_em_index, log_em_values)
    
        
        ax[1].plot(em.index[min_lagtime:], np.exp(intercept + slope * log_em_index), color='#006400')
        ax[1].set_xscale('log')
        ax[1].set_yscale('log')
        ax[1].set(ylabel=r'MSD (mm²)',
                  xlabel='$t$ (s)')
        # ax[1].set_title("EMSD model")
        
        # plot of the Brownian motion prediction
        ax[1].scatter(em.index, em.index*(micron_par_pixels*l_mean)**2/(Tp_mean+l_mean/vit), color="red",alpha=0.05,s = 6)
        ax[1].plot(em.index[min_lagtime:], em.index[min_lagtime:]*(micron_par_pixels*l_mean)**2/(Tp_mean+l_mean/vit),color='#8B0000')
        # plot of the ballistic straight swimmer prediction
        ax[1].scatter(em.index, em.index**2*(micron_par_pixels*l_mean)**2/(Tp_mean+l_mean/vit), color="blue",alpha=0.05,s = 6)
        ax[1].plot(em.index[min_lagtime:], em.index[min_lagtime:]**2*(micron_par_pixels*l_mean)**2/(Tp_mean+l_mean/vit)**2,color='#000080')
        
        plt.tight_layout()
        
        plt.savefig(output_img + 'I_EMSD.png', dpi=300, pad_inches=0.1)

        plt.show()


    if display_graph_MSD:

        print("n : ", n_MSD)
        print("A : ", A_MSD)
    

    return n_MSD, A_MSD

# Plot MSD

if plot_MSD:
    x_list = []
    y_list = []
    
    nb_traj_square = 50 # number of iteration
    m = 500 # number of hops
    
    vit = 3
    proba_sa = 0.7
    var_alpha = 1.2
    l_mean = 3 
    l_var = 0.5 
    l_fix = True
    Tp_mean = 0.2 
    Tp_var = 0.2
        
    micron_par_pixels = 1000
    dt = 0.2
    frames_par_sec = 1/dt

    
    display_graph_MSD = True
    
    typical_track_duration = m*(l_mean/vit+Tp_mean) # in seconds
    print('typical_track_duration in seconds=', typical_track_duration)

    fraction_of_track = 0.2 # to reduce the influence of low statistics part
    max_lagtime = int(fraction_of_track*typical_track_duration/dt) # max lag time in number of frames for msd calculation.
    print('max lag time in number of frames =',max_lagtime)

    min_hops = 3 # minimum number of ops after which we expect diffusive behavior to appear in the fit
    min_hops_duration = min_hops*(l_mean/vit+Tp_mean) # in seconds
    min_lagtime = int(min_hops_duration/dt)  # min lag time in number of frames for msd calculation.
    print('min lag time in number of frames for fit=',min_lagtime)
    
    # Générer les trajectoires
    for _ in range(nb_traj_square):
        traj_x,traj_y,t_end_hop_ind,t_end_pause_ind = create_tracks_sa(m, x0, y0,proba_sa, vit,var_alpha,l_mean,l_var,Tp_mean,Tp_var,l_fix)
        x, y, t = time_discretisation(traj_x,traj_y,t_end_hop_ind,t_end_pause_ind,dt)
        
        t = np.array(t).flatten()
        
        x_list.append(np.array(x))
        y_list.append(np.array(y))

    n,_ = calc_MSD(x_list, y_list, micron_par_pixels, frames_par_sec, display_graph_MSD, var_alpha, proba_sa, m, nb_traj_square, max_lagtime, min_lagtime)
    print(n)


# =============================================================================
# Calculation of Uptake and Tortuosity
# =============================================================================

def uptake(x, y, t, grid_precision, plot_uptake):
    min_x, max_x = min(x), max(x)
    min_y, max_y = min(y), max(y)
    
    taille_x_grid = int((max_x - min_x) * grid_precision) + 1
    taille_y_grid = int((max_y - min_y) * grid_precision) + 1
    
    # grid[x][y]
    grid = np.ones((taille_y_grid, taille_x_grid), dtype=bool)
    grid = grid.astype(int) # transfo boolean into 0 and 1
    
    for pos_x, pos_y in zip(x, y):
        grid[int((max_y - pos_y) * grid_precision), int((pos_x - min_x) * grid_precision)] = False
        
    
    # faire ratio True/False
    U = np.sum(grid) # number of 1
    
    
    if plot_uptake:
        plt.imshow(grid, cmap='gray', origin='upper', extent=[min_x, max_x, min_y, max_y], alpha=0.5)
        
        for i in range(nb_traj):
            
            plt.plot(traj_x, traj_y, linestyle = '-', color = 'black', label='hop')
            plt.scatter(x0,y0,color='red',label='start')
    
            if draw_pauses:
                plt.scatter(traj_x,traj_y,color='black',label='pauses')
    
        plt.title('trajectoires')
        plt.xlabel('x (mm)')
        plt.ylabel('y (mm)')
        plt.legend()
        plt.gca().set_aspect('equal')
        
        plt.show()
        
    return grid,U

def calc_Tortuosity(x,y):
    """

    Parameters
    ----------
    x : array
        trajectory on x axis.
    y : array
        trajectory on y axis.

    Returns
    -------
    tortuosity : float
        T = 1 for total_length = straight line ; 0 for circle

    """
    # shorter length
    shorter_length = np.sqrt((x[-1] - x[0])**2 + (y[-1] - y[0])**2)
    
    # total length
    # np.diff renvoie un vecteur du type [x2-x1, x3-x2, x4-x3] => même calcul que pour shorter_length mais pour chaque segments
    total_length =  np.sum(np.sqrt(np.diff(x)**2 + np.diff(y)**2))
    
    tortuosity = (total_length / shorter_length)/100
    
    return tortuosity

if graph_trajectory:
    tortuosity = calc_Tortuosity(traj_x, traj_y)
    print("tortuosity = ", tortuosity)
    grid,nb_T = uptake(x,y,t,grid_precision, plot_uptake)
    print(grid)
    print("nb_T = ",nb_T)
    
# =============================================================================
# Heatmap
# =============================================================================

def heatmap_MSD_Tortuosity_Uptake(m_MSD,size_sa, size_alpha, pas_precision_sa,pas_precision_alpha, nb_traj_square, display_tortuosity_and_uptake,plot_uptake,grid_precision):
    """

    Parameters
    ----------
    m_MSD : integer
        number of hop, specific for heatmap.
    size_sa : integer
        ength of matrix on axis y of the heatmap from 0 to 0.9.
    size_alpha : integer
        length of matrix on axis x of the heatmap from 0.1 to inf.
    pas_precision : float
        step on each lines and columns of the matrix.
    nb_traj_square : integer
        replicates for eachs square.
    display_tortuosity_and_uptake : boolean
        if True, display also the heatmap of tortuosity.
    plot_uptake : TYPE
        DESCRIPTION.
    grid_precision : TYPE
        DESCRIPTION.


    Returns
    -------
    all_traj : dictionnary
        all trajectories in a dictionnary like : 
        all_traj = {coord matrice[x,y] : replicat_square} where replicat_square  {index:([array_1],[array_2],...)}

    """
    
    # creation of matrix (x,y) <=> (var_alpha, proba_sa) # -1 car on ne prend pas en compte la dernière valeur
    matrice = np.empty((size_sa, size_alpha), dtype=object)

    # et non pas in range(size_sa+1) car on évite la dernière valeur proba_sa = 1
    x = np.arange(0, size_alpha * pas_precision_alpha, pas_precision_alpha)
    y = np.arange(0, size_sa * pas_precision_sa, pas_precision_sa)
    
    xx, yy = np.meshgrid(x, y)
    
    for i in range(size_sa):
        for j in range(size_alpha):
            matrice[i][j] = [xx[i][j], yy[i][j]]
    # delete first col because of impossibility of var_alpha == 0
    matrice = matrice[:,1:]
    
    matrice_n = np.zeros((size_sa, size_alpha))
    matrice_T = np.zeros((size_sa, size_alpha))
    matrice_U = np.zeros((size_sa, size_alpha))

    matrice_var_T = np.zeros((size_sa, size_alpha))
    matrice_var_U = np.zeros((size_sa, size_alpha))
    
    
    matrice_n = matrice_n[:,1:]
    matrice_T = matrice_T[:,1:]
    matrice_U = matrice_U[:,1:]
    
    matrice_var_T = matrice_var_T[:,1:]
    matrice_var_U = matrice_var_U[:,1:]

    
    # # add a first colum = 0.1 when large heatmap
    # if size_alpha >11:
    #     new_column = np.empty((size_sa, 1), dtype=object)
    #     new_column[0][0] = [0.1,0]
    #     new_column[1][0] = [0.1, 0.32]
    #     new_column[2][0] = [0.1, 0.64]
    #     new_column[3][0] = [0.1, 0.96]
        
    #     matrice = np.hstack((new_column, matrice))
        
    #     matrice_n = np.zeros((size_sa, size_alpha))
    #     matrice_T = np.zeros((size_sa, size_alpha))
    #     matrice_U = np.zeros((size_sa, size_alpha))

    #     matrice_var_T = np.zeros((size_sa, size_alpha+1))
    #     matrice_var_U = np.zeros((size_sa, size_alpha+1))
    
        

    
    dim_n = np.shape(matrice_n)
    dim_T = np.shape(matrice_T)
    dim_U=  np.shape(matrice_U)
    
    # recup all traj x,y et t pour export en csv et traiter l'uptake
    all_traj ={}
    
    all_grid = []
    
    # iteration on each coord (x,y)
    for index_x,ligne in enumerate(matrice):
       for index_y, coordonnees in enumerate(ligne):
           
           x,y = coordonnees
           var_alpha = x 
           proba_sa = y
           # print("index_x : ",index_x,x)
           # print("index_y : ",index_y,y,"\n")
           print("\n","Iteration suivante")
           print("var_alpha : ", round(var_alpha,2), "  (",x,")")
           print("proba_sa : ", round(proba_sa,2), "  (",y,")")
           
           # 2 matrix for all exposant of MSD and all tortuosity
           all_T = []
           all_U = []
           x_list = []
           y_list = []
           
           # dictionnaire pour recuperer chaque trajectoires de chaque replicat
           pos_x = round(x,2)
           pos_y = round(y,2)
           
           nb_square = f"[{pos_x},{pos_y}]"
           replicat_square = {}
           
           # replicate each square for MSD, tortuosity and uptake ----------------------------------------
           for i in range(nb_traj_square):
               
               traj_x,traj_y,t_end_hop_ind,t_end_pause_ind = create_tracks_sa(m_MSD, x0, y0,proba_sa, vit,var_alpha,l_mean,l_var,Tp_mean,Tp_var,l_fix)
               x, y, t = time_discretisation(traj_x,traj_y,t_end_hop_ind,t_end_pause_ind,dt)
               t = np.array(t).flatten()
               grid,U = uptake(x,y,t,grid_precision, plot_uptake)
               

               t = np.array(t).flatten()
            
               x_list.append(np.array(x))
               y_list.append(np.array(y))
               
               if plot_uptake:
                   print(grid)
                   plot_graph_traj(traj_x,traj_y,lim_graph,draw_pauses,nb_traj)

               tortuosity = calc_Tortuosity(traj_x, traj_y)
                              
               all_U.append(U)
               all_T.append(tortuosity)
               
               replicat_square[i+1] = x,y,t
           
           typical_track_duration = m_MSD*(l_mean/vit+Tp_mean) # in seconds
            
           fraction_of_track = 0.2 # to reduce the influence of low statistics part
           max_lagtime = int(fraction_of_track*typical_track_duration/dt) # max lag time in number of frames for msd calculation.
            
           min_hops = 3 # minimum number of ops after which we expect diffusive behavior to appear in the fit
           min_hops_duration = min_hops*(l_mean/vit+Tp_mean) # in seconds
           min_lagtime = int(min_hops_duration/dt)  # min lag time in number of frames for msd calculation.
    
           n,_ = calc_MSD(x_list, y_list, micron_par_pixels, frames_par_sec, False, var_alpha, proba_sa, m, nb_traj_square, max_lagtime, min_lagtime)
           print("n: ",n)
           
           all_traj[nb_square] = replicat_square
           # print(all_traj)

           matrice_n[index_x][index_y] = n 
                    
           T_mean = np.mean(all_T)
           T_var = np.var(all_T)
           matrice_T[index_x][index_y] = T_mean 
           matrice_var_T[index_x][index_y] = T_var
           print("T_mean : ", T_mean)
           
           U_mean = np.mean(all_U)
           U_var = np.var(all_U)
           matrice_U[index_x][index_y] = U_mean
           matrice_var_U[index_x][index_y] = U_var
           
           all_grid.append(U_mean)
           
           print("U_mean : ", U_mean)  
    
    # normalisation uptake
    max_U = max(all_grid)
    matrice_U = matrice_U/max_U
    matrice_var_U = matrice_var_U/max_U
    
    plt.imshow(matrice_n, cmap='Reds', interpolation='nearest')
    plt.colorbar(label='gradient of n',orientation='horizontal',pad=0.2)
    plt.xlabel(r'variation angle ($\beta$)') # = perte persistance 
    plt.ylabel('probability \n self-avoidance')
    

    # Définir les ticks des axes
    x_ticks = np.arange(size_alpha-1)
    y_ticks = np.arange(size_sa)
    x_tick_labels = np.round(np.arange(pas_precision_alpha, size_alpha * pas_precision_alpha, pas_precision_alpha), 2)
    y_tick_labels = np.round(np.arange(0, size_sa * pas_precision_sa, pas_precision_sa), 2)
    
    if size_alpha>11:
        x_tick_labels = [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15]


    plt.xticks(ticks=x_ticks, labels=x_tick_labels)
    plt.yticks(ticks=y_ticks, labels=y_tick_labels)

    # Txt explicatif
    plt.text(-2.5,5, f'Jumps = {m_MSD}\n Iterations = {nb_traj_square}',
             color='black', fontsize=9)


    # save la fig en png
    plt.savefig(output_img_heatmap + 'heatmap_n_MSD.png', dpi=300, pad_inches=0.1)
    
    plt.tight_layout()
    plt.show()
    
    np.save(output_data_heatmap + "matrice_n",matrice_n)
    
    pd.DataFrame(all_traj).to_csv(output_data_heatmap + "all_traj.csv",index=False)
    np.save(output_data_heatmap + "all_traj",all_traj)
    
    np.save(output_data_heatmap + "matrice_T",matrice_T)
    np.save(output_data_heatmap + "matrice_var_T",matrice_var_T)
    
    np.save(output_data_heatmap + "matrice_U",matrice_T)
    np.save(output_data_heatmap + "matrice_var_U",matrice_var_T)
    
    
    if display_tortuosity_and_uptake:

        plt.imshow(matrice_T, cmap='Greens', interpolation='nearest')
        plt.colorbar(label='gradient of tortuosity',orientation='horizontal',pad=0.2)
        plt.xlabel(r'variation angle ($\beta$)') # = perte persistance 
        plt.ylabel('probability \n self-avoidance')
        
        # Définir les ticks des axes
        x_ticks = np.arange(size_alpha - 1)
        y_ticks = np.arange(size_sa)
        x_tick_labels = np.round(np.arange(pas_precision_alpha, size_alpha * pas_precision_alpha, pas_precision_alpha), 2)
        y_tick_labels = np.round(np.arange(0, size_sa * pas_precision_sa, pas_precision_sa), 2)
        
        if size_alpha>11:
            x_tick_labels = [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15]
        
        plt.xticks(ticks=x_ticks, labels=x_tick_labels)
        plt.yticks(ticks=y_ticks, labels=y_tick_labels)

        # Txt explicatif
        plt.text(-2.5,5, f'Jumps = {m_MSD}\n Iterations = {nb_traj_square}',
                 color='black', fontsize=9)

        plt.savefig(output_img_heatmap + 'heatmap_tortuosity.png', dpi=300, pad_inches=0.1)

        plt.tight_layout()
        plt.show()
        
        plt.imshow(matrice_U, cmap='Blues', interpolation='nearest')
        plt.colorbar(label='gradient of uptake (fraction of maximum reached)',orientation='horizontal',pad=0.2)
        plt.xlabel(r'variation angle ($\beta$)') # = perte persistance 
        plt.ylabel('probability \n self-avoidance')
        
        plt.xticks(ticks=x_ticks, labels=x_tick_labels)
        plt.yticks(ticks=y_ticks, labels=y_tick_labels)
        
        # Txt explicatif
        plt.text(-2.5,5, f'Jumps = {m_MSD}\n Iterations = {nb_traj_square}',
             color='black', fontsize=9)
        
        plt.savefig(output_img_heatmap + 'heatmap_Uptake.png', dpi=300, pad_inches=0.1)
        
        plt.tight_layout()
        plt.show()
        
    return all_traj

# Parameters heatmap
display_tortuosity_and_uptake = True

nb_traj_square = 700 # number of iteration
m_MSD = 800 # number of hops

vit = 3
l_mean = 1
l_var = 0.5 
l_fix = True
Tp_mean = 0.6
Tp_var = 0.8
    
micron_par_pixels = 1000
dt = 0.2
frames_par_sec = 1/dt

Param_Heatmap = "large"

if Param_Heatmap == "small":
    size_sa = 4 # dimension proba_sa => do not touch !
    size_alpha = 11 # dimension var_alpha
    pas_precision_sa = 0.31 # incrementation of each values of proba_sa 
    pas_precision_alpha = 0.1 # incrementation of each values of var_alpha

if Param_Heatmap == "large":
    size_sa = 4 # dimension proba_sa => do not touch !
    size_alpha = 16 # dimension var_alpha
    pas_precision_sa = 0.31 # incrementation of each values of proba_sa 
    pas_precision_alpha = 1 # incrementation of each values of var_alpha


# for uptake
plot_uptake = False
grid_precision = 8

if plot_heatmaps:
    all_traj = heatmap_MSD_Tortuosity_Uptake(m_MSD,size_sa, size_alpha, pas_precision_sa,pas_precision_alpha, nb_traj_square, display_tortuosity_and_uptake,plot_uptake,grid_precision)

# =============================================================================
# Test de convergence for estimation parameters m and nb_traj_square
# =============================================================================

if convergence_test:
    # convergence sur un seul square où convergence vers 1

    vit = 3
    proba_sa = 0.0
    var_alpha = 20
    l_mean = 3 
    l_var = 0.5 
    l_fix = True
    Tp_mean = 0.2 
    Tp_var = 0.2
        
    micron_par_pixels = 1000
    dt = 0.2
    frames_par_sec = 1/dt
    
    # gamme de valeurs nombre sauts et nombre traj
    m_values = [100,120]
    nb_traj_square_values = [40,60]
    
    display_graph_MSD = False
    
    convergence = []
    x_list = []
    y_list = []
    

    
    # Boucle sur chaque combinaison de m_MSD et nb_traj_square
    for m_conv in m_values:
        for nb_traj_square_conv in nb_traj_square_values:
            all_n = []
            print(m_conv,nb_traj_square_conv," En cours...")
            
            typical_track_duration = m_conv*(l_mean/vit+Tp_mean) # in seconds
            
            fraction_of_track = 0.2 # to reduce the influence of low statistics part
            max_lagtime = int(fraction_of_track*typical_track_duration/dt) # max lag time in number of frames for msd calculation.

            min_hops = 3 # minimum number of ops after which we expect diffusive behavior to appear in the fit
            min_hops_duration = min_hops*(l_mean/vit+Tp_mean) # in seconds
            min_lagtime = int(min_hops_duration/dt)  # min lag time in number of frames for msd calculation.
            # min_lagtime  = 0       
            print("min :",min_lagtime," track :",typical_track_duration," max :",max_lagtime)
            
            # Générer les trajectoires
            for _ in range(nb_traj_square_conv):
                
                traj_x,traj_y,t_end_hop_ind,t_end_pause_ind = create_tracks_sa(m_conv, x0, y0,proba_sa, vit,var_alpha,l_mean,l_var,Tp_mean,Tp_var,l_fix)
                x, y, t = time_discretisation(traj_x,traj_y,t_end_hop_ind,t_end_pause_ind,dt)
                
                t = np.array(t).flatten()
                
                x_list.append(np.array(x))
                y_list.append(np.array(y))
        
            n,_ = calc_MSD(x_list, y_list, micron_par_pixels, frames_par_sec, display_graph_MSD, var_alpha, proba_sa, m, nb_traj_square_conv, max_lagtime, min_lagtime)
            print(n)
                    
            convergence.append((m_conv, nb_traj_square_conv, m_conv * nb_traj_square_conv, n))
    
    # pour affichage ordre croissant du produit
    convergence = np.array(convergence)
    convergence = convergence[convergence[:, 2].argsort()]
    
    # extraction col
    all_m = convergence[:, 0] # nombre de sauts
    all_N = convergence[:, 1] # nombre de trajectoires
    products = convergence[:, 2] # produit des 2
    n_means = convergence[:, 3] 
    
    # graphs
    plt.subplot(2,1,1)
    for m in m_values:
        subset = convergence[convergence[:, 0] == m]
        plt.plot(subset[:, 1], subset[:, 3], marker='o', label=f'm = {m}')
    
    #plt.plot(all_m,1,color='black')
    plt.xlabel('Nombre de trajectoires')
    plt.ylabel('n')
    plt.title('Convergence de n en fonction du nombre de trajectoires pour différentes valeurs de m')
    plt.tight_layout()
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True)
    
    plt.subplot(2,1,2)
    # plt.plot(all_m,1,color='black')
    plt.plot(products, n_means, marker='o',color='black')
    plt.xlabel('nombre sauts m * nombre de trajectoires')
    plt.ylabel('n')
    plt.title('Convergence de n en fonction du nombre de trajectoires * nombre de sauts m')
    plt.grid(True)
    
    plt.tight_layout() 
    plt.savefig(output_img_heatmap + 'convergence_n.png', dpi=300, pad_inches=0.1)
    
    plt.show()

# =============================================================================
# Video animation
# =============================================================================

plt.rcParams['animation.ffmpeg_path'] = "C:/Users/serre/OneDrive/Bureau/STAGE/CODE/ffmpeg-7.0-essentials_build/ffmpeg-7.0-essentials_build/bin/ffmpeg.exe"

if create_animation and all_trajectories_plot:
    traj_x_1, traj_y_1, t_1 = time_discretisation(traj_x_1,traj_y_1,t_end_hop_ind_1,t_end_pause_ind_1,dt)
    traj_x_2, traj_y_2, t_2 = time_discretisation(traj_x_2,traj_y_2,t_end_hop_ind_2,t_end_pause_ind_2,dt)
    traj_x_3, traj_y_3, t_3 = time_discretisation(traj_x_3,traj_y_3,t_end_hop_ind_3,t_end_pause_ind_3,dt)
    traj_x_4, traj_y_4, t_4 = time_discretisation(traj_x_4,traj_y_4,t_end_hop_ind_4,t_end_pause_ind_4,dt)





def update(frame):
    """

    Parameters
    ----------
    frame : interger
        frame of trajectory discretised.

    Returns
    -------
    None.

    """
    
    plt.cla()
    
    plt.plot(traj_x_1[:frame], traj_y_1[:frame], color='red', label=f'Pa = {proba_sa_min}; β = {var_alpha_min}',alpha=alpha)
    plt.scatter(traj_x_1[frame], traj_y_1[frame], color='red')
    
    plt.plot(traj_x_2[:frame], traj_y_2[:frame], color='blue', label=f'Pa = {proba_sa_min}; β = {var_alpha_max}',alpha=alpha)
    plt.scatter(traj_x_2[frame], traj_y_2[frame], color='blue')
    
    plt.plot(traj_x_3[:frame], traj_y_3[:frame], color='green', label=f'Pa = {proba_sa_max}; β = {var_alpha_moy}',alpha=alpha)
    plt.scatter(traj_x_3[frame], traj_y_3[frame], color='green')
    
    plt.plot(traj_x_4[:frame], traj_y_4[:frame], color='orange')
    plt.scatter(traj_x_4[frame], traj_y_4[frame], color='orange', label=f'Pa = {proba_sa_max}; β = {var_alpha_max}',alpha=alpha)
    
    plt.xlabel('x (mm)')
    plt.ylabel('y( mm)')
    
    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5),fontsize=8)
    plt.gca().set_aspect('equal')
    plt.tight_layout(pad=2.0)

    
    # Calculate the limits based on trajectories 2, 3, and 4
    min_x = min(min(traj_x_2), min(traj_x_3), min(traj_x_4)) - 1
    max_x = max(max(traj_x_2), max(traj_x_3), max(traj_x_4)) + 1
    min_y = min(min(traj_y_2), min(traj_y_3), min(traj_y_4)) - 1
    max_y = max(max(traj_y_2), max(traj_y_3), max(traj_y_4)) + 1
    
    # Setting the limits of the plot
    plt.xlim(min_x, max_x)
    plt.ylim(min_y, max_y)
    
    temps_sec = round(frame * dt,1)
    
    plt.text(0.5, 0.95, f'Time: {temps_sec} (s) ', ha='center', va='top', transform=plt.gca().transAxes)


def create_anim():
    """
    
    Create animation

    Returns
    -------
    None.

    """
    fig = plt.figure()
    ani = FuncAnimation(fig, update, frames=len(t_4), interval=100)
    
    ani.save(path_ani, fps=fps_video, extra_args=['-vcodec', 'libx264'])
    
    plt.show()

if create_animation and all_trajectories_plot:
    create_anim()







def update_one_traj(frame):
    """

    Parameters
    ----------
    frame : interger
        frame of trajectory discretised.

    Returns
    -------
    None.

    """
    
    plt.cla()
    
    plt.plot(x[:frame], y[:frame], color='black')
    plt.scatter(x[frame], y[frame], color='red')
    
    plt.xlim(min(x)-1, max(x)+1)
    plt.ylim(min(y)-1, max(y)+1)
    
    plt.xlabel('x (mm)')
    plt.ylabel('y( mm)')
    
    temps_sec = round(frame * dt,1)
    
    plt.text(0.5, 0.95, f'Time: {temps_sec} (s) ', ha='center', va='top', transform=plt.gca().transAxes)
    
def create_anim_one_traj():
    """
    
    Create animation

    Returns
    -------
    None.

    """
    fig = plt.figure()
    ani = FuncAnimation(fig, update_one_traj, frames=len(t), interval=100)
    
    ani.save(path_ani, fps=fps_video, extra_args=['-vcodec', 'libx264'])
    
    plt.show()

if not all_trajectories_plot and not create_anim:
    create_anim_one_traj()
