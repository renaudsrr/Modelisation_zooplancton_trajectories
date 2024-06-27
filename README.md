# Model for Building Self-Avoiding Trajectories in 2D and Quantitative Analysis of These Trajectories

Author: Renaud SERRE

Creation Date: 15.04.2024
Last Modified: 27/06/2024

The goal is to compare this general model (model_generate_and_analysis_trajectories.py) to a biological case of resource acquisition in 2D for Scapholeberis mucronata, a zooplankton that feeds on organic matter at the air-water interface. Two other scripts respectively allow obtaining experimental trajectories (image_analysis_processing.py) and analyzing the quantitative parameters of the trajectories and the swimming parameters of S. mucronata (analysis_trajectories.py).

The three scripts can run locally (parameter connection_cluster = False) or on a server (connection_cluster = True). In this case, please specify the paths for obtaining figures and data. A "print_console" parameter allows displaying or not verification data in the terminal.

The packages needed to run the code are:
- matplotlib
- numpy
- pandas
- trackpy
- pims 

If needed, install them in the terminal with: *pip install <package name>*
For server execution, run the code as: *python3.11 <filename.py>* and add "<&>" at the end for offline execution.

### Model_generate_and_analysis_trajectories.py

The construction of the **trajectory generator** includes the functions:
1. *sample_param* : draws the 4 jump parameters: angle alpha (mean var_alpha, absolute exponential distribution), length l (mean l_mean and variance l_var, log-normal distribution), pause time Tp (mean Tp_mean and variance Tp_var, log-normal distribution), and Ts = l/speed
2. *create_tracks_sa* : builds a trajectory with m jumps based on a self-avoidance probability proba_sa.
3. *plot_graph_traj* : displays the graphs
4. *time_discretisation* : discretizes the theoretical trajectories based on a time step dt

Displaying the **distributions** of the 4 swimming parameters:
1. *calc_densite_expo*, calc_densite_lognormal et calc_densite_lognormal_Th : fits of the drawn distributions
2. *print_graphs* : displays the distributions of the swimming parameters

Analysis of the **3 quantitative parameters** is done with the functions:
=> **MSD** : 
- *fit_powerlaw* : a copy of a trackpy function to perform a linear regression in log-log dimensions and retrieve the distribution parameters of the power law MSD = A.t^n
- *calc_MSD* : calculates the MSD for a trajectory
- *plot_MSD_trajectory* : calculates the MSD for a given number of iterations to obtain coherent values for heatmap display
  
=> **Tortuosity** : 
- *calc_tortuosity* : calculates the tortuosity of the trajectories
  
=> **Uptake** (resource acquisition rate) : 
- *uptake* : calculates uptake based on a grid_precision parameter representing the division of each x and y axis to represent the scale factor (corresponding to the size of individuals in experimental analysis) and a range_radius parameter representing an individual's ability to acquire distant resources, which is a multiplicative factor of the grid_precision parameter.

=>=> Heatmaps : 
- *heatmap_MSD_Tortuosity_Uptake* : this function returns any value of the 3 quantitative parameters for all combinations between a self-avoidance probability and a directional persistence (var_alpha). Parameters allow changing the precision bounds of these combinations.

A convergence test allows using significant iteration and jump values for each combination. 
Finally, functions create an animation in .mp4 format with a pre-installed and properly configured ffmpeg file. 
At the very end of the program, you can change all the parameters to build and analyze the trajectories.

### Image_analysis_processing.py

For analyzing real trajectories with the scripts [] and [], please specify the folder name containing the photos in .tiff format.

The different particle tracking parameters are refined to correspond to S. mucronata:
var_file = 100 # movement variance of individuals => if very low, little movement so it is not an individual
diameter = 9 # particle diameter
minmass = 1 # minimum mass in terms of brightness => sum of px values
noise_size = 1 # 1 by default. Gaussian blurring kernel => filters signal impurities, higher values filter larger signals
smoothing_size = 11 # diameter by default, removes large-scale biases (e.g., luminosity gradient)
separation = 15 # distance below which we group the same particle

memory = 0 # number of frames to keep an individual in memory if it disappears
max_deplacement = 80 # maximum number of pixels between each movement

### Analysis_trajectories.py

First, we separate particles based on their size (adult/juvenile) according to an arbitrary threshold mass_ind_heavy = 3.8. We can now retrieve the MSD for each particle (trackpy's emsd function), their instantaneous speeds, instantaneous rotation angles (depending on the image acquisition frequency/s), tortuosity, and uptake of the real trajectories of the individuals.
