### Modèle permettant la construction de trajectoires auto évitantes en 2 dimensions et l'analyse quantitatives de ces trajectoires.

Auteur : Renaud SERRE 
Date création : 15.04.2024

L'objectif est de comparer ce modèle général ([nom model]) à un cas biologique d'acquisition de ressource en 2 dimensions chez Scapholeberis mucronata, un zooplancton se nourrissant de la matière organique à l'interface entre l'air et l'eau. 2 autres scripts permettent respectivement l'obtention des trajectoires expérimentales ([nom script]) et l'analyse des paramètres quantitatifs des trajectoires et des paramètres de nages instantanés de S. mucronata.

Les 3 scripts peuvent s'exécuter en local (paramètre connection_cluster = False) ou sur un serveur (connection_cluster = True). Dans ce cas merci de préciser les chemins d'accès pour l'obtention des figures et des données. Un paramètre "print_console" permet d'afficher ou non les données de vérification dans le terminal.

Les packages nécessaire pour l'exécution du code sont : 
- matplotlib
- numpy
- pandas
- trackpy
- pims 

Si besoin installez les dans le terminal avec  : pip install [nom package]

Dans le cas d'une exécution sur un serveur, l'exécution du code se fait sous la forme : python3.11 nom_fichier.py et ajouter "&" à la fin pour une exécution hors ligne.

### [model]

La construction du générateur de trajectoire se compose des fonctions : 

1. sample_param : tire les 4 paramètres de saut : angle alpha (moyenne var_alpha, distribution absolue exponentielle), longueur l (moyenne l_mean et variance l_var, distribution log-normale), temps de pause Tp (moyenne Tp_mean et variance Tp_var, distribution log-normale), et Ts = l/vitesse 
2. create_tracks_sa  pour la construction d'une trajectoire avec m saut en fonction d'une proba d'auto évitement proba_sa. 
3. plot_graph_traj pour l'affichage du graphique
4. time discretisation pour discretiser en fonction d'un pas de temps dt les trajectoires théoriques

L'affichage des distributions des 4 paramètres de nage : 
1. calc_densite_expo, calc_densite_lognormal et calc_densite_lognormal_Th pour retrouver les fit des distributions tirés 
2. print_graphs pour l'affichage des distributions des paramètres de nage

L'analyse des paramètres quantitatifs se fait avec les fonctions : 
=> MSD : - fit_powerlaw : copie d'une fonction de trackpy pour effectuer une régression linéaire en dimensions log-log et retrouver les paramètres de la distribution de la loi de puissance MSD = A.t^n
	 - calc_MSD : calcule le MSD pour une trajectoire
	 - plot_MSD_trajectory : calcule le MSD pour un nombre donné d'itération en vu d'obtenir des valeurs cohérentes 	pour l'affichage sur les cartes de chaleur
=> Tortuosité : - calc_tortuosity : calcule la tortuosité des trajectoires
=> Uptake (taux acquisition des ressource) : - uptake : calcule l'uptake en fonction d'un paramètre grid_precision qui représente la division de chaque axe x et y pour représenter le facteur d'échelle (correspond à la taille des individus en analyse expérimentale) et d'un paramètre range_radius qui représente la capacité d'un individu à acquérir des ressources lointaines qui représente un facteur multiplicatif du paramètre grid_precision.

=>=> Cartes de chaleurs : heatmap_MSD_Tortuosity_Uptake : cette fonction permet de retourner n'importe quel valeur des 3 paramètres quantitatifs pour l'ensemble des combinaisons entre une probabilité d'auto évitement et une persistance directionnelle (var_alpha). Des paramètres permettent de changer les bornes de précisions de ces combinaisons.


Un test de convergence permet d'utiliser des valeurs significatives d'itération et de sauts pour chaque combinaisons. 

Enfin, des fonction permettent de créer une animation en format .mp4 grâce à un fichier ffmpeg préalablement installé et correctement configuré.

A la toute fin du programme vous pouvez changer l'ensembles des paramètres permettant de construire les trajectoires et de les analyser

### []

Pour l'analyse des réelles trajectoires avec les script [] et [] merci de préciser le nom du dossier contenant les photos en format .tiff 

Les différents paramètres de suivi des particules sont affiné pour correspondre à S. mucronata : 
diameter = 9 
minmass = 1 
noise_size = 1 
smoothing_size = 11 
separation = 15

memory = 0 
max_deplacement = 80 

### []

Dans un premier temps on sépare les particules en fonction de leurs taille (adulte/juvéniles) selon un seuil arbitraire mass_ind_heavy = 3.8. 
On peut maintenant retrouver le MSD pour chaque particules (fonction emsd de trackpy), leurs vitesses instantannés, l'angle de rotation instantannés (dépendant de la fréquence/s d'aquisition d'images), la tortuosité et l'uptake des réelles trajectoires trajectoires des individus)
