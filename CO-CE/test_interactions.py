"""
Test CO lateral interaction correction OLS model
with 5 additional structures
"""

'''
Import librarys
'''
import os
import sys
import time
import platform

import pandas as pd
import numpy as np  
import json
import matplotlib.pyplot as plt
import matplotlib

from sklearn.metrics import mean_squared_error, r2_score
from ase.io import read, write
from ase.data import covalent_radii

HomePath = os.path.expanduser('~')
ProjectPath = os.path.join(HomePath, 'Documents', 'GitHub', 'Pdn-CO-Stability')

if platform.system() == 'Linux':
    ProjectPath = '/work/ccei_biomass/users/wangyf/cluster_project/CE_opt'

# CO binding model directory
binding_path = os.path.join(ProjectPath, 'CO-CE')
# Energy model directory
energy_path = os.path.join(ProjectPath, 'Pdn-CE')


sys.path.append(binding_path)
sys.path.append(energy_path)

import energy_functions as energy
import binding_functions as binding

# Set plotting resolution
dpi = 300.

font = {'size'   : 20}

matplotlib.rc('font', **font)
matplotlib.rcParams['axes.linewidth'] = 1.5
matplotlib.rcParams['xtick.major.size'] = 8
matplotlib.rcParams['xtick.major.width'] = 2
matplotlib.rcParams['ytick.major.size'] = 8
matplotlib.rcParams['ytick.major.width'] = 2

# set unit length for CO lattice
Pdr =  covalent_radii[46]
unit_length = 2*Pdr 


#%% user define functions

def predict_config_Ebind_total(atoms_object, CO_sites_occ):
    """predict total CO binding energy by summing up all binding energy given occupied site list
    
    :param atoms_object: atoms object
    :type atoms_object: atoms object
    :param CO_sites_occ: occupied CO site indices in CO sites
    :type CO_sites_occ: list
    :return: CO_configs, CO_pos, sitetype_list, config_Ebind_total
    :rtype: CO configuration list. CO position list, site type list, total binding energy for each configuration
    """
    # Predict CO binding energy given all sites
    y_pred, COsites, CO_pos, sitetype_list =  binding.predict_binding_Es_fast(atoms_object, view_flag = False, output_descriptor= False)
    
    # Find the corresponding binding energies
    CO_configs = [] # occupied CO config indices in CO sites 
    config_Ebinds = [] # binding energy list
    
    for config_i in CO_sites_occ:
        Ebinds = []
        CO_config = []
        for site_i in config_i:
            CO_config_i = COsites.index(site_i)
            Ebinds.append(y_pred[CO_config_i])
            CO_config.append(CO_config_i)
        CO_configs.append(CO_config)
        config_Ebinds.append(Ebinds)
        
        
    # Sum Ebind for each configuration
    config_Ebind_total = np.array([sum(Ebinds) for Ebinds in config_Ebinds])
    
    return CO_configs, CO_pos, sitetype_list, config_Ebind_total


def read_Ebind_real(config_Ebind_total, CO_ads_csv):
    """read the real binding energy from a csv file
    
    :param config_Ebind_total: the list contain predicted binding energy
    :type config_Ebind_total:list
    :param CO_ads_csv: co adsorption csv file path
    :type CO_ads_csv: str
    :return: real co binding energy list
    :rtype: list
    """
    # m - number of configurations
    # Read the csv file containing m DFT binding energies
    n_config = len(config_Ebind_total)                   
                   
    COm_data = pd.read_csv(CO_ads_csv)
    COm_Ebind_total = np.array(COm_data['Eads_total'])[0:n_config]
    
    return COm_Ebind_total


def cal_correlation_mat(CO_sites_occ, CO_pos, sitetype_list, interactions_list):
    """calculate the correlation matrix by counting the number of interactions
    """
    # initialize the matrix
    count_matrix = np.zeros((len(CO_sites_occ), len(interactions_list)))
    # iterate through the configuration
    for i, ci in enumerate(CO_sites_occ):
        interactions = binding.CO_interactions(CO_pos, sitetype_list, ci, interactions_list) # create an object
        count_matrix[i,:] = interactions.count_interactions() # count the interaction for this configuration
    
    return count_matrix
    
    

    
#%%
# Define CO sites for configurations in a 2D list
pd9_CO_sites = [[[501, 506],[502, 507], [500, 507], [506, 508], [507, 508], [505, 508]]] # 6CO


pd13_CO_sites = [[[506], [507], [502], [501, 510], [500, 509], [503], [505, 510, 512]], #7CO
                 [[506], [507], [502], [501, 510], [500, 509], [503], [508, 511], [511, 512], [509, 510, 512], [505, 510]]] #10CO

pd15_CO_sites = [[[508, 514], [505, 511], [502, 509], [501, 510], [501, 510], [503, 512], [500], [507, 514], [506, 513]], #8CO
                  [[508, 514], [505, 511], [502, 509], [501, 510], [501, 510], [503, 512], [500], [507, 514], [511, 513], [506, 513], [509]]] #11CO 

# Set atoms object and Ebinding csv file path
# pd9_atoms = read(os.path.join(os.path.dirname(__file__), 'pd9-no-CO.CONTCAR'))
# pd13_atoms = read(os.path.join(os.path.dirname(__file__), 'pd13-no-CO.CONTCAR'))
# pd15_atoms = read(os.path.join(os.path.dirname(__file__), 'pd15-no-CO.CONTCAR'))

# pd9_CO_ads_csv = os.path.join(os.path.dirname(__file__), 'adsorption_constant_Pd9_mCO.csv')
# pd13_CO_ads_csv = os.path.join(os.path.dirname(__file__), 'adsorption_constant_Pd13_mCO.csv')
# pd15_CO_ads_csv = os.path.join(os.path.dirname(__file__), 'adsorption_constant_Pd15_mCO.csv')

bare_cluster_path = os.path.join(ProjectPath, 'dataset', 'DFT_structures', 'bare')
interaction_path =  os.path.join(ProjectPath, 'dataset', 'interactions')
pd9_atoms = read(os.path.join(bare_cluster_path, 'pd9-no-CO.CONTCAR'))
pd13_atoms = read(os.path.join(bare_cluster_path, 'pd13-no-CO.CONTCAR'))
pd15_atoms = read(os.path.join(bare_cluster_path, 'pd15-no-CO.CONTCAR'))

pd9_CO_ads_csv = os.path.join(interaction_path, 'adsorption_constant_Pd9_mCO.csv')
pd13_CO_ads_csv = os.path.join(interaction_path, 'adsorption_constant_Pd13_mCO.csv')
pd15_CO_ads_csv = os.path.join(interaction_path, 'adsorption_constant_Pd15_mCO.csv')

# Predict overall CO binding energy without interactions
pd9_CO_configs, pd9_CO_pos, pd9_sitetype_list, pd9_Ebinds_total_pred = predict_config_Ebind_total(pd9_atoms, pd9_CO_sites)
pd13_CO_configs, pd13_CO_pos, pd13_sitetype_list, pd13_Ebinds_total_pred = predict_config_Ebind_total(pd13_atoms, pd13_CO_sites)
pd15_CO_configs, pd15_CO_pos, pd15_sitetype_list, pd15_Ebinds_total_pred = predict_config_Ebind_total(pd15_atoms, pd15_CO_sites)
# combine into one vector
Ebinds_total_pred = np.concatenate((pd9_Ebinds_total_pred, pd13_Ebinds_total_pred, pd15_Ebinds_total_pred))


# Get the real total binding energy
pd9_Ebinds_total_real = read_Ebind_real(pd9_Ebinds_total_pred, pd9_CO_ads_csv)
pd13_Ebinds_total_real = read_Ebind_real(pd13_Ebinds_total_pred, pd13_CO_ads_csv)
pd15_Ebinds_total_real = read_Ebind_real(pd15_Ebinds_total_pred, pd15_CO_ads_csv)
# combine into one vector
Ebinds_total_real = np.concatenate((pd9_Ebinds_total_real, pd13_Ebinds_total_real, pd15_Ebinds_total_real))

#%% Compute the interactions using trained interaction model
pd9_interactions = binding.CO_interactions(pd9_CO_pos, pd9_sitetype_list, pd9_CO_configs[0])
pd9_total_interactions = pd9_interactions.cal_interactions()
pd9_Ebinds_total_pred_w_interaction = pd9_total_interactions + pd9_Ebinds_total_pred[0]

pd13_interactions = binding.CO_interactions(pd13_CO_pos, pd13_sitetype_list, pd13_CO_configs[0])
pd13_total_interactions = pd13_interactions.cal_interactions()
pd13_Ebinds_total_pred_w_interaction_0 = pd13_total_interactions + pd13_Ebinds_total_pred[0]

pd13_interactions = binding.CO_interactions(pd13_CO_pos, pd13_sitetype_list, pd13_CO_configs[1])
pd13_total_interactions = pd13_interactions.cal_interactions()
pd13_Ebinds_total_pred_w_interaction_1 = pd13_total_interactions + pd13_Ebinds_total_pred[1]

pd15_interactions = binding.CO_interactions(pd15_CO_pos, pd15_sitetype_list, pd15_CO_configs[0])
pd15_total_interactions = pd15_interactions.cal_interactions()
pd15_Ebinds_total_pred_w_interaction_0 = pd15_total_interactions + pd15_Ebinds_total_pred[0]

pd15_interactions = binding.CO_interactions(pd15_CO_pos, pd15_sitetype_list, pd15_CO_configs[1])
pd15_total_interactions = pd15_interactions.cal_interactions()
pd15_Ebinds_total_pred_w_interaction_1 = pd15_total_interactions + pd15_Ebinds_total_pred[1]

# combine into one vector
Ebinds_total_pred_w_interactions = np.array([pd9_Ebinds_total_pred_w_interaction,
                                             pd13_Ebinds_total_pred_w_interaction_0,
                                             pd13_Ebinds_total_pred_w_interaction_1,
                                             pd15_Ebinds_total_pred_w_interaction_0,
                                             pd15_Ebinds_total_pred_w_interaction_1])


#%% Evaluate the model performance

# Try calculate the total Ebind by adding the interaction to the model
all_sites = pd9_CO_sites + pd13_CO_sites + pd15_CO_sites
n_all_sites = np.array([len(si) for si in all_sites])

# Performance metrics before correction
testing_before_r2 = r2_score(Ebinds_total_real, Ebinds_total_pred)
testing_before_rmse =  np.sqrt(mean_squared_error(Ebinds_total_real, Ebinds_total_pred))
print('The total E r2 is {} before correction'.format(testing_before_r2))
print('The total E RMSE is {} before correction'.format(testing_before_rmse))
# normalize based on the number of adsorbates 
normalized_rmse =  np.sqrt(mean_squared_error(Ebinds_total_pred/n_all_sites, Ebinds_total_real/n_all_sites))
print('The normalzied E RMSE is {} before correction'.format(normalized_rmse))

# Performance metrics after correction
testing_after_r2 = r2_score(Ebinds_total_real, Ebinds_total_pred_w_interactions)
testing_after_rmse =  np.sqrt(mean_squared_error(Ebinds_total_real, Ebinds_total_pred_w_interactions))
print('The total E r2 is {} after correction'.format(testing_after_r2))
print('The total E RMSE is {} after correction'.format(testing_after_rmse))

# normalize based on the number of adsorbates 
normalized_rmse =  np.sqrt(mean_squared_error(Ebinds_total_pred_w_interactions/n_all_sites, Ebinds_total_real/n_all_sites))
print('The normalzied E RMSE is {} after correction'.format(normalized_rmse))

# Parity plots
# set plotting limits
lims = [-25, -7]
# before interaction correction parity
fig, ax = plt.subplots(figsize = (6,6))
ax.scatter(Ebinds_total_real, Ebinds_total_pred, s = 100, alpha = 0.6)
# lims = [
#         np.min([ax.get_xlim(), ax.get_ylim()]),  # min of both axes
#         np.max([ax.get_xlim(), ax.get_ylim()]),  # max of both axes
#     ]
ticks = np.around(np.arange(lims[0], lims[1]+1, 4), decimals = 0)
ax.plot(lims, lims, 'k--', alpha=0.75, zorder=0)
ax.set_xlim(lims)
ax.set_ylim(lims)
ax.set_xticks(ticks)
ax.set_yticks(ticks)
ax.set_xlabel(r"$\rm DFT\ E^{mCO-ads}_{\ Pd_{n-i}/CeO_2}\ (eV)$")
ax.set_ylabel(r"$\rm RF\ Predicted\ E^{mCO-ads}_{\ Pd_{n-i}/CeO_2}\ (eV)$")

# after interaction correction parity
fig, ax = plt.subplots(figsize = (6,6))
ax.scatter(Ebinds_total_real, Ebinds_total_pred_w_interactions, s = 100, alpha = 0.6)
ticks = np.around(np.arange(lims[0], lims[1]+1, 4), decimals = 0)
ax.plot(lims, lims, 'k--', alpha=0.75, zorder=0)
ax.set_xlim(lims)
ax.set_ylim(lims)
ax.set_xticks(ticks)
ax.set_yticks(ticks)
ax.set_xlabel(r"$\rm DFT\ E^{mCO-ads}_{\ Pd_{n-i}/CeO_2}\ (eV)$")
ax.set_ylabel(r"$\rm RF\ Predicted\ E^{mCO-ads}_{\ Pd_{n-i}/CeO_2}\ (eV)$")

