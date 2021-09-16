"""
Train CO literal interaction correction OLS model
Note that each run may result in slight changes 
in the interaction values 
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
    COm_Ebind_total = np.array(COm_data['Eads_total'])[1:n_config+1]
    
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
# Pd4 sites
pd4_CO_sites = [[[97, 98, 99], [96, 98]],
                   [[96],[99], [97, 98]],
                   [[97, 99], [96, 98],[96, 97], [98,99]],
                   [[96], [97, 99], [96, 98],[96, 97], [98, 99]]]


# Pd7 sites
# pd7_CO_sites = [[[99, 102], [101, 102]], #2CO
#                     [[99, 102], [101, 102], [96, 98, 99]], #3CO
#                     [[99, 102], [101, 102], [96, 98, 99], [100, 102]], #4CO
#                     [[99, 102], [101, 102], [96, 98, 99], [100, 102], [96, 100]], #5CO
#                     [[97, 99, 102], [101, 102], [98, 99], [100, 102], [96, 100], [102]], #6CO
#                     [[100, 102], [101, 102], [96, 99], [97, 99], [96, 100], [102], [98]], #7CO
#                     [[97, 99], [101, 102], [96, 98, 99], [100, 102], [96, 100], [102], [98], [99]]] #8CO


pd7_CO_sites = [[[99, 102], [101, 102]], #2CO
                    [[99, 102], [101, 102], [96, 98, 99]], #3CO
                    [[99, 102], [101, 102], [96, 98, 99], [100, 102]], #4CO
                    [[99, 102], [101, 102], [96, 98, 99], [100, 102], [96, 100]], #5CO
                    [[97, 99, 102], [101, 102], [98, 99], [100, 102], [96, 100], [102]], #6CO
                    [[100, 102], [101, 102], [96, 99], [97, 99], [96, 100], [102], [98]], #7CO
                    [[97, 99], [101, 102], [96, 98, 99], [100, 102], [96, 100], [102], [98], [99]], #8CO
                    [[97, 102], [101, 102], [98, 99], [100, 101], [96, 100], [102], [99], [99], [100]], #9CO
                    [[101, 102], [96, 98, 99], [100, 101], [96, 100], [102], [99], [99], [96], [97], [98]]] #10CO

pd20_CO_sites = [[[106, 107], [106, 113], [102, 103], [108, 114]], #4CO
                   [[106, 107], [106, 113], [102, 103], [108, 114], [108, 109]], #5CO
                   [[106, 107], [106, 113], [103, 113, 114], [102, 103], [108, 114], [108, 109]], #6CO
                   [[106, 107], [106, 113], [103, 113, 114],  [102, 103], [108, 114], [108, 109], [114, 115]], #7CO
                   [[106, 107], [106, 113], [100, 107], [113, 115], [102, 103], [108, 114], [108, 109], [114, 115]], #8CO  
                   [[106, 107], [106, 113], [100, 107], [103, 113, 114], [115], [102, 103], [108, 114], [108, 109], [114, 115]], #9CO  
                   [[106, 107], [106, 113], [100], [103, 113], [115], [112, 113], [102, 103], [108, 114], [108, 109], [114, 115]]] #10CO 

# Set atoms object and Ebinding csv file path
bare_cluster_path = os.path.join(ProjectPath, 'dataset', 'DFT_structures', 'bare')
interaction_path =  os.path.join(ProjectPath, 'dataset', 'interactions')
pd4_atoms = read(os.path.join(bare_cluster_path, 'pd4-no-CO-CONTCAR'))
pd7_atoms = read(os.path.join(bare_cluster_path, 'pd7-no-CO-CONTCAR'))
pd20_atoms = read(os.path.join(bare_cluster_path, 'pd20-no-CO-CONTCAR'))

pd4_CO_ads_csv = os.path.join(interaction_path, 'adsorption_constant_Pd4_mCO.csv')
pd7_CO_ads_csv = os.path.join(interaction_path, 'adsorption_constant_Pd7_mCO.csv')
pd20_CO_ads_csv = os.path.join(interaction_path, 'adsorption_constant_Pd20_mCO.csv')


# Predict overall CO binding energy
pd4_CO_configs, pd4_CO_pos, pd4_sitetype_list, pd4_Ebinds_total_pred = predict_config_Ebind_total(pd4_atoms, pd4_CO_sites)
pd7_CO_configs, pd7_CO_pos, pd7_sitetype_list, pd7_Ebinds_total_pred = predict_config_Ebind_total(pd7_atoms, pd7_CO_sites)
pd20_CO_configs, pd20_CO_pos, pd20_sitetype_list, pd20_Ebinds_total_pred = predict_config_Ebind_total(pd20_atoms, pd20_CO_sites)


# Calculate the real total binding energy
pd4_Ebinds_total_real = read_Ebind_real(pd4_Ebinds_total_pred, pd4_CO_ads_csv)
pd7_Ebinds_total_real = read_Ebind_real(pd7_Ebinds_total_pred, pd7_CO_ads_csv)
pd20_Ebinds_total_real = read_Ebind_real(pd20_Ebinds_total_pred, pd20_CO_ads_csv)




#%% determine the coefficient matrix 

# define the interaction list
top_top = {'length': 1 * unit_length,
           'edge_type': [['top', 'top']]} 

hollow_hollow_1 = {'length': np.sqrt(3)/3 * unit_length,
                   'edge_type': [['hollow', 'hollow']]} 

hollow_hollow_2 = {'length': 1 * unit_length,
                   'edge_type': [['hollow', 'hollow']]} 

bridge_bridge_1 = {'length': 1/2 * unit_length,
                 'edge_type': [['bridge', 'bridge']]} 

bridge_bridge_2 = {'length': np.sqrt(3)/2 * unit_length,
                 'edge_type': [['bridge', 'bridge']]} 

bridge_bridge_3 = {'length': 1 * unit_length,
                 'edge_type': [['bridge', 'bridge']]} 

top_bridge_1 = {'length': 1/2 * unit_length,
              'edge_type': [['top', 'bridge'], ['bridge', 'top']]} 

top_bridge_2 = {'length': np.sqrt(3)/2 * unit_length,
              'edge_type': [['top', 'bridge'], ['bridge', 'top']]} 

bridge_hollow_1 = {'length': np.sqrt(3)/ 6 * unit_length,
                   'edge_type': [['bridge', 'hollow'], ['hollow', 'bridge']]} 

bridge_hollow_2 = {'length': np.sqrt(21)/ 6 * unit_length,
                   'edge_type': [['bridge', 'hollow'], ['hollow', 'bridge']]} 

top_hollow = {'length': np.sqrt(3)/ 3 * unit_length,
                   'edge_type': [['top', 'hollow'], ['hollow', 'top']]} 


# a dictionary contains all interactions
interactions = {'interactions':[top_top, hollow_hollow_1, hollow_hollow_2, 
                      bridge_bridge_1, bridge_bridge_2, bridge_bridge_3, 
                      top_bridge_1, top_bridge_2, bridge_hollow_1, bridge_hollow_2, top_hollow]}


# Calculate the correlation (count) matrix
pd4_X = cal_correlation_mat(pd4_CO_configs, pd4_CO_pos, pd4_sitetype_list, interactions['interactions'])
pd7_X = cal_correlation_mat(pd7_CO_configs, pd7_CO_pos, pd7_sitetype_list, interactions['interactions'])
pd20_X = cal_correlation_mat(pd20_CO_configs, pd20_CO_pos, pd20_sitetype_list, interactions['interactions'])


# Concate the correlation matrix X and delta E
X = np.concatenate((pd4_X, pd7_X, pd20_X), axis = 0)
Ebinds_total_pred = np.concatenate((pd4_Ebinds_total_pred, pd7_Ebinds_total_pred, pd20_Ebinds_total_pred))
Ebinds_total_real = np.concatenate((pd4_Ebinds_total_real, pd7_Ebinds_total_real, pd20_Ebinds_total_real))


# Calculate the delta
# delta = real - predicted
# real = delta + predicted
delta_Ebinds = Ebinds_total_real - Ebinds_total_pred

# OLS regression to get the interaction terms
interactions_ev = np.linalg.lstsq(X, delta_Ebinds, rcond = None)[0]

# Create a filter for small values by setting them as 0
interactions_ev[np.where(np.abs(interactions_ev) <=  1e-5)]  = 0

#%% Update the interaction list and save it to a json file

for inter_i, ei in zip(interactions['interactions'], interactions_ev):
    inter_i['E'] = ei
    
with open('co_interactions_new.json', 'w') as outfile:
    json.dump(interactions, outfile, indent = 4)


#%% Access the model performance

# Try calculate the total Ebind by adding the interaction to the model
delta_Ebinds_pred = np.dot(X, interactions_ev)
Ebinds_validate = delta_Ebinds_pred + Ebinds_total_pred

all_sites = pd4_CO_sites + pd7_CO_sites + pd20_CO_sites
n_all_sites = np.array([len(si) for si in all_sites])


# Performance metrics on delta E
delta_r2 = r2_score(delta_Ebinds, delta_Ebinds_pred)
delta_rmse =  np.sqrt(mean_squared_error(delta_Ebinds, delta_Ebinds_pred))
print('The delta E r2 is {}'.format(delta_r2))
print('The delta E RMSE is {}'.format(delta_rmse))


# Parity plot
fig, ax = plt.subplots(figsize = (6,6))
ax.scatter(delta_Ebinds, delta_Ebinds_pred, s = 100, alpha = 0.6)
lims = [
        np.min([ax.get_xlim(), ax.get_ylim()]),  # min of both axes
        np.max([ax.get_xlim(), ax.get_ylim()]),  # max of both axes
    ]
ticks = np.around(np.arange(lims[0], lims[1]+1, 2), decimals = 0)
ax.plot(lims, lims, 'k--', alpha=0.75, zorder=0)
ax.set_xlim(lims)
ax.set_ylim(lims)
ax.set_xticks(ticks)
ax.set_yticks(ticks)
ax.set_xlabel(r"$\rm DFT\ \Delta E\ (eV)$")
ax.set_ylabel(r"$\rm Predicted\ \Delta E\ (eV)$")


# Performance metrics on Ebind total
total_r2_before = r2_score(Ebinds_total_pred, Ebinds_total_real)
total_rmse_before= np.sqrt(mean_squared_error(Ebinds_total_pred, Ebinds_total_real))
print('The total E r2 is {} before correction'.format(total_r2_before))
print('The total E RMSE is {} before correction'.format(total_rmse_before))

normalized_rmse =  np.sqrt(mean_squared_error(Ebinds_total_pred/n_all_sites, Ebinds_total_real/n_all_sites))
print('The normalzied E RMSE is {} after correction'.format(normalized_rmse))


total_r2 = r2_score(Ebinds_validate, Ebinds_total_real)
total_rmse =  np.sqrt(mean_squared_error(Ebinds_validate, Ebinds_total_real))
print('The total E r2 is {} after correction'.format(total_r2))
print('The total E RMSE is {} after correction'.format(total_rmse))

# normalize based on the number of adsorbates 
normalized_rmse =  np.sqrt(mean_squared_error(Ebinds_validate/n_all_sites, Ebinds_total_real/n_all_sites))
print('The normalzied E RMSE is {} after correction'.format(normalized_rmse))


# before interaction correction parity
fig, ax = plt.subplots(figsize = (6,6))
ax.scatter(Ebinds_total_real, Ebinds_total_pred, s = 100, alpha = 0.6)
lims = [
        np.min([ax.get_xlim(), ax.get_ylim()]),  # min of both axes
        np.max([ax.get_xlim(), ax.get_ylim()]),  # max of both axes
    ]
ticks = np.around(np.arange(lims[0], lims[1]+1, 4), decimals = 0)
ax.plot(lims, lims, 'k--', alpha=0.75, zorder=0)
ax.set_xlim(lims)
ax.set_ylim(lims)
ax.set_xticks(ticks)
ax.set_yticks(ticks)
ax.set_xlabel(r"$\rm DFT\ E^{mCO-ads}\ (eV)$")
ax.set_ylabel(r"$\rm RF\ Predicted\ E^{mCO-ads}\ (eV)$")


# after interaction correction parity
fig, ax = plt.subplots(figsize = (6,6))
ax.scatter(Ebinds_total_real, Ebinds_validate, s = 100, alpha = 0.6)
# lims = [
#         np.min([ax.get_xlim(), ax.get_ylim()]),  # min of both axes
#         np.max([ax.get_xlim(), ax.get_ylim()]),  # max of both axes
#     ]
# ticks = np.around(np.arange(lims[0], lims[1]+1, 4), decimals = 0)
# now plot both limits against eachother
ax.plot(lims, lims, 'k--', alpha=0.75, zorder=0)
ax.set_xlim(lims)
ax.set_ylim(lims)
ax.set_xticks(ticks)
ax.set_yticks(ticks)
ax.set_xlabel(r"$\rm DFT\ E^{mCO-ads}_{\ Pd_n/CeO_2}\ (eV)$")
ax.set_ylabel(r"$\rm CE\ Predicted\ E^{mCO-ads}_{\ Pd_n/CeO_2}\ (eV)$")


