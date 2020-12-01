# -*- coding: utf-8 -*-
"""
Created on Fri Mar  6 10:55:13 2020

@author: Yifan Wang
"""

"""
Initial optimal structures
"""

import os
import sys
import pandas as pd
import numpy as np
import time

import platform
HomePath = os.path.expanduser('~')
ProjectPath = os.path.join(HomePath, 'Documents', 'GitHub', 'Pdn-Dynamics-Model')

if platform.system() == 'Linux':
    ProjectPath = '/work/ccei_biomass/users/wangyf/cluster_project/CE_opt'


# CO binding model directory
binding_path = os.path.join(ProjectPath, 'CO-adsorption', 'binding', 'v1')
# Energy model directory
energy_path = os.path.join(ProjectPath, 'Cluster-Expansion', 'v11_annealing')

# LASSO model directory
selected_batches = [0, 1, 2, 3]
lasso_model_name = 'lasso' + '_' + ''.join(str(i) for i in selected_batches)
lasso_path = os.path.join(energy_path, lasso_model_name)
lasso_file = os.path.join(lasso_path, lasso_model_name + '.p')


sys.path.append(binding_path)
sys.path.append(energy_path)

import energy_functions as energy
import binding_functions as binding

from generate_clusters_super_cell import super_mother
from set_ce_lattice import dz

from ase.visualize import view

local_view_flag = False

import GCMC_functions as GCMC

'''
Load energy object
'''
Pdn = energy.Pdn(lasso_file, mother = super_mother, super_cell_flag = local_view_flag)

def create_single_atom_structure(npd, seed = 0):
    """Create initial structure with nPd atoms on the support, n <= 25 
    """
    # set random seed
    np.random.seed(seed)
    # Create an random SAC based individual
    # the index for base layer atoms in super cell
    pd_base_indices = np.where(super_mother[:,2] ==  dz)[0]
    pd_atom_separate_indices = [500, 502, 504, 551, 553, 510, 512, 514, 561, 563, 520, 522, 524, 571, 573, 530, 532, 534, 581, 583, 540, 542, 544, 591, 593]
    pd_sa_separate_indices = sorted([pd_base_indices[i-500] for i in pd_atom_separate_indices])

    pd_base_occ_indices = np.unique(np.random.choice(pd_sa_separate_indices, npd, replace = False))
    
    # Initialize the Pd individual configuration in one hot encoding
    pdx = np.zeros(len(super_mother),dtype = int)
    pdx[pd_base_occ_indices] = 1
    
    return list(pdx)

def view_config(pdn_config, view_flag = True):
    """
    View the configuration in ASE object
    """
    pdn_atoms, _ = energy.append_support(pdn_config, super_mother, view_flag = local_view_flag)
    if view_flag: view(pdn_atoms)
    
    return pdn_atoms
    
def view_pdx(pdx, view_flag = True):
    """
    View the configuration in ASE object
    """
    pdn_config = energy.one_hot_to_index(pdx)
    pdn_atoms, _ = energy.append_support(pdn_config, super_mother, view_flag = local_view_flag)
    if view_flag: view(pdn_atoms)
    
    return pdn_atoms




#%%
    
npdx = 20
T_GA = 300  # Ensemble temperature
filename_hof = os.path.join(os.getcwd(), 'GA_hofs', 'ga_hall_of_fame_' + str(npdx) + '_' + str(T_GA) + 'k' + '.csv')
hof_inds, hof_fitnesses= GCMC.read_history(filename_hof)


counter = 0

#%%

# set an energy cutoff to select nhof
Ecutoff = 0.5 # eV

# the index of energy column corresponding to size
Eindices = {5: 3,
            6: 3, 
            7: 3, 
            8: 3,
            9: 3,
            10: 3,
            11: 3,
            12: 3,
            13: 3,
            14: 3,
            15: 3,
            16: 2,
            17: 2,
            18: 2,
            19: 2,
            20: 2,
            21: 2}

# set the maximum number of hof
nhof = 10

# select the qualified individuals with energy within the cutoff
qualified = list(np.where(hof_fitnesses[:, Eindices[npdx]] < hof_fitnesses[0, Eindices[npdx]] + Ecutoff)[0])
nqualified = len(qualified)


#%%
# Configurations that only contains single clusters 
hof_inds_single = []
while counter < nhof and counter < nqualified:
    
    hof_index = qualified[counter]
    
    pdx_i = hof_inds[hof_index]
    ncluster_i = GCMC.count_clusters(pdx_i)
    if ncluster_i == 1:
        hof_inds_single.append(pdx_i)
    counter += 1
    
hof_inds_config = [energy.one_hot_to_index(ci) for ci in  hof_inds_single]
    

OutputPath = os.path.join(os.getcwd(), 'initial structures')
if not os.path.exists(OutputPath): os.makedirs(OutputPath)
SizePath = os.path.join(OutputPath, 'pd' + str(npdx))
if not os.path.exists(SizePath): os.makedirs(SizePath)

pdx_name_list = []
for i in range(len(hof_inds_config)):
    pdx_name_list.append(str(npdx) +'_i'+ str(i) )

# save the configurations as POV

for name_i, config_i in zip(pdx_name_list, hof_inds_config):
    atoms_i = view_config(config_i, view_flag = False)
    binding.save_POV(Pdi = name_i, index = 0, atoms = atoms_i, output_dir= SizePath)


     
    
     