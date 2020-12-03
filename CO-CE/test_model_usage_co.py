# -*- coding: utf-8 -*-
"""
Created on Thu Dec  3 10:15:44 2020

@author: Yifan Wang
"""

'''
Test for CO-CE model predictions
'''

#%% Import session

import os
import sys
import platform
from ase.io import read, write

# Set the path
HomePath = os.path.expanduser('~')
ProjectPath = os.path.join(HomePath, 'Documents', 'GitHub', 'Pdn-CO-Stability')

if platform.system() == 'Linux':
    ProjectPath = '/work/ccei_biomass/users/wangyf/cluster_project/CE_opt'

# Energy model directory 
energy_path = os.path.join(ProjectPath, 'Pdn-CE')
model_path = os.path.join(ProjectPath,'CO-CE')
sys.path.append(energy_path)
sys.path.append(model_path)

import binding_functions as binding

#%% 
'''
Predict single CO adsorption energies and geometric descriptors for a Pd20 structure
'''
# Import a CONTCAR file from DFT
CONTCAR_filename = 'pd20-no-CO-CONTCAR'
bare_cluster_path = os.path.join(ProjectPath, 'dataset', 'DFT_structures', 'bare')
pd20_atoms = read(os.path.join(bare_cluster_path, CONTCAR_filename))
#the Pd atoms involved in binding
Pd_interest = [107, 96, 98, 111, 106, 99, 110, 113, 112, 115] 

# Use one line function
binding_Es, COsites, CO_pos, sitetype_list, GCNs, CN1s, CN2s, ratio_surface =  binding.predict_binding_Es_fast(pd20_atoms, 
                                                                                                           Pd_interest , 
                                                                                                           view_flag = False, 
                                                                                                           output_descriptor= True,
                                                                                                           top_only= False)

# # Longer version of the code
# COsites = binding.find_sites(Pd_interest, pd20_atoms)
# #specific the ML model
# spca = binding.pca_model('spca')
# spca.predict_binding_E(pd20_atoms, COsites)
# y_bind = spca.y_bind
# y_pred = y_bind.copy()
# sitetype_list = spca.sitetype_list

'''
Assume some COs are occupied, predict the interactions and total energy for the adlayer
'''
co_config = [0, 1, 2, 36, 35]
# Use the interactions class
interactions = binding.CO_interactions(CO_pos, sitetype_list, co_config)
# calculate the lateral interactions
total_interactions = interactions.cal_interactions()
# calculate the total adlayer enegry
total_E = interactions.cal_binding_Es_total(binding_Es)


'''
Visualize atoms
'''
PdnCOm_obj = binding.PdnCOm(pd20_atoms, Pd_interest)
PdnCOm_atoms = PdnCOm_obj.append_COs(co_config, view_flag = True)

    
    
