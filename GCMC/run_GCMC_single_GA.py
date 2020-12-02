# -*- coding: utf-8 -*-
"""
Created on Mon Sep 14 10:50:08 2020

@author: yifan
"""


"""
Run GCMC single on just one GA structure
"""

#%% Import session

import os
import sys
import pandas as pd
import numpy as np
import time
import pickle

import platform
HomePath = os.path.expanduser('~')
ProjectPath = os.path.join(HomePath, 'Documents', 'GitHub', 'Pdn-CO-Stability')

if platform.system() == 'Linux':
    ProjectPath = '/work/ccei_biomass/users/wangyf/cluster_project/CE_opt'
    

from mpi4py import MPI
import GCMC_functions as GCMC
from initial_structures import *
from ase.visualize import view

# CO binding model directory
binding_path = os.path.join(ProjectPath, 'CO-CE')
sys.path.append(binding_path)
import binding_functions as binding

start_time = time.time()

# MPI setup
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

global_progress_flag = False
global_plot_flag = True


#%% User define functions

def read_history(filename):
    '''
    Read the history type csv output files
    return the population in a list  
    return the fitnesses in a numpy array
    '''
    # Convert to a dataframe
    df = pd.read_csv(filename, index_col=False)
    
    # Convert the pandas value to one hot eqncoding
    population = []
    for indi in df['individual']:
        population.append([int(i) for i in indi.strip('][').split(', ')])
    
    # Save the fitness value into a 2D array
    fitnesses = np.column_stack((df['fitness1'], df['fitness2'], df['fitness3'], df['fitness4'], df['fitness5']))
    
    return population, fitnesses


def run_pdx_equilibirum(T, PCO, pdx, pdx_name):
    """
    run the simulation to reach the equilirium
    """
    # Initialize the Pdx GCMC simulation trajectory
    traj = GCMC.TrajectoryRejectionFree(T, PCO, pdx, batch_name = pdx_name, progress_flag = global_progress_flag)
    traj.set_subdirectories()
    if len(pdx) > 1:
        traj.nsteps = 500
    else: 
        traj.nsteps = 100
    traj.run()
    
    

def analyze_pdx_equilibrium(T, PCO, pdx, pdx_name):
    """
    Analysis the simulation results at equilirium
    """
    # Initialize the Pdx GCMC simulation trajectory
    traj = GCMC.TrajectoryRejectionFree(T, PCO, pdx,  batch_name = pdx_name, progress_flag = global_progress_flag)
    traj.set_subdirectories()
    # Get the final/equilibrium status
    nCO, mu = traj.analysis(plot_flag = global_plot_flag)
    
    return nCO, mu

def get_pickle_path(npdx, InputPath = os.path.abspath(os.getcwd()), ResultsName = 'resultsRejectionFree'):
    '''set path for pickle file'''
    name = 'pd' + str(npdx)
    
    # level 1 directory contains all simulation results
    ResultPath = os.path.join(InputPath, ResultsName)
    if not os.path.exists(ResultPath): os.makedirs(ResultPath)
    
    # level 2 directory contains results for specific size
    SizePath = os.path.join(ResultPath, name)
    if not os.path.exists(SizePath): os.makedirs(SizePath)
    
    return SizePath
        
        
#%% Simulation specifications
npdx= 6
T_GA = 300  # Ensemble temperature
filename_hof = os.path.join(os.getcwd(), 'GA_hofs', 'ga_hall_of_fame_' + str(npdx) + '_' + str(T_GA) + 'k' + '.csv')
hof_inds, hof_fitnesses= read_history(filename_hof)

counter = 0


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
            21: 2,
            25: 2,
            30: 2,
            38: 2,
            55: 2}

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

# pd_config = energy.one_hot_to_index(hof_inds_single[0])
# GCMC.view_config(pd_config)


# Select the best individuals from hof for GCMC
pdx_sim = hof_inds_single

nsim = len(pdx_sim) # the best individuals selected from hall of frame

pdx_name = 'pd' + str(npdx) # folder name


#%%
index = 3
pdx = pdx_sim[index]
pdx_name = 'pdx'

OutputPathName = 'results'

start_time = time.time()
#%%
# Reactionc conditions
T = 300 # Temperature of the simulation in K
PCO = 1e-1 # Partial pressure of CO in bar
global_progress_flag = True
global_plot_flag = True


# Generate the Pd1 GCMC simulation trajectory
pdx_traj = GCMC.TrajectoryRejectionFree(T, PCO, pdx, pdx_name, progress_flag = global_progress_flag)
pdx_traj.set_subdirectories()
pdx_traj.rseed = 1
pdx_traj.nsteps = 1000
pdx_traj.run()


#%%
# Get the final/equilibrium status
pdx_nCO, pdx_mu = pdx_traj.analysis(plot_flag = global_plot_flag)
view(pdx_traj.PdnCOm_atoms)


end_time = time.time()
all_time = (end_time - start_time)/60

print(pdx_mu)
stable_config_info = pdx_traj.analyze_sites()
