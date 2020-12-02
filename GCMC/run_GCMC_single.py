#  -*- coding: utf-8 -*-
"""
Created on Fri Mar 13 15:35:01 2020

@author: Yifan Wang
"""

"""
GCMC main function using class
Compare Pd1 and Pd20
"""

import GCMC_functions as GCMC
from initial_structures import pd1, pd2, pd3, pd4_2d, pd4_3d, pd5_3d, pd5_2d, pd6_2d, pd20s, pd20, pd19_stable,  pd20_1layer, pd13_1layer
from ase.visualize import view
import binding_functions as binding
import time

pdx = pd6_2d
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