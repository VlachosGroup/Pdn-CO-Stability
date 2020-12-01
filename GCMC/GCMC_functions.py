# -*- coding: utf-8 -*-
"""
Created on Fri Mar  6 15:13:03 2020

@author: yifan
"""

"""
Modular GCMC functions
"""

#%% Import session

import os
import sys
import pandas as pd
import numpy as np
import networkx as nx
import time
import platform
from ase.visualize import view
from heapq import heapify, heappop
import collections
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
import lattice_functions as lf 

from generate_clusters_super_cell import super_mother
from set_ce_lattice import dz


# matplotlib settings
import matplotlib
if platform.system() == 'Linux':
    matplotlib.use('Agg')
    print('switched')
import matplotlib.pyplot as plt   
if platform.system() == 'Linux':
    plt.switch_backend('agg')
    
font = {'size'   : 20}

matplotlib.rc('font', **font)
matplotlib.rcParams['axes.linewidth'] = 1.5
matplotlib.rcParams['xtick.major.size'] = 8
matplotlib.rcParams['xtick.major.width'] = 2
matplotlib.rcParams['ytick.major.size'] = 8
dpi = 300.
matplotlib.rcParams['figure.dpi'] = dpi



#%% helper functions
def split_pd_list(pd_column):
    """function to split the input configuration list from dataframe
    take out [] as str and convert number to integers
    
    :param pd_column: a column in dataframe
    :type pd_column: a column in dataframe
    :return: 2d list
    :rtype: 2d list
    """
    pd_list = []
    for indi in pd_column:
        indi_split = indi.strip('][').split(', ')
        try:
            pd_list.append([int(i) for i in indi_split])
            
        except:
            pd_list.append([])
    return pd_list
 

def linear_search(arr, x): 
    """return index of i in a list"""
    for i in range(len(arr)): 
  
        if arr[i][0] == x[0]: 
            return i 
    return -1

def get_k(E, T):
    """calculate propensity given energy"""
    E_copy = np.array(E)
    return np.exp(-E_copy/kb/T)

def get_t(k):
    """calculate random time interval given propensity"""
    if isinstance(k, float): rands = np.random.rand()
    else:
        rands = np.random.rand(len(k)) 
    k_copy = np.array(k)   
    return - 1/k_copy * np.log(rands)


def get_t_heap(T, dmu, site_indices):  
    """generate a heap for time intervals based on dmu"""
    t_heap  = []
    
    for dmu_i, si in zip(dmu, site_indices):
        # propensity for each event
        k = get_k(dmu_i, T)  
        # time for each event
        t = get_t(k)
        # zip time with site indices
        item_i = (t, si)
        t_heap.append(item_i)
            
    return t_heap


def count_clusters(pdx):
    """input pdx and return the number of clusters in a configuration"""
    
    pdx_config = energy.one_hot_to_index(pdx)
    pdConfiguration = Configuration( pdx_config)
    clusters = pdConfiguration.break_clusters()
    n_clusters = len(clusters)
    
    return n_clusters
    
    

def view_config(pdn_config, view_flag = True):
    """
    View the configuration in ASE object
    """
    pdn_atoms, _ = energy.append_support(pdn_config, super_mother, view_flag = view_flag)
    if view_flag: view(pdn_atoms)
    
    return pdn_atoms
    
def view_pdx(pdx, view_flag = True):
    """
    View the configuration in ASE object
    """
    pdn_config = energy.one_hot_to_index(pdx)
    pdn_atoms, _ = energy.append_support(pdn_config, super_mother, view_flag = view_flag)
    if view_flag: view(pdn_atoms)
    
    return pdn_atoms


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



#%%
'''
Load energy object
'''
Pdn = energy.Pdn(lasso_file, mother = super_mother, super_cell_flag = True)    
    
'''
Define constants
'''

# Boltzmann constant in eV/K
kb = 8.617333262145e-05
    
# Distribution factor from Boltzmann distributin
w = 0

# Standard presse in bar
P0 = 1.01325
    
# Referecen: https://janaf.nist.gov/tables/C-093.html
# use delat (H - Hr) at T - (H - Hr)  at 0 K and convert to eV
mu_CO_p0 = -0.52468



class Configuration():
    """
    For each kmc snapshot
    """
    def __init__(self, config, mother = super_mother, dz = dz):
        """Initialize the kmc configuration
        
        :param mother: lattice coordinates
        :type mother: numpy array 
        :param dz: the z distance in lattice
        :type dz: float
        :param config: one hot encoding for kmc configuration
        :type config: list
        """        
        self.mother = mother
        self.config = config
        self.dz = dz

    def initialize_graph_object(self):
        """initialize graph object
        
        :return: lattice as a graph
        :rtype: networkx graph
        """
        '''
        NN1 == 0, draw all nodes
        NN1 == 1, only draw 1st nearest neighbors
        NN1 == 2, connect both 1st nearest neighbors 
        and edges with length smaller than 1NN
        '''
        NN1 = 1
        '''
        Initialize graph object function
        '''
        empty = 'grey' 
        filled = 'r'
        occ = [empty, filled]
        '''
        Draw mother/conifgurations/clusters?
        '''
        draw = [0, 0, 0]
    
        Graphs = lf.graphs(occ, NN1, draw)
        Graphs.dz = self.dz # set z distance
                
        return Graphs

    def build_graph(self):
        """build the kmc lattice graph
        return the graph
        """
        
        graphs = self.initialize_graph_object()
        Graph = graphs.gclusters_kmc(self.mother, self.config)

        return Graph

    def break_clusters(self):
        """Iterate through all atoms break a configuration into all
        """
        # build the lattice graph
        Graph = self.build_graph()

        # 2D lists of cluster indices
        clusters = [list(si) for si in list(nx.connected_components(Graph))]
        
        for i, cluster_i in enumerate(clusters):
            for j, node_j in enumerate(cluster_i):
                clusters[i][j] = self.config[node_j]
        
        # for i, ci in enumerate(self.config):
        #     cluster = []
        #     for j, cj in enumerate(self.config):
        #         if nx.has_path(Graph, i, j): cluster.append(cj)
        #     clusters.append(cluster)
        
        # clusters_unique = []
        # [clusters_unique.append(x) for x in clusters if x not in clusters_unique]
        
        return clusters

    
    
class Simulation():
    """
    Main class for GCMC simulation
    """
    
    def __init__(self, T, PCO, OutputPath):
        """Set constants and default simulation conditions"""
        
        # set output path
        self.OutputPath = OutputPath

        # Temperature of the simulation in K
        self.T = T
        
        # Partial pressure of CO in bar
        self.PCO = PCO
        
        # Chemical potential of CO at T
        self.mu_CO = mu_CO_p0 + kb * self.T * np.log(self.PCO/P0)
        #print(self.mu_CO)
        
        # random seed
        self.rseed = 2
         
        #Successful step Counter
        self.iaccept = 0
        
        # View the atom objects during the simulation?
        self.sim_view_flag = False
        
        # Check CO - CO distance flag, a parameter to ensure no multiple CO were adsorbed onto one Pd
        self.check_co_neighbor_flag  = True
        
        # Include CO lateral interaction flag
        self.interaction_flag = True
        
    def write_output_on_event(self, sim_stats_list):
        
        columns = self.stats_columns
        
        sim_stats =  dict([(columns[0], sim_stats_list[0]),  #mu
                           (columns[1], sim_stats_list[1]),  #E_pd
                           (columns[2], [sim_stats_list[2]]), #pd_config 
                           (columns[3], [sim_stats_list[3]]),  #co_config
                           (columns[4], sim_stats_list[4]) #event name
                           ])
        
        # Convert to a dataframe
        df = pd.DataFrame(sim_stats)
         
        with open(self.filename_df, 'a') as f:
            df.to_csv(f, header=f.tell()==0)
        
        
    def write_output_initial(self, sim_stats_list):
        
        # filename for dataframe
        self.filename_df = os.path.join(self.OutputPath, 'GCMC_trajectory.csv')
        
        # Delete the file generated previously
        if os.path.exists(self.filename_df):  os.remove(self.filename_df)
        
        # Save the simulation results to a pandas dataframe
        self.stats_columns = ['mu', 'E_pd', 'pd_config', 'co_config', 'event name']
        
        self.write_output_on_event(sim_stats_list)


    def initialize(self,  pdx):
        
        # Set random seed
        np.random.seed(seed = self.rseed)

        """
        Initialize the simulation with a structure
        """
        pdx = np.array(pdx) # convert to np array for pdx
        '''
        Start recording setups
        '''
        #Predict energy for initial configuration
        pd_config = energy.one_hot_to_index(pdx)
        co_config, cosites, co_pos, binding_Es = [], [], [], []
        cox = np.array([]) #0/1s for CO sites, dynamically changing with Pd configuration
        
        
        E_pd, _  = Pdn.predict_E(pd_config) # the energy of metal clusters only

        #Create initial atom object
        Pdn_atoms, super_mother_with_support = energy.append_support(pd_config, super_mother, view_flag = self.sim_view_flag)
        
        #Update the binding sites
        binding_Es, cosites, co_pos, sitetype_list = binding.predict_binding_Es_fast(Pdn_atoms, view_flag = self.sim_view_flag)
        
        cox = np.zeros(len(cosites))
        nCO = len(co_config)
        
        # keep track of chemical potential
        # Initial chemical potential
        mu = E_pd + np.sum(binding_Es[co_config]) - nCO * self.mu_CO
        
        # Assign the value to self
        # Coordinates and sites
        self.super_mother_with_support = super_mother_with_support
        self.sitetype_list = sitetype_list
        # Pd information
        self.pdx = pdx
        self.pd_config = pd_config
        self.Pdn_atoms = Pdn_atoms
        # CO information
        self.cox = cox
        self.co_pos = co_pos
        self.co_config = co_config
        self.cosites = cosites
        self.nCOsites = len(cosites)
        # energies
        self.binding_Es = binding_Es
        self.E_pd = E_pd
        self.mu = mu        
        
        # Save to csv file
        sim_stats_list = [self.mu, self.E_pd, self.pd_config, self.co_config, 'Initial']      
        self.write_output_initial(sim_stats_list)

    def displace_Pd(self):
        """Displace a particle, Pd
        """
        global w
        
        # Generate a random walk, swap an occupied site and an empty site
        pdx_new, pd_chosen_empty_i, pd_chosen_occ_i  =  Pdn.swap_occ_empty_fast(self.pdx)    
        pd_config_new = energy.one_hot_to_index(pdx_new)
        

        # check if the old node is not attached to CO and the new node does not overlap with CO
        nCO = len(self.co_config) 

        CO_flag = True # default true, CO is not attached to Pd
        # if nCO > 0: 
        #     CO_flag = binding.check_Pd_CO_distance(pd_chosen_empty_i, pd_chosen_occ_i,  
        #                                             self.super_mother_with_support, 
        #                                             self.co_config, self.co_pos)
        # else: pass

        if CO_flag: 
            
            # Initalize new variables
            binding_Es_total = 0
            co_config_new = self.co_config
            cox_new  = self.cox
            cosites_new = self.cosites
            potential_flag = False
            
            # predict the new energy
            E_pd_new, _  = Pdn.predict_E(pd_config_new)
            Pdn_atoms_new = self.Pdn_atoms.copy()
            
            # Screen the surface, find all sites
            Pdn_atoms_new, super_mother_with_support_new = energy.append_support(pd_config_new, super_mother, view_flag = self.sim_view_flag)
            
            if nCO > 0: # if there are adsorbed COs
                
                # update the CO indices, add the choesn CO and desorb the previous CO
                cox_new, co_config_new, binding_Es_new, cosites_new, co_pos_new, sitetype_list_new = binding.update_COsites(Pdn_atoms_new, self.co_config, self.co_pos)
                # include lateral interaction for binding energies
                if self.interaction_flag: 
                    interactions = binding.CO_interactions(co_pos_new, sitetype_list_new, co_config_new)
                    binding_Es_total = interactions.cal_binding_Es_total(binding_Es_new) 
                else: 
                    binding_Es_total = np.sum(binding_Es_new[co_config_new])
                    
            else:
                #Update the binding sites
                binding_Es_new, cosites_new, co_pos_new, sitetype_list_new = binding.predict_binding_Es_fast(Pdn_atoms_new, view_flag = self.sim_view_flag)
                
            # Update the chemical potential
            mu_new = E_pd_new + binding_Es_total - len(co_config_new) * self.mu_CO
            delta_mu = mu_new - self.mu

            # accept the change if energy going downhill
            if delta_mu <= 0: 
                potential_flag = True 
                
            # test using Boltzmann distribution
            else:
                if self.T > 0: w = np.exp(-delta_mu/kb/self.T)
                if np.random.rand() <= w: 
                    potential_flag = True
            
            print(potential_flag) 
            if potential_flag: 
                
                # update all the variables
                # Coordinates and sites
                self.super_mother_with_support = super_mother_with_support_new
                self.sitetype_list = sitetype_list_new
                # Pd information
                self.pdx = pdx_new
                self.pd_config = pd_config_new
                self.Pdn_atoms = Pdn_atoms_new
                
                # CO information
                self.cox = cox_new
                self.co_config = co_config_new
                self.co_pos = co_pos_new
                self.nCOsites = len(cosites_new)
                self.cosites = cosites_new

                # energies
                self.binding_Es = binding_Es_new
                self.E_pd = E_pd_new
                self.mu = mu_new  
        

                # append to the csv file
                sim_stats_list = [self.mu, self.E_pd, self.pd_config, self.co_config, 'displace Pd']      
                self.write_output_on_event(sim_stats_list)
        
                self.iaccept += 1
            
            else:pass
        
        else: pass    

   
    def displace_CO(self):
        """Displace a particle, CO
        """
        
        # displace a CO 
        
        # Generate a random walk, swap an occupied site and an empty site
        #print(co_config)
        global w
        
        cox_new, co_chosen_empty_i, co_chosen_occ_i =  energy.swap_occ_empty(self.cox)    
        co_config_new = energy.one_hot_to_index(cox_new)
         
        
        co_neighbor_flag = True
        if len(co_config_new) > 1 and self.check_co_neighbor_flag: # chekc the distance between two COs
            co_neighbor_flag = binding.check_CO_CO_distance(co_config_new, co_chosen_empty_i, self.co_pos)
        
        if co_neighbor_flag:
            
            # add the choesn CO and desorb the previous CO
            if self.interaction_flag: 
                interactions = binding.CO_interactions(self.co_pos, self.sitetype_list, co_config_new)
                binding_Es_total = interactions.cal_binding_Es_total(self.binding_Es) 
            else: 
                binding_Es_total = np.sum(self.binding_Es[co_config_new])
            
            mu_new = self.E_pd + binding_Es_total - len(co_config_new) * self.mu_CO
            
            potential_flag = False
            delta_mu = mu_new - self.mu
    
            # accept the change if energy going downhill
            if delta_mu <= 0: 
                potential_flag = True 
            # test using Boltzmann distribution
            else:
                if self.T > 0: w = np.exp(-delta_mu/kb/self.T)
                if np.random.rand() <= w: 
                    potential_flag = True
                    
            if potential_flag:
                
                self.cox = cox_new
                self.co_config = co_config_new
                self.mu = mu_new

                # append to the csv file
                sim_stats_list = [self.mu, self.E_pd, self.pd_config, self.co_config, 'displace CO']      
                self.write_output_on_event(sim_stats_list)
        
                self.iaccept += 1
                
            else: pass
        
        else: pass
        
    
    def add_CO(self):
        """
        Add a CO
        """
        global w
        
        nCO = len(self.co_config)
        co_empty_sites = np.where(self.cox == 0.0)[0]
        nCO_empty_sites = len(co_empty_sites)
        
            
        # Choose an empty site and add a CO
        coi = np.random.choice(co_empty_sites, 1, replace = False)[0] # site index
        # check CO distance 
        co_neighbor_flag = True
        
        if self.check_co_neighbor_flag: 
            co_neighbor_flag = binding.check_CO_CO_distance(self.co_config, coi, self.co_pos)
        
        if co_neighbor_flag:
            
            #print(co_neighbor_flag)
            cox_new = self.cox.copy()
            cox_new[coi] = 1 # change occupancy to 1
            co_config_new = energy.one_hot_to_index(cox_new)
            
            if self.interaction_flag: 
                interactions = binding.CO_interactions(self.co_pos, self.sitetype_list, co_config_new)
                binding_Es_total = interactions.cal_binding_Es_total(self.binding_Es)
            else: 
                binding_Es_total = np.sum(self.binding_Es[co_config_new])
            
            mu_new = self.E_pd + binding_Es_total - len(co_config_new) * self.mu_CO

            potential_flag = False
            delta_mu = mu_new - self.mu

            # accept the change if energy going downhill
            if delta_mu <= 0: 
                potential_flag = True 
                # test using Boltzmann distribution
            else:
                if self.T > 0: w = np.exp(-delta_mu/kb/self.T)
                if np.random.rand() <= w: 
                    potential_flag = True
                    
            if potential_flag:
                
                self.cox = cox_new
                self.co_config = co_config_new
                self.mu = mu_new
                
                # append to the csv file
                sim_stats_list = [self.mu, self.E_pd, self.pd_config, self.co_config, 'add CO']      
                self.write_output_on_event(sim_stats_list)
                
                self.iaccept += 1
                
            else: pass
        
        else: pass
    
    def remove_CO(self):
        """
        Remove a CO
        
        """
        global w
    

        # Choose a occupied site and remove a CO
        coi = np.random.choice(self.co_config, 1, replace = False)[0] # site index
        
        cox_new = self.cox.copy()
        cox_new[coi] = 0 # change occupancy to 0
        co_config_new = energy.one_hot_to_index(cox_new)
        
        
        if self.interaction_flag: 
                interactions = binding.CO_interactions(self.co_pos, self.sitetype_list, co_config_new)
                binding_Es_total = interactions.cal_binding_Es_total(self.binding_Es)
        else: 
            binding_Es_total = np.sum(self.binding_Es[co_config_new])
        
        mu_new = self.E_pd + binding_Es_total - len(co_config_new) * self.mu_CO
        potential_flag = False
        delta_mu = mu_new - self.mu
     
        
        # accept the change if energy going downhill
        if delta_mu <= 0: 
            potential_flag = True 
            # test using Boltzmann distribution
        else:
            if self.T > 0: w = np.exp(-delta_mu/kb/self.T)
            if np.random.rand() <= w: 
                potential_flag = True
                  
        if potential_flag:
            
            self.cox = cox_new
            self.co_config = co_config_new
            self.mu = mu_new
            
            # append to the csv file
            sim_stats_list = [self.mu, self.E_pd, self.pd_config, self.co_config, 'remove CO']      
            self.write_output_on_event(sim_stats_list)
            
            self.iaccept += 1
                    
        else: pass 

        
        
class Trajectory():
    """GCMC simulation trajectory class
    """

    def __init__(self, T, PCO, pdx, name = None, progress_flag = True, InputPath = os.path.abspath(os.getcwd()), ResultsName = 'results'):
        """start a simulation trajectory
        
        :param T: temperature
        :type T: float
        :param PCO: pressure of CO
        :type PCO: float
        :param pdx: the pd configuration list in one hot
        :type pdx: list 

        """
        # set up simulation specifics
        self.rseed = 2
        self.nsteps = 10000
        self.rdisplacement = 0.1 
        np.random.rand(self.rseed)

        # Reaction conditions
        self.T = T
        self.PCO = PCO    
        
        # Initialize the configuration
        self.pdx = pdx
        self.nPd = np.sum(self.pdx)

        # Set output flag
        self.progress_flag = progress_flag

        # set initial csv filename
        self.df_file = None

        # Set I/O path
        if name == None: self.name = 'Pd_' + str(self.nPd)
        else: self.name = name

        OutputPathName = 'run_' + self.name + '_' + str(self.T)+'k_' + str(self.PCO) + '_bar_' + str(self.rseed)
        
        # level 1 directory contains all simulation results
        ResultPath = os.path.join(InputPath, ResultsName)
        if not os.path.exists(ResultPath): os.makedirs(ResultPath)
        
        # level 2 directory contains results for specific configuration
        self.TrajectoryPath = os.path.join(ResultPath, name)
        if not os.path.exists(self.TrajectoryPath): os.makedirs(self.TrajectoryPath)
        
        # level 3 directory contains results for specific trajectory
        self.OutputPath = os.path.join(self.TrajectoryPath, OutputPathName)
        if not os.path.exists(self.OutputPath): os.makedirs(self.OutputPath)

        
    def run(self):
        """run the GCMC simulation at given parameters
        """
        # Start timing
        start_time = time.time() 

        simulation = Simulation(self.T, self.PCO, self.OutputPath)
        simulation.rseed = self.rseed
        simulation.check_co_neighbor_flag  = True
        simulation.initialize(self.pdx)

        # calculate the CO site statis
        nCOsites = simulation.nCOsites
        nCO = len(simulation.co_config)
        nCO_empty_sites = len(np.where(simulation.cox == 0.0)[0])


        # Main body for MC iterations
        for istep in range(0, self.nsteps):
            
            r1 = np.random.rand()
            
            # displacement move
            if  nCO > 0  and nCO_empty_sites > 0 and r1 < self.rdisplacement:
                simulation.displace_CO()
                nCO = len(simulation.co_config)
                nCO_empty_sites = len(np.where(simulation.cox == 0.0)[0])
            
            # if r1 < self.rdisplacement:
            #     simulation.displace_CO()
            #     nCO = len(simulation.co_config)
            #     nCOsites = simulation.nCOsites
            #     nCO_empty_sites = len(np.where(simulation.cox == 0.0)[0])
            
            # add and remove move
            if r1 >= self.rdisplacement:
                
                r2 = np.random.rand()
                # add CO
                if nCO_empty_sites > 0 and r2 < 0.5:
                    simulation.add_CO()
                    nCO = len(simulation.co_config)
                    nCO_empty_sites = len(np.where(simulation.cox == 0.0)[0])
                
                # remove CO
                else:
                    r3 = np.random.rand()
                    if nCO > 0 and r3 < nCO/nCOsites:
                        simulation.remove_CO()
                        nCO = len(simulation.co_config)
                        nCO_empty_sites = len(np.where(simulation.cox == 0.0)[0])    
                    
            # Update the index and print simulation progress
            if self.progress_flag:
                progress = np.around((istep+1)/self.nsteps*100, decimals = 3)
                print('{}% done!,  {} accpted, Chemical Potential = {}'.format(progress, simulation.iaccept,  simulation.mu))
    
        # Record the time
        end_time = time.time()
        self.sim_time = (end_time - start_time) / 60
        print('Pd{0} at {1} K, {2:.2e} CO pressure: simulation takes {3:.4f} minutes'.format(self.nPd, self.T, self.PCO, self.sim_time))

        # Return the simulation file name
        self.df_file = simulation.filename_df 
        


    def analysis(self, plot_flag = True, mean_flag = True, save_png_flag = True): 
        """analysis a single trajectory, plot average CO and chemical potential per npd 
        
        :return: final nco and chemical potential
        :rtype: float, float
        """
        if self.df_file == None:
            self.df_file = os.path.join(self.OutputPath, 'GCMC_trajectory.csv')
        
        df = pd.read_csv(open(self.df_file, 'r'))

        # process the pd configurations
        pd_config_history = split_pd_list(df['pd_config'][1:])
        # process the co configurations
        co_config_history = [] + split_pd_list(df['co_config'][1:])
        
        
        mu_traj = np.array(df['mu'])
        nCO_traj = np.array([len(ncoi) for ncoi in co_config_history])

        # average based on the cluster size
        mu_traj_mean = mu_traj/self.nPd
        nCO_traj_mean = nCO_traj/self.nPd
        
        if plot_flag:
            if mean_flag:
                mu_traj_plot = mu_traj_mean.copy()
                nCO_traj_plot = nCO_traj_mean.copy()
            else:
                mu_traj_plot = mu_traj.copy()
                nCO_traj_plot = nCO_traj.copy()
            # Make trajectory plots
            fig, ax = plt.subplots(figsize= (6,6))
            ax.plot(mu_traj_plot)
            ax.set_xscale('log')
            ax.set_xlabel('# Moves')
            ax.set_ylabel(r'$\rm Relative\ \mu (eV)$')
            fig.tight_layout()
            fig.savefig(os.path.join(self.OutputPath, 'mu' + '_trajectory.PNG'), dpi = dpi)

            fig, ax = plt.subplots(figsize= (6,6))
            ax.plot(nCO_traj_plot)
            ax.set_xscale('log')
            ax.set_xlabel('# Moves')
            ax.set_ylabel('# CO')
            fig.tight_layout()
            fig.savefig(os.path.join(self.OutputPath, 'nCO' + '_trajectory.PNG'), dpi = dpi)
        
        
        '''
        view the final configuration with COs in ASE
        '''
        pd_config = energy.one_hot_to_index(self.pdx)
        E_pd, _  = Pdn.predict_E(pd_config) # the energy of metal clusters only
        
        # Extract the final values
        if len(co_config_history) > 0:
            co_config_final = co_config_history[-1]
            self.nCO_final_mean= nCO_traj_mean[-1]
            self.mu_final_mean = mu_traj_mean[-1]
        else: 
            co_config_final = []
            self.nCO_final_mean= 0
            self.mu_final_mean = E_pd/self.nPd
            
        
        
        # view the configuration in ase if not saving to a pov file
        if save_png_flag:
            view_flag = False
        else: view_flag = True
        
        #Create initial atom object
        if len(pd_config_history) > 0:
            pd_config_final = pd_config_history[-1]
        else: pd_config_final= pd_config
        
        self.Pdn_atoms, _ = energy.append_support(pd_config_final, super_mother, view_flag = False)
        PdnCOm_obj = binding.PdnCOm(self.Pdn_atoms)
        self.PdnCOm_atoms = PdnCOm_obj.append_COs(co_config_final, view_flag = view_flag)

        if save_png_flag:
            binding.save_POV(Pdi = self.name, index = 0, atoms = self.PdnCOm_atoms, output_dir= self.OutputPath)
            
        return self.nCO_final_mean, self.mu_final_mean

#%% Rejection free scheme
# 
class SimulationRejectionFree():
    """
    Main class for GCMC simulation
    """
    
    def __init__(self, T, PCO, OutputPath):
        """Set constants and default simulation conditions"""
        
        # set output path
        self.OutputPath = OutputPath

        # Temperature of the simulation in K
        self.T = T
        
        # Partial pressure of CO in bar
        self.PCO = PCO
        
        # Chemical potential of CO at T
        self.mu_CO = mu_CO_p0 + kb * self.T * np.log(self.PCO/P0)
        #print(self.mu_CO)
        
        # random seed
        self.rseed = 2
         
        #Successful step Counter
        self.iaccept = 0
        
        # View the atom objects during the simulation?
        self.sim_view_flag = False
        
        # Check CO - CO distance flag, a parameter to ensure no multiple CO were adsorbed onto one Pd
        self.check_co_neighbor_flag  = True
        
        
    def write_output_on_event(self, sim_stats_list):
        
        columns = self.stats_columns
        
        sim_stats =  dict()

        for column, sim_stats_i in zip(columns, sim_stats_list):
            sim_stats[column] =  sim_stats_i
        
        # Convert to a dataframe
        df = pd.DataFrame(sim_stats)
         
        with open(self.filename_df, 'a') as f:
            df.to_csv(f, header=f.tell()==0)
        
        
    def write_output_initial(self, sim_stats_list):
        
        # filename for dataframe
        self.filename_df = os.path.join(self.OutputPath, 'GCMC_trajectory.csv')
        
        # Delete the file generated previously
        if os.path.exists(self.filename_df):  os.remove(self.filename_df)
        
        # Save the simulation results to a pandas dataframe
        self.stats_columns = ['i', 'dt', 'mu_val', 'mu_cur',  'co_config', 'event name']
        
        self.write_output_on_event(sim_stats_list)


    def initialize(self,  pdx):
        
        # Set random seed
        np.random.seed(seed = self.rseed)

        """
        Initialize the simulation with a structure
        """
        pdx = np.array(pdx) # convert to np array for pdx
        '''
        Start recording setups
        '''
        #Predict energy for initial configuration
        pd_config = energy.one_hot_to_index(pdx)
        co_config, cosites, co_pos, binding_Es = [], [], [], []
        cox = np.array([]) #0/1s for CO sites, dynamically changing with Pd configuration
        
        
        E_pd, _  = Pdn.predict_E(pd_config) # the energy of metal clusters only

        #Create initial atom object
        Pdn_atoms, super_mother_with_support = energy.append_support(pd_config, super_mother, view_flag = self.sim_view_flag)
        
        #Update the binding sites
        binding_Es, cosites, co_pos, sitetype_list = binding.predict_binding_Es_fast(Pdn_atoms, view_flag = self.sim_view_flag)
        
        cox = np.zeros(len(cosites))
        nCO = len(co_config)
        
        # keep track of chemical potential
        # Initial chemical potential
        mu = E_pd + np.sum(binding_Es[co_config]) - nCO * self.mu_CO
        
        # Assign the value to self
        # Coordinates and sites
        self.super_mother_with_support = super_mother_with_support
        self.sitetype_list = sitetype_list
        # Pd information
        self.pdx = pdx
        self.pd_config = pd_config
        self.Pdn_atoms = Pdn_atoms
        # CO information
        self.cox = cox
        self.co_pos = co_pos
        self.cosites = cosites
        self.nCOsites = len(cosites)
        # energies
        self.binding_Es = binding_Es
        self.E_pd = E_pd
        
        # Save to csv file
        sim_stats_list = [0, 0,  mu, mu, [co_config], 'Initial']      
        self.write_output_initial(sim_stats_list)


        
class TrajectoryRejectionFree():
    """GCMC simulation trajectory class rejection free
    """
    def __init__(self, T, PCO, pdx, name = None, batch_name = None,  progress_flag = True):
        """start a simulation trajectory
        
        :param T: temperature
        :type T: float
        :param PCO: pressure of CO
        :type PCO: float
        :param pdx: the pd configuration list in one hot
        :type pdx: list 

        """
        # set up simulation specifics
        self.nsteps = 100

        # Reaction conditions
        self.T = T
        self.PCO = PCO    
        
        # Initialize the configuration
        self.pdx = pdx
        self.nPd = np.sum(self.pdx)
        self.pdx_config = energy.one_hot_to_index(self.pdx)
        E_pd, _  = Pdn.predict_E(self.pdx_config) 
        self.E_pd = E_pd # the energy of metal clusters only

        # Set output flag
        self.progress_flag = progress_flag
        
        # Set I/O path
        if name == None: self.name = 'pd' + str(self.nPd)
        else: self.name = name
        
        if batch_name == None: self.batch_name = 'i0'
        else: self.batch_name = batch_name

        
    def set_subdirectories(self, rseed = 0, InputPath = os.path.abspath(os.getcwd()), ResultsName = 'resultsRejectionFree'):
        
        self.rseed = rseed
        np.random.rand(self.rseed)
        
        # set initial csv filename
        self.df_file = None

        OutputPathName = 'run_' + self.name + '_' + self.batch_name + '_' + str(self.T)+'k_' + str(self.PCO) + '_bar_' + str(self.rseed)
        
        # level 1 directory contains all simulation results
        ResultPath = os.path.join(InputPath, ResultsName)
        if not os.path.exists(ResultPath): os.makedirs(ResultPath)
        
        # level 2 directory contains results for specific size
        self.SizePath = os.path.join(ResultPath, self.name)
        if not os.path.exists(self.SizePath): os.makedirs(self.SizePath)
                                                                
        # level 3 directory contains results for specific configuration - batch 
        self.BatchPath = os.path.join(self.SizePath, self.batch_name)
        if not os.path.exists(self.BatchPath): os.makedirs(self.BatchPath)
        
        # level 4 directory contains results for specific trajectory
        self.OutputPath = os.path.join(self.BatchPath, OutputPathName)
        if not os.path.exists(self.OutputPath): os.makedirs(self.OutputPath)


    def cal_pair_interactions(self, site_new, site_indices, interactions):
        """calculate the pairwise interactions 
        between site i and other sites on the surface"""
        
        pair_interactions = []
        for site_i in site_indices:  
            co_config_i = [site_new, site_i]
            interactions.define_config(co_config_i)
            total_interactions = interactions.cal_interactions() 
            pair_interactions.append(total_interactions)
            
        return pair_interactions 

    def update_dmu(self, dmu_init, all_pair_interactions, sites_indices, sites_occ):
        """Update the dmu vector given the current occupied sites"""
        dmu_new = []
        for si in sites_indices:
            if si in sites_occ:
                # negate the mu and remove all the interactions at site si
                dmu_new_i = - dmu_init[si] - np.sum([all_pair_interactions[si][sj] for sj in sites_occ])
            else: 
                # add all interactions with the occupied sites j at site si
                dmu_new_i =  dmu_init[si] + np.sum([all_pair_interactions[sj][si] for sj in sites_occ])

            dmu_new.append(dmu_new_i)
            
        return dmu_new

    def get_config_mu(self, co_config, binding_Es, mu_CO, interactions):
        """calculate mu for a given configuration"""
        # validation for mu curr calculated during the simulation
        co_config = sorted(co_config)
        
        interactions.define_config(co_config)
        mu_curr = interactions.cal_binding_Es_total(binding_Es) - len(co_config)*mu_CO
        
        return mu_curr

    def run(self):
            """run the GCMC simulation at given parameters
            """
            # Start timing
            start_time = time.time() 

            # Initialize the simulation object
            simulation = SimulationRejectionFree(self.T, self.PCO, self.OutputPath)
            simulation.rseed = self.rseed
            simulation.check_co_neighbor_flag  = True
            simulation.initialize(self.pdx)

            # Extract site information
            binding_Es = simulation.binding_Es
            co_pos = simulation.co_pos
            cosites = simulation.cosites
            self.cosites = cosites
            sitetype_list = simulation.sitetype_list
            self.sitetype_list = sitetype_list

            # Extract other environmental constants
            mu_CO = simulation.mu_CO

            # Initialize interaction object
            interactions = binding.CO_interactions(co_pos, sitetype_list)


            # total number of CO sites
            nCOsites = len(cosites) 
            sites_indices = list(range(0, nCOsites))

            # occupied site list
            sites_occ = []
            # sites being too closed to the current occupied sites 
            sites_banned = []

            # dictionary of pair interactions between all site pairs 
            # key the current site, value the interaction value with other site
            all_pair_interactions = {}
            for si in sites_indices:
                all_pair_interactions[si] = self.cal_pair_interactions(si, sites_indices, interactions)

            # Initialize the free energy with cluster energy
            mu_curr = self.E_pd

            # Initial free energy change for each event (each one is adjusted by a mu_CO)
            dmu_init = list(binding_Es - mu_CO)

            # Initialize the heap
            t_heap = get_t_heap(self.T, dmu_init, sites_indices)

            # Simulation main loop

            nsteps = self.nsteps
            counter_step = 0
            dmu = dmu_init.copy()
            dt = 0 # time increments

            while counter_step < nsteps:
                
                # select the top element with the smallest time
                heapify(t_heap)
                item_i = heappop(t_heap)
                site_i = item_i[1]
                
                # look down the heap if the site is banned
                while site_i in sites_banned:
                    item_i = heappop(t_heap)
                    site_i = item_i[1]
                
                # update the current co configuration, sites_occ
                # if this is a new site, check distance for adsorption
                if not site_i in sites_occ:
                    event_type = 'ads'
                    co_neighbor_flag = binding.check_CO_CO_distance(sites_occ, site_i, co_pos)    
                    if co_neighbor_flag: 
                        sites_occ.append(site_i)
                    else: 
                        sites_banned.append(site_i)
                # if the site is occupied, desorption
                else: 
                    event_type = 'des'
                    sites_occ.remove(site_i)
                    sites_banned = []
                    
                # if the co-co distance is allowed or desorption    
                if event_type == 'des' or co_neighbor_flag:
                        
                    # update current mu
                    dmu_i = dmu[site_i]
                    
                    # time increment
                    dt = item_i[0] 

                    mu_curr += dmu_i                            
                    mu_valid = self.get_config_mu(sites_occ, binding_Es, mu_CO, interactions)

                    # Update dmu
                    dmu = self.update_dmu(dmu_init, all_pair_interactions, sites_indices, sites_occ)

                    # Update the t heap
                    t_heap = get_t_heap(self.T, dmu, sites_indices)
                
                    # increment the counter
                    counter_step += 1
                
                    co_config = sites_occ[:]  # shallow copy of the list     

                    # append to the csv file
                    sim_stats_list = [counter_step, dt, mu_valid, mu_curr, [co_config] , event_type]      
                    simulation.write_output_on_event(sim_stats_list)
 

            # Record the time
            end_time = time.time()
            self.sim_time = (end_time - start_time) / 60
            print('Pd{0} at {1} K, {2:.2e} CO pressure: simulation takes {3:.4f} minutes'.format(self.nPd, self.T, self.PCO, self.sim_time))

            # Return the simulation file name
            self.df_file = simulation.filename_df 


    def analysis(self, plot_flag = True, save_png_flag = True, time_average = True): 
        """analysis a single trajectory, plot average CO and chemical potential per npd 
        
        :return: final nco and chemical potential
        :rtype: float, float
        """
        if self.df_file == None:
            self.df_file = os.path.join(self.OutputPath, 'GCMC_trajectory.csv')
        
        df = pd.read_csv(open(self.df_file, 'r'))
 
        # process the co configurations
        co_config_history = split_pd_list(df['co_config'])
        n_iter = len(co_config_history)
        dt_traj = np.array(df['dt'])
        mu_traj = np.array(df['mu_cur'])
        nCO_traj = np.array([len(co_config_i) for co_config_i in co_config_history])

        # average based on the cluster size
        mu_traj = mu_traj/self.nPd
        nCO_traj = nCO_traj/self.nPd
        
        '''
        Generate trajectory plots
        '''
        if plot_flag:
            mu_traj_plot = mu_traj.copy()
            nCO_traj_plot = nCO_traj.copy()
            
            # Make trajectory plots
            fig, ax = plt.subplots(figsize= (5,5))
            ax.plot(mu_traj_plot)
            ax.set_xscale('log')
            ax.set_xlabel('# Moves')
            ax.set_ylabel(r'$\rm \overline{G}\ (eV)$')
            fig.tight_layout()
            fig.savefig(os.path.join(self.OutputPath, 'mu' + '_trajectory.PNG'), dpi = dpi)

            fig, ax = plt.subplots(figsize= (5,5))
            ax.plot(nCO_traj_plot)
            ax.set_xscale('log')
            ax.set_xlabel('# Moves')
            ax.set_ylabel(r'$\rm \overline{m}_{CO}\ (eV)$')
            fig.tight_layout()
            fig.savefig(os.path.join(self.OutputPath, 'nCO' + '_trajectory.PNG'), dpi = dpi)
        
        '''
        Obtain the low energy structure
        '''
        # view the configuration in ase if not saving to a pov file
        if save_png_flag:
            view_flag = False
        else: view_flag = True

        # Extract the configuration with lowest mu
        mu_opt_index = np.argmin(mu_traj)
        self.mu_opt = mu_traj[mu_opt_index]
        self.co_config_opt = co_config_history[mu_opt_index]
        self.nCO_opt = nCO_traj[mu_opt_index]
   

        # Visualize atoms
        self.Pdn_atoms, _ = energy.append_support(self.pdx_config, super_mother, view_flag = False)
        PdnCOm_obj = binding.PdnCOm(self.Pdn_atoms)
        self.PdnCOm_atoms = PdnCOm_obj.append_COs(self.co_config_opt, view_flag = view_flag)


        if save_png_flag:
            binding.save_POV(Pdi = self.name, index = 0, atoms = self.PdnCOm_atoms, output_dir= self.OutputPath)
    
        '''
        Compute the time average
        '''
        if time_average:

            # set start index to discard the 1/5 of non-equilirium state
            start_index = int(n_iter/5)

            dt_v = dt_traj[start_index + 1:]
            mu_v = mu_traj[start_index: -1]
            nCO_v = nCO_traj[start_index: -1]

            self.mu_mean = np.dot(mu_v, dt_v)/np.sum(dt_v)
            self.nCO_mean = np.dot(nCO_v, dt_v)/np.sum(dt_v)
            
            
            return self.nCO_mean, self.mu_mean
        
        else:
            return self.nCO_opt, self.mu_opt
        
        
    def analyze_sites(self):
        """Analyze the type of sites for a CO configurations"""
        # Calculate structure descriptors for the most stable configuration
        
        sitetype_list, GCNs, CN1s, CN2s, ratio_surface, Z_pos = binding.predict_binding_Es_fast(self.Pdn_atoms, ind_index =  self.co_config_opt,  stable_info = True)
        
        stable_config_info = dict()

        #From the trajectory
        stable_config_info['n'] = self.nPd
        stable_config_info['T'] = self.T
        stable_config_info['PCO'] = self.PCO
        
        stable_config_info['rseed'] = self.rseed
        stable_config_info['mu_mean'] = self.mu_mean
        stable_config_info['nCO_mean'] = self.nCO_mean

        # descriptors for the Pdn structure only
        stable_config_info['GCN'] = np.mean(GCNs)
        stable_config_info['CN1']  = np.mean(CN1s)
        stable_config_info['CN2'] = np.mean(CN2s)
        stable_config_info['ratio_surface']  = np.mean(ratio_surface)

        

        sitestype_occ = []
        for si in self.co_config_opt:
            sitestype_occ.append(sitetype_list[si])
        sitetype_counter = collections.Counter(sitetype_list)
        sitetype_counter_occ = collections.Counter(sitestype_occ) 
        
        total_top = sitetype_counter['top']
        total_bridge = sitetype_counter['bridge']
        total_hollow = sitetype_counter['hollow']
        
        #average per pd atom
        stable_config_info['total_top']  = total_top/self.nPd
        stable_config_info['total_bridge']  = total_bridge/self.nPd
        stable_config_info['total_hollow']  = total_hollow/self.nPd
        stable_config_info['total_sites'] =  total_top + total_bridge + total_hollow

        
        stable_config_info['occ_top'] = sitetype_counter_occ['top']/self.nPd
        stable_config_info['occ_bridge'] = sitetype_counter_occ['bridge']/self.nPd
        stable_config_info['occ_hollow'] = sitetype_counter_occ['hollow']/self.nPd
        # descriptors for occupied CO adsorption sites 
        stable_config_info['Z_pos']  =  np.mean(Z_pos) 
        

        return stable_config_info

#%% Plotting functions for GCMC
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.pyplot import cm
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

plot_legend = False

def get_pickle_path(npdx, InputPath = os.path.abspath(os.getcwd()), ResultsName = 'resultsRejectionFree'):
    '''set path for pickle file'''
    name = 'pd' + str(npdx)
    
    # level 1 directory contains all simulation results
    ResultPath = os.path.join(InputPath, ResultsName)
    if not os.path.exists(ResultPath): os.makedirs(ResultPath)
    
    # level 2 directory contains results for specific size
    SizePath = os.path.join(ResultPath, name)
    if not os.path.exists(SizePath): os.makedirs(SizePath)
    
    return ResultPath


def plot_heatmap(xv, yv, vmin, vmax, y_heatmap, interpolate = 'gaussian'):
    """
    Heatmap plotting function
    """
    
    x_left, x_right = np.min(np.log10(xv)), np.max(np.log10(xv))
    y_left, y_right = np.min(yv), np.max(yv)
    # set number of ticksections
    n_tick_sections = 4
    fig,ax = plt.subplots(figsize=(5, 7))
    ax.set_aspect('equal', adjustable='box')
    im = ax.imshow(y_heatmap, cmap = 'Spectral_r', interpolation = 'gaussian', aspect = 'auto', origin = 'upper',  vmin = vmin, vmax = vmax)

    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.2)
    plt.colorbar(im, cax = cax)
    # set xy tick locations
    xtick_loc = [int(i) for i in np.linspace(0, xv.shape[0]-1, n_tick_sections)]
    ytick_loc = [int(i) for i in np.linspace(0, yv.shape[0]-1, n_tick_sections)]
    #ax.axis('equal')
    

    ax.set_xticks(xtick_loc)
    ax.set_yticks(ytick_loc)
    ax.set_xticklabels(np.linspace(x_left, x_right, n_tick_sections, dtype = int))
    ax.set_yticklabels(np.linspace(y_left, y_right, n_tick_sections, dtype = int))
    ax.set_xlabel(r'$\rm log(P_{CO}\ (bar)) $')
    ax.set_ylabel(r'$\rm Temperature\ (K) $')
    ax.invert_yaxis()

    ax.minorticks_on()
    ax.tick_params(which='minor', length=4, width=1.5, direction='in')
    ax.tick_params(which='major', direction='in')
    #plt.tight_layout()
    plt.show()
    
    return fig

def plot_heatmap_cluster_size(xv, yv, vmin, vmax, y_heatmap, interpolate = 'gaussian'):
    """
    Heatmap plotting function
    """
    
    x_left, x_right = np.min(np.log10(xv)), np.max(np.log10(xv))
    y_left, y_right = np.min(yv), np.max(yv)
    # set number of ticksections
    n_tick_sections = 6
    fig,ax = plt.subplots(figsize=(7, 10))
    ax.set_aspect('equal', adjustable='box')
    im = ax.imshow(y_heatmap, cmap = 'Spectral_r', interpolation = 'gaussian', aspect = 'auto', origin = 'upper',  vmin = vmin, vmax = vmax)

    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size ="5%", pad=0.2)
    cticklables = [0, 5, 10, 15, 20]
    cbar = plt.colorbar(im, cax = cax, ticks = cticklables)
    #cbar.ax.set_yticks() 
    cbar.ax.set_yticklabels(cticklables) 
    # set xy tick locations
    xtick_loc = np.linspace(-0.3, xv.shape[0]-0.5, xv.shape[0])
    ytick_loc = np.linspace(-0.3, yv.shape[0]-0.5, yv.shape[0])
    #ax.axis('equal')
    

    ax.set_xticks(xtick_loc)
    ax.set_yticks(ytick_loc)
    ax.set_xticklabels([int(np.log10(xi)) for xi in xv])
    ax.set_yticklabels([int(yi) for yi in yv])
    ax.set_xlabel(r'$\rm log_{10}(P_{CO}\ (bar)) $')
    ax.set_ylabel(r'$\rm Temperature\ (K) $')
    ax.invert_yaxis()

    ax.minorticks_on()
    ax.tick_params(which='minor', length=4, width=1.5, direction='in')
    ax.tick_params(which='major', direction='in')
    #plt.tight_layout()
    plt.show()
    
    return fig

def plot_isobar(Pslide, Pv, Tv, y_mu, y_mco, iso_names):
    """Plot 1D isobars at the chosen Pslide value"""
    y_mu = y_mu.copy()
    y_mco = y_mco.copy() 
    if len(y_mco.shape) == 2: # add one more dimension if only two provided
        y_mu = y_mu[np.newaxis, :]
        y_mco = y_mco[np.newaxis, :]
        
    n_sim, nT, nP =  y_mu.shape
    Pmin, Pmax = np.min(Pv), np.max(Pv)
    Tmin, Tmax = np.min(Tv), np.max(Tv)
    
    x = np.where(Pv == Pslide)[0]
    
    # set x tick locations
    n_tick_sections = 4   
    
    #print(x)
    # isobars for mu
    pdix_mu_v = np.reshape(y_mu[:, :, x], (n_sim, nT))
    fig1,ax = plt.subplots(figsize=(5, 5))
    for n_sim_i, isoname_i in zip(range(0, n_sim), iso_names) :
        ax.plot(Tv, pdix_mu_v[n_sim_i, :], label = isoname_i,  markersize= 7, marker = 'o', fillstyle = 'none')
    
    ax.set_xlim(Tmin, Tmax)
    ax.set_xticks(np.linspace(Tmin, Tmax, n_tick_sections, dtype = int))
    ax.set_ylim(-4, 0)
    ax.set_xlabel('Temperature (K)')
    ax.set_ylabel(r'$\rm \overline{G}\ (eV)$')
    if plot_legend: ax.legend(bbox_to_anchor=(1.1, 1.1), loc='upper left')
    #
    #ax.legend(loc='upper center', fontsize=14, bbox_to_anchor=(0.5, -0.2), ncol = 5) #loc='upper left', bbox_to_anchor=(1, 1)
    
    
    # isobars for mco
    pdix_mco_v = np.reshape(y_mco[:, :, x], (n_sim, nT))
    fig2,ax = plt.subplots(figsize=(5, 5))
    for n_sim_i, isoname_i in zip(range(0, n_sim), iso_names):
        ax.plot(Tv, pdix_mco_v[n_sim_i, :], label = isoname_i, markersize= 7, marker = 'o', fillstyle = 'none')
    
    ax.set_xlim(Tmin, Tmax)
    ax.set_xticks(np.linspace(Tmin, Tmax, n_tick_sections, dtype = int))
    ax.set_ylim(0, 2)
    ax.set_xlabel('Temperature (K)')
    ax.set_ylabel(r'$\rm \overline{m}_{CO}\ (eV)$')
    if plot_legend: ax.legend(bbox_to_anchor=(1.1, 1.1), loc='upper left')
    #ax.legend(loc='upper center', fontsize=14, bbox_to_anchor=(0.5, -0.2), ncol = 5) #loc='upper left', bbox_to_anchor=(1, 1)
    plt.show()
    
    return fig1, fig2
    

def plot_isotherm(Tslide, Pv, Tv, y_mu, y_mco, iso_names):
    """Plot 1D isobars at the chosen Pslide value"""
    y_mu = y_mu.copy()
    y_mco = y_mco.copy() 
    if len(y_mco.shape) == 2: # add one more dimension if only two provided
        y_mu = y_mu[np.newaxis, :]
        y_mco = y_mco[np.newaxis, :]
        
    n_sim, nT, nP =  y_mu.shape
    Pmin, Pmax = np.min(Pv), np.max(Pv)
    Tmin, Tmax = np.min(Tv), np.max(Tv)

    x = np.where(np.abs(Tv - Tslide) <= 0.1)[0]
    #print(x)
    

    # isotherms for mu
    pdix_mu_v = np.reshape(y_mu[:, x, :], (n_sim, nP))
    fig1,ax = plt.subplots(figsize=(5, 5))
    for n_sim_i, isoname_i in zip(range(0, n_sim), iso_names) :
        ax.plot(Pv, pdix_mu_v[n_sim_i, :], label = isoname_i, markersize= 7, marker = 'o', fillstyle = 'none')
        
    # set x tick locations
    n_tick_sections = 4    
    
    ax.set_xlim(Pmin, Pmax)
    ax.set_ylim(-4, 0)
    ax.set_xscale('log')
    ax.set_xticks([10** int(i) for i in np.linspace(np.log10(Pmin), np.log10(Pmax), n_tick_sections, dtype = int)])
    ax.set_xlabel(r'$\rm P_{CO}\ (bar)$')
    ax.set_ylabel(r'$\rm \overline{G}\ (eV)$')
    if plot_legend: ax.legend(bbox_to_anchor=(1.1, 1.1), loc='upper left')
    #ax.legend(loc='upper center', fontsize=14, bbox_to_anchor=(0.5, -0.2), ncol = 5) #loc='upper left', bbox_to_anchor=(1, 1)
    
    # isotherms for mco
    pdix_mco_v = np.reshape(y_mco[:, x, :], (n_sim, nP))
    fig2,ax = plt.subplots(figsize=(5, 5))
    for n_sim_i, isoname_i in zip(range(0, n_sim), iso_names):
        ax.plot(Pv, pdix_mco_v[n_sim_i, :], label = isoname_i, markersize= 7, marker = 'o', fillstyle = 'none')
    
    ax.set_xlim(Pmin, Pmax)
    ax.set_ylim(0, 2)
    ax.set_xscale('log')
    ax.set_xticks([10** int(i) for i in np.linspace(np.log10(Pmin), np.log10(Pmax), n_tick_sections, dtype = int)])
    ax.set_xlabel(r'$\rm P_{CO}\ (bar)$')
    ax.set_ylabel(r'$\rm \overline{m}_{CO}\ (eV)$')
    if plot_legend: ax.legend(bbox_to_anchor=(1.1, 1.1), loc='upper left')
    #ax.legend(loc='upper center', fontsize=14, bbox_to_anchor=(0.5, -0.2), ncol = 5) #loc='upper left', bbox_to_anchor=(1, 1)
    plt.show()
    
    return fig1, fig2


def plot_pdn(Px, Tx, pdi_nCO_v, pdi_mu_v, Tmin, Tmax, Pmin, Pmax, iso_names):
    """Generate plots for a size n"""
    
    # extract the dimensions of the simulations
    pdi_nCO_v = pdi_nCO_v.copy()
    pdi_mu_v = pdi_mu_v.copy() 
    if len(pdi_nCO_v.shape) == 2: # add one more dimension if only two provided
        pdi_nCO_v = pdi_nCO_v[np.newaxis, :]
        pdi_mu_v = pdi_mu_v[np.newaxis, :]
    
    # the tensor has 3 dimension: the isomer structure, temperature, and pressure 
    # set the dimensions to be consistent with the input
    n_sim, nT, nP =  pdi_nCO_v.shape
    
    # reaction condition vector
    Tv = np.linspace(Tmin, Tmax, nT)
    Pv = np.logspace(Pmin, Pmax, nP) # Partial pressure of CO in bar
    
    for xi in range(n_sim):
        # Select an isomer
        # Plot the heatmap, Tv is for the rows so it becomes yv, Pv is for the columns for xv
        pdix_mu_v = pdi_mu_v[xi, :, :]
        pdix_nCO_v = pdi_nCO_v[xi, :, :]
    
        # for chemical potential
        mu_fig_i = plot_heatmap(Pv, Tv, -4, 0, pdix_mu_v)
        
        # for the number of CO
        mco_fig_i = plot_heatmap(Pv, Tv, 0, 2 , pdix_nCO_v)
        



def plot_isobar_spread(Pslide, Pv, Tv, y_mu, y_mco, iso_names):
    """Plot 1D isobars at the chosen Pslide value"""
    y_mu = y_mu.copy()
    y_mco = y_mco.copy() 
    if len(y_mco.shape) == 2: # add one more dimension if only two provided
        y_mu = y_mu[np.newaxis, :]
        y_mco = y_mco[np.newaxis, :]
        
    n_sim, nT, nP =  y_mu.shape
    Pmin, Pmax = np.min(Pv), np.max(Pv)
    Tmin, Tmax = np.min(Tv), np.max(Tv)
    
    x = np.where(Pv == Pslide)[0]
    #print(x)
    # isobars for mu
    pdix_mu_v = np.reshape(y_mu[:, :, x], (n_sim, nT))
    mean_mu = np.mean(pdix_mu_v, axis = 0)
    min_mu = np.min(pdix_mu_v, axis = 0)
    max_mu = np.max(pdix_mu_v, axis = 0)
    

    # set x tick locations
    n_tick_sections = 4
        
    fig1,ax = plt.subplots(figsize=(5, 5))
    ax.plot(Tv, mean_mu, 'black', label = 'Mean')
    ax.plot(Tv, min_mu, 'steelblue')
    ax.plot(Tv, max_mu, 'steelblue', linewidth = 0.8)
    ax.fill_between(Tv, min_mu, max_mu, color = 'steelblue', alpha = 0.3, label = 'Range')
    
    ax.set_xlim(Tmin, Tmax)
    ax.set_xticks(np.linspace(Tmin, Tmax, n_tick_sections, dtype = int))
    ax.set_ylim(-4, 0)
    ax.set_xlabel('Temperature (K)')
    ax.set_ylabel(r'$\rm \overline{G}\ (eV)$')
    #ax.legend(bbox_to_anchor=(1.1, 1.1), loc='upper left')
    
    
    # isobars for mco
    pdix_mco_v = np.reshape(y_mco[:, :, x], (n_sim, nT))
    mean_mco = np.mean(pdix_mco_v, axis = 0)
    min_mco = np.min(pdix_mco_v, axis = 0)
    max_mco = np.max(pdix_mco_v, axis = 0)
    
    fig2,ax = plt.subplots(figsize=(5, 5))
    ax.plot(Tv, mean_mco, 'black', label = 'Mean')
    ax.plot(Tv, min_mco, 'steelblue')
    ax.plot(Tv, max_mco, 'steelblue', linewidth = 0.8)
    ax.fill_between(Tv, min_mco, max_mco, color = 'steelblue', alpha = 0.3, label = 'Range')
    
    ax.set_xlim(Tmin, Tmax)
    ax.set_xticks(np.linspace(Tmin, Tmax, n_tick_sections, dtype = int))
    ax.set_ylim(0, 2)
    ax.set_xlabel('Temperature (K)')
    ax.set_ylabel(r'$\rm \overline{m}_{CO}\ (eV)$')
    #ax.legend(bbox_to_anchor=(1.1, 1.1), loc='upper left')
    plt.show()
    
    return fig1, fig2
    

def plot_isotherm_spread(Tslide, Pv, Tv, y_mu, y_mco, iso_names):
    """Plot 1D isobars at the chosen Pslide value"""
    y_mu = y_mu.copy()
    y_mco = y_mco.copy() 
    if len(y_mco.shape) == 2: # add one more dimension if only two provided
        y_mu = y_mu[np.newaxis, :]
        y_mco = y_mco[np.newaxis, :]
        
    n_sim, nT, nP =  y_mu.shape
    Pmin, Pmax = np.min(Pv), np.max(Pv)
    Tmin, Tmax = np.min(Tv), np.max(Tv)
    
    x = np.where(np.abs(Tv - Tslide) <= 0.1)[0]
    #print(x)

    # isotherms for mu
    pdix_mu_v = np.reshape(y_mu[:, x, :], (n_sim, nP))
    mean_mu = np.mean(pdix_mu_v, axis = 0)
    min_mu = np.min(pdix_mu_v, axis = 0)
    max_mu = np.max(pdix_mu_v, axis = 0)
    
    # set x tick locations
    n_tick_sections = 4
    
    fig1,ax = plt.subplots(figsize=(5, 5))
    
    ax.plot(Pv, mean_mu, 'black', label = 'Mean')
    ax.plot(Pv, min_mu, 'steelblue')
    ax.plot(Pv, max_mu, 'steelblue', linewidth = 0.8)
    ax.fill_between(Pv, min_mu, max_mu, color = 'steelblue', alpha = 0.3, label = 'Range')
    
    ax.set_xlim(Pmin, Pmax)
    ax.set_ylim(-4, 0)
    ax.set_xscale('log')
    ax.set_xticks([10** int(i) for i in np.linspace(np.log10(Pmin), np.log10(Pmax), n_tick_sections, dtype = int)])
    ax.set_xlabel(r'$\rm P_{CO}\ (bar)$')
    ax.set_ylabel(r'$\rm \overline{G}\ (eV)$')
    #ax.legend(bbox_to_anchor=(1.1, 1.1), loc='upper left')
    
    # isotherms for mco
    pdix_mco_v = np.reshape(y_mco[:, x, :], (n_sim, nP))
    mean_mco = np.mean(pdix_mco_v, axis = 0)
    min_mco = np.min(pdix_mco_v, axis = 0)
    max_mco = np.max(pdix_mco_v, axis = 0)
    
    fig2,ax = plt.subplots(figsize=(5, 5))
    
    ax.plot(Pv, mean_mco, 'black', label = 'Mean')
    ax.plot(Pv, min_mco, 'steelblue')
    ax.plot(Pv, max_mco, 'steelblue', linewidth = 0.8)
    ax.fill_between(Pv, min_mco, max_mco, color = 'steelblue', alpha = 0.3, label = 'Range')
    
    
    ax.set_xlim(Pmin, Pmax)
    ax.set_ylim(0, 2)
    ax.set_xscale('log')
    ax.set_xticks([10** int(i) for i in np.linspace(np.log10(Pmin), np.log10(Pmax), n_tick_sections, dtype = int)])
    ax.set_xlabel(r'$\rm P_{CO}\ (bar)$')
    ax.set_ylabel(r'$\rm \overline{m}_{CO}\ (eV)$')
    #ax.legend(bbox_to_anchor=(1.1, 1.1), loc='upper left')
    plt.show()
    
    return fig1, fig2



def cal_stats_T_slide(y, slide):
    """calculate the stats for 2d dimensional matrix at a slice of 3D matrix"""
    y = y.copy()
    if len(y.shape) == 2: # add one more dimension if only two provided
        y = y[np.newaxis, :]
        
    n_sim, nT, nP =  y.shape
    #print(x)

    # isotherms for mu
    pdix_y_v = np.reshape(y[:, slide, :], (n_sim, nP))
    mean_y = np.mean(pdix_y_v, axis = 0)
    min_y = np.min(pdix_y_v, axis = 0)
    max_y = np.max(pdix_y_v, axis = 0)
    
    return mean_y, min_y, max_y
    

def cal_stats_P_slide(y, slide):
    """calculate the stats for 2d dimensional matrix at a slice of 3D matrix"""
    y = y.copy()
    if len(y.shape) == 2: # add one more dimension if only two provided
        y = y[np.newaxis, :]
        
    n_sim, nT, nP =  y.shape
    #print(x)
    # isotherms for mu
    pdix_y_v = np.reshape(y[:, :, slide], (n_sim, nT))
    mean_y = np.mean(pdix_y_v, axis = 0)
    min_y = np.min(pdix_y_v, axis = 0)
    max_y = np.max(pdix_y_v, axis = 0)
    
    return mean_y, min_y, max_y    

def cal_min_sim(y): 
    """Calculate the minimal value given multiple trajectories from different isomers"""
    y = y.copy()
    if len(y.shape) == 2: # add one more dimension if only two provided
        y = y[np.newaxis, :]
    n_sim, nT, nP =  y.shape
    
    y_min_sim = np.min(y, axis = 0)
    
    return y_min_sim
    
def get_min_index(y): 
    """Calculate the minimal value given multiple trajectories from different isomers
    return the isomer index"""
    y = y.copy()
    if len(y.shape) == 2: # add one more dimension if only two provided
        y = y[np.newaxis, :]
    n_sim, nT, nP =  y.shape
    
    y_min_index = np.argmin(y, axis = 0)
    
    return y_min_index    

def plot_isotherm_multi(Tslide, Pv, Tv, y_mu_list, y_mco_list, y_mu_min, iso_names, plot_spread = True):
    """Plot 1D isobars at the chosen Pslide value"""
    
    Pmin, Pmax = np.min(Pv), np.max(Pv)
    Tmin, Tmax = np.min(Tv), np.max(Tv)
    
    x = np.where(np.abs(Tv - Tslide) <= 0.1)[0]

    color_list = cm.jet(np.linspace(0, 1, len(y_mu_list)))
    # isotherms for mu
    fig1,ax = plt.subplots(figsize=(5, 5))
    
    for y_mu_i, color_i, name_i in zip(y_mu_list, color_list, iso_names):
        mean_mu, min_mu, max_mu = cal_stats_T_slide(y_mu_i, x)
        ax.plot(Pv, min_mu, color = color_i, linewidth = 0.8, label = name_i)
        #ax.plot(Pv, mean_mu, color = color_i)
        #ax.plot(Pv, max_mu, color = color_i, linewidth = 0.8)
        if plot_spread: ax.fill_between(Pv, min_mu, max_mu, color = color_i, alpha = 0.3) #, label = name_i)
    
    # plot the hull
    _, hull_mu, _ = cal_stats_T_slide(y_mu_min, x)
    ax.plot(Pv, hull_mu, 'k--', linewidth = 2)
    # set x tick locations
    n_tick_sections = 4

    ax.set_xlim(Pmin, Pmax)
    ax.set_ylim(-4, 0)
    ax.set_xscale('log')
    ax.set_xticks([10** int(i) for i in np.linspace(np.log10(Pmin), np.log10(Pmax), n_tick_sections, dtype = int)])
    ax.set_xlabel(r'$\rm P_{CO}\ (bar)$')
    ax.set_ylabel(r'$\rm \overline{G}\ (eV)$')
    if plot_legend: ax.legend(bbox_to_anchor=(1.1, 1.1), loc='upper left')
    #ax.legend(loc='upper center', fontsize=14, bbox_to_anchor=(0.5, -0.2), ncol = 7) #loc='upper left', bbox_to_anchor=(1, 1)
    
    # isotherms for mco
    fig2,ax = plt.subplots(figsize=(5, 5))
    
    for y_mco_i, color_i, name_i in zip(y_mco_list, color_list, iso_names):
        mean_mco, min_mco, max_mco = cal_stats_T_slide(y_mco_i, x)
        ax.plot(Pv, min_mco, color = color_i,linewidth = 0.8, label = name_i)
        #ax.plot(Pv, mean_mco, color = color_i)
        #ax.plot(Pv, max_mco, color = color_i, linewidth = 0.8)
        if plot_spread: ax.fill_between(Pv, min_mco, max_mco, color = color_i, alpha = 0.3) #, label = name_i)
    
    
    ax.set_xlim(Pmin, Pmax)
    ax.set_ylim(0, 2)
    ax.set_xscale('log')
    ax.set_xticks([10** int(i) for i in np.linspace(np.log10(Pmin), np.log10(Pmax), n_tick_sections, dtype = int)])
    ax.set_xlabel(r'$\rm P_{CO}\ (bar)$')
    ax.set_ylabel(r'$\rm \overline{m}_{CO}\ (eV)$')
    if plot_legend: ax.legend(bbox_to_anchor=(1.1, 1.1), loc='upper left')
    #ax.legend(loc='upper center', fontsize=14, bbox_to_anchor=(0.5, -0.2), ncol = 7) #loc='upper left', bbox_to_anchor=(1, 1)

    plt.show()
    
    return fig1, fig2


def plot_isobar_multi(Pslide, Pv, Tv, y_mu_list, y_mco_list, y_mu_min, iso_names, plot_spread = True):
    """Plot 1D isobars at the chosen Pslide value"""
    
    Pmin, Pmax = np.min(Pv), np.max(Pv)
    Tmin, Tmax = np.min(Tv), np.max(Tv)
    
    x = np.where(Pv == Pslide)[0]
    
    color_list = cm.jet(np.linspace(0, 1, len(y_mu_list)))
    # isotherms for mu
    fig1,ax = plt.subplots(figsize=(5, 5))
    
    for y_mu_i, color_i, name_i in zip(y_mu_list, color_list, iso_names):
        mean_mu, min_mu, max_mu = cal_stats_P_slide(y_mu_i, x)
        ax.plot(Tv, min_mu, color = color_i, linewidth = 0.8, label = name_i)
        #ax.plot(Tv, max_mu, color = color_i, linewidth = 0.8)
        #ax.plot(Pv, mean_mu, color = color_i)
        if plot_spread: ax.fill_between(Tv, min_mu, max_mu, color = color_i, alpha = 0.3) #, label = name_i)
    
    # plot the hull
    _, hull_mu, _ = cal_stats_P_slide(y_mu_min, x)
    ax.plot(Tv, hull_mu, 'k--', linewidth = 2)
    
    # set x tick locations
    n_tick_sections = 4

    ax.set_xlim(Tmin, Tmax)
    ax.set_xticks(np.linspace(Tmin, Tmax, n_tick_sections, dtype = int))
    ax.set_ylim(-4, 0)
    ax.set_xlabel('Temperature (K)')
    ax.set_ylabel(r'$\rm \overline{G}\ (eV)$')
    if plot_legend: ax.legend(bbox_to_anchor=(1.1, 1.1), loc='upper left')
    #ax.legend(loc='upper center', fontsize=14, bbox_to_anchor=(0.5, -0.2), ncol = 7) #loc='upper left', bbox_to_anchor=(1, 1)
    
    # isotherms for mco
    fig2,ax = plt.subplots(figsize=(5, 5))
    
    for y_mco_i, color_i, name_i in zip(y_mco_list, color_list, iso_names):
        mean_mco, min_mco, max_mco = cal_stats_P_slide(y_mco_i, x)
        ax.plot(Tv, min_mco, color = color_i, linewidth = 0.8, label = name_i)
        #ax.plot(Tv, max_mco, color = color_i, linewidth = 0.8)
        #ax.plot(Tv, mean_mco, color = color_i)
        if plot_spread: ax.fill_between(Tv, min_mco, max_mco, color = color_i, alpha = 0.3) #, label = name_i)
    
    
    ax.set_xlim(Tmin, Tmax)
    ax.set_xticks(np.linspace(Tmin, Tmax, n_tick_sections, dtype = int))
    ax.set_ylim(0, 2)
    ax.set_xlabel('Temperature (K)')
    ax.set_ylabel(r'$\rm \overline{m}_{CO}\ (eV)$')
    if plot_legend: ax.legend(bbox_to_anchor=(1.1, 1.1), loc='upper left')
    #ax.legend(loc='upper center', fontsize=14, bbox_to_anchor=(0.5, -0.2), ncol = 7) #loc='upper left', bbox_to_anchor=(1, 1)

    plt.show()
    
    return fig1, fig2
