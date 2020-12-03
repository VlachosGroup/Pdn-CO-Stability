# Grand Canonical Monte Carlo (GCMC) Simulations 

### Flow chart and the type of Monte Carlo moves
![Flowchart](CGMC_flowchart.svg)

## Usage 

GCMC is performed by executing `run_CGMC_single.py` for a smaller cluster (size smaller than 6)

Or by executing `run_CGMC_single_GA.py` for a larger cluster 

### Simulation parameters 
- pdx, initial structure in one-hot encoding
- index, isomer index (0 being the lowest energy for the bare clusters)
- T, temperature
- PCO, CO partial pressure 
- rseed, random seed for the simulation trajectory
- nsteps, the number of Monte Carlo moves 

### Output files describing a GCMC trajectory
- `results/pdx/i(index)/run_pdx_(index)_(T)k_(PCO)bar_(rseed)/GCMC_trajectory`, the record for each GCMC trajectory
- Include Pdn configuration, CO configuration, free energy (mu) and the number of CO (nCO)


### Descriptor data
- `equilibrium_config\pd_all_config_all_conditions.csv` descriptor values for all equilibrated structures (n = 1-21)
- `equilibrium_config_large\pd_large_config_300K.csv` descriptor values for all equilibrated structures (n = 25, 30, 38, 55)
