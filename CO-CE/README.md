# CO cluster expansion (CE)

Two machine learning models are used to predict the energy of CO adsorbate layer

- A random forest model (RF) for Single CO adsorption energy 
- A ordinary least square (OLS) regression for CO lateral interactions 

## Training 
Model training is performed by `train_rf_model.py` and `train_interactions.py`  

Run `train_rf_model.py` would generate a model pickle file, 'rf_estimator.p' 

Run `train_interactions.py`  would save the interaction values in a json file, 'co_interactions_new.json'

Both files can be used directly for prediction


## Usage 
See `test_model_usage.py` for details

- Predict the energy for structures in configuration dataset on a 5 by 5 lattice (mother)
```
from set_config_constants import config
from set_ce_lattice import mother
import energy_functions as energy
from set_ce_lattice import dz
from generate_clusters_super_cell import super_mother

# Load energy object
Pdn = energy.Pdn(lasso_file, mother=mother, super_cell_flag=False)

# select a Pd single atom from configuration dataset
config_SA = config[0][0]

# Predict energy for a single atom
E_pred_SA, _  = Pdn.predict_E(config_SA)

# Visualize the atomic configuration
atoms_SA = energy.append_support(config_SA, mother, view_flag=True)
```

- Generate a random structure and calculate its energy on a 10 by 10 lattice (super_mother) 
```
from set_config_constants import config
from set_ce_lattice import mother
import energy_functions as energy
from set_ce_lattice import dz
from generate_clusters_super_cell import super_mother

# Load energy object
Pdn_super = energy.Pdn(lasso_file, mother=super_mother, super_cell_flag=True)

# Single atoms to put on the base layer
n_seeds = 20

# the index for base layer atoms in super cell
base_indices = np.where(super_mother[:,2] == dz)[0]
base_occ_indices = np.unique(np.random.choice(base_indices, n_seeds, replace = False))


# Initialize the individual configuration in one hot encoding
rnd_individual = np.zeros(len(super_mother),dtype = int)
rnd_individual[base_occ_indices] = 1


# Predict energy for initial configuration
config_rnd = energy.one_hot_to_index(rnd_individual)
E_pred_rnd, _  = Pdn_super.predict_E(config_rnd)

# Visualize the atomic configuration
atoms_rnd = energy.append_support(config_rnd, super_mother, view_flag=True)
```

