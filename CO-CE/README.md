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
See `test_model_usage_co.py` for details


- Import session
```
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
```

- Predict single CO adsorption energies a Pd20 structures 
```
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
```
Longer version of the code
```
COsites = binding.find_sites(Pd_interest, pd20_atoms)
#specific the ML model
spca = binding.pca_model('spca')
spca.predict_binding_E(pd20_atoms, COsites)
y_bind = spca.y_bind
y_pred = y_bind.copy()
sitetype_list = spca.sitetype_list
```



- Predict the interactions and total energy for the adlayer
```
# Assume some COs are occupied
co_config = [0, 1, 2, 36, 35]
# Use the interactions class
interactions = binding.CO_interactions(CO_pos, sitetype_list, co_config)
# calculate the lateral interactions
total_interactions = interactions.cal_interactions()
# calculate the total adlayer enegry
total_E = interactions.cal_binding_Es_total(binding_Es)
```

- Visualize structures
```
PdnCOm_obj = binding.PdnCOm(pd20_atoms, Pd_interest)
PdnCOm_atoms = PdnCOm_obj.append_COs(co_config, view_flag = True)
```

