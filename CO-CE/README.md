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
See `binding_fuctions.py` for details

- Predict CO adsorption energies a Pd20 structures 
```

# Given a CONTCAR file name
bare_cluster_path = os.path.join(ProjectPath, 'dataset', 'DFT_structures', 'bare')
pd20_atoms = read(os.path.join(bare_cluster_path, 'pd20-no-CO-CONTCAR'))
atoms = read(filename)

COsites = find_all_surface_sites(atoms)
Pd_interest = [107, 96, 98, 111, 106, 99, 110, 113, 112, 115]#[x for x in Pd_indices if x not in Pd_no_interest]
COsites = find_sites(Pd_interest, atoms)

spca = pca_model('spca')
spca.predict_binding_E(atoms, COsites)
y_bind = spca.y_bind

# make it possible for drawing_Pd20 to plot
# the tidy difference deviates from the symmtery might caused by relaxed cooridnates from a CONTCAR structure
y_pred = y_bind.copy()
sitetype_list = spca.sitetype_list
```

or one line function

```
y_pred, COsites, CO_pos, sitetype_list, GCNs, CN1s, CN2s, ratio_surface =  predict_binding_Es_fast(pd20_atoms, 
																					[107, 96, 98, 111, 106, 99, 110, 113, 112, 115], 
																					view_flag = True, 
																					output_descriptor= True,
																					top_only= False)
```



