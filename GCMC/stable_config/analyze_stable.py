# -*- coding: utf-8 -*-
"""
Created on Tue Oct 20 23:14:25 2020

Descriptor analysis for all stabel configurations
"""
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt 
import matplotlib.patches as mpatches
from matplotlib.pyplot import cm
import matplotlib

font = {'size'   : 18}
matplotlib.rc('font', **font)
matplotlib.rcParams['axes.linewidth'] = 1
matplotlib.rcParams['xtick.major.size'] = 6
matplotlib.rcParams['xtick.major.width'] = 1
matplotlib.rcParams['ytick.major.size'] = 6
dpi = 300.
matplotlib.rcParams['figure.dpi'] = dpi

def set_axis_style(ax, labels):
    ax.get_xaxis().set_tick_params(direction='out')
    ax.xaxis.set_ticks_position('bottom')
    ax.set_xticks(np.arange(1, len(labels) + 1))
    ax.set_xticklabels(labels)
    ax.set_xlim(0.25, len(labels) + 0.75)
    
def set_color(violin_parts, c):
    for vi in violin_parts['bodies']: 
        vi.set_color(c) 
    violin_parts['cmeans'].set_color(c)
    violin_parts['cmaxes'].set_color(c)
    violin_parts['cmins'].set_color(c)
    violin_parts['cbars'].set_color(c)


# read the data of stable configurations
filename_stable = os.path.join(os.getcwd(),  'pdall_stable_config_all_conditions.csv')   
df_all = pd.read_csv(filename_stable, index_col=False)

# Preprocess the data
z_support = 14
# fill the nan value in the z position column
df_all = df_all.fillna(value = z_support)
# rescale the z position back 0
df_all['Z_pos'] = df_all['Z_pos'] - z_support

# Put PCO into log scale
df_all['PCO'] = np.log10(df_all['PCO'])

# take absolute value for mu
df_all['mu_mean'] = np.abs(df_all['mu_mean'])

#%%
from sklearn.decomposition import PCA 
from sklearn.preprocessing import StandardScaler 

descriptors= ['n', 'T', 'PCO', 'mu_mean','nCO_mean', 'GCN', 'CN1', 'CN2',  'total_top', 'total_bridge', 'total_hollow', 'total_sites', 'occ_top', 'occ_bridge','occ_hollow', 'Z_pos']
descriptors_rf = ['n', 'GCN', 'CN1', 'CN2',  'total_top', 'total_bridge', 'total_hollow', 'total_sites', 'occ_top', 'occ_bridge','occ_hollow', 'Z_pos']

# descriptors used in plot labels
descriptors_plot = ['n', 'T', r'$\rm P_{CO}$', r'$\rm \overline{|G|}$', r'$\rm \overline{m}_{CO}$', 
                    r'$\rm \overline{GCN}$', r'$\rm \overline{CN1}$', r'$\rm \overline{CN2}$', r'$\rm N_{top}$', r'$\rm N_{bridge}$', r'$\rm N_{hollow}$', 
                    r'$\rm N_{sites}$', r'$\rm \overline{m}_{CO-top}$',  r'$\rm \overline{m}_{CO-bridge}$', r'$\rm \overline{m}_{CO-hollow}$',  
                    r'$\rm \overline{Z}_{CO}$']

descriptors_rf_plot = ['n', r'$\rm \overline{GCN}$', r'$\rm \overline{CN1}$', r'$\rm \overline{CN2}$', r'$\rm N_{top}$', r'$\rm N_{bridge}$', r'$\rm N_{hollow}$', 
                    r'$\rm N_{sites}$', r'$\rm \overline{m}_{CO-top}$',  r'$\rm \overline{m}_{CO-bridge}$', r'$\rm \overline{m}_{CO-hollow}$',  
                    r'$\rm \overline{Z}_{CO}$']


nDescriptors = len(descriptors)

descriptors_rf_indices = [0] + list(range(5, nDescriptors))
X =  np.array(df_all.loc[:,descriptors], dtype = float)
X_rf =  np.array(df_all.loc[:,descriptors_rf], dtype = float)

y_mu = np.abs(df_all['mu_mean'])
y_mco = np.abs(df_all['nCO_mean'])

pc_reg = 3


#%% PCA 
# Normalize the data
#X_std = StandardScaler().fit_transform(X)
scaler = StandardScaler().fit(X)
X_std = scaler.transform(X)

# Covriance matrix of original data
cov_mat = np.cov(X_std.T) 

# PCA use sklearn
pca = PCA().fit(X_std)    
Xpc = pca.transform(X_std) 
# Covriance matrix from PCA
cov_pc = np.cov(Xpc.T) 


# Plot Covariance structure 
fig, ax1 = plt.subplots(figsize=(12,10))
ax1.set_xticks(range(0,nDescriptors))
ax1.set_xticklabels(descriptors_plot, rotation = 70)
ax1.set_yticks(range(0,nDescriptors))
ax1.set_yticklabels(descriptors_plot)
im1 = ax1.imshow(cov_mat, vmin = -1, vmax = 1)
fig.colorbar(im1, ax = ax1, shrink = 0.5)


#eig_vals, eig_vecs = np.linalg.eig(cov_mat)
eig_vals = pca.explained_variance_ #eigenvalues 
eig_vecs = pca.components_  # eigenvector
#tot = sum(eig_vals)
#var_exp = [(i / tot)*100 for i in sorted(eig_vals, reverse=True)]
var_exp = pca.explained_variance_ratio_ #explained variance ratio
cum_var_exp = np.cumsum(var_exp) #cumulative variance ratio
#print('Eigenvectors \n%s' %eig_vecs)
#print('\nEigenvalues \n%s' %eig_vals)

'''
Scree plot for PCs
'''
                      
plt.figure(figsize=(10, 10))

plt.bar(range(nDescriptors), var_exp, alpha=0.5, align='center',
        label='Individual Explained Variance')
plt.step(range(nDescriptors), cum_var_exp, where='mid',
         label='Cumulative Explained Variance')
plt.ylabel('Explained Variance Ratio')
plt.xlabel('Principal Components')
plt.xticks(np.arange(nDescriptors), 
            ['PC%i'%(w+1) for w in range(nDescriptors)],  rotation=60)
plt.legend(loc = 'right'	)
plt.tight_layout()

#%% Plot the normalized desciptor loading
'''
Need to be completed
'''
ind = 0
yvals = []
ylabels = []
bar_vals = []
space = 0.2

cm_cur =  cm.jet(np.linspace(0,1,nDescriptors))
fig = plt.figure(figsize=(6,6))


ax = fig.add_subplot(111)
n = len(descriptors)
width = (1 - space) / (len(descriptors))
indeces = np.arange(0, pc_reg) + 0.5  

PC_loadings = []
# Create a set of bars at each position
for i, pci in enumerate(eig_vecs[:pc_reg]):
    
    vals = pci/np.sum(np.absolute(pci))
    PC_loadings.append(vals)
    pos = width*np.arange(n) + i 
    ax.bar(pos, vals, width=width, label=str(i+1), color = cm_cur, alpha = 1) 
        
linex = np.arange(np.arange(0, pc_reg).min() -0.1  , np.arange(0, pc_reg).max()+ 1)

ax.set_xticks(indeces) 
ax.set_xticklabels(list(np.arange(0, pc_reg)+1))
ax.set_ylabel("Normalized Descriptoor Loading")
ax.set_xlabel("Principal Component (PC) #")    
  
# Add legend using color patches
patches = []
for c in range(n):
    patches.append(mpatches.Patch(color=cm_cur[c]))
plt.legend(patches, descriptors, 
           bbox_to_anchor = (1.02, 1),loc= 'upper left', frameon=False)

plt.plot(linex, linex*0, c = 'k', lw = 1.5)
plt.show()

#%% PC1, PC2 2D PC loading plot
plt.figure(figsize=(6, 6))
for i in range(n):
    xl = PC_loadings[0][i]
    yl = PC_loadings[1][i]
    plt.scatter(xl, yl,  c = cm_cur[i], s = 120)
    plt.plot(np.array([0, xl]), np.array([0, yl]), '--', c = cm_cur[i], label = descriptors[i])
plt.xlabel('PC1 Loading')
plt.ylabel('PC2 Loading')

# Add legend using color patches
patches = []
for c in range(n):
    patches.append(mpatches.Patch(color=cm_cur[c]))
plt.legend(patches, descriptors, 
           bbox_to_anchor = (1.02, 1),loc= 'upper left', frameon=False)
plt.show()

#%%
# Helper fuction for loading plots
def plot_loadings(loadings, axes=None, pc=(0,1), varlabels=[],
                  scale=False, **kwargs):
    xl = loadings[:,pc[0]]
    yl = loadings[:,pc[1]]
    if scale:
        xl = (xl-xl.mean())/(xl.max() - xl.min())
        yl = (yl-yl.mean())/(yl.max() - yl.min())
    
    if axes:
        fig = None
        ax = axes
    else:
        fig = plt.figure(**kwargs)
        ax = fig.add_subplot(111)
    
    for i, (x, y) in enumerate(zip(xl,yl)):
        ax.plot([0, x],[0, y],  '--', c = cm_cur[i])
        ax.scatter(x, y,  c = cm_cur[i], s = 100)
    
    # if list(varlabels):
    #     for i, txt in enumerate(varlabels):
    #         ax.annotate(txt, xy=(xl[i]*0.95, yl[i]*1.1), fontsize = 16)
    
    xlim = abs(max(xl.max(),xl.min(),key=abs))*1.1
    ylim = abs(max(yl.max(),yl.min(),key=abs))*1.1
    ax.set_xlim(-0.5,0.5)
    ax.set_ylim(-0.5,0.5)
    
    ax.set_xlabel("Principle Component {}".format(pc[0]+1))
    ax.set_ylabel("Principle Component {}".format(pc[1]+1))

    return fig, ax

pca = PCA()
scores = pca.fit_transform(X_std)
loadings = np.transpose(pca.components_)
plot_loadings(
    loadings=loadings,
    pc=(0,1),
    figsize=(8,8),
    varlabels=descriptors_plot
)

plot_loadings(
    loadings=loadings,
    pc=(0,2),
    figsize=(8,8),
    varlabels=descriptors_plot
)


#%%
# random forest for feature importance on a regression problem
from sklearn.datasets import make_regression
from sklearn.ensemble import RandomForestRegressor
import time 
from sklearn.model_selection import RepeatedKFold, cross_validate, LeaveOneOut, train_test_split
from sklearn.metrics import r2_score, mean_squared_error
import warnings 
from sklearn.exceptions import UndefinedMetricWarning


# Turn off warnings for calculating r2 of a single point
warnings.filterwarnings("ignore", category=UndefinedMetricWarning)

def plot_importances(estimator, descriptors):
    '''
    Plot feature importance reported by random forest
    '''
    fig,ax = plt.subplots(figsize=(8,8))
    n_descriptor = len(descriptors)
    importance =  estimator.feature_importances_
    importance_sorted = np.sort(importance)
    descirptors_sorted = [x for _,x in sorted(zip(importance,descriptors))]
    
    ax.barh(np.arange(n_descriptor),importance_sorted, color= cm_cur[descriptors_rf_indices])
    ax.set_yticks(np.arange(n_descriptor))
    ax.set_yticklabels(descirptors_sorted, rotation=0)
    #ax.set_ylabel('Descriptor')
    ax.set_xlabel('Importance')
    plt.show()
    
    
    
def cross_validation(X, y, estimator): 
    '''
    Cross-validation
    '''
    #rkf = RepeatedKFold(n_splits = 10, n_repeats = 3) #change this to leave one out
    loo = LeaveOneOut()
    scores  = cross_validate(estimator, X, y, cv = loo,
                                scoring=('neg_mean_squared_error', 'r2'),
                                return_train_score=True)
    # RMSE for repeated 10 fold test data 
    test_RMSE = np.sqrt(np.abs(scores['test_neg_mean_squared_error'])) 
    test_RMSE_mean = np.mean(test_RMSE)
    test_RMSE_std = np.std(test_RMSE)
    
    train_r2 = scores['train_r2'] 
    train_r2_mean =  np.mean(train_r2)
    train_r2_std = np.std(train_r2)
    
    return [test_RMSE_mean, test_RMSE_std, train_r2_mean, train_r2_std]



X_train_mu, X_test_mu, y_train_mu, y_test_mu = train_test_split(X_rf, y_mu, test_size=0.1, random_state=0)
X_train_mco, X_test_mco, y_train_mco, y_test_mco = train_test_split(X_rf, y_mco, test_size=0.1, random_state=0)


# Train the random forest model using 2D Grid search
train_flag = False

if train_flag:
    # grid search for two hyperparameters 
    n_estimators_grid = range(1,1001, 500)
    max_depth_grid = range(1,51, 25)
    test_RMSE_m = np.zeros((len(n_estimators_grid), len(max_depth_grid))) 
    progress = 0
    
    start_time = time.time()
    for i, n_estimators_i in enumerate(n_estimators_grid):
        for j, max_depth_i in enumerate(max_depth_grid):
            progress += 1
            per = progress/len(n_estimators_grid)/len(max_depth_grid) *100
            print('Training {0:.5f} % Done!'.format(per))
            
            rf = RandomForestRegressor(n_estimators = n_estimators_i,
                                         max_depth = max_depth_i,
                                         random_state=0)
    
            [test_RMSE_mean, test_RMSE_std, train_r2_mean, train_r2_std] = cross_validation(X_train_mu, y_train_mu, rf)
            test_RMSE_m[i, j] = test_RMSE_mean
    
    end_time = time.time()
    
    print("Tuning hyperparameter takes {0:.2f} minutes".format((end_time-start_time)/60.0))

    # Final the optimal model
    test_RMSE_m = np.around(test_RMSE_m, decimals = 3)
    opt_ind = np.unravel_index(np.argmin(test_RMSE_m, axis=None), test_RMSE_m.shape)
    n_estimators_opt = n_estimators_grid[opt_ind[0]]
    max_depth_opt = max_depth_grid[opt_ind[1]]
    
else: 
    n_estimators_opt = 500
    max_depth_opt = 20

print('The optimal number of estimator is: {}'.format(n_estimators_opt))
print('The optimal number of max depth is: {}'.format(max_depth_opt))


#%% The final optimal model for mu    
rf_opt_mu = RandomForestRegressor(n_estimators = n_estimators_opt,
                                     max_depth = max_depth_opt,
                                     random_state=0)
rf_opt_mu.fit(X_train_mu, y_train_mu)

#%%
# Error on test data
y_predict_test_mu = rf_opt_mu.predict(X_test_mu)
RMSE_test_mu = np.sqrt(mean_squared_error(y_test_mu, y_predict_test_mu))

# Error on all data
y_predict_mu = rf_opt_mu.predict(X_rf)

#%%
plot_importances(rf_opt_mu, descriptors_rf_plot)

# #%% The final optimal model for mco    
# rf_opt_mco = RandomForestRegressor(n_estimators = n_estimators_opt,
#                                      max_depth = max_depth_opt,
#                                      random_state=0)
# rf_opt_mco.fit(X_train_mco, y_train_mco)

# y_predict_test_mco = rf_opt_mco.predict(X_test_mco)
# RMSE_test_mco = np.sqrt(mean_squared_error(y_test_mco, y_predict_test_mco))


# # Error on all data
# y_predict_mco = rf_opt_mco.predict(X_rf)
# plot_importances(rf_opt_mco, descriptors_rf)

# plot the actual data and prediction

fig, ax = plt.subplots(figsize=(6, 6))
X0_min, X0_max = 0, 4
X_plot = np.linspace(X0_min, X0_max, 50)
ax.scatter(y_mu, y_predict_mu, s = 20, alpha = 0.5)
ax.plot(X_plot, X_plot, color = 'r')
ax.set_xlabel(r'$\rm \overline{|G|}$')
ax.set_ylabel(r'$\rm Model\ Predicted\ \overline{|G|}$')

