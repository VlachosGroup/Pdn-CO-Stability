# -*- coding: utf-8 -*-
"""
Created on Tue Jan 28 16:23:28 2020

@author: Yifan Wang
"""

"""
Test file for random forest regression
"""

from sklearn.decomposition import PCA 
from sklearn.preprocessing import StandardScaler 
from sklearn import linear_model 
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.model_selection import RepeatedKFold, cross_validate, LeaveOneOut, train_test_split


import os
import pandas as pd
import numpy as np
import pickle

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt 
import matplotlib
from matplotlib.patches import Ellipse

import time

matplotlib.rcParams['font.size'] = 16
matplotlib.rcParams['axes.linewidth'] = 1.5
matplotlib.rcParams['xtick.major.size'] = 8
matplotlib.rcParams['xtick.major.width'] = 2
matplotlib.rcParams['ytick.major.size'] = 8
matplotlib.rcParams['ytick.major.width'] = 2
dpi = 300.
matplotlib.rcParams['figure.dpi'] = dpi

HomePath = os.path.expanduser('~')
ProjectPath = os.path.join(HomePath, 'Documents', 'GitHub', 'Pdn-CO-Stability')


# CO binding model directory
binding_path = os.path.join(ProjectPath, 'CO-CE')
# Energy model directory
energy_path = os.path.join(ProjectPath, 'Pdn-CE')
# data directory
data_path = os.path.join(ProjectPath, 'dataset')

from matplotlib import cm
viridis = cm.get_cmap('viridis', 5)
#cm = ['r', 'coral', 'pink',  'orange',  'gold', 'y','lightgreen', 'lightblue',  'c', 'mediumpurple', 'brown', 'grey', 'orchid']

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


def parity_plot_st(yobj, ypred, RMSE_test, method):
    '''
    Plot the parity plot of y vs ypred
    return R2 score and MSE for the model
    colorcode different site types
    '''
    RMSE_all = np.sqrt(np.mean((yobj - ypred)**2))
    r2_all = r2_score(yobj, ypred)
    fig, ax = plt.subplots(figsize = (6,6))
    for site, col in zip(('top', 'bridge', 'hollow'),
                    ('red', 'green', 'blue')):
        indices = np.where(np.array(sitetype_list) == site)[0]
        ax.scatter(yobj[indices],
                    ypred[indices],
                    label=site,
                    facecolor = col, 
                    alpha = 0.5,
                    s  = 60)
    ax.plot([yobj.min(), yobj.max()], [yobj.min(), yobj.max()], 'k--', lw=2)
    
    ax.set_xlabel('DFT-Calculated ')
    ax.set_ylabel('Model Prediction')
    plt.title(r'{}, RMSE-{:.2}, $r^2$ -{:.2}'.format(method, RMSE_test, r2_all))
    plt.legend(bbox_to_anchor = (1.02, 1),loc= 'upper left', frameon=False)
    plt.show()
    
    return RMSE_all, r2_all



def plot_importances(estimator, descriptors):
    '''
    Plot feature importance reported by random forest
    '''
    fig,ax = plt.subplots(figsize=(6,6))
    n_descriptor = len(descriptors)
    importance =  estimator.feature_importances_
    importance_sorted = np.sort(importance)
    descirptors_sorted = [x for _,x in sorted(zip(importance,descriptors))]
    
    ax.barh(np.arange(n_descriptor),importance_sorted, color=viridis.colors)
    ax.set_yticks(np.arange(n_descriptor))
    ax.set_yticklabels(descirptors_sorted, rotation=90)
    #ax.set_ylabel('Descriptor')
    ax.set_xlabel('Importance')
    plt.show()
    
    
#%% Processs descriptor_data.cvs
fdata = pd.read_csv(os.path.join(data_path, 'descriptors', 'descriptor_data.csv'))

#possible descriptors
descriptors_all =  ['NPd', 'CN1', 'CN2','GCN', 'Z', 'Charge', 'Nsites', 'Pd1C', 'Pd2C', 'Pd3C', 'CeCN1', 'OCN1'] #10 in total
descriptors_g =  ['NPd', 'Nsites', 'Z', 'CN1', 'CN2', 'GCN'] #6 geometric descriptors
descriptors_g5 =  ['NPd', 'Nsites', 'Z',  'CN1', 'CN2'] #['NPd',  'CN1', 'CN2',  'Z', 'Nsites' ] #5 geometric descriptors
descriptor_plot_g = ['n', r'$\rm S_{type}$', 'Z', 'CN1', 'CN2', 'GCN'] # strings used for plotting
descriptor_plot_g5 = ['n', r'$\rm S_{type}$', 'Z', 'CN1', 'CN2'] # strings used for plotting

Xg =  np.array(fdata.loc[:,descriptors_g], dtype = float)

# standardize the data 
scaler = StandardScaler().fit(Xg)
X_std_g = scaler.transform(Xg)

# Covriance matrix of original data, remove GCN 
cov_mat = np.abs(np.cov(X_std_g.T))

# Plot Covariance structure 
fig,ax = plt.subplots(figsize=(6,6))
#ax.set_title('X')
ax.set_xticks(range(0,len(descriptors_g)))
ax.set_xticklabels(descriptor_plot_g, rotation = 0)
ax.set_yticks(range(0,len(descriptors_g)))
ax.set_yticklabels(descriptor_plot_g)
im1 = ax.imshow(cov_mat)
fig.colorbar(im1, ax = ax, shrink = 0.8)


#%% Update and X and y for regression
X_reg = np.array(fdata.loc[:,descriptors_g5], dtype = float)
scaler = StandardScaler().fit(X_reg)
X_std = scaler.transform(X_reg)
X = X_std
nDescriptors = X.shape[1]
Eads = np.array(fdata.loc[:,'Eads'], dtype = float)
filename_list = list(fdata.loc[:,'Filename'])
sitetype_list = list(fdata.loc[:,'SiteType'])
y = Eads
# Count the number of sites 
ntop = (np.array(sitetype_list) == 'top').astype(int).sum()
nbridge = (np.array(sitetype_list) == 'bridge').astype(int).sum()
nhollow = (np.array(sitetype_list) == 'hollow').astype(int).sum()

# show one example -Pd10 hollow
fdata.loc[1]
nDescriptors = X.shape[1]


#%% Implement random forest
from sklearn.ensemble import RandomForestRegressor
import warnings 
from sklearn.exceptions import UndefinedMetricWarning


# Turn off warnings for calculating r2 of a single point
warnings.filterwarnings("ignore", category=UndefinedMetricWarning)
    
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=0)

# Train the random forest model using 2D Grid search
train_flag = False

if train_flag:
    # grid search for two hyperparameters 
    n_estimators_grid = range(1,51, 1)
    max_depth_grid = range(1,31, 1)
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
    
            [test_RMSE_mean, test_RMSE_std, train_r2_mean, train_r2_std] = cross_validation(X_train, y_train, rf)
            test_RMSE_m[i, j] = test_RMSE_mean
    
    end_time = time.time()
    
    print("Tuning hyperparameter takes {0:.2f} minutes".format((end_time-start_time)/60.0))

    # Final the optimal model
    test_RMSE_m = np.around(test_RMSE_m, decimals = 3)
    opt_ind = np.unravel_index(np.argmin(test_RMSE_m, axis=None), test_RMSE_m.shape)
    n_estimators_opt = n_estimators_grid[opt_ind[0]]
    max_depth_opt = max_depth_grid[opt_ind[1]]

else: 
    n_estimators_opt = 50
    max_depth_opt = 6

print('The optimal number of estimator is: {}'.format(n_estimators_opt))
print('The optimal number of max depth is: {}'.format(max_depth_opt))

#%% The final optimal model accessment    
rf_opt = RandomForestRegressor(n_estimators = n_estimators_opt,
                                     max_depth = max_depth_opt,
                                     random_state=0)
rf_opt.fit(X_train, y_train)
# validation errors for the optimal model
[cv_RMSE_mean, cv_RMSE_std, _, _] = cross_validation(X_train, y_train, rf_opt)

# Error on test data
y_predict_test = rf_opt.predict(X_test)
RMSE_test = np.sqrt(mean_squared_error(y_test, y_predict_test))

# Error on all data
y_predict = rf_opt.predict(X)
RMSE_all, r2_all = parity_plot_st(y, y_predict, RMSE_test, method = 'Random Forest')
plot_importances(rf_opt, descriptor_plot_g5)


# Pickle the models 
import pickle
pickle.dump([rf_opt, scaler], open('rf_estimator.p','wb'))


#%% Seaborn lasso join plot
import matplotlib.pyplot as plt 
import seaborn as sns;   
dpi = 300.
'''
Warning: the marginal axis labels do not work in matplotlib 3.1.0 but work in version 2.8.1
Consider downgrading for plotting jointplot
'''

y_predict_all = rf_opt.predict(X)
model_name = 'rf'


lims = [
    np.min([y.min(), y_predict_all.min()]),  # min of both axes
    -1.0 #np.max([y.max(), y_predict_all.max()]),  # max of both axes
]


top_indices = np.where(np.array(sitetype_list) == 'top')[0]
bridge_indices = np.where(np.array(sitetype_list) == 'bridge')[0]
hollow_indices = np.where(np.array(sitetype_list) == 'hollow')[0]


df1 = pd.DataFrame(np.array([y[top_indices], y_predict_all[top_indices]]).T, columns=["DFT Cluster Energy (eV)","Predicted Cluster Energy (eV)"])
df2 = pd.DataFrame(np.array([y[bridge_indices], y_predict_all[bridge_indices]]).T, columns=["DFT Cluster Energy (eV)","Predicted Cluster Energy (eV)"])
df3 = pd.DataFrame(np.array([y[hollow_indices], y_predict_all[hollow_indices]]).T, columns=["DFT Cluster Energy (eV)","Predicted Cluster Energy (eV)"])

df1['site_type'] = 'top'
df2['site_type'] = 'bridge'
df3['site_type'] = 'hollow'
df=pd.concat([df1, df2, df3])


def multivariateGrid(col_x, col_y, col_k, df, model_name, k_is_color=False, scatter_alpha=.6):
    def colored_scatter(x, y, c=None):
        def scatter(*args, **kwargs):
            args = (x, y)
            if c is not None:
                kwargs['c'] = c
            kwargs['alpha'] = scatter_alpha
            plt.scatter(*args, **kwargs, s = 50)

        return scatter

    g = sns.JointGrid(
        x=col_x,
        y=col_y,
        data=df
    )
    color = ['green', 'blue', 'r']
    i = 0
    legends=[]
    for name, df_group in df.groupby(col_k):
        legends.append(name)
        if k_is_color:
            color=name
        g.plot_joint(
            colored_scatter(df_group[col_x],df_group[col_y],color[i]),
        )
        sns.scatterplot(
            df_group[col_x].values, df_group[col_y].values - df_group[col_x].values, 
            ax=g.ax_marg_x, s = 30, color =color[i], alpha = 0.6
        )
        sns.distplot(
            df_group[col_y].values,
            ax=g.ax_marg_y,
            bins=np.arange(lims[0], lims[1], 0.1), #set bin size to 0.1 eV
            color=color[i],            
            vertical=True,
            norm_hist= True
        )
        i += 1
    # Do also global Hist:
#    sns.scatterplot(
#        df[col_x].values, df[col_y].values - df[col_x].values, 
#        ax=g.ax_marg_x, s = 30, color ='b', alpha = 0.8
#    )
    g.ax_marg_x.plot(lims, np.zeros(len(lims)), 'k--' )

    # sns.distplot(
    #     df[col_y].values.ravel(),
    #     ax=g.ax_marg_y,
    #     color='grey',
    #     vertical=True,
    #     norm_hist= True
    # )
    plt.legend(legends, loc= 'upper left', frameon=False)
    g.ax_joint.plot(lims, lims, '--k')

    g.set_axis_labels(r"$\rm DFT\ E^{CO-ads^{(i)}}_{\ Pd_n/CeO_2}\ (eV)$", r"$\rm Model\ Predicted\ E^{CO-ads^{(i)}}_{\ Pd_n/CeO_2}\ (eV)$")
    g.ax_marg_x.set_ylabel(r"$\rm \Delta E(eV)$")
    g.ax_marg_y.set_xlabel("Frequency")
    g.ax_marg_x.set_ylim([-0.5,0.5]) # convert to eV
    g.ax_marg_y.set_xticks([0,3]) # not the actual frequency

    g.fig.set_size_inches(7,7)

    plt.setp(g.ax_marg_x.get_yticklabels(), visible=True)
    plt.setp(g.ax_marg_y.get_xticklabels(), visible=True)    
    plt.setp(g.ax_marg_x.yaxis.get_majorticklines(), visible=True)
    plt.setp(g.ax_marg_x.yaxis.get_minorticklines(), visible=True)
    plt.setp(g.ax_marg_y.xaxis.get_majorticklines(), visible=True)
    plt.setp(g.ax_marg_y.xaxis.get_minorticklines(), visible=True)
    if matplotlib.__version__ == '2.2.0':
        g.savefig(os.path.join(os.getcwd(), model_name + '_binding_combo.png'), dpi = dpi)
    else: plt.show()

    
multivariateGrid("DFT Cluster Energy (eV)","Predicted Cluster Energy (eV)", 'site_type', df=df, model_name = model_name)





