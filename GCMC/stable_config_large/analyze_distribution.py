# -*- coding: utf-8 -*-
"""
Created on Tue Oct 20 23:14:25 2020

Analyse the isomer and size distributions
"""
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt 
import matplotlib

font = {'size'   : 20}
matplotlib.rc('font', **font)
matplotlib.rcParams['axes.linewidth'] = 1.5
matplotlib.rcParams['xtick.major.size'] = 8
matplotlib.rcParams['xtick.major.width'] = 2
matplotlib.rcParams['ytick.major.size'] = 8
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

pdx_names = [25, 30, 38, 55]  

# read the data of stable configurations
filename_stable = os.path.join(os.getcwd(),  'pd_large_config_300K.csv')  
df_all = pd.read_csv(filename_stable, index_col=False)

#%%
# select 300K and 1e-1
T = 300
PCO = 1e-1
df_T = df_all.loc[(df_all['PCO'] == PCO )]
df_TP = df_T.loc[(df_T['T'] == T )]

size_labels =  pdx_names
# plot violin for all isomers 
mu_all_list = []
mco_all_list = []

# the previous lowest free energy n = 1-21 
mu_all_mean_small = -2.936011156
# mu_all_array = np.array(df_TP['mu_mean'])
# mu_all_min = np.min(mu_all_array)

for i in pdx_names:
    df_TP_xi =  df_TP.loc[(df_TP['n'] == i)]
    mu_xi = np.array(df_TP_xi['mu_mean'])
    mco_xi = np.array(df_TP_xi['nCO_mean'])
    
    mu_all_list.append(mu_xi)
    mco_all_list.append(mco_xi)


fig, ax1 = plt.subplots(figsize=(6, 3))
violin_parts= ax1.violinplot(mu_all_list, showmeans = True, showextrema = True)
ax1.set_xlabel('Cluster Size (n)')
ax1.set_ylabel(r'$\rm \overline{G}\ (eV)$')
ax1.set_ylim([-4, -0.9])
ax1.axhline(y=mu_all_mean_small , color='k', linestyle='--')
#ax1.set_xlim([0, 10])
set_axis_style(ax1, size_labels)


fig, ax2= plt.subplots(figsize=(6, 3))
violin_parts =ax2.violinplot(mco_all_list, showmeans = True, showextrema = True)
set_color(violin_parts, 'r')
ax2.set_xlabel('Cluster Size (n)')
ax2.set_ylabel(r'$\rm \overline{m}_{CO}\ $')
ax2.set_ylim([0, 2])
#ax2.set_xlim([0, 10])
set_axis_style(ax2,size_labels)


#%%
# plot volin for the isomers at the same size 
# At 300K and 1e-1

# def plot_isomer_distribution(xi):
#     """Plot isomer distribution at size xi"""
xi = 55
df_xi = df_TP.loc[(df_TP['n'] == xi)]
iso_indices = list(np.arange(0, np.max(df_xi['iso_index'])+1))
iso_labels = [r'$\rm i_{' + str(i) + '}$' for i in iso_indices]
mu_iso_list = []
mco_iso_list = []

mu_xi_array = np.array(df_xi['mu_mean'])
mu_xi_min = np.min(mu_xi_array)
# select the min size (-1 for indexing)
mu_min_iso_indices = np.array(df_xi.loc[(df_xi['mu_mean'] == mu_xi_min)]['iso_index'])
n_xi_min = mu_min_iso_indices[0] + 0.7

mu_min_iso = []
for i in iso_indices:
    sub_df = df_xi.loc[df_xi['iso_index']== i]
    mu_i = np.array(sub_df['mu_mean'])
    mco_i = np.array(sub_df['nCO_mean'])
    mu_iso_list.append(mu_i)
    mco_iso_list.append(mco_i)
    mu_min_iso.append(mu_xi_min - np.min(mu_i))
    
fig, ax1 = plt.subplots(figsize=(5, 2.5))
violin_parts= ax1.violinplot(mu_iso_list, showmeans = True, showextrema = True)
ax1.set_xlabel(r'$\rm Pd_{' + str(xi)+ '}\ Isomer\ labels$')
ax1.set_ylabel(r'$\rm \overline{G}\ (eV)$')
ax1.set_ylim([-3.6, -1])
ax1.axhline(y=mu_xi_min , color='k', linestyle='--')
ax1.annotate(r'$\rm \overline{G}_{min}$', xy=(n_xi_min, mu_xi_min -0.5), fontsize =18)
#ax1.set_xlim([0, 10])
set_axis_style(ax1, iso_labels)

fig, ax2= plt.subplots(figsize=(5,2.5))
violin_parts =ax2.violinplot(mco_iso_list, showmeans = True, showextrema = True)
set_color(violin_parts, 'r')
ax2.set_xlabel(r'$\rm Pd_{' + str(xi)+ '}\ Isomer\ labels$')
ax2.set_ylabel(r'$\rm \overline{m}_{CO}\ $')

ax2.set_ylim([0.5, 2])
#ax2.set_xlim([0, 10])
set_axis_style(ax2, iso_labels)


#plot_isomer_distribution(13)

