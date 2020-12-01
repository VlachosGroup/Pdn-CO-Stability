# -*- coding: utf-8 -*-
"""
Created on Thu Oct 22 10:41:37 2020

@author: yifan
"""
"""
Save the all data into a csv file 
including 300K and other conditions
"""
import os
import pandas as pd
import numpy as np



pdx_names = ['x'] + list(np.arange(5, 22, 1))
df_list = []

# Read the physics file
for pdi_name in pdx_names:
    filename_stable = os.path.join(os.getcwd(),  'pd'+  str(pdi_name) + '_stable_config.csv')
    df_i = pd.read_csv(filename_stable, index_col=False)
    df_list.append(df_i)

# Append the configs at 300K
filename_stable_300K = os.path.join(os.getcwd(),  'pdall_stable_config_300K.csv')   
df_300K = pd.read_csv(filename_stable_300K, index_col=False)
df_list.append(df_300K)


df_all = pd.concat(df_list)

df_all.to_csv('pdall_stable_config_all_conditions.csv', index = False)