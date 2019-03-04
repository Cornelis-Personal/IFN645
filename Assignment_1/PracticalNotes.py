# -*- coding: utf-8 -*-
"""
Created on Mon Mar  4 16:56:02 2019

@author: n8891974
"""

import pandas as pd
import numpy as np

data = pd.read_csv('datasets/veteran.csv')

# This can be used to find out of there are any null values
print(data.info().unique()) 

# This can be used to easily calssify data for grouping continious data
print(data['DemAge'].value_counts(bins=10))

# This can be used to find values that are illegical
print(data['DemAge'].value_counts())


# Used for mapping values
dem_home_owner_map = {'U':0, 'H': 1}
data['DemHomeOwner'] = data['DemHomeOwner'].map(dem_home_owner_map)

# Used for setting NANs
mask = data['DemMedIncome'] < 1
data.loc[mask, 'DemMedIncome'] = np.nan

# Fill missing values
data['DemAge'].fillna(data['DemAge'].mean(), inplace=True)

# drop Columns
data.drop(['ID', 'TargetD'], axis=1, inplace=True)