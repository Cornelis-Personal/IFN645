# -*- coding: utf-8 -*-
"""
Created on Mon Mar  4 17:29:50 2019

@author: n8891974
"""
import pandas as pd
from datetime import datetime


data = pd.read_csv("Kick.csv")

# Inconsistencies

### Illogical Data ###
# Drop the PurchaseID
data.drop(['PurchaseID'], axis=1, inplace=True)

timeSeries = df_copy = pd.DataFrame().reindex_like(data["PurchaseTimestamp"])

counter = 0
for i in data["PurchaseTimestamp"]:
    timeSeries[counter]= datetime.utcfromtimestamp(i).strftime('%Y-%m-%d %H:%M:%S')
    counter = counter + 1
print (data["PurchaseTimestamp"])