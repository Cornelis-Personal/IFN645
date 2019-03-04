# -*- coding: utf-8 -*-
"""
Created on Mon Mar  4 17:29:50 2019

@author: n8891974
"""

# Imports
import pandas as pd
import numpy as np

# Ignore Warnings
import warnings
warnings.filterwarnings("ignore")

### Globals ###
# Create a mapping for Wheel Type
WheelTypeMapping = {'Alloy':1,  'Covers':2, 'Special':3}

# Do the data analysis
data = pd.read_csv("Kick.csv", na_values = '?')

def PreProcessing (data):
    print("Pre-Processing Step")
    
    # Handle Bad Columns drop Columns
    print("Drop PRIMEUNIT due to incufficient data amount")
    print("Drop AUCGUART due to incufficient data amount and is leaky data")
    print("Drop wheel type due to it being a dupplicate of wheelType ID.")
    data.drop(['PRIMEUNIT', 'AUCGUART', 'WheelTypeID'], axis=1, inplace=True)
    
    # Handle Missing Values
    i = 0            # Python's indexing starts at zero
    for item in data['TopThreeAmericanName']:   # Python's for loops are a "for each" loop 
        if data['TopThreeAmericanName'][i] == np.nan and  data['Make'][i] == 'Hyundai':
            data['TopThreeAmericanName'][i] = 'HYUNDAI'
        i += 1
        
    i = 0            # Python's indexing starts at zero
    for item in data['TopThreeAmericanName']:   # Python's for loops are a "for each" loop 
        if data['TopThreeAmericanName'][i] == np.nan and  data['Make'][i] == 'Jeep':
            data['TopThreeAmericanName'][i] = 'JEEP'
        i += 1
    
    print(data.groupby(['TopThreeAmericanName'])['Make'].value_counts())
    
def Question1(data):
    print("Question 1.")
    
    # check to see if the are any odd labaels
    
    counts = data["IsBadBuy"].value_counts()
    total = data["IsBadBuy"].count()
    
    # calculate the percentage
    # kick is where IsBadBuy == 1
    kickPersentage = counts[1] / total * 100
    print ("Questin 1.1 :")
    print( kickPersentage, "%")



### RUN THE OUTPUT ###
data = PreProcessing(data)
Question1(data)