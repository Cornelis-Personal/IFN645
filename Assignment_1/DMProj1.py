# -*- coding: utf-8 -*-
"""
Created on Mon Mar  4 17:29:50 2019

@author: n8891974
"""

# Imports
import pandas as pd
import numpy as np
import datetime 
# Ignore Warnings
import warnings
warnings.filterwarnings("ignore")

### Globals ###
# Create a mapping for Wheel Type
WheelTypeMapping = {'Alloy':1,  'Covers':2, 'Special':3}

# Do the data analysis
data = pd.read_csv("Kick.csv", na_values = '?')

def PreProcessing (df):
    print("Pre-Processing Step")
    
    # Handle Bad Columns drop Columns
    print("Drop PRIMEUNIT due to incufficient data amount")
    print("Drop AUCGUART due to incufficient data amount and is leaky data")
    print("Drop wheel type due to it being a dupplicate of wheelType ID.")
    print("Drop Purchase date as it is a duplicate of time stamp")
    
    df.drop(['PRIMEUNIT', 'AUCGUART', 'WheelTypeID', 'PurchaseDate'], axis=1, inplace=True)
    
    # Handle Missing Values
    i = 0            # Python's indexing starts at zero
    for item in data['TopThreeAmericanName']:   # Python's for loops are a "for each" loop 
        if df['TopThreeAmericanName'][i] == np.nan and  df['Make'][i] == 'Hyundai':
            df['TopThreeAmericanName'][i] = 'HYUNDAI'
        i += 1
        
    i = 0            # Python's indexing starts at zero
    for item in data['TopThreeAmericanName']:   # Python's for loops are a "for each" loop 
        if df['TopThreeAmericanName'][i] == np.nan and  df['Make'][i] == 'Jeep':
            df['TopThreeAmericanName'][i] = 'JEEP'
        i += 1
        
    # Change time stamp integer to datetime
    print("Convert timestamp to datetime")
    df['PurchaseTimestamp'] = pd.to_datetime(data['PurchaseTimestamp'], unit='s')
    
    # create a new weekday column
    df['Weekday'] = np.nan
     
    for i in range(1,len(df['PurchaseTimestamp'])):
        df['Weekday'] = df['PurchaseTimestamp'][i].weekday()
        
    # remove the rows where the cars are not for sale
    df = df[df.ForSale == 'No']
    df = df[df.ForSale == '0']
    
    # now remove the ForSale column    
    df.drop(['ForSale'], axis=1, inplace=True)
    return df

def Question1(df):
    print("Question 1.")
    
    # check to see if the are any odd labaels
    print(df)
    counts = df["IsBadBuy"].value_counts()
    total = df["IsBadBuy"].count()
    
    # calculate the percentage
    # kick is where IsBadBuy == 1
    kickPersentage = counts[1] / total * 100
    print ("Questin 1.1 :")
    print( kickPersentage, "%")



### RUN THE OUTPUT ###
data = PreProcessing(data)
Question1(data)