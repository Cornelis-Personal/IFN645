# -*- coding: utf-8 -*-
"""
Created on Mon Mar  4 17:29:50 2019

@author: n8891974
"""

# Imports
import pandas as pd
import numpy as np
import scipy.stats as stats

# Import plotting libraries
import matplotlib.pyplot as plt
import seaborn as sns


# Ignore Warnings
import warnings
warnings.filterwarnings("ignore")

### Globals ###


# Do the data analysis
data = pd.read_csv("Kick.csv", 
                   index_col = 'PurchaseID', 
                   na_values = ('?',  '#VALUE!'))

def PreProcessing (data):
    print("Pre-Processing Step")
    
    # Check if there are any missing target variables
    if data['IsBadBuy'].isnull().values.any() == True:
        print("Missing Target Variables")
    else:
        print("No missing Target Variables")
    
    
    # Handle Bad Columns drop Columns
    print("Drop PRIMEUNIT due to insufficient data amount")
    print("Drop AUCGUART due to insufficient and data amount and leaky data")
    print("Drop WheelTypeID due to it being a duplicate of WheelType")
    print("Drop ForSale due to data skew")
    print("Drop PurchaseDate due to it being a duplicate of PurchaseTimestamp")
    data.drop(['PRIMEUNIT', 'AUCGUART', 'WheelTypeID', 'ForSale', 'PurchaseDate'], 
              axis=1, 
              inplace=True)
    
    
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
        
        
    """ DOES THIS HAVE TO BE THERE """
    # print(data.groupby(['TopThreeAmericanName'])['Make'].value_counts())
    """ DOES THIS HAVE TO BE THERE """
    
    
    
    # Standardise the capitilization across all object rows
    obj_cols = data.select_dtypes(include='object').columns # Create a list of col names
    for i in obj_cols:              # Interate over the obj_cols list
        data[i] = data[i].str.upper()   # Convert all strings to uppercase
        
    # Standardize USA to AMERICA
    data['Nationality'].replace({'USA' : 'AMERICAN'}, inplace = True)
    
    # Turn Transmission into a binary variable with Auto = 1 and Manual = 0
    data.rename({'Transmission' : 'Auto'}, axis = 1, inplace = True)
    # Replace binary columns with 1s and 0s
    data['Auto'].replace({'MANUAL' : 0, 'AUTO' : 1}, inplace=True)
    
    
    """
    #This is the significance test for VNST
    # Check to see if VNST is a statisically significant variable
    # Create a distribution of IsBuyBad for VNST
    VNST_badBuy = pd.crosstab(data['IsBadBuy'], data['VNST']).loc[0]
    VNST_goodBuy = pd.crosstab(data['IsBadBuy'], data['VNST']).loc[1]
    categoricalPlot('VNST')
    
    # Use a Chi2 test to test if there is any corrilation between them, if there is
    # (p < 0.05) discarde the variable
    fScore, pValue = stats.f_oneway(VNST_badBuy, VNST_goodBuy)
    print("The pValaue is " + str(pValue) + " which is significant enough to reject null hypothesis")
    """
    print("Drop VNST due to statistical insignificance")
    data.drop('VNST', axis=1, inplace = True)
    
    
    # Seperate the Size feature into Size and Body
    tempSize = data['Size'].str.split(' ', expand = True) # Create temp var with split column
    data['Size'] = tempSize[0] # Save the temp var back into data
    data['Body'] = tempSize[1] # Save the temp var back into data
    data['Body'].fillna('CITY', inplace = True) # Assume any other cars are 'City'
    
    
    # Replace all non 0, 1 values in IsOnlineSale to 1
    maskOnlineSale = data['IsOnlineSale'] != 0 # Any value that isn't 0 will be set to 1
    data.loc[maskOnlineSale, 'IsOnlineSale'] = 1 # Set the values to 1
    
    """ This should be the last thing done """
    # Convert all categorical variables into one hot representations
    
    print("The number of features before one hot encoding is " + str(data.shape[1]))
    data_OH = pd.get_dummies(data, columns = ['Auction', 'Make', 'Color', 'VehYear'
                                              'Nationality', 'Size', 'Body', 'TopThreeAmericanName', 'WheelType'])
    data_OH.rename({'Size_VAN' : 'Body_VAN'}) # Standardize the naming convention
    print("The number of features after one hot encoding is " + str(data_OH.shape[1]))
       
    
    return data, data_OH

# Define a function to plot catgorical variables with relation to another cat, default is IsBadBuy
def categoricalPlot(cat, cat2 = 'IsBadBuy'): # Cat is the carigorical as a string i.e 'Size'
    pd.crosstab(data[cat],data[cat2]).plot(kind="bar")


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
data, data_OH = PreProcessing(data)
Question1(data)
categoricalPlot('Size')