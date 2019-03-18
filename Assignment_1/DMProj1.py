# -*- coding: utf-8 -*-
"""
Created on Mon Mar  4 17:29:50 2019

@author: n8891974
"""

# Imports
import pandas as pd
import numpy as np
import datetime 
import scipy.stats as stats

# Import plotting libraries
import matplotlib.pyplot as plt
import seaborn as sns

# Import ML libraries
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report, accuracy_score
from sklearn.model_selection import GridSearchCV
rs = 10 # Set a random state const

# Ignore Warnings
import warnings
warnings.filterwarnings("ignore")

### Globals ###


# Do the data analysis
data = pd.read_csv("Kick.csv", 
                   index_col = 'PurchaseID', 
                   na_values = ('?',  '#VALUE!'))

def PreProcessing (df):
    print("Pre-Processing Step")
    
    # Check if there are any missing target variables
    if df['IsBadBuy'].isnull().values.any() == True:
        print("Missing Target Variables")
    else:
        print("No missing Target Variables")
    
    
    # Handle Bad Columns drop Columns
    print("Drop PRIMEUNIT due to insufficient data amount")
    print("Drop AUCGUART due to insufficient and data amount and leaky data")
    print("Drop WheelTypeID due to it being a duplicate of WheelType")
    print("Drop ForSale due to data skew")
    print("Drop PurchaseDate due to it being a duplicate of PurchaseTimeStamp")
    df.drop(['PRIMEUNIT', 'AUCGUART', 'WheelTypeID', 'PurchaseDate'], 
              axis=1, 
              inplace=True)
    
    
    # Handle Missing Values
    i = 0            # Python's indexing starts at zero
    for item in df['TopThreeAmericanName']:   # Python's for loops are a "for each" loop 
        if df['TopThreeAmericanName'][i] == np.nan and  df['Make'][i] == 'Hyundai':
            df['TopThreeAmericanName'][i] = 'HYUNDAI'
        i += 1
        
    i = 0            # Python's indexing starts at zero
    for item in df['TopThreeAmericanName']:   # Python's for loops are a "for each" loop 
        if df['TopThreeAmericanName'][i] == np.nan and  df['Make'][i] == 'Jeep':
            df['TopThreeAmericanName'][i] = 'JEEP'
        i += 1
        
        # Change time stamp integer to datetime
    print("Convert timestamp to datetime")
    df['PurchaseTimestamp'] = pd.to_datetime(df['PurchaseTimestamp'], unit='s')
    
    # create a new weekday column
    df['Weekday'] = np.nan
     
    for i in range(1,len(df['PurchaseTimestamp'])):
        df['Weekday'] = df['PurchaseTimestamp'][i].weekday()
        
    # remove the rows where the cars are not for sale
    df = df[df.ForSale == 'No']
    df = df[df.ForSale == '0']
    
    # now remove the ForSale column    
    df.drop(['ForSale'], axis=1, inplace=True)
        
    """ DOES THIS HAVE TO BE THERE """
    # print(data.groupby(['TopThreeAmericanName'])['Make'].value_counts())
    """ DOES THIS HAVE TO BE THERE """
    
    # Standardise the capitilization across all object rows
    obj_cols = df.select_dtypes(include='object').columns # Create a list of col names
    for i in obj_cols:              # Interate over the obj_cols list
        df[i] = df[i].str.upper()   # Convert all strings to uppercase
        
    # Standardize USA to AMERICA
    df['Nationality'].replace({'USA' : 'AMERICAN'}, inplace = True)
    
    # Turn Transmission into a binary variable with Auto = 1 and Manual = 0
    df.rename({'Transmission' : 'Auto'}, axis = 1, inplace = True)
    # Replace binary columns with 1s and 0s
    df['Auto'].replace({'MANUAL' : 0, 'AUTO' : 1}, inplace=True)
    
    # Remove NOT AVAIL in color and place it in the NaN section
    df['Color'].replace({'NOT AVAIL': np.nan}, inplace = True)
    
    
    """ I'll have to check with teach if this is correct
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
    df.drop('VNST', axis=1, inplace = True)
    
    
    # Seperate the Size feature into Size and Body
    tempSize = df['Size'].str.split(' ', expand = True) # Create temp var with split column
    df['Size'] = tempSize[0] # Save the temp var back into data
    df['Body'] = tempSize[1] # Save the temp var back into data
    df['Body'].fillna('CITY', inplace = True) # Assume any other cars are 'City'
    df.loc[df.Size == 'VAN', 'Body'] = 'Van' # Convert Van into a body type
    df.loc[df.Size == 'VAN', 'Size'] =  np.nan # Take van away from size, shouldn't matter once OH is done
    
    
    # Replace all non 0, 1 values in IsOnlineSale to 1
    maskOnlineSale = df['IsOnlineSale'] != 0 # Any value that isn't 0 will be set to 1
    df.loc[maskOnlineSale, 'IsOnlineSale'] = 1 # Set the values to 1
    
    
    # Converting the TimeStamp into Quater
    Quater = [] # Create empty string
    for i, _ in enumerate(df.PurchaseTimestamp): # Loop over the entire dataset
        # Convert the epoch datetime into the quater and append to list
        Quater.append(pd.Timestamp(df.PurchaseTimestamp.loc[i], unit = 's').quarter)        
    df['Quater'] = Quater # Create the column with list
    df.drop('PurchaseTimestamp', axis=1, inplace = True) # Drop old TimeStamp
    
    
    
    """ This will take care of any Null values we don't specifically take care of
    by replaceing the missing data with data from the same distibution"""
    for i in df.columns: # Loop over dataset
        if df[i].isna().any(): # Check to see if there is a NaN is the feature
            dist = df[i].value_counts(normalize=True) # Find the distrabution of the column
            missing = df[i].isna() # Find where the NaN are
            # Replace the NaNs with values from the same distrabution of the column
            df.loc[missing, i] = np.random.choice(dist.index, size=len(df[missing]),p=dist.values)       
            print("Converted all of " + i + "s missing values into the same distrubution")
    
    
    """ This should be the last thing done """
    # Convert all categorical variables into one hot representations
    
    print("The number of features before one hot encoding is " + str(df.shape[1]))
    data_OH = pd.get_dummies(data, columns = ['Auction', 'Make', 'Color', 'VehYear', 
                                              'Nationality', 'Size', 'Body', 'TopThreeAmericanName', 
                                              'WheelType', 'Quater'])
    print("The number of features after one hot encoding is " + str(data_OH.shape[1]))
    
    """ Now we will start the numerical analysis """
    # so WarrantyCost and Ratios don't have null data
    # Foreach of the data fill in the mean
    df["MMRAcquisitionAuctionAveragePrice"] = FillZeroWithMean(df["MMRAcquisitionAuctionAveragePrice"])
    df["MMRAcquisitionAuctionCleanPrice"] = FillZeroWithMean(df["MMRAcquisitionAuctionCleanPrice"])
    df["MMRAcquisitionRetailAveragePrice"] = FillZeroWithMean(df["MMRAcquisitionRetailAveragePrice"])
    df["MMRAcquisitonRetailCleanPrice"] = FillZeroWithMean(df["MMRAcquisitonRetailCleanPrice"])
    df["MMRCurrentAuctionAveragePrice"] = FillZeroWithMean(df["MMRCurrentAuctionAveragePrice"])
    df["MMRCurrentAuctionCleanPrice"] = FillZeroWithMean(df["MMRCurrentAuctionCleanPrice"])
    df["MMRCurrentRetailAveragePrice"] = FillZeroWithMean(df["MMRCurrentRetailAveragePrice"])
    df["MMRCurrentRetailCleanPrice"] = FillZeroWithMean(df["MMRCurrentRetailCleanPrice"])
    
    # Test the correlations between Average and Clean
    print (np.corrcoef(df["MMRAcquisitionAuctionAveragePrice"], df["MMRAcquisitionAuctionCleanPrice"] ))
    print (np.corrcoef(df["MMRAcquisitionRetailAveragePrice"], df["MMRAcquisitonRetailCleanPrice"] ))
    print (np.corrcoef(df["MMRCurrentAuctionAveragePrice"], df["MMRCurrentAuctionCleanPrice"] ))
    print (np.corrcoef(df["MMRCurrentRetailAveragePrice"], df["MMRCurrentRetailCleanPrice"] ))  
    
    # We have high correlation so we drop the tables
    # Create new columns with the averaged amount due to high correlation
    df["AcquisitionAuctionprice"] = CreateAveragedColumn(df["MMRAcquisitionAuctionAveragePrice"], df["MMRAcquisitionAuctionCleanPrice"])
    df["AcquisitionRetailPrice"] = CreateAveragedColumn(df["MMRAcquisitionRetailAveragePrice"], df["MMRAcquisitonRetailCleanPrice"])
    df["MMRCurrentAuctionPrice"] = CreateAveragedColumn(df["MMRCurrentAuctionAveragePrice"], df["MMRCurrentAuctionCleanPrice"])
    df["MMRCurrentRetailPrice"] = CreateAveragedColumn(df["MMRCurrentRetailAveragePrice"], df["MMRCurrentRetailCleanPrice"])
    
    # drop the other tables
    df.drop(['MMRAcquisitionAuctionAveragePrice',
               'MMRAcquisitionAuctionCleanPrice', 
               'MMRAcquisitionRetailAveragePrice', 
               'MMRAcquisitonRetailCleanPrice', 
               'MMRCurrentAuctionAveragePrice', 
               'MMRCurrentAuctionCleanPrice', 
               'MMRCurrentRetailAveragePrice', 
               'MMRCurrentRetailCleanPrice'], 
              axis=1, 
              inplace=True)
    
    # Now do a box plots
    df["AcquisitionAuctionprice"].hist()
    df["AcquisitionRetailPrice"].hist()
    df["MMRCurrentAuctionPrice"].hist()
    df["MMRCurrentRetailPrice"].hist()
    
    ## Identify and Remove outiers
    outliers = (IdentifyOutliers(df["AcquisitionAuctionprice"]) and
                IdentifyOutliers(df["AcquisitionRetailPrice"]) and
                IdentifyOutliers(df["MMRCurrentAuctionPrice"]) and
                IdentifyOutliers(df["MMRCurrentRetailPrice"]))
    
    # Select lines that aren't outliers
    num_lines =  len(df["MMRCurrentRetailPrice"])
    df = df[[not i for i in outliers]]
    print ("Lines Removed: ", num_lines - len(df["MMRCurrentRetailPrice"]))
                
    # dealta (Q1, Q3) -> IQR
    # lower min = Q1 - 1.5 * IQR
    # higher value = Q3 + 1.5 * IQR
	
    return df, data_OH


# Define a function to plot catgorical variables with relation to another cat, default is IsBadBuy
def categoricalPlot(cat, cat2 = 'IsBadBuy'): # Cat is the carigorical as a string i.e 'Size'
    pd.crosstab(data[cat],data[cat2]).plot(kind="bar")


def Question1(data):
    print("Question 1.")
    
    # check to see if the are any odd labaels
    print(data)
    counts = data["IsBadBuy"].value_counts()
    total = data["IsBadBuy"].count()
    
    # calculate the percentage
    # kick is where IsBadBuy == 1
    kickPersentage = counts[1] / total * 100
    print ("Questin 1.1 :")
    print( kickPersentage, "%")

def HasZero (column):
    return len(column[column == 0]) > 0

def FillZeroWithMean (column):
    # get the mean of the non zero
    m = column[column != 0].mean()
    column=column.replace(0,m) 
            
    return column

def CreateAveragedColumn (A, B):
    C = [None] * len( A )
    for i in range(len(A)):
        C[i] = (A[i] + B[i])/2
    
    return C

def IdentifyOutliers (column):
    data_described = column.describe()
    Q1 = data_described["25%"]
    Q3 = data_described["75%"]
    IQR = Q3-Q1
    lowerLimit = Q1 - 1.5 * IQR
    upperLimit = Q3 + 1.5 * IQR
    output = [None] * len( column )
    for i in range(len(column)):
        value = column[i]
        if value < lowerLimit or value > upperLimit:
            output[i] = True
        else:
            output[i] = False
    return output
            
### RUN THE OUTPUT ###
data, data_OH = PreProcessing(data)
Question1(data)
