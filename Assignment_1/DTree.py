# Imports
import pandas as pd
import numpy as np
import scipy.stats as stats

# Import plotting libraries
import matplotlib.pyplot as plt
import seaborn as sns


from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report, accuracy_score
from sklearn.model_selection import GridSearchCV

from DMProj1 import *

# Ignore Warnings
import warnings
warnings.filterwarnings("ignore")

# Define target variable and features
y = data_OH['IsBadBuy']
X = data_OH.drop(['IsBadBuy'], axis=1)


# Split the data into test and train groups with a test size of 20%
X_mat = X.values # Turn X into a matrix
X_train, X_test, y_train, y_test = train_test_split(X_mat, y, test_size=0.2, 
                                                    stratify=y, random_state=rs)

# Build the decision Tree
model = DecisionTreeClassifier(random_state=rs) # Define the model
model.fit(X_train, y_train) # Fit the data
y_pred = model.predict(X_test)


print("With no parameter tuning")
print("Train accuracy:", model.score(X_train, y_train))
print("Test accuracy:", model.score(X_test, y_test))
print(classification_report(y_test, y_pred))

# grab feature importances from the model and feature name from the original X
importances = model.feature_importances_
feature_names = X.columns

# sort them out in descending order
indices = np.argsort(importances)
indices = np.flip(indices, axis=0)

# limit to 20 features, you can leave this out to print out everything
indices = indices[:20]

for i in indices:
    print(feature_names[i], ':', importances[i])
    

# Test the model on different depths and plot the accuracy
test_score = [] # Define empty set for test scores
train_score = [] # Define empty set for train scores

# check the model performance for max depth from 2-20
for max_depth in range(2, 21):
    model = DecisionTreeClassifier(max_depth=max_depth, random_state=rs) # Define the model
    model.fit(X_train, y_train) # Fit the model to the data
    
    # Append the score into empty sets
    test_score.append(model.score(X_test, y_test)) 
    train_score.append(model.score(X_train, y_train))
    
    
# Plot the accuracy scores and see the best range for depth
plt.plot(range(2, 21), train_score, 'b', range(2,21), test_score, 'r')
plt.xlabel('max_depth\nBlue = training acc. Red = test acc.')
plt.ylabel('accuracy')
plt.show()

# Perform a grid search over the best hyperparameters
params = {'criterion': ['gini', 'entropy'], # What criterion to check
          'max_depth': range(2, 7), # Check the depth, use the graph generated above
          'min_samples_leaf': range(5, 26, 5)} # Define the min sample leafs

cv = GridSearchCV(param_grid=params, estimator=DecisionTreeClassifier(random_state=rs), cv=10) # Define the model
cv.fit(X_train, y_train) # Fit the data to the model
y_pred = cv.predict(X_test) # test the best model

print("Using grid search the accuracy is")
print("Train accuracy:", cv.score(X_train, y_train))
print("Test accuracy:", cv.score(X_test, y_test))
print(classification_report(y_test, y_pred))

# print parameters of the best model
print(cv.best_params_)

# Use these parameters to refine the model
params = {'max_depth': range(2, cv.best_params_['max_depth']+2),
          'min_samples_leaf': range(cv.best_params_['min_samples_leaf']-4, 
                                    cv.best_params_['min_samples_leaf']+5)}

cv = GridSearchCV(param_grid=params, estimator=DecisionTreeClassifier(criterion= cv.best_params_['criterion'], random_state=rs), cv=10)
cv.fit(X_train, y_train)

print("Using the refinded parameters")
print("Train accuracy:", cv.score(X_train, y_train))
print("Test accuracy:", cv.score(X_test, y_test))

# test the best model
y_pred = cv.predict(X_test)
print(classification_report(y_test, y_pred))

# print parameters of the best model
print(cv.best_params_)