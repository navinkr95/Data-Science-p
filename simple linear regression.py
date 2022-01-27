# -*- coding: utf-8 -*-
"""
Created on Tue Jan 25 02:58:34 2022

@author: Navin
"""

# Importing the libraries
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# Importing the dataset
dataset = pd.read_csv('D:\SKILLEDGES PUNE\Datasets/stud_reg.csv')
print(type(dataset))

X = dataset.iloc[:,:-1].values
y = dataset.iloc[:, 1].values

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 1/3, random_state = 0)

#Note: The parameter 'random_state' is used to randomly bifurcate the dataset into training &
#testing datasets. That number should be supplied as arguments to parameter 'random_state'
#which helps us get the max accuracy. And that number is decided by hit & trial method.
    
# Fitting Simple Linear Regression to the Training set
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, y_train)

#Calculating the coefficients:
print(regressor.coef_)

#coefficent is use to find out the relationship between dependent and independend
#variables. If i get the coefficent value positive ,then there is a direct relationship
#between the variables, As the placement rate increase, no of application increases,
#If i get a negative values, then there is a inverse relationship between the variables.
#As the placement rate increase, no of applicatio decreases.

#Calculating the intercept:
print(regressor.intercept_)


#we have determined that the intercept is 15016.89 and the coefficent for
# placement rate is -192.10

# Therefore the complete regresssion equation is:
# Application= 15016.85 + (-192.10 * placement rate)

# this equatio0n telss us that the predicted no of applications for king's college
# for master in anlaytics will decrease by 192 students for every one percent increase
# in the placemnt rate.


# Predicting the Test set results
y_pred = regressor.predict(X_test)

# Accuracy of the model

#Calculating the r squared value:
from sklearn.metrics import r2_score
r2_score(y_test,y_pred)

from sklearn.metrics import mean_squared_error
rmse=np.sqrt(mean_squared_error(y_test,y_pred))
rmse
 
#Create a DataFrame
df1 = {'Actual Applicants':y_test,
'Predicted Applicants':y_pred}
df1 = pd.DataFrame(df1,columns=['Actual Applicants','Predicted Applicants'])
print(df1)
 
# Visualising the predicted results
line_chart1 = plt.plot(X_test,y_pred, '--', c ='red')
line_chart2 = plt.plot(X_test,y_test, ':', c='blue')

# Here i got the accuracy is 70%. so i have to find out the the no of application for 
# 70% placement rate.

# Application= 15016.89 + (-192.1 * 70) 
#             1570

# if our rate placement rate is 70% then we will receives 1570 no of application.


