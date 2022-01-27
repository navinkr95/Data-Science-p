# -*- coding: utf-8 -*-
"""
Created on Wed Jan 26 23:55:58 2022

@author: Navin
"""

#Multiple Regression

# Importing the libraries
import pandas as pd
import seaborn as sns

# Importing the dataset
dataset = pd.read_csv('D:\SKILLEDGES PUNE\Datasets/stud_reg_2.csv')
print(type(dataset))

#Data Visualization:
sns.heatmap(dataset)

X = dataset.iloc[:, :-1].values
y = dataset.iloc[:,2].values

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 1)

#Note: The parameter 'random_state' is used to randomly bifurcate the dataset into training &
#testing datasets. That number should be supplied as arguments to parameter 'random_state'
#which helps us get the max accuracy. And that number is decided by hit & trial method.

# Fitting Linear Regression to the Training set
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, y_train)

#Calculating the coefficients:
print(regressor.coef_)

#Calculating the intercept:
print(regressor.intercept_)

# Predicting the Test set results
y_pred = regressor.predict(X_test)

# Accuracy of the model

#Calculating the r squared value:
from sklearn.metrics import r2_score
r2_score(y_test,y_pred)

#Create a DataFrame
df1 = {'Actual Applicants':y_test,
'Predicted Applicants':y_pred}
df1 = pd.DataFrame(df1,columns=['Actual Applicants','Predicted Applicants'])
print(df1)

# Visualising the predicted results
import matplotlib.pyplot as plt
line_chart1 = plt.plot(y_pred,X_test, '--',c='green')
line_chart2 = plt.plot(y_test,X_test, ':', c='red')
plt.show()


# To find out when placement rate is 90% and number of graduation is 15000.
# then how many no of application can be received

# from this output, we have determined that the intercept is 3442, the coefficent
# for the placement rate is -127 and the coefficent for number of graduating students is 0.61.
 

#therefore the complete regression equation is:
  #  Application= 3442 + (-127*90) + (15000*0.61)
     #           = 1162

# if placement rate is 90% and no of graduate student per year 15000 then we might can :
# received 1162 no of appliaction.     
     
# This equation tell us that the predicted mumner of appliaction for king's college
# for master in Analytics will:
#decreases by 127 students for every 1 percent increases in the placement rate
# increase by 1 students for every 2 percent increase in the no. of graduating students.

















