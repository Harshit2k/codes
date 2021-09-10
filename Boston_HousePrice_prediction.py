# -*- coding: utf-8 -*-
"""Task3.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1ZkY2AHpU309h3LwJe7h47ZZ1rMZjFU7R

**House Price Prediction**

**Importing libraries**
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import ShuffleSplit

from google.colab import files
files.upload()

import visuals as vs

from google.colab import files
uploaded=files.upload()

import io
data=pd.read_csv(io.BytesIO(uploaded['HousingData.csv']))

"""**Data Preprocessing**"""

data.isnull().sum()

data.fillna(0,inplace=True)

# Commented out IPython magic to ensure Python compatibility.
# %matplotlib inline
prices = data['MEDV']
features = data.drop('MEDV', axis = 1)
print("Boston housing dataset has {} data points with {} variables each.".format(*data.shape))

"""**Observation**

-> Houses with more rooms(higher 'RM' value) will worth more. Usually houses with more rooms bigger and can fit more people, so it is reasonable that they cost more money.

-> Neighborhoods with more lower class workers(high 'LSTAT' value) will worth less.If the percentage of lower working people is higher, it is likely that they have low purchasing power and therefore, these houses will cost less.

-> Neighbors with more students to teachers ratio(higher 'PTRATIO' value) will be worth less. If the percentage of students to teachers ratio people is higher, it is likely that in the neighborhood there are less schools, this could be because there is less tax income which could be because in that neighborhood people earn less money.
"""

import matplotlib.pyplot as plt
import seaborn as sns
sns.pairplot(data)
plt.tight_layout()

correlation_matrix = data.corr().round(2)
sns.heatmap(data=correlation_matrix, annot=True)

sns.set(rc={'figure.figsize':(11.7,8.27)})
sns.distplot(data['MEDV'], bins=30)
plt.show()

"""**Calculating Satistics**"""

# TODO: Minimum price of the data
minimum_price = np.min(prices)

# TODO: Maximum price of the data
maximum_price = np.max(prices)

# TODO: Mean price of the data
mean_price = np.mean(prices)

# TODO: Median price of the data
median_price = np.median(prices)

# TODO: Standard deviation of prices of the data
std_price = np.std(prices)

# Show the calculated statistics
print ("Statistics for Boston housing dataset:\n")
print ("Minimum price: ${:,.2f}".format(minimum_price))
print ("Maximum price: ${:,.2f}".format(maximum_price))
print ("Mean price: ${:,.2f}".format(mean_price))
print ("Median price ${:,.2f}".format(median_price))
print ("Standard deviation of prices: ${:,.2f}".format(std_price))

"""**Feature Observation**"""

import matplotlib.pyplot as plt
import seaborn as sns
for var in features.columns:
    sns.regplot(data[var],prices)
    plt.show()

"""**Performance of Metric**"""

from sklearn.metrics import r2_score
def performance_metric(y_true, y_predict):
  score = r2_score(y_true,y_predict)
  return score

score = performance_metric([3, -0.5, 2, 7, 4.2], [2.5, 0.0, 2.1, 7.8, 5.3])
print ("Model has a coefficient of determination, R^2, of {:.3f}.".format(score))

"""**Splitting of Training and Testing data**"""

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(features,prices,test_size=0.2,random_state=100)
print ("Training and testing split was successful.")

#Percentage_of_training_and_testing_data
print(features.shape[0])
print(float(X_train.shape[0])/float(features.shape[0]))
print(float(X_test.shape[0]/float(features.shape[0])))

"""**Model Fitting**"""

#Fitting_a_model
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import make_scorer
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import ShuffleSplit
def fit_model(X, y):
  cv_sets = ShuffleSplit(n_splits=10, test_size = 0.20, random_state = 0)
  regressor = DecisionTreeRegressor()
  params = {'max_depth': range(1,11)}
  scoring_fnc = make_scorer(performance_metric)
  grid = GridSearchCV(regressor,params,scoring_fnc,cv=cv_sets)
  grid = grid.fit(X, y)
  return grid.best_estimator_

reg = fit_model(X_train, y_train)
print ("Parameter 'max_depth' is {} for the optimal model.".format(reg.get_params()['max_depth']))

"""**Model's Accuracy**"""

from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

lin_model = LinearRegression()
lin_model.fit(X_train, y_train)
y_train_predict = lin_model.predict(X_train)
rmse = (np.sqrt(mean_squared_error(y_train, y_train_predict)))
r2 = r2_score(y_train, y_train_predict)

print("The model performance for training set")
print("--------------------------------------")
print('RMSE is {}'.format(rmse))
print('R2 score is {}'.format(r2))
print("\n")

# model evaluation for testing set
y_test_predict = lin_model.predict(X_test)
rmse = (np.sqrt(mean_squared_error(y_test, y_test_predict)))
r2 = r2_score(y_test, y_test_predict)

print("The model performance for testing set")
print("--------------------------------------")
print('RMSE is {}'.format(rmse))
print('R2 score is {}'.format(r2))

"""**Price Prediction**"""

#Predicting_Selling_Prices
client_data = [[5, 17, 15],
               [4, 32, 22], 
               [8, 3, 12]] 
for i, price in enumerate(reg.predict(client_data)):
    print ("Predicted selling price for Client {}'s home: ${:,.2f}".format(i+1, price))

import matplotlib.pyplot as plt
plt.hist(prices, bins = 20)
for price in reg.predict(client_data):
    plt.axvline(price, lw = 5, c = 'r')

"""**Sites Which helped me during project:**

-> towardsdatascience.com

-> ritchieng.com

-> wikipedia.com

-> stackoverflow.com
"""