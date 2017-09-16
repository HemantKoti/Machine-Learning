# Polynomial Regression

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Position_Salaries.csv')
X = dataset.iloc[:, 1:2].values
y = dataset.iloc[:, -1].values

# Splitting the dataset into the Training set and Test set
# For smaller data sets we dont need to split the data set into test and training set for accurate results
"""from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)"""

# For comparing results we build both the models - Linear and Polynomial
# Fitting Linear Regression to the dataset
from sklearn.linear_model import LinearRegression
linear_regressor = LinearRegression()
linear_regressor.fit(X,y)

# Fitting Polynomial Regression to the dataset
from sklearn.preprocessing import PolynomialFeatures
polynomial_regressor = PolynomialFeatures(degree=4)
X_poly = polynomial_regressor.fit_transform(X)
linear_regressor2 = LinearRegression()
linear_regressor2.fit(X_poly,y)

# Visualising the Linear Regression results
plt.scatter(X,y,color = 'red')
plt.plot(X, linear_regressor.predict(X), color = 'blue')
plt.title('Truth or bluff (Linear Regression)')
plt.xlabel('Position Level')
plt.ylabel('Salary')
plt.show()

# Visualising the Polynomial Regression results
X_grid = np.arange(min(X), max(X), 0.1)
X_grid = X_grid.reshape(len(X_grid), 1)
plt.scatter(X,y,color = 'red')
plt.plot(X_grid, linear_regressor2.predict(polynomial_regressor.fit_transform(X_grid)), color = 'blue')
plt.title('Truth or bluff (Polynomial Regression)')
plt.xlabel('Position Level')
plt.ylabel('Salary')
plt.show()

# Predicting Linear Regression results
linear_regressor.predict(6.5)

# Predicting Polynomial Regression results
linear_regressor2.predict(polynomial_regressor.fit_transform(6.5))