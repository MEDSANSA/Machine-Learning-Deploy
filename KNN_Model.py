import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import numpy as np
import pickle

happy = pd.read_csv('final_happy.csv', sep=',')

happy = happy.drop(['Country name', 'year'], axis=1)

y = happy['Life Ladder']
X = happy.drop(['Life Ladder'], axis=1)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

k = 3
knn_reg = KNeighborsRegressor(n_neighbors=k)
knn_reg.fit(X_train, y_train)

yhat = knn_reg.predict(X_test)

test_r2 = r2_score(y_test, yhat)
test_mse = mean_squared_error(y_test, yhat)
test_mae = mean_absolute_error(y_test, yhat)

print("Test set R-squared: ", test_r2)
print("Test set MSE: ", test_mse)
print("Test set MAE: ", test_mae)

pickle.dump(knn_reg, open("knn_model.pkl", "wb"))