import pandas as pd
import numpy as np
import shap 
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score

data = {
'Cijena': [400000, 150000, 275000, 520000, 90000, 600000, 180000, 350000, 210000, 480000, 320000,
195000, 530000, 110000, 650000, 230000, 380000, 290000, 440000, 170000, 580000, 205000, 410000, 310000,
490000, 95000, 620000, 220000, 360000, 270000, 450000, 130000, 670000, 240000, 390000, 330000, 460000,
105000, 700000, 250000, 370000, 280000, 470000, 140000, 720000, 260000, 400000, 300000, 500000, 115000,
750000],
'Kvart': [1,1,2,3,2,3,1,2,1,3,2,1,3,2,3,1,2,1,3,2,3,1,2,1,3,2,3,1,2,1,3,2,3,1,2,1,3,2,3,1,2,1,3,2,3,1,2,1,3,2,3],
'Kvadrati': [100,30,65,120,25,150,45,80,55,110,75,50,125,35,160,60,85,70,105,40,140,52,90,68,115,28,155,58,82,62,108,32,170,63,88,72,112,33,180,65,84,67,118,38,190,70,95,75,130,36,200],
'Dizalo': [0,0,1,1,0,1,0,1,0,1,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1],
'Kat': [2,1,3,5,1,6,2,4,3,4,3,2,5,1,7,3,4,4,4,2,6,2,5,3,5,1,7,3,4,3,4,1,8,4,5,4,5,1,9,4,5,3,5,2,10,5,6,4,6,2,11],
'Sobe': [3,1,2,4,1,5,2,3,2,3,2,2,4,1,5,2,3,3,3,1,4,2,3,2,4,1,5,2,3,2,3,1,6,3,3,3,4,1,6,3,3,2,4,1,6,3,4,3,5,1,7],
'Balkon': [2,0,1,2,0,3,1,1,0,2,1,1,2,0,3,1,1,1,1,0,2,1,2,0,2,0,3,1,1,0,1,0,3,1,2,1,2,0,3,1,1,0,2,0,3,1,2,1,2,0,4],
'Parking': [2,1,1,2,0,2,1,1,0,2,1,0,2,0,2,1,1,1,2,1,2,1,1,0,2,0,2,1,1,0,2,1,2,1,1,1,2,0,2,1,1,0,2,1,2,1,1,1,2,0,3],
'Godina_izgradnje': [1985,1990,2005,2015,1978,2020,1995,2010,1980,2018,2008,1992,2016,1985,2021,1998,2012,2000,2017,1993,2019,
1996,2014,1988,2020,1975,2022,1999,2011,1982,2018,1987,2023,2001,2013,2003,2021,1980,2022,2002,2015,1984,2020,
1991,2023,2004,2016,2006,2021,1989,2024]
}

df = pd.DataFrame(data)

X = df.drop('Cijena', axis=1)
y = df['Cijena']
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

model = RandomForestRegressor(random_state=42)

param_grid = {
'n_estimators': [50, 100, 200],
'max_depth': [None, 10, 20],
'min_samples_split': [2, 5],
'min_samples_leaf': [1, 2]
}

grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=5, n_jobs=-1,
scoring='neg_mean_squared_error', verbose=1)

grid_search.fit(X_train, y_train)
print(f"Najbolji model: {grid_search.best_params_}")

best_model = grid_search.best_estimator_

y_pred_best_model = best_model.predict(X_val)
mse_best_model = mean_squared_error(y_val, y_pred_best_model)
r2_best_model = r2_score(y_val, y_pred_best_model)

print(f"Mean Squared Error (MSE) na setu za validaciju (optimizirani model): {mse_best_model:.2f}")
print(f"R2 Score na setu za validaciju (optimizirani model): {r2_best_model:.4f}")

model.fit(X_train, y_train)

y_pred_basic_model = model.predict(X_val)
mse_basic_model = mean_squared_error(y_val, y_pred_basic_model)
r2_basic_model = r2_score(y_val, y_pred_basic_model)

print(f"Mean Squared Error (MSE) na setu za validaciju (osnovni model): {mse_basic_model:.2f}")
print(f"R2 Score na setu za validaciju (osnovni model): {r2_basic_model:.4f}")

print("\nUsporedba modela:")
print(f"Osnovni model - MSE: {mse_basic_model:.2f}, R2: {r2_basic_model:.4f}")
print(f"Optimizirani model - MSE: {mse_best_model:.2f}, R2: {r2_best_model:.4f}")