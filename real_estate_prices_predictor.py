#fetch data 
import imp
import os
from random import Random
import tarfile
from tkinter import Label
from six.moves import urllib 

DOWNLOAD_ROOT = "https://raw.githubusercontent.com/ageron/handson-ml/master/"
HOUSING_URL = DOWNLOAD_ROOT + "datasets/housing/housing.tgz"
HOUSING_PATH = os.path.join("zestawy danych", "mieszkania")


def fetch_housing_data(housing_url=HOUSING_URL, housing_path=HOUSING_PATH):
    if not os.path.isdir(housing_path):
        os.makedirs(housing_path)
    tgz_path = os.path.join(housing_path, "housing.tgz")
    urllib.request.urlretrieve(housing_url, tgz_path)
    housing_tgz = tarfile.open(tgz_path)
    housing_tgz.extractall(path=housing_path)
    housing_tgz.close()

#load data 
import pandas as pd 

def load_housing_data(housing_path=HOUSING_PATH):
    csv_file_path = os.path.join(housing_path, "housing.csv")
    return pd.read_csv(csv_file_path)

fetch_housing_data()
housing_data = load_housing_data()
housing_data.head() #5 pierwszych wierszy
housing_data.info() #lista kolumn i ich typow danych 
housing_data["ocean_proximity"].value_counts() #ilośc poszczególnych wartości dla konkretnej kolumny 
housing_data.describe() #tabela z danymi statystycznymi

#wizualizacja danych - histogram 

import matplotlib.pyplot as plt

housing_data.hist(bins=50, figsize=(20, 15))

#wydzielenie zbioru testowego 
import numpy as np 

def split_train_test(data, test_ratio):
    np.random.seed(42)
    shuffled_indices= np.random.permutation(len(housing_data))   
    test_set_size = int(test_ratio*len(data))
    test_indices = shuffled_indices[:test_set_size]
    train_indices = shuffled_indices[test_set_size:]
    return data.iloc[train_indices], data.iloc[test_indices]

train_set, test_set = split_train_test(housing_data, 0.2)  
print("Uczące:  ", len(train_set), ", testowe:   ", len(test_set))


#wydzielenie zbioru testowego z użyciem wbudowanej funckji
from sklearn.model_selection import train_test_split

train_set, test_set = train_test_split(housing_data, test_size=0.2, random_state=42)
print("Uczące2:  ", len(train_set), ", testowe2:   ", len(test_set))


#warstwy (mediana dochodów)
housing_data["income_cat"] = np.ceil(housing_data["median_income"]/1.5)
housing_data["income_cat"].where(housing_data["income_cat"]<5, 5.0, inplace=True)

#próbkowanie warstwowe 
from sklearn.model_selection import StratifiedShuffleSplit

split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
for train_index, test_index in split.split(housing_data, housing_data["income_cat"]):
    strat_train_set = housing_data.loc[train_index]
    strat_test_set = housing_data.loc[test_index]

housing_data["income_cat"].value_counts() / len(housing_data)

#3.0    0.350581
#2.0    0.318847
#4.0    0.176308
#5.0    0.114438
#1.0    0.039826

#usunięcie atrybutu income_cat 
for set_ in(strat_train_set, strat_test_set):
    set_.drop("income_cat", axis=1, inplace=True) 

#usunięcie atrybutu ocean_proximity 
for set_ in(strat_train_set, strat_test_set):
    set_.drop("ocean_proximity", axis=1, inplace=True)


#dane 

housing_data = strat_train_set.drop("median_house_value", axis=1)
housing_labels = strat_train_set["median_house_value"].copy()

#potok transformujący 
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer

num_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy="median")), 
    ('std_scaller', StandardScaler())
])

housing_data_prepared = num_pipeline.fit_transform(housing_data)

#wybór i uczenie modelu
#regresja liniowa: 
from sklearn.linear_model import LinearRegression

lin_reg = LinearRegression()
lin_reg.fit(housing_data_prepared, housing_labels)

some_data = housing_data.iloc[:5]
some_labels = housing_labels.iloc[:5]
 
some_data_prepared = num_pipeline.transform(some_data)
#wyniki regresji:
print("prognozy:", lin_reg.predict(some_data_prepared))
#rzeczywiste wartości:
print("etykiety:", list(some_labels))

from sklearn.metrics import mean_squared_error
housing_predictions = lin_reg.predict(housing_data_prepared)

lin_mse = mean_squared_error(housing_labels, housing_predictions)
lin_rmse = np.sqrt(lin_mse)
print("linear regression rmse:", lin_rmse)

#drzewo decyzyjne:
from sklearn.tree import DecisionTreeRegressor

tree_reg = DecisionTreeRegressor()
tree_reg.fit(housing_data_prepared, housing_labels)

housing_predictions = tree_reg.predict(housing_data_prepared)
tree_mse = mean_squared_error(housing_predictions, housing_labels)
tree_rmse = np.sqrt(tree_mse)
print("decision tree regressor rmse:", tree_rmse)

#test krzyzowy (kroswalidacja) dla obu modeli 
from sklearn.model_selection import cross_val_score
tree_reg_new = DecisionTreeRegressor()
scores  =  cross_val_score(tree_reg_new, housing_data_prepared, housing_labels, scoring="neg_mean_squared_error", cv=10)
tree_rmse_scores = np.sqrt(-scores)
print("decision tree crossvalidation regressor rmse:", tree_rmse_scores)
print("średnia:", scores.mean())
print("odchylenie standardowe:", scores.std())

lin_reg_new = LinearRegression()
scores_lin =  cross_val_score(lin_reg_new, housing_data_prepared, housing_labels, scoring="neg_mean_squared_error", cv=10)
reg_lin_rmse_scores = np.sqrt(-scores_lin)
print("linear regression crossvalidation regressor rmse:", reg_lin_rmse_scores)
print("średnia:", scores_lin.mean())
print("odchylenie standardowe:", scores_lin.std())


#model lasu losowego:
from sklearn.ensemble import RandomForestRegressor
forest_reg = RandomForestRegressor()
forest_reg.fit(housing_data_prepared, housing_labels)
forest_scores = cross_val_score(forest_reg, housing_data_prepared, housing_labels, scoring="neg_mean_squared_error", cv=10)
forest_rmse_scores = np.sqrt(-forest_scores)
print("random forest  regressor crossvalidation regressor rmse:", forest_rmse_scores)
print("średnia:", forest_rmse_scores.mean())
print("odchylenie standardowe:", forest_rmse_scores.std())

#dostrajanie modelu
from sklearn.model_selection import GridSearchCV 

param_grid = [
    {'n_estimators': [3, 10, 30], 'max_features': [2, 4, 6, 8]}, 
    {'bootstrap': [False], 'n_estimators': [3, 10], 'max_features': [2, 3, 4]}
]

forest_reg = RandomForestRegressor()
grid_search = GridSearchCV(forest_reg, param_grid, cv=5, scoring='neg_mean_squared_error')
grid_search.fit(housing_data_prepared, housing_labels)
print("najlepsze hiperparametry", grid_search.best_params_)

#ostateczny model
final_model = grid_search.best_estimator_
X_test = strat_test_set.drop("median_house_value", axis=1)
y_test = strat_test_set["median_house_value"].copy()

X_test_prepared = num_pipeline.transform(X_test)
final_predictions = final_model.predict(X_test_prepared)
final_mse = mean_squared_error(y_test, final_predictions)
final_rmse = np.sqrt(final_mse)
print("final rmse:", final_rmse)