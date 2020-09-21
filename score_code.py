import pandas as pd
import numpy as np
import sys
import pickle
import xgboost as xgb
#model evaluation metrics
from sklearn.metrics import mean_squared_error, r2_score
from math import sqrt

# define model evaluation rmse function
def rmse(y, pred):
    rmse = sqrt(mean_squared_error(y, pred))
    return rmse

# define accuracy evaluation function
def accuracy(y, pred, cut_off=3):
    accuracy = (np.abs(y - pred) <= cut_off).astype(int)
    return np.sum(accuracy)/len(accuracy)

# define model evaluate function
def model_evaluate(reg, X, y):

    #prediction results
    pred = reg.predict(X)
    pred[pred > 850] = 850
    pred[pred < 300] = 300
    temp_y = y.ravel()

    rmse_val = rmse(temp_y, pred)
    acc_val = accuracy(temp_y, pred, 3)
    r2_val = r2_score(temp_y, pred)

    print("\n")
    print("RMSE - ", rmse_val)
    print("\n")
    print("Accuracy - ", acc_val)
    print("\n")
    print("R squared - ", r2_val)

    return rmse_val, acc_val, r2_val, pred
