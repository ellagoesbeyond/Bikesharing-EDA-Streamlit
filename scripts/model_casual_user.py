# Autor: Elisabth Oeljeklaus
# Date: 2023-11-11

from sklearn.preprocessing import MinMaxScaler, PolynomialFeatures
from sklearn.linear_model import Ridge
from xgboost import XGBRegressor as XGBoost
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
import lightgbm as lgb
from sklearn.base import BaseEstimator, TransformerMixin
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import plotly.subplots as sp
import pickle
import sys
from sklearn.metrics import mean_absolute_percentage_error as MAPE  
from sklearn.metrics import mean_absolute_error as MAE 
from sklearn.metrics import mean_squared_error as MSE 
from math import sqrt


# Import the function from the module located in the specified directory
sys.path.append('scripts/pipeline_df_for_streamlit.py')  
from scripts.pipeline_df_for_streamlit import preped_data

sys.path.append ('scripts/')
from scripts.important_pipeline import OutlierClipper
from scripts.important_pipeline import RemoveRedundant
from scripts.important_pipeline import MinMaxScaler


def casual_model():
    data = preped_data()


    # Split predictors and target
    target='casual'
    non_target='registered'

    # Based on results of the model we introduce AR(2) to the data and MA(4)
    data['lag_1_casual']=data[target].shift(1)
    data['lag_2_casual']=data[target].shift(2)
    data['lag_1_registered']=data[non_target].shift(1)
    data['lag_2_registered']=data[non_target].shift(2)


    data['hour_sin'] = np.sin(2 * np.pi * data['hr'] / 12.0)  # to account for daily seasonality
    data['hour_cos'] = np.cos(2 * np.pi * data['hr'] / 12.0) # to account for daily seasonality
    data['lag_week_casual']=data[target].shift(24*7)  # to account for weekly seasonality
    data['lag_24_casual']=data[target].shift(24) # to account for daily seasonality
    data['lag_week_registered']=data[non_target].shift(24*7)  # to account for weekly seasonality
    data['lag_24_registered']=data[non_target].shift(24)  # to account for daily seasonality


    #data['ma_4']=data[target].rolling(window=4).mean()

    data.dropna(inplace=True)  # drop the rows that are NA created through the lagging. As the dataset is quite extensive dropping is no big thing

    X=data.drop(columns=[target,'dteday','datetime','cnt',non_target],inplace=False)
    y=data[target]


    print(X.dtypes)
    print(X.shape,y.shape)

    train_portion=0.7
    train_set_size=int(train_portion*len(X))
    X_train=X[:train_set_size]
    y_train=y[:train_set_size]


    X_test=X[train_set_size:]
    y_test=y[train_set_size:]

    # here i want to u the pipline with the following estimators. 

    # Load the dictionary from the file
    with open('models/casual_best_estimator.pkl', 'rb') as file:
        pipelines = pickle.load(file)

    # Now you can use the pipelines as needed
    # For example, to use the LightGBM pipeline:
    casual_pipeline = pipelines['lightgbm']

    # use the lightgbm pipeline to predict
    Y_pred = casual_pipeline.predict(X_test)

    #read the best cv resutls 
    best_cv_results=pd.read_csv('models/best_cv_results_casual_lightbm.csv',index_col=0)
    val_scores={}
    val_scores['RMSE'] = best_cv_results[best_cv_results['rank_test_RMSE'] == 1]['mean_test_RMSE'].values[0]
    val_scores['MAPE'] = best_cv_results[best_cv_results['rank_test_RMSE'] == 1]['mean_test_MAPE'].values[0]
    val_scores['MAE'] = best_cv_results[best_cv_results['rank_test_RMSE'] == 1]['mean_test_MAE'].values[0]


    test_scores={}
    test_scores['RMSE']=sqrt(MSE(y_test,Y_pred))
    test_scores['MAPE']=MAPE(y_test,Y_pred)
    test_scores['MAE']=MAE(y_test,Y_pred)


    dates_train=pd.to_datetime(data.iloc[0:len(X_train),0])
    dates_test=pd.to_datetime(data.iloc[len(X_train):,0])

    return casual_pipeline,dates_train,dates_test,y_train,y_test,Y_pred,val_scores,test_scores

