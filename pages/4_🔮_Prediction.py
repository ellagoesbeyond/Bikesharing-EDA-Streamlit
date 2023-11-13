
# Autor: Elisabth Oeljeklaus
# Date: 2023-11-08

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import plotly.subplots as sp
import sys
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from math import sqrt
sys.path.append ('scripts/')
from scripts.important_pipeline import OutlierClipper
from scripts.important_pipeline import RemoveRedundant
from scripts.important_pipeline import MinMaxScaler


# Import the function from the module located in the specified directory
sys.path.append('scripts/pipeline_df_for_streamlit.py')  
from scripts.pipeline_df_for_streamlit import preped_data

sys.path.append ('defaults/defaultforlibs.py')
from defaults.defaultforlibs import default_plt

sys.path.append ('scripts/model_casual_user.py')
from scripts.model_casual_user import casual_model

sys.path.append ('scripts/model_registered_user.py')
from scripts.model_registered_user import registered_model


st.title("Joined Model Prediction")
final_df = pd.read_csv("bike_sharing_output.csv")
sns.lineplot
st.line_chart (final_df[["total_pred_lightgbm"]],)

tab1, tab2 = st.tabs(["Try it out!"])
 
with tab1: 
    st.header("Predict the hourly demand of bikes")

    st.subheader("Input Parameters for Prediction")

    # Add sliders for numerical inputs
    temp = st.slider('Temperature (in Celsius)', min_value=-10.0, max_value=40.0, value=20.0, step=0.5)
    humidity = st.slider('Humidity (in %)', min_value=0, max_value=100, value=50)
    windspeed = st.slider('Wind Speed (in km/h)', min_value=0.0, max_value=50.0, value=10.0, step=1.0)

    # Add input boxes for categorical or discrete values
    season = st.selectbox('Season', options=['Spring', 'Summer', 'Fall', 'Winter'])
    holiday = st.selectbox('Is it a holiday?', options=['Yes', 'No'])
    workingday = st.selectbox('Is it a working day?', options=['Yes', 'No'])
    weather = st.selectbox('Weather condition', options=['Clear', 'Mist', 'Light Rain/Snow', 'Heavy Rain/Snow'])
    hours = st.slider('Wind Speed (in km/h)', min_value=0, max_value=23, value=9, step=1)
    # Convert categorical inputs to numerical (if necessary)

    
    holiday = 1 if holiday == 'Yes' else 0
    workingday = 1 if workingday == 'Yes' else 0
    weekday = st.slider('Weekday (in km/h)', min_value=0, max_value=6, value=3, step=1)
    mnth = st.slider('Month (in km/h)', min_value=0, max_value=12, value=3, step=1)
    # Map other categorical variables as needed
    if season =="Spring":
        season = 1
    elif season =="Summer":
        season = 2
    elif season =="Fall":
        season = 3
    else:
        season = 4

    if weather =="Clear":
        weather = 1
    elif weather =="Mist":
        weather = 2
    elif weather =="Light Rain/Snow":
        weather = 3
    else:
        weather = 4

    user_input = pd.DataFrame(
        data=[[temp, humidity, windspeed, season, holiday, workingday, weather,hours,weekday, mnth]],
        columns=['temp', 'hum', 'windspeed', 'season', 'holiday', 'workingday', 'weathersit',"hr","weekday","mnth"]
    )
    target=st.multiselect("Select the hour of the day", options=["Casual", "Registered"])
    # Button to make prediction
    if st.button('Predict Demand'):
        st.spinner(text='Prediction progress...')
        col1, col2 = st.columns(2)
       
        user_input ["yr"]=1
        user_input['hour_sin'] = np.sin(2 * np.pi * user_input['hr'] / 12.0)  # to account for daily seasonality
        user_input['hour_cos'] = np.cos(2 * np.pi * user_input['hr'] / 12.0) # to account for daily seasonality
       
        st.write("Predicted Demand Total Users:")
        with col1:
        
            st.write("Predicted Demand CASUAL Users:")
            user_input['lag_2_casual']=35.964151 # to account for weekly seasonality
            user_input['lag_1_registered']=35.964906	 # to account for daily seasonality
            user_input['lag_2_registered']=35.964151	# to account for weekly seasonality
            user_input['lag_1_casual']=35.964906 # to account for daily seasonality
            
            user_input['lag_week_casual']=35.890651 # to account for weekly seasonality
            user_input['lag_24_casual']=35.948173 # to account for daily seasonality
            user_input['lag_week_registered']=35.890651	# to account for weekly seasonality
            user_input['lag_24_registered']=35.948173 # to account for daily seasonality

            casual_pipeline,dates_train,dates_test,y_train,y_test,Y_pred,val_scores,test_scores=casual_model()
            Y_pred_casual=casual_pipeline.predict(user_input)
                    
            st.dataframe(Y_pred_casual)
                   
        with col2:
            st.write("Predicted Demand REGISTERED Users:")
            user_input['lag_2_casual']=154.791238 # to account for weekly seasonality
            user_input['lag_1_registered']=154.793737	 # to account for daily seasonality
            user_input['lag_2_registered']=154.791238	# to account for weekly seasonality
            user_input['lag_1_casual']=154.793737 # to account for daily seasonality
            user_input['lag_week_casual']=154.693219 # to account for weekly seasonality
            user_input['lag_24_casual']=154.742781 # to account for daily seasonality
            user_input['lag_week_registered']=154.693219	# to account for weekly seasonality                user_input['lag_24_registered']=154.742781 #to account for daily seasonality
                            
            registered_pipeline,dates_train,dates_test,y_train,y_test,Y_pred,val_scores,test_scores=registered_model()
            Y_pred_registered=registered_pipeline.predict(user_input)
            st.dataframe(Y_pred_registered)

        st.divider()
        st.write("Predicted Demand Total Users:")
        st.write(Y_pred_casual+Y_pred_registered)           


















        
