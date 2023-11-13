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
from math import sqrt
sys.path.append ('02/group06/')
from important_pipeline import OutlierClipper
from important_pipeline import RemoveRedundant
from important_pipeline import MinMaxScaler


# Import the function from the module located in the specified directory
sys.path.append('02/group06/pipeline_df_for_streamlit.py')  
from pipeline_df_for_streamlit import preped_data

sys.path.append ('02/group06/defaultforlibs.py')
from defaultforlibs import default_plt

sys.path.append ('02/group06/model_casual_user.py')
from model_casual_user import casual_model

sys.path.append ('02/group06/model_registered_user.py')
from model_registered_user import registered_model
# Title
st.title("Modelling")
st.subheader("Overview ⚗️ ")

multi="""
We tested 3 approaches as you can see in the tab. \n
1. **Predicting the casual user rides** \n
2. **Predicting the registered user rides** \n
3. **Joining the models output** \n

➡️ And for each approach we tested a **Lightgbm** and **XGBoost** model to find the perfect performance.\n
➡️ The last 30 % of the data was used for testing and the rest for training."""
st.markdown(multi)

st.subheader('Boosting 🪃')
st.write ("Here you see what boosting is and how it works")
st.markdown("![Alt Text](https://substackcdn.com/image/fetch/f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2Fe5658e59-9617-4ff8-bc76-880c9b3aa77f_722x472.gif)")
mutli="""
- is an iterative training process
- the subsequent model puts more focus on misclassified samples from the previous model
- the final prediction is a weighted combination of all predictions"""

st.markdown(mutli)
expander =st.expander("Explanation Refit Metric")
with expander:
    multi="""
    ### Why We Choose RMSE Over MAPE and MAE for Model Refitting

    When refining our predictive models, it's crucial to select the right metric that aligns with our business objectives. Here's why we opted for RMSE (Root Mean Square Error) over MAPE (Mean Absolute Percentage Error) and MAE (Mean Absolute Error):

    **1. Limitations of MAPE:** 
    - **Issue with Zero Actual Values:** MAPE becomes problematic when dealing with actual values of zero. If the actual value is 0 and the predicted value is even slightly off (say, 0.0001), MAPE skyrockets to infinity, as it involves dividing by the actual value. This makes MAPE unreliable for our dataset.

    **2. Choosing RMSE Over MAE:**
    - **Penalizing Larger Errors:** While MAE provides a straightforward average of errors, it doesn't distinguish between small and large errors. For our purpose, larger errors are more critical as they can lead to significant business costs, like unnecessary investments in new bikes based on inflated demand forecasts.
    - **Emphasizing Accuracy:** RMSE, by squaring the errors before averaging, gives more weight to larger errors. This means that a few large errors will have a substantial impact on RMSE, encouraging models that are consistent and accurate across all predictions. 
    - **Business Context:** In our scenario, it's more beneficial to be consistently close to the right prediction rather than being exactly right most of the time but significantly off on rare occasions. The costs of such rare but large errors can be substantial, making RMSE a more suitable choice.

    By focusing on RMSE, we aim to build a model that minimizes large errors, ensuring more reliable and cost-effective predictions for our business needs.
    """
    st.markdown(multi)
st.divider()

#Create Tabs on sidebar 
st.sidebar.title("Options")

#define the target options
target_options = st.sidebar.radio("Choose what you wanna see",["Preprocessing Steps","Casual Users", "Registered Users"])

if target_options == "Preprocessing Steps":
    st.header("Preprocessing Steps")
    multi="""
   The pipeline consists of the following steps:\n
    1. **Clipping Outliers**: Modify extreme data values to reduce their impact.\n
    2. **Creating Polynomial Features**: Generate new features by combining and transforming existing ones.\n
    3. **Removing Redundant Features**: Eliminate features that are highly correlated with others.\n
    4. **MinMax Scaling**: Normalize the data to a consistent range for all features.\n
    5. **Fitting the Model**: Train different models using the processed data.\n
    ➡️ This approach streamlines the process of applying a consistent set of preprocessing steps to various models, enhancing efficiency and accuracy."""
    st.markdown(multi)
    expander1 =st.expander("Pipeline")
    with expander1:
        st.write( "The following is our created pipeline with a **list comprehension of pipelines** to iterate over different models and apply pipeline processing steps")
        code = '''model_running = [ Pipeline([

                ('poly', Polynomial_trans),  # Feature generation (polynomial features)
                ('clipper',OutlierClipper()), # Clip values to factor*IQR
                ('RmRedundant',RemoveRedundant(redundant=0.8)),  # edundacy as of 0.7 corr between predictors  (was 0.7)
                ('scaling',MinMaxScaler()),  # MinMax Scaling of ALL variables
                (model_name,model)   # respective model iterated through in the list comprehension
                ])
            for model_name,model in models
            ]'''
        st.code(code, language='python')

    expander3 =st.expander("Variable Importance")
    with expander3:
        st.write("The following are the variable importance plots for each user group")
        st.write("This is without feature engineering")
        # Variable Importances of target=casual
        data = preped_data()
        correlation_matrix= data.corr()
        variable_importance=pd.DataFrame()
        target='casual'
        variable_importance[target]=abs(correlation_matrix.loc[~correlation_matrix.index.isin([target,'registered','cnt']),target]).sort_values(ascending=False)

        # Variable Importances of target=registered
        target='registered'
        variable_importance[target]=abs(correlation_matrix.loc[~correlation_matrix.index.isin([target,'casual','cnt']),target]).sort_values(ascending=False)
        # joint variable importance
        sorted_df = variable_importance.sort_values(by=['casual', 'registered'], ascending=False)
        st.dataframe(sorted_df.loc[:, ['casual', 'registered']])
        #st.dataframe(pd.DataFrame(variable_importance.loc[:,['casual','registered']]))

    expander2 =st.expander("Feature Engineering")
    with expander2:
        ## Feature Engineering Summary
        multi=("""
        The feature engineering process introduces new features to enhance the predictive power of the model for forecasting 'casual' and 'registered' user counts.
        
        1. **Lagged Features for 'Casual' and 'Registered':**
        - `lag_1_casual`: Represents the 'casual' count shifted by 1 time step (e.g., previous hour).
        - `lag_2_casual`: Represents the 'casual' count shifted by 2 time steps.
        - `lag_1_registered`: Represents the 'registered' count shifted by 1 time step.
        - `lag_2_registered`: Represents the 'registered' count shifted by 2 time steps.
               
        ➡️ **These features help capture immediate past trends.**

        2. **Seasonality Features:**
        - `hour_sin` & `hour_cos`: These are sinusoidal transformations of the hour of the day to capture the daily seasonality in a continuous manner. `hour_sin` and `hour_cos` provide the sine and cosine transformations, respectively, facilitating the model to understand the cyclic nature of daily patterns.
        - `lag_week_casual`: Represents the 'casual' count shifted by 168 hours (24 hours * 7 days), capturing weekly seasonality.
        - `lag_24_casual`: Represents the 'casual' count shifted by 24 hours, capturing daily seasonality.
        - `lag_week_registered`: Similar to `lag_week_casual` but for 'registered' count.
        - `lag_24_registered`: Similar to `lag_24_casual` but for 'registered' count.
               
        ➡️ **These features are crucial to model periodic patterns on daily and weekly bases.**"""
        )
        st.markdown(multi)
    

if target_options == "Casual Users":
    st.header("Casual Users")
    casual_pipeline,dates_train,dates_test,y_train,y_test,Y_pred,val_scores,test_scores=casual_model()
    multi=("""We tried out different models for the casual users. The best scores were achieved with the **LightGBM** model.""")
    st.markdown(multi)
    #show picture of the model pipeline 
    st.subheader("Pipeline")
    st.write("This is the pipeline for the casual users LightGBM model!")
    st.image("overview_casual_model_pipeline.png")

    st.divider()
    # Print metrics
    st.subheader("Metrics")
    st.write("The following are the metrics for the casual users LightGBM model!")
    col1,col2 =st.columns(2)
    with col1:
        st.write("Validation Scores")
        st.write(f"RMSE: {round(val_scores['RMSE'],2)}")
        st.write(f"MAE: {round(val_scores['MAE'],2)}")
        st.write(f"MAPE: {round(val_scores['MAPE'],2)}")
    with col2:
        st.write("Test Scores")
        st.write(f"RMSE: {round(test_scores['RMSE'],2)}")
        st.write(f"MAE: {round(test_scores['MAE'],2)}")
        st.write(f"MAPE: {round(test_scores['MAPE'],2)}")
    
    st.divider()
    
    mulit="""
    **TRAIN** Dates where: \n
    - Start: 2011-01-01 \n
    - End: 2012-05-28 \n
    **TEST** Dates where: \n
    - Start: 2012-05-29 \n
    - End: 2012-12-31 \n
    """
    st.markdown(mulit)
    #plot
    plt.figure(figsize=(35, 10))
    plt.title(f"LIGHTXBG Forecast period acutal vs. error")
    sns.lineplot(x=dates_test,y= y_test, color='seagreen', label='actual')
    sns.lineplot(x=dates_test, y=Y_pred, color='darksalmon', linestyle='dashed', label='Forecast')    
    plt.title(f"LIGHTXBG Forecast period acutal vs. error")
    st.pyplot(plt)

    # Plot historical data (train data)
    plt.figure(figsize=(35, 10))
    sns.lineplot(x=dates_train,y= y_train, label='Observed', color='seagreen')
    # plot actual
    sns.lineplot(x=dates_test,y= y_test, color='seagreen', label='actual')
    sns.lineplot(x=dates_test, y=Y_pred, color='darksalmon', linestyle='dashed', label='Forecast')   
    st.pyplot(plt)


if target_options == "Registered Users":

    st.header("Registered Users")
    registered_pipeline,dates_train,dates_test,y_train,y_test,Y_pred,val_scores,test_scores=registered_model()
    multi=("""We tried out different models for the casual users. The best scores were achieved with the **LightGBM** model.""")
    st.markdown(multi)
    #show picture of the model pipeline 
    st.subheader("Pipeline")
    st.write("This is the pipeline for the casual users LightGBM model!")
    st.image("overview_registered_model_pipeline.png")

    st.divider()
    mulit="""
    **TRAIN** Dates where: \n
    - Start: 2011-01-01 \n
    - End: 2012-05-28 \n
    **TEST** Dates where: \n
    - Start: 2012-05-29 \n
    - End: 2012-12-31 \n
    """
    st.markdown(mulit)
    # Print metrics
    st.subheader("Metrics")
    st.write("The following are the metrics for the casual users LightGBM model!")
    col1,col2 =st.columns(2)
    with col1:
        st.write("Validation Scores")
        st.write(f"RMSE: {round(val_scores['RMSE'],2)}")
        st.write(f"MAE: {round(val_scores['MAE'],2)}")
        st.write(f"MAPE: {round(val_scores['MAPE'],2)}")
    with col2:
        st.write("Test Scores")
        st.write(f"RMSE: {round(test_scores['RMSE'],2)}")
        st.write(f"MAE: {round(test_scores['MAE'],2)}")
        st.write(f"MAPE: {round(test_scores['MAPE'],2)}")
    
    st.divider()
    
    #plot
    plt.figure(figsize=(35, 10))
    plt.title(f"LIGHTXBG Forecast period acutal vs. error")
    sns.lineplot(x=dates_test, y=Y_pred, color='darksalmon', linestyle='dashed', label='Forecast')

    sns.lineplot(x=dates_test,y= y_test, color='seagreen', label='actual')
  
    
    st.pyplot(plt)

    # Plot historical data (train data)
    plt.figure(figsize=(35, 10))
    sns.lineplot(x=dates_train,y= y_train, label='Observed', color='seagreen')
    # plot actual
    sns.lineplot(x=dates_test,y= y_test, color='seagreen', label='actual')
    sns.lineplot(x=dates_test, y=Y_pred, color='darksalmon', linestyle='dashed', label='Forecast')
    st.pyplot(plt)


