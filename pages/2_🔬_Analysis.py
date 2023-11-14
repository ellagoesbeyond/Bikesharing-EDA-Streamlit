# Autor: Elisabth Oeljeklaus
# Date: 2023-11-07

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly as px
import sys

sys.path.append('scripts/pipeline_df_for_streamlit.py')  
from scripts.pipeline_df_for_streamlit import preped_data

sys.path.append ('defaults/defaultforlibs.py')
from defaults.defaultforlibs import default_plt

import streamlit as st
import pandas as pd
from defaults.defaultforlibs import add_logo
add_logo()
# defaults for plots
default_plt()
df= preped_data()

# Titel of the App
st.title("EDA of Bike Sharing Dataset")
st.header("Overview 📊")
col1,col2,col3 = st.columns(3,gap="small")
with col1:
    date_range = pd.date_range(start=df['dteday'].min(), end=df['dteday'].max())
    num_days = len(date_range)
    st.metric(label="Days", value=num_days)
    st.write("---")
    st.metric(label="Entries", value=df.shape[0])
with col2:
    st.metric(label="Dates between",value=df['dteday'].min().strftime("%Y-%m-%d"))
    st.write("and")
    st.metric(label="",value=df['dteday'].max().strftime("%Y-%m-%d"))

with col3:  
    # Amount of Columns
    st.metric(label="Amount of Columns", value=df.shape[1])
    st.write("---")
    # Amount of Null Values
    st.metric(label="Amount of Null Values", value=df.isnull().sum().sum())

expand_col_def = st.expander(label='Column Discription!')
with expand_col_def:
    st.markdown(""" 
        **The following are the columns our csv files contains:**
         - `instant`: record index
        - `dteday` : date
        - `season` : season 1:winter, 2:spring, 3:summer, 4:fall
        - `yr` : year (0: 2011, 1:2012)
        - `mnth` : month ( 1 to 12)
        - `hr` : hour (0 to 23)
        - `holiday` : weather day is holiday or not (extracted from http://dchr.dc.gov/page/holiday-schedule)
        - `weekday` : day of the week
        - `workingday` : if day is neither weekend nor holiday is 1, otherwise is 0.
        - `weathersit` :
                - 1: Clear, Few clouds, Partly cloudy
                - 2: Mist + Cloudy, Mist + Broken clouds, Mist + Few clouds, Mist
                - 3: Light Snow, Light Rain + Thunderstorm + Scattered clouds, Light Rain + Scattered clouds
                - 4: Heavy Rain + Ice Pallets + Thunderstorm + Mist, Snow + Fog
        - `temp` : Normalized temperature in Celsius. The values are divided to 41 (max)
        - `atemp`: Normalized feeling temperature in Celsius. The values are divided to 50 (max)
        - `hum`: Normalized humidity. The values are divided to 100 (max)
        - `windspeed`: Normalized wind speed. The values are divided to 67 (max)
        - `casual`: count of casual users
        - `registered`: count of registered users
        - `cnt`: count of total rental bikes including both casual and registered
            """ )

st.divider()
# Sidebar
st.sidebar.title("Options")
#define the target options
target_options = st.sidebar.radio("Choose what you want to see",["Overview 📊","First Analysis", "Outlier Analysis", "Correlation Analysis"])

if target_options == "Overview 📊":
    # 1. DATA OVERVIEW 
    st.header("Data Overview 📊")
    df= preped_data()

    plt.figure(figsize=(35, 10))
    plt.title('Targets daily mean ')
    df_agg= pd.DataFrame(df.groupby(['dteday'])[['registered','casual']].sum())
    plt.title('Targets daily')
    sns.lineplot(data=df_agg,x='dteday',y='registered',label='registered',color='black',errorbar=None)
    sns.lineplot(data=df_agg,x='dteday',y='casual',label='casual',color='green',errorbar=None)
    plt.legend()
    st.pyplot(plt)
    st.divider()
    st.dataframe(df)


if target_options == "First Analysis":
    #  2. KPI's (Amount of Row, Consitency of Data, Amount of Columns, Amount of Null Values, Amount of Duplicates)
    st.header("First Analysis")  
    st.subheader("Here you can see first analysis we got of the data")
    
    # average amount of bike used per user group 
    st.markdown(f"""Amount of Duplicates:   **{df.duplicated().sum()}**""")
    # Consitency of Data
    # Generate a sequence of dates from start date to end date
    start_date = '2011-01-01 00:00:00'
    end_date = '2012-12-31 00:00:00'
    date_range = pd.date_range(start=start_date, end=end_date, freq='H')  # Assuming hourly frequency
    # Check for missing dates
    missing = set(date_range) - set(df['datetime'])
    # If there are missing dates, the 'missing_dates' set will contain those dates
    if missing:
        print("Missing dates in 'dteday' column:")
        print(sorted(missing))

    else:
        print("No missing dates in 'dteday' column.")


    len(missing)
    st.markdown(f"""Dates not present in Dataset: **{len(missing)}**.\n  --> All of which ar on the weekend.""")

        
    df_agg= df.groupby(['dteday'])[['registered','casual']].sum()
    # calculate average amount of bike used per user group
    casual_mean = df['casual'].mean()
    registered_mean = df['registered'].mean()
    # print results
    st.markdown(f"Average HOURLY Bike Use per Casual User: **{int(casual_mean)}**")
    st.markdown(f"Average HOURLY Bike Use per Registered User: **{int(registered_mean)}**")

    st.divider()
    # Donut Chart count of registered and casual users in dataset
    fig = px.express.pie(df, values=[df['casual'].sum(),df['registered'].sum()], names=['casual','registered'], title='Count of registered and casual users in dataset')
    st.plotly_chart(fig)

    st.divider()
    plt.figure()
    plt.title('Missing Entries/Data points in Dataset')
    plt.plot(df['datetime'], df['cnt'])
    # Add missing values as points
    for date in missing:
        plt.scatter(date, 0, color='red', marker='x')

    # Use st.pyplot to display the Matplotlib plot in Streamlit
    st.pyplot(plt)
    

    st.divider()
    

    col1,col2,col3 = st.columns(3,gap="small")
    with col1:
        # Average Amount of Rides per Weekday per User Group number 
        st.subheader("Amount of Rides per Weekday per User Group")
        # create a new dataframe with the amount of rides per weekday per user group
        df_agg= pd.DataFrame(df.groupby(['weekday'])[['registered','casual']].mean())
        # plot the dataframe
        st.dataframe(df_agg)
    with col2: 
        # Average Amount of Rides per Month per User Group
        st.subheader("Amount of Rides per Month per User Group")
        # create a new dataframe with the amount of rides per month per user group
        df_agg= pd.DataFrame(df.groupby(['mnth'])[['registered','casual']].mean())
        # plot the dataframe
        st.dataframe(df_agg)
    with col3:
        st.subheader("Amount of Rides per Hour per User Group")
        # create a new dataframe with the amount of rides per hour per user group
        df_agg= pd.DataFrame(df.groupby(['hr'])[['registered','casual']].mean())
        # plot the dataframe
        st.dataframe(df_agg)

    
if target_options == "Outlier Analysis":
    # 3. OUTLIERS
    st.header("Outlier Analysis")
    tab1, tab2 = st.tabs(["Summary 📄", "Insights 👀"])
    with tab1:
        st.subheader("Summary of Outlier Analysis")
        multi= ("""
        1. `Casual`, `registered`, and `count` categories display significant outliers, with values reaching maximum extremes.
        2. `Weather situations` are predominantly categorized as 1 or 2, with category 3 being less common and category 4 extremely rare.
        3. `Humidity` levels at 0 are very uncommon in the dataset.
        4. `Wind speed` typically ranges between 1.8 to 0.23, with a few notable extremes, particularly values above the 75th percentile, which range from 0.5 to 0.8.
        5. `Casual users` are generally observed in the range of 0 to 50, with the 75th percentile around 110, but there are some outliers stretching up to 360.
        6. `Registered users` usually fall between 50 to 210, with the 75th percentile nearing 460; however, many outliers exist, reaching as high as approximately 800 times the typical values.
        7. The `count of total users` typically lies between 100 to 500, with the 75th percentile at about 700, but numerous outliers are present, with the maximum reaching around 900 times the median values."""
        )
        st.markdown(multi)
    with tab2:
        expand_boxplot = st.expander(label='Boxplot Analysis')
        with expand_boxplot:
            st.subheader("Here you can see the distribution of the data in the columns")
            
            # Create a dictionary mapping original column names to user-friendly names
            column_mapping = {
                'hum': 'humidity',
                'weathersit': 'weather situation',
                "atemp": "feels like temperature",
                "windspeed": "wind speed",
                "cnt": "count",
                "temp": "temperature",
                "yr": "year",
                "mnth": "month",
                "holiday": "holiday",
                "weekday": "weekday",  
                "workingday": "workingday",
                "season": "season",
                "instant": "instant",
                "dteday": "date",
                "casual": "casual users",
                "registered": "registered users",
                "hr": "hour"
                # Add more mappings as needed
            }
            
            # Dropdown for selecting column with user-friendly names
            selected_column = st.selectbox("Select a Column", list(column_mapping.values()))
            
            # Reverse the mapping to get the original column name based on user-friendly name
            original_column_name = [key for key, value in column_mapping.items() if value == selected_column][0]
            
            # Display box plot based on user selection

            plt.title(selected_column)  # Set the title based on user-friendly column name
            sns.set(rc={'figure.figsize':(10,6)})  # Set the plot size
            sns.boxplot(data=df[original_column_name].values)
            plt.xlabel(original_column_name)  # Set the x-axis label
            plt.ylabel("Values")  # Set the y-axis label
            st.pyplot(plt)  # Display the plot in Streamlit app

        # Insights for Outlier Analysis
        st.subheader("Insights for Outlier Analysis")
        
        multi= ("""
        **OBSERVATION**:
        - Casual, registered and count have some severe outliers to max
        - `weather situation` --> mostly between 1-2 rarely 3 and very rare is 4
        - `humidity` --> very rare at 0
        - `wind speed` : mostly between 1.8-0.23 a couple of extreme values (above 75Percentile at 0.5 above up until 0.8x)
        - `casual users`: mostly 0-50 75Percentile~110, some outliers above up until 360
        - `registered users`: mostly between 50-210 75% at ~460 , many outliers above , max is 800x
        - `count`: mostly between 100-500 75% at ~700, many outliers above, max is 900x"""
        )
        st.markdown(multi)
        
  
    
if target_options == "Correlation Analysis":
    # 4. CORRELATION ANALYSIS
    st.header("Correlation Analysis")
    correlation_matrix= df.corr()
    plt.figure(figsize=(10, 6))  
    sns.heatmap(df.corr())
    st.pyplot(plt)
    st.subheader("Summary of Correlation Analysis")
    multi= ("""
    **SUMMARY**:
        
    1. `Casual users`: Most influence of bike usage has the **temperature**.
    2. `Registered users`: Most influence of bike usage has the **hour of the day**""")
    st.markdown(multi)
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

  








