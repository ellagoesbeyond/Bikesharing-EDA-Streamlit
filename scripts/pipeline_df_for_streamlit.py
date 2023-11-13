# Autor: Elisabth Oeljeklaus
# Date: 2023-11-07
import pandas as pd
import numpy as np

def preped_data():
    # Define file paths
    input_file = 'bike-sharing-hourly.csv'

    # Read in input file
    data = pd.read_csv(data/input_file)

    data.set_index('instant',inplace=True) # set instant as index as this is the index of the dataset

    data['hr_2']=data['hr'].astype(str).str.zfill(2) # add a leading zero to the hour to have a consistent format
    data['datetime']=data['dteday']+' '+data['hr_2']+':00' # create a datetime column
    data['datetime']=pd.to_datetime(data['datetime']) # convert the datetime column to datetime format
    data['dteday']=pd.to_datetime(data['dteday']) # convert the dteday column to datetime format
    data.drop(columns='hr_2',inplace=True) # drop the hr_2 column
    return data


