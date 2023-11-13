# Autor: Elisabth Oeljeklaus
# Date: 2023-11-07

import matplotlib.pyplot as plt
import plotly.express as px
#use the code below  for import in the other files
def default_plt():
    #viridis_colors = px.colors.sequential.Viridis  # Access the Viridis sequential color scale
    colors = {
    'casual': "#440154",      # First color in the viridis palette
    'registered':"#5ec962"  # Fifth color in the viridis palette
    }
    
    font= plt.rcParams['font.family'] = 'IBM Plex Sans'
    plt.rcParams['font.size'] = 12
    plt.rcParams['axes.labelsize'] = 12
    plt.rcParams['axes.labelweight'] = 'bold'
    plt.rcParams['axes.titlesize'] = 14
    plt.rcParams['axes.titleweight'] = 'bold'
    plt.rcParams['xtick.labelsize'] = 10
    plt.rcParams['ytick.labelsize'] = 10
    plt.rcParams['legend.fontsize'] = 10
    plt.rcParams['figure.titlesize'] = 16
    plt.rcParams['figure.titleweight'] = 'bold'
    plt.rcParams['figure.figsize'] = [10, 6]
    plt.rcParams['figure.dpi'] = 100
    plt.rcParams['figure.facecolor'] = '#ebedf0'
    return colors