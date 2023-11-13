# Autor: Elisabth Oeljeklaus
# Date: 2023-11-07

import matplotlib.pyplot as plt
import plotly.express as px
import streamlit as st
#use the code below  for import in the other files
def default_plt():
    #viridis_colors = px.colors.sequential.Viridis  # Access the Viridis sequential color scale
    colors = {
    'casual': "#440154",      # First color in the viridis palette
    'registered':"#5ec962"  # Fifth color in the viridis palette
    }
    
    
    return colors

def add_logo():

    st.markdown(
        """
        <style>
            [data-testid="stSidebarNav"] {
                background-image: url(https://images.squarespace-cdn.com/content/v1/5a7a33842aeba5b53517c812/1521604039742-KVLQSXYYUIM1JW98VXF9/CCC.png?format=2500w);
                background-repeat: no-repeat;
                padding-top: 120px;
                background-position: 30px 30px;
            }
            [data-testid="stSidebarNav"]::before {
                content: "";
                margin-left: 20px;
                margin-top: 20px;
                font-size: 30px;
                position: relative;
                top: 100px;
            }
        </style>
        """,
        unsafe_allow_html=True,
    )
