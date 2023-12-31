import streamlit as st
from defaults.defaultforlibs import add_logo

add_logo()

st.title("Washington D.C Bike Sharing")
st.write("# Welcome to Group 6 Analysis 👋🚲")
st.image("pics/header.jpg", use_column_width=True)

#st.sidebar.success("Select a page.")
st.markdown("""
      *The administration of Washington D.C wants to make a deeper analysis of the usage of the bike-sharing service present in the city 
      in order to build a predictor model that helps the public transport department anticipate better the provisioning of bikes in the city. 
      For these purposes, some data is available for the years 2011 and 2012.*

      **👈 Select a page you are interested in from the side bar**""")
col1, col2 = st.columns(2)
with col1:
  st.markdown(
      """
      ### Table of Contents
      1. [Data Exploration](https://bikesharing-group06.streamlit.app/Analysis)
      2. [Data Visualization](https://bikesharing-group06.streamlit.app/Visualization)
      3. [Data Modelling](https://bikesharing-group06.streamlit.app/Modelling)
      4. [Data Prediction](https://bikesharing-group06.streamlit.app/Prediction)
      5. [Data Conclusion](https://bikesharing-group06.streamlit.app/Actions)
      """
    
  )
with col2:
  st.image("pics/qr_code.png", width=200)
my_expander = st.expander(label='Assignment Instructions')
with my_expander:
  """ **Main goals**
As part of the team of the hired consultancy firm, your work is to build an interactive, insightful and complete report about the bike-sharing service in the city for the head of transportation services of the local government. As part of the requirements, there are two big points:
- The customer wants a deep analysis of the bike-sharing service. He wants to know how the citizens are using the service in order to know if some changes must be made to optimize costs or provide a better service.
- The customer also wants a predictive model able to predict the total number of bicycle users on an hourly basis. They believe this will help them in the optimization of bike provisioning and will optimize the costs incurred from the outsourcing transportation company.
    """
   
my_expander_2 = st.expander(label='Group Members')
with my_expander_2:
  """ **Members**
- Ekiomoado Akhigbe
- Elisabeth Oeljeklaus 
- Leopold von Hugo
- Rahul Rohilla 
- Kohei Hayashi
- Raouf Ammar

*Please note this is a final group assignment for Python 2 course by Daniel Garcia Hernandez at IE School of Science and Technology*
    """

