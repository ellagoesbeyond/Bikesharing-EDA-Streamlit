# Autor: Elisabth Oeljeklaus
# Date: 2023-11-08

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import plotly.subplots as sp
import sys
import streamlit as st
import  streamlit_toggle as tog

# Import the function from the module located in the specified directory
sys.path.append('02/group06/pipeline_df_for_streamlit.py')  
from pipeline_df_for_streamlit import preped_data

sys.path.append ('02/group06/defaultforlibs.py')
from defaultforlibs import default_plt

# defaults for plots
colors=default_plt()

# Load the data
data = preped_data()

tab1, tab2 = st.tabs(["📈 Chart", "🗃 Summary"])

with tab1:
     
    st.title("Insights about our User Groups")
    import streamlit as st

    # Set the sidebar title
    st.sidebar.title("Options")
    #define the target options
    target_options = st.sidebar.multiselect("Select User Groups ", ["Casual users", "Registered users"])

    st.divider()

    tabs = ["Choose","Hourly Usage", "Weekly Usage", "Monthly Usage", "Holidays and Workingdays", "Seasons","Weather"]
    selected_tab = st.sidebar.selectbox("User Group Behaviour by ", tabs)
   
    # Display the selected tab
    if selected_tab == "Choose":
        st.write("Please select a Topic you want to gather Insights about.")

    elif selected_tab == "Hourly Usage":
        st.header(f"by {selected_tab}")
        st.subheader("OBSERVATION")
        st.markdown("As suspected we see 2 completely different user behaviours.")
        st.markdown("- `Casual users`: peak in the afternoon (hence likely to be tourists, people who use the service for leisure).")
        st.markdown("- `Registered users`: peak in the morning and the evening (hence likely to be working people, people who use the service to get to work or school).")
        selected_metrics = []    
        if "Casual users" in target_options:
            selected_metrics.append('casual')
        if "Registered users" in target_options:
            selected_metrics.append('registered')
        if selected_metrics:
            fig = px.bar(data, x='hr', y=selected_metrics, labels={'value': 'Amount of Users', 'hr': 'Hour'},barmode="group", color_discrete_map=colors)
            fig.update_layout(legend_title_text='User Groups')
            st.plotly_chart(fig)
        else:
            st.write("Please select at least one user type.")

    elif selected_tab == "Weekly Usage":
        st.header(f"by {selected_tab}")
        st.subheader("OBSERVATION")
        st.markdown("As suspected we see 2 completely different user behaviours.")
        st.markdown("- `Casual users`: peak during the weekends (hence likely to be all sorts of people, who use the service for leisure).")
        st.markdown("- `Registered users`: peak during the week (hence likely to be people with a similar daily routine during the week, like commute to work).")
        selected_metrics = []
        if "Casual users" in target_options:
            selected_metrics.append('casual')
        if "Registered users" in target_options:
            selected_metrics.append('registered')
        if selected_metrics:
            fig = px.bar(data, x='weekday', y=selected_metrics, labels={'value': 'Amount of Users', 'weekday': 'Day of Week'},barmode="group",color_discrete_map=colors)
            fig.update_xaxes(type='category', tickmode='array', tickvals=[0, 1, 2, 3, 4, 5, 6], ticktext=['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'])
            fig.update_layout(legend_title_text='User Groups')
            st.plotly_chart(fig)
        else:
            st.write("Please select at least one user type.")

    elif selected_tab == "Monthly Usage":
        st.header(f"by {selected_tab}")
        st.subheader("OBSERVATION")
        st.markdown("We see the same user behaviour for both groups.")
        st.markdown("- `Casual users` and `Registered users`: Are using the bike sharing service more between June till September with almmost constant use.")
        st.markdown("- Only noting that registered users use the bikes slightly less in July compared to the other months from June till September.")
        selected_metrics = []
        if "Casual users" in target_options:
            selected_metrics.append('casual')
        if "Registered users" in target_options:
            selected_metrics.append('registered')
        if selected_metrics:
            fig = px.bar(data, x='mnth', y=selected_metrics, labels={'value': 'Amount of Users', 'mnth': 'Month'},barmode="group",color_discrete_map=colors)
            fig.update_xaxes(type='category', tickmode='array', tickvals=list(range(1, 13)), ticktext=['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'])
            st.plotly_chart(fig)
        else:
            st.write("Please select at least one user type.")

    elif selected_tab == "Holidays and Workingdays":
        st.header(f"by {selected_tab}")
        st.subheader("OBSERVATION")
        st.markdown("As suspected we see 2 completely different user behaviours")
        st.markdown("- `Casual users`: Use the bike sharing service more on weekends and holidays.")
        st.markdown("- `Registered users`: Use the bike sharing service more on working days.")
        selected_metrics = []
        
        if "Casual users" in target_options:
            selected_metrics.append('casual')
        if "Registered users" in target_options:
            selected_metrics.append('registered')

        if selected_metrics:
            # Create subplots: 1 row, 2 columns
            fig = sp.make_subplots(rows=1, cols=2, subplot_titles=('Holiday', 'Working Day'))
        
            #Plot Holiday Usage
            for metric in selected_metrics:
                fig.add_trace(
                    go.Bar(x=data['holiday'], y=data[metric], name=metric, marker_color=colors[metric]),
                    row=1, col=1
                )

            # Plot Working Day Usage
            for metric in selected_metrics:
                fig.add_trace(
                    go.Bar(x=data['workingday'], y=data[metric], name=metric, marker_color=colors[metric]),
                    row=1, col=2
                )
            fig.update_xaxes(type='category', tickmode='array', tickvals=[0, 1], ticktext=['Non-Holiday', 'Holiday'])
            fig.update_layout(legend_title_text='User Groups')
            st.plotly_chart(fig)
        
        else:
            st.write("Please select at least one user type.")

    elif selected_tab == "Seasons":
        st.header(f"by {selected_tab}")
        st.subheader("OBSERVATION")
        st.markdown("We see the same user behaviour for both groups.")
        st.markdown("- `Casual users` and `Registered users`: Use the bike sharing service more during fall and summer.")

        selected_metrics = []
        if "Casual users" in target_options:
            selected_metrics.append('casual')
        if "Registered users" in target_options:
            selected_metrics.append('registered')

        if selected_metrics:
            fig = px.bar(data, x='season', y=selected_metrics, labels={'value': 'Amount of Users', 'season': 'Season'},barmode="group",color_discrete_map=colors)
            fig.update_xaxes(type='category', tickmode='array', tickvals=[1, 2, 3, 4], ticktext=['Spring', 'Summer', 'Autumn', 'Winter'])
            fig.update_layout(legend_title_text='User Groups')
            st.plotly_chart(fig)
        else:
            st.write("Please select at least one user type.")

    elif selected_tab == "Weather":
        st.header(f"by {selected_tab}")
        st.subheader("OBSERVATION")
        st.markdown("We see as expected the same user behaviour for both groups.")
        st.markdown("- `Casual users` and `Registered users`:") 
        st.markdown("   - Are more likley to use the service when the weather is clear or mist.")
        st.markdown("   - Are more likley to use the service when its less windy and less humid.")
        st.markdown("   - Are more likley to use the service when the temperature is between 20-30 degrees.")
        st.markdown("--> The better the Weather the more both Groups use the service")

        expander = st.expander("Click here to see the weather conditions")
        expander.markdown("1: **Clear** --> Description: Clear, Few clouds, Partly cloudy")
        expander.markdown("2: **Mist** --> Description: Mist + Cloudy, Mist + Broken clouds, Mist + Few clouds, Mist")
        expander.markdown("3: **Light Precipitation** --> Description: Light Snow, Light Rain + Thunderstorm + Scattered clouds, Light Rain + Scattered clouds")
        expander.markdown("4: **Heavy Precipitation** --> Description: Heavy Rain + Ice Pallets + Thunderstorm + Mist, Snow + Fog")

        selected_metrics = []
        if "Casual users" in target_options:
            selected_metrics.append('casual')
        if "Registered users" in target_options:
            selected_metrics.append('registered')

        if selected_metrics:
            #WEATHER SITUATION
            fig = px.bar(data, x='weathersit', y=selected_metrics, labels={'value': 'Amount of Users', 'weathersit': 'Weather Situation'},barmode="group",color_discrete_map=colors)
            fig.update_xaxes(type='category', tickmode='array', tickvals=[1, 2, 3, 4], ticktext=['Clear', 'Mist', 'Light Precipitation', 'Heavy Precipitation'])
            fig.update_layout(legend_title_text='User Groups')
            st.plotly_chart(fig)
            st.divider()

            
            #TEMPERATURE 
            # Create subplots with 1 row and 2 columns if there are two metrics selected
            #subplot_rows = 1 if len(selected_metrics) > 1 else 2
            #subplot_cols = 2 if len(selected_metrics) > 1 else 1
            fig = sp.make_subplots(rows=2, cols=2, subplot_titles=('Casual Users Feeling Temperature', 'Registered Users Feeling Temperature',
                                                       'Casual Users Real Temperature', 'Registered Users Real Temperature'))
            #fig = sp.make_subplots(rows=2, cols=2)
          
            # Track the current column
            col_counter = 1

            for metric in selected_metrics:
                # Add scatter plot for the current metric on the left for 'atemp' (row 1)
                fig.add_trace(
                    go.Scatter(
                        x=data['atemp'],
                        y=data[metric],
                        mode='markers',
                        marker=dict(color=colors[metric]),  # Use the color from the colors dictionary
                        name=metric
                    ),
                    row=1, col=col_counter
                )
                
                # Add scatter plot for the current metric on the left for 'temp' (row 2)
                fig.add_trace(
                    go.Scatter(
                        x=data['temp'],
                        y=data[metric],
                        mode='markers',
                        marker=dict(color=colors[metric]),  # Use the color from the colors dictionary
                        name=metric
                    ),
                    row=2, col=col_counter
                )
                
                # Increment the column counter when there is more than one metric
                if len(selected_metrics) > 1:
                    col_counter += 1

            # Update xaxis properties for both 'atemp' and 'temp' scatter plots
            fig.update_xaxes(title_text='Feeling Temperature (atemp)', row=1, col=1)
            #fig.update_xaxes(title_text='Feeling Temperature (atemp)', row=1, col=2)
            fig.update_xaxes(title_text='Real Temperature (temp)', row=2, col=1)
            #fig.update_xaxes(title_text='Real Temperature (temp)', row=2, col=2)

            # Update layout
            fig.update_layout(height=600, width=1200, title_text="Temperature Influence on User Groups")
            # Display the figure
            st.plotly_chart(fig)

            st.divider()
            fig = px.bar(data, x='hum', y=selected_metrics, labels={'value': 'Amount of Users', 'hum': 'Humiditiy'},barmode="group",color_discrete_map=colors)
            fig.update_layout(legend_title_text='User Groups')
            st.plotly_chart(fig)

            st.divider()
            fig = px.bar(data, x='windspeed', y=selected_metrics, labels={'value': 'Amount of Users', 'windspeed': 'Wind'},barmode="group",color_discrete_map=colors)
            fig.update_layout(legend_title_text='User Groups')
            st.plotly_chart(fig)
        
with tab2: 
    st.header("What are the differences between our users?")
    st.subheader("Casual Users")
    multi  = """
    - peak in the afternoon (hence likely to be tourists, people who use the service for leisure.
    - peak during the weekends (hence likely to be all sorts of people, who use the service for leisure.
    - Are using the bike sharing service more between.
    - Use the bike sharing service more on weekends and holidays.
    """
    st.markdown(multi)
    st.divider()
    st.subheader("Registered Users")
    multi ="""
    - peak in the morning and the evening (hence likely to be working people, people who use the service to get to work or school.
    - peak during the week (hence likely to be people with a similar daily routine during the week, like commute to work.
    - Only noting that registered users use the bikes slightly less in July compared to the other months from June till September.
    - Use the bike sharing service more on working days.          
    """
    st.markdown(multi)