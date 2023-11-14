
# Autor: Elisabth Oeljeklaus
# Date: 2023-11-08
import streamlit as st#
from defaults.defaultforlibs import add_logo
add_logo()
# Function to display short-term initiatives
def short_term_initiatives():
    st.header('Short-Term Initiatives')
    multi=("""
    1. **Peak Hour Availability**: 
        - Increase bike availability during peak hours for `registered users`.
        - Increase bike availability during peak hours for `casual users`.
    2. **Provisioning**: 
        - Ensure higher bike availability during peak times in the commute times (early mornings and evenings) to attract more `casual users`.
    3. **Promotional Offers**: 
        - Introduce promotional offers for `casual users`during high-demand seasons (e.g. workingdays, early morning & evening hours, winter etc.)
        - Offer discounts for `registered users` during low-demand seasons (e.g. weekends,noon hours, winter, holiday season etc.)
    4. **Bike Maintenance**: 
        - Intensify maintenance checks to ensure optimal bike conditions.
    5. **Weather Preparedness**: 
        - Ensure that the bike supply meets the expected weather-based demand. 
        - You can use the weather forecast to predict demand and use historical data with our Machine Learning model to predict supply.
    """)
    st.markdown(multi, unsafe_allow_html=True)

# Function to display long-term initiatives
def long_term_initiatives():
    st.header('Long-Term Initiatives')
    multi =("""
    1. **Model Deployment, Maintaining & Improvement**:
        - Deploy our predictive model to optimize bike availability and to manage demand, maintanance and maximize revenue.
    2. **Loyalty Programs**: 
        - Implement loyalty programs to encourage consistent use throughout the year.
    3. **Dynamic Pricing Models**:
        - Develop dynamic pricing strategies to manage demand.
    4. **Real-Time Tracking App**: 
        - Develop a mobile application for real-time bike availability.
    5. **Predictive Deployment**: 
        - Use predictive modeling for real-time bike deployment.
    6. **Public-Private Partnerships**: 
        - Collaborate with local businesses for subsidized memberships.
    7. **Urban Planning Considerations**: 
        - Work with city planners to optimize station placement.
    8. **Educational Outreach**: 
        - Conduct public outreach to highlight the benefits of bike-sharing.
    9. **Flexible Workday Adjustments**:
        - Promote flexible commuting options in response to weather forecasts.
    """)
    st.markdown(multi, unsafe_allow_html=True)

def notes():
    txt = st.text_area(
    "Here you can write your notes.",(
    "Feel free to write anything you want.\n"
    "This is a multiline text area.\n"
    "It will be saved as a string.\n"
    "If your done click the button below."
    ))
    st.write(f'You wrote {len(txt)} characters.')

    if st.download_button('Save your notes!', txt):
        st.write('Your notes have been saved!')

def actions():
    multi =("""
    # Business Insights on Bike Usage Patterns

    ## User Patterns

    ### Casual User Pattern: "Leisure Users"
    - **Primary Usage**: During weekends and good weather.
    - **Demand Peak**: Around noon.

    ### Registered User Pattern: "Get to Work Users"
    - **Primary Usage**: During weekdays, less sensitive to weather.
    - **Demand Peak**: Morning and afternoon (start/end of working day).
    - **Note**: When the weather worsens, both casual and registered bike users significantly reduce their usage. However, it's observed that in conditions of heavy precipitation, those who continue to use bikes are more likely to be registered users.
    ----------------------
    ## Demand Overview
    - Over a two-year period, **~80%** of daily demand comes from registered users.
    - Year-over-Year (YoY) demand increase is higher for registered users compared to casual users.
    ----------------------
    ## Strategic Advantages

    1. **Boost Customer Satisfaction**: Anticipating demand peaks to adjust bike supply, enhancing satisfaction and loyalty.
    2. **Operational Costs**: Optimizing bike supply to match demand, reducing excess supply costs.
    3. **Bike Availability**: Predicting demand fluctuations to schedule optimal times for bike redistribution and maintenance.
    4. **Demand Boosting**: Offering discounts during low demand periods (nights, cold weather) to attract price-sensitive customers.
    5. **Alternative Revenue Streams**: Renting out excess bikes in low demand periods (winter, night) to delivery services.
    -----------------------
    ## Considerations Beyond Current Analysis

    - **Bike Location**: Lack of location data could lead to unavailability in certain city regions, affecting revenue.
    - **Station Capacity**: Absence of station capacity data might deter potential users due to parking issues.
    - **District Analysis**: Bike location data could identify districts with high or low bike supply.
    - **Travel Distance**: Information on travel distances could reveal more detailed usage patterns and preferences, like the need for E-bikes for longer trips.

    """)
    st.markdown(multi, unsafe_allow_html=True)

# Main app structure
def main():
    st.title("Actions to take! 🫵🏽")
    
    st.write("We have divided them into short-term and long-term initiatives")
    st.write("Feel free to take a look at our notes and write your own.")
    
    tab1, tab2, tab3,tab4= st.tabs(["Summary of Actions 🫱🏽‍🫲🏼",'Short-Term Initiatives 🪄', 'Long-Term Initiatives ⏳',"Notes 📝"])

    with tab1:
        actions()
    with tab2:
        short_term_initiatives()
    with tab3:
        long_term_initiatives()
    with tab4:
        notes()

main()

