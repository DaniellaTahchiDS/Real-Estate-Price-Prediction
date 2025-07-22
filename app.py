# Create streamlit app
import streamlit as st
import pandas as pd
import pickle
# Load the model and scaler
with open('model.pkl', 'rb') as file:
    model = pickle.load(file)

with open('scaler.pkl', 'rb') as file:
    scaler = pickle.load(file)
# Define the input fields
st.title("House Price Prediction")
st.write("Enter the details below to predict the house price of unit area.")
# Input fields for user to enter data
distance_to_mrt = st.number_input("Distance to MRT (km)", min_value=0.0, max_value=100000.0, value=0.0)
number_of_convenience_stores = st.number_input("Number of Convenience Stores", min_value=0, max_value=100, value=0)
latitude = st.number_input("Latitude", min_value=-90.0, max_value=90.0, value=0.0)
longitude = st.number_input("Longitude", min_value=-180.0, max_value=180.0, value=0.0)  
# Create a button to make predictions
if st.button("Predict"):
    # Prepare the input data
    input_data = pd.DataFrame({
        'Distance to the nearest MRT station': [distance_to_mrt],
        'Number of convenience stores': [number_of_convenience_stores],
        'Latitude': [latitude],
        'Longitude': [longitude]
    })
    
    # Scale the input data
    input_data_scaled = scaler.transform(input_data)
    
    # Make the prediction
    prediction = model.predict(input_data_scaled)
    
    # Display the prediction
    st.write(f"Predicted House Price of Unit Area: {prediction[0]:.2f}")
