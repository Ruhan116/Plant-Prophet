import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import OneHotEncoder

# Load the trained model
model = joblib.load('rainfall_prediction_model.pkl')

# Define a function to preprocess input data
def preprocess_input(year, month, station, weather_df):
    # Create a dataframe for the input
    input_df = pd.DataFrame({
        'Year': [year],
        'Month': [month],
        'Station': [station]
    })
    
    # One-hot encode the Station column
    station_encoder = OneHotEncoder(drop='first', sparse_output=False)
    station_encoder.fit(weather_df[['Station']])
    station_encoded = station_encoder.transform(input_df[['Station']])
    station_columns = station_encoder.get_feature_names_out(['Station'])
    input_encoded_df = pd.DataFrame(station_encoded, columns=station_columns)
    
    # Merge the encoded station data with year and month
    input_df = pd.concat([input_df[['Year', 'Month']], input_encoded_df], axis=1)
    
    return input_df

# Example weather dataset for reference to the stations
weather = pd.read_csv("weather.csv")
weather = weather[weather['Year'] >= 1986]

# Get unique station names
stations = weather['Station'].unique()

# Print station names with corresponding numbers
print("Available Stations:")
for idx, station in enumerate(stations):
    print(f"{idx + 1}. {station}")

# User input
year = int(input("Enter the year (after 2014): "))
month = int(input("Enter the month (1-12): "))
station_num = int(input("Enter the station number: "))

# Validate station number
if station_num < 1 or station_num > len(stations):
    raise ValueError("Invalid station number.")

# Get the selected station name
station = stations[station_num - 1]

# Preprocess the input
input_data = preprocess_input(year, month, station, weather)

# Predict the rainfall
predicted_rainfall = model.predict(input_data)[0]

print(f"Predicted monthly rainfall for {station} in {year}-{month:02d}: {predicted_rainfall:.2f} mm")
