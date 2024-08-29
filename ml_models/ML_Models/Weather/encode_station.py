import pandas as pd
from sklearn.preprocessing import OneHotEncoder
import joblib

# Load your dataset
weather = pd.read_csv("weather.csv")

# Filter data if necessary
weather = weather[weather['Year'] >= 1986]

# Create the OneHotEncoder
station_encoder = OneHotEncoder(drop='first', sparse_output=False)

# Fit the encoder on the 'Station' column
station_encoder.fit(weather[['Station']])

# Save the encoder to a file
joblib.dump(station_encoder, 'station_encoder.pkl')
