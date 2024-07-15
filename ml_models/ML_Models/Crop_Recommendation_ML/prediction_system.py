import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, StandardScaler

# Load the saved model
model = joblib.load('crop_recommendation_model.pkl')

# Load data to get scalers
crop = pd.read_csv("Crop_recommendation.csv")

# Split data for scalers
x = crop.drop(['label'], axis=1)

# Scale data
ms = MinMaxScaler()
x = ms.fit_transform(x)

sc = StandardScaler()
x = sc.fit_transform(x)

def preprocess_input(data):
    data = np.array(data).reshape(1, -1)
    data = ms.transform(data)
    data = sc.transform(data)
    return data

# Crop dictionary to decode the numerical prediction back to crop names
crop_dict = {
    1: 'rice', 2: 'maize', 3: 'jute', 4: 'cotton', 5: 'coconut', 6: 'papaya',
    7: 'orange', 8: 'apple', 9: 'muskmelon', 10: 'watermelon', 11: 'grapes',
    12: 'mango', 13: 'banana', 14: 'pomegranate', 15: 'lentil', 16: 'blackgram',
    17: 'mungbean', 18: 'mothbeans', 19: 'pigeonpeas', 20: 'kidneybeans',
    21: 'chickpea', 22: 'coffee'
}

def predict_crop(N, P, K, temperature, humidity, pH, rainfall):
    data = [N, P, K, temperature, humidity, pH, rainfall]
    preprocessed_data = preprocess_input(data)
    prediction = model.predict(preprocessed_data)
    predicted_crop = crop_dict[prediction[0]]
    return predicted_crop

if __name__ == "__main__":
    print("Enter the following values:")
    N = float(input("Nitrogen (N): "))
    P = float(input("Phosphorus (P): "))
    K = float(input("Potassium (K): "))
    temperature = float(input("Temperature (°C): "))
    humidity = float(input("Humidity (%): "))
    pH = float(input("pH level: "))
    rainfall = float(input("Rainfall (mm): "))
    
    recommended_crop = predict_crop(N, P, K, temperature, humidity, pH, rainfall)
    print(f"The recommended crop for the given conditions is: {recommended_crop}")
