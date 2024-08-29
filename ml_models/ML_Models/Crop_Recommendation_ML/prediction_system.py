import numpy as np
import joblib
import pandas as pd

# Load the crop dictionary from the training script
crop_dict = {
    'rice': 1, 'maize': 2, 'jute': 3, 'cotton': 4, 'coconut': 5, 'papaya': 6,
    'orange': 7, 'apple': 8, 'muskmelon': 9, 'watermelon': 10, 'grapes': 11,
    'mango': 12, 'banana': 13, 'pomegranate': 14, 'lentil': 15, 'blackgram': 16,
    'mungbean': 17, 'mothbeans': 18, 'pigeonpeas': 19, 'kidneybeans': 20,
    'chickpea': 21, 'coffee': 22
}

# Inverse crop dictionary for decoding
inverse_crop_dict = {v: k for k, v in crop_dict.items()}

def recommend_top_5_crops(input_data):
    # Load the pipeline model
    model_pipeline = joblib.load('crop_recommendation_model.pkl')
    
    # Ensure input data is in the right shape (e.g., one sample with all features)
    input_data = np.array(input_data).reshape(1, -1)
    
    # Predict probabilities using the pipeline (it handles scaling internally)
    probabilities = model_pipeline.predict_proba(input_data)[0]
    
    # Get the top 5 crop indices
    top_5_indices = np.argsort(probabilities)[-5:][::-1]
    
    # Get the corresponding crop names
    top_5_crops = [inverse_crop_dict[idx + 1] for idx in top_5_indices]
    
    return top_5_crops

# Example usage:
if __name__ == "__main__":
    # Example input data (replace with actual input)
    example_input = [80, 40, 45, 28.0, 80.0, 6.5, 300.0]  # Example feature values
    top_5_crops = recommend_top_5_crops(example_input)
    print("Top 5 recommended crops:", top_5_crops)
