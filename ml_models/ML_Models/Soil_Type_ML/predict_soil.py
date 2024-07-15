import argparse
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array, load_img

# Load the trained model
model = load_model('soil.h5')

# Soil type labels (update these to match your dataset)
soil_labels = [
    'Alluvial Soil', 'Black Soil', 'Cinder Soil', 'Clayey soils',
    'Laterite Soil', 'Loamy Soil', 'Peat Soil', 'Red Soil',
    'Sandy Loam Soil', 'Sandy Soil', 'Yellow Soil'
]

def preprocess_image(image_path, target_size=(256, 256)):
    # Load the image
    img = load_img(image_path, target_size=target_size)
    # Convert the image to array
    img_array = img_to_array(img)
    # Scale the image (normalizing)
    img_array = img_array / 255.0
    # Expand dimensions to match the input shape of the model
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

def predict_soil_type(image_path):
    # Preprocess the image
    img_array = preprocess_image(image_path)
    # Make the prediction
    predictions = model.predict(img_array)
    # Get the index of the highest predicted probability
    predicted_class = np.argmax(predictions[0])
    # Get the corresponding label
    soil_type = soil_labels[predicted_class]
    return soil_type

if __name__ == '__main__':
    # Set up argument parsing
    parser = argparse.ArgumentParser(description='Soil Type Prediction from Image')
    parser.add_argument('image_path', type=str, help='Path to the image file')
    
    args = parser.parse_args()
    
    # Predict the soil type
    soil_type = predict_soil_type(args.image_path)
    
    # Print the result
    print(f'The predicted soil type is: {soil_type}')