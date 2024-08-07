import numpy as np
import joblib
import os
import requests
from flask import Flask, request, render_template, redirect, url_for, session
from werkzeug.utils import secure_filename
from tensorflow.keras.models import load_model  # type: ignore
from tensorflow.keras.preprocessing import image  # type: ignore
from PIL import Image
from authlib.integrations.flask_client import OAuth
from datetime import datetime, timedelta
from dotenv import load_dotenv
from functools import wraps

app = Flask(__name__)

# Load environment variables from .env file
load_dotenv()

# App config
app.secret_key = os.getenv('SECRET_KEY')
app.config['SESSION_COOKIE_NAME'] = 'google-login-session'
app.config['PERMANENT_SESSION_LIFETIME'] = timedelta(minutes=5)

# OAuth Setup
oauth = OAuth(app)
google = oauth.register(
    name='google',
    client_id=os.getenv('GOOGLE_CLIENT_ID'),
    client_secret=os.getenv('GOOGLE_CLIENT_SECRET'),
    access_token_url='https://accounts.google.com/o/oauth2/token',
    access_token_params=None,
    authorize_url='https://accounts.google.com/o/oauth2/auth',
    authorize_params=None,
    api_base_url='https://www.googleapis.com/oauth2/v1/',
    userinfo_endpoint='https://openidconnect.googleapis.com/v1/userinfo',
    client_kwargs={'scope': 'email profile'},
    server_metadata_url='https://accounts.google.com/.well-known/openid-configuration'
)

# Set the upload folder
UPLOAD_FOLDER = 'uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Ensure the upload folder exists
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Load the models
soil_model = load_model('ml_models/soil.h5')
crop_model = joblib.load('ml_models/crop_recommendation_model.pkl')

# Define allowed file extensions
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# Decorator for routes that should be accessible only by logged-in users
def login_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if 'profile' not in session:
            return redirect(url_for('login'))
        return f(*args, **kwargs)
    return decorated_function

@app.route('/', methods=['GET', 'POST'])
def index():
    api_key = os.getenv('OPENWEATHER_KEY')
    weather_data = None
    error_message = None
    
    if api_key and request.method == 'POST':
        city = request.form.get('city', 'Your_City_Name')  # Default city name if not specified
        url = f'http://api.openweathermap.org/data/2.5/weather?q={city}&appid={api_key}&units=metric'
        response = requests.get(url)
        if response.status_code == 200:
            weather_data = response.json()
        else:
            error_message = f"Could not retrieve weather data for {city}. Please try again."

    return render_template('index.html', weather_data=weather_data, error_message=error_message)

@app.route('/crop_recommendation', methods=['GET', 'POST'])
@login_required
def crop_recommendation():
    if request.method == 'POST':
        try:
            # Get the file from the POST request
            file = request.files['image']
            if file and allowed_file(file.filename):
                filename = secure_filename(file.filename)
                filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
                file.save(filepath)
                
                # Predict soil type
                img = image.load_img(filepath, target_size=(256, 256))
                img_array = image.img_to_array(img)
                img_array = np.expand_dims(img_array, axis=0) / 255.0
                predictions = soil_model.predict(img_array)
                predicted_class = np.argmax(predictions, axis=1)[0]
                soil_types = ['Alluvial Soil', 'Black Soil', 'Cinder Soil', 'Clayey Soils', 'Laterite Soil', 'Loamy Soil', 'Peat Soil', 'Red Soil', 'Sandy Loam Soil', 'Sandy Soil', 'Yellow Soil']
                predicted_soil_type = soil_types[predicted_class]
                
                # Get weather data
                api_key = os.getenv('OPENWEATHER_KEY')
                city = request.form['city']
                url = f'http://api.openweathermap.org/data/2.5/weather?q={city}&appid={api_key}&units=metric'
                response = requests.get(url)
                weather_data = response.json()
                temperature = weather_data['main']['temp']
                humidity = weather_data['main']['humidity']

                # Get rainfall data
                current_year = datetime.now().year
                current_month = datetime.now().month
                # Assuming you have a function or API to get the expected rainfall
                station = request.form['station']
                rainfall = get_expected_rainfall(station, current_year, current_month)
                
                # Assuming these values for N, P, K, and pH based on the predicted soil type
                soil_npk_ph = {
                    'Alluvial Soil': (50, 20, 30, 6.5),
                    'Black Soil': (60, 30, 40, 7.0),
                    'Cinder Soil': (40, 10, 20, 6.0),
                    'Clayey Soils': (70, 40, 50, 7.5),
                    'Laterite Soil': (30, 20, 10, 5.5),
                    'Loamy Soil': (80, 50, 60, 7.0),
                    'Peat Soil': (20, 10, 15, 5.0),
                    'Red Soil': (50, 30, 40, 6.0),
                    'Sandy Loam Soil': (60, 40, 50, 6.5),
                    'Sandy Soil': (30, 10, 20, 6.0),
                    'Yellow Soil': (50, 20, 30, 6.5)
                }
                
                N, P, K, ph = soil_npk_ph[predicted_soil_type]
                
                input_data = np.array([[N, P, K, temperature, humidity, ph, rainfall]])
                probabilities = crop_model.predict_proba(input_data)
                top_5_indices = np.argsort(probabilities[0])[-5:][::-1]
                
                crop_dict = {
                    1: 'rice', 2: 'maize', 3: 'jute', 4: 'cotton', 5: 'coconut', 6: 'papaya',
                    7: 'orange', 8: 'apple', 9: 'muskmelon', 10: 'watermelon', 11: 'grapes',
                    12: 'mango', 13: 'banana', 14: 'pomegranate', 15: 'lentil', 16: 'blackgram',
                    17: 'mungbean', 18: 'mothbeans', 19: 'pigeonpeas', 20: 'kidneybeans',
                    21: 'chickpea', 22: 'coffee'
                }
                
                recommended_crops = [crop_dict[i + 1] for i in top_5_indices]
                
                return render_template('crop_recommendation.html', recommended_crops=recommended_crops, predicted_soil_type=predicted_soil_type)
        except Exception as e:
            return str(e)
    return render_template('crop_recommendation.html')

def get_expected_rainfall(station, year, month):
    # Placeholder function to simulate getting expected rainfall data
    # You should replace this with actual logic or API call to get the expected rainfall
    return 100.0  # Dummy value for rainfall

@app.route('/login')
def login():
    print("Accessing login route")
    google = oauth.create_client('google')
    redirect_uri = url_for('authorize', _external=True)
    print(f"Redirect URI: {redirect_uri}")
    return google.authorize_redirect(redirect_uri)

@app.route('/authorize')
def authorize():
    google = oauth.create_client('google')
    token = google.authorize_access_token()
    resp = google.get('userinfo')
    user_info = resp.json()
    session['profile'] = user_info
    session.permanent = True
    return redirect('/')

@app.route('/logout')
def logout():
    for key in list(session.keys()):
        session.pop(key)
    return redirect('/')

if __name__ == '__main__':
    app.run(debug=True)
