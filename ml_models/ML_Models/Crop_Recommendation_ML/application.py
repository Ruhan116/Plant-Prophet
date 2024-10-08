import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier, ExtraTreeClassifier
from sklearn.ensemble import RandomForestClassifier, BaggingClassifier, GradientBoostingClassifier, AdaBoostClassifier
from sklearn.metrics import accuracy_score
from sklearn.pipeline import Pipeline
from sklearn.model_selection import StratifiedKFold

# Load data
crop = pd.read_csv("Crop_recommendation.csv")

# Encode labels
crop_dict = {
    'rice': 1, 'maize': 2, 'jute': 3, 'cotton': 4, 'coconut': 5, 'papaya': 6,
    'orange': 7, 'apple': 8, 'muskmelon': 9, 'watermelon': 10, 'grapes': 11,
    'mango': 12, 'banana': 13, 'pomegranate': 14, 'lentil': 15, 'blackgram': 16,
    'mungbean': 17, 'mothbeans': 18, 'pigeonpeas': 19, 'kidneybeans': 20,
    'chickpea': 21, 'coffee': 22
}
crop['crop_num'] = crop['label'].map(crop_dict)

# Split data
x = crop.drop(['crop_num', 'label'], axis=1)
y = crop['crop_num']
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

# Define models
models = {
    'Logistic Regression': LogisticRegression(),
    'Naive Bayes': GaussianNB(),
    'Support Vector Machine': SVC(probability=True),  # Enable probability estimates
    'K-Nearest Neighbors': KNeighborsClassifier(),
    'Decision Tree': DecisionTreeClassifier(),
    'Random Forest': RandomForestClassifier(),
    'Bagging': BaggingClassifier(),
    'AdaBoost': AdaBoostClassifier(algorithm='SAMME.R'),  # Default is SAMME.R
    'Gradient Boosting': GradientBoostingClassifier(),
    'Extra Trees': ExtraTreeClassifier(),
}

# Evaluate models and find the best one using cross-validation
best_model = None
best_accuracy = 0
best_model_name = ""

for name, model in models.items():
    pipeline = Pipeline([
        ('scaler', StandardScaler()),  # Using StandardScaler, as it generally works well for most models
        ('model', model)
    ])
    
    # Use StratifiedKFold to ensure that each fold has the same proportion of classes
    scores = cross_val_score(pipeline, x_train, y_train, cv=StratifiedKFold(n_splits=5), scoring='accuracy')
    mean_accuracy = scores.mean()
    print(f"{name} with accuracy: {mean_accuracy}")

    if mean_accuracy > best_accuracy:
        best_accuracy = mean_accuracy
        best_model = pipeline
        best_model_name = name

print(f"Best model: {best_model_name} with accuracy: {best_accuracy}")

# Train the best model on the entire training set
best_model.fit(x_train, y_train)

# Save the best model and scaler
joblib.dump(best_model, 'crop_recommendation_model.pkl')

# Test the best model on the test set
ypred = best_model.predict(x_test)
print(f"{best_model_name} accuracy on test set: {accuracy_score(y_test, ypred)}")
