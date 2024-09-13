print("Starting app.py")
import sys
print(f"Python version: {sys.version}")
print(f"Python path: {sys.executable}")

print("Importing Flask...")
from flask import Flask, request, jsonify
print("Importing numpy...")
import numpy as np
print("Importing pandas...")
import pandas as pd
print("Importing sklearn...")
from sklearn.linear_model import LassoCV
from sklearn.preprocessing import StandardScaler, OneHotEncoder
print("Importing joblib...")
import joblib
import os

print("Creating Flask app...")
app = Flask(__name__)

# Global variables for our models and preprocessors
data = None
screw_config_encoder = None
scaler = None
lasso_coverage = None
lasso_number = None
X_interaction = None
y_coverage = None
y_number = None

def load_and_preprocess_data():
    global data, screw_config_encoder, scaler, X_interaction, y_coverage, y_number
    
    print("Loading data...")
    try:
        data = pd.read_csv('Model_data.csv')
        print("Data loaded successfully")
    except FileNotFoundError:
        print("Error: Model_data.csv not found!")
        return False
    
    print("Preprocessing data...")
    screw_config_encoder = OneHotEncoder(drop='first')
    screw_config_encoded = screw_config_encoder.fit_transform(data[['Screw_Configuration']]).toarray()
    
    scaler = StandardScaler()
    data[['Screw_speed_norm', 'Liquid_binder_norm', 'Liquid_content_norm']] = scaler.fit_transform(
        data[['Screw_speed', 'Liquid_binder', 'Liquid_content']]
    )
    
    X = np.hstack([data[['Screw_speed_norm', 'Liquid_binder_norm', 'Liquid_content_norm']].values, screw_config_encoded])
    y_coverage = data['Seed_coverage'].values
    y_number = data['number_seeded'].values
    
    X_interaction = np.hstack([
        X,
        (X[:, 0] * X[:, 1]).reshape(-1, 1),
        (X[:, 0] * X[:, 2]).reshape(-1, 1),
        (X[:, 1] * X[:, 2]).reshape(-1, 1),
        (X[:, 0] ** 2).reshape(-1, 1),
        (X[:, 1] ** 2).reshape(-1, 1),
        (X[:, 2] ** 2).reshape(-1, 1)
    ])
    print("Data preprocessing completed")
    return True

def train_or_load_models():
    global lasso_coverage, lasso_number
    
    if os.path.exists('lasso_coverage.joblib') and os.path.exists('lasso_number.joblib'):
        print("Loading pre-trained models...")
        lasso_coverage = joblib.load('lasso_coverage.joblib')
        lasso_number = joblib.load('lasso_number.joblib')
        print("Models loaded successfully")
    else:
        print("Training new models...")
        lasso_coverage = LassoCV(cv=10).fit(X_interaction, y_coverage)
        lasso_number = LassoCV(cv=10).fit(X_interaction, y_number)
        
        print("Saving trained models...")
        joblib.dump(lasso_coverage, 'lasso_coverage.joblib')
        joblib.dump(lasso_number, 'lasso_number.joblib')
        print("Models saved successfully")

print("Initializing data and models...")
if load_and_preprocess_data():
    train_or_load_models()
else:
    print("Failed to initialize data and models")

@app.route('/')
def home():
    return "Welcome to the Flask API! Use the /predict endpoint for predictions."

def predict_with_confidence_intervals(X_new, model, y_train, X_train):
    predictions = model.predict(X_new)
    residuals = y_train - model.predict(X_train)
    se_residuals = np.std(residuals)
    ci_range = 1.96 * se_residuals * np.sqrt(1 + 1 / len(X_train))
    ci = np.array([predictions - ci_range, predictions + ci_range]).T
    return predictions, ci

@app.route('/predict', methods=['POST'])
def predict():
    if not all([data is not None, screw_config_encoder is not None, scaler is not None, 
                lasso_coverage is not None, lasso_number is not None]):
        return jsonify({"error": "Models not initialized properly"}), 500
    
    try:
        input_data = request.get_json(force=True)
        features = np.array(input_data['features']).reshape(1, -1)
        screw_config_input = np.array(input_data['screw_config']).reshape(1, -1)

        features_scaled = scaler.transform(features)
        screw_config_encoded = screw_config_encoder.transform(screw_config_input).toarray()

        X_new = np.hstack([features_scaled, screw_config_encoded])
        X_new = np.hstack([
            X_new,
            (X_new[:, 0] * X_new[:, 1]).reshape(-1, 1),
            (X_new[:, 0] * X_new[:, 2]).reshape(-1, 1),
            (X_new[:, 1] * X_new[:, 2]).reshape(-1, 1),
            (X_new[:, 0] ** 2).reshape(-1, 1),
            (X_new[:, 1] ** 2).reshape(-1, 1),
            (X_new[:, 2] ** 2).reshape(-1, 1)
        ])

        coverage_prediction, coverage_ci = predict_with_confidence_intervals(X_new, lasso_coverage, y_coverage, X_interaction)
        number_prediction, number_ci = predict_with_confidence_intervals(X_new, lasso_number, y_number, X_interaction)

        coverage_prediction = np.clip(coverage_prediction, 0, 100)
        coverage_ci = np.clip(coverage_ci, 0, 100)

        return jsonify({
            'coverage_prediction': coverage_prediction[0],
            'coverage_ci': coverage_ci[0].tolist(),
            'number_prediction': number_prediction[0],
            'number_ci': number_ci[0].tolist()
        })
    except Exception as e:
        print(f"Error during prediction: {str(e)}")
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    print("Running app in debug mode")
    app.run(debug=True)
else:
    print("App imported, ready to be run by WSGI server")