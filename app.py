import sys
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import numpy as np
import pandas as pd
from sklearn.linear_model import RidgeCV
from sklearn.preprocessing import StandardScaler, OneHotEncoder
import joblib
import os

app = Flask(__name__)
CORS(app)

# Global variables
data = None
screw_config_encoder = None
scaler = None
ridge_coverage = None
ridge_number = None
X_interaction = None
y_coverage = None
y_number = None

def predict_with_confidence_intervals(X_new, model, y_train, X_train):
    predictions = model.predict(X_new)
    residuals = y_train - model.predict(X_train)
    se_residuals = np.std(residuals)
    ci_range = 1.96 * se_residuals * np.sqrt(1 + 1 / len(X_train))
    ci = np.array([predictions - ci_range, predictions + ci_range]).T
    return predictions, ci

def load_and_preprocess_data():
    global data, screw_config_encoder, scaler, X_interaction, y_coverage, y_number
    
    try:
        data = pd.read_csv('Model_data.csv')
        print(f"Data loaded successfully. Shape: {data.shape}")
    except FileNotFoundError:
        print("Error: Model_data.csv not found!")
        return False
    
    screw_config_encoder = OneHotEncoder(drop='first', sparse_output=False)
    screw_config_encoded = screw_config_encoder.fit_transform(data[['Screw_Configuration']])
    
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(data[['Screw_speed', 'Liquid_content', 'Liquid_binder']])
    
    X = np.hstack([scaled_features, screw_config_encoded])
    y_coverage = data['Seed_coverage'].values
    y_number = data['number_seeded'].values
    
    # Create interaction and quadratic terms
    X_interaction = np.column_stack([
        X,
        X[:, 0] * X[:, 1],  # Screw_speed * Liquid_content
        X[:, 0] * X[:, 2],  # Screw_speed * Liquid_binder
        X[:, 1] * X[:, 2],  # Liquid_content * Liquid_binder
        X[:, 0]**2,         # Screw_speed^2
        X[:, 1]**2,         # Liquid_content^2
        X[:, 2]**2          # Liquid_binder^2
    ])
    
    print(f"X_interaction shape: {X_interaction.shape}")
    print(f"y_coverage shape: {y_coverage.shape}")
    print(f"y_number shape: {y_number.shape}")
    
    return True

def train_or_load_models():
    global ridge_coverage, ridge_number
    
    if os.path.exists('ridge_coverage.joblib') and os.path.exists('ridge_number.joblib'):
        ridge_coverage = joblib.load('ridge_coverage.joblib')
        ridge_number = joblib.load('ridge_number.joblib')
    else:
        ridge_coverage = RidgeCV(alphas=[0.1, 1.0, 10.0], cv=5).fit(X_interaction, y_coverage)
        ridge_number = RidgeCV(alphas=[0.1, 1.0, 10.0], cv=5).fit(X_interaction, y_number)
        joblib.dump(ridge_coverage, 'ridge_coverage.joblib')
        joblib.dump(ridge_number, 'ridge_number.joblib')
    
    print(f"Coverage model coefficients: {ridge_coverage.coef_}")
    print(f"Number model coefficients: {ridge_number.coef_}")

if load_and_preprocess_data():
    train_or_load_models()
else:
    print("Failed to initialize data and models")

@app.route('/')
def serve_html():
    return send_from_directory('static', 'index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        input_data = request.get_json(force=True)
        
        features = np.array(input_data['features']).reshape(1, -1)
        screw_config_input = np.array(input_data['screw_config']).reshape(1, -1)
        
        print(f"Input features: {features}")
        print(f"Input screw config: {screw_config_input}")
        
        features_scaled = scaler.transform(features)
        screw_config_encoded = screw_config_encoder.transform(screw_config_input)
        
        X_new = np.hstack([features_scaled, screw_config_encoded])
        
        # Create interaction and quadratic terms
        X_interaction_new = np.column_stack([
            X_new,
            X_new[:, 0] * X_new[:, 1],  # Screw_speed * Liquid_content
            X_new[:, 0] * X_new[:, 2],  # Screw_speed * Liquid_binder
            X_new[:, 1] * X_new[:, 2],  # Liquid_content * Liquid_binder
            X_new[:, 0]**2,             # Screw_speed^2
            X_new[:, 1]**2,             # Liquid_content^2
            X_new[:, 2]**2              # Liquid_binder^2
        ])
        
        print(f"X_interaction_new shape: {X_interaction_new.shape}")
        
        coverage_prediction, coverage_ci = predict_with_confidence_intervals(X_interaction_new, ridge_coverage, y_coverage, X_interaction)
        number_prediction, number_ci = predict_with_confidence_intervals(X_interaction_new, ridge_number, y_number, X_interaction)
        
        # Ensure non-negative predictions and clip coverage to 0-100%
        coverage_prediction = np.clip(coverage_prediction, 0, 100)
        coverage_ci = np.clip(coverage_ci, 0, 100)
        number_prediction = np.maximum(number_prediction, 0)
        number_ci = np.maximum(number_ci, 0)
        
        # Sanity check: if prediction is zero, use mean of training data
        if coverage_prediction[0] == 0:
            coverage_prediction[0] = np.mean(y_coverage)
        if number_prediction[0] == 0:
            number_prediction[0] = np.mean(y_number)
        
        print(f"Coverage prediction: {coverage_prediction}")
        print(f"Number prediction: {number_prediction}")
        
        return jsonify({
            'coverage_prediction': float(coverage_prediction[0]),
            'coverage_ci': coverage_ci[0].tolist(),
            'number_prediction': float(number_prediction[0]),
            'number_ci': number_ci[0].tolist()
        })
    except Exception as e:
        print(f"Error during prediction: {str(e)}")
        return jsonify({
            "error": "An error occurred during prediction",
            "error_message": str(e)
        }), 500

@app.route('/feature_importance', methods=['GET'])
def feature_importance():
    if ridge_coverage is None or ridge_number is None:
        return jsonify({"error": "Models not initialized"}), 500
    
    feature_names = ['Screw_speed', 'Liquid_content', 'Liquid_binder'] + [f'Screw_Config_{cat}' for cat in screw_config_encoder.categories_[0][1:]]
    feature_names += [
        'Screw_speed * Liquid_content',
        'Screw_speed * Liquid_binder',
        'Liquid_content * Liquid_binder',
        'Screw_speed^2',
        'Liquid_content^2',
        'Liquid_binder^2'
    ]
    
    coverage_importance = dict(zip(feature_names, ridge_coverage.coef_))
    number_importance = dict(zip(feature_names, ridge_number.coef_))
    
    return jsonify({
        'coverage_importance': coverage_importance,
        'number_importance': number_importance
    })

if __name__ == '__main__':
    app.run(debug=True)
else:
    print("App imported, ready to be run by WSGI server")