import sys
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import numpy as np
import pandas as pd
from sklearn.linear_model import LassoCV
from sklearn.preprocessing import StandardScaler, OneHotEncoder
import joblib
import os

app = Flask(__name__)
CORS(app)

# Global variables
data = None
screw_config_encoder = None
scaler = None
lasso_coverage = None
lasso_number = None
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
    
    return True

def train_or_load_models():
    global lasso_coverage, lasso_number
    
    if os.path.exists('lasso_coverage.joblib') and os.path.exists('lasso_number.joblib'):
        lasso_coverage = joblib.load('lasso_coverage.joblib')
        lasso_number = joblib.load('lasso_number.joblib')
    else:
        lasso_coverage = LassoCV(cv=10, positive=True).fit(X_interaction, y_coverage)
        lasso_number = LassoCV(cv=10, positive=True).fit(X_interaction, y_number)
        joblib.dump(lasso_coverage, 'lasso_coverage.joblib')
        joblib.dump(lasso_number, 'lasso_number.joblib')

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
        
        coverage_prediction, coverage_ci = predict_with_confidence_intervals(X_interaction_new, lasso_coverage, y_coverage, X_interaction)
        number_prediction, number_ci = predict_with_confidence_intervals(X_interaction_new, lasso_number, y_number, X_interaction)
        
        # Ensure non-negative predictions and clip coverage to 0-100%
        coverage_prediction = np.clip(coverage_prediction, 0, 100)
        coverage_ci = np.clip(coverage_ci, 0, 100)
        number_prediction = np.maximum(number_prediction, 0)
        number_ci = np.maximum(number_ci, 0)
        
        return jsonify({
            'coverage_prediction': float(coverage_prediction[0]),
            'coverage_ci': coverage_ci[0].tolist(),
            'number_prediction': float(number_prediction[0]),
            'number_ci': number_ci[0].tolist()
        })
    except Exception as e:
        return jsonify({
            "error": "An error occurred during prediction",
            "error_message": str(e)
        }), 500

@app.route('/feature_importance', methods=['GET'])
def feature_importance():
    if lasso_coverage is None or lasso_number is None:
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
    
    coverage_importance = dict(zip(feature_names, lasso_coverage.coef_))
    number_importance = dict(zip(feature_names, lasso_number.coef_))
    
    return jsonify({
        'coverage_importance': coverage_importance,
        'number_importance': number_importance
    })

if __name__ == '__main__':
    app.run(debug=True)
else:
    print("App imported, ready to be run by WSGI server")