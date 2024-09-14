print("Starting app.py")
import sys
print(f"Python version: {sys.version}")
print(f"Python path: {sys.executable}")

print("Importing Flask and related modules...")
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
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
CORS(app)  # Enable CORS for all routes

# Global variables for our models and preprocessors
data = None
screw_config_encoder = None
scaler = None
lasso_coverage = None
lasso_number = None
X_interaction = None
y_coverage = None
y_number = None

# Add this function definition here
def predict_with_confidence_intervals(X_new, model, y_train, X_train):
    predictions = model.predict(X_new)
    residuals = y_train - model.predict(X_train)
    se_residuals = np.std(residuals)
    ci_range = 1.96 * se_residuals * np.sqrt(1 + 1 / len(X_train))
    ci = np.array([predictions - ci_range, predictions + ci_range]).T
    return predictions, ci

@app.route('/test', methods=['GET'])
def test():
    return jsonify({"message": "Test route is working"}), 200

def load_and_preprocess_data():
    global data, screw_config_encoder, scaler, X_interaction, y_coverage, y_number, num_features
    
    print("Loading data...")
    try:
        data = pd.read_csv('Model_data.csv')
        print("Data loaded successfully")
    except FileNotFoundError:
        print("Error: Model_data.csv not found!")
        return False
    
    print("Preprocessing data...")
    screw_config_encoder = OneHotEncoder(drop='first', sparse_output=False)
    screw_config_encoded = screw_config_encoder.fit_transform(data[['Screw_Configuration']])
    
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(data[['Screw_speed', 'Liquid_binder', 'Liquid_content']])
    
    X = np.hstack([scaled_features, screw_config_encoded])
    y_coverage = data['Seed_coverage'].values
    y_number = data['number_seeded'].values
    
    num_features = X.shape[1]
    
    # Include interactions with screw configuration
    X_interaction = X.copy()
    for i in range(3):  # For each continuous feature
        for j in range(3, X.shape[1]):  # For each one-hot encoded feature
            X_interaction = np.column_stack((X_interaction, X[:, i] * X[:, j]))
    
    # Add quadratic terms for continuous features
    for i in range(3):
        X_interaction = np.column_stack((X_interaction, X[:, i] ** 2))
    
    print(f"X_interaction shape: {X_interaction.shape}")  # Debug print
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
def serve_html():
    return send_from_directory('static', 'index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if not all([data is not None, screw_config_encoder is not None, scaler is not None, 
                lasso_coverage is not None, lasso_number is not None]):
        return jsonify({"error": "Models not initialized properly"}), 500
    
    try:
        input_data = request.get_json(force=True)
        print(f"Received input data: {input_data}")  # Debug print
        
        features = np.array(input_data['features']).reshape(1, -1)
        screw_config_input = np.array(input_data['screw_config']).reshape(1, -1)
        
        print(f"Features shape: {features.shape}")  # Debug print
        print(f"Screw config input: {screw_config_input}")  # Debug print

        features_scaled = scaler.transform(features)
        screw_config_encoded = screw_config_encoder.transform(screw_config_input)
        
        print(f"Encoded screw config shape: {screw_config_encoded.shape}")  # Debug print

        X_new = np.hstack([features_scaled, screw_config_encoded])
        
        print(f"X_new shape: {X_new.shape}")  # Debug print
        
        # Create interaction terms
        X_interaction_new = X_new.copy()
        for i in range(3):  # For each continuous feature
            for j in range(3, num_features):  # Use num_features instead of X_new.shape[1]
                X_interaction_new = np.column_stack((X_interaction_new, X_new[:, i] * X_new[:, j]))
        
        # Add quadratic terms for continuous features
        for i in range(3):
            X_interaction_new = np.column_stack((X_interaction_new, X_new[:, i] ** 2))
        
        print(f"X_interaction_new shape: {X_interaction_new.shape}")  # Debug print
        print(f"Expected shape based on training: {X_interaction.shape[1]}")  # Debug print

        coverage_prediction, coverage_ci = predict_with_confidence_intervals(X_interaction_new, lasso_coverage, y_coverage, X_interaction)
        number_prediction, number_ci = predict_with_confidence_intervals(X_interaction_new, lasso_number, y_number, X_interaction)
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
        print(f"Error type: {type(e)}")
        import traceback
        error_traceback = traceback.format_exc()
        print(f"Traceback: {error_traceback}")
        return jsonify({
            "error": "An error occurred during prediction",
            "error_message": str(e),
            "error_type": str(type(e)),
            "traceback": error_traceback
        }), 500

@app.route('/debug_screw_config', methods=['POST'])
def debug_screw_config():
    input_data = request.get_json(force=True)
    screw_config_input = np.array(input_data['screw_config']).reshape(1, -1)
    encoded = screw_config_encoder.transform(screw_config_input)
    return jsonify({
        "input": screw_config_input.tolist(),
        "encoded": encoded.tolist(),
        "categories": screw_config_encoder.categories_
    })

@app.route('/feature_importance', methods=['GET'])
def feature_importance():
    if lasso_coverage is None or lasso_number is None:
        return jsonify({"error": "Models not initialized"}), 500
    
    feature_names = ['Screw_speed', 'Liquid_binder', 'Liquid_content'] + [f'Screw_Config_{cat}' for cat in screw_config_encoder.categories_[0][1:]]
    
    # Add interaction term names
    for i, name1 in enumerate(feature_names[:3]):
        for name2 in feature_names[3:]:
            feature_names.append(f'{name1} * {name2}')
    
    # Add quadratic term names
    for name in feature_names[:3]:
        feature_names.append(f'{name}^2')
    
    coverage_importance = dict(zip(feature_names, lasso_coverage.coef_))
    number_importance = dict(zip(feature_names, lasso_number.coef_))
    
    return jsonify({
        'coverage_importance': coverage_importance,
        'number_importance': number_importance
    })

if __name__ == '__main__':
    print("Running app in debug mode")
    app.run(debug=True)
else:
    print("App imported, ready to be run by WSGI server")