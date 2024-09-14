import sys
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import cross_val_predict
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator, TransformerMixin
import joblib
import os

app = Flask(__name__)
CORS(app)

# Global variables
data = None
model_pipeline_coverage = None
model_pipeline_number = None
y_coverage = None
y_number = None

class PolynomialFeatures(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        numeric_cols = X[:, :3]
        categorical_cols = X[:, 3:]
        poly_features = np.column_stack([
            numeric_cols,
            numeric_cols ** 2,
            np.prod(numeric_cols[:, :2], axis=1).reshape(-1, 1),
            np.prod(numeric_cols[:, [0, 2]], axis=1).reshape(-1, 1),
            np.prod(numeric_cols[:, 1:3], axis=1).reshape(-1, 1),
            categorical_cols
        ])
        return poly_features

def predict_with_confidence_intervals(X_new, model, y_train):
    y_pred = model.predict(X_new)
    y_cv_pred = cross_val_predict(model, X_new, y_train, cv=5)
    residuals = y_train - y_cv_pred
    std_residuals = np.std(residuals)
    ci = 1.96 * std_residuals / np.sqrt(len(y_train))
    return y_pred, np.array([y_pred - ci, y_pred + ci]).T

def load_and_preprocess_data():
    global data, model_pipeline_coverage, model_pipeline_number, y_coverage, y_number
    
    try:
        data = pd.read_csv('Model_data.csv')
        print(f"Data loaded successfully. Shape: {data.shape}")
    except FileNotFoundError:
        print("Error: Model_data.csv not found!")
        return False
    
    y_coverage = data['Seed_coverage'].values
    y_number = data['number_seeded'].values

    numeric_features = ['Screw_speed', 'Liquid_content', 'Liquid_binder']
    categorical_features = ['Screw_Configuration']

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numeric_features),
            ('cat', OneHotEncoder(drop='first', sparse_output=False), categorical_features)
        ])

    model_pipeline_coverage = Pipeline([
        ('preprocessor', preprocessor),
        ('poly', PolynomialFeatures()),
        ('regressor', Ridge(alpha=1.0))
    ])

    model_pipeline_number = Pipeline([
        ('preprocessor', preprocessor),
        ('poly', PolynomialFeatures()),
        ('regressor', Ridge(alpha=1.0))
    ])

    model_pipeline_coverage.fit(data[numeric_features + categorical_features], y_coverage)
    model_pipeline_number.fit(data[numeric_features + categorical_features], y_number)

    print("Models trained successfully")
    return True

if load_and_preprocess_data():
    print("Data loaded and models trained successfully")
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
        screw_config = input_data['screw_config'][0]
        
        print(f"Input features: {features}")
        print(f"Input screw config: {screw_config}")
        
        input_df = pd.DataFrame(features, columns=['Screw_speed', 'Liquid_content', 'Liquid_binder'])
        input_df['Screw_Configuration'] = screw_config
        
        coverage_prediction, coverage_ci = predict_with_confidence_intervals(input_df, model_pipeline_coverage, y_coverage)
        number_prediction, number_ci = predict_with_confidence_intervals(input_df, model_pipeline_number, y_number)
        
        # Ensure non-negative predictions and clip coverage to 0-100%
        coverage_prediction = np.clip(coverage_prediction, 0, 100)
        coverage_ci = np.clip(coverage_ci, 0, 100)
        number_prediction = np.maximum(number_prediction, 0)
        number_ci = np.maximum(number_ci, 0)
        
        print(f"Coverage prediction: {coverage_prediction}")
        print(f"Coverage CI: {coverage_ci}")
        print(f"Number prediction: {number_prediction}")
        print(f"Number CI: {number_ci}")
        
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
    if model_pipeline_coverage is None or model_pipeline_number is None:
        return jsonify({"error": "Models not initialized"}), 500
    
    preprocessor = model_pipeline_coverage.named_steps['preprocessor']
    feature_names = (
        preprocessor.named_transformers_['num'].get_feature_names_out().tolist() +
        preprocessor.named_transformers_['cat'].get_feature_names_out().tolist()
    )
    feature_names += [
        'Screw_speed^2', 'Liquid_content^2', 'Liquid_binder^2',
        'Screw_speed * Liquid_content',
        'Screw_speed * Liquid_binder',
        'Liquid_content * Liquid_binder'
    ]
    
    coverage_importance = dict(zip(feature_names, model_pipeline_coverage.named_steps['regressor'].coef_))
    number_importance = dict(zip(feature_names, model_pipeline_number.named_steps['regressor'].coef_))
    
    return jsonify({
        'coverage_importance': coverage_importance,
        'number_importance': number_importance
    })

if __name__ == '__main__':
    app.run(debug=True)
else:
    print("App imported, ready to be run by WSGI server")