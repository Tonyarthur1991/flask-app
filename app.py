import numpy as np
import pandas as pd
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LassoCV
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_val_predict

app = Flask(__name__)
CORS(app)

# Global variables
data = None
model_coverage = None
model_number = None
X = None
y_coverage = None
y_number = None
data_means = None
data_stds = None
config_categories = None

def load_and_preprocess_data():
    global data, model_coverage, model_number, X, y_coverage, y_number, data_means, data_stds, config_categories
    
    try:
        data = pd.read_csv('Model_data.csv')
        print(f"Data loaded successfully. Shape: {data.shape}")
    except FileNotFoundError:
        print("Error: Model_data.csv not found!")
        return False
    
    # Convert Screw_Configuration to categorical
    data['Screw_Configuration'] = pd.Categorical(data['Screw_Configuration'])
    config_categories = data['Screw_Configuration'].cat.categories.tolist()

    # Normalize continuous variables
    continuous_vars = ['Screw_speed', 'Liquid_content', 'Liquid_binder']
    data_means = data[continuous_vars].mean()
    data_stds = data[continuous_vars].std()
    for var in continuous_vars:
        data[f'{var}_norm'] = (data[var] - data_means[var]) / data_stds[var]

    # Prepare the data for Lasso
    X = data[['Screw_speed_norm', 'Liquid_content_norm', 'Liquid_binder_norm']]
    y_coverage = data['Seed_coverage']
    y_number = data['number_seeded']

    # Create dummy variables for Screw_Configuration
    config_dummies = pd.get_dummies(data['Screw_Configuration'], drop_first=True)
    X = pd.concat([X, config_dummies], axis=1)

    # Add interaction terms and quadratic terms
    X['Speed_Content'] = X['Screw_speed_norm'] * X['Liquid_content_norm']
    X['Speed_Binder'] = X['Screw_speed_norm'] * X['Liquid_binder_norm']
    X['Content_Binder'] = X['Liquid_content_norm'] * X['Liquid_binder_norm']
    X['Speed_Squared'] = X['Screw_speed_norm'] ** 2
    X['Content_Squared'] = X['Liquid_content_norm'] ** 2
    X['Binder_Squared'] = X['Liquid_binder_norm'] ** 2

    # Apply Lasso Regression
    model_coverage = LassoCV(cv=10).fit(X, y_coverage)
    model_number = LassoCV(cv=10).fit(X, y_number)

    print("Models trained successfully")
    return True

def predict_model_lasso_with_ci(model, X_new, X_train, y_train, is_coverage=False):
    predictions = model.predict(X_new)
    y_cv_pred = cross_val_predict(model, X_train, y_train, cv=5)
    residuals = y_train - y_cv_pred
    se_residuals = np.std(residuals)
    n = len(y_train)
    ci_range = 1.96 * se_residuals * np.sqrt(1 + 1/n)
    ci = np.array([predictions - ci_range, predictions + ci_range]).T

    if is_coverage:
        predictions = np.clip(predictions, 0, 100)
        ci = np.clip(ci, 0, 100)
    else:
        predictions = np.maximum(predictions, 0)
        ci = np.maximum(ci, 0)

    return predictions, ci

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
        
        # Normalize new data
        X_new = (features - data_means.values) / data_stds.values
        
        # Create dummy variables for Screw_Configuration
        config_dummy = np.zeros((1, len(config_categories) - 1))
        if screw_config in config_categories[1:]:
            config_dummy[0, config_categories[1:].index(screw_config)] = 1
        
        X_new = np.hstack([X_new, config_dummy])
        
        # Add interaction terms and quadratic terms
        X_new = np.column_stack([
            X_new,
            X_new[:, 0] * X_new[:, 1],  # Speed * Content
            X_new[:, 0] * X_new[:, 2],  # Speed * Binder
            X_new[:, 1] * X_new[:, 2],  # Content * Binder
            X_new[:, 0] ** 2,           # Speed^2
            X_new[:, 1] ** 2,           # Content^2
            X_new[:, 2] ** 2            # Binder^2
        ])
        
        coverage_prediction, coverage_ci = predict_model_lasso_with_ci(model_coverage, X_new, X, y_coverage, is_coverage=True)
        number_prediction, number_ci = predict_model_lasso_with_ci(model_number, X_new, X, y_number)
        
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
    if model_coverage is None or model_number is None:
        return jsonify({"error": "Models not initialized"}), 500
    
    feature_names = X.columns.tolist()
    
    coverage_importance = dict(zip(feature_names, model_coverage.coef_))
    number_importance = dict(zip(feature_names, model_number.coef_))
    
    return jsonify({
        'coverage_importance': coverage_importance,
        'number_importance': number_importance
    })

@app.route('/screw_configs', methods=['GET'])
def get_screw_configs():
    return jsonify({
        'screw_configs': config_categories.tolist()
    })

if __name__ == '__main__':
    app.run(debug=True)
else:
    print("App imported, ready to be run by WSGI server")