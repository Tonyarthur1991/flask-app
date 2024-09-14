import numpy as np
import pandas as pd
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LinearRegression
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
preprocessor = None
config_categories = None

def load_and_preprocess_data():
    global data, model_coverage, model_number, X, y_coverage, y_number, preprocessor, config_categories
    
    try:
        data = pd.read_csv('Model_data.csv')
        print(f"Data loaded successfully. Shape: {data.shape}")
        print(f"Columns: {data.columns}")
        print(f"Data types:\n{data.dtypes}")
        print(f"First few rows:\n{data.head()}")
    except FileNotFoundError:
        print("Error: Model_data.csv not found!")
        return False
    
    # Convert Screw_Configuration to categorical
    data['Screw_Configuration'] = pd.Categorical(data['Screw_Configuration'])
    config_categories = data['Screw_Configuration'].cat.categories.tolist()
    print(f"Screw Configuration categories: {config_categories}")

    # Prepare the data
    X = data[['Screw_speed', 'Liquid_content', 'Liquid_binder', 'Screw_Configuration']]
    y_coverage = data['Seed_coverage']
    y_number = data['number_seeded']

    # Create preprocessor
    numeric_features = ['Screw_speed', 'Liquid_content', 'Liquid_binder']
    categorical_features = ['Screw_Configuration']
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numeric_features),
            ('cat', OneHotEncoder(drop='first', sparse_output=False), categorical_features)
        ])

    # Fit preprocessor and transform data
    X_preprocessed = preprocessor.fit_transform(X)
    print(f"Preprocessed X shape: {X_preprocessed.shape}")
    print(f"Preprocessed X sample:\n{X_preprocessed[:5]}")
    
    # Add interaction terms and quadratic terms
    X_interaction = np.column_stack([
        X_preprocessed,
        X_preprocessed[:, 0] * X_preprocessed[:, 1],  # Speed * Content
        X_preprocessed[:, 0] * X_preprocessed[:, 2],  # Speed * Binder
        X_preprocessed[:, 1] * X_preprocessed[:, 2],  # Content * Binder
        X_preprocessed[:, 0] ** 2,                    # Speed^2
        X_preprocessed[:, 1] ** 2,                    # Content^2
        X_preprocessed[:, 2] ** 2                     # Binder^2
    ])
    print(f"X_interaction shape: {X_interaction.shape}")
    print(f"X_interaction sample:\n{X_interaction[:5]}")

    # Train models
    model_coverage = LinearRegression().fit(X_interaction, y_coverage)
    model_number = LinearRegression().fit(X_interaction, y_number)

    print("Models trained successfully")
    print(f"Coverage model coefficients: {model_coverage.coef_}")
    print(f"Coverage model intercept: {model_coverage.intercept_}")
    print(f"Number model coefficients: {model_number.coef_}")
    print(f"Number model intercept: {model_number.intercept_}")
    return True

def predict_with_confidence_intervals(model, X_new, X_train, y_train, is_coverage=False):
    predictions = model.predict(X_new)
    
    # Use the preprocessed X_train for cross-validation
    X_train_preprocessed = preprocessor.transform(X_train)
    X_train_interaction = np.column_stack([
        X_train_preprocessed,
        X_train_preprocessed[:, 0] * X_train_preprocessed[:, 1],
        X_train_preprocessed[:, 0] * X_train_preprocessed[:, 2],
        X_train_preprocessed[:, 1] * X_train_preprocessed[:, 2],
        X_train_preprocessed[:, 0] ** 2,
        X_train_preprocessed[:, 1] ** 2,
        X_train_preprocessed[:, 2] ** 2
    ])
    
    y_cv_pred = cross_val_predict(model, X_train_interaction, y_train, cv=5)
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
        
        # Prepare input data
        X_new = pd.DataFrame(features, columns=['Screw_speed', 'Liquid_content', 'Liquid_binder'])
        X_new['Screw_Configuration'] = screw_config
        
        # Preprocess input data
        X_new_preprocessed = preprocessor.transform(X_new)
        print(f"Preprocessed input: {X_new_preprocessed}")
        
        # Add interaction terms and quadratic terms
        X_new_interaction = np.column_stack([
            X_new_preprocessed,
            X_new_preprocessed[:, 0] * X_new_preprocessed[:, 1],  # Speed * Content
            X_new_preprocessed[:, 0] * X_new_preprocessed[:, 2],  # Speed * Binder
            X_new_preprocessed[:, 1] * X_new_preprocessed[:, 2],  # Content * Binder
            X_new_preprocessed[:, 0] ** 2,                        # Speed^2
            X_new_preprocessed[:, 1] ** 2,                        # Content^2
            X_new_preprocessed[:, 2] ** 2                         # Binder^2
        ])
        
        print(f"Final X_new shape: {X_new_interaction.shape}")
        print(f"Final X_new: {X_new_interaction}")
        
        # Manual prediction for coverage
        manual_pred_coverage = model_coverage.predict(X_new_interaction)
        print(f"Manual prediction for coverage: {manual_pred_coverage}")
        
        # Manual prediction for number
        manual_pred_number = model_number.predict(X_new_interaction)
        print(f"Manual prediction for number: {manual_pred_number}")
        
        coverage_prediction, coverage_ci = predict_with_confidence_intervals(model_coverage, X_new_interaction, X, y_coverage, is_coverage=True)
        number_prediction, number_ci = predict_with_confidence_intervals(model_number, X_new_interaction, X, y_number)
        
        print(f"Raw coverage prediction: {coverage_prediction}")
        print(f"Raw coverage CI: {coverage_ci}")
        print(f"Raw number prediction: {number_prediction}")
        print(f"Raw number CI: {number_ci}")
        
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
    
    feature_names = ['Screw_speed', 'Liquid_content', 'Liquid_binder'] + [f'Screw_Config_{cat}' for cat in config_categories[1:]]
    feature_names += [
        'Speed * Content', 'Speed * Binder', 'Content * Binder',
        'Speed^2', 'Content^2', 'Binder^2'
    ]
    
    coverage_importance = dict(zip(feature_names, model_coverage.coef_))
    number_importance = dict(zip(feature_names, model_number.coef_))
    
    return jsonify({
        'coverage_importance': coverage_importance,
        'number_importance': number_importance
    })

@app.route('/screw_configs', methods=['GET'])
def get_screw_configs():
    return jsonify({
        'screw_configs': config_categories
    })

if __name__ == '__main__':
    app.run(debug=True)
else:
    print("App imported, ready to be run by WSGI server")