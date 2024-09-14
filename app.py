from flask import Flask, request, jsonify, send_from_directory
import numpy as np
from sklearn.linear_model import LassoCV
from sklearn.preprocessing import StandardScaler, OneHotEncoder

app = Flask(__name__)

def run_predictions(features, screw_config):
    # Assuming features are Screw_speed, Liquid_content, Liquid_binder
    # Assuming screw_config is a list of one value representing the configuration

    # Normalizing continuous variables
    scaler = StandardScaler()
    features = scaler.fit_transform(np.array(features).reshape(1, -1))
    
    # One-hot encoding for Screw_Configuration
    ohe = OneHotEncoder(drop='first')
    config_dummies = ohe.fit_transform([[screw_config[0]]]).toarray()
    X = np.hstack((features, config_dummies))
    
    # Add interaction terms and quadratic terms
    X = np.hstack((X,
                   X[:, 0] * X[:, 1],
                   X[:, 0] * X[:, 2],
                   X[:, 1] * X[:, 2],
                   X[:, 0]**2,
                   X[:, 1]**2,
                   X[:, 2]**2))
    
    # Dummy Lasso Regression models (normally you'd train these models beforehand)
    coef_coverage = np.random.rand(X.shape[1])
    intercept_coverage = np.random.rand(1)
    
    coef_number = np.random.rand(X.shape[1])
    intercept_number = np.random.rand(1)

    # Mock predictions
    coverage_prediction = np.dot(X, coef_coverage) + intercept_coverage
    number_prediction = np.dot(X, coef_number) + intercept_number

    # Mock confidence intervals
    coverage_ci = [coverage_prediction - 5, coverage_prediction + 5]
    number_ci = [number_prediction - 3, number_prediction + 3]

    return {
        'coverage_prediction': coverage_prediction[0],
        'coverage_ci': [coverage_ci[0][0], coverage_ci[1][0]],
        'number_prediction': number_prediction[0],
        'number_ci': [number_ci[0][0], number_ci[1][0]]
    }

@app.route('/')
def index():
    return send_from_directory('static', 'index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        features = data['features']
        screw_config = data['screw_config']
        results = run_predictions(features, screw_config)
        return jsonify(results)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
