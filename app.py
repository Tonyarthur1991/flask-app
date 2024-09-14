from flask import Flask, request, jsonify
import pandas as pd
import numpy as np
from sklearn.linear_model import LassoCV
from sklearn.preprocessing import StandardScaler, OneHotEncoder

app = Flask(__name__)

def run_predictions(data):
    # Convert Screw_Configuration to categorical
    data['Screw_Configuration'] = data['Screw_Configuration'].astype('category')

    # Normalizing continuous variables
    continuous_features = ['Screw_speed', 'Liquid_content', 'Liquid_binder']
    scaler = StandardScaler()
    data[continuous_features] = scaler.fit_transform(data[continuous_features])

    # Prepare the data for Lasso
    X = data[continuous_features]
    y_coverage = data['Seed_coverage']
    y_number = data['number_seeded']

    # One-hot encoding for Screw_Configuration
    ohe = OneHotEncoder(drop='first')
    config_dummies = ohe.fit_transform(data[['Screw_Configuration']]).toarray()
    X = np.hstack((X, config_dummies))

    # Add interaction terms and quadratic terms
    X = np.hstack((X,
                   X[:, 0] * X[:, 1].reshape(-1, 1),
                   X[:, 0] * X[:, 2].reshape(-1, 1),
                   X[:, 1] * X[:, 2].reshape(-1, 1),
                   X[:, 0]**2,
                   X[:, 1]**2,
                   X[:, 2]**2))

    # Apply Lasso Regression for y_coverage
    lasso_coverage = LassoCV(cv=10).fit(X, y_coverage)

    # Apply Lasso Regression for y_number
    lasso_number = LassoCV(cv=10).fit(X, y_number)

    # Extract coefficients and intercepts
    coef_coverage = lasso_coverage.coef_
    intercept_coverage = lasso_coverage.intercept_

    coef_number = lasso_number.coef_
    intercept_number = lasso_number.intercept_

    return {
        'coef_coverage': coef_coverage.tolist(),
        'intercept_coverage': intercept_coverage,
        'coef_number': coef_number.tolist(),
        'intercept_number': intercept_number
    }

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400

    file = request.files['file']
    try:
        data = pd.read_csv(file)
        results = run_predictions(data)
        return jsonify(results)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
