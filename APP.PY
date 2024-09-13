from flask import Flask, request, jsonify
import numpy as np
import pandas as pd
from sklearn.linear_model import LassoCV
from sklearn.preprocessing import StandardScaler, OneHotEncoder

app = Flask(__name__)

# Root route to display a welcome message
@app.route('/')
def home():
    return "Welcome to the Flask API! Use the /predict endpoint for predictions."

# Step 1: Load the data
data = pd.read_csv('Model_data.csv')

# Step 2: Convert Screw_Configuration to categorical and one-hot encode it
screw_config_encoder = OneHotEncoder(drop='first')  # Drop the first category to avoid multicollinearity
screw_config_encoded = screw_config_encoder.fit_transform(data[['Screw_Configuration']]).toarray()

# Step 3: Normalize continuous variables
scaler = StandardScaler()
data[['Screw_speed_norm', 'Liquid_binder_norm', 'Liquid_content_norm']] = scaler.fit_transform(
    data[['Screw_speed', 'Liquid_binder', 'Liquid_content']]
)

# Step 4: Prepare the data for Lasso
# Combine the normalized continuous features and the one-hot encoded categorical features
X = np.hstack([data[['Screw_speed_norm', 'Liquid_binder_norm', 'Liquid_content_norm']].values, screw_config_encoded])
y_coverage = data['Seed_coverage'].values
y_number = data['number_seeded'].values

# Step 5: Add interaction terms and quadratic terms
X_interaction = np.hstack([
    X,
    (X[:, 0] * X[:, 1]).reshape(-1, 1),  # Interaction between Screw speed and Liquid binder
    (X[:, 0] * X[:, 2]).reshape(-1, 1),  # Interaction between Screw speed and Liquid content
    (X[:, 1] * X[:, 2]).reshape(-1, 1),  # Interaction between Liquid binder and Liquid content
    (X[:, 0] ** 2).reshape(-1, 1),       # Quadratic term for Screw speed
    (X[:, 1] ** 2).reshape(-1, 1),       # Quadratic term for Liquid binder
    (X[:, 2] ** 2).reshape(-1, 1)        # Quadratic term for Liquid content
])

# Step 6: Apply Lasso regression for seed coverage and number of seeded granules
lasso_coverage = LassoCV(cv=10).fit(X_interaction, y_coverage)
lasso_number = LassoCV(cv=10).fit(X_interaction, y_number)

# Step 7: Define prediction function with confidence intervals
def predict_with_confidence_intervals(X_new, model, y_train, X_train):
    predictions = model.predict(X_new)
    residuals = y_train - model.predict(X_train)
    se_residuals = np.std(residuals)
    ci_range = 1.96 * se_residuals * np.sqrt(1 + 1 / len(X_train))
    ci = np.array([predictions - ci_range, predictions + ci_range]).T
    return predictions, ci

# Step 8: Prediction endpoint
@app.route('/predict', methods=['POST'])
def predict():
    input_data = request.get_json(force=True)
    features = np.array(input_data['features']).reshape(1, -1)
    screw_config_input = np.array(input_data['screw_config']).reshape(1, -1)

    # Normalize continuous variables
    features_scaled = scaler.transform(features)

    # One-hot encode Screw Configuration
    screw_config_encoded = screw_config_encoder.transform(screw_config_input).toarray()

    # Combine all features for prediction
    X_new = np.hstack([features_scaled, screw_config_encoded])
    X_new = np.hstack([
        X_new,
        (X_new[:, 0] * X_new[:, 1]).reshape(-1, 1),  # Interaction terms
        (X_new[:, 0] * X_new[:, 2]).reshape(-1, 1),
        (X_new[:, 1] * X_new[:, 2]).reshape(-1, 1),
        (X_new[:, 0] ** 2).reshape(-1, 1),           # Quadratic terms
        (X_new[:, 1] ** 2).reshape(-1, 1),
        (X_new[:, 2] ** 2).reshape(-1, 1)
    ])

    # Predict coverage and number of seeded granules with confidence intervals
    coverage_prediction, coverage_ci = predict_with_confidence_intervals(X_new, lasso_coverage, y_coverage, X_interaction)
    number_prediction, number_ci = predict_with_confidence_intervals(X_new, lasso_number, y_number, X_interaction)

    # Cap seed coverage prediction at 100%
    coverage_prediction = np.clip(coverage_prediction, 0, 100)
    coverage_ci = np.clip(coverage_ci, 0, 100)

    return jsonify({
        'coverage_prediction': coverage_prediction[0],
        'coverage_ci': coverage_ci[0].tolist(),
        'number_prediction': number_prediction[0],
        'number_ci': number_ci[0].tolist()
    })

if __name__ == '__main__':
    app.run(debug=True)
