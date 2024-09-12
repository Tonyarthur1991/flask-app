from flask import Flask, request, jsonify
import numpy as np
from sklearn.linear_model import LassoCV
from sklearn.preprocessing import StandardScaler, OneHotEncoder

# Initialize Flask app
app = Flask(__name__)

# Example training data
data = np.array([[150, 0.1, 0.05], [200, 0.12, 0.07], [300, 0.14, 0.09]])  # Example training data
y_coverage = np.array([60, 80, 95])  # Example target for seed coverage
y_number = np.array([50, 65, 85])  # Example target for number of seeded granules

# Fit the Lasso model
scaler = StandardScaler()
X_scaled = scaler.fit_transform(data)
lasso_coverage = LassoCV(cv=3).fit(X_scaled, y_coverage)
lasso_number = LassoCV(cv=3).fit(X_scaled, y_number)

@app.route('/predict', methods=['POST'])
def predict():
    # Get input features from the request JSON body
    input_data = request.get_json(force=True)
    features = np.array(input_data['features']).reshape(1, -1)
    
    # Normalize the input features
    features_scaled = scaler.transform(features)
    
    # Make predictions
    coverage_prediction = lasso_coverage.predict(features_scaled)
    number_prediction = lasso_number.predict(features_scaled)
    
    # Return predictions as JSON
    return jsonify({
        'coverage_prediction': coverage_prediction[0],
        'number_prediction': number_prediction[0]
    })
