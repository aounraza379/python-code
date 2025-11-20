from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import joblib
import numpy as np
import os

app = Flask(__name__, template_folder='templates')
CORS(app)

# Load models once
poly = joblib.load('poly_features.pkl')
regressor = joblib.load('linear_regression_model.pkl')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json(force=True)
    rooms = data['rooms']
    area = data['area']
    age = data['age']
    features = np.array([[rooms, area, age]])
    features_poly = poly.transform(features)
    prediction = regressor.predict(features_poly)
    return jsonify({'predicted_price': prediction[0]})

@app.route('/')
def home():
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)
