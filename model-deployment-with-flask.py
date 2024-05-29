# Example Code Snippet for Model Deployment with Flask

from flask import Flask, request, jsonify
import joblib
import numpy as np

# Load the trained model
model = joblib.load('model.pkl')

# Initialize Flask app
app = Flask(__name__)

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json(force=True)
    prediction = model.predict(np.array(data['features']).reshape(1, -1))
    return jsonify({'prediction': int(prediction[0])})

if __name__ == '__main__':
    app.run(debug=True)


###############################################################################################
# Dockerfile for Containerization

# Use an official Python runtime as a parent image
FROM python:3.8-slim

# Set the working directory in the container
WORKDIR /app

# Copy the current directory contents into the container at /app
COPY . /app

# Install any needed packages specified in requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Make port 5000 available to the world outside this container
EXPOSE 5000

# Define environment variable
ENV NAME World

# Run app.py when the container launches
CMD ["python", "app.py"]

###############################################################################################
# Monitoring Example

from flask import Flask, request, jsonify
from prometheus_client import Counter, generate_latest, Summary

# Metrics
REQUEST_COUNT = Counter('request_count', 'Number of requests received')
REQUEST_LATENCY = Summary('request_latency_seconds', 'Time spent processing request')

app = Flask(__name__)

@app.route('/predict', methods=['POST'])
@REQUEST_LATENCY.time()
def predict():
    REQUEST_COUNT.inc()
    data = request.get_json(force=True)
    prediction = model.predict(np.array(data['features']).reshape(1, -1))
    return jsonify({'prediction': int(prediction[0])})

@app.route('/metrics', methods=['GET'])
def metrics():
    return generate_latest(), 200

if __name__ == '__main__':
    app.run(debug=True)

