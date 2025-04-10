from flask import Flask, request, jsonify
import pickle
import numpy as np

app = Flask(__name__)

with open("models/iris_model.pkl", "rb") as f:
    iris_model = pickle.load(f)

@app.route("/")
def home():
    return "ML Model is Running"

@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()

    if not data:
        return jsonify({ "error": "No input data provided" }), 400

    if "features" not in data:
        return jsonify({ "error": '"features" key is missing' }), 400
    
    features = data["features"]
    if not isinstance(features, list) or not all(
        isinstance(row, list) and len(row) == 4 and all(isinstance(x, (int, float)) for x in row)
        for row in features
    ):
        return jsonify({ "error": 'Each item in "features" must be a list of exactly 4 numbers values' }), 400

    input_features = np.array(features)
    predictions = iris_model.predict(input_features)
    return jsonify({ "prediction": predictions.tolist() })

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=9000)
