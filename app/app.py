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
    input_features = np.array(data["features"])
    predictions = iris_model.predict(input_features)
    return jsonify({ "prediction": predictions.tolist() })

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=9000)
