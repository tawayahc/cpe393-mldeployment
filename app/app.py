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
    input_features = np.array(data["features"]).reshape(1, -1)
    
    prediction = iris_model.predict(input_features)[0]
    proba = iris_model.predict_proba(input_features)[0]
    confidence = float(np.max(proba))
    
    return jsonify({
        "prediction": int(prediction),
        "confidence": round(confidence, 2)
    })

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=9000)
