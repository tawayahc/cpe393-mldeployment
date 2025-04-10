# **CPE393 ML DEPLOYMENT LAB**

## ⚙️ Setup Project

**Model Export** - Run `iris_train.py` (`iris_model.pkl` will be saved in models folder)
```
python scripts/iris_train.py
```

**Build Docker image**
```
docker build -t ml-model .
```

**Run Docker container**
```
docker run -p 9000:9000 ml-model
```

**Test the API in new terminal or Postman**
```
curl -X POST http://localhost:9000/predict \
     -H "Content-Type: application/json" \
     -d '{"features": [5.1, 3.5, 1.4, 0.2]}'
```

**Expected output**
```json
{
     "prediction": 0
}
```

**Output Screenshot**
> ![Setup Project](images/setup_project.png)


## ✅ Exercise 1: Add Confidence Scores
> **Task**: Update the `/predict` endpoint to return the prediction and the confidence score using `predict_proba()`

**Expected Output Example:**
```json
{
  "prediction": 0,
  "confidence": 0.97
}
```

**Updated Function**
```python
@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()
    input_features = np.array(data["features"]).reshape(1, -1)
    
    prediction = iris_model.predict(input_features)[0]
    proba = iris_model.predict_proba(input_features)[0]     # Added: Predict confidence
    confidence = float(np.max(proba))                       # Added: Extract the highest confidence of the predicted class
    
    return jsonify({
        "prediction": int(prediction),
        "confidence": round(confidence, 2)                  # Added: Round decimals
    })
```

**Output Screenshot**
> ![Exercise 1](images/exercise_1.png)