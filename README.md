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