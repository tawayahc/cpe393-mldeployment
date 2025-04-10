import pickle
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier

iris = load_iris()
X, y = iris.data, iris.target

model = RandomForestClassifier()
model.fit(X, y)

with open("models/iris_model.pkl", "wb") as f:
    pickle.dump(model, f)
