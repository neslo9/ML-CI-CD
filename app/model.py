from sklearn.datasets import load_breast_cancer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import numpy as np

def load_model_and_data():
    data = load_breast_cancer()
    X_train, X_test, y_train, y_test = train_test_split(data.data, data.target, test_size=0.2, random_state=42)
    model = LogisticRegression(max_iter=10000)
    model.fit(X_train, y_train)
    return model, data

def predict_sample(model, input_data):
    from sklearn.datasets import load_breast_cancer
    data = load_breast_cancer()
    prediction = model.predict(input_data)[0]
    label = data.target_names[prediction]
    return {"prediction": int(prediction), "label": label}

def train_and_predict():
    data = load_breast_cancer()
    X_train, X_test, y_train, y_test = train_test_split(data.data, data.target, test_size=0.2, random_state=42)
    model = LogisticRegression(max_iter=10000)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    return y_pred, y_test, acc