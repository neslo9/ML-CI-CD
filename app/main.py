import os
import numpy as np
from fastapi import FastAPI, HTTPException
from app.model import load_model_and_data, predict_sample
from app.schemas import PredictRequest

API_KEY = os.getenv("API_KEY")
if not API_KEY:
    raise RuntimeError("Brak wymaganego API_KEY w zmiennych środowiskowych")

model, data = load_model_and_data()

app = FastAPI(title="API do serwowania modelu ML")

@app.get("/")
def read_root():
    return {"message": "Witamy w API serwującym model ML"}

@app.post("/predict")
def predict(request: PredictRequest):
    try:
        input_data = np.array(request.features).reshape(1, -1)
        if input_data.shape[1] != len(data.feature_names):
            raise HTTPException(status_code=400, detail=f"Oczekiwano {len(data.feature_names)} cech.")
        prediction = predict_sample(model, input_data)
        return prediction
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.get("/info")
def get_info():
    return {
        "model": "LogisticRegression",
        "n_features": len(data.feature_names),
        "feature_names": data.feature_names.tolist()
    }

@app.get("/health")
def health_check():
    return {"status": "ok"}

@app.get("/config")
def read_config():
    return {"api_key": API_KEY}