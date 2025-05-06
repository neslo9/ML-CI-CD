from pydantic import BaseModel, Field
from typing import List
from sklearn.datasets import load_breast_cancer

data = load_breast_cancer()

class PredictRequest(BaseModel):
    features: List[float] = Field(..., min_items=len(data.feature_names), max_items=len(data.feature_names))