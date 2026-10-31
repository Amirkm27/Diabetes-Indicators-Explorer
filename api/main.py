from fastapi import FastAPI
import joblib
import pandas as pd
from pydantic import BaseModel

app = FastAPI()

artifact = joblib.load("models/diabetes_model.joblib")

model = artifact["model"]
threshold = artifact["threshold"]


@app.get("/health")
def health():
    return {
        "status": "ok",
        "model_loaded": True,
        "threshold": threshold
    }

class PredictionInput(BaseModel):
    HighBP: int
    HighChol: int
    CholCheck: int
    BMI: float
    Smoker: int
    Stroke: int
    HeartDiseaseorAttack: int
    PhysActivity: int
    Fruits: int
    Veggies: int
    HvyAlcoholConsump: int
    AnyHealthcare: int
    NoDocbcCost: int
    GenHlth: int
    MentHlth: int
    PhysHlth: int
    DiffWalk: int
    Sex: int
    Age: int
    Education: int
    Income: int

@app.post("/predict")
def predict(data: PredictionInput):

    X = pd.DataFrame([data.model_dump()])
    X["Age_BMI"] = X["Age"] * X["BMI"]

    probability = model.predict_proba(X)[0, 1]
    prediction = int(probability >= threshold)

    return {
        "prediction": prediction,
        "probability": round(float(probability), 4),
        "threshold": threshold
    }