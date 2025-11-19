from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import numpy as np
import pandas as pd
from preprocess import preprocess_input
from predicttest import make_prediction

# Define input data model
class SensorData(BaseModel):
    Temperature: float
    Humidity: float
    Light: float
    CO2: float
    HumidityRatio: float

    class Config:
        schema_extra = {
            "example": {
                "Temperature": 23.1,
                "Humidity": 27.0,
                "Light": 300.0,
                "CO2": 900.0,
                "HumidityRatio": 0.004
            }
        }

app = FastAPI(
    title="Occupancy Detection API",
    description="Predicts occupancy using sensor data",
    version="1.0"
)

# Load model + scaler
model = joblib.load('model.pkl')
scaler = joblib.load('scaler.pkl')
features = joblib.load('features.pkl')

@app.get("/")
def home():
    return {"message": "Occupancy Prediction API is running."}

@app.post("/predict")
def predict(data: SensorData):

    input_dict = data.dict()   # This is already correct

    # Create df for debugging or logging (optional)
    df = pd.DataFrame([input_dict])

    # FIX: pass dictionary, not df
    X = preprocess_input(input_dict, features, scaler)

    pred = make_prediction(model, X)

    return {
        "input": input_dict,
        "prediction": int(pred[0]),
        "occupancy": "Occupied" if pred[0] == 1 else "Not Occupied"
    }


# Add this to run directly
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8005)