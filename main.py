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
    """
    Predict occupancy using sensor data
    
    - **Temperature**: Temperature in Celsius
    - **Humidity**: Humidity percentage
    - **Light**: Light intensity
    - **CO2**: CO2 level in ppm
    - **HumidityRatio**: Humidity ratio
    """
    # Convert Pydantic model to dict
    input_dict = data.dict()
    
    df = pd.DataFrame([input_dict])
    X = preprocess_input(df, features, scaler)
    pred = make_prediction(model, X)

    return {
        "input": input_dict,
        "prediction": int(pred[0]),
        "occupancy": "Occupied" if pred[0] == 1 else "Not Occupied"
    }

# Add this to run directly
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)