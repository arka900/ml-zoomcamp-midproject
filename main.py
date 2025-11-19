from fastapi import FastAPI
from pydantic import BaseModel
from predict import predict   

# FastAPI app
app = FastAPI(
    title="Occupancy Detection API",
    description="Predicts occupancy using sensor data",
    version="1.0"
)

# Input schema
class SensorData(BaseModel):
    Temperature: float
    Humidity: float
    Light: float
    CO2: float
    HumidityRatio: float

    class Config:
        json_schema_extra = {
            "example": {
                "Temperature": 23.1,
                "Humidity": 27.0,
                "Light": 300.0,
                "CO2": 900.0,
                "HumidityRatio": 0.004
            }
        }


@app.get("/")
def home():
    return {"message": "Occupancy Prediction API is running."}


@app.post("/predict")
def predict_api(data: SensorData):

    # Convert request into dictionary
    input_dict = data.model_dump()

    # Call your unified prediction function
    result = predict(input_dict)

    return {
        "input": input_dict,
        "occupancy_prediction": result["occupancy_prediction"],
        "occupancy_probability": result["occupancy_probability"],
        "status": "Occupied" if result["occupancy_prediction"] == 1 else "Not Occupied"
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
