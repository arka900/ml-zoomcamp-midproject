# predict.py

import pickle
import numpy as np
import pandas as pd

# Load artifacts
with open("model.pkl", "rb") as f:
    model = pickle.load(f)

with open("scaler.pkl", "rb") as f:
    scaler = pickle.load(f)

with open("features.pkl", "rb") as f:
    feature_list = pickle.load(f)


def predict(input_dict: dict):
    """
    input_dict example:
    {
        "Temperature": 23.5,
        "Humidity": 27.2,
        "Light": 450,
        "CO2": 950,
        "HumidityRatio": 0.004
    }
    """

    # Convert to DataFrame with same column order
    df = pd.DataFrame([input_dict], columns=feature_list)

    # Scale using training scaler
    df_scaled = scaler.transform(df)

    # Predict
    pred = model.predict(df_scaled)[0]
    prob = model.predict_proba(df_scaled)[0, 1]

    return {
        "occupancy_prediction": int(pred),
        "occupancy_probability": float(prob)
    }


if __name__ == "__main__":
    sample = {
        "Temperature": 23.5,
        "Humidity": 27.2,
        "Light": 450,
        "CO2": 950,
        "HumidityRatio": 0.004
    }

    print(predict(sample))
