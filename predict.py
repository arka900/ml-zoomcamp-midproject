import pandas as pd
from artifacts import model, scaler, feature_list

def predict(input_dict: dict):
    # Convert to DataFrame in correct order
    df = pd.DataFrame([input_dict], columns=feature_list)

    # Scale
    df_scaled = scaler.transform(df)

    # Predict
    pred = model.predict(df_scaled)[0]
    prob = model.predict_proba(df_scaled)[0, 1]

    return {
        "occupancy_prediction": int(pred),
        "occupancy_probability": float(prob)
    }
