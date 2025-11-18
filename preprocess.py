import pandas as pd
import numpy as np

def preprocess_input(input_dict, features, scaler):
    """
    input_dict: raw JSON dict from FastAPI input
    features: list of required features (from features.pkl)
    scaler: loaded sklearn scaler
    """

    # Convert dict → DataFrame with ONE row
    df = pd.DataFrame([input_dict])

    # Ensure correct feature order
    df = df[features]

    # Scale
    df_scaled = scaler.transform(df)

    return df_scaled
