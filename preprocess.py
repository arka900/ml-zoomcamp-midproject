# preprocess.py

import numpy as np
import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier

from ucimlrepo import fetch_ucirepo

def load_and_clean_data():
    # Load dataset
    data = fetch_ucirepo(id=357)
    X = data.data.features
    y = data.data.targets.squeeze()

    # Drop unused columns
    if "date" in X.columns:
        X = X.drop("date", axis=1)

    # Drop NaN values
    mask = ~y.isna()
    X = X[mask]
    y = y[mask]

    # Convert numeric
    X = X.apply(pd.to_numeric, errors="coerce")

    return X, y


def train_model(X, y):
    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)

    # Choose final model (RandomForest is reliable and robust)
    model = RandomForestClassifier(n_estimators=300, random_state=42)

    # Train model
    model.fit(X_train_scaled, y_train)

    return model, scaler, X.columns.tolist()


def save_artifacts(model, scaler, features):
    with open("model.pkl", "wb") as f:
        pickle.dump(model, f)

    with open("scaler.pkl", "wb") as f:
        pickle.dump(scaler, f)

    with open("features.pkl", "wb") as f:
        pickle.dump(features, f)

    print("Artifacts saved: model.pkl, scaler.pkl, features.pkl")


if __name__ == "__main__":
    X, y = load_and_clean_data()
    model, scaler, features = train_model(X, y)
    save_artifacts(model, scaler, features)
    print("Training pipeline complete.")
