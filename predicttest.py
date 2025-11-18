def make_prediction(model, processed_X):
    """
    model: loaded ML model
    processed_X: scaled feature vector from preprocess_input
    """
    prediction = model.predict(processed_X)
    return prediction.tolist()
