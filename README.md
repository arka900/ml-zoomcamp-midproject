
---

# 🏠 Occupancy Detection – ML Zoomcamp Midterm Project

This project predicts whether a room is **occupied** based on environmental sensor readings such as temperature, humidity, CO₂ levels, and light intensity.

It includes:

* A full **machine learning pipeline**
* A **FastAPI** deployment
* A **Dockerized API** for easy reproduction
* Clean modular code (`preprocess.py`, `predict.py`)
* Pickled model artifacts

---

## 📁 Project Structure

```
ml-zoomcamp-midproject/
│
├── preprocess.py          # Training & preprocessing pipeline
├── predict.py             # Load model + return predictions
├── main.py                # FastAPI app
├── environment.yaml       # Conda environment
├── features.pkl           # Feature order for prediction
├── model.pkl              # Trained ML model
├── scaler.pkl             # StandardScaler
├── Dockerfile             # Containerization
├── test_data.json         # Sample payload
└── README.md
```

---

## 📦 1. Create the Conda Environment

```bash
conda env create -f environment.yaml
conda activate midproject
```

---

## 🧠 2. Train the Model (Optional)

If you want to retrain the model:

```bash
python preprocess.py
```

This will generate:

* `model.pkl`
* `scaler.pkl`
* `features.pkl`

These files are **ignored by Git** using `.gitignore`:

```
*.pkl
```

---

## 🔮 3. Make Predictions Locally

You can test prediction directly in Python:

```bash
python predict.py
```

Or import the function:

```python
from predict import predict

sample = {
    "Temperature": 23.5,
    "Humidity": 27.2,
    "Light": 450,
    "CO2": 950,
    "HumidityRatio": 0.004
}

print(predict(sample))
```

---

## ⚡ 4. Run the FastAPI App Locally

```bash
uvicorn main:app --reload
```

Open the browser:

* API root: **[http://127.0.0.1:8000](http://127.0.0.1:8000)**
* Docs UI: **[http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)**

Use the POST `/predict` endpoint.

---

## 🐳 5. Build & Run the Docker Image

### **Build the image**

```bash
docker build -t occupancy-api .
```

### **Run the container**

```bash
docker run -p 8000:8000 occupancy-api
```

API will be available at:

```
http://127.0.0.1:8000
http://127.0.0.1:8000/docs
```

---

## 🧪 6. Example API Payload

`test_data.json`:

```json
{
  "Temperature": 23.5,
  "Humidity": 27.2,
  "Light": 450,
  "CO2": 950,
  "HumidityRatio": 0.004
}
```

Use with curl:

```bash
curl -X POST http://127.0.0.1:8000/predict \
     -H "Content-Type: application/json" \
     -d @test_data.json
```
