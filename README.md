
---

# ✅ **README Instructions (Copy-Paste Ready)**

# 🏠 Occupancy Prediction API

This project trains classifier models on the UCI Occupancy Detection dataset and exposes it as a REST API using **FastAPI**.
The project includes training, artifact generation, and full Docker deployment.

---


# 🚀 **How to Replicate This Project**

## **1️⃣ Clone the Repository**

```bash
git clone <your-repo-url>
cd ml-zoomcamp-midproject
```

---

# **2️⃣ Create the Conda Environment (Optional for Local Dev)**

If you want to run everything locally:

```bash
conda env create -f environment.yaml
conda activate midproject
```

---

# **3️⃣ Train the Model and Generate Artifacts**

Run:

```bash
python train.py
```

This will create:

```
model.pkl
scaler.pkl
features.pkl
```

These files are automatically loaded by the FastAPI app.

---

# **4️⃣ Run the API Locally (without Docker)**

```bash
uvicorn main:app --reload
```

Then open:

```
http://127.0.0.1:8000/docs
```

You’ll see Swagger UI where you can test predictions.

---

# **5️⃣ Build the Docker Image**

Make sure **Docker Desktop is running**, then run:

```bash
docker build -t occupancy-api .
```

---

# **6️⃣ Run the Container**

```bash
docker run -p 8000:8000 occupancy-api
```

The API is now available at:

```
http://127.0.0.1:8000/docs
```

---

# **7️⃣ Making Predictions**

### POST `/predict`

Example JSON:

```json
{
  "Temperature": 23.18,
  "Humidity": 27.20,
  "Light": 426.0,
  "CO2": 721.25,
  "HumidityRatio": 0.0048
}
```

Response:

```json
{
  "prediction": 1,
  "label": "Occupied"
}
```

---

# 🧪 **8️⃣ Re-training with Modified Model**

Simply re-run:

```bash
python train.py
```

Artifacts will update and Docker will use them on next build.

---

# 📄 **9️⃣ .gitignore Notes**

To avoid committing large or sensitive files, your `.gitignore` should include:

```
model.pkl
scaler.pkl
features.pkl
__pycache__/
```

---

# 🎯 **10️⃣ Summary**

| Step | Action                           |
| ---- | -------------------------------- |
| 1    | Clone the repo                   |
| 2    | Create Conda environment         |
| 3    | Train the model & save artifacts |
| 4    | Run API locally using FastAPI    |
| 5    | Build Docker image               |
| 6    | Run container                    |
| 7    | Test predictions in Swagger      |
