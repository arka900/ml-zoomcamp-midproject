
---

# 🏠 Occupancy Detection – ML Zoomcamp Midterm Project

This project predicts whether a room is **occupied** based on environmental sensor readings such as temperature, humidity, CO₂ levels, and light intensity.

The dataset has been fetched from UCI Machine Learning Repository and is automatically downloaded within the script.

It includes:

* A full **machine learning pipeline**
* A **FastAPI** deployment
* A **Dockerized API** for easy reproduction
* Clean modular code (`preprocess.py`, `predict.py`)
* Pickled model artifacts

---

Exploratory Data Analysis has been conducted in the notebook.ipynb file. Additionally, it contains:

* Data Cleaning
* Model Selection and Hyperparameter Tuning
* Feature Importance


---
## Instructions:

## 1. Clone the repository:
```
git clone <repository-url>
cd <repository-directory>
```

## 2. Create the virtual environment using conda along with the necessary libraries (if done locally)

```bash
conda env create -f environment.yaml
conda activate midproject
```
```bash
pip install -r requirements.txt
```

---

## 3. Train the Model (Locally)

```bash
python train.py
```

This will generate:

* `model.pkl`
* `scaler.pkl`
* `features.pkl`

---

## 4. Run the Prediction Service

```bash
python predict.py
```
---

## 5. Run the FastAPI App Locally

```bash
uvicorn main:app --reload
```

Open the browser:

* API root: **[http://127.0.0.1:8000](http://127.0.0.1:8000)**
* Docs UI: **[http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)**

Use the POST `/predict` endpoint.

---

## 6. Interact with the service: 

Run the curl.py script in a separate terminal to send a POST request to the prediction service

```bash
python curl.py
```

## 6. Build & Run the Docker Image

### **Build the image**

```bash
docker build -t occupancy-api .
```

###  **Run the container**

```bash
docker run -p 8000:8000 occupancy-api
```

API will be available at:

```
http://127.0.0.1:8000
http://127.0.0.1:8000/docs
```

---

### 7. Cloud Deployment

I have used railway.app (http://railway.app) to host the docker file. 

It lives on at (ml-zoomcamp-midproject-production.up.railway.app).


![screenshot](D:\ml-zoomcamp-midproject\screenshots\Screenshot 2025-11-20 015449.png)

