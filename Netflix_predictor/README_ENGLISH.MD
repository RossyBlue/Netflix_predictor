# Netflix Content Popularity Predictor 🎬📊

An end-to-end, decoupled Machine Learning service that predicts Netflix content popularity from structured metadata and delivers real-time recommendations via a REST API and interactive interface.

![Python](https://img.shields.io/badge/Python-3.10+-blue)
![FastAPI](https://img.shields.io/badge/FastAPI-2.0-009688)
![Streamlit](https://img.shields.io/badge/Streamlit-Interactive-FF4B4B)
![XGBoost](https://img.shields.io/badge/XGBoost-ML-green)

---

## 🎯 System Overview

Designed with a decoupled microservice-style architecture, this system processes **~32,000 tabular records** of movies and TV shows to forecast popularity scores and enable data-driven catalog prioritization.

`[ User UI ] <---> (Streamlit Frontend) <--- JSON / REST API ---> (FastAPI Backend) <---> [ XGBoost Model ]`

### Key Engineering Features
* **Decoupled Architecture:** Clean separation of concerns between API inference engine and UI presentation layer.
* **Vectorized Inference:** High-performance prediction pipeline optimized using NumPy and Pandas.
* **Model Interpretability:** Integrated **SHAP (SHapley Additive Explanations)** analysis to identify top prediction features (Main Genre, Country, Content Type).
* **Production-Ready API:** Includes `/health` monitoring checks, dynamic genre querying (`/genres`), structured logging, and OpenAPI 3.1 schema.
* **Caching & UX:** Client-side query caching, real-time interactive filters, and CSV export capabilities.

---

## 🛠️ Tech Stack

* **Backend & API:** FastAPI, Uvicorn, REST Architecture
* **Machine Learning & Data:** XGBoost, Scikit-learn, SHAP, Pandas, NumPy, Joblib
* **Frontend:** Streamlit
* **Development & Tooling:** Python 3, Git, Virtual Environments

---

## 🧠 Machine Learning Approach

* **Dataset:** Cleaned Netflix Movies & TV Shows dataset (~32k records).
* **Target Variable:** Continuous `popularity` score.
* **Selected Algorithm:** **XGBoost Regressor**, chosen after benchmarking against baseline models (Linear Regression, Random Forest, CatBoost, and Deep MLPs) due to its superior MAE/RMSE performance and rapid inference speed on tabular datasets.
* **Top Explanatory Features (SHAP):** Primary Genre, Country of Origin, Rating, Content Type, and Release Year.

---

## 📡 REST API Reference

### Core Endpoints

| Method | Endpoint | Description |
| :--- | :--- | :--- |
| `GET` | `/health` | Service, model, and dataset health check |
| `GET` | `/genres` | Dynamic list of available genres in catalog |
| `POST` | `/recommend` | Generates top-N recommendations based on predicted popularity |

#### Request Payload (`POST /recommend`)
```json
{
  "generos": ["Drama", "Romance"],
  "anio_min": 2015,
  "anio_max": 2023,
  "rating_min": 7.0,
  "top_n": 10
}

Example

[
  {
    "title": "Example Content Title",
    "release_year": 2021,
    "rating": 7.8,
    "score_ia": 145.32,
    "type": "Movie"
  }
]

📂 Repository Structure
├── Backend/             # FastAPI application & dataset
│   ├── main.py
│   └── netflix_cleaned.csv
├── Frontend/            # Streamlit dashboard
│   └── app.py
├── models/              # Serialized XGBoost model (.pkl)
├── notebooks/           # Exploratory Data Analysis & Model Benchmarking
├── data/                # Raw datasets
├── requirements.txt
└── README_English/Spanish.md

🚀 Local Setup & Run
git clone <YOUR_REPOSITORY_LINK>
cd <REPOSITORY_NAME>

python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt


//Start Backend API

uvicorn Backend.main:app --reload
API base URL: http://127.0.0.1:8000

Interactive API Docs (Swagger UI): http://127.0.0.1:8000/docs

//Start Frontend Dashboard

In a separate terminal session:
streamlit run Frontend/app.py
Access dashboard at http://localhost:8501

🔮 Roadmap & Next Steps
[ ] Containerize services using Docker & Docker Compose for streamlined orchestration.

[ ] Implement CI/CD pipeline using GitHub Actions for automated unit testing.

[ ] Integrate Natural Language Processing (NLP) on movie synopses to enhance feature space.

[ ] Deploy backend to cloud infrastructure (AWS / GCP / Railway).

