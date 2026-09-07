# Netflix Popularity Predictor 🎬📊

Sistema de aprendizaje automático orientado a la predicción de popularidad de contenidos de Netflix utilizando técnicas de Machine Learning y una arquitectura desacoplada basada en microservicios.

El proyecto integra un modelo predictivo desarrollado con XGBoost, una API REST construida con FastAPI y una interfaz interactiva implementada con Streamlit.

---

# 📌 Objetivo del Proyecto

El objetivo principal del proyecto es predecir la popularidad de películas y series de Netflix a partir de metadatos estructurados, permitiendo apoyar decisiones relacionadas con:

- análisis de contenido,
- priorización de catálogo,
- recomendaciones,
- posicionamiento de contenido,
- análisis de patrones de popularidad.

El sistema busca transformar datos estructurados en información útil mediante modelos de aprendizaje automático supervisado.

---

# 🎯 Problema a Resolver

Las plataformas de streaming operan en un entorno altamente competitivo, donde comprender qué factores influyen en la popularidad de un contenido resulta fundamental para mejorar la experiencia del usuario y optimizar decisiones de negocio.

La popularidad de películas y series depende de múltiples factores, por lo que este proyecto busca modelar parcialmente este fenómeno utilizando variables estructuradas disponibles en datasets públicos.

---

# 📂 Dataset Utilizado

Se utilizó el dataset:

**Netflix Movies and TV Shows till 2025**

El dataset contiene información relacionada con:
- películas,
- series,
- géneros,
- países,
- ratings,
- popularidad,
- años de lanzamiento.

## Características principales

- ~32.000 registros
- Datos de películas y series
- Variables numéricas y categóricas
- Dataset integrado y preprocesado

## Variables utilizadas en el modelo

### Variables de entrada
- `release_year`
- `rating`
- `main_country`
- `main_genre`
- `type` (película o serie)

### Variable objetivo
- `popularity`

---

# 🧠 Modelos Evaluados

Durante el desarrollo se evaluaron múltiples técnicas de Machine Learning y Deep Learning:

| Modelo | Tipo |
|---|---|
| Regresión Lineal | Machine Learning |
| Random Forest | Machine Learning |
| XGBoost | Boosting |
| CatBoost | Boosting |
| MLP Base | Deep Learning |
| MLP Deep | Deep Learning |
| MLP Regularizada | Deep Learning |

---

# 📈 Métricas Utilizadas

El desempeño de los modelos fue evaluado mediante métricas de regresión:

| Métrica | Descripción |
|---|---|
| MAE | Error absoluto promedio |
| RMSE | Penaliza errores grandes |
| R² | Capacidad explicativa del modelo |

---

# 🏆 Modelo Seleccionado

## XGBoost Regressor

El modelo final seleccionado fue XGBoost debido a que presentó el mejor desempeño relativo durante las pruebas experimentales.

## Resultados principales

| Métrica | Resultado |
|---|---|
| MAE | 32.24 |
| RMSE | 112.13 |
| R² | 0.082 |

## Justificación de selección

XGBoost fue seleccionado porque:
- obtuvo el menor MAE,
- presentó mejor RMSE,
- mostró mejor capacidad de generalización,
- capturó mejor relaciones no lineales,
- tuvo mejor desempeño sobre datos tabulares.

---

# 🚀 Arquitectura del Proyecto

El sistema utiliza una arquitectura desacoplada compuesta por frontend y backend separados.

## 🔹 Backend
Desarrollado con FastAPI (versión 2.0.0).

Responsabilidades:
- cargar modelo entrenado,
- procesar solicitudes con inferencia vectorizada,
- ejecutar predicciones con XGBoost,
- registrar trazabilidad de consultas mediante logging,
- retornar resultados al frontend.

## 🔹 Frontend
Desarrollado con Streamlit.

Responsabilidades:
- interfaz gráfica de usuario,
- selección de filtros interactivos,
- visualización de resultados con score de popularidad,
- caché de peticiones para mejor rendimiento,
- descarga de resultados en CSV.

---

# 🛠️ Tecnologías Utilizadas

## Backend
- FastAPI
- Uvicorn
- Joblib
- Pandas
- NumPy
- Scikit-learn
- XGBoost

## Frontend
- Streamlit

## Machine Learning
- Scikit-learn
- XGBoost
- CatBoost
- TensorFlow / Keras
- SHAP

## Desarrollo
- Python 3
- Google Colab
- GitHub

---

# 📂 Estructura del Repositorio

```text
Acif104/
│
├── Backend/
│   ├── main.py
│   └── netflix_cleaned.csv
│
├── Frontend/
│   └── app.py
│
├── notebooks/
│   └── Semana_9_Sumativa_2_fase3.ipynb
│
├── data/
│   ├── netflix_movies_detailed_up_to_2025.csv
│   └── netflix_tv_shows_detailed_up_to_2025.csv
│
├── docs/
│   └── acif104_s9_BCastillo_SHerrera_RGarrote.pdf
│
├── models/
│   └── modelo_final_netflix-2.pkl
│
├── requirements.txt
├── README.md
└── .gitignore
```

---

# ⚙️ Instalación y Configuración

## 1️⃣ Clonar el repositorio

```bash
git clone <ENLACE_GITHUB>
cd Acif104
```

---

## 2️⃣ Crear entorno virtual

### Windows

```bash
python -m venv venv
venv\Scripts\activate
```

### Linux / Mac

```bash
python3 -m venv venv
source venv/bin/activate
```

---

## 3️⃣ Instalar dependencias

```bash
pip install -r requirements.txt
```

---

# ▶️ Ejecución del Proyecto

## 🔹 Ejecutar Backend (FastAPI)

```bash
uvicorn Backend.main:app --reload
```

Backend disponible en:

```
http://127.0.0.1:8000
```

Documentación Swagger UI:

```
http://127.0.0.1:8000/docs
```

---

## 🔹 Ejecutar Frontend (Streamlit)

En otra terminal:

```bash
streamlit run Frontend/app.py
```

Frontend disponible en:

```
http://localhost:8501
```

---

# 🔄 Flujo del Sistema

1. El usuario ingresa parámetros desde Streamlit.
2. El frontend envía la solicitud al backend.
3. FastAPI procesa la información y ejecuta inferencia vectorizada.
4. El modelo XGBoost genera la predicción de popularidad.
5. El backend registra la consulta mediante logging estructurado.
6. El backend retorna el resultado ordenado por score.
7. Streamlit muestra las recomendaciones con visualización interactiva.

---

# 📡 Endpoints de la API

| Método | Endpoint | Descripción |
|---|---|---|
| GET | `/health` | Estado del servicio, modelo y dataset |
| GET | `/genres` | Géneros disponibles en el catálogo |
| POST | `/recommend` | Recomendaciones personalizadas por popularidad predicha |

## Ejemplo JSON — POST /recommend

```json
{
  "generos": ["Drama", "Romance"],
  "anio_min": 2015,
  "anio_max": 2023,
  "rating_min": 7.0,
  "top_n": 10
}
```

## Ejemplo de respuesta

```json
[
  {
    "title": "Nombre del contenido",
    "release_year": 2021,
    "rating": 7.8,
    "score_ia": 145.32,
    "type": "Movie"
  }
]
```

---

# 🔍 Interpretabilidad del Modelo

Durante el desarrollo del proyecto se exploraron técnicas de interpretabilidad como SHAP (SHapley Additive Explanations) con el objetivo de analizar la influencia de las variables sobre las predicciones.

El análisis SHAP evidenció que las variables con mayor contribución en la estimación de popularidad fueron:
- género principal,
- país de origen,
- rating,
- tipo de contenido,
- año de lanzamiento.

Estas variables fueron posteriormente utilizadas como base para el sistema de filtrado y recomendación implementado en el backend.

---

# 📈 Características del Sistema

✅ Arquitectura desacoplada frontend/backend  
✅ API REST documentada con FastAPI + Swagger UI (OAS 3.1)  
✅ Interfaz interactiva con Streamlit  
✅ Modelo XGBoost integrado (32.000 registros)  
✅ Inferencia vectorizada con NumPy/Pandas  
✅ Logging estructurado con trazabilidad de consultas  
✅ Health check y monitoreo del servicio (`/health`)  
✅ Géneros dinámicos obtenidos desde el dataset (`/genres`)  
✅ Caché de peticiones en el frontend  
✅ Descarga de resultados en CSV  
✅ Sistema reproducible y escalable  
✅ Manejo avanzado de errores en frontend y backend  

---

# 🔮 Mejoras Futuras

- Incorporación de NLP sobre descripciones y sinopsis
- Optimización avanzada de hiperparámetros (Grid Search, Bayesian Optimization)
- Validación cruzada extendida
- Despliegue en la nube (AWS / GCP / Railway)
- Integración con bases de datos para persistencia de consultas
- Autenticación de usuarios
- Monitoreo continuo del modelo
- Dashboards explicativos con SHAP interactivo

---

# 📚 Referencias

- Scikit-learn Documentation
- XGBoost Documentation
- FastAPI Documentation
- Streamlit Documentation
- SHAP Documentation

---

