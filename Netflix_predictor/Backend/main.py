from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
import joblib
import pandas as pd
import os
import numpy as np
from pydantic import BaseModel, Field
from typing import List, Optional
import time
import logging

# ──────────────────────────────────────────────
# Configuración de logging para trazabilidad
# ──────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Netflix Popularity Predictor API",
    description="Predice popularidad de contenidos Netflix usando XGBoost sobre metadatos estructurados.",
    version="2.0.0",
)

# Permitir peticiones desde el frontend Streamlit (localhost)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

# ──────────────────────────────────────────────
# Carga de artefactos (modelo + dataset)
# ──────────────────────────────────────────────
BASE_PATH = os.path.dirname(__file__)
MODEL_PATH = os.path.join(BASE_PATH, "..", "models", "modelo_final_netflix-2.pkl")
CSV_PATH = os.path.join(BASE_PATH, "netflix_cleaned.csv")

try:
    model = joblib.load(MODEL_PATH)
    df = pd.read_csv(CSV_PATH, sep=";")
    logger.info(f"Artefactos cargados correctamente. Dataset: {len(df)} registros.")
except Exception as e:
    logger.critical(f"Error crítico al cargar artefactos: {e}")
    model = None
    df = None


# ──────────────────────────────────────────────
# Modelos Pydantic
# ──────────────────────────────────────────────
class FilterQuery(BaseModel):
    generos: List[str] = Field(..., min_items=1, description="Lista de géneros a filtrar")
    anio_min: int = Field(..., ge=1900, le=2030, description="Año mínimo de lanzamiento")
    anio_max: int = Field(..., ge=1900, le=2030, description="Año máximo de lanzamiento")
    rating_min: float = Field(..., ge=0.0, le=10.0, description="Rating mínimo (0-10)")
    top_n: Optional[int] = Field(default=10, ge=1, le=50, description="Cantidad de resultados a retornar")


class ContentItem(BaseModel):
    title: str
    release_year: int
    rating: float
    score_ia: float
    type: Optional[str] = None


# ──────────────────────────────────────────────
# Endpoints
# ──────────────────────────────────────────────
@app.get("/health", summary="Health check del servicio")
def health_check():
    """Verifica que el modelo y el dataset estén disponibles."""
    if model is None or df is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Modelo o dataset no disponibles.",
        )
    return {
        "status": "ok",
        "model_loaded": True,
        "dataset_rows": len(df),
        "feature_count": len(model.feature_names_in_),
    }


@app.get("/genres", summary="Géneros disponibles en el catálogo")
def get_genres():
    """Retorna la lista de géneros únicos disponibles en el dataset."""
    if df is None:
        raise HTTPException(status_code=503, detail="Dataset no disponible.")
    genres = sorted(df["genre_1"].dropna().unique().tolist())
    return {"genres": genres}


@app.post(
    "/recommend",
    response_model=List[ContentItem],
    summary="Recomendaciones personalizadas",
    description="Retorna el top N de contenidos predichos como más populares por el modelo XGBoost.",
)
def recommend(data: FilterQuery):
    """
    Filtra el catálogo según los parámetros recibidos y ejecuta inferencia con XGBoost
    para retornar los contenidos con mayor popularidad predicha.
    """
    if model is None or df is None:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="El modelo o la base de datos no están disponibles en el servidor.",
        )

    # Validación de rango de años
    if data.anio_min > data.anio_max:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="anio_min no puede ser mayor que anio_max.",
        )

    try:
        start_time = time.time()
        logger.info(
            f"Solicitud recibida | géneros={data.generos} | años={data.anio_min}-{data.anio_max} | rating>={data.rating_min}"
        )

        # 1. FILTRAR EL DATASET
        mask = (
            df["genre_1"].isin(data.generos)
            & (df["release_year"] >= data.anio_min)
            & (df["release_year"] <= data.anio_max)
            & (df["rating"] >= data.rating_min)
        )
        candidatos = df[mask].copy().reset_index(drop=True)

        if candidatos.empty:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=(
                    "No hay contenidos que cumplan con los filtros seleccionados. "
                    "Intenta reduciendo el rating mínimo o ampliando el rango de años."
                ),
            )

        logger.info(f"Candidatos encontrados: {len(candidatos)}")

        # 2. CONSTRUIR MATRIZ DE FEATURES (vectorizado)
        X = pd.DataFrame(
            0,
            index=np.arange(len(candidatos)),
            columns=model.feature_names_in_,
            dtype=np.float32,
        )

        # Variables numéricas (asignación directa — O(1))
        if "release_year" in X.columns:
            X["release_year"] = candidatos["release_year"].values
        if "rating" in X.columns:
            X["rating"] = candidatos["rating"].values

        # One-Hot Encoding vectorizado para género
        genre_dummies = (
            "main_genre_" + candidatos["genre_1"].fillna("Unknown")
        )
        valid_genre_cols = genre_dummies[genre_dummies.isin(X.columns)]
        for col in valid_genre_cols.unique():
            idx = valid_genre_cols[valid_genre_cols == col].index
            X.loc[idx, col] = 1

        # One-Hot Encoding vectorizado para país
        country_dummies = (
            "main_country_" + candidatos["country_1"].fillna("Unknown")
        )
        for col in country_dummies.unique():
            idx = country_dummies[country_dummies == col].index
            if col in X.columns:
                X.loc[idx, col] = 1
            else:
                # Categoría no vista → "Other" o "Unknown" como fallback
                fallback = next(
                    (c for c in ["main_country_Other", "main_country_Unknown"] if c in X.columns),
                    None,
                )
                if fallback:
                    X.loc[idx, fallback] = 1

        # One-Hot Encoding vectorizado para tipo de contenido
        if "type" in candidatos.columns:
            type_dummies = "type_" + candidatos["type"].fillna("Unknown")
            for col in type_dummies.unique():
                if col in X.columns:
                    idx = type_dummies[type_dummies == col].index
                    X.loc[idx, col] = 1

        # 3. INFERENCIA
        candidatos["score_ia"] = model.predict(X)

        elapsed = time.time() - start_time
        logger.info(
            f"Inferencia completada | candidatos={len(candidatos)} | tiempo={elapsed:.4f}s"
        )

        # 4. ORDENAR Y RETORNAR TOP N
        columnas_retorno = ["title", "release_year", "rating", "score_ia"]
        if "type" in candidatos.columns:
            columnas_retorno.append("type")

        top_n = (
            candidatos.sort_values(by="score_ia", ascending=False)
            .head(data.top_n)[columnas_retorno]
            .to_dict(orient="records")
        )

        return top_n

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error interno en /recommend: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error interno en el procesamiento de la predicción: {str(e)}",
        )