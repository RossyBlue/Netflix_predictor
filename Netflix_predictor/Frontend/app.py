import streamlit as st
import requests
import pandas as pd

# ──────────────────────────────────────────────
# Configuración de página
# ──────────────────────────────────────────────
st.set_page_config(
    page_title="Netflix AI Finder",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded",
)

BACKEND_URL = "http://127.0.0.1:8000"

# ──────────────────────────────────────────────
# Utilidades
# ──────────────────────────────────────────────
def tipo_icono(tipo: str | None) -> str:
    mapping = {"Movie": "🎥", "TV Show": "📺", "Show": "📺"}
    return mapping.get(tipo, "⭐")


def score_color(score: float, max_score: float) -> str:
    """Retorna color de barra según proporción del score."""
    pct = score / max_score if max_score > 0 else 0
    if pct >= 0.66:
        return "#2ecc71"   # verde
    elif pct >= 0.33:
        return "#f39c12"   # naranja
    return "#e74c3c"       # rojo


def barra_score(score: float, max_score: float) -> str:
    """HTML de barra de progreso para el score."""
    pct = min(score / max_score * 100, 100) if max_score > 0 else 0
    color = score_color(score, max_score)
    return (
        f'<div style="background:#2c2c2c;border-radius:6px;height:10px;width:100%;">'
        f'<div style="background:{color};width:{pct:.1f}%;height:10px;border-radius:6px;'
        f'transition:width 0.4s ease;"></div></div>'
    )


@st.cache_data(ttl=300, show_spinner=False)
def obtener_generos_api() -> list[str]:
    """Obtiene géneros desde el backend con caché de 5 minutos."""
    try:
        r = requests.get(f"{BACKEND_URL}/genres", timeout=5)
        r.raise_for_status()
        return r.json().get("genres", [])
    except Exception:
        # Fallback con géneros conocidos si el backend no responde
        return [
            "Action", "Adventure", "Animation", "Comedy", "Crime",
            "Documentary", "Drama", "Family", "Fantasy", "Horror",
            "Romance", "Science Fiction", "Thriller",
        ]


@st.cache_data(ttl=60, show_spinner=False)
def verificar_backend() -> dict:
    """Verifica el estado del backend."""
    try:
        r = requests.get(f"{BACKEND_URL}/health", timeout=4)
        r.raise_for_status()
        return r.json()
    except Exception:
        return None


# ──────────────────────────────────────────────
# Header
# ──────────────────────────────────────────────
col_logo, col_titulo = st.columns([1, 8])
with col_logo:
    st.markdown("## 🎬")
with col_titulo:
    st.title("Netflix AI Finder")
    st.caption("Motor de recomendación basado en **XGBoost** entrenado sobre metadatos del catálogo Netflix.")

st.divider()

# ──────────────────────────────────────────────
# Estado del backend (indicador superior)
# ──────────────────────────────────────────────
health = verificar_backend()
if health:
    st.sidebar.success(
        f"✅ Backend activo  \n"
        f"📦 {health.get('dataset_rows', '?'):,} registros  \n"
        f"🧠 {health.get('feature_count', '?')} features"
    )
else:
    st.sidebar.error(
        "❌ Backend no disponible.  \n"
        "Ejecuta: `uvicorn main:app --reload`"
    )

# ──────────────────────────────────────────────
# Barra lateral — Filtros
# ──────────────────────────────────────────────
st.sidebar.header("🔍 Filtros de búsqueda")

generos_disponibles = obtener_generos_api()
seleccion_gen = st.sidebar.multiselect(
    "Géneros",
    generos_disponibles,
    default=["Drama"],
    help="Selecciona uno o más géneros para filtrar el catálogo.",
)

rango_anio = st.sidebar.slider(
    "Rango de años",
    min_value=1990,
    max_value=2025,
    value=(2015, 2025),
    step=1,
)

min_rating = st.sidebar.slider(
    "Rating mínimo",
    min_value=0.0,
    max_value=10.0,
    value=5.0,
    step=0.5,
)

top_n = st.sidebar.select_slider(
    "Resultados a mostrar",
    options=[5, 10, 15, 20],
    value=10,
)

st.sidebar.divider()
st.sidebar.markdown(
    "**Acerca del modelo**  \n"
    "XGBoost entrenado sobre metadatos estructurados del catálogo Netflix.  \n"
    "Variables: género, país, año, rating y tipo de contenido.  \n\n"
    "📊 Métricas (test set):  \n"
    "- MAE: 32.24  \n"
    "- RMSE: 112.13  \n"
    "- R²: 0.082"
)

# ──────────────────────────────────────────────
# Acción principal
# ──────────────────────────────────────────────
buscar = st.button("🚀 Encontrar mi Top de contenidos", use_container_width=True, type="primary")

if buscar:
    if not seleccion_gen:
        st.warning("⚠️ Selecciona al menos un género en la barra lateral.")
    else:
        with st.spinner("La IA está analizando el catálogo..."):
            try:
                payload = {
                    "generos": seleccion_gen,
                    "anio_min": rango_anio[0],
                    "anio_max": rango_anio[1],
                    "rating_min": min_rating,
                    "top_n": top_n,
                }
                res = requests.post(f"{BACKEND_URL}/recommend", json=payload, timeout=15)
                res.raise_for_status()
                datos = res.json()

            except requests.exceptions.ConnectionError:
                st.error(
                    "❌ **Error de conexión.** No se pudo contactar al backend.  \n"
                    "Asegúrate de tener activa la terminal con: `uvicorn main:app --reload`"
                )
                st.stop()
            except requests.exceptions.Timeout:
                st.error("⏳ **Tiempo de espera agotado.** El servidor tardó demasiado. Intenta acotar los filtros.")
                st.stop()
            except requests.exceptions.HTTPError:
                try:
                    detalle = res.json().get("detail", "Error en el procesamiento.")
                    st.warning(f"⚠️ {detalle}")
                except Exception:
                    st.error(f"❌ Error del servidor (HTTP {res.status_code})")
                st.stop()
            except Exception as e:
                st.error(f"🚨 Error inesperado: {e}")
                st.stop()

        # ──────────────────────────────────────────────
        # Resultados
        # ──────────────────────────────────────────────
        st.markdown(
            f"### 🏆 Top {len(datos)} recomendaciones "
            f"· {', '.join(seleccion_gen)} "
            f"· {rango_anio[0]}–{rango_anio[1]} "
            f"· Rating ≥ {min_rating}"
        )

        max_score = max((p["score_ia"] for p in datos), default=1)

        # Vista de tarjetas (2 columnas)
        cols = st.columns(2)
        for i, peli in enumerate(datos):
            with cols[i % 2]:
                icono = tipo_icono(peli.get("type"))
                tipo_label = peli.get("type") or "Contenido"
                score = peli["score_ia"]

                with st.container(border=True):
                    st.markdown(f"**{i + 1}. {icono} {peli['title']}**")
                    
                    c1, c2, c3 = st.columns(3)
                    c1.metric("Año", peli["release_year"])
                    c2.metric("Rating", f"{peli['rating']:.1f}")
                    c3.metric("Tipo", tipo_label)

                    st.markdown(
                        f"**Popularidad predicha:** `{score:.1f}`",
                        unsafe_allow_html=False,
                    )
                    st.markdown(
                        barra_score(score, max_score),
                        unsafe_allow_html=True,
                    )

        st.divider()

        # ──────────────────────────────────────────────
        # Tabla resumen descargable
        # ──────────────────────────────────────────────
        with st.expander("📋 Ver tabla completa y descargar"):
            df_resultado = pd.DataFrame(datos)
            df_resultado.index += 1
            df_resultado.columns = [c.replace("_", " ").title() for c in df_resultado.columns]
            st.dataframe(df_resultado, use_container_width=True)

            csv_bytes = df_resultado.to_csv(index=False).encode("utf-8")
            st.download_button(
                label="⬇️ Descargar como CSV",
                data=csv_bytes,
                file_name="recomendaciones_netflix.csv",
                mime="text/csv",
            )

      