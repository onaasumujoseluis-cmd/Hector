# domustela.py
# Professional Dashboard for Domustela Analytics
# Version: 4.0 - LO QUE EL CLIENTE REALMENTE NECESITA
# Author: JLON Data Solutions

import logging
import re
from typing import Optional, Dict, Any

import altair as alt
import pandas as pd
import streamlit as st
from sqlalchemy import create_engine, text

# Optional dependencies
try:
    from rapidfuzz import process, fuzz
    HAS_RAPIDFUZZ = True
except ImportError:
    HAS_RAPIDFUZZ = False

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Streamlit page configuration
st.set_page_config(page_title="Domustela - ROI por Anuncio", layout="wide")

# Constants
DEFAULT_MYSQL_PORT = 3306
CACHE_TTL_SECONDS = 300  # Aumentado para mejor performance
FUZZY_MATCH_THRESHOLD = 70

# Database Configuration
def get_database_config() -> Dict[str, Any]:
    try:
        config = {
            "host": st.secrets["MYSQL_HOST"],
            "port": int(st.secrets.get("MYSQL_PORT", DEFAULT_MYSQL_PORT)),
            "user": st.secrets["MYSQL_USER"],
            "password": st.secrets["MYSQL_PASSWORD"],
            "database": st.secrets["MYSQL_DB"]
        }
        return config
    except KeyError as e:
        st.error(f"Missing secret: {e}")
        st.stop()

def create_database_engine(config: Dict[str, Any]):
    engine_str = f"mysql+mysqlconnector://{config['user']}:{config['password']}@{config['host']}:{config['port']}/{config['database']}"
    try:
        return create_engine(engine_str, pool_pre_ping=True, pool_recycle=3600)
    except Exception as e:
        st.error(f"Database error: {e}")
        st.stop()

# Initialize
db_config = get_database_config()
engine = create_database_engine(db_config)

# Data Loading
@st.cache_data(ttl=CACHE_TTL_SECONDS)
def load_table_safe(query: str, params: Optional[Dict] = None) -> pd.DataFrame:
    try:
        return pd.read_sql(query, engine, params=params)
    except Exception as e:
        logger.error(f"Query failed: {e}")
        return pd.DataFrame()

# Helper functions
def calculate_roi_por_anuncio():
    """
    CALCULA LO MÁS IMPORTANTE: ROI POR ANUNCIO
    Combina gasto de Meta con ventas subidas
    """
    # Cargar datos de Meta
    df_meta = load_table_safe("""
        SELECT 
            campaign_name,
            SUM(spend_eur) as total_gastado,
            SUM(results) as total_leads,
            AVG(cpl) as cpl_promedio
        FROM meta_campaign_metrics 
        GROUP BY campaign_name
        HAVING total_gastado > 0
    """)
    
    # Cargar ventas
    df_ventas = load_table_safe("""
        SELECT nombre_anuncio, SUM(precio) as total_ventas, COUNT(*) as cantidad_ventas
        FROM ventas_domustela 
        WHERE estado = 'completada' AND precio > 0
        GROUP BY nombre_anuncio
    """)
    
    if df_meta.empty or df_ventas.empty:
        return pd.DataFrame()
    
    # Hacer matching inteligente
    resultados = []
    campaigns_list = df_meta["campaign_name"].astype(str).fillna("").tolist()
    
    for _, venta in df_ventas.iterrows():
        anuncio_venta = str(venta["nombre_anuncio"])
        
        # Buscar mejor coincidencia
        best_match = None
        best_score = 0
        
        for campaign in campaigns_list:
            if HAS_RAPIDFUZZ:
                score = fuzz.WRatio(anuncio_venta, campaign)
            else:
                anuncio_lower = anuncio_venta.lower()
                campaign_lower = campaign.lower()
                if anuncio_lower in campaign_lower or campaign_lower in anuncio_lower:
                    score = 80
                else:
                    score = 0
            
            if score > best_score and score >= FUZZY_MATCH_THRESHOLD:
                best_score = score
                best_match = campaign
        
        if best_match:
            meta_data = df_meta[df_meta["campaign_name"] == best_match]
            if not meta_data.empty:
                gastado = meta_data["total_gastado"].iloc[0]
                ventas_total = venta["total_ventas"]
                
                roi = ((ventas_total - gastado) / gastado * 100) if gastado > 0 else 0
                
                resultados.append({
                    "anuncio_closer": anuncio_venta,
                    "campaign_name": best_match,
                    "gastado_meta": gastado,
                    "ventas_generadas": ventas_total,
                    "cantidad_ventas": venta["cantidad_ventas"],
                    "roi_porcentaje": roi,
                    "match_score": best_score
                })
    
    return pd.DataFrame(resultados)

# Navigation - SIMPLIFICADO PARA EL CLIENTE
SECTION_OPTIONS = [
    "Dashboard ROI",
    "Meta Ads",
    "Landings",
    "Ventas & ROI",
    "Webinar"
]

section = st.sidebar.selectbox(
    "Navegación",
    SECTION_OPTIONS,
    index=0
)

# ---------------------------
# 1) DASHBOARD ROI - LO MÁS IMPORTANTE
# ---------------------------
if section == "Dashboard ROI":
    st.title("DASHBOARD ROI - ¿DÓNDE PONER MÁS DINERO?")
    
    st.subheader("ROI POR ANUNCIO (Lo que más importa)")
    
    df_roi = calculate_roi_por_anuncio()
    
    if not df_roi.empty:
        df_roi = df_roi.sort_values("roi_porcentaje", ascending=False)
        
        def color_roi(val):
            if val > 100:
                return 'background-color: #4CAF50; color: white; font-weight: bold;'
            elif val > 0:
                return 'background-color: #FFEB3B;'
            else:
                return 'background-color: #F44336; color: white;'
        
        total_gastado = df_roi["gastado_meta"].sum()
        total_ventas = df_roi["ventas_generadas"].sum()
        roi_total = ((total_ventas - total_gastado) / total_gastado * 100) if total_gastado > 0 else 0
        
        col1, col2, col3 = st.columns(3)
        col1.metric("Total Gastado", f"{total_gastado:,.0f} €")
        col2.metric("Total Ventas", f"{total_ventas:,.0f} €")
        col3.metric("ROI Total", f"{roi_total:.1f}%", 
                   delta="Positivo" if roi_total > 0 else "Negativo")
        
        st.dataframe(
            df_roi.style.applymap(color_roi, subset=['roi_porcentaje']).format({
                'gastado_meta': '{:,.0f} €',
                'ventas_generadas': '{:,.0f} €',
                'roi_porcentaje': '{:.1f}%',
                'match_score': '{:.0f}%'
            }),
            use_container_width=True
        )
        
        st.subheader("RECOMENDACIONES DE INVERSIÓN")
        
        top_anuncios = df_roi.head(3)
        if not top_anuncios.empty:
            st.success("ESCALAR ESTOS ANUNCIOS:")
            for _, row in top_anuncios.iterrows():
                if row['roi_porcentaje'] > 50:
                    st.markdown(f"""
                    **{row['campaign_name']}**
                    • Gastado: {row['gastado_meta']:,.0f}€
                    • Ventas: {row['ventas_generadas']:,.0f}€ ({row['cantidad_ventas']} ventas)
                    • ROI: {row['roi_porcentaje']:.1f}%
                    • ACCIÓN: Aumentar presupuesto en 30-50%
                    """)
        
        negativos = df_roi[df_roi['roi_porcentaje'] <= 0]
        if not negativos.empty:
            st.warning("REVISAR / PAUSAR ESTOS ANUNCIOS:")
            for _, row in negativos.iterrows():
                st.markdown(f"""
                **{row['campaign_name']}**
                • Gastado: {row['gastado_meta']:,.0f}€
                • Ventas: {row['ventas_generadas']:,.0f}€
                • ROI: {row['roi_porcentaje']:.1f}%
                • ACCIÓN: Reducir presupuesto o pausar
                """)
    else:
        st.info("""
        Para ver el ROI por anuncio:
        1. Los closers suben ventas en la pestaña "Ventas & ROI"
        2. El sistema conecta ventas con anuncios
        3. Aquí verás qué anuncios escalar
        """)

# ---------------------------
# 5) WEBINAR
# ---------------------------
elif section == "Webinar":
    st.title("Webinar - Registros")
    dfw = load_table_safe("SELECT * FROM webinar_registros ORDER BY fecha_registro DESC LIMIT 100")
    st.dataframe(dfw, use_container_width=True)

# Footer
st.sidebar.markdown("---")
st.sidebar.markdown("PARA EL CLIENTE:")
st.sidebar.markdown("""
Ver pestaña: "Dashboard ROI"

Ahí verás:
1. ROI por anuncio
2. Qué anuncios escalar
3. Qué anuncios pausar
""")
st.sidebar.caption("Domustela Dashboard v4.0 - ROI Focus")
