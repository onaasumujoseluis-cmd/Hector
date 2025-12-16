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
st.set_page_config(page_title="Domustela - ROI por Anuncio", layout="wide", page_icon="💰")

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
                # Matching simple
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
            # Encontrar datos de Meta para esta campaña
            meta_data = df_meta[df_meta["campaign_name"] == best_match]
            if not meta_data.empty:
                gastado = meta_data["total_gastado"].iloc[0]
                ventas_total = venta["total_ventas"]
                
                # Calcular ROI
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
    " Dashboard ROI",
    " Meta Ads",
    " Landings",
    " Ventas & ROI",
    " Webinar"
]

section = st.sidebar.selectbox(
    "Navegación",
    SECTION_OPTIONS,
    index=0
)

# ---------------------------
# 1) DASHBOARD ROI - LO MÁS IMPORTANTE
# ---------------------------
if section == " Dashboard ROI":
    st.title(" DASHBOARD ROI - ¿DÓNDE PONER MÁS DINERO?")
    
    # ROI por anuncio
    st.subheader(" ROI POR ANUNCIO (Lo que más importa)")
    
    df_roi = calculate_roi_por_anuncio()
    
    if not df_roi.empty:
        # Ordenar por ROI
        df_roi = df_roi.sort_values("roi_porcentaje", ascending=False)
        
        # Mostrar con colores según ROI
        def color_roi(val):
            if val > 100:
                return 'background-color: #4CAF50; color: white; font-weight: bold;'
            elif val > 0:
                return 'background-color: #FFEB3B;'
            else:
                return 'background-color: #F44336; color: white;'
        
        # KPIs principales
        total_gastado = df_roi["gastado_meta"].sum()
        total_ventas = df_roi["ventas_generadas"].sum()
        roi_total = ((total_ventas - total_gastado) / total_gastado * 100) if total_gastado > 0 else 0
        
        col1, col2, col3 = st.columns(3)
        col1.metric(" Total Gastado", f"{total_gastado:,.0f} €")
        col2.metric("Total Ventas", f"{total_ventas:,.0f} €")
        col3.metric(" ROI Total", f"{roi_total:.1f}%", 
                   delta="Positivo" if roi_total > 0 else "Negativo")
        
        # Tabla de ROI
        st.dataframe(
            df_roi.style.applymap(color_roi, subset=['roi_porcentaje']).format({
                'gastado_meta': '{:,.0f} €',
                'ventas_generadas': '{:,.0f} €',
                'roi_porcentaje': '{:.1f}%',
                'match_score': '{:.0f}%'
            }),
            use_container_width=True
        )
        
        # Recomendaciones CLARAS
        st.subheader("🚀 RECOMENDACIONES DE INVERSIÓN")
        
        # Top 3 anuncios con mejor ROI
        top_anuncios = df_roi.head(3)
        if not top_anuncios.empty:
            st.success("** ESCALAR ESTOS ANUNCIOS:**")
            for idx, row in top_anuncios.iterrows():
                if row['roi_porcentaje'] > 50:
                    st.markdown(f"""
                    **{row['campaign_name']}**
                    • Gastado: {row['gastado_meta']:,.0f}€
                    • Ventas: {row['ventas_generadas']:,.0f}€ ({row['cantidad_ventas']} ventas)
                    • **ROI: {row['roi_porcentaje']:.1f}%** 🚀
                    • **ACCIÓN: Aumentar presupuesto en 30-50%**
                    """)
        
        # Anuncios con ROI negativo
        negativos = df_roi[df_roi['roi_porcentaje'] <= 0]
        if not negativos.empty:
            st.warning("** REVISAR/PAUSAR ESTOS ANUNCIOS:**")
            for idx, row in negativos.iterrows():
                st.markdown(f"""
                **{row['campaign_name']}**
                • Gastado: {row['gastado_meta']:,.0f}€
                • Ventas: {row['ventas_generadas']:,.0f}€
                • **ROI: {row['roi_porcentaje']:.1f}%** ❌
                • **ACCIÓN: Reducir presupuesto o pausar**
                """)
    else:
        st.info("""
        ** Para ver el ROI por anuncio:**
        
        1. **Los closers suben ventas** en la pestaña " Ventas & ROI"
        2. **El sistema automáticamente** conecta ventas con anuncios
        3. **Aquí verás** qué anuncios dan más ROI para escalar
        """)
    
    # Gráfico de ROI
    if not df_roi.empty:
        st.subheader(" ROI Visual por Anuncio")
        
        # Preparar datos para gráfico
        chart_data = df_roi.copy()
        chart_data = chart_data[chart_data['roi_porcentaje'].abs() < 1000]  # Filtrar outliers
        
        # Crear gráfico de barras
        bars = alt.Chart(chart_data).mark_bar().encode(
            x=alt.X('campaign_name:N', title='Anuncio', sort='-y'),
            y=alt.Y('roi_porcentaje:Q', title='ROI %'),
            color=alt.condition(
                alt.datum.roi_porcentaje > 0,
                alt.value('#4CAF50'),  # Verde para positivo
                alt.value('#F44336')   # Rojo para negativo
            ),
            tooltip=['campaign_name', 'gastado_meta', 'ventas_generadas', 'roi_porcentaje']
        ).properties(height=400)
        
        st.altair_chart(bars, use_container_width=True)

# ---------------------------
# 2) META ADS
# ---------------------------
elif section == " Meta Ads":
    st.title(" Meta Ads - Rendimiento")
    df = load_table_safe("SELECT fecha_corte, campaign_name, spend_eur, impressions, clicks, results, ctr_pct, cpc, cpl FROM meta_campaign_metrics")
    
    if df.empty:
        st.info("No hay datos de Meta Ads.")
    else:
        df["fecha_corte"] = pd.to_datetime(df["fecha_corte"])
        
        # Selector de tipo de gráfico
        st.subheader("📈 Evolución de Inversión")
        chart_type = st.radio(
            "Tipo de gráfico:",
            ["Línea", "Barras"],
            horizontal=True
        )
        
        # Agrupar por día
        daily = df.groupby("fecha_corte", as_index=False)["spend_eur"].sum()
        
        if chart_type == "Línea":
            chart = alt.Chart(daily).mark_line(point=True).encode(
                x="fecha_corte:T",
                y="spend_eur:Q",
                tooltip=["fecha_corte:T", "spend_eur:Q"]
            )
        else:
            chart = alt.Chart(daily).mark_bar().encode(
                x="fecha_corte:T",
                y="spend_eur:Q",
                tooltip=["fecha_corte:T", "spend_eur:Q"]
            )
        
        st.altair_chart(chart.properties(height=350), use_container_width=True)
        
        # Top campañas por gasto
        st.subheader(" Top Campañas por Inversión")
        top_campaigns = df.groupby("campaign_name")["spend_eur"].sum().sort_values(ascending=False).head(10)
        st.bar_chart(top_campaigns)

# ---------------------------
# 3) LANDINGS
# ---------------------------
elif section == " Landings":
    st.title(" Landings - Conversiones")
    df = load_table_safe("SELECT fecha, landing_nombre, sessions, leads, conv_pct FROM landings_performance_new")
    
    if not df.empty:
        df["fecha"] = pd.to_datetime(df["fecha"])
        
        # Conversion rate por landing
        conv = df.groupby("landing_nombre", as_index=False)["conv_pct"].mean().sort_values("conv_pct", ascending=False)
        
        st.subheader(" Top Landings por Conversión")
        st.altair_chart(alt.Chart(conv).mark_bar().encode(
            x=alt.X("landing_nombre:N", sort="-y"),
            y="conv_pct:Q"
        ).properties(height=350), use_container_width=True)

# ---------------------------
# 4) VENTAS & ROI - PROCESO COMPLETO
# ---------------------------
elif section == " Ventas & ROI":
    st.title(" Ventas & ROI - Proceso Closers")
    
    # Dos pestañas: Subir ventas y Ver conexiones
    tab1, tab2 = st.tabs(["📤 Subir Ventas", " Ver Conexiones"])
    
    with tab1:
        st.header(" Subir Ventas (Para Closers)")
        
        st.markdown("""
        
        
        # Subida de archivo
        uploaded = st.file_uploader("Sube Excel/CSV con ventas", type=["csv", "xlsx"])
        
        if uploaded:
            try:
                if uploaded.name.endswith(".xlsx"):
                    df_new = pd.read_excel(uploaded)
                else:
                    df_new = pd.read_csv(uploaded)
                
                st.success(f" Archivo cargado: {len(df_new)} ventas")
                st.dataframe(df_new.head())
                
                if st.button(" Insertar Ventas", type="primary"):
                    try:
                        df_new.columns = [c.strip() for c in df_new.columns]
                        if "fecha_compra" in df_new.columns:
                            df_new["fecha_compra"] = pd.to_datetime(df_new["fecha_compra"], errors="coerce")
                        
                        df_new.to_sql("ventas_domustela", engine, if_exists="append", index=False)
                        st.success(" Ventas insertadas")
                        st.balloons()
                    except Exception as e:
                        st.error(f"Error: {e}")
                        
            except Exception as e:
                st.error(f"Error leyendo archivo: {e}")
    
    with tab2:
        st.header(" Conexiones Ventas-Anuncios")
        
        # Mostrar matching actual
        df_ventas = load_table_safe("SELECT nombre_anuncio, SUM(precio) as total_ventas FROM ventas_domustela GROUP BY nombre_anuncio")
        df_meta = load_table_safe("SELECT campaign_name, SUM(spend_eur) as total_gastado FROM meta_campaign_metrics GROUP BY campaign_name")
        
        if not df_ventas.empty and not df_meta.empty:
            st.markdown("### 🔍 **Matching Automático Detectado:**")
            
            for _, venta in df_ventas.iterrows():
                anuncio = str(venta["nombre_anuncio"])
                mejor_match = None
                mejor_puntaje = 0
                
                for campaign in df_meta["campaign_name"]:
                    if HAS_RAPIDFUZZ:
                        score = fuzz.WRatio(anuncio, str(campaign))
                    else:
                        score = 80 if anuncio.lower() in str(campaign).lower() else 0
                    
                    if score > mejor_puntaje and score >= 70:
                        mejor_puntaje = score
                        mejor_match = campaign
                
                if mejor_match:
                    gastado = df_meta[df_meta["campaign_name"] == mejor_match]["total_gastado"].iloc[0]
                    ventas = venta["total_ventas"]
                    roi = ((ventas - gastado) / gastado * 100) if gastado > 0 else 0
                    
                    emoji = "" if roi > 0 else ""
                    st.markdown(f"""
                    {emoji} **{anuncio}** → **{mejor_match}**
                    • Gastado: {gastado:,.0f}€
                    • Ventas: {ventas:,.0f}€
                    • **ROI: {roi:.1f}%**
                    """)
                else:
                    st.markdown(f" **{anuncio}** → Sin match claro")

# ---------------------------
# 5) WEBINAR
# ---------------------------
elif section == "🎥 Webinar":
    st.title("🎥 Webinar - Registros")
    dfw = load_table_safe("SELECT * FROM webinar_registros ORDER BY fecha_registro DESC LIMIT 100")
    st.dataframe(dfw, use_container_width=True)

# Footer
st.sidebar.markdown("---")
st.sidebar.markdown("###  **PARA EL CLIENTE:**")
st.sidebar.markdown("""
**Ver pestaña: " Dashboard ROI"**

Ahí verás:
1. ** ROI por anuncio**
2. ** Qué anuncios escalar**
3. ** Qué anuncios pausar**
""")
st.sidebar.caption("Domustela Dashboard v4.0 - ROI Focus")

