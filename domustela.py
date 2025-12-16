# domustela.py
# Professional Dashboard for Domustela Analytics
# Version: 3.1 - CORREGIDO
# Author: JLON Data Solutions
# Description: Streamlit-based dashboard for Meta Ads, GA4, Sales, and Webinar data analysis

import logging
import os
import re
from datetime import datetime, timedelta
from typing import Optional, List, Dict, Any

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

# Streamlit page configuration must be the first Streamlit command
st.set_page_config(page_title="Domustela Dashboard", layout="wide", page_icon="📊")

# Constants
DEFAULT_MYSQL_PORT = 3306
CACHE_TTL_SECONDS = 60
FUZZY_MATCH_THRESHOLD = 70

# Database Configuration
# Reads credentials from Streamlit secrets (recommended for security)
def get_database_config() -> Dict[str, Any]:
    """
    Retrieve database configuration from Streamlit secrets.

    Returns:
        Dict containing database connection parameters.

    Raises:
        KeyError: If required secrets are missing.
    """
    try:
        config = {
            "host": st.secrets["MYSQL_HOST"],
            "port": int(st.secrets.get("MYSQL_PORT", DEFAULT_MYSQL_PORT)),
            "user": st.secrets["MYSQL_USER"],
            "password": st.secrets["MYSQL_PASSWORD"],
            "database": st.secrets["MYSQL_DB"]
        }
        logger.info("Database configuration loaded successfully.")
        return config
    except KeyError as e:
        error_msg = f"Missing required secret: {e}. Please ensure MYSQL_HOST, MYSQL_USER, MYSQL_PASSWORD, MYSQL_DB are set in .streamlit/secrets.toml."
        logger.error(error_msg)
        st.error(error_msg)
        st.stop()

def create_database_engine(config: Dict[str, Any]):
    """
    Create SQLAlchemy engine with connection pooling.

    Args:
        config: Database configuration dictionary.

    Returns:
        SQLAlchemy engine instance.

    Raises:
        Exception: If engine creation fails.
    """
    engine_str = f"mysql+mysqlconnector://{config['user']}:{config['password']}@{config['host']}:{config['port']}/{config['database']}"
    try:
        engine = create_engine(engine_str, pool_pre_ping=True, pool_recycle=3600)
        logger.info("Database engine created successfully.")
        return engine
    except Exception as e:
        error_msg = f"Failed to create database engine: {e}"
        logger.error(error_msg)
        st.error(error_msg)
        st.stop()

# Initialize database connection
db_config = get_database_config()
engine = create_database_engine(db_config)

# Database Schema Management
def ensure_indexes_and_tables() -> bool:
    """
    Ensure required database indexes and tables exist.
    Creates indexes and webinar table if they don't exist.

    Returns:
        bool: True if successful, False otherwise.
    """
    index_queries = [
        "CREATE INDEX IF NOT EXISTS idx_meta_fecha ON meta_campaign_metrics (fecha_corte(10))",
        # Note: MySQL syntax for index on varchar/text columns
    ]

    webinar_table_query = """
    CREATE TABLE IF NOT EXISTS webinar_registros (
        id INT AUTO_INCREMENT PRIMARY KEY,
        fecha_registro DATETIME,
        nombre_cliente VARCHAR(255),
        email VARCHAR(255),
        telefono VARCHAR(50),
        asistio TINYINT(1),
        hora_entrada DATETIME,
        hora_salida DATETIME,
        duracion_minutos INT,
        notas TEXT,
        fecha_insercion TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;
    """

    try:
        with engine.begin() as conn:
            # Create webinar table
            try:
                conn.execute(text(webinar_table_query))
                logger.info("Webinar table ensured.")
            except Exception as e:
                logger.warning(f"Failed to create webinar table: {e}")

            # Create indexes
            for query in index_queries:
                try:
                    conn.execute(text(query))
                except Exception as e:
                    logger.warning(f"Failed to create index: {e}")
        return True
    except Exception as e:
        logger.error(f"Database schema setup failed: {e}")
        return False

# Initialize database schema
schema_initialized = ensure_indexes_and_tables()
if not schema_initialized:
    st.warning("Some database schema elements could not be initialized. Functionality may be limited.")

# Data Loading Utilities
@st.cache_data(ttl=CACHE_TTL_SECONDS)
def load_table_safe(query: str, params: Optional[Dict] = None) -> pd.DataFrame:
    """
    Safely load data from database with caching.

    Args:
        query: SQL query string.
        params: Optional query parameters.

    Returns:
        DataFrame with query results, or empty DataFrame on error.
    """
    try:
        df = pd.read_sql(query, engine, params=params)
        logger.info(f"Query executed successfully, returned {len(df)} rows.")
        return df
    except Exception as e:
        logger.error(f"Query failed: {query[:100]}... Error: {e}")
        return pd.DataFrame()

# Helpers
# Data Processing Helpers
def safe_mean(series: pd.Series) -> Optional[float]:
    """
    Calculate mean safely, handling non-numeric values.

    Args:
        series: Pandas series to calculate mean for.

    Returns:
        Mean value or None if calculation fails.
    """
    try:
        s = pd.to_numeric(series, errors="coerce")
        s = s.replace([pd.NA, float("inf"), float("-inf")], pd.NA).dropna()
        return s.mean() if not s.empty else None
    except Exception as e:
        logger.warning(f"Failed to calculate safe mean: {e}")
        return None

def normalize_landing(raw: Any) -> Optional[str]:
    """
    Normalize landing page names for consistent matching.

    Args:
        raw: Raw landing page value.

    Returns:
        Normalized landing key or None.
    """
    if pd.isna(raw):
        return None
    s = str(raw).lower().strip()
    match = re.search(r"landing[:\s/_-]*(\d+)", s)
    if match:
        return f"landing{match.group(1)}"
    for token in s.replace("/", " ").split():
        if token.startswith("l") and token[1:].isdigit():
            return f"landing{token[1:]}"
        if token.isdigit() and len(token) <= 4:
            return f"landing{token}"
    return s.replace(" ", "_")[:80]

def merge_sales_with_meta(df_sales: pd.DataFrame, df_meta: pd.DataFrame) -> pd.DataFrame:
    """
    Merge sales data with Meta campaign data using fuzzy matching.

    Args:
        df_sales: Sales DataFrame.
        df_meta: Meta campaigns DataFrame.

    Returns:
        Merged DataFrame with campaign matches.
    """
    if df_sales.empty or df_meta.empty:
        return df_sales.assign(campaign_name=None) if not df_sales.empty else pd.DataFrame()

    campaigns = df_meta["campaign_name"].astype(str).fillna("").unique().tolist()

    def map_campaign(name: Any) -> Optional[str]:
        if pd.isna(name):
            return None
        s = str(name)
        if HAS_RAPIDFUZZ:
            best = process.extractOne(s, campaigns, scorer=fuzz.WRatio)
            if best and best[1] >= FUZZY_MATCH_THRESHOLD:
                return best[0]
        else:
            for c in campaigns:
                if str(s).lower() in str(c).lower() or str(c).lower() in str(s).lower():
                    return c
        return None

    df = df_sales.copy()
    if "nombre_anuncio" in df.columns:
        df["campaign_name"] = df["nombre_anuncio"].apply(map_campaign)
    else:
        df["campaign_name"] = None
    if "landing" in df.columns:
        df["landing_key"] = df["landing"].apply(normalize_landing)
    else:
        df["landing_key"] = None
    return df

def storytelling_block(text: str) -> None:
    """
    Display a styled text block for storytelling/insights.

    Args:
        text: Text to display.
    """
    st.markdown(f"<div style='color:#333;font-size:14px;padding:4px 0'>{text}</div>", unsafe_allow_html=True)

def generate_storytelling_inversion(df: pd.DataFrame) -> str:
    """
    Generate storytelling based on investment evolution data.
    
    Args:
        df: DataFrame with fecha_corte and spend_eur columns
    
    Returns:
        String with storytelling analysis
    """
    if df.empty:
        return "No hay datos suficientes para generar análisis."
    
    try:
        # Calculate metrics
        total_investment = df["spend_eur"].sum()
        avg_daily_investment = df["spend_eur"].mean()
        max_investment = df["spend_eur"].max()
        min_investment = df["spend_eur"].min()
        investment_range = max_investment - min_investment
        
        # Calculate trends
        if len(df) > 1:
            df_sorted = df.sort_values("fecha_corte")
            last_investment = df_sorted["spend_eur"].iloc[-1]
            first_investment = df_sorted["spend_eur"].iloc[0]
            trend = last_investment - first_investment
            trend_percentage = (trend / first_investment * 100) if first_investment > 0 else 0
            
            if trend > 0:
                trend_text = f"📈 **Tendencia alcista**: La inversión ha aumentado {trend_percentage:.1f}% desde {first_investment:.0f}€ hasta {last_investment:.0f}€"
            elif trend < 0:
                trend_text = f"📉 **Tendencia bajista**: La inversión ha disminuido {abs(trend_percentage):.1f}% desde {first_investment:.0f}€ hasta {last_investment:.0f}€"
            else:
                trend_text = "➡️ **Tendencia estable**: La inversión se mantiene constante"
        else:
            trend_text = "📊 **Datos insuficientes** para determinar tendencia"
        
        # Generate storytelling
        story = f"""
        **📊 Storytelling de Inversión Meta Ads**
        
        **Resumen financiero:**
        • **Inversión total**: {total_investment:,.0f}€
        • **Promedio diario**: {avg_daily_investment:,.0f}€
        • **Rango de inversión**: {min_investment:,.0f}€ - {max_investment:,.0f}€ (variación de {investment_range:,.0f}€)
        
        **Análisis de tendencia:**
        {trend_text}
        
        **Recomendaciones:**
        """
        
        if investment_range > avg_daily_investment * 0.5:
            story += "• ⚠️ **Alta volatilidad**: Considera estabilizar el presupuesto diario para mejor predictibilidad\n"
        else:
            story += "• ✅ **Baja volatilidad**: Estrategia de inversión estable y predecible\n"
            
        if avg_daily_investment < 1000:
            story += "• 💡 **Oportunidad de escala**: Si el ROI es positivo, considera aumentar la inversión gradualmente\n"
        else:
            story += "• 🔍 **Monitoreo constante**: Asegúrate que el ROI justifique el nivel actual de inversión\n"
            
        if trend > avg_daily_investment * 0.2:
            story += "• 🚀 **Escalando agresivamente**: Valida que la eficiencia (CPL) se mantenga con el aumento\n"
        elif trend < -avg_daily_investment * 0.2:
            story += "• ⚠️ **Reducción significativa**: Investiga si es por resultados pobres o estrategia deliberada\n"
        
        return story
        
    except Exception as e:
        logger.error(f"Error generating storytelling: {e}")
        return "Error generando storytelling. Revise los datos."

# Navigation - MODIFICADO (quitamos la sección duplicada)
SECTION_OPTIONS = [
    "Dashboard General",
    "Meta Ads",
    "Landings Pages",
    "Ventas",  # AQUÍ está TODO el proceso de closers
    "Webinar"
]

section = st.sidebar.radio(
    "Navegación",
    SECTION_OPTIONS,
    index=0
)

# ---------------------------
# 1) DASHBOARD GENERAL
# ---------------------------
if section == "Dashboard General":
    st.title("Dashboard General — Domustela")
    df_meta = load_table_safe("SELECT fecha_corte, campaign_name, spend_eur, impressions, clicks, results, ctr_pct, cpc, cpl FROM meta_campaign_metrics")
    df_ga = load_table_safe("SELECT fecha, landing_nombre, sessions, leads, conv_pct FROM landings_performance_new")
    df_sales = load_table_safe("SELECT id, fecha_compra, precio FROM ventas_domustela")

    # KPIs
    total_spend = df_meta["spend_eur"].sum() if not df_meta.empty else 0
    total_leads_meta = int(df_meta["results"].sum()) if not df_meta.empty else 0
    total_sessions = int(df_ga["sessions"].sum()) if not df_ga.empty else 0
    total_revenue = df_sales["precio"].sum() if not df_sales.empty else 0

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Inversión (periodo)", f"{total_spend:,.2f} €")
    c2.metric("Leads (Meta)", f"{total_leads_meta:,}")
    c3.metric("Sesiones (GA4)", f"{total_sessions:,}")
    c4.metric("Ingresos (ventas)", f"{total_revenue:,.2f} €")

    storytelling_block(
        "Resumen ejecutivo: inversión vs leads vs sesiones vs ingresos. "
        "Utiliza estas métricas para ver rápidamente si el gasto está generando resultados (leads/ventas)."
    )

    # Mini gráficas: inversión diaria y top landings por conversión
    if not df_meta.empty:
        df_meta["fecha_corte"] = pd.to_datetime(df_meta["fecha_corte"])
        meta_daily = df_meta.groupby("fecha_corte", as_index=False)["spend_eur"].sum()
        st.subheader("Evolución inversión (Meta)")
        st.altair_chart(alt.Chart(meta_daily).mark_line(point=True).encode(
            x="fecha_corte:T", y="spend_eur:Q", tooltip=["fecha_corte:T", "spend_eur:Q"]
        ).properties(height=240), use_container_width=True)

    if not df_ga.empty:
        df_ga["fecha"] = pd.to_datetime(df_ga["fecha"])
        conv = df_ga.groupby("landing_nombre", as_index=False)["conv_pct"].mean().sort_values("conv_pct", ascending=False).head(6)
        st.subheader("Top landings por conversión (GA4)")
        st.altair_chart(alt.Chart(conv).mark_bar().encode(
            x=alt.X("landing_nombre:N", sort="-y"), y="conv_pct:Q", tooltip=["landing_nombre", "conv_pct"]
        ).properties(height=240), use_container_width=True)
        storytelling_block("Las landings mostradas son las que mejor convierten. Priorizar revisar copy/UX para las que están por debajo.")

# ---------------------------
# 2) META ADS - CON GRÁFICO DE BARRAS/LÍNEA CORREGIDO
# ---------------------------
elif section == "Meta Ads":
    st.title("Meta Ads")  # Título simplificado
    df = load_table_safe("SELECT fecha_corte, campaign_name, spend_eur, impressions, clicks, results, ctr_pct, cpc, cpl FROM meta_campaign_metrics")
    
    if df.empty:
        st.info("No hay datos de Meta Ads.")
    else:
        df["fecha_corte"] = pd.to_datetime(df["fecha_corte"])
        
        # Filtros
        campaigns = sorted(df["campaign_name"].dropna().unique().tolist())
        sel = st.multiselect("Selecciona campañas (vacío = todas)", campaigns, default=campaigns[:6])
        df_sel = df[df["campaign_name"].isin(sel)] if sel else df.copy()
        
        # KPIs
        col1, col2, col3 = st.columns(3)
        col1.metric("Inversión total", f"{df_sel['spend_eur'].sum():,.2f} €")
        col2.metric("CPL medio", f"{safe_mean(df_sel['cpl']):,.2f}" if not df_sel.empty else "-")
        col3.metric("CTR medio (%)", f"{safe_mean(df_sel['ctr_pct']):,.2f}" if not df_sel.empty else "-")
        
        # Inversión por día (selección)
        daily = df_sel.groupby("fecha_corte", as_index=False)["spend_eur"].sum().sort_values("fecha_corte")
        
        # Gráfico de evolución de inversión CON SELECTOR
        st.subheader("Evolución inversión")
        
        # Selector de tipo de gráfico
        chart_type = st.radio(
            "Selecciona tipo de gráfico:",
            ["Gráfico de Línea", "Gráfico de Barras"],
            horizontal=True,
            index=0
        )
        
        if not daily.empty:
            # Crear gráfico según selección
            if chart_type == "Gráfico de Línea":
                chart = alt.Chart(daily).mark_line(point=True).encode(
                    x=alt.X("fecha_corte:T", title="Fecha"),
                    y=alt.Y("spend_eur:Q", title="Inversión (€)"),
                    tooltip=["fecha_corte:T", "spend_eur:Q"]
                ).properties(height=350, title="Evolución de Inversión (Línea)")
            else:  # Gráfico de Barras
                chart = alt.Chart(daily).mark_bar(size=30).encode(
                    x=alt.X("fecha_corte:T", title="Fecha"),
                    y=alt.Y("spend_eur:Q", title="Inversión (€)"),
                    tooltip=["fecha_corte:T", "spend_eur:Q"],
                    color=alt.condition(
                        alt.datum.spend_eur == daily["spend_eur"].max(),
                        alt.value("#4CAF50"),  # Verde para el día con máxima inversión
                        alt.value("#2196F3")   # Azul para los demás
                    )
                ).properties(height=350, title="Evolución de Inversión (Barras)")
            
            st.altair_chart(chart, use_container_width=True)
            
            # Storytelling de negocio
            st.subheader("📈 Análisis y Recomendaciones")
            st.markdown(generate_storytelling_inversion(daily))
        else:
            st.info("No hay datos para el período seleccionado.")
        
        storytelling_block("KPIs calculados sobre la selección de campañas. CPL y CTR ayudan a decidir pausar/escalar.")
        
        # Tabla resumen
        st.subheader("Resumen por campaña")
        to_show = df_sel[["fecha_corte","campaign_name","spend_eur","impressions","clicks","results","ctr_pct","cpc","cpl"]].sort_values("spend_eur", ascending=False)
        st.dataframe(to_show, use_container_width=True)

# ---------------------------
# 3) LANDINGS PAGES (ANTES "GOOGLE ANALYTICS (GA4)")
# ---------------------------
elif section == "Landings Pages":
    st.title("Landings Pages — Conversiones")
    df = load_table_safe("SELECT fecha, landing_nombre, sessions, leads, conv_pct FROM landings_performance_new")
    if df.empty:
        st.info("No hay datos GA4.")
    else:
        df["fecha"] = pd.to_datetime(df["fecha"])
        # filtros landings
        lands = sorted(df["landing_nombre"].dropna().unique().tolist())
        sel_landings = st.multiselect("Filtrar landings", lands, default=lands[:6])
        df_sel = df[df["landing_nombre"].isin(sel_landings)] if sel_landings else df.copy()

        st.subheader("Conversion rate por landing (media)")
        conv = df_sel.groupby("landing_nombre", as_index=False)["conv_pct"].mean().sort_values("conv_pct", ascending=False)
        st.altair_chart(alt.Chart(conv).mark_bar().encode(
            x=alt.X("landing_nombre:N", sort="-y"),
            y="conv_pct:Q",
            tooltip=["landing_nombre","conv_pct"]
        ).properties(height=360), use_container_width=True)
        storytelling_block("Identifica las landings con mejor rendimiento y optimiza las que convienden menos.")

        st.subheader("Conversiones por landing")
        st.dataframe(df_sel.sort_values(["landing_nombre","fecha"], ascending=[True, False]), use_container_width=True)

# ---------------------------
# 4) VENTAS - TODO EL PROCESO DE CLOSERS AQUÍ
# ---------------------------
elif section == "Ventas":
    st.title("Ventas — Proceso de Closers")
    
    # Información para closers
    st.markdown("---")
    st.subheader("🎯 GUÍA RÁPIDA PARA CLOSERS")
    
    # Instrucciones visuales
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        ### ✅ **LO QUE SÍ NECESITAS:**
        
        1. **Preguntar al cliente:**
           > "¿De qué anuncio de Facebook/Instagram viniste?"
        
        2. **Anotar en Excel:**
           - `nombre_anuncio`: Lo que te diga el cliente
           - `fecha_compra`: Fecha de venta
           - `nombre_cliente`: Nombre del cliente
           - `precio`: Monto pagado
        
        3. **Ejemplos reales:**
           - "Del anuncio del curso gratis"
           - "De Instagram del webinar"
           - "Del que sale en Facebook"
        """)
    
    with col2:
        st.markdown("""
        ### ❌ **LO QUE NO NECESITAS:**
        
        - ❌ URLs complejas
        - ❌ Parámetros UTM
        - ❌ Códigos de seguimiento
        - ❌ Acceso a Ad Manager
        - ❌ Conocimiento técnico
        
        ### 📊 **EL SISTEMA HACE:**
        
        - 🔍 **Búsqueda inteligente** entre nombres
        - 🤝 **Conecta automáticamente** ventas con anuncios
        - 📈 **Muestra ROI por anuncio** en tiempo real
        """)
    
    st.markdown("---")
    
    # Visualización de match actual
    st.subheader("🔍 Match Actual entre Ventas y Anuncios")
    
    # Cargar datos de ventas y Meta
    df_sales = load_table_safe("SELECT id, fecha_compra, nombre_cliente, nombre_anuncio, landing, precio, estado FROM ventas_domustela")
    df_meta = load_table_safe("SELECT DISTINCT campaign_name FROM meta_campaign_metrics")
    
    if df_sales.empty:
        st.info("No hay ventas registradas aún.")
    else:
        # Mostrar tabla de ventas
        st.dataframe(df_sales, use_container_width=True)
        
        # Mostrar matching con campañas
        if not df_meta.empty and "nombre_anuncio" in df_sales.columns:
            st.subheader("📊 Matching Automático Detectado")
            
            # Filtrar ventas con nombre_anuncio
            ventas_con_nombre = df_sales[~df_sales["nombre_anuncio"].isna()]
            
            if not ventas_con_nombre.empty:
                for idx, venta in ventas_con_nombre.head(10).iterrows():
                    anuncio = str(venta["nombre_anuncio"])
                    if HAS_RAPIDFUZZ:
                        campaigns = df_meta["campaign_name"].astype(str).fillna("").unique().tolist()
                        best = process.extractOne(anuncio, campaigns, scorer=fuzz.WRatio)
                        if best and best[1] >= FUZZY_MATCH_THRESHOLD:
                            st.markdown(f"✅ **Venta de {venta['nombre_cliente']}**: `{anuncio}` → **{best[0]}** ({best[1]:.0f}% coincidencia)")
                        else:
                            st.markdown(f"⚠️ **Venta de {venta['nombre_cliente']}**: `{anuncio}` → *Sin match claro*")
                    else:
                        st.markdown(f"📝 **Venta de {venta['nombre_cliente']}**: `{anuncio}`")
    
    # Subida de ventas
    st.markdown("---")
    st.subheader("📤 Subir Excel/CSV de Ventas")
    
    uploaded = st.file_uploader(
        "Arrastra tu Excel aquí o haz clic para buscar",
        type=["csv", "xlsx"],
        help="Archivo debe tener columnas: fecha_compra, nombre_cliente, nombre_anuncio, precio"
    )
    
    if uploaded:
        try:
            # Leer archivo
            if uploaded.name.endswith(".xlsx"):
                df_new = pd.read_excel(uploaded)
            else:
                df_new = pd.read_csv(uploaded)
            
            st.success(f"✅ Archivo cargado: {uploaded.name} ({len(df_new)} registros)")
            
            # Mostrar vista previa
            st.write("**Vista previa (primeras 5 filas):**")
            st.dataframe(df_new.head())
            
            # Verificar columnas requeridas
            required_columns = ["fecha_compra", "nombre_cliente", "precio"]
            missing_columns = [col for col in required_columns if col not in df_new.columns]
            
            if missing_columns:
                st.warning(f"⚠️ Faltan columnas: {', '.join(missing_columns)}")
            else:
                st.info("✅ Columnas requeridas presentes")
            
            # Botón para insertar
            if st.button("🚀 Insertar Ventas en Base de Datos", type="primary"):
                try:
                    # Preparar datos
                    df_new.columns = [c.strip() for c in df_new.columns]
                    
                    # Convertir fecha
                    if "fecha_compra" in df_new.columns:
                        df_new["fecha_compra"] = pd.to_datetime(df_new["fecha_compra"], errors="coerce")
                    
                    # Insertar en base de datos
                    df_new.to_sql("ventas_domustela", engine, if_exists="append", index=False)
                    
                    st.success("✅ ¡Ventas insertadas correctamente!")
                    st.balloons()
                    
                    # Actualizar página
                    st.rerun()
                    
                except Exception as e:
                    st.error(f"❌ Error al insertar: {str(e)}")
                    
        except Exception as e:
            st.error(f"❌ Error al leer archivo: {str(e)}")
    
    # Plantilla de Excel para descargar
    st.markdown("---")
    st.subheader("📥 Plantilla de Excel")
    
    # Crear DataFrame de ejemplo
    template_data = {
        "fecha_compra": ["2024-01-15", "2024-01-15", "2024-01-16"],
        "nombre_cliente": ["Juan Pérez", "María López", "Carlos Ruiz"],
        "nombre_anuncio": ["ESCALADO FINAL - DIC", "anuncio webinar", "Instagram promo"],
        "landing": ["1", "landing2", "3"],
        "precio": [997, 997, 997],
        "estado": ["completada", "completada", "pendiente"],
        "notas": ["Cliente satisfecho", "Pago con tarjeta", "Seguimiento pendiente"]
    }
    
    template_df = pd.DataFrame(template_data)
    
    # Convertir a Excel
    template_excel = template_df.to_csv(index=False).encode('utf-8')
    
    st.download_button(
        label="📋 Descargar Plantilla de Excel",
        data=template_excel,
        file_name="plantilla_ventas_closers.csv",
        mime="text/csv",
        help="Usa esta plantilla para subir tus ventas"
    )

# ---------------------------
# 5) WEBINAR
# ---------------------------
elif section == "Webinar":
    st.title("Webinar — Registros")
    dfw = load_table_safe("SELECT id, fecha_registro, nombre_cliente, email, telefono, asistio, duracion_minutos, notas FROM webinar_registros ORDER BY fecha_registro DESC LIMIT 500")
    st.subheader("Últimos registros de webinar")
    st.dataframe(dfw, use_container_width=True)

    st.subheader("Subir CSV de registros")
    uploaded_w = st.file_uploader("Sube CSV con columnas: fecha_registro,nombre_cliente,email,telefono,asistio,hora_entrada,hora_salida,duracion_minutos,notas", type=["csv"], key="webinar_up")
    if uploaded_w:
        try:
            df_new_w = pd.read_csv(uploaded_w)
            st.dataframe(df_new_w.head(8))
            if st.button("Insertar registros webinar"):
                try:
                    # normalizar fecha si existe
                    if "fecha_registro" in df_new_w.columns:
                        df_new_w["fecha_registro"] = pd.to_datetime(df_new_w["fecha_registro"], errors="coerce")
                    df_new_w.to_sql("webinar_registros", engine, if_exists="append", index=False)
                    st.success("✅ Registros insertados.")
                except Exception as e:
                    st.error(f"❌ Error al insertar registros webinar: {e}")
        except Exception as e:
            st.error(f"❌ Error leyendo CSV: {e}")

# Footer
st.sidebar.markdown("---")
st.sidebar.markdown("### 📋 Para Closers:")
st.sidebar.markdown("""
1. **Pregunta:** ¿De qué anuncio viniste?
2. **Anota:** nombre_anuncio en Excel
3. **Sube:** Excel en "Ventas"
4. **Listo:** Sistema hace el resto
""")
st.sidebar.caption("JLON Data Solutions — Domustela Dashboard v3.1")
st.sidebar.caption("Built with Streamlit & Professional Analytics")
