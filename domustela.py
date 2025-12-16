# domustela.py
# Professional Dashboard for Domustela Analytics
# Version: 3.0
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
                trend_text = f" **Tendencia alcista**: La inversión ha aumentado {trend_percentage:.1f}% desde {first_investment:.0f}€ hasta {last_investment:.0f}€"
            elif trend < 0:
                trend_text = f" **Tendencia bajista**: La inversión ha disminuido {abs(trend_percentage):.1f}% desde {first_investment:.0f}€ hasta {last_investment:.0f}€"
            else:
                trend_text = " **Tendencia estable**: La inversión se mantiene constante"
        else:
            trend_text = "**Datos insuficientes** para determinar tendencia"
        
        # Generate storytelling
        story = f"""
        **Storytelling de Inversión Meta Ads**
        
        **Resumen financiero:**
        • **Inversión total**: {total_investment:,.0f}€
        • **Promedio diario**: {avg_daily_investment:,.0f}€
        • **Rango de inversión**: {min_investment:,.0f}€ - {max_investment:,.0f}€ (variación de {investment_range:,.0f}€)
        
        **Análisis de tendencia:**
        {trend_text}
        
        **Recomendaciones:**
        """
        
        if investment_range > avg_daily_investment * 0.5:
            story += "•  **Alta volatilidad**: Considera estabilizar el presupuesto diario para mejor predictibilidad\n"
        else:
            story += "• **Baja volatilidad**: Estrategia de inversión estable y predecible\n"
            
        if avg_daily_investment < 1000:
            story += "•  **Oportunidad de escala**: Si el ROI es positivo, considera aumentar la inversión gradualmente\n"
        else:
            story += "•  **Monitoreo constante**: Asegúrate que el ROI justifique el nivel actual de inversión\n"
            
        if trend > avg_daily_investment * 0.2:
            story += "•  **Escalando agresivamente**: Valida que la eficiencia (CPL) se mantenga con el aumento\n"
        elif trend < -avg_daily_investment * 0.2:
            story += "•  **Reducción significativa**: Investiga si es por resultados pobres o estrategia deliberada\n"
        
        return story
        
    except Exception as e:
        logger.error(f"Error generating storytelling: {e}")
        return "Error generando storytelling. Revise los datos."

# Navigation - MODIFICADO
SECTION_OPTIONS = [
    "Dashboard General",
    "Meta Ads",
    "Landings Pages",  # Cambiado de "Google Analytics (GA4)"
    "Meta Ads - Anuncios",  # Cambiado de "Meta + GA4 (Funnel / Merge)"
    "Ventas",
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
# 2) META ADS - MODIFICADO CON GRÁFICO DE BARRAS Y STORYTELLING
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
        
        # Selector de tipo de gráfico
        col1, col2 = st.columns([2, 1])
        with col2:
            chart_type = st.selectbox(
                "Tipo de gráfico para Evolución Inversión",
                ["Línea", "Barras"],
                index=0
            )
        
        # KPIs
        col1, col2, col3 = st.columns(3)
        col1.metric("Inversión total", f"{df_sel['spend_eur'].sum():,.2f} €")
        col2.metric("CPL medio", f"{safe_mean(df_sel['cpl']):,.2f}" if not df_sel.empty else "-")
        col3.metric("CTR medio (%)", f"{safe_mean(df_sel['ctr_pct']):,.2f}" if not df_sel.empty else "-")
        
        # Inversión por día (selección)
        daily = df_sel.groupby("fecha_corte", as_index=False)["spend_eur"].sum().sort_values("fecha_corte")
        
        # Gráfico de evolución de inversión
        st.subheader("Evolución inversión")
        
        if not daily.empty:
            # Selección de tipo de gráfico
            if chart_type == "Línea":
                chart = alt.Chart(daily).mark_line(point=True).encode(
                    x="fecha_corte:T", 
                    y=alt.Y("spend_eur:Q", title="Inversión (€)"),
                    tooltip=["fecha_corte:T", "spend_eur:Q"]
                ).properties(height=320)
            else:  # Barras
                chart = alt.Chart(daily).mark_bar().encode(
                    x=alt.X("fecha_corte:T", title="Fecha"),
                    y=alt.Y("spend_eur:Q", title="Inversión (€)"),
                    tooltip=["fecha_corte:T", "spend_eur:Q"],
                    color=alt.value("#4CAF50")  # Color verde para barras
                ).properties(height=320)
            
            st.altair_chart(chart, use_container_width=True)
            
            # Storytelling de negocio
            st.subheader("Storytelling de Negocio")
            st.markdown(generate_storytelling_inversion(daily))
        else:
            st.info("No hay datos para el período seleccionado.")
        
        storytelling_block("KPIs calculados sobre la selección de campañas. CPL y CTR ayudan a decidir pausar/escalar.")
        
        # Tabla resumen
        st.subheader("Resumen por campaña")
        to_show = df_sel[["fecha_corte","campaign_name","spend_eur","impressions","clicks","results","ctr_pct","cpc","cpl"]].sort_values("spend_eur", ascending=False)
        st.dataframe(to_show, use_container_width=True)

# ---------------------------
# 3) LANDINGS PAGES (ANTES "GOOGLE ANALYTICS (GA4)") - TÍTULO MODIFICADO
# ---------------------------
elif section == "Landings Pages":
    st.title("Landings Pages — Conversiones")  # Título modificado
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

        st.subheader("Conversiones por landing")  # Título modificado
        st.dataframe(df_sel.sort_values(["landing_nombre","fecha"], ascending=[True, False]), use_container_width=True)

# ---------------------------
# 4) META ADS - ANUNCIOS (ANTES "META + GA4 (FUNNEL / MERGE)") - TÍTULO MODIFICADO
# ---------------------------
elif section == "Meta Ads - Anuncios":
    st.title("Meta Ads - Anuncios")  # Título modificado
    df_meta = load_table_safe("SELECT fecha_corte, campaign_name, spend_eur, clicks, results FROM meta_campaign_metrics")
    df_ga = load_table_safe("SELECT fecha, landing_nombre, sessions, leads FROM landings_performance_new")
    
    # Sección adicional: Proceso de ventas para closers
    st.markdown("---")
    st.subheader("Proceso de Ventas para Closers")
    
    proceso_ventas = """
    ** Objetivo:** Seguimiento de ventas sin depender de URLs/UTMs complejos
    
    **Flujo de trabajo actual:**
    1. **Lanzamiento anuncios** → Landing page → Agenda con equipo de ventas
    2. **Closer recibe lead** en Excel con datos básicos
    3. **Seguimiento manual:** Nombre, anuncio origen, estado de venta
    4. **Problema actual:** Dificultad para correlacionar anuncio → venta en tiempo real
    
    **Solución implementada en este dashboard:**
    
    **1. Subida de ventas simplificada:**
    - Los closers suben Excel con: fecha_compra, nombre_cliente, nombre_anuncio, landing, precio, estado
    - No necesitan saber URLs/UTMs, solo el nombre del anuncio que les dijeron
    
    **2. Match automático:**
    - El sistema hace "fuzzy matching" entre "nombre_anuncio" (ventas) y "campaign_name" (Meta Ads)
    - Ejemplo: "Escalado Final Dic" se matchea con "ESCALADO FINAL - DICIEMBRE 2025"
    
    **3. Visualización para decisiones:**
    - En esta sección puedes ver qué anuncios generan más ventas
    - Filtra por fecha para ver rendimiento en lanzamientos específicos
    
    **Para los closers:**
    - Solo necesitan guardar el **nombre del anuncio** que aparece en Meta
    - El sistema hace el resto automáticamente
    - Pueden ver en tiempo real qué anuncios están generando ventas
    """
    
    st.markdown(proceso_ventas)
    st.markdown("---")
    
    if df_meta.empty or df_ga.empty:
        st.info("Se requieren datos de Meta y GA4 para esta sección.")
    else:
        df_meta["fecha_corte"] = pd.to_datetime(df_meta["fecha_corte"])
        df_ga["fecha"] = pd.to_datetime(df_ga["fecha"])
        # agregar diario
        meta_daily = df_meta.groupby("fecha_corte", as_index=False).agg(spend_eur=("spend_eur","sum"), clicks=("clicks","sum"), leads_meta=("results","sum"))
        ga_daily = df_ga.groupby("fecha", as_index=False).agg(sessions=("sessions","sum"), leads_ga=("leads","sum"))
        funnel = pd.merge(meta_daily, ga_daily, left_on="fecha_corte", right_on="fecha", how="inner")
        if funnel.empty:
            st.warning("No hay intersección temporal entre Meta y GA4 (mismo rango de fechas).")
        else:
            funnel["ctr_pct"] = (funnel["clicks"] / funnel["spend_eur"].replace(0, pd.NA)).fillna(0) * 100
            st.subheader("Funnel diario combinado")
            st.dataframe(funnel.sort_values("fecha_corte", ascending=False), use_container_width=True)
            storytelling_block("El funnel muestra la relación inversión → clicks → sesiones → leads. Identifica caídas para investigar landings o campañas.")

        # Merge ventas vs campañas (si existen ventas)
        st.subheader("Merge ventas (si existen) — Preview")
        df_sales = load_table_safe("SELECT id, fecha_compra, nombre_anuncio, landing, precio FROM ventas_domustela")
        merged_preview = merge_sales_with_meta(df_sales, df_meta)
        if merged_preview.empty:
            st.info("No hay ventas para mostrar merge.")
        else:
            st.dataframe(merged_preview.head(200), use_container_width=True)
            storytelling_block("Merge tentativa entre ventas y campañas (fuzzy match si rapidfuzz disponible). Revísalo y valida para crear reglas de atribución.")

# ---------------------------
# 5) VENTAS (subir archivo, insertar)
# ---------------------------
elif section == "Ventas":
    st.title("Ventas — Subir / Gestionar")
    
    # Información para closers
    with st.expander(" Instrucciones para Closers", expanded=True):
        st.markdown("""
        **Para subir ventas NO necesitas:**
        - URLs complejas
        - Parámetros UTM
        - Códigos de seguimiento
        
        ** Solo necesitas guardar en tu Excel:**
        1. **nombre_anuncio**: El nombre como aparece en Meta (ej: "ESCALADO FINAL - DIC")
        2. **landing**: Número o nombre simple (ej: "landing1" o "1")
        3. **fecha_compra**: Fecha de la venta
        4. **nombre_cliente**: Nombre del cliente
        5. **precio**: Monto de la venta
        6. **estado**: "completada", "pendiente", etc.
        
        ** El sistema automáticamente:**
        - Busca coincidencias entre "nombre_anuncio" y las campañas de Meta
        - Asocia cada venta con su campaña correspondiente
        - Te permite ver qué anuncios generan más ventas
        """)
    
    st.subheader("Ventas actuales")
    df_sales = load_table_safe("SELECT id, fecha_compra, nombre_cliente, email, telefono, origen, nombre_anuncio, landing, precio, estado, notas FROM ventas_domustela")
    st.dataframe(df_sales, use_container_width=True)

    st.subheader("Subir Excel/CSV de ventas (opcional)")
    uploaded = st.file_uploader("Sube CSV/XLSX con columnas: fecha_compra,nombre_cliente,email,telefono,origen,nombre_anuncio,landing,precio,estado,notas", type=["csv","xlsx"])
    if uploaded:
        try:
            if uploaded.name.endswith(".xlsx"):
                df_new = pd.read_excel(uploaded)
            else:
                df_new = pd.read_csv(uploaded)
            st.write("Vista previa:")
            st.dataframe(df_new.head(8))
            
            # Mostrar match con campañas
            if "nombre_anuncio" in df_new.columns:
                st.write("**🔍 Matching con campañas Meta:**")
                df_meta = load_table_safe("SELECT DISTINCT campaign_name FROM meta_campaign_metrics")
                unique_ads = df_new["nombre_anuncio"].dropna().unique()
                
                for ad in unique_ads[:10]:  # Mostrar primeros 10
                    if HAS_RAPIDFUZZ and not df_meta.empty:
                        campaigns = df_meta["campaign_name"].astype(str).fillna("").unique().tolist()
                        best = process.extractOne(str(ad), campaigns, scorer=fuzz.WRatio)
                        if best and best[1] >= FUZZY_MATCH_THRESHOLD:
                            st.write(f"• `{ad}` → **{best[0]}** ({best[1]:.0f}% match)")
                        else:
                            st.write(f"• `{ad}` → *Sin match claro*")
                    else:
                        st.write(f"• `{ad}`")
            
            if st.button("Insertar (append) en ventas_domustela"):
                try:
                    # normalizar nombres columnas
                    df_new.columns = [c.strip() for c in df_new.columns]
                    if "fecha_compra" in df_new.columns:
                        df_new["fecha_compra"] = pd.to_datetime(df_new["fecha_compra"], errors="coerce")
                    df_new.to_sql("ventas_domustela", engine, if_exists="append", index=False)
                    st.success(" Ventas subidas correctamente.")
                    st.balloons()
                except Exception as e:
                    st.error(f"❌ Error al insertar ventas: {e}")
        except Exception as e:
            st.error(f"❌ Error leyendo archivo: {e}")

# ---------------------------
# 6) WEBINAR (subida y tabla)
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
                    st.success(" Registros insertados.")
                except Exception as e:
                    st.error(f" Error al insertar registros webinar: {e}")
        except Exception as e:
            st.error(f" Error leyendo CSV: {e}")

# Footer
st.sidebar.markdown("---")
st.sidebar.markdown("###  Flujo de Trabajo para Closers")
st.sidebar.markdown("""
**Simplificado:**
1. **Recibe lead** → Agenda en calendario
2. **Cierra venta** → Anota nombre_anuncio
3. **Sube Excel** → Solo datos básicos
4. **Mira dashboard** → Ve qué anuncios venden

**No necesita:** URLs, UTMs, códigos
""")
st.sidebar.caption("JLON Data Solutions — Domustela Dashboard v3.0")
st.sidebar.caption("Built with Streamlit & Professional Analytics")
