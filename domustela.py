# domustela.py
# Professional Dashboard for Domustela Analytics
# Version: 5.1 - Con Análisis de ROI por Anuncio (Plan B)
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
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Streamlit page configuration
st.set_page_config(page_title="Domustela Analytics", layout="wide")

# Constants
DEFAULT_MYSQL_PORT = 3306
CACHE_TTL_SECONDS = 300
FUZZY_MATCH_THRESHOLD = 70

# Database Configuration
def get_database_config() -> Dict[str, Any]:
    try:
        return {
            "host": st.secrets["MYSQL_HOST"],
            "port": int(st.secrets.get("MYSQL_PORT", DEFAULT_MYSQL_PORT)),
            "user": st.secrets["MYSQL_USER"],
            "password": st.secrets["MYSQL_PASSWORD"],
            "database": st.secrets["MYSQL_DB"]
        }
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
    except Exception:
        return pd.DataFrame()

# Helper functions
def generate_investment_analysis(df: pd.DataFrame) -> str:
    """Generate simple investment analysis without emojis"""
    if df.empty:
        return "No hay datos suficientes para el análisis."
    
    try:
        total_investment = df["spend_eur"].sum()
        avg_daily = df["spend_eur"].mean()
        max_investment = df["spend_eur"].max()
        min_investment = df["spend_eur"].min()
        
        if len(df) > 1:
            df_sorted = df.sort_values("fecha_corte")
            last = df_sorted["spend_eur"].iloc[-1]
            first = df_sorted["spend_eur"].iloc[0]
            trend = last - first
            trend_pct = (trend / first * 100) if first > 0 else 0
            
            if trend > 0:
                trend_text = f"Tendencia alcista: aumento del {trend_pct:.1f}%"
            elif trend < 0:
                trend_text = f"Tendencia bajista: disminución del {abs(trend_pct):.1f}%"
            else:
                trend_text = "Tendencia estable"
        else:
            trend_text = "Datos insuficientes para tendencia"
        
        return f"""
        Análisis de Inversión:
        
        Resumen:
        - Inversión total: {total_investment:,.0f}€
        - Promedio diario: {avg_daily:,.0f}€
        - Rango: {min_investment:,.0f}€ - {max_investment:,.0f}€
        
        Tendencia:
        {trend_text}
        
        Recomendación:
        {"Considerar aumentar inversión si ROI es positivo" if avg_daily < 1000 else "Monitorear ROI actual"}
        """
    except Exception:
        return "Error en análisis de datos."

# Navigation - Simple and clean
SECTION_OPTIONS = [
    "Dashboard General",
    "Meta Ads",
    "Landings Pages",
    "Ventas",
    "Performance por Anuncio",  # <-- NUEVA SECCIÓN (PLAN B)
    "Webinar"
]

section = st.sidebar.selectbox(
    "Navegación",
    SECTION_OPTIONS,
    index=0
)

# ---------------------------
# 1) DASHBOARD GENERAL (CONSULTA CORREGIDA)
# ---------------------------
if section == "Dashboard General":
    st.title("Dashboard General - Domustela")
    
    # CONSULTA CORREGIDA: Ahora SÍ incluye la columna 'results'
    df_meta = load_table_safe("SELECT fecha_corte, spend_eur, results FROM meta_campaign_metrics")
    df_ga = load_table_safe("SELECT fecha, sessions, leads FROM landings_performance_new")
    df_sales = load_table_safe("SELECT precio FROM ventas_domustela")
    
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Inversión", f"{df_meta['spend_eur'].sum():,.0f} €")
    # Ahora la columna 'results' existe y se sumará correctamente
    col2.metric("Leads Meta", f"{df_meta['results'].sum():,.0f}" if 'results' in df_meta.columns else "0")
    col3.metric("Sesiones GA4", f"{df_ga['sessions'].sum():,.0f}")
    col4.metric("Ingresos", f"{df_sales['precio'].sum():,.0f} €")
    
    if not df_meta.empty:
        df_meta["fecha_corte"] = pd.to_datetime(df_meta["fecha_corte"])
        meta_daily = df_meta.groupby("fecha_corte", as_index=False)["spend_eur"].sum()
        st.subheader("Evolución Inversión (Meta)")
        st.altair_chart(alt.Chart(meta_daily).mark_line().encode(
            x="fecha_corte:T", y="spend_eur:Q"
        ).properties(height=250), use_container_width=True)

# ---------------------------
# 2) META ADS - With dual charts
# ---------------------------
elif section == "Meta Ads":
    st.title("Meta Ads - Rendimiento")
    
    df = load_table_safe("""
        SELECT fecha_corte, campaign_name, spend_eur, impressions, clicks, results, ctr_pct, cpc, cpl 
        FROM meta_campaign_metrics
    """)
    
    if not df.empty:
        df["fecha_corte"] = pd.to_datetime(df["fecha_corte"])
        
        # Campaign filter
        campaigns = sorted(df["campaign_name"].dropna().unique().tolist())
        selected = st.multiselect("Filtrar campañas", campaigns, default=campaigns[:5])
        df_filtered = df[df["campaign_name"].isin(selected)] if selected else df
        
        # KPIs
        col1, col2, col3 = st.columns(3)
        col1.metric("Inversión total", f"{df_filtered['spend_eur'].sum():,.0f} €")
        cpl_mean = df_filtered['cpl'].mean() if 'cpl' in df_filtered.columns else 0
        col2.metric("CPL medio", f"{cpl_mean:,.2f}")
        ctr_mean = df_filtered['ctr_pct'].mean() if 'ctr_pct' in df_filtered.columns else 0
        col3.metric("CTR medio", f"{ctr_mean:.2f}%")
        
        # Investment evolution with dual chart option
        st.subheader("Evolución de Inversión")
        
        # Single selector for chart type
        chart_option = st.selectbox(
            "Visualización:",
            ["Gráfico de Línea", "Gráfico de Barras"],
            index=0
        )
        
        daily = df_filtered.groupby("fecha_corte", as_index=False)["spend_eur"].sum()
        
        if not daily.empty:
            if chart_option == "Gráfico de Línea":
                chart = alt.Chart(daily).mark_line(point=True).encode(
                    x="fecha_corte:T",
                    y="spend_eur:Q",
                    tooltip=["fecha_corte", "spend_eur"]
                )
            else:
                chart = alt.Chart(daily).mark_bar().encode(
                    x="fecha_corte:T",
                    y="spend_eur:Q",
                    tooltip=["fecha_corte", "spend_eur"]
                )
            
            st.altair_chart(chart.properties(height=350), use_container_width=True)
            
            # Simple investment analysis
            st.text_area("Análisis de Inversión:", 
                        value=generate_investment_analysis(daily),
                        height=150)
        
        # Campaign summary
        st.subheader("Resumen por Campaña")
        st.dataframe(df_filtered[["fecha_corte", "campaign_name", "spend_eur", "impressions", 
                                 "clicks", "results"]].sort_values("spend_eur", ascending=False),
                    use_container_width=True)

# ---------------------------
# 3) LANDINGS PAGES
# ---------------------------
elif section == "Landings Pages":
    st.title("Landings Pages - Conversiones")
    
    df = load_table_safe("SELECT fecha, landing_nombre, sessions, leads, conv_pct FROM landings_performance_new")
    
    if not df.empty:
        df["fecha"] = pd.to_datetime(df["fecha"])
        
        st.subheader("Conversion Rate por Landing")
        conv = df.groupby("landing_nombre", as_index=False)["conv_pct"].mean().sort_values("conv_pct", ascending=False)
        
        st.altair_chart(alt.Chart(conv).mark_bar().encode(
            x=alt.X("landing_nombre:N", sort="-y"),
            y="conv_pct:Q"
        ).properties(height=350), use_container_width=True)
        
        st.subheader("Detalle por Fecha")
        st.dataframe(df.sort_values(["landing_nombre", "fecha"], ascending=[True, False]),
                    use_container_width=True)

# ---------------------------
# 4) VENTAS - Simple and clean
# ---------------------------
elif section == "Ventas":
    st.title("Ventas - Gestión")
    
    # Simple 4-line guide in a box
    with st.expander("Guía para Closers (4 pasos)", expanded=False):
        st.markdown("""
        1. Preguntar al cliente: ¿De qué anuncio viniste?
        2. Anotar en Excel: nombre_anuncio, fecha_compra, nombre_cliente, precio
        3. Subir el archivo Excel aquí
        4. El sistema conecta automáticamente con los datos de Meta
        """)
    
    # Show current sales
    df_sales = load_table_safe("""
        SELECT fecha_compra, nombre_cliente, nombre_anuncio, landing, precio, estado 
        FROM ventas_domustela 
        ORDER BY fecha_compra DESC
    """)
    
    if not df_sales.empty:
        st.subheader("Ventas Registradas")
        st.dataframe(df_sales, use_container_width=True)
    else:
        st.info("No hay ventas registradas.")
    
    # File upload section
    st.subheader("Subir Archivo de Ventas")
    
    uploaded = st.file_uploader(
        "Selecciona archivo Excel o CSV",
        type=["csv", "xlsx"],
        help="Columnas requeridas: fecha_compra, nombre_cliente, nombre_anuncio, precio"
    )
    
    if uploaded:
        try:
            if uploaded.name.endswith(".xlsx"):
                df_new = pd.read_excel(uploaded)
            else:
                df_new = pd.read_csv(uploaded)
            
            st.success(f"Archivo cargado: {uploaded.name}")
            st.write("Vista previa:")
            st.dataframe(df_new.head())
            
            if st.button("Insertar Ventas en Base de Datos"):
                try:
                    df_new.columns = [c.strip() for c in df_new.columns]
                    if "fecha_compra" in df_new.columns:
                        df_new["fecha_compra"] = pd.to_datetime(df_new["fecha_compra"], errors="coerce")
                    
                    df_new.to_sql("ventas_domustela", engine, if_exists="append", index=False)
                    st.success("Ventas insertadas correctamente")
                    st.rerun()
                except Exception as e:
                    st.error(f"Error al insertar: {str(e)}")
                    
        except Exception as e:
            st.error(f"Error al leer archivo: {str(e)}")
    
    # Download template
    st.subheader("Plantilla")
    template_df = pd.DataFrame({
        "fecha_compra": ["2024-01-15"],
        "nombre_cliente": ["Ejemplo Cliente"],
        "nombre_anuncio": ["Nombre del anuncio"],
        "landing": ["1"],
        "precio": [997],
        "estado": ["completada"],
        "notas": [""]
    })
    
    csv = template_df.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="Descargar Plantilla CSV",
        data=csv,
        file_name="plantilla_ventas.csv",
        mime="text/csv"
    )

# ---------------------------
# 5) PERFORMANCE POR ANUNCIO (PLAN B - Fuzzy Matching)
# ---------------------------
elif section == "Performance por Anuncio":
    st.title("Performance por Anuncio - ROI Estimado (Plan B)")
    
    st.markdown("""
    **Objetivo**: Conectar automáticamente las ventas subidas por los closers con las campañas de Meta Ads para estimar el rendimiento (ROI).
    *El sistema busca coincidencias inteligentes (Fuzzy Matching) entre los nombres anotados por closers y los nombres reales de campañas en Meta.*
    """)
    
    # Cargar datos de ventas
    df_ventas = load_table_safe("""
        SELECT nombre_anuncio, precio 
        FROM ventas_domustela 
        WHERE estado = 'completada' AND nombre_anuncio IS NOT NULL
    """)
    
    # Cargar datos de campañas de Meta
    df_meta = load_table_safe("""
        SELECT campaign_name, SUM(spend_eur) as gasto_total
        FROM meta_campaign_metrics 
        GROUP BY campaign_name
        HAVING gasto_total > 0
    """)
    
    if df_ventas.empty or df_meta.empty:
        st.warning("Se necesitan datos de ventas completadas y campañas de Meta para este análisis.")
    else:
        # 1. Agrupar ventas por el nombre que pusieron los closers
        ventas_agrupadas = df_ventas.groupby('nombre_anuncio').agg(
            total_ventas=('precio', 'count'),
            ingresos_totales=('precio', 'sum')
        ).reset_index()
        
        # 2. Fuzzy Matching: Conectar con los nombres reales de Meta
        resultados = []
        campañas_meta = df_meta['campaign_name'].tolist()
        
        for _, fila in ventas_agrupadas.iterrows():
            nombre_closer = str(fila['nombre_anuncio'])
            mejor_coincidencia = None
            mejor_puntaje = 0
            
            # Buscar la campaña de Meta que más se parezca
            for campaña_meta in campañas_meta:
                # Usar rapidfuzz si está disponible, si no, una búsqueda simple
                if HAS_RAPIDFUZZ:
                    puntaje = fuzz.WRatio(nombre_closer, str(campaña_meta))
                else:
                    # Búsqueda simple
                    if nombre_closer.lower() in str(campaña_meta).lower():
                        puntaje = 80
                    else:
                        puntaje = 0
                
                if puntaje > mejor_puntaje and puntaje >= FUZZY_MATCH_THRESHOLD:
                    mejor_puntaje = puntaje
                    mejor_coincidencia = campaña_meta
            
            # Si encontramos una buena coincidencia, guardar los datos combinados
            if mejor_coincidencia:
                gasto_campaña = df_meta.loc[df_meta['campaign_name'] == mejor_coincidencia, 'gasto_total'].iloc[0]
                roi = ((fila['ingresos_totales'] - gasto_campaña) / gasto_campaña * 100) if gasto_campaña > 0 else 0
                
                resultados.append({
                    'Nombre en Excel (Closer)': nombre_closer,
                    'Campaña Meta (Match)': mejor_coincidencia,
                    'Confianza Match': f"{mejor_puntaje:.0f}%",
                    'Total Ventas': fila['total_ventas'],
                    'Ingresos Totales (€)': fila['ingresos_totales'],
                    'Gasto en Meta (€)': gasto_campaña,
                    'ROI Estimado (%)': roi
                })
            else:
                # Si no hay match, mostrar igual la venta
                resultados.append({
                    'Nombre en Excel (Closer)': nombre_closer,
                    'Campaña Meta (Match)': "NO ENCONTRADA",
                    'Confianza Match': "0%",
                    'Total Ventas': fila['total_ventas'],
                    'Ingresos Totales (€)': fila['ingresos_totales'],
                    'Gasto en Meta (€)': 0,
                    'ROI Estimado (%)': 0
                })
        
        # 3. Mostrar resultados
        if resultados:
            df_resultados = pd.DataFrame(resultados)
            
            # --- TABLA PRINCIPAL ---
            st.subheader("Conexión Ventas - Anuncios")
            df_display = df_resultados.copy()
            df_display['Ingresos Totales (€)'] = df_display['Ingresos Totales (€)'].map('€ {:,.0f}'.format)
            df_display['Gasto en Meta (€)'] = df_display['Gasto en Meta (€)'].map('€ {:,.0f}'.format)
            df_display['ROI Estimado (%)'] = df_display['ROI Estimado (%)'].map('{:+.1f}%'.format)
            
            st.dataframe(df_display, use_container_width=True, hide_index=True)
            
            # --- ANÁLISIS PARA DECISIONES ---
            st.subheader("Resumen para Toma de Decisiones")
            
            df_con_match = df_resultados[df_resultados['Campaña Meta (Match)'] != "NO ENCONTRADA"]
            if not df_con_match.empty:
                mejor_anuncio = df_con_match.loc[df_con_match['ROI Estimado (%)'].idxmax()]
                st.success(f"**ANUNCIO A PRIORIZAR:** *{mejor_anuncio['Campaña Meta (Match)']}*")
                st.caption(f"ROI estimado: **{mejor_anuncio['ROI Estimado (%)']:+.1f}%**. Ventas: {mejor_anuncio['Total Ventas']}. Ingresos: {mejor_anuncio['Ingresos Totales (€)']:,.0f}€")
        
        else:
            st.info("No se encontraron conexiones. Revisa que los nombres en 'nombre_anuncio' sean similares a los nombres reales de las campañas en Meta.")

# ---------------------------
# 6) WEBINAR
# ---------------------------
elif section == "Webinar":
    st.title("Webinar - Registros")
    
    dfw = load_table_safe("""
        SELECT fecha_registro, nombre_cliente, email, telefono, asistio, duracion_minutos 
        FROM webinar_registros 
        ORDER BY fecha_registro DESC 
        LIMIT 100
    """)
    
    if not dfw.empty:
        st.dataframe(dfw, use_container_width=True)
    else:
        st.info("No hay registros de webinar.")

# Footer
st.sidebar.markdown("---")
st.sidebar.caption("Domustela Dashboard v5.1 - Plan B Activo")
st.sidebar.caption("JLON Data Solutions")
