# domustela.py
# Versión final completa — limpio, sin debug visible, con secciones solicitadas
# Requisitos (pip): streamlit, pandas, sqlalchemy, mysql-connector-python, altair, rapidfuzz (opcional)

import os
import pandas as pd
import altair as alt
from sqlalchemy import create_engine, text
from datetime import datetime, timedelta

import streamlit as st  # <- st.set_page_config must be the first Streamlit command used
st.set_page_config(page_title="Domustela Dashboard", layout="wide")

# ---------------------------
# CONFIG: leer secrets (desde .streamlit/secrets.toml)
# ---------------------------
try:
    MYSQL_HOST = st.secrets["MYSQL_HOST"]
    MYSQL_PORT = st.secrets.get("MYSQL_PORT", 3306)
    MYSQL_USER = st.secrets["MYSQL_USER"]
    MYSQL_PASSWORD = st.secrets["MYSQL_PASSWORD"]
    MYSQL_DB = st.secrets["MYSQL_DB"]
except Exception as e:
    st.error("No se encontraron credenciales en .streamlit/secrets.toml. Asegúrate de tener MYSQL_HOST, MYSQL_PORT, MYSQL_USER, MYSQL_PASSWORD, MYSQL_DB.")
    st.stop()

# Motor SQLAlchemy (mysqlconnector)
ENGINE_STR = f"mysql+mysqlconnector://{MYSQL_USER}:{MYSQL_PASSWORD}@{MYSQL_HOST}:{MYSQL_PORT}/{MYSQL_DB}"
try:
    engine = create_engine(ENGINE_STR, pool_pre_ping=True, pool_recycle=3600)
except Exception as e:
    st.error(f"No se pudo crear engine SQLAlchemy: {e}")
    st.stop()

# ---------------------------
# FUNCIONES INTERNAS (AUTOMÁTICAS, INVISIBLES)
# ---------------------------
def ensure_indexes_and_tables():
    """Crea índices recomendados y tabla webinar si no existen (silencioso)."""
    idx_sqls = [
        "CREATE INDEX IF NOT EXISTS idx_meta_fecha ON meta_campaign_metrics (fecha_corte(10))",
        # Note: MySQL syntax: CREATE INDEX idx ON table (col(length)) only allowed for varchar/text; adjust if needed.
        # Keep tries inside try/except to avoid errores si no compatible.
    ]
    create_webinar = """
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
            try:
                conn.execute(text(create_webinar))
            except Exception:
                pass
            # ejecutar índices de forma segura (silenciar errores)
            for q in idx_sqls:
                try:
                    conn.execute(text(q))
                except Exception:
                    pass
        return True
    except Exception:
        return False

# Ejecutar ensure automáticamente (no visible)
_ = ensure_indexes_and_tables()

# ---------------------------
# UTIL / CARGA SQL con cache
# ---------------------------
@st.cache_data(ttl=60)
def load_table_safe(query, params=None):
    try:
        df = pd.read_sql(query, engine, params=params)
        return df
    except Exception as e:
        # retornar df vacío si falla
        return pd.DataFrame()

# Intento de importar rapidfuzz (fuzzy match) - opcional
try:
    from rapidfuzz import process, fuzz
    HAS_RAPIDFUZZ = True
except Exception:
    HAS_RAPIDFUZZ = False

# Helpers
def safe_mean(series):
    s = pd.to_numeric(series, errors="coerce")
    s = s.replace([pd.NA, float("inf"), float("-inf")], pd.NA).dropna()
    return s.mean() if not s.empty else None

def normalize_landing(raw):
    if pd.isna(raw): return None
    s = str(raw).lower().strip()
    import re
    m = re.search(r"landing[:\s/_-]*(\d+)", s)
    if m:
        return f"landing{m.group(1)}"
    for token in s.replace("/", " ").split():
        if token.startswith("l") and token[1:].isdigit():
            return f"landing{token[1:]}"
        if token.isdigit() and len(token) <= 4:
            return f"landing{token}"
    return s.replace(" ", "_")[:80]

def merge_sales_with_meta(df_sales, df_meta):
    if df_sales.empty or df_meta.empty:
        return df_sales.assign(campaign_name=None) if not df_sales.empty else pd.DataFrame()
    campaigns = df_meta["campaign_name"].astype(str).fillna("").unique().tolist()
    def map_campaign(name):
        if pd.isna(name): return None
        s = str(name)
        if HAS_RAPIDFUZZ:
            best = process.extractOne(s, campaigns, scorer=fuzz.WRatio)
            if best and best[1] >= 70:
                return best[0]
            return None
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

def storytelling_block(text):
    st.markdown(f"<div style='color:#333;font-size:14px;padding:4px 0'>{text}</div>", unsafe_allow_html=True)

# ---------------------------
# SIDEBAR: Navegación (las secciones que pediste)
# ---------------------------
section = st.sidebar.radio(
    "Navegación",
    [
        "Dashboard General",
        "Meta Ads",
        "Google Analytics (GA4)",
        "Meta + GA4 (Funnel / Merge)",
        "Ventas",
        "Webinar"
    ],
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
# 2) META ADS
# ---------------------------
elif section == "Meta Ads":
    st.title("Meta Ads — Rendimiento y detalle")
    df = load_table_safe("SELECT fecha_corte, campaign_name, spend_eur, impressions, clicks, results, ctr_pct, cpc, cpl FROM meta_campaign_metrics")
    if df.empty:
        st.info("No hay datos de Meta Ads.")
    else:
        df["fecha_corte"] = pd.to_datetime(df["fecha_corte"])
        # filtros
        campaigns = sorted(df["campaign_name"].dropna().unique().tolist())
        sel = st.multiselect("Selecciona campañas (vacío = todas)", campaigns, default=campaigns[:6])
        df_sel = df[df["campaign_name"].isin(sel)] if sel else df.copy()

        # KPIs
        col1, col2, col3 = st.columns(3)
        col1.metric("Inversión total", f"{df_sel['spend_eur'].sum():,.2f} €")
        col2.metric("CPL medio", f"{safe_mean(df_sel['cpl']):,.2f}" if not df_sel.empty else "-")
        col3.metric("CTR medio (%)", f"{safe_mean(df_sel['ctr_pct']):,.2f}" if not df_sel.empty else "-")

        storytelling_block("KPIs calculados sobre la selección de campañas. CPL y CTR ayudan a decidir pausar/escalar.")

        # Inversión por día (selección)
        daily = df_sel.groupby("fecha_corte", as_index=False)["spend_eur"].sum().sort_values("fecha_corte")
        st.subheader("Evolución inversión")
        st.altair_chart(alt.Chart(daily).mark_line(point=True).encode(
            x="fecha_corte:T", y="spend_eur:Q", tooltip=["fecha_corte:T", "spend_eur:Q"]
        ).properties(height=320), use_container_width=True)

        # Tabla resumen
        st.subheader("Resumen por campaña")
        to_show = df_sel[["fecha_corte","campaign_name","spend_eur","impressions","clicks","results","ctr_pct","cpc","cpl"]].sort_values("spend_eur", ascending=False)
        st.dataframe(to_show, use_container_width=True)

# ---------------------------
# 3) GOOGLE ANALYTICS (GA4)
# ---------------------------
elif section == "Google Analytics (GA4)":
    st.title("Google Analytics (GA4) — Landings")
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
        storytelling_block("Identifica las landings con mejor rendimiento y optimiza las que convierten menos.")

        st.subheader("Detalle GA4 (por fecha)")
        st.dataframe(df_sel.sort_values(["landing_nombre","fecha"], ascending=[True, False]), use_container_width=True)

# ---------------------------
# 4) META + GA4 (Funnel / Merge)
# ---------------------------
elif section == "Meta + GA4 (Funnel / Merge)":
    st.title("Meta + GA4 — Funnel y Merge")
    df_meta = load_table_safe("SELECT fecha_corte, campaign_name, spend_eur, clicks, results FROM meta_campaign_metrics")
    df_ga = load_table_safe("SELECT fecha, landing_nombre, sessions, leads FROM landings_performance_new")
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
            if st.button("Insertar (append) en ventas_domustela"):
                try:
                    # normalizar nombres columnas
                    df_new.columns = [c.strip() for c in df_new.columns]
                    if "fecha_compra" in df_new.columns:
                        df_new["fecha_compra"] = pd.to_datetime(df_new["fecha_compra"], errors="coerce")
                    df_new.to_sql("ventas_domustela", engine, if_exists="append", index=False)
                    st.success("Ventas subidas correctamente.")
                except Exception as e:
                    st.error(f"Error al insertar ventas: {e}")
        except Exception as e:
            st.error(f"Error leyendo archivo: {e}")

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
                    st.success("Registros insertados.")
                except Exception as e:
                    st.error(f"Error al insertar registros webinar: {e}")
        except Exception as e:
            st.error(f"Error leyendo CSV: {e}")

# ---------------------------
# FOOTER / HELP (breve)
# ---------------------------
st.sidebar.markdown("---")
st.sidebar.caption("JLON Data Solutions — Domustela")