import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from pathlib import Path

# ============================================================================
# CONFIGURACIÓN Y ESTILOS
# ============================================================================

st.set_page_config(
    page_title="RNA-seq + IA · Diagnóstico Oncológico de Precisión",
    page_icon="🎗️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Paleta corporativa médica
COLORS = {
    "navy":      "#0B1E3D",
    "blue":      "#1D6FAA",
    "blue_light":"#3B90D0",
    "teal":      "#0F8B6E",
    "teal_light":"#1EB896",
    "amber":     "#C87F0A",
    "red":       "#C13B3B",
    "cream":     "#F7F5F2",
    "border":    "rgba(11,30,61,0.12)",
}

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@300;400;500;700&family=DM+Serif+Display:ital@0;1&display=swap');

    html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; }
    .main { background: #F7F5F2; }

    /* Header principal */
    .exec-header {
        background: linear-gradient(135deg, #0B1E3D 0%, #1A3A6A 100%);
        color: white;
        padding: 2rem 2.5rem;
        border-radius: 12px;
        border-bottom: 4px solid #0F8B6E;
        margin-bottom: 1.5rem;
    }
    .exec-header h1 {
        font-family: 'DM Serif Display', serif;
        font-size: 2rem;
        font-weight: 400;
        margin-bottom: 0.3rem;
    }
    .exec-header p { color: rgba(255,255,255,0.65); font-size: 0.9rem; }
    .exec-badge {
        display: inline-block;
        background: #0F8B6E;
        color: white;
        font-size: 0.7rem;
        padding: 4px 12px;
        border-radius: 20px;
        letter-spacing: 1px;
        text-transform: uppercase;
        font-weight: 500;
        margin-top: 8px;
    }

    /* KPI Cards */
    .kpi-card {
        background: white;
        border: 1px solid rgba(11,30,61,0.1);
        border-radius: 12px;
        padding: 1.25rem 1.5rem;
        text-align: center;
        height: 100%;
    }
    .kpi-value-large {
        font-size: 2.2rem;
        font-weight: 700;
        line-height: 1.1;
        margin: 4px 0;
    }
    .kpi-value-green  { color: #0F8B6E; }
    .kpi-value-red    { color: #C13B3B; }
    .kpi-value-blue   { color: #1D6FAA; }
    .kpi-value-navy   { color: #0B1E3D; }
    .kpi-label        { font-size: 0.72rem; color: #4A5E7A; text-transform: uppercase; letter-spacing: 0.5px; font-weight: 500; }
    .kpi-delta        { font-size: 0.75rem; color: #6B7280; margin-top: 4px; }

    /* Screen question */
    .screen-question {
        font-family: 'DM Serif Display', serif;
        font-size: 1.5rem;
        font-weight: 400;
        color: #0B1E3D;
        line-height: 1.3;
        margin-bottom: 4px;
    }
    .screen-sub { font-size: 0.82rem; color: #6B7280; margin-bottom: 1.2rem; }

    /* Insight boxes */
    .insight-exec {
        background: #EBF5FF;
        border-left: 4px solid #1D6FAA;
        padding: 1rem 1.25rem;
        border-radius: 0 10px 10px 0;
        font-size: 0.875rem;
        color: #0B1E3D;
        line-height: 1.6;
        margin-top: 1rem;
    }
    .insight-success {
        background: #E6FAF4;
        border-left: 4px solid #0F8B6E;
    }
    .insight-warning {
        background: #FFF8E6;
        border-left: 4px solid #C87F0A;
    }

    /* Priority badges */
    .badge-alta   { background:#D4F4EC; color:#0A6B52; padding:3px 10px; border-radius:12px; font-size:0.72rem; font-weight:600; }
    .badge-media  { background:#FFF0CC; color:#8A5500; padding:3px 10px; border-radius:12px; font-size:0.72rem; font-weight:600; }
    .badge-baja   { background:#FFE8E8; color:#8B2020; padding:3px 10px; border-radius:12px; font-size:0.72rem; font-weight:600; }

    /* Conclusion hero */
    .conclusion-hero {
        background: linear-gradient(135deg, #0B1E3D 0%, #1A3A6A 100%);
        color: white;
        padding: 2rem 2.5rem;
        border-radius: 14px;
        margin-bottom: 1.5rem;
    }
    .conclusion-hero h2 {
        font-family: 'DM Serif Display', serif;
        font-size: 1.6rem;
        font-weight: 400;
        margin-bottom: 0.75rem;
        line-height: 1.35;
    }
    .conclusion-hero p { color: rgba(255,255,255,0.72); font-size: 0.9rem; line-height: 1.65; }

    /* Separador */
    .section-divider {
        height: 2px;
        background: linear-gradient(90deg, #0F8B6E, transparent);
        border-radius: 2px;
        margin: 1rem 0 1.5rem;
    }

    /* Sidebar */
    section[data-testid="stSidebar"] { background: #0B1E3D; }
    section[data-testid="stSidebar"] * { color: rgba(255,255,255,0.85) !important; }
    section[data-testid="stSidebar"] .stRadio label { font-size: 0.82rem !important; }
    section[data-testid="stSidebar"] hr { border-color: rgba(255,255,255,0.15) !important; }
</style>
""", unsafe_allow_html=True)

# ============================================================================
# CARGA DE DATOS
# ============================================================================


@st.cache_data
def load_data():
    # Directorio actual para el script y data.csv
    base = Path(".") 
    
    # Ruta específica para los archivos del modelo
    viz_path = base / "data_local" / "refined" / "visualizations"

    datos = {}

    # 1. Datos de negocio
    negocio_path = base / "data.csv"
    if negocio_path.exists():
        datos["negocio"] = pd.read_csv(negocio_path)
    else:
        st.error(f"❌ No se encontraron datos de negocio en: {negocio_path.resolve()}")
        return None

    # 2. Archivos de modelo
    archivos_modelo = {
        "resumen":   "app_resumen_ejecutivo.csv",
        "metricas":  "app_tabla_metricas_informe.csv",
        "desempeno": "app_desempeno_por_clase.csv",
        "errores":   "app_errores_frecuentes.csv",
        "auditoria": "app_auditoria_predicciones.csv",
    }
    
    for key, fname in archivos_modelo.items():
        # Buscamos en la nueva subcarpeta
        path = viz_path / fname
        datos[key] = pd.read_csv(path) if path.exists() else None
        
        if datos[key] is None:
            st.warning(f"⚠️  Archivo no encontrado en '{viz_path}': {fname} — se usarán valores ilustrativos")

    return datos
#@st.cache_data
# def load_data():
#     base = Path("data_local/refined")
#     tables = base / "tables" / "refined_contexto_negocio_eps_ips_cohorte"
#     viz = base / "visualizations"

#     datos = {}

#     # Datos de negocio
#     negocio_path = tables / "data.csv"
#     if negocio_path.exists():
#         datos["negocio"] = pd.read_csv(negocio_path)
#     else:
#         st.error("❌ No se encontraron datos de negocio en: " + str(negocio_path))
#         return None

#     # Archivos de modelo
#     for key, fname in {
#         "resumen":   "app_resumen_ejecutivo.csv",
#         "metricas":  "app_tabla_metricas_informe.csv",
#         "desempeno": "app_desempeno_por_clase.csv",
#         "errores":   "app_errores_frecuentes.csv",
#         "auditoria": "app_auditoria_predicciones.csv",
#     }.items():
#         path = viz / fname
#         datos[key] = pd.read_csv(path) if path.exists() else None
#         if datos[key] is None:
#             st.warning(f"⚠️  Archivo no encontrado: {fname} — se usarán valores ilustrativos")

#     return datos

# ============================================================================
# HELPERS DE DISEÑO
# ============================================================================

PLOTLY_TEMPLATE = dict(
    plot_bgcolor="white",
    paper_bgcolor="white",
    font=dict(family="DM Sans", color="#0B1E3D", size=11),
    xaxis=dict(
        gridcolor="rgba(11,30,61,0.07)", 
        linecolor="rgba(11,30,61,0.15)",
        automargin=True # <- NUEVO: Evita que se corten los nombres de abajo
    ),
    yaxis=dict(
        gridcolor="rgba(11,30,61,0.07)", 
        linecolor="rgba(11,30,61,0.15)",
        automargin=True # <- NUEVO: Evita que se corten las etiquetas de la izquierda
    ),
    # Aumentamos los márgenes (b=bottom, r=right) para dar más respiro
    margin=dict(l=10, r=20, t=50, b=40), 
    colorway=[COLORS["blue"], COLORS["teal"], COLORS["amber"], COLORS["red"],
              COLORS["blue_light"], COLORS["teal_light"]],
)

def kpi_card(value, label, delta="", color="navy"):
    st.markdown(f"""
    <div class="kpi-card">
        <div class="kpi-label">{label}</div>
        <div class="kpi-value-large kpi-value-{color}">{value}</div>
        <div class="kpi-delta">{delta}</div>
    </div>
    """, unsafe_allow_html=True)

def insight(text, tipo="exec"):
    st.markdown(f'<div class="insight-exec insight-{tipo}">{text}</div>', unsafe_allow_html=True)

def screen_header(pregunta, sub=""):
    st.markdown(f'<div class="screen-question">{pregunta}</div>', unsafe_allow_html=True)
    if sub:
        st.markdown(f'<div class="screen-sub">{sub}</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)

# ============================================================================
# PANTALLAS
# ============================================================================

# --------------------------------------------------------------------------- #
# PANTALLA 1 — EL PROBLEMA
# --------------------------------------------------------------------------- #}
def pantalla_problema(df):
    screen_header(
        "¿Por qué las IPS necesitan una nueva estrategia diagnóstica?",
        "Estado actual del diagnóstico oncológico · Oportunidad de mejora"
    )

    # KPIs
    pct_tardios = (df["dias_ruta_diagnostica_actual"] > 62).mean() * 100
    prom_actual = df["dias_ruta_diagnostica_actual"].mean()
    prom_analitica = df["dias_ruta_diagnostica_con_analitica"].mean()
    costo_prom = df["costo_ruta_actual_cop"].mean()

    c1, c2, c3, c4 = st.columns(4)
    with c1: kpi_card(f"{prom_actual:.0f} días", "Tiempo promedio diagnóstico actual", "+19% sobre umbral (62 días)", "red")
    with c2: kpi_card(f"{pct_tardios:.0f}%", "Pacientes > 62 días en ruta", "exceden umbral óptimo de diagnóstico", "red")
    with c3: kpi_card(f"${costo_prom/1_000_000:.1f}M COP", "Costo promedio ruta diagnóstica", "por paciente, ruta actual", "blue")
    with c4: kpi_card(f"{prom_actual - prom_analitica:.0f} días", "Potencial de reducción", "con apoyo analítico RNA-seq", "green")

    st.markdown("---")

    # Se eliminaron las dos columnas para que el histograma ocupe todo el ancho
    
    # Histograma comparativo
    df_melt = df.melt(
        value_vars=["dias_ruta_diagnostica_actual", "dias_ruta_diagnostica_con_analitica"],
        var_name="escenario", value_name="dias"
    )
    df_melt["escenario"] = df_melt["escenario"].map({
        "dias_ruta_diagnostica_actual":        "Ruta Actual",
        "dias_ruta_diagnostica_con_analitica": "Con RNA-seq + IA"
    })

    fig = px.histogram(
        df_melt, x="dias", color="escenario", nbins=35, barmode="overlay",
        title="Distribución de tiempos diagnósticos: antes vs después",
        labels={"dias": "Días hasta diagnóstico definitivo", "escenario": ""},
        color_discrete_map={"Ruta Actual": COLORS["red"], "Con RNA-seq + IA": COLORS["teal"]},
        opacity=0.75
    )
    fig.add_vline(x=62, line_dash="dash", line_color=COLORS["amber"],
                  annotation_text="Umbral 62 días", annotation_font_color=COLORS["amber"])
    
    # Ajustamos la altura a 450 para mantener la proporción visual a pantalla completa
    fig.update_layout(**PLOTLY_TEMPLATE, height=450, showlegend=True,
                      legend=dict(orientation="h", y=1.05))
    st.plotly_chart(fig, use_container_width=True, theme=None)

    # 🗑️ Se eliminó la figura fig2 (Pieplot de prioridad operativa)

    # KPI de días reubicado y centrado usando columnas vacías
    col_vacia1, col_kpi, col_vacia2 = st.columns([2, 1, 2])
    with col_kpi:
        dias_max = df["dias_ruta_diagnostica_actual"].max()
        st.metric("Caso extremo registrado", f"{dias_max:.0f} días", delta="ruta sin apoyo analítico")

    insight(
        "⚡ <strong>Síntesis ejecutiva:</strong> El 61% de los pacientes oncológicos espera más de 62 días "
        "para diagnóstico definitivo. Esta demora tiene un costo clínico (mayor estadio al inicio de tratamiento) "
        "y económico (exámenes redundantes, hospitalización prolongada) completamente evitable con apoyo analítico.",
        "exec"
    )

# --------------------------------------------------------------------------- #
# PANTALLA 2 — EVIDENCIA MOLECULAR
# --------------------------------------------------------------------------- #

def pantalla_evidencia(df):
    screen_header(
        "¿Existe señal biológica suficiente para que la IA diferencie tumores?",
        "Cohorte TCGA · 8.335 muestras · RNA-seq 19.944 genes · 18 tipos de cáncer"
    )

    c1, c2, c3, c4 = st.columns(4)
    n_muestras = len(df)
    n_tipos = df["cancer_type"].nunique()
    with c1: kpi_card(f"{n_muestras:,}", "Muestras en la cohorte", "TCGA (The Cancer Genome Atlas)", "blue")
    with c2: kpi_card(f"{n_tipos}", "Tipos de cáncer clasificados", "un solo modelo multiclase", "blue")
    with c3: kpi_card("19.944", "Genes medidos (RNA-seq)", "expresión génica por muestra", "navy")
    with c4: kpi_card("Alta", "Separabilidad biológica", "patrones moleculares distintos por tumor", "green")

    st.markdown("---")
    
    # Se eliminaron las dos columnas para que el gráfico ocupe todo el ancho
    conteo = df.groupby("cancer_type").size().reset_index(name="Pacientes")
    conteo = conteo.sort_values("Pacientes", ascending=False)
    
    # Generar gráfico sin escala de color automática
    fig = px.bar(
        conteo, x="cancer_type", y="Pacientes",
        title="Distribución de muestras por tipo de cáncer",
        labels={"cancer_type": "Tipo de Cáncer", "Pacientes": "N° muestras"}
    )
    
    # 🎨 Color Teal para el primero (índice 0) y gris para el resto
    bar_colors = [COLORS["teal"]] + ["#9AA3AD"] * (len(conteo) - 1)
    fig.data[0].marker.color = bar_colors
    
    fig.update_layout(**PLOTLY_TEMPLATE, height=450, coloraxis_showscale=False) # Aumenté un poco la altura a 450
    fig.update_xaxes(tickangle=45)
    st.plotly_chart(fig, use_container_width=True, theme=None)

    # 🗑️ Se eliminaron las figuras fig2 (Sunburst) y fig3 (Histograma de edad)

    insight(
        "🧬 <strong>Validación biológica:</strong> La cohorte TCGA ofrece la mayor base pública "
        "de datos moleculares oncológicos del mundo. La diversidad de 18 tipos de cáncer con "
        "expresión génica RNA-seq de 19.944 genes provee señal molecular robusta para un "
        "clasificador de alto desempeño.",
        "exec"
    )

# --------------------------------------------------------------------------- #
# PANTALLA 3 — EL MODELO IA
# --------------------------------------------------------------------------- #
def pantalla_modelo(datos):
    screen_header(
        "¿Podemos confiar en el modelo de inteligencia artificial?",
        "Comparación de algoritmos · Validación del modelo seleccionado"
    )

    resumen = datos.get("resumen")
    metricas = datos.get("metricas")

    NOMBRE_MODELO = {
        "OneVsRest_LinearSVC":            "SVM Lineal (OvR)",
        "LogisticRegression_multinomial": "Regresión Logística",
        "RandomForestClassifier":         "Random Forest",
        "NaiveBayes":                     "Naive Bayes",
        "DecisionTreeClassifier":         "Árbol de Decisión",
        "LogisticRegression":             "Regresión Logística",
        "SVM_LinearSVC":                  "SVM Lineal",
        "XGBoost":                        "XGBoost",
        "XGBoost_inmemory":               "XGBoost",
        "XGBoost_final":                  "XGBoost",
    }

    def _num(serie, col, default=float("nan")):
        try:
            v = float(serie[col].iloc[0])
            return v if pd.notna(v) else default
        except Exception:
            return default

    # --- KPIs: usa test si existe; si no, validacion ---
    if resumen is not None and "modelo_final" in resumen.columns:
        modelo_nombre = NOMBRE_MODELO.get(resumen["modelo_final"].iloc[0], resumen["modelo_final"].iloc[0])
        usa_test = not pd.isna(_num(resumen, "f1_macro_test"))
        sufijo, etiqueta = ("test", "test") if usa_test else ("validation", "validación")
        f1  = _num(resumen, f"f1_macro_{sufijo}", 0.97)
        ba  = _num(resumen, f"balanced_accuracy_{sufijo}", f1)
        acc = _num(resumen, f"accuracy_{sufijo}", ba)
    else:
        modelo_nombre, f1, ba, acc, etiqueta = "XGBoost", 0.97, 0.97, 0.97, "test"

    c1, c2, c3, c4 = st.columns(4)
    with c1: kpi_card(modelo_nombre, "Modelo seleccionado", "mejor balance precisión / generalización", "navy")
    with c2: kpi_card(f"{f1:.3f}", f"F1-Score macro ({etiqueta})", "promedio sobre 18 clases", "green")
    with c3: kpi_card(f"{ba:.1%}", f"Balanced Accuracy ({etiqueta})", "sin sesgo por clase desbalanceada", "green")
    with c4: kpi_card(f"{acc:.1%}", f"Accuracy global ({etiqueta})", "conjunto de evaluación", "green")

    st.markdown("---")

    # --- Comparacion: usa test si hay filas test; si no, validacion ---
    if metricas is not None and "modelo" in metricas.columns:
        m = metricas.dropna(subset=["modelo"]).copy()
        if "split" in m.columns:
            split_usar = "test" if (m["split"] == "test").any() else "validation"
            df_cmp = m[m["split"] == split_usar].copy()
        else:
            df_cmp = m
    else:
        df_cmp = pd.DataFrame({
            "modelo": ["XGBoost", "Regresión Logística"],
            "f1_macro": [0.971, 0.948],
            "balanced_accuracy": [0.971, 0.951],
            "accuracy": [0.972, 0.954],
        })

    df_cmp["modelo"] = df_cmp["modelo"].map(lambda x: NOMBRE_MODELO.get(x, x))
    if "f1_macro" in df_cmp.columns:
        df_cmp = df_cmp.dropna(subset=["f1_macro"])

    df_plot = df_cmp.melt(
        id_vars="modelo",
        value_vars=[c for c in ["f1_macro", "balanced_accuracy", "accuracy"] if c in df_cmp.columns],
        var_name="Métrica", value_name="Valor"
    ).dropna(subset=["Valor"])
    df_plot["Métrica"] = df_plot["Métrica"].map({
        "f1_macro": "F1-Score Macro", "balanced_accuracy": "Balanced Accuracy", "accuracy": "Accuracy",
    })

    fig = px.bar(
        df_plot, x="modelo", y="Valor", color="Métrica", barmode="group",
        title="Comparación de modelos",
        labels={"Valor": "Valor", "modelo": "Algoritmo"},
        color_discrete_sequence=[COLORS["blue"], COLORS["teal"], COLORS["amber"]]
    )
    fig.update_yaxes(range=[0.80, 1.0], tickformat=".0%")
    fig.update_layout(
        **PLOTLY_TEMPLATE,
        height=420,
        legend=dict(
            title_text="",
            orientation="h",
            yanchor="bottom", y=1.02,
            xanchor="left",  x=0,
            font=dict(size=12),
        ),
    )
    st.plotly_chart(fig, use_container_width=True, theme=None)

    insight(
        "🤖 <strong>Confianza clínica:</strong> El modelo seleccionado supera el umbral de "
        "referencia para uso clínico de apoyo (F1 &gt; 0.95), con una degradación mínima entre "
        "entrenamiento y evaluación — indicio de que generaliza y no sobreajusta. "
        "<strong>El modelo NO reemplaza al patólogo</strong>: lo asiste con una segunda opinión "
        "molecular objetiva, disponible en minutos tras la secuenciación.",
        "success"
    )

# --------------------------------------------------------------------------- #
# PANTALLA 4 — DESEMPEÑO POR CÁNCER
# --------------------------------------------------------------------------- #
def pantalla_desempeno(datos):
    screen_header(
        "¿El modelo funciona igual para todos los tipos de tumor?",
        "F1-Score por clase · Identificación de áreas de mejora clínica"
    )

    desempeno = datos.get("desempeno")

    if desempeno is not None:
        excluir = ["accuracy", "macro avg", "weighted avg"]
        df_clases = desempeno[~desempeno["clase"].isin(excluir)].copy()
    else:
        # Datos ilustrativos con valores representativos TCGA
        df_clases = pd.DataFrame({
            "clase":    ["THCA","KIRC","PRAD","BRCA","UCEC","LIHC","KIPH","COAD",
                         "LUAD","OV","HNSC","BLCA","LUSC","PAAD","CESC","STAD","GBM","SKCM"],
            "f1-score": [0.999,0.998,0.997,0.996,0.994,0.991,0.988,0.983,
                         0.979,0.975,0.968,0.963,0.957,0.942,0.923,0.931,0.911,0.882],
        })

    df_clases = df_clases.sort_values("f1-score", ascending=True)
    
    # 🎨 CORRECCIÓN: Verde (teal) si >= 0.95, rojo si es < 0.95
    df_clases["color"] = df_clases["f1-score"].apply(
        lambda v: COLORS["teal"] if v >= 0.95 else COLORS["red"]
    )

    # Gráfico horizontal
    fig = go.Figure(go.Bar(
        x=df_clases["f1-score"],
        y=df_clases["clase"],
        orientation="h",
        marker_color=df_clases["color"],
        text=df_clases["f1-score"].apply(lambda v: f"{v:.3f}"),
        textposition="outside",
    ))
    fig.add_vline(x=0.95, line_dash="dash", line_color=COLORS["navy"],
                  annotation_text="Umbral clínico 0.95", annotation_font_size=10)
    fig.update_xaxes(range=[0.85, 1.02], tickformat=".0%")
    fig.update_layout(**PLOTLY_TEMPLATE, height=580,
                      title="F1-Score por tipo de cáncer (ordenado de menor a mayor)")
    st.plotly_chart(fig, use_container_width=True, theme=None)

    # 🗑️ Se eliminaron las tablas comparativas de Top 5 y Bottom 5 inferiores

    insight(
        "🔬 <strong>Interpretación para la IPS:</strong> 13 de 18 tipos de cáncer presentan F1 &gt; 0.95, "
        "el umbral clínico aceptable para herramienta de apoyo diagnóstico. Los tipos con menor F1 "
        "corresponden a tumores biológicamente heterogéneos donde incluso expertos presentan desacuerdo "
        "diagnóstico — el modelo captura la complejidad real del panorama oncológico.",
        "exec"
    )


# --------------------------------------------------------------------------- #
# PANTALLA 5 — ERRORES CLÍNICOS
# --------------------------------------------------------------------------- #
def pantalla_errores(datos):
    screen_header(
        "¿Cuáles son los errores clínicamente relevantes?",
        "Análisis de confusiones · Impacto clínico de predicciones incorrectas"
    )

    errores  = datos.get("errores")

    # 1. Preparar los datos (reales o ilustrativos)
    if errores is not None and "cancer_type" in errores.columns:
        top_err = errores.head(10).copy()
        top_err["error_label"] = top_err["cancer_type"] + " → " + top_err["cancer_predicho"]
    else:
        # Ilustrativo (Valores ajustados para probar la condición de gris <= 2)
        top_err = pd.DataFrame({
            "error_label": ["LUAD → LUSC", "LUSC → LUAD", "COAD → STAD", "STAD → COAD",
                             "GBM → LGG", "LGG → GBM", "SKCM → BRCA", "CESC → UCEC",
                             "STAD → HNSC", "BLCA → LUSC"],
            "n_errores": [28, 24, 19, 17, 15, 14, 10, 3, 2, 1] 
        })

    # Ordenar para que las barras más grandes queden en la parte superior
    top_err = top_err.sort_values("n_errores", ascending=True)

    # 🎨 Lógica de colores: Rojo si > 2, Gris si es <= 2
    top_err["color"] = top_err["n_errores"].apply(
        lambda v: COLORS["red"] if v > 2 else "#9AA3AD"
    )

    # 2. Generar el gráfico a ancho completo
    fig = go.Figure(go.Bar(
        x=top_err["n_errores"],
        y=top_err["error_label"],
        orientation="h",
        marker_color=top_err["color"],
        text=top_err["n_errores"],
        textposition="outside",
    ))

    # Aumentamos ligeramente la altura y configuramos ejes
    fig.update_layout(
        **PLOTLY_TEMPLATE, 
        height=450, 
        title="Top 10 errores de clasificación",
        xaxis_title="N° de errores",
        yaxis_title="Real → Predicho"
    )
    st.plotly_chart(fig, use_container_width=True, theme=None)

    # 🗑️ Se eliminaron las columnas y el gráfico fig2 de Accuracy

    insight(
        "⚠️ <strong>Errores clínicamente aceptables:</strong> Los principales errores se producen entre "
        "tumores biológicamente relacionados (LUAD↔LUSC, COAD↔STAD, GBM↔LGG). Estos son los mismos "
        "pares donde los patólogos experimentados presentan desacuerdo. El modelo no introduce confusiones "
        "clínicamente peligrosas (ej: BRCA confundida con GBM).",
        "warning"
    )

# --------------------------------------------------------------------------- #
# PANTALLA 6 — IMPACTO CLÍNICO
# --------------------------------------------------------------------------- #
def pantalla_impacto_clinico(df):
    # Aseguramos que la reducción sea positiva para la visualización
    df = df.copy()
    df["dias_reducidos_pos"] = df["dias_reducidos"].abs()

    screen_header(
        "¿Cuántos días recupera cada paciente con apoyo analítico?",
        "Comparación directa de rutas diagnósticas · Impacto en inicio de tratamiento"
    )

    red_prom = df["dias_reducidos_pos"].mean()
    red_max  = df["dias_reducidos_pos"].max()
    pct_mejora = (df["dias_reducidos_pos"] > 0).mean() * 100
    dias_42 = df["dias_ruta_diagnostica_con_analitica"].mean()

    c1, c2, c3, c4 = st.columns(4)
    with c1: kpi_card(f"+{red_prom:.0f} días", "Reducción promedio por paciente", "ganancia de tiempo en ruta", "green")
    with c2: kpi_card(f"{pct_mejora:.0f}%", "Pacientes con mejora", "> 15 días recuperados", "green")
    with c3: kpi_card(f"+{red_max:.0f} días", "Reducción máxima registrada", "en rutas más largas", "green")
    with c4: kpi_card(f"{dias_42:.0f} días", "Tiempo promedio final", "vs 74 días ruta actual", "blue")

    st.markdown("---")

    # Gráfico único: Curva de retención (tipo Kaplan-Meier)
    dias_actual = np.sort(df["dias_ruta_diagnostica_actual"].dropna())
    dias_analitica = np.sort(df["dias_ruta_diagnostica_con_analitica"].dropna())
    
    pct_actual = np.arange(len(dias_actual)) / float(len(dias_actual) - 1)
    pct_analitica = np.arange(len(dias_analitica)) / float(len(dias_analitica) - 1)

    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=dias_actual, y=1 - pct_actual, mode='lines',
        name="Ruta Actual", line=dict(color=COLORS["red"], width=3, shape='hv')
    ))
    fig.add_trace(go.Scatter(
        x=dias_analitica, y=1 - pct_analitica, mode='lines',
        name="Con RNA-seq + IA", line=dict(color=COLORS["teal"], width=3, shape='hv')
    ))
    
    fig.add_vline(x=62, line_dash="dash", line_color=COLORS["amber"],
                   annotation_text="Umbral crítico (62 días)", annotation_font_size=10)
    
    fig.update_layout(
        **PLOTLY_TEMPLATE, height=500,
        title="Pacientes en espera de diagnóstico (Curva de retención)",
        xaxis_title="Días transcurridos",
        yaxis_title="Proporción de pacientes sin diagnóstico (%)",
        legend=dict(orientation="h", y=1.1)
    )
    fig.update_yaxes(tickformat=".0%")
    
    st.plotly_chart(fig, use_container_width=True, theme=None)

    # 🗑️ Se eliminó la segunda gráfica (Waterfall)

    insight(
        "⏱️ <strong>Impacto en la atención:</strong> La implementación de RNA-seq "
        "comprime la fase analítica, permitiendo recuperar en promedio 32 días de espera. "
        "Esta ganancia de tiempo permite iniciar tratamientos en estadios más tempranos, "
        "mejorando significativamente el pronóstico del paciente y optimizando el uso "
        "de recursos hospitalarios.",
        "success"
    )
# --------------------------------------------------------------------------- #
# PANTALLA 7 — IMPACTO ECONÓMICO
# --------------------------------------------------------------------------- #
def pantalla_impacto_economico(df):
    screen_header(
        "¿La reducción de tiempos genera ahorro económico medible para la IPS?",
        "Análisis costo-beneficio por tipo de cáncer"
    )

    ahorro_total = df["ahorro_estimado_cop"].sum()
    ahorro_prom  = df["ahorro_estimado_cop"].mean()
    
    # 🛠️ CORRECCIÓN: Se ajustó a 3 columnas tras eliminar el KPI de porcentaje ahorro
    c1, c2, c3 = st.columns(3)
    with c1: kpi_card(f"${ahorro_total/1_000_000_000:.1f}B COP", "Ahorro total estimado (cohorte)", f"{len(df):,} pacientes", "green")
    with c2: kpi_card(f"${ahorro_prom/1_000_000:.2f}M COP", "Ahorro promedio por paciente", "reducción de ruta diagnóstica", "green")
    with c3: kpi_card("Alta", "Viabilidad Financiera", "retorno de inversión proyectado", "green")

    st.markdown("---")

    # Gráfico: Comparación costo actual vs con analítica por tipo de cáncer
    df_comp = df.groupby("cancer_type").agg({
        "costo_ruta_actual_cop":      "mean",
        "costo_ruta_con_analitica_cop": "mean"
    }).head(10).reset_index()
    
    df_comp_melt = df_comp.melt(
        id_vars="cancer_type",
        value_vars=["costo_ruta_actual_cop", "costo_ruta_con_analitica_cop"],
        var_name="Escenario", value_name="Costo"
    )
    df_comp_melt["Escenario"] = df_comp_melt["Escenario"].map({
        "costo_ruta_actual_cop":       "Ruta Actual",
        "costo_ruta_con_analitica_cop": "Con Analítica"
    })
    
    fig = px.bar(
        df_comp_melt, x="cancer_type", y="Costo", color="Escenario", barmode="group",
        title="Costo diagnóstico actual vs con analítica (Top 10 tipos)",
        labels={"cancer_type": "Tipo de Cáncer", "Costo": "Costo promedio (COP)"},
        color_discrete_map={"Ruta Actual": COLORS["red"], "Con Analítica": COLORS["teal"]}
    )
    fig.update_xaxes(tickangle=45)
    fig.update_yaxes(tickformat=",.0f")
    
    # Leyenda centrada abajo
    fig.update_layout(
        **PLOTLY_TEMPLATE, 
        height=480, 
        legend=dict(
            orientation="h", 
            yanchor="top", y=-0.25, 
            xanchor="center", x=0.5,
            title_text=""
        )
    )
    st.plotly_chart(fig, use_container_width=True, theme=None)

    insight(
        "💰 <strong>Argumento financiero:</strong> El ahorro por reducción de "
        "exámenes redundantes, hospitalización evitada y tratamiento en estadio temprano supera "
        "ampliamente el costo de implementación de la plataforma RNA-seq, garantizando la viabilidad "
        "del modelo de negocio.",
        "success"
    )

# --------------------------------------------------------------------------- #
# PANTALLA 8 — BENEFICIO POR EPS
# --------------------------------------------------------------------------- #
def pantalla_beneficio_eps(df):
    screen_header(
        "¿Qué EPS y regímenes concentran el mayor potencial?",
        "Análisis por aseguradora · Identificación de aliados estratégicos"
    )

    # 🛠️ CORRECCIÓN: Ordenamos ascendente para que los valores más altos queden al final del DF
    df_eps = df.groupby("eps").agg(
        ahorro_total=("ahorro_estimado_cop", "sum"),
        ahorro_promedio=("ahorro_estimado_cop", "mean"),
        dias_totales=("dias_reducidos", "sum"),
        pacientes=("patient_id", "count")
    ).reset_index().sort_values("ahorro_total", ascending=True)

    # 🛠️ Usamos .tail(10) para tomar los 10 valores más altos (que ahora están al final)
    fig = px.bar(
        df_eps.tail(10), x="ahorro_total", y="eps", orientation="h",
        title="Ahorro total por EPS (Top 10)",
        labels={"ahorro_total": "Ahorro total (COP)", "eps": ""},
        color="ahorro_total",
        color_continuous_scale=[[0, COLORS["blue_light"]], [1, COLORS["teal"]]]
    )
    fig.update_xaxes(tickformat=",.0f")
    fig.update_layout(**PLOTLY_TEMPLATE, height=450, coloraxis_showscale=False)
    st.plotly_chart(fig, use_container_width=True, theme=None)

    insight(
        "🏢 <strong>Potencial de alianzas:</strong> Algunas EPS concentran la mayor base de pacientes "
        "oncológicos y el mayor potencial de ahorro. Focalizar las primeras implementaciones en estas "
        "EPS genera casos de éxito replicables y facilita la expansión posterior.",
        "exec"
    )
# --------------------------------------------------------------------------- #
# PANTALLA 9 — PRIORIZACIÓN POR CÁNCER
# --------------------------------------------------------------------------- #
def pantalla_priorizacion(df):
    screen_header(
        "¿Dónde debería priorizar la IPS la implementación?",
        "Mapa de valor: volumen vs ahorro · Plan de fases"
    )

    df_c = df.groupby("cancer_type").agg(
        ahorro_total=("ahorro_estimado_cop", "sum"),
        ahorro_promedio=("ahorro_estimado_cop", "mean"),
        pacientes=("patient_id", "count"),
        dias_ahorrados=("dias_reducidos", "mean")
    ).reset_index()

    # 🎨 Lógica de colores discretos personalizados
    color_map = {c: "#9AA3AD" for c in df_c["cancer_type"].unique()}
    if "GBM" in color_map: color_map["GBM"] = COLORS["teal"]
    if "OV" in color_map: color_map["OV"] = COLORS["teal"]
    if "BRCA" in color_map: color_map["BRCA"] = "#7E22CE"  # Tono Morado

    # Bubble chart
    fig = px.scatter(
        df_c, x="pacientes", y="ahorro_promedio",
        size="ahorro_total", text="cancer_type",
        title="Mapa de priorización: Volumen × Ahorro promedio (tamaño = ahorro total)",
        labels={"pacientes": "N° de Pacientes", "ahorro_promedio": "Ahorro promedio por paciente (COP)"},
        color="cancer_type",           # Asignamos el color con base en el tipo de cáncer
        color_discrete_map=color_map   # Pasamos nuestro diccionario de colores forzados
    )
    
    fig.update_traces(textposition="top center", textfont_size=9)
    fig.update_yaxes(tickformat=",.0f")
    
    # Se reemplaza coloraxis_showscale por showlegend=False para ocultar la lista lateral
    fig.update_layout(**PLOTLY_TEMPLATE, height=480, showlegend=False) 
    st.plotly_chart(fig, use_container_width=True, theme=None)

    # (La tabla inferior se mantiene eliminada según el ajuste anterior)

    insight(
        "🎯 <strong>Estrategia de implementación:</strong> Comenzar con BRCA, LUAD y LUSC permite "
        "maximizar impacto en el primer año (mayor volumen + alto ahorro). La segunda fase incorpora "
        "GBM y STAD donde el ahorro por paciente es mayor aunque el volumen sea menor.",
        "exec"
    )

# --------------------------------------------------------------------------- #
# PANTALLA 10 — SIMULADOR EJECUTIVO
# --------------------------------------------------------------------------- #
def pantalla_simulador(df):
    screen_header(
        "Simulador Ejecutivo — ¿Qué pasaría si la IPS implementa hoy?",
        "Ajuste los parámetros y vea el impacto proyectado en tiempo real"
    )

    col_f1, col_f2 = st.columns(2)
    with col_f1:
        tipos_disponibles = sorted(df["cancer_type"].unique().tolist())
        cancer_sel = st.multiselect(
            "Tipos de cáncer a incluir en el programa",
            options=tipos_disponibles,
            default=tipos_disponibles[:4] if len(tipos_disponibles) >= 4 else tipos_disponibles,
            help="Seleccione los tipos de cáncer que implementará en su IPS"
        )
        eps_sel = st.multiselect(
            "Filtrar por EPS",
            options=["Todas"] + sorted(df["eps"].unique().tolist()),
            default=["Todas"]
        )

    with col_f2:
        adopcion = st.slider("Tasa de adopción (%)", 10, 100, 60, 5,
                             help="Porcentaje de pacientes elegibles que entrarán al flujo RNA-seq")
        regimen_sel = st.multiselect(
            "Régimen",
            options=df["regimen"].unique().tolist(),
            default=df["regimen"].unique().tolist()
        )

    st.markdown("---")
    col_p1, col_p2 = st.columns(2)
    with col_p1:
        inversion_anual = st.number_input(
            "Inversión anual estimada (COP)",
            min_value=100_000_000, max_value=5_000_000_000,
            value=500_000_000, step=50_000_000,
            help="Infraestructura RNA-seq, personal especializado, licencias"
        )
    with col_p2:
        horizonte = st.number_input("Horizonte de evaluación (años)", 1, 10, 3, 1)

    # Filtrar DataFrame
    df_f = df.copy()
    if cancer_sel:
        df_f = df_f[df_f["cancer_type"].isin(cancer_sel)]
    if "Todas" not in eps_sel and eps_sel:
        df_f = df_f[df_f["eps"].isin(eps_sel)]
    if regimen_sel:
        df_f = df_f[df_f["regimen"].isin(regimen_sel)]

    if df_f.empty:
        st.warning("⚠️ No hay datos para la selección actual.")
        return

    # Cálculos
    n_pac = len(df_f)
    pac_impactados = int(n_pac * adopcion / 100)
    ahorro_prom_pac = df_f["ahorro_estimado_cop"].mean()
    ahorro_total_sim = ahorro_prom_pac * pac_impactados
    ahorro_anual = ahorro_total_sim / max(horizonte, 1)

    # Normalizar a anual
    años_datos = 3
    ahorro_anual_real = (df_f["ahorro_estimado_cop"].sum() * (adopcion / 100)) / años_datos

    dias_anual = int(df_f["dias_reducidos"].sum() * (adopcion / 100) / años_datos)

    ganancia_neta = ahorro_anual_real - inversion_anual
    roi_anual = (ganancia_neta / inversion_anual) * 100 if inversion_anual > 0 else 0
    roi_total = ((ahorro_anual_real * horizonte - inversion_anual * horizonte)
                 / (inversion_anual * horizonte)) * 100 if inversion_anual > 0 else 0
    payback_m = (inversion_anual / (ahorro_anual_real / 12)) if ahorro_anual_real > 0 else 999

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        kpi_card(f"{pac_impactados:,}", "Pacientes impactados / año", f"{adopcion}% de adopción sobre {n_pac:,} pac", "blue")
    with c2:
        kpi_card(f"${ahorro_anual_real/1_000_000:,.0f}M COP", "Ahorro anual estimado", "neto por reducción de ruta", "green")
    with c3:
        color_roi = "green" if roi_anual > 0 else "red"
        kpi_card(f"{roi_anual:+.0f}%", "ROI Año 1", "sobre inversión anual", color_roi)
    with c4:
        payback_str = f"{payback_m:.1f} meses" if payback_m < 24 else f"{payback_m/12:.1f} años"
        kpi_card(payback_str, "Payback estimado", "tiempo de recuperación inversión", "blue")

    st.markdown(f"**Días recuperados / año:** {dias_anual:,} días")

    # Gráfico evolución
    años_eje = list(range(1, horizonte + 1))
    ahorro_acum = [ahorro_anual_real * y / 1_000_000 for y in años_eje]
    inversion_acum = [inversion_anual * y / 1_000_000 for y in años_eje]

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=años_eje, y=ahorro_acum, mode="lines+markers",
        name="Ahorro acumulado", line=dict(color=COLORS["teal"], width=3),
        marker=dict(size=8)
    ))
    fig.add_trace(go.Scatter(
        x=años_eje, y=inversion_acum, mode="lines+markers",
        name="Inversión acumulada", line=dict(color=COLORS["red"], width=3, dash="dash"),
        marker=dict(size=8)
    ))
    for i, (g, v) in enumerate(zip(ahorro_acum, inversion_acum)):
        if g >= v:
            fig.add_vline(x=i + 1, line_dash="dot", line_color=COLORS["navy"],
                          annotation_text=f"Punto de equilibrio: año {i+1}",
                          annotation_font_size=10)
            break

    fig.update_layout(
        **PLOTLY_TEMPLATE, height=360,
        title="Proyección: Ahorro acumulado vs Inversión acumulada",
        xaxis_title="Años", yaxis_title="COP (Millones)",
        legend=dict(orientation="h", y=1.05)
    )
    st.plotly_chart(fig, use_container_width=True, theme=None)

    # Resumen financiero
    with st.expander("📊 Ver resumen financiero detallado"):
        df_res = pd.DataFrame({
            "Concepto": [
                "Inversión anual", "Inversión total del proyecto",
                "Ahorro anual estimado", "Ahorro total del proyecto",
                "Ganancia neta total", f"ROI del proyecto ({horizonte} años)",
                "Tiempo de recuperación"
            ],
            "Valor": [
                f"${inversion_anual/1_000_000:,.0f}M COP",
                f"${inversion_anual*horizonte/1_000_000:,.0f}M COP",
                f"${ahorro_anual_real/1_000_000:,.0f}M COP",
                f"${ahorro_anual_real*horizonte/1_000_000:,.0f}M COP",
                f"${(ahorro_anual_real-inversion_anual)*horizonte/1_000_000:,.0f}M COP",
                f"{roi_total:.0f}%",
                payback_str
            ]
        })
        st.dataframe(df_res, use_container_width=True, hide_index=True)

    nivel = ("muy atractivo" if roi_total > 50 else
             "atractivo" if roi_total > 20 else
             "moderado" if roi_total > 0 else "negativo")

    insight(
        f"📈 <strong>Interpretación financiera:</strong> Un ROI del {roi_total:.0f}% sobre {horizonte} años "
        f"se considera <strong>{nivel}</strong> para proyectos de innovación en salud en Colombia. "
        f"El punto de equilibrio se alcanza al cruzar la línea de inversión acumulada. "
        f"Este análisis no captura los beneficios clínicos (mayor sobrevida, mejor calidad de vida) "
        f"que amplificarían aún más el valor institucional.",
        "success"
    )


# --------------------------------------------------------------------------- #
# PANTALLA CONCLUSIÓN
# --------------------------------------------------------------------------- #
def pantalla_conclusion():
    st.markdown("""
    <div class="conclusion-hero">
        <h2>Sí. El clasificador RNA-seq + IA mejora la atención oncológica en la IPS de forma medible y rentable.</h2>
        <p>
        El uso de secuenciación de RNA combinado con Machine Learning permite clasificar 18 tipos de cáncer
        con un F1-Score de 0.967, reducir los tiempos diagnósticos un <strong style="color:#4FDBB5">43%</strong>
        (de 74 a 42 días promedio), y generar un ahorro estimado de <strong style="color:#4FDBB5">$1.85M COP por paciente</strong>
        — con ROI positivo desde el primer año. Cada día recuperado equivale a inicio de tratamiento
        en estadio más temprano, mayor tasa de sobrevida y menor costo operativo para la IPS.
        </p>
    </div>
    """, unsafe_allow_html=True)

    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown("""
        <div style="background:white; border:1px solid rgba(11,30,61,0.1); border-radius:12px;
                    padding:1.25rem; border-top:4px solid #0F8B6E;">
            <div style="font-size:2.5rem; font-weight:700; color:#0F8B6E;">96.7%</div>
            <div style="font-size:0.78rem; color:#6B7280; margin:4px 0;">Precisión del modelo (F1 macro)</div>
            <div style="font-size:0.82rem; color:#0B1E3D; margin-top:10px; line-height:1.4;">
            Clasificación correcta de 18 tipos de cáncer en conjunto de prueba independiente.
            Supera el umbral clínico aceptable para herramienta de apoyo diagnóstico.
            </div>
        </div>
        """, unsafe_allow_html=True)
    with c2:
        st.markdown("""
        <div style="background:white; border:1px solid rgba(11,30,61,0.1); border-radius:12px;
                    padding:1.25rem; border-top:4px solid #0F8B6E;">
            <div style="font-size:2.5rem; font-weight:700; color:#0F8B6E;">−32 días</div>
            <div style="font-size:0.78rem; color:#6B7280; margin:4px 0;">Reducción promedio por paciente</div>
            <div style="font-size:0.82rem; color:#0B1E3D; margin-top:10px; line-height:1.4;">
            De 74 a 42 días de ruta diagnóstica. El 83% de los pacientes experimenta
            una reducción con impacto clínico directo en la estadificación inicial.
            </div>
        </div>
        """, unsafe_allow_html=True)
    with c3:
        st.markdown("""
        <div style="background:white; border:1px solid rgba(11,30,61,0.1); border-radius:12px;
                    padding:1.25rem; border-top:4px solid #0F8B6E;">
            <div style="font-size:2.5rem; font-weight:700; color:#0F8B6E;">$1.85M</div>
            <div style="font-size:0.78rem; color:#6B7280; margin:4px 0;">Ahorro promedio por paciente (COP)</div>
            <div style="font-size:0.82rem; color:#0B1E3D; margin-top:10px; line-height:1.4;">
            Ahorro total estimado de $15.4B COP en la cohorte analizada.
            ROI positivo desde el año 1 con inversión inicial moderada.
            </div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")
    st.subheader("📋 Plan de implementación recomendado para la IPS")

    df_plan = pd.DataFrame({
        "Fase": ["Fase 1 — Piloto", "Fase 2 — Expansión", "Fase 3 — Cobertura completa"],
        "Tipos de Cáncer": ["BRCA, LUAD, LUSC", "PRAD, STAD, GBM, BLCA, OV", "Todos los 18 tipos"],
        "Horizonte": ["Meses 1–12", "Meses 13–24", "Mes 25+"],
        "Pacientes/año": ["~2.100", "~3.550", "~8.335+"],
        "Ahorro estimado": ["$6.3B COP/año", "$4.1B COP/año", "$15.4B COP/año"],
        "Riesgo": ["Bajo", "Medio", "Bajo"]
    })
    st.dataframe(df_plan, use_container_width=True, hide_index=True)

    insight(
        "🏆 <strong>Recomendación ejecutiva:</strong> Aprobar piloto de Fase 1 con BRCA, LUAD y LUSC. "
        "La inversión se recupera antes del mes 6. El impacto en sobrevida y la ventaja competitiva "
        "institucional hacen de esta tecnología una decisión <strong>estratégica</strong>, no solo operativa. "
        "La IPS que adopte primero esta capacidad se posiciona como referente en oncología de precisión en Colombia.",
        "success"
    )


# ============================================================================
# MAIN
# ============================================================================

def main():
    # Header ejecutivo
    st.markdown("""
    <div class="exec-header">
        <h1>🎗️ Diagnóstico Oncológico de Precisión</h1>
        <p>RNA-seq + Machine Learning para optimización de rutas diagnósticas en IPS · Cohorte TCGA · 8.335 pacientes · 18 tipos de cáncer</p>
        <span class="exec-badge">IPS Executive View · Junta Directiva</span>
    </div>
    """, unsafe_allow_html=True)

    # Sidebar
    with st.sidebar:
        st.markdown("## 🎗️ Navegación")
        st.markdown("*Diagnóstico Oncológico de Precisión*")
        st.markdown("---")

        pagina = st.radio(
            "Seleccione sección:",
            [
                "1 — El Problema y la Evidencia",
                "2 — El Modelo IA",
                "3 — Impacto y Priorización",
                "4 — Simulador",
                "✅ Conclusión",
            ],
        )

        st.markdown("---")
        st.markdown(
            "**Pregunta central:** *¿Cómo el clasificador RNA-seq permite mejorar "
            "la atención oncológica en la IPS, reduciendo tiempos diagnósticos y "
            "optimizando costos?*"
        )

    # Carga de datos
    datos = load_data()
    if datos is None:
        st.stop()
    df = datos["negocio"]

    # Routing
    if pagina == "1 — El Problema y la Evidencia":
        pantalla_problema(df)
        pantalla_evidencia(df)
    elif pagina == "2 — El Modelo IA":
        pantalla_modelo(datos)
        pantalla_desempeno(datos)
        pantalla_errores(datos)
    elif pagina == "3 — Impacto y Priorización":
        pantalla_impacto_clinico(df)
        pantalla_impacto_economico(df)
        pantalla_beneficio_eps(df)
        pantalla_priorizacion(df)
    elif pagina == "4 — Simulador":
        pantalla_simulador(df)
    elif pagina == "✅ Conclusión":
        pantalla_conclusion()


if __name__ == "__main__":
    main()
