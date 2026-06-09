# Dashboard — Clasificación de Cáncer con RNA-seq + IA

Dashboard interactivo (Streamlit + Plotly) que comunica los resultados de un proyecto de
clasificación de 18 tipos de cáncer a partir de perfiles de expresión génica RNA-seq (cohorte TCGA),
y su impacto clínico/económico para una IPS.

## Contenido

```
streamlit_app.py        # la aplicación Streamlit
data.csv                # datos de contexto (negocio)
data_local/refined/visualizations/
    app_resumen_ejecutivo.csv
    app_tabla_metricas_informe.csv
    app_desempeno_por_clase.csv
    app_errores_frecuentes.csv
    app_auditoria_predicciones.csv
requirements.txt
```

La app lee estos CSV (no necesita Spark ni reentrenar nada): son las salidas ya procesadas del pipeline.

## Correr en local

```bash
pip install -r requirements.txt
streamlit run streamlit_app.py
```

Se abre en el navegador (por defecto http://localhost:8501).

> En Windows, si el comando `streamlit` no se reconoce, usa:
> `python -m streamlit run streamlit_app.py`

## Desplegar en Streamlit Community Cloud

1. Sube esta carpeta a un repositorio de GitHub.
2. En https://share.streamlit.io conecta el repo.
3. Archivo principal: `streamlit_app.py`.
4. Streamlit instala `requirements.txt` y publica la app.

## Secciones del dashboard

1. El Problema y la Evidencia
2. El Modelo IA (comparación de modelos, desempeño por clase, errores)
3. Impacto y Priorización (clínico, económico, beneficio por EPS, priorización)
4. Simulador
5. Conclusión
