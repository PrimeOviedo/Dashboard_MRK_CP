import streamlit as st
import pandas as pd
import pydeck as pdk
import hashlib
import glob
import streamlit.components.v1 as components
import plotly.graph_objects as go
import matplotlib.colors as mcolors

st.set_page_config(page_title="Dashboard Maestro RTM - MRK", layout="wide")

st.markdown("""
    <style>
    /* 🪄 Selecciona el bloque padre que contenga la marca glass-marker */
    div[data-testid="stVerticalBlock"] > div:has(.glass-marker) {
        background: rgba(255, 255, 255, 0.08);
        display: none;
        backdrop-filter: blur(12px);
        -webkit-backdrop-filter: blur(12px);
        border-radius: 18px;
        padding: 25px;
        border: 1px solid rgba(255, 255, 255, 0.25);
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.37);
        margin-top: 20px;
        margin-bottom: 20px;
    }
    </style>
""", unsafe_allow_html=True)

st.title('Dashboard Mi Ruta KOF Centro-Pacifico')

df_mrk = pd.read_parquet('bdd_mrk_cp.parquet')

df_mrk['Jefatura'] = df_mrk['Jefatura'].astype(str).str.strip()
df_mrk = df_mrk[df_mrk['Jefatura'].notna() & (df_mrk['Jefatura'] != '')]

col1_1, col1_2, col1_3 = st.columns([1.5,2,5])

with col1_1:
    import base64

    with open("assets/logo_mrk.png", "rb") as f:
        data = base64.b64encode(f.read()).decode()

    st.markdown(
        f"""
        <div style="display: flex; justify-content: center; align-items: center; height: 150px;">
            <img src="data:image/png;base64,{data}" width="250">
        </div>
        """,
        unsafe_allow_html=True
    )

with col1_2:
    # 📅 Filtro de fechas
    st.subheader("📆 Filtro de Fechas")

    fecha_min = df_mrk["Fecha inicio"].min().date()
    fecha_max = df_mrk["Fecha inicio"].max().date()

    rango_fechas = st.date_input(
        "Selecciona un rango de fechas:",
        value=(fecha_min, fecha_max),
        min_value=fecha_min,
        max_value=fecha_max
    )

    if isinstance(rango_fechas, tuple) and len(rango_fechas) == 2:
        fecha_inicio, fecha_fin = rango_fechas
        df_mrk = df_mrk[
            (df_mrk["Fecha inicio"].dt.date >= fecha_inicio) &
            (df_mrk["Fecha inicio"].dt.date <= fecha_fin)
            ]

df_zona = df_mrk.pivot_table(index='Zona', values=['Geoeficiencia','Geoefectividad','Efectividad omnicanal','Primer cliente (%)','Tiempo de servicio (%)'], aggfunc='mean')

df_mrk_trad = df_mrk[~df_mrk['Región'].isin(['MODERNO','MAYORISTAS'])]
df_regiones_trad = df_mrk_trad.pivot_table(index='Región', values=['Geoeficiencia','Geoefectividad','Efectividad omnicanal','Primer cliente (%)','Tiempo de servicio (%)'], aggfunc='mean')

df_mrk_mod = df_mrk[df_mrk['Región'].isin(['MODERNO','MAYORISTAS'])]
df_regiones_mod = df_mrk_mod.pivot_table(index='Región', values=['Geoeficiencia','Geoefectividad','Efectividad omnicanal','Primer cliente (%)','Tiempo de servicio (%)'], aggfunc='mean')
# 📈 Multiplicar todos los valores por 100
df_regiones_trad = df_regiones_trad * 100
df_regiones_mod = df_regiones_mod * 100

st.subheader("🗺️ Indicadores a Total Zona")
col2_1, col2_2 = st.columns([5, 4])

with col2_1:
    st.dataframe(df_regiones_trad.style.format("{:.1f}%"))
    st.dataframe(df_regiones_mod.style.format("{:.1f}%"))

with col2_2:

    import plotly.graph_objects as go

    # 📌 Variables y valores de la zona
    indicadores = df_zona.columns.tolist()
    zona = df_zona.index[0]  # solo hay una
    valores = df_zona.loc[zona].values

    # 🎨 Color personalizado
    color_base = '#fc0000'  # 🔴 Rojo Coca-Cola (puedes cambiarlo)

    # 📈 Polígono principal
    fig = go.Figure()

    fig.add_trace(go.Scatterpolar(
        r = valores * 100,  # 👉 escalar también aquí
        theta = indicadores,
        fill = 'toself',
        name = zona,
        line=dict(color=color_base),
        fillcolor=color_base,
        opacity=0.5
    ))

    # 📍 Etiquetas de porcentaje en cada punto
    fig.add_trace(go.Scatterpolar(
        r = (valores * 100),
        theta = indicadores,
        mode = 'markers',
        textposition = 'middle center',
        textfont=dict(size=16),          # 👈 más pequeño para que se vea más adentro
        marker=dict(color=color_base, size=12),
        showlegend=False
    ))

    # 📍 Etiquetas de porcentaje en cada punto
    fig.add_trace(go.Scatterpolar(
        r = (valores * 100) * 0.8,  # 👈 se empujan un poco hacia adentro
        theta = indicadores,
        mode = 'text',
        text = [f'{v*100:.1f}%' for v in valores],  # 👈 multiplicar por 100 para mostrar bien
        textposition = 'middle center',
        textfont=dict(size=16),
        showlegend=False
    ))

    # 🛠 Diseño
    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                showticklabels=False,
                range=[0, (df_zona.max().max() * 100)],
                tickformat=".0f",
                tickangle=0,
                color='white',# 👉 color del texto de los ejes
            ),
            angularaxis=dict(rotation=-54,
                             tickfont=dict(size=18, color="white")
                             ),
            bgcolor="rgba(0,0,0,0)"  # 🌑 fondo de la zona polar
        ),  # 🖤 fondo general
        font=dict(color="white"),  # ✨ texto en blanco para contraste
        showlegend=False,
        height=500,
        margin=dict(l=40, r=40, t=40, b=40)
    )

    st.plotly_chart(fig, use_container_width=True)

st.divider()

col_kpi_1, col_kpi_2 = st.columns([1,5])

with col_kpi_1:
    # 📌 Lista de indicadores disponibles
    indicadores_disponibles = ['Geoeficiencia', 'Geoefectividad', 'Efectividad omnicanal', 'Primer cliente (%)', 'Tiempo de servicio (%)']

    # 🧭 Selector de indicador
    indicador_seleccionado = st.selectbox(
        "📊 Selecciona el indicador a visualizar:",
        options=indicadores_disponibles
    )
with col_kpi_2:
    st.subheader("")

def asignar_color(valor):
    if valor > 89.99:
        return '#7ED957'  # Verde
    elif valor >= 50:
        return '#FFBD59'  # Amarillo
    else:
        return '#FF5757'  # Rojo

# 🧮 Ordenar dinámicamente por el indicador seleccionado
df_top5 = df_regiones_trad.sort_values(by=indicador_seleccionado, ascending=False).head(5)
df_bottom5 = df_regiones_trad.sort_values(by=indicador_seleccionado, ascending=True).head(5)

# 🎨 Asignar color según el valor del indicador seleccionado
df_top5 = df_top5.assign(color=df_top5[indicador_seleccionado].apply(asignar_color))
df_bottom5 = df_bottom5.assign(color=df_bottom5[indicador_seleccionado].apply(asignar_color))

# 🧭 Indicador actual
indicador = indicador_seleccionado

# 📊 Agrupar por Región y Jefatura SOLO con el DF filtrado
df_jef = (
    df_mrk[df_mrk['Región'].isin(df_top5.index.tolist() + df_bottom5.index.tolist())]
    .groupby(['Región', 'Jefatura'])[indicador]
    .mean()
    .reset_index() * 100
)

# 📌 Ordenar de mayor a menor
df_jef = df_jef.sort_values(by=['Región', indicador], ascending=[True, False])
df_jef[indicador] = df_jef[indicador].fillna(0)

# 🏆 Top 5 por región
df_top_jef = df_jef.groupby('Región').head(5).copy()
df_bottom_jef = df_jef.groupby('Región').tail(5).copy()

df_top_jef = df_top_jef.drop_duplicates(subset=['Región', 'Jefatura'])
df_bottom_jef = df_bottom_jef.drop_duplicates(subset=['Región', 'Jefatura'])

def color_por_valor(valor):
    if valor > 89.99:
        return '#7ED957'
    elif valor >= 50:
        return '#FFBD59'
    else:
        return '#FF5757'

df_top_jef['Color'] = df_top_jef[indicador].apply(color_por_valor)
df_bottom_jef['Color'] = df_bottom_jef[indicador].apply(color_por_valor)

import plotly.express as px

# 🥇 Top 5
fig_top = px.bar(
    df_top5.reset_index(),
    x='Región',
    y=indicador_seleccionado,
    text=indicador_seleccionado,
)
fig_top.update_traces(
    marker_color=df_top5['color'],
    texttemplate='%{text:.1f}%',
    textposition='outside'
)
fig_top.update_layout(
    title=f'🏆 Top 5 {indicador_seleccionado}',
    yaxis_title=f"{indicador_seleccionado} (%)",
    xaxis_title="Región",
    plot_bgcolor='rgba(0,0,0,0)',
    paper_bgcolor='rgba(0,0,0,0)',
    font=dict(color='white'),
    height=420,
)

# 🪫 Bottom 5
fig_bottom = px.bar(
    df_bottom5.reset_index(),
    x='Región',
    y=indicador_seleccionado,
    text=indicador_seleccionado,
)
fig_bottom.update_traces(
    marker_color=df_bottom5['color'],
    texttemplate='%{text:.1f}%',
    textposition='outside'
)
fig_bottom.update_layout(
    title=f'🪫 Bottom 5 {indicador_seleccionado}',
    yaxis_title=f"{indicador_seleccionado} (%)",
    xaxis_title="Región",
    plot_bgcolor='rgba(0,0,0,0)',
    paper_bgcolor='rgba(0,0,0,0)',
    font=dict(color='white'),
    height=420,
)

# 🎨 Función para aplicar colores solo a la columna del indicador
def aplicar_color_tabla(df):
    def color_celda(valor):
        if valor > 89.99:
            return 'background-color: #7ED957; color: black;'
        elif valor >= 50:
            return 'background-color: #FFBD59; color: black;'
        else:
            return 'background-color: #FF5757; color: white;'
    styled = df.style.format("{:.1f}%")
    # Usa applymap para máxima compatibilidad
    styled = styled.map(color_celda, subset=[indicador_seleccionado])
    return styled

with st.container():

    col0_1, col0_2 = st.columns([7,12])
    with col0_1:
        st.subheader("🏆 Top 5 y Bottom 5 Regiones por Geoeficiencia")
        st.plotly_chart(
            fig_top,
            config={"responsive": True},
            use_container_width=True
        )
        st.plotly_chart(
            fig_bottom,
            config={"responsive": True},
            use_container_width=True
        )

    with col0_2:
        st.subheader(f"📊 Indicadores por Región — {indicador_seleccionado}")
        modo_tabla = st.radio(
            "📊 Ver tablas por:",
            options=["Jefatura", "Ruta"],
            horizontal=True,
            key="modo_tabla_selector"
        )

        regiones_unicas = sorted(df_mrk_trad["Región"].dropna().unique())
        regiones_top = regiones_unicas[:5]
        regiones_bottom = regiones_unicas[5:]

        # 🧾 Top regiones
        cols_top = st.columns(len(regiones_top))
        for i, region in enumerate(regiones_top):
            with cols_top[i]:
                df_region = df_mrk_trad[df_mrk_trad["Región"] == region]
                if df_region.empty:
                    st.info(f"⚠️ {region} sin datos")
                    continue
                df_tabla = (
                    df_region.pivot_table(
                        index=['Jefatura' if modo_tabla == "Jefatura" else 'Ruta'],
                        values=[indicador_seleccionado],
                        aggfunc='mean'
                    ) * 100
                ).sort_values(by=indicador_seleccionado, ascending=True)
                st.markdown(f"**{region}**")
                st.dataframe(aplicar_color_tabla(df_tabla), height=300, use_container_width=True)

        # 🧾 Bottom regiones
        cols_bottom = st.columns(len(regiones_bottom))
        for i, region in enumerate(regiones_bottom):
            with cols_bottom[i]:
                df_region = df_mrk_trad[df_mrk_trad["Región"] == region]
                if df_region.empty:
                    st.info(f"⚠️ {region} sin datos")
                    continue
                df_tabla = (
                    df_region.pivot_table(
                        index=['Jefatura' if modo_tabla == "Jefatura" else 'Ruta'],
                        values=[indicador_seleccionado],
                        aggfunc='mean'
                    ) * 100
                ).sort_values(by=indicador_seleccionado, ascending=True)
                st.markdown(f"**{region}**")
                st.dataframe(aplicar_color_tabla(df_tabla), height=300, use_container_width=True)

st.divider()