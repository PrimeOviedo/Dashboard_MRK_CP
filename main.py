import streamlit as st
import pandas as pd

# 🔻 Ocultar "Built with Streamlit"
hide_streamlit_style = """
    <style>
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    .stDeployButton {display:none;}
    </style>
"""
st.markdown(hide_streamlit_style, unsafe_allow_html=True)

def agregar_columna_primer_cliente(df_tabla, df_region, modo):
    if "Primer cliente (hr)" in df_tabla.columns:
        df_tabla = df_tabla.drop(columns=["Primer cliente (hr)"])
    df_tabla_hora = (
        df_region.pivot_table(
            index=['Jefatura' if modo == "Jefatura" else 'Ruta'],
            values=['Primer cliente (hr)'],
            aggfunc='mean'
        )
    )
    df_tabla_hora['Primer cliente (hr)'] = df_tabla_hora['Primer cliente (hr)'].apply(
        lambda x: (pd.Timestamp('1900-01-01') + x).strftime('%H:%M:%S') if pd.notnull(x) else ''
    )
    return df_tabla.join(df_tabla_hora)

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

file_id = "1Pyr09HsOanL9c_e1sAf_Et6xl6zHFYKc"
#url = f"https://drive.google.com/uc?id={file_id}"

df_mrk = pd.read_parquet('bdd_mrk_cp.parquet')

#df_mrk = pd.read_parquet(url)

df_mrk['Jefatura'] = df_mrk['Jefatura'].astype(str).str.strip()
df_mrk = df_mrk[df_mrk['Jefatura'].notna() & (df_mrk['Jefatura'] != '')]

# Conversión de columna de hora a timedelta
df_mrk['Primer cliente (hr)'] = pd.to_timedelta(df_mrk['Primer cliente (hr)'])

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

    # Comparación de fechas
    activar_comparacion = st.checkbox("Activar comparación con otro rango de fechas")
    rango_comparativo = None
    if activar_comparacion:
        rango_comparativo = st.date_input(
            "Selecciona un rango comparativo:",
            value=(fecha_min, fecha_max),
            min_value=fecha_min,
            max_value=fecha_max,
            key="rango_comparativo"
        )

# --- Filtrado base y comparativo ---
if isinstance(rango_fechas, tuple) and len(rango_fechas) == 2:
    fecha_inicio, fecha_fin = rango_fechas
    df_base = df_mrk[
        (df_mrk["Fecha inicio"].dt.date >= fecha_inicio) &
        (df_mrk["Fecha inicio"].dt.date <= fecha_fin)
    ]
else:
    df_base = df_mrk.copy()

with col1_3:

    # 📌 Opción para incluir o excluir Farmer Comercial (solo afecta a Tradicional)
    farmer_option = st.radio(
        "👤 Filtrado de Farmer Comercial:",
        options=["Incluir todo", "Solo Farmer Comercial", "Excluir Farmer Comercial"],
        horizontal=True,
        key="farmer_option"
    )

# --- Filtrado base TRADICIONAL (sin moderno/mayoristas) ---
df_mrk_trad_base = df_mrk[~df_mrk['Región'].isin(['MODERNO', 'MAYORISTAS'])]

if farmer_option == "Solo Farmer Comercial":
    df_mrk_trad_base = df_mrk_trad_base[df_mrk_trad_base['Descripción Tipo'] == 'Farmer Comercial']
elif farmer_option == "Excluir Farmer Comercial":
    df_mrk_trad_base = df_mrk_trad_base[df_mrk_trad_base['Descripción Tipo'] != 'Farmer Comercial']

# --- Aplicar filtro de fechas a TRADICIONAL ---
if isinstance(rango_fechas, tuple) and len(rango_fechas) == 2:
    fecha_inicio, fecha_fin = rango_fechas
    df_mrk_trad = df_mrk_trad_base[
        (df_mrk_trad_base["Fecha inicio"].dt.date >= fecha_inicio) &
        (df_mrk_trad_base["Fecha inicio"].dt.date <= fecha_fin)
    ]
else:
    df_mrk_trad = df_mrk_trad_base.copy()

# --- MODERNO/MAYORISTAS (NO SE TOCA por Farmer Comercial) ---
if isinstance(rango_fechas, tuple) and len(rango_fechas) == 2:
    df_mrk_mod = df_mrk[
        (df_mrk["Fecha inicio"].dt.date >= fecha_inicio) &
        (df_mrk["Fecha inicio"].dt.date <= fecha_fin) &
        (df_mrk['Región'].isin(['MODERNO', 'MAYORISTAS']))
    ]
else:
    df_mrk_mod = df_mrk[df_mrk['Región'].isin(['MODERNO', 'MAYORISTAS'])]

df_comp = pd.DataFrame()
if activar_comparacion and rango_comparativo is not None and isinstance(rango_comparativo, tuple) and len(rango_comparativo) == 2:
    fecha_inicio_comp, fecha_fin_comp = rango_comparativo
    df_comp = df_mrk[
        (df_mrk["Fecha inicio"].dt.date >= fecha_inicio_comp) &
        (df_mrk["Fecha inicio"].dt.date <= fecha_fin_comp)
    ]

 # --- KPIs base ---
# Si estás incluyendo/excluyendo Farmer, usar df_mrk_trad
if farmer_option != "Incluir todas":
    df_zona = df_mrk_trad.pivot_table(
        index='Zona',
        values=['Geoeficiencia','Geoefectividad','Efectividad omnicanal','Primer cliente (%)','Tiempo de servicio (%)'],
        aggfunc='mean'
    )
else:
    df_zona = df_base.pivot_table(
        index='Zona',
        values=['Geoeficiencia','Geoefectividad','Efectividad omnicanal','Primer cliente (%)','Tiempo de servicio (%)'],
        aggfunc='mean'
    )

# Usar df_mrk_trad para tradicional (ya filtrado por Farmer Comercial)
df_regiones_trad = df_mrk_trad.pivot_table(
    index='Región',
    values=['Geoeficiencia','Geoefectividad','Efectividad omnicanal','Primer cliente (%)','Tiempo de servicio (%)'],
    aggfunc='mean'
) * 100

# Para MODERNO/MAYORISTAS, usar df_mrk_mod (no afectado por Farmer Comercial)
df_regiones_mod = df_mrk_mod.pivot_table(
    index='Región',
    values=['Geoeficiencia','Geoefectividad','Efectividad omnicanal','Primer cliente (%)','Tiempo de servicio (%)'],
    aggfunc='mean'
) * 100

# Diccionario de abreviaciones de KPIs (mover a global)
KPI_ABREV = {
    "Geoeficiencia": "Geoef.",
    "Geoefectividad": "Geoefec.",
    "Efectividad omnicanal": "Efect. Omn.",
    "Primer cliente (%)": "1er Cl.",
    "Tiempo de servicio (%)": "T. Serv."
}

# --- KPIs comparativo ---
df_zona_comp = df_regiones_trad_comp = df_regiones_mod_comp = pd.DataFrame()
df_zona_diff = df_regiones_trad_diff = df_regiones_mod_diff = pd.DataFrame()
if activar_comparacion and not df_comp.empty:
    df_zona_comp = df_comp.pivot_table(index='Zona', values=['Geoeficiencia','Geoefectividad','Efectividad omnicanal','Primer cliente (%)','Tiempo de servicio (%)'], aggfunc='mean')
    df_mrk_trad_comp = df_comp[~df_comp['Región'].isin(['MODERNO','MAYORISTAS'])]
    df_regiones_trad_comp = df_mrk_trad_comp.pivot_table(index='Región', values=['Geoeficiencia','Geoefectividad','Efectividad omnicanal','Primer cliente (%)','Tiempo de servicio (%)'], aggfunc='mean') * 100
    df_mrk_mod_comp = df_comp[df_comp['Región'].isin(['MODERNO','MAYORISTAS'])]
    df_regiones_mod_comp = df_mrk_mod_comp.pivot_table(index='Región', values=['Geoeficiencia','Geoefectividad','Efectividad omnicanal','Primer cliente (%)','Tiempo de servicio (%)'], aggfunc='mean') * 100
    # Diferencias
    if not df_zona_comp.empty:
        df_zona_diff = (df_zona * 100 - df_zona_comp * 100)
    if not df_regiones_trad_comp.empty:
        # Solo las regiones que están en ambos dataframes
        idx_inter = df_regiones_trad.index.intersection(df_regiones_trad_comp.index)
        df_regiones_trad_diff = df_regiones_trad.loc[idx_inter] - df_regiones_trad_comp.loc[idx_inter]
    if not df_regiones_mod_comp.empty:
        idx_inter_mod = df_regiones_mod.index.intersection(df_regiones_mod_comp.index)
        df_regiones_mod_diff = df_regiones_mod.loc[idx_inter_mod] - df_regiones_mod_comp.loc[idx_inter_mod]

st.subheader("🗺️ Indicadores a Total Zona")
col2_1, col2_2 = st.columns([5, 4])

with col2_1:

    def style_delta_columns(df, delta_cols):
        # Color verde para positivo, rojo para negativo, gris para cero. Solo color de texto, fondo transparente.
        def color_delta(val):
            if pd.isnull(val):
                return ''
            if val > 0:
                return 'color: #7ED957; background-color: transparent;'
            elif val < 0:
                return 'color: #FF5757; background-color: transparent;'
            else:
                return 'color: #aaa; background-color: transparent;'
        # Aplica formato a todas las columnas delta
        style_dict = {col: "{:+.1f}%" for col in delta_cols}
        styled = df.style.format({col: "{:.1f}%" for col in df.columns if col not in delta_cols} | style_dict)
        for col in delta_cols:
            styled = styled.map(color_delta, subset=[col])
        return styled

    def build_display_df(base_df, comp_df=None):
        # Si no hay comparación, solo formatear valores
        if comp_df is None or comp_df.empty:
            return base_df.copy(), []
        # Intersección de índices
        idx_inter = base_df.index.intersection(comp_df.index)
        base_df = base_df.loc[idx_inter].copy()
        comp_df = comp_df.loc[idx_inter].copy()
        display_df = base_df.copy()
        delta_cols = []
        # Calcular columnas delta y agregarlas a display_df (pero reordenar después)
        for col in base_df.columns:
            if col in comp_df.columns:
                # Usar abreviatura para el nombre de la columna delta
                col_abrev = KPI_ABREV.get(col, col)
                delta_col = f"Δ {col_abrev}"
                display_df[delta_col] = base_df[col] - comp_df[col]
                delta_cols.append(delta_col)
        # Reordenar: insertar cada columna delta justo después de su KPI
        reordered_df = pd.DataFrame(index=display_df.index)
        for col in base_df.columns:
            reordered_df[col] = display_df[col]
            col_abrev = KPI_ABREV.get(col, col)
            delta_col = f"Δ {col_abrev}"
            if delta_col in display_df.columns:
                reordered_df[delta_col] = display_df[delta_col]
        return reordered_df, delta_cols

    # --- Visualización consolidada para tradicional ---
    df_regiones_trad_display, trad_delta_cols = build_display_df(df_regiones_trad, df_regiones_trad_comp if activar_comparacion else None)
    st.markdown("**Regiones Tradicional**")
    st.dataframe(
        style_delta_columns(df_regiones_trad_display, trad_delta_cols),
        # width='stretch' removed per instructions
    )

    # --- Visualización consolidada para moderno ---
    df_regiones_mod_display, mod_delta_cols = build_display_df(df_regiones_mod, df_regiones_mod_comp if activar_comparacion else None)
    st.markdown("**Regiones Moderno/Mayoristas**")
    st.dataframe(
        style_delta_columns(df_regiones_mod_display, mod_delta_cols),
        # width='stretch' removed per instructions
    )

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
# Top/Bottom 5 sobre df_regiones_trad (ya filtrado por Farmer Comercial)
df_top5 = df_regiones_trad.sort_values(by=indicador_seleccionado, ascending=False).head(5)
df_bottom5 = df_regiones_trad.sort_values(by=indicador_seleccionado, ascending=True).head(5)

# Si comparación, calcular diferencias y agregar columna
if activar_comparacion and not df_regiones_trad_comp.empty:
    # Solo regiones presentes en ambos
    idx_inter_top = df_top5.index.intersection(df_regiones_trad_comp.index)
    idx_inter_top = idx_inter_top.intersection(df_top5.index)
    df_top5 = df_top5.reindex(idx_inter_top).dropna()
    df_top5_comp = df_regiones_trad_comp.loc[idx_inter_top]
    if not df_top5.empty and not df_top5_comp.empty:
        df_top5['Diferencia'] = df_top5[indicador_seleccionado] - df_top5_comp[indicador_seleccionado]

    idx_inter_bottom = df_bottom5.index.intersection(df_regiones_trad_comp.index)
    idx_inter_bottom = idx_inter_bottom.intersection(df_bottom5.index)
    df_bottom5 = df_bottom5.reindex(idx_inter_bottom).dropna()
    df_bottom5_comp = df_regiones_trad_comp.loc[idx_inter_bottom]
    if not df_bottom5.empty and not df_bottom5_comp.empty:
        df_bottom5['Diferencia'] = df_bottom5[indicador_seleccionado] - df_bottom5_comp[indicador_seleccionado]

# 🎨 Asignar color según el valor del indicador seleccionado
df_top5 = df_top5.assign(color=df_top5[indicador_seleccionado].apply(asignar_color))
df_bottom5 = df_bottom5.assign(color=df_bottom5[indicador_seleccionado].apply(asignar_color))

# 🧭 Indicador actual
indicador = indicador_seleccionado

# 📊 Agrupar por Región y Jefatura SOLO con el DF base filtrado (df_mrk_trad ya filtrado por Farmer Comercial)
df_jef = (
    df_mrk_trad[df_mrk_trad['Región'].isin(df_top5.index.tolist() + df_bottom5.index.tolist())]
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
    text=indicador_seleccionado if 'Diferencia' not in df_top5.columns else None,
)
if 'Diferencia' in df_top5.columns:
    # Etiqueta: valor (+/-diferencia)
    fig_top.update_traces(
        marker_color=df_top5['color'],
        text=[f"{row[indicador_seleccionado]:.1f}% ({row['Diferencia']:+.1f}%)" for idx, row in df_top5.iterrows()],
        texttemplate='%{text}',
        textposition='outside'
    )
else:
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
    text=indicador_seleccionado if 'Diferencia' not in df_bottom5.columns else None,
)
if 'Diferencia' in df_bottom5.columns:
    fig_bottom.update_traces(
        marker_color=df_bottom5['color'],
        text=[f"{row[indicador_seleccionado]:.1f}% ({row['Diferencia']:+.1f}%)" for idx, row in df_bottom5.iterrows()],
        texttemplate='%{text}',
        textposition='outside'
    )
else:
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



def style_tabla_ruta(df, delta_col):
    def color_celda(valor):
        if pd.isnull(valor):
            return ''

        # 🕒 Si el valor es un string con formato HH:MM:SS (Primer cliente (hr))
        if isinstance(valor, str) and ':' in valor:
            try:
                hora = pd.to_datetime(valor, format='%H:%M:%S').time()
                if hora > pd.to_datetime('09:00:00', format='%H:%M:%S').time():
                    return 'background-color: #FF5757; color: white;'  # Rojo si > 9AM
                else:
                    return 'background-color: #7ED957; color: black;'  # Verde si <= 9AM
            except Exception:
                return ''

        # 📊 Si el valor es numérico
        try:
            val = float(valor)
        except (ValueError, TypeError):
            return ''

        if val > 89.99:
            return 'background-color: #7ED957; color: black;'
        elif val >= 50:
            return 'background-color: #FFBD59; color: black;'
        else:
            return 'background-color: #FF5757; color: white;'

    def color_delta(val):
        if pd.isnull(val):
            return ''
        if val > 0:
            return 'color: #7ED957; background-color: transparent;'
        elif val < 0:
            return 'color: #FF5757; background-color: transparent;'
        else:
            return 'color: #aaa; background-color: transparent;'

    # Detectar todas las columnas KPI y Delta de la tabla
    kpi_cols = [c for c in df.columns if not c.startswith("Δ")]
    delta_cols = [c for c in df.columns if c.startswith("Δ")]

    # Excluir explícitamente la columna de horas del formateo numérico
    format_dict = {}
    for col in kpi_cols:
        if col != "Primer cliente (hr)":
            format_dict[col] = "{:.1f}%"
    for col in delta_cols:
        format_dict[col] = "{:+.1f}%"

    styled = df.style.format(format_dict)

    # Aplicar colores: fondo para KPIs y texto para Deltas
    if kpi_cols:
        styled = styled.map(color_celda, subset=kpi_cols)
    if delta_cols:
        styled = styled.map(color_delta, subset=delta_cols)

    return styled

with st.container():

    col0_1, col0_2 = st.columns([7,12])
    with col0_1:
        st.subheader(f"🏆 Top 5 y Bottom 5 Regiones por {indicador_seleccionado}")
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
                )
                # --- Añadir promedio de la columna 'Primer cliente (hr)' solo si se selecciona ese KPI ---
                if indicador_seleccionado == "Primer cliente (%)":
                    df_tabla_hora = (
                        df_region.pivot_table(
                            index=['Jefatura' if modo_tabla == "Jefatura" else 'Ruta'],
                            values=['Primer cliente (hr)'],
                            aggfunc='mean'
                        )
                    )
                    df_tabla_hora['Primer cliente (hr)'] = df_tabla_hora['Primer cliente (hr)'].apply(
                        lambda x: (pd.Timestamp('1900-01-01') + x).strftime('%H:%M:%S') if pd.notnull(x) else ''
                    )
                    df_tabla = df_tabla.join(df_tabla_hora)
                # --- Añadir promedio de la columna 'Primer cliente (hr)' solo si se selecciona ese KPI ---
                if indicador_seleccionado == "Primer cliente (%)":
                    df_tabla = agregar_columna_primer_cliente(df_tabla, df_region, modo_tabla)
                df_tabla = df_tabla.sort_values(by=indicador_seleccionado, ascending=True)
                delta_col = None
                # Soporte para delta cuando comparación está activa
                if activar_comparacion and rango_comparativo is not None and isinstance(rango_comparativo, tuple) and len(rango_comparativo) == 2:
                    # Nueva lógica: usar df_mrk_trad_comp si existe y está no vacía
                    if 'df_mrk_trad_comp' in globals() and not df_mrk_trad_comp.empty:
                        df_region_comp = df_mrk_trad_comp[df_mrk_trad_comp['Región'] == region]
                    else:
                        df_region_comp = pd.DataFrame()
                    if not df_region_comp.empty:
                        df_tabla_comp = (
                            df_region_comp.pivot_table(
                                index=['Jefatura' if modo_tabla == "Jefatura" else 'Ruta'],
                                values=[indicador_seleccionado],
                                aggfunc='mean'
                            ) * 100
                        )
                        # Intersección de índices, reindex para asegurar todos presentes y fillna(0)
                        idx_inter = df_tabla.index.intersection(df_tabla_comp.index)
                        df_tabla_int = df_tabla.reindex(idx_inter).fillna(0)
                        df_tabla_comp_int = df_tabla_comp.reindex(idx_inter).fillna(0)
                        if not df_tabla_int.empty and not df_tabla_comp_int.empty:
                            df_delta = df_tabla_int[indicador_seleccionado] - df_tabla_comp_int[indicador_seleccionado]
                            delta_col = f"Δ {KPI_ABREV[indicador_seleccionado]}"
                            # Insertar la columna delta justo después del indicador
                            df_tabla_int.insert(1, delta_col, df_delta)
                            st.markdown(f"**{region}**")
                            st.dataframe(
                                style_tabla_ruta(df_tabla_int, delta_col),
                                height=350
                            )
                            continue
                # Si no hay delta o comparación, mostrar tabla normal
                st.markdown(f"**{region}**")
                st.dataframe(
                    style_tabla_ruta(df_tabla, None),
                    height=350
                )

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
                )
                if indicador_seleccionado == "Primer cliente (%)":
                    df_tabla = agregar_columna_primer_cliente(df_tabla, df_region, modo_tabla)
                df_tabla = df_tabla.sort_values(by=indicador_seleccionado, ascending=True)
                delta_col = None
                # Soporte para delta cuando comparación está activa
                if activar_comparacion and rango_comparativo is not None and isinstance(rango_comparativo, tuple) and len(rango_comparativo) == 2:
                    # Nueva lógica: usar df_mrk_trad_comp si existe y está no vacía
                    if 'df_mrk_trad_comp' in globals() and not df_mrk_trad_comp.empty:
                        df_region_comp = df_mrk_trad_comp[df_mrk_trad_comp['Región'] == region]
                    else:
                        df_region_comp = pd.DataFrame()
                    if not df_region_comp.empty:
                        df_tabla_comp = (
                            df_region_comp.pivot_table(
                                index=['Jefatura' if modo_tabla == "Jefatura" else 'Ruta'],
                                values=[indicador_seleccionado],
                                aggfunc='mean'
                            ) * 100
                        )
                        # Intersección de índices, reindex para asegurar todos presentes y fillna(0)
                        idx_inter = df_tabla.index.intersection(df_tabla_comp.index)
                        df_tabla_int = df_tabla.reindex(idx_inter).fillna(0)
                        df_tabla_comp_int = df_tabla_comp.reindex(idx_inter).fillna(0)
                        if not df_tabla_int.empty and not df_tabla_comp_int.empty:
                            df_delta = df_tabla_int[indicador_seleccionado] - df_tabla_comp_int[indicador_seleccionado]
                            delta_col = f"Δ {KPI_ABREV[indicador_seleccionado]}"
                            # Insertar la columna delta justo después del indicador
                            df_tabla_int.insert(1, delta_col, df_delta)
                            st.markdown(f"**{region}**")
                            st.dataframe(
                                style_tabla_ruta(df_tabla_int, delta_col),
                                height=350
                            )
                            continue
                # Si no hay delta o comparación, mostrar tabla normal
                st.markdown(f"**{region}**")
                st.dataframe(
                    style_tabla_ruta(df_tabla, None),
                    height=350
                )

st.divider()

df_uo_mod = df_mrk_mod.pivot_table(
    index='Nombre UO',
    values=['Geoeficiencia','Geoefectividad','Efectividad omnicanal','Primer cliente (%)','Tiempo de servicio (%)'],
    aggfunc='mean'
) * 100


# 📊 Sección para MODERNO / MAYORISTAS
with st.container():
    # ------------------------------------------
    # FILTRO DE REGION MODERNO/MAYORISTAS
    # ------------------------------------------
    filtro_region_mod = st.radio(
        "Filtrar por:",
        options=["Todas", "Moderno", "Mayoristas"],
        horizontal=True,
        key="filtro_region_mod"
    )

    col_mod_1, col_mod_2 = st.columns([7, 12])

    # Aplicar filtro ANTES de calcular top/bottom y tablas
    df_mrk_mod_filtrado = df_mrk_mod.copy()
    if filtro_region_mod == "Moderno":
        df_mrk_mod_filtrado = df_mrk_mod_filtrado[df_mrk_mod_filtrado['Región'] == 'MODERNO']
    elif filtro_region_mod == "Mayoristas":
        df_mrk_mod_filtrado = df_mrk_mod_filtrado[df_mrk_mod_filtrado['Región'] == 'MAYORISTAS']
    # Si "Todas", no filtra

    # Recalcular tabla de UOs con el filtro aplicado
    df_uo_mod = df_mrk_mod_filtrado.pivot_table(
        index='Nombre UO',
        values=['Geoeficiencia','Geoefectividad','Efectividad omnicanal','Primer cliente (%)','Tiempo de servicio (%)'],
        aggfunc='mean'
    ) * 100

    # 🧮 Top y Bottom 5 MOD
    df_top5_mod = df_uo_mod.sort_values(by=indicador_seleccionado, ascending=False).head(5)
    df_bottom5_mod = df_uo_mod.sort_values(by=indicador_seleccionado, ascending=True).head(5)

    # 🔹 Asegurar que 'Nombre UO' sea columna antes de graficar
    df_top5_mod = df_top5_mod.reset_index().rename(columns={"index": "Nombre UO"})
    df_bottom5_mod = df_bottom5_mod.reset_index().rename(columns={"index": "Nombre UO"})

    # 🖌️ Colores según valor
    df_top5_mod = df_top5_mod.assign(color=df_top5_mod[indicador_seleccionado].apply(asignar_color))
    df_bottom5_mod = df_bottom5_mod.assign(color=df_bottom5_mod[indicador_seleccionado].apply(asignar_color))

    # --- Calcular diferencias si hay comparación activa ---
    if activar_comparacion and 'df_mrk_mod_comp' in globals() and not df_mrk_mod_comp.empty:
        df_comp_mod_uo = df_mrk_mod_comp.pivot_table(
            index='Nombre UO',
            values=[indicador_seleccionado],
            aggfunc='mean'
        ) * 100

        # ✅ Top 5 diferencia
        if not df_top5_mod.empty:
            idx_inter_top_mod = set(df_top5_mod['Nombre UO']).intersection(set(df_comp_mod_uo.index))
            df_top5_mod = df_top5_mod[df_top5_mod['Nombre UO'].isin(idx_inter_top_mod)]
            df_top5_mod = df_top5_mod.merge(
                df_comp_mod_uo[indicador_seleccionado],
                left_on='Nombre UO',
                right_index=True,
                suffixes=('', '_comp')
            )
            df_top5_mod['Diferencia'] = df_top5_mod[indicador_seleccionado] - df_top5_mod[
                f'{indicador_seleccionado}_comp']

        # ✅ Bottom 5 diferencia
        if not df_bottom5_mod.empty:
            idx_inter_bottom_mod = set(df_bottom5_mod['Nombre UO']).intersection(set(df_comp_mod_uo.index))
            df_bottom5_mod = df_bottom5_mod[df_bottom5_mod['Nombre UO'].isin(idx_inter_bottom_mod)]
            df_bottom5_mod = df_bottom5_mod.merge(
                df_comp_mod_uo[indicador_seleccionado],
                left_on='Nombre UO',
                right_index=True,
                suffixes=('', '_comp')
            )
            df_bottom5_mod['Diferencia'] = df_bottom5_mod[indicador_seleccionado] - df_bottom5_mod[
                f'{indicador_seleccionado}_comp']

    # 🥇 Top 5 MOD
    fig_top_mod = px.bar(
        df_top5_mod,
        x='Nombre UO',
        y=indicador_seleccionado,
        text=indicador_seleccionado if 'Diferencia' not in df_top5_mod.columns else None,
    )
    if 'Diferencia' in df_top5_mod.columns:
        fig_top_mod.update_traces(
            marker_color=df_top5_mod['color'],
            text=[f"{row[indicador_seleccionado]:.1f}% ({row['Diferencia']:+.1f}%)" for idx, row in
                  df_top5_mod.iterrows()],
            texttemplate='%{text}',
            textposition='outside'
        )
    else:
        fig_top_mod.update_traces(
            marker_color=df_top5_mod['color'],
            texttemplate='%{text:.1f}%',
            textposition='outside'
        )

    fig_top_mod.update_layout(
        title=f'🏆 Top 5 {indicador_seleccionado} {filtro_region_mod}',
        yaxis_title=f"{indicador_seleccionado} (%)",
        xaxis_title="Nombre UO",
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(color='white'),
        height=420,
    )

    # 🪫 Bottom 5 MOD
    fig_bottom_mod = px.bar(
        df_bottom5_mod,
        x='Nombre UO',
        y=indicador_seleccionado,
        text=indicador_seleccionado if 'Diferencia' not in df_bottom5_mod.columns else None,
    )
    if 'Diferencia' in df_bottom5_mod.columns:
        fig_bottom_mod.update_traces(
            marker_color=df_bottom5_mod['color'],
            text=[f"{row[indicador_seleccionado]:.1f}% ({row['Diferencia']:+.1f}%)" for idx, row in
                  df_bottom5_mod.iterrows()],
            texttemplate='%{text}',
            textposition='outside'
        )
    else:
        fig_bottom_mod.update_traces(
            marker_color=df_bottom5_mod['color'],
            texttemplate='%{text:.1f}%',
            textposition='outside'
        )

    fig_bottom_mod.update_layout(
        title=f'🪫 Bottom 5 {indicador_seleccionado} {filtro_region_mod}',
        yaxis_title=f"{indicador_seleccionado} (%)",
        xaxis_title="Nombre UO",
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(color='white'),
        height=420,
    )

    # -------------------------------
    # 📈 Gráficas Top & Bottom 5
    # -------------------------------
    with col_mod_1:
        st.subheader(f"🏆 Top 5 y Bottom 5 {filtro_region_mod} por {indicador_seleccionado}")
        st.plotly_chart(
            fig_top_mod,
            config={"responsive": True},
            use_container_width=True,
            key="fig_top_mod"
        )
        st.plotly_chart(
            fig_bottom_mod,
            config={"responsive": True},
            use_container_width=True,
            key="fig_bottom_mod"
        )

    # -------------------------------
    # 📊 Tablas por región
    # -------------------------------
    with col_mod_2:
        # 📊 Sección para MODERNO / MAYORISTAS — Tablas agrupadas por Nombre UO
        with st.container():
            st.subheader(f"📊 Indicadores por Región — {indicador_seleccionado} {filtro_region_mod}")
            modo_tabla_mod_uo = st.radio(
                "📊 Ver tablas por:",
                options=["Jefatura", "Ruta"],
                horizontal=True,
                key="modo_tabla_selector_mod_uo"
            )

            uos_unicas_mod = sorted(df_mrk_mod_filtrado["Nombre UO"].dropna().unique())
            uos_top_mod = uos_unicas_mod[:5]
            uos_bottom_mod = uos_unicas_mod[5:]

            # 🧾 Top UOs MOD
            if len(uos_top_mod) > 0:
                cols_top_mod = st.columns(len(uos_top_mod))
                for i, uo in enumerate(uos_top_mod):
                    with cols_top_mod[i]:
                        df_region = df_mrk_mod_filtrado[df_mrk_mod_filtrado["Nombre UO"] == uo]
                        if df_region.empty:
                            st.info(f"⚠️ {uo} sin datos")
                            continue
                        df_tabla = (
                                df_region.pivot_table(
                                    index=['Jefatura' if modo_tabla_mod_uo == "Jefatura" else 'Ruta'],
                                    values=[indicador_seleccionado],
                                    aggfunc='mean'
                                ) * 100
                        )
                        # --- Añadir promedio de la columna 'Primer cliente (hr)' solo si se selecciona ese KPI ---
                        if indicador_seleccionado == "Primer cliente (%)":
                            df_tabla = agregar_columna_primer_cliente(df_tabla, df_region, modo_tabla_mod_uo)
                        df_tabla = df_tabla.sort_values(by=indicador_seleccionado, ascending=True)

                        delta_col = None
                        # Soporte para delta cuando comparación está activa
                        if activar_comparacion and rango_comparativo is not None and isinstance(rango_comparativo,
                                                                                                tuple) and len(
                                rango_comparativo) == 2:
                            if 'df_mrk_mod_comp' in globals() and not df_mrk_mod_comp.empty:
                                df_region_comp = df_mrk_mod_comp[df_mrk_mod_comp['Nombre UO'] == uo]
                            else:
                                df_region_comp = pd.DataFrame()
                            if not df_region_comp.empty:
                                df_tabla_comp = (
                                        df_region_comp.pivot_table(
                                            index=['Jefatura' if modo_tabla_mod_uo == "Jefatura" else 'Ruta'],
                                            values=[indicador_seleccionado],
                                            aggfunc='mean'
                                        ) * 100
                                )
                                idx_inter = df_tabla.index.intersection(df_tabla_comp.index)
                                df_tabla_int = df_tabla.reindex(idx_inter).fillna(0)
                                df_tabla_comp_int = df_tabla_comp.reindex(idx_inter).fillna(0)
                                if not df_tabla_int.empty and not df_tabla_comp_int.empty:
                                    df_delta = df_tabla_int[indicador_seleccionado] - df_tabla_comp_int[
                                        indicador_seleccionado]
                                    delta_col = f"Δ {KPI_ABREV[indicador_seleccionado]}"
                                    df_tabla_int.insert(1, delta_col, df_delta)
                                    st.markdown(f"**{uo}**")
                                    st.dataframe(
                                        style_tabla_ruta(df_tabla_int, delta_col),
                                        height=350
                                    )
                                    continue
                        st.markdown(f"**{uo}**")
                        st.dataframe(
                            style_tabla_ruta(df_tabla, None),
                            height=350
                        )
            else:
                st.info("⚠️ No hay Regiones en el Top para esta categoría")

            # 🧾 Bottom UOs MOD
            if len(uos_bottom_mod) > 0:
                cols_bottom_mod = st.columns(len(uos_bottom_mod))
                for i, uo in enumerate(uos_bottom_mod):
                    with cols_bottom_mod[i]:
                        df_region = df_mrk_mod_filtrado[df_mrk_mod_filtrado["Nombre UO"] == uo]
                        if df_region.empty:
                            st.info(f"⚠️ {uo} sin datos")
                            continue
                        df_tabla = (
                                df_region.pivot_table(
                                    index=['Jefatura' if modo_tabla_mod_uo == "Jefatura" else 'Ruta'],
                                    values=[indicador_seleccionado],
                                    aggfunc='mean'
                                ) * 100
                        )
                        # --- Añadir promedio de la columna 'Primer cliente (hr)' solo si se selecciona ese KPI ---
                        if indicador_seleccionado == "Primer cliente (%)":
                            df_tabla = agregar_columna_primer_cliente(df_tabla, df_region, modo_tabla_mod_uo)
                        df_tabla = df_tabla.sort_values(by=indicador_seleccionado, ascending=True)

                        delta_col = None
                        if activar_comparacion and rango_comparativo is not None and isinstance(rango_comparativo,
                                                                                                tuple) and len(
                                rango_comparativo) == 2:
                            if 'df_mrk_mod_comp' in globals() and not df_mrk_mod_comp.empty:
                                df_region_comp = df_mrk_mod_comp[df_mrk_mod_comp['Nombre UO'] == uo]
                            else:
                                df_region_comp = pd.DataFrame()
                            if not df_region_comp.empty:
                                df_tabla_comp = (
                                        df_region_comp.pivot_table(
                                            index=['Jefatura' if modo_tabla_mod_uo == "Jefatura" else 'Ruta'],
                                            values=[indicador_seleccionado],
                                            aggfunc='mean'
                                        ) * 100
                                )
                                idx_inter = df_tabla.index.intersection(df_tabla_comp.index)
                                df_tabla_int = df_tabla.reindex(idx_inter).fillna(0)
                                df_tabla_comp_int = df_tabla_comp.reindex(idx_inter).fillna(0)
                                if not df_tabla_int.empty and not df_tabla_comp_int.empty:
                                    df_delta = df_tabla_int[indicador_seleccionado] - df_tabla_comp_int[
                                        indicador_seleccionado]
                                    delta_col = f"Δ {KPI_ABREV[indicador_seleccionado]}"
                                    df_tabla_int.insert(1, delta_col, df_delta)
                                    st.markdown(f"**{uo}**")
                                    st.dataframe(
                                        style_tabla_ruta(df_tabla_int, delta_col),
                                        height=350
                                    )
                                    continue
                        st.markdown(f"**{uo}**")
                        st.dataframe(
                            style_tabla_ruta(df_tabla, None),
                            height=350
                        )
            else:
                st.info("⚠️ No hay Regiones en el Bottom para esta categoría")
# =====================================================================
# 📍 Indicadores por GEC (v3)
#   - Gráficas y tabla principal: SIEMPRE por Región
#   - Debajo: selector de Región + radio (Jefatura/Ruta) para la segunda tabla
#   - Respeta filtros globales (Farmer, GEC, rangos, comparativo)
# =====================================================================
st.divider()
st.subheader("📍 Indicadores por GEC")

# --- Utilidad: filtrar por rango (debe existir ANTES de usarse) ---
def filtrar_por_rango(df, rango):
    if isinstance(rango, tuple) and len(rango) == 2:
        f_ini, f_fin = rango
        return df[(df["Fecha inicio"].dt.date >= f_ini) & (df["Fecha inicio"].dt.date <= f_fin)]
    return df

# --- Cargar parquet y asegurar tipos ---
df_gec = pd.read_parquet("bdd_mrk_cp_gec.parquet")

# Tipos consistentes (evita warnings)
df_gec["Fecha inicio"] = pd.to_datetime(df_gec["Fecha inicio"], format="%Y-%m-%d", errors="coerce")
df_gec["Hora de llegada"] = pd.to_datetime(df_gec["Hora de llegada"], format="%H:%M:%S", errors="coerce")
df_gec["Tiempo de atencion"] = pd.to_timedelta(df_gec["Tiempo de atencion"], errors="coerce")

col_f1, col_f2, col_f3 = st.columns([2,1,5])

with col_f1:
    # --- Filtro de GEC ---
    gecs_disponibles = sorted(df_gec["GEC"].dropna().unique())
    gec_seleccionado = st.selectbox(
        "Selecciona un GEC:",
        options=gecs_disponibles,
        key="filtro_gec"
    )

with col_f2:
    st.subheader("")

with col_f3:

    # --- Filtro Farmer ---
    farmer_option = st.radio(
        "Filtrar rutas Farmer:",
        options=["Incluir todo", "Solo Farmer Comercial", "Excluir Farmer Comercial"],
        horizontal=True,
        key="filtro_farmer_gec"
    )
    if farmer_option == "Solo Farmer Comercial":
        df_gec = df_gec[df_gec["Descripción Tipo"] == "Farmer Comercial"]
    elif farmer_option == "Excluir Farmer Comercial":
        df_gec = df_gec[df_gec["Descripción Tipo"] != "Farmer Comercial"]

# Evita errores si no hay GEC seleccionado
if not gec_seleccionado:
    st.warning("⚠️ Por favor selecciona un GEC para continuar.")
    st.stop()

df_gec = df_gec[df_gec["GEC"] == gec_seleccionado]

# --- Aplicar rangos ---
df_gec_base = filtrar_por_rango(df_gec, rango_fechas)
df_gec_comp = filtrar_por_rango(df_gec, rango_comparativo) if activar_comparacion else pd.DataFrame()

# --- Promedio de hora válida (05:00–23:00) ---
def hora_promedio(series):
    horas = series if pd.api.types.is_datetime64_any_dtype(series) else pd.to_datetime(series, errors="coerce")
    horas_validas = horas[(horas.dt.hour >= 5) & (horas.dt.hour <= 23)]
    if horas_validas.empty:
        return pd.NaT
    segundos = (horas_validas.dt.hour * 3600 + horas_validas.dt.minute * 60 + horas_validas.dt.second).mean()
    return pd.to_datetime("1900-01-01") + pd.to_timedelta(segundos, unit="s")

# --- Calcular indicadores genérico ---
def calcular_indicadores(df, columna_grupo):
    if df.empty:
        return pd.DataFrame()
    g = (
        df.groupby(columna_grupo)
          .agg({
              "Geoeficiencia": "sum",
              "Geoefectividad": "sum",
              "Pedido Omnicanal": "sum",
              "Id cliente": "sum",
              "Tiempo de atencion": "mean",
              "Hora de llegada": hora_promedio,
          })
          .reset_index()
    )
    g["Geoeficiencia (%)"] = (g["Geoeficiencia"] / g["Id cliente"]) * 100
    g["Geoefectividad (%)"] = (g["Geoefectividad"] / g["Id cliente"]) * 100
    g["Pedido Omnicanal (%)"] = (g["Pedido Omnicanal"] / g["Id cliente"]) * 100
    g["Hora de llegada"] = g["Hora de llegada"].dt.strftime("%H:%M:%S")
    g["Tiempo de atencion"] = g["Tiempo de atencion"].apply(
        lambda x: str(x).split(" days ")[-1].split(".")[0] if pd.notnull(x) else None
    )
    g.rename(columns={columna_grupo: "Nivel"}, inplace=True)
    return g[["Nivel", "Geoeficiencia (%)", "Geoefectividad (%)", "Pedido Omnicanal (%)", "Tiempo de atencion", "Hora de llegada"]]

# ================
# 1) Región (fijo)
# ================
REGION_COL = "Región Comercial_Act 2026"
df_ind = calcular_indicadores(df_gec_base, REGION_COL)
df_ind_comp = calcular_indicadores(df_gec_comp, REGION_COL) if not df_gec_comp.empty else pd.DataFrame()

# --- Comparativos (Δ) para la tabla principal por Región ---
df_diff = pd.DataFrame()
if activar_comparacion and not df_ind_comp.empty and not df_ind.empty:
    comunes = df_ind["Nivel"].isin(df_ind_comp["Nivel"])
    base = df_ind[comunes].set_index("Nivel")
    comp = df_ind_comp.set_index("Nivel")
    df_diff = base.copy()
    for col in ["Geoeficiencia (%)", "Geoefectividad (%)", "Pedido Omnicanal (%)"]:
        # Abreviar solo en los Δ para que no se alargue
        abrev = {
            "Geoeficiencia (%)": "Geoefic.",
            "Geoefectividad (%)": "Geoefect.",
            "Pedido Omnicanal (%)": "Omnicanal",
        }[col]
        df_diff[f"Δ {abrev}"] = (base[col] - comp[col]).round(1)

    # Δ de tiempo de atención (string con flechas)
    base_t = pd.to_timedelta(base["Tiempo de atencion"], errors="coerce")
    comp_t = pd.to_timedelta(comp["Tiempo de atencion"], errors="coerce")
    delta_seg = (base_t - comp_t).dt.total_seconds()

    def formatear_delta_tiempo(seg):
        if pd.isna(seg): return ""
        signo = "↑ +" if seg > 0 else "↓ -" if seg < 0 else "= "
        seg = abs(int(seg))
        m, s = divmod(seg, 60)
        h, m = divmod(m, 60)
        if h > 0: return f"{signo}{h} h {m} m"
        elif m > 0: return f"{signo}{m} m {s} s"
        else: return f"{signo}{s} s"

    df_diff["Δ Tiempo"] = delta_seg.apply(formatear_delta_tiempo)
    df_diff = df_diff.reset_index()

# --- Estilo de color (se aplica a columnas Δ) ---
def color_delta(val):
    if pd.isnull(val) or val == "":
        return ""

    # 🟢🔴 Si es numérico puro
    if isinstance(val, (float, int)):
        if val > 0:
            return "color: #7ED957; font-weight: bold;"
        elif val < 0:
            return "color: #FF5757; font-weight: bold;"
        else:
            return "color: #aaa;"

    # 🟢🔴 Si es texto formateado con signo (+ / -)
    if isinstance(val, str):
        if any(s in val for s in ["+", "↑"]):
            return "color: #7ED957; font-weight: bold;"
        elif any(s in val for s in ["-", "↓"]):
            return "color: #FF5757; font-weight: bold;"
        else:
            return "color: #aaa;"

    return ""

# ============================
#  Top / Bottom 5 por Región
# ============================
col_top, col_table = st.columns([7, 12])
with col_top:
    st.subheader("Top y Bottom por GEC - Geoefectividad")
    import plotly.express as px
    df_top5 = df_ind.sort_values("Geoefectividad (%)", ascending=False).head(5)
    df_bottom5 = df_ind.sort_values("Geoefectividad (%)", ascending=True).head(5)

    def etiqueta(row):
        base_val = f"{row['Geoefectividad (%)']:.1f}%"
        if activar_comparacion and not df_ind_comp.empty and row["Nivel"] in df_ind_comp["Nivel"].values:
            delta = row["Geoefectividad (%)"] - df_ind_comp.set_index("Nivel").loc[row["Nivel"], "Geoefectividad (%)"]
            return f"{base_val} ({delta:+.1f}%)"
        return base_val

    fig_top = px.bar(
        df_top5, x="Nivel", y="Geoefectividad (%)",
        text=[etiqueta(r) for _, r in df_top5.iterrows()],
        color_discrete_sequence=["#7ED957"]
    )
    fig_bottom = px.bar(
        df_bottom5, x="Nivel", y="Geoefectividad (%)",
        text=[etiqueta(r) for _, r in df_bottom5.iterrows()],
        color_discrete_sequence=["#FF5757"]
    )
    for fig in [fig_top, fig_bottom]:
        fig.update_traces(textposition="outside")
        fig.update_layout(
            font=dict(color="white"), plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)", height=430,
        )
    fig_top.update_layout(title="🏆 Top 5")
    fig_bottom.update_layout(title="🪫 Bottom 5")
    st.plotly_chart(fig_top, config={"responsive": True, "displayModeBar": False}, use_container_width=True)
    st.plotly_chart(fig_bottom, config={"responsive": True, "displayModeBar": False}, use_container_width=True)

# ============================
#  Tabla principal por Región
# ============================
with col_table:
    st.markdown("#### 📊 Tabla comparativa por Región")
    if not df_diff.empty:
        # Reordenar: KPI seguido de su Δ abreviado
        columnas = ["Nivel",
                    "Geoeficiencia (%)", "Δ Geoefic.",
                    "Geoefectividad (%)", "Δ Geoefect.",
                    "Pedido Omnicanal (%)", "Δ Omnicanal",
                    "Tiempo de atencion", "Δ Tiempo",
                    "Hora de llegada"]
        columnas = [c for c in columnas if c in df_diff.columns or c == "Nivel"]
        df_display = df_diff[columnas].copy()
        # Mantener Δ como numérico; formateo y color con Styler
        delta_cols = [c for c in df_display.columns if c in ["Δ Geoefic.", "Δ Geoefect.", "Δ Omnicanal", "Δ Tiempo"]]
        styler = df_display.style.format({
            "Geoeficiencia (%)": "{:.1f}%",
            "Geoefectividad (%)": "{:.1f}%",
            "Pedido Omnicanal (%)": "{:.1f}%",
            **{c: "{:+.1f} %" for c in df_display.columns if c in ["Δ Geoefic.", "Δ Geoefect.", "Δ Omnicanal"]}
        }).map(color_delta, subset=delta_cols)
        st.dataframe(
            styler,
            width='stretch', hide_index=True,
            column_config={"Nivel": st.column_config.TextColumn("Región", pinned="left")}
        )
    else:
        st.dataframe(
            df_ind.style.format({
                "Geoeficiencia (%)": "{:.1f}%",
                "Geoefectividad (%)": "{:.1f}%",
                "Pedido Omnicanal (%)": "{:.1f}%",
            }),
            width='stretch', hide_index=True,
            column_config={"Nivel": st.column_config.TextColumn("Región", pinned="left")}
        )

    # =============================================================
    # 2) Detalle por Región seleccionada → radio Jefatura / Ruta
    # =============================================================
    st.markdown(" ")

    regiones_disp = sorted(df_gec_base[REGION_COL].dropna().unique().tolist())
    if len(regiones_disp) == 0:
        st.info("⚠️ No hay regiones disponibles con los filtros actuales.")
    else:
        region_sel = st.selectbox(
            "Elige la región a detallar:", options=regiones_disp, key="detalle_region"
        )

        vista_detalle = st.radio(
            "📊 Ver detalle por:", options=["Jefatura", "Ruta"], horizontal=True, key="modo_detalle_selector"
        )

        col_jef_exist = "Jefatura" if "Jefatura" in df_gec_base.columns else ("Jefatura_y" if "Jefatura_y" in df_gec_base.columns else None)
        if vista_detalle == "Jefatura" and col_jef_exist is None:
            st.error("No se encontró la columna de Jefatura en el dataset.")
        else:
            col_agr = col_jef_exist if vista_detalle == "Jefatura" else "Ruta"

            # Filtrar región
            df_reg_base = df_gec_base[df_gec_base[REGION_COL] == region_sel].copy()
            df_reg_comp = df_gec_comp[df_gec_comp[REGION_COL] == region_sel].copy() if activar_comparacion and not df_gec_comp.empty else pd.DataFrame()

            base_det = calcular_indicadores(df_reg_base, col_agr)
            comp_det = calcular_indicadores(df_reg_comp, col_agr) if not df_reg_comp.empty else pd.DataFrame()

            # Comparativos para el detalle
            df_det_diff = pd.DataFrame()
            if activar_comparacion and not comp_det.empty and not base_det.empty:
                comunes = base_det["Nivel"].isin(comp_det["Nivel"])
                b = base_det[comunes].set_index("Nivel")
                c = comp_det.set_index("Nivel")
                df_det_diff = b.copy()
                # % (abreviadas en Δ)
                abrev_map = {"Geoeficiencia (%)": "Geoefic.", "Geoefectividad (%)": "Geoefect.", "Pedido Omnicanal (%)": "Omnicanal"}
                for col in abrev_map:
                    if col in b.columns and col in c.columns:
                        df_det_diff[f"Δ {abrev_map[col]}"] = (b[col] - c[col]).round(1)
                # Tiempo
                bt = pd.to_timedelta(b["Tiempo de atencion"], errors="coerce")
                ct = pd.to_timedelta(c["Tiempo de atencion"], errors="coerce")
                dseg = (bt - ct).dt.total_seconds()
                df_det_diff["Δ Tiempo"] = dseg.apply(lambda seg: (
                    "" if pd.isna(seg) else (
                        ("↑ +" if seg > 0 else ("↓ -" if seg < 0 else "= ")) + (
                            (lambda s: (f"{s//3600} h {(s%3600)//60} m" if s>=3600 else (f"{(s//60)} m {s%60} s" if s>=60 else f"{s} s")))(abs(int(seg)))
                        )
                    )
                ))
                df_det_diff = df_det_diff.reset_index()

            st.markdown(f"#### 📊 Detalle por {vista_detalle} — {region_sel}")

            if not df_det_diff.empty:
                cols = ["Nivel",
                        "Geoeficiencia (%)", "Δ Geoefic.",
                        "Geoefectividad (%)", "Δ Geoefect.",
                        "Pedido Omnicanal (%)", "Δ Omnicanal",
                        "Tiempo de atencion", "Δ Tiempo",
                        "Hora de llegada"]
                cols = [c for c in cols if c in df_det_diff.columns or c == "Nivel"]
                show_df = df_det_diff[cols].copy()
                # Mantener Δ como numérico; formateo y color con Styler
                delta_cols_det = [c for c in show_df.columns if c.startswith("Δ")]
                styler_det = show_df.style.format({
                    "Geoeficiencia (%)": "{:.1f}%",
                    "Geoefectividad (%)": "{:.1f}%",
                    "Pedido Omnicanal (%)": "{:.1f}%",
                    **{c: "{:+.1f} %" for c in show_df.columns if c in ["Δ Geoefic.", "Δ Geoefect.", "Δ Omnicanal"]}
                }).map(color_delta, subset=delta_cols_det)
                st.dataframe(
                    styler_det,
                    width='stretch', hide_index=True,
                    column_config={"Nivel": st.column_config.TextColumn(vista_detalle, pinned="left")}
                )
            else:
                st.dataframe(
                    base_det.style.format({
                        "Geoeficiencia (%)": "{:.1f}%",
                        "Geoefectividad (%)": "{:.1f}%",
                        "Pedido Omnicanal (%)": "{:.1f}%",
                    }),
                    width='stretch',height=250, hide_index=True,
                    column_config={"Nivel": st.column_config.TextColumn(vista_detalle, pinned="left")}
                )
# =====================================================================
# 📱 Indicadores Clientes con Recurrencia Digital
# =====================================================================
st.divider()
st.subheader("📱 Indicadores Clientes con Recurrencia Digital")

# --- Cargar parquet ---
df_class = pd.read_parquet("bdd_mrk_cp_class.parquet")

# --- Filtro Farmer Comercial (sección Recurrencia Digital) ---
col_class_f1, col_class_f2 = st.columns([2, 10])
with col_class_f1:
    filtro_farmer_class = st.radio(
        "Filtrar rutas Farmer:",
        options=["Incluir todo", "Solo Farmer Comercial", "Excluir Farmer Comercial"],
        horizontal=True,
        key="filtro_farmer_class"
    )

# --- Filtrar df_class según opción seleccionada ---
if filtro_farmer_class == "Solo Farmer Comercial":
    df_class = df_class[df_class["Descripción Tipo"] == "Farmer Comercial"]
elif filtro_farmer_class == "Excluir Farmer Comercial":
    df_class = df_class[df_class["Descripción Tipo"] != "Farmer Comercial"]
# Si "Incluir todo", no filtra

# --- Asegurar tipos correctos ---
df_class["Fecha inicio"] = pd.to_datetime(df_class["Fecha inicio"], errors="coerce")
df_class["Hora de llegada"] = pd.to_datetime(df_class["Hora de llegada"], format="%H:%M:%S", errors="coerce")
df_class["Tiempo de atencion"] = pd.to_timedelta(df_class["Tiempo de atencion"], errors="coerce")

# --- Aplicar filtros globales de fechas ---
df_class_base = filtrar_por_rango(df_class, rango_fechas)
df_class_comp = filtrar_por_rango(df_class, rango_comparativo) if activar_comparacion else pd.DataFrame()

# --- Calcular indicadores principales ---
REGION_COL = "Región Comercial_Act 2026"
df_class_ind = calcular_indicadores(df_class_base, REGION_COL)
df_class_ind_comp = calcular_indicadores(df_class_comp, REGION_COL) if not df_class_comp.empty else pd.DataFrame()

# --- Comparativos ---
df_class_diff = pd.DataFrame()
if activar_comparacion and not df_class_ind_comp.empty and not df_class_ind.empty:
    comunes = df_class_ind["Nivel"].isin(df_class_ind_comp["Nivel"])
    base = df_class_ind[comunes].set_index("Nivel")
    comp = df_class_ind_comp.set_index("Nivel")
    df_class_diff = base.copy()
    for col in ["Geoeficiencia (%)", "Geoefectividad (%)", "Pedido Omnicanal (%)"]:
        abrev = {
            "Geoeficiencia (%)": "Geoefic.",
            "Geoefectividad (%)": "Geoefect.",
            "Pedido Omnicanal (%)": "Omnicanal",
        }[col]
        df_class_diff[f"Δ {abrev}"] = (base[col] - comp[col]).round(1)

    base_t = pd.to_timedelta(base["Tiempo de atencion"], errors="coerce")
    comp_t = pd.to_timedelta(comp["Tiempo de atencion"], errors="coerce")
    delta_seg = (base_t - comp_t).dt.total_seconds()

    def formatear_delta_tiempo(seg):
        if pd.isna(seg): return ""
        signo = "↑ +" if seg > 0 else "↓ -" if seg < 0 else "= "
        seg = abs(int(seg))
        m, s = divmod(seg, 60)
        h, m = divmod(m, 60)
        if h > 0: return f"{signo}{h} h {m} m"
        elif m > 0: return f"{signo}{m} m {s} s"
        else: return f"{signo}{s} s"

    df_class_diff["Δ Tiempo"] = delta_seg.apply(formatear_delta_tiempo)
    df_class_diff = df_class_diff.reset_index()

# =============================
#  Tabla principal por Región
# =============================
st.markdown("#### 📊 Tabla comparativa por Región")
if not df_class_diff.empty:
    columnas = ["Nivel",
                "Geoeficiencia (%)", "Δ Geoefic.",
                "Geoefectividad (%)", "Δ Geoefect.",
                "Pedido Omnicanal (%)", "Δ Omnicanal",
                "Tiempo de atencion", "Δ Tiempo",
                "Hora de llegada"]
    columnas = [c for c in columnas if c in df_class_diff.columns or c == "Nivel"]
    df_display = df_class_diff[columnas].copy()
    delta_cols = [c for c in df_display.columns if c.startswith("Δ")]
    styler = df_display.style.format({
        "Geoeficiencia (%)": "{:.1f}%",
        "Geoefectividad (%)": "{:.1f}%",
        "Pedido Omnicanal (%)": "{:.1f}%",
        **{c: "{:+.1f} %" for c in df_display.columns if c in ["Δ Geoefic.", "Δ Geoefect.", "Δ Omnicanal"]}
    }).map(color_delta, subset=delta_cols)
    st.dataframe(
        styler,
        width='stretch', hide_index=True,
        column_config={"Nivel": st.column_config.TextColumn("Región", pinned="left")}
    )
else:
    st.dataframe(
        df_class_ind.style.format({
            "Geoeficiencia (%)": "{:.1f}%",
            "Geoefectividad (%)": "{:.1f}%",
            "Pedido Omnicanal (%)": "{:.1f}%",
        }),
        width='stretch', hide_index=True,
        column_config={"Nivel": st.column_config.TextColumn("Región", pinned="left")}
    )

# ========================================
#  Detalle inferior — Jefatura o Ruta
# ========================================
st.markdown(" ")

regiones_disp = sorted(df_class_base[REGION_COL].dropna().unique().tolist())
if len(regiones_disp) == 0:
    st.info("⚠️ No hay regiones disponibles con los filtros actuales.")
else:
    region_sel = st.selectbox(
        "Elige la región a detallar:", options=regiones_disp, key="detalle_region_class"
    )

    vista_detalle = st.radio(
        "📊 Ver detalle por:", options=["Jefatura", "Ruta"], horizontal=True, key="modo_detalle_class"
    )

    col_jef_exist = "Jefatura" if "Jefatura" in df_class_base.columns else ("Jefatura_y" if "Jefatura_y" in df_class_base.columns else None)
    if vista_detalle == "Jefatura" and col_jef_exist is None:
        st.error("No se encontró la columna de Jefatura en el dataset.")
    else:
        col_agr = col_jef_exist if vista_detalle == "Jefatura" else "Ruta"

        df_reg_base = df_class_base[df_class_base[REGION_COL] == region_sel].copy()
        df_reg_comp = df_class_comp[df_class_comp[REGION_COL] == region_sel].copy() if activar_comparacion and not df_class_comp.empty else pd.DataFrame()

        base_det = calcular_indicadores(df_reg_base, col_agr)
        comp_det = calcular_indicadores(df_reg_comp, col_agr) if not df_reg_comp.empty else pd.DataFrame()

        df_det_diff = pd.DataFrame()
        if activar_comparacion and not comp_det.empty and not base_det.empty:
            comunes = base_det["Nivel"].isin(comp_det["Nivel"])
            b = base_det[comunes].set_index("Nivel")
            c = comp_det.set_index("Nivel")
            df_det_diff = b.copy()
            abrev_map = {"Geoeficiencia (%)": "Geoefic.", "Geoefectividad (%)": "Geoefect.", "Pedido Omnicanal (%)": "Omnicanal"}
            for col in abrev_map:
                if col in b.columns and col in c.columns:
                    df_det_diff[f"Δ {abrev_map[col]}"] = (b[col] - c[col]).round(1)
            bt = pd.to_timedelta(b["Tiempo de atencion"], errors="coerce")
            ct = pd.to_timedelta(c["Tiempo de atencion"], errors="coerce")
            dseg = (bt - ct).dt.total_seconds()
            df_det_diff["Δ Tiempo"] = dseg.apply(lambda seg: (
                "" if pd.isna(seg) else (
                    ("↑ +" if seg > 0 else ("↓ -" if seg < 0 else "= ")) + (
                        (lambda s: (f"{s//3600} h {(s%3600)//60} m" if s>=3600 else (f"{(s//60)} m {s%60} s" if s>=60 else f"{s} s")))(abs(int(seg)))
                    )
                )
            ))
            df_det_diff = df_det_diff.reset_index()

        st.markdown(f"#### 📊 Detalle por {vista_detalle} — {region_sel}")

        if not df_det_diff.empty:
            cols = ["Nivel",
                    "Geoeficiencia (%)", "Δ Geoefic.",
                    "Geoefectividad (%)", "Δ Geoefect.",
                    "Pedido Omnicanal (%)", "Δ Omnicanal",
                    "Tiempo de atencion", "Δ Tiempo",
                    "Hora de llegada"]
            cols = [c for c in cols if c in df_det_diff.columns or c == "Nivel"]
            show_df = df_det_diff[cols].copy()
            delta_cols_det = [c for c in show_df.columns if c.startswith("Δ")]
            styler_det = show_df.style.format({
                "Geoeficiencia (%)": "{:.1f}%",
                "Geoefectividad (%)": "{:.1f}%",
                "Pedido Omnicanal (%)": "{:.1f}%",
                **{c: "{:+.1f} %" for c in show_df.columns if c in ["Δ Geoefic.", "Δ Geoefect.", "Δ Omnicanal"]}
            }).map(color_delta, subset=delta_cols_det)
            st.dataframe(
                styler_det,
                width='stretch', hide_index=True,
                column_config={"Nivel": st.column_config.TextColumn(vista_detalle, pinned="left")}
            )
        else:
            st.dataframe(
                base_det.style.format({
                    "Geoeficiencia (%)": "{:.1f}%",
                    "Geoefectividad (%)": "{:.1f}%",
                    "Pedido Omnicanal (%)": "{:.1f}%",
                }),
                width='stretch', height=250, hide_index=True,
                column_config={"Nivel": st.column_config.TextColumn(vista_detalle, pinned="left")}
            )