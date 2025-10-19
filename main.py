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

df_mrk = pd.read_parquet('bdd_mrk_cp.parquet')

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
st.divider()
st.subheader("")