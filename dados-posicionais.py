import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import numpy as np
from io import StringIO
import warnings
warnings.filterwarnings('ignore')

# Configuração da página
st.set_page_config(
    page_title="Análise de Percurso do Atleta",
    page_icon="🏃",
    layout="wide"
)

# Título do app
st.title("🏃 Análise de Percurso do Atleta durante o Jogo")
st.markdown("---")

# ==================== FUNÇÕES OTIMIZADAS DE ENTROPIA ====================

@st.cache_data(ttl=3600)
def compute_entropy_fast(data, bins=30):
    """Entropia de Shannon otimizada"""
    hist, _ = np.histogram(data, bins=bins, density=True)
    hist = hist[hist > 0]
    if len(hist) == 0:
        return 0
    # Cálculo direto sem scipy.stats.entropy
    return -np.sum(hist * np.log2(hist))

@st.cache_data(ttl=3600)
def sample_entropy_fast(data, m=2, r=0.2):
    """
    Entropia Amostral otimizada (Sample Entropy)
    Versão mais rápida para séries grandes
    """
    data = np.asarray(data, dtype=np.float64)
    N = len(data)
    
    if N < m + 1:
        return np.nan
    
    # Normalizar pelo desvio padrão
    std_data = np.std(data)
    if std_data == 0:
        return 0
    r = r * std_data
    
    # Função para contar matches de forma otimizada
    def _count_matches(m):
        count = 0
        total = 0
        for i in range(N - m):
            template = data[i:i+m]
            # Calcular distâncias máximas de forma vetorizada
            for j in range(i + 1, N - m + 1):
                diff = np.abs(template - data[j:j+m])
                if np.max(diff) <= r:
                    count += 1
                total += 1
        return count / total if total > 0 else 0
    
    phi_m = _count_matches(m)
    phi_m1 = _count_matches(m + 1)
    
    if phi_m == 0 or phi_m1 == 0:
        return 0
    
    return -np.log(phi_m1 / phi_m)

@st.cache_data(ttl=3600)
def rolling_entropy_fast(data, window_size=50, step=10, m=2, r=0.2):
    """
    Entropia por janela deslizante otimizada
    """
    N = len(data)
    entropies = []
    positions = []
    
    # Usar apenas algumas janelas para acelerar
    num_windows = min(30, (N - window_size) // step + 1)
    
    for i in range(0, N - window_size + 1, max(step, (N - window_size) // num_windows)):
        window = data[i:i+window_size]
        if len(window) > 10 and np.std(window) > 0.01:
            ent = sample_entropy_fast(window, m=m, r=r)
            if not np.isnan(ent):
                entropies.append(ent)
                positions.append(i + window_size // 2)
    
    return np.array(positions), np.array(entropies)

# ==================== FUNÇÃO DE CARREGAMENTO DE DADOS ====================

@st.cache_data
def load_data(uploaded_file):
    """Carrega e processa os dados do arquivo CSV enviado"""
    try:
        content = uploaded_file.getvalue().decode('utf-8')
        df = pd.read_csv(StringIO(content), skiprows=7, sep=';')
        
        df.columns = ['Timestamp', 'Seconds', 'Velocity', 'Acceleration', 'Odometer', 
                      'Latitude', 'Longitude', 'HeartRate', 'PlayerLoad', 
                      'PositionalQuality', 'HDOP', 'Sats']
        
        numeric_cols = ['Seconds', 'Velocity', 'Acceleration', 'Odometer', 
                        'Latitude', 'Longitude', 'HeartRate', 'PlayerLoad', 
                        'PositionalQuality', 'HDOP', 'Sats']
        
        for col in numeric_cols:
            df[col] = pd.to_numeric(df[col].astype(str).str.replace(',', '.'), errors='coerce')
        
        # Tratamento flexível para o Timestamp
        df['Timestamp'] = df['Timestamp'].astype(str).str.strip()
        
        try:
            df['Timestamp'] = pd.to_datetime(df['Timestamp'], format='%d/%m/%Y %H:%M:%S.%f', errors='coerce')
        except:
            try:
                df['Timestamp'] = pd.to_datetime(df['Timestamp'], format='%d/%m/%Y %H:%M:%S', errors='coerce')
            except:
                try:
                    df['Timestamp'] = pd.to_datetime(df['Timestamp'], dayfirst=True, errors='coerce')
                except:
                    df['Timestamp'] = pd.to_datetime(df['Timestamp'], errors='coerce')
        
        if df['Timestamp'].isna().all():
            df['Timestamp'] = df['Timestamp'].astype(str)
        
        df = df.dropna(subset=['Latitude', 'Longitude', 'Velocity', 'HeartRate'])
        
        return df
    except Exception as e:
        st.error(f"Erro ao carregar arquivo: {e}")
        return None

# ==================== SIDEBAR ====================

st.sidebar.header("📁 Upload do Arquivo")
st.sidebar.markdown("""
### Instruções:
1. Clique em "Browse files"
2. Selecione o arquivo CSV exportado pelo OpenField
3. Aguarde o processamento automático
""")

uploaded_file = st.sidebar.file_uploader(
    "Escolha o arquivo CSV",
    type=['csv'],
    help="Arquivo exportado pelo sistema OpenField no formato CSV"
)

with st.sidebar.expander("📋 Exemplo de formato esperado"):
    st.markdown("""
    O arquivo deve conter as seguintes colunas:
    - **Timestamp**: Data/hora da leitura
    - **Seconds**: Tempo em segundos
    - **Velocity**: Velocidade (km/h)
    - **Latitude**: Coordenada de latitude
    - **Longitude**: Coordenada de longitude
    - **HeartRate**: Frequência cardíaca (bpm)
    """)

# ==================== PROCESSAMENTO PRINCIPAL ====================

if uploaded_file is not None:
    st.sidebar.success(f"✅ Arquivo carregado: {uploaded_file.name}")
    st.sidebar.info(f"📊 Tamanho: {uploaded_file.size / 1024:.1f} KB")
    
    with st.spinner('Processando dados...'):
        df = load_data(uploaded_file)
    
    if df is not None and len(df) > 0:
        # Extrair informações do cabeçalho
        content = uploaded_file.getvalue().decode('utf-8')
        lines = content.split('\n')
        
        periodo = "Não identificado"
        atleta = "Não identificado"
        for line in lines[:10]:
            if 'Period:' in line:
                try:
                    periodo = line.split('"')[1] if '"' in line else line.split(':')[1].strip().strip('"')
                except:
                    periodo = "Não identificado"
            if 'Athlete:' in line:
                try:
                    atleta = line.split('"')[1] if '"' in line else line.split(':')[1].strip().strip('"')
                except:
                    atleta = "Não identificado"
        
        st.sidebar.markdown("---")
        st.sidebar.subheader("🏅 Informações do Atleta")
        st.sidebar.write(f"**Atleta:** {atleta}")
        st.sidebar.write(f"**Jogo:** {periodo}")
        
        st.sidebar.markdown("---")
        st.sidebar.subheader("📊 Estatísticas")
        st.sidebar.write(f"Registros: {len(df):,}")
        if 'Seconds' in df.columns and df['Seconds'].max() > 0:
            st.sidebar.write(f"Duração: {df['Seconds'].max():.0f} seg")
        if 'Odometer' in df.columns and df['Odometer'].max() > 0:
            st.sidebar.write(f"Distância: {df['Odometer'].max():.0f} m")
        
        # Filtros
        st.sidebar.markdown("---")
        st.sidebar.subheader("⚙️ Filtros")
        
        if 'Seconds' in df.columns and df['Seconds'].max() > 0:
            max_time = df['Seconds'].max()
            time_range = st.sidebar.slider(
                "Intervalo de tempo (segundos)",
                min_value=0.0,
                max_value=float(max_time),
                value=(0.0, float(max_time)),
                step=5.0
            )
            time_filter = (df['Seconds'] >= time_range[0]) & (df['Seconds'] <= time_range[1])
        else:
            time_filter = pd.Series([True] * len(df))
            time_range = (0, df['Seconds'].max() if 'Seconds' in df.columns else 100)
        
        if 'Velocity' in df.columns and df['Velocity'].max() > 0:
            max_speed = df['Velocity'].max()
            min_speed = df['Velocity'].min()
            speed_range = st.sidebar.slider(
                "Filtro por velocidade (km/h)",
                min_value=float(min_speed),
                max_value=float(max_speed),
                value=(float(min_speed), float(max_speed)),
                step=0.5
            )
            speed_filter = (df['Velocity'] >= speed_range[0]) & (df['Velocity'] <= speed_range[1])
        else:
            speed_filter = pd.Series([True] * len(df))
            speed_range = (0, 100)
        
        df_filtered = df[time_filter & speed_filter].copy()
        
        if len(df_filtered) == 0:
            st.warning("⚠️ Nenhum dado encontrado com os filtros selecionados.")
            st.stop()
        
        # Métricas principais
        st.markdown("### 📈 Métricas de Desempenho")
        col1, col2, col3, col4, col5 = st.columns(5)
        
        with col1:
            dist_total = df_filtered['Odometer'].max() if 'Odometer' in df_filtered.columns else 0
            st.metric("Distância total", f"{dist_total:.0f} m")
        with col2:
            vel_max = df_filtered['Velocity'].max() if 'Velocity' in df_filtered.columns else 0
            st.metric("Velocidade máxima", f"{vel_max:.1f} km/h")
        with col3:
            vel_media = df_filtered['Velocity'].mean() if 'Velocity' in df_filtered.columns else 0
            st.metric("Velocidade média", f"{vel_media:.1f} km/h")
        with col4:
            fc_media = df_filtered['HeartRate'].mean() if 'HeartRate' in df_filtered.columns else 0
            st.metric("FC média", f"{fc_media:.0f} bpm")
        with col5:
            fc_max = df_filtered['HeartRate'].max() if 'HeartRate' in df_filtered.columns else 0
            st.metric("FC máxima", f"{fc_max:.0f} bpm")
        
        # Criar abas
        tab1, tab2, tab3, tab4, tab5 = st.tabs(["🗺️ Mapa do Percurso", "📈 Gráficos de Desempenho", 
                                                  "⚡ Velocidade e Aceleração", "❤️ Frequência Cardíaca",
                                                  "📊 Análise de Entropia"])
        
        # ==================== TAB 1: MAPA ====================
        with tab1:
            st.subheader("Percurso do Atleta no Campo")
            
            center_lat = df_filtered['Latitude'].mean() if 'Latitude' in df_filtered.columns else -23.5505
            center_lon = df_filtered['Longitude'].mean() if 'Longitude' in df_filtered.columns else -46.6333
            
            map_style = st.selectbox(
                "Estilo do mapa",
                ["open-street-map", "carto-positron", "carto-darkmatter", "stamen-terrain"],
                index=0
            )
            
            fig_map = go.Figure()
            
            fig_map.add_trace(go.Scattermapbox(
                lat=df_filtered['Latitude'],
                lon=df_filtered['Longitude'],
                mode='lines+markers',
                line=dict(width=3, color='red'),
                marker=dict(
                    size=6,
                    color=df_filtered['Velocity'] if 'Velocity' in df_filtered.columns else 'blue',
                    colorscale='Viridis',
                    showscale=True,
                    colorbar=dict(title="Velocidade<br>(km/h)", x=1.02)
                ),
                text=[f"<b>Tempo:</b> {t:.1f}s<br><b>Velocidade:</b> {v:.1f} km/h" 
                      for t, v in zip(df_filtered['Seconds'] if 'Seconds' in df_filtered.columns else range(len(df_filtered)), 
                                    df_filtered['Velocity'] if 'Velocity' in df_filtered.columns else [0]*len(df_filtered))],
                hoverinfo='text',
                name='Percurso'
            ))
            
            fig_map.add_trace(go.Scattermapbox(
                lat=[df_filtered['Latitude'].iloc[0], df_filtered['Latitude'].iloc[-1]],
                lon=[df_filtered['Longitude'].iloc[0], df_filtered['Longitude'].iloc[-1]],
                mode='markers',
                marker=dict(size=12, color=['green', 'red'], symbol=['marker', 'marker']),
                text=['🏁 Início', '🏁 Fim'],
                hoverinfo='text',
                name='Pontos'
            ))
            
            fig_map.update_layout(
                mapbox=dict(
                    style=map_style,
                    center=dict(lat=center_lat, lon=center_lon),
                    zoom=15
                ),
                margin=dict(l=0, r=0, t=30, b=0),
                height=650,
                title={'text': f"Trajetória de {atleta} - {periodo}", 'x': 0.5, 'xanchor': 'center'},
                hovermode='closest'
            )
            
            st.plotly_chart(fig_map, use_container_width=True)
        
        # ==================== TAB 2: GRÁFICOS DE DESEMPENHO ====================
        with tab2:
            st.subheader("Análise de Desempenho ao Longo do Tempo")
            
            col1, col2 = st.columns([1, 3])
            with col1:
                var_options = {}
                if 'Velocity' in df_filtered.columns:
                    var_options['Velocity'] = 'Velocidade (km/h)'
                if 'HeartRate' in df_filtered.columns:
                    var_options['HeartRate'] = 'Frequência Cardíaca (bpm)'
                if 'Acceleration' in df_filtered.columns:
                    var_options['Acceleration'] = 'Aceleração (m/s²)'
                if 'PlayerLoad' in df_filtered.columns:
                    var_options['PlayerLoad'] = 'Carga do Jogador'
                
                selected_var = st.selectbox(
                    "Selecionar variável",
                    options=list(var_options.keys()),
                    format_func=lambda x: var_options[x]
                )
            
            fig = make_subplots(rows=2, cols=2, 
                               subplot_titles=(var_options[selected_var], 
                                             "Velocidade vs FC", 
                                             "Distância Acumulada",
                                             "Mapa de Calor de Velocidade"),
                               specs=[[{"type": "scatter"}, {"type": "scatter"}],
                                      [{"type": "scatter"}, {"type": "heatmap"}]])
            
            if 'Seconds' in df_filtered.columns:
                fig.add_trace(
                    go.Scatter(x=df_filtered['Seconds'], y=df_filtered[selected_var],
                              mode='lines', name=var_options[selected_var],
                              line=dict(width=2), fill='tozeroy'),
                    row=1, col=1
                )
                fig.update_xaxes(title_text="Tempo (s)", row=1, col=1)
            
            if 'Velocity' in df_filtered.columns and 'HeartRate' in df_filtered.columns:
                fig.add_trace(
                    go.Scatter(x=df_filtered['Velocity'], y=df_filtered['HeartRate'],
                              mode='markers', name='Dados',
                              marker=dict(size=5, color=df_filtered['Seconds'] if 'Seconds' in df_filtered.columns else 'blue', 
                                        colorscale='Viridis', showscale=True)),
                    row=1, col=2
                )
                fig.update_xaxes(title_text="Velocidade (km/h)", row=1, col=2)
                fig.update_yaxes(title_text="Frequência Cardíaca (bpm)", row=1, col=2)
            
            if 'Odometer' in df_filtered.columns and 'Seconds' in df_filtered.columns:
                fig.add_trace(
                    go.Scatter(x=df_filtered['Seconds'], y=df_filtered['Odometer'],
                              mode='lines', name='Distância',
                              line=dict(color='green', width=2), fill='tozeroy'),
                    row=2, col=1
                )
                fig.update_xaxes(title_text="Tempo (s)", row=2, col=1)
                fig.update_yaxes(title_text="Distância (m)", row=2, col=1)
            
            fig.update_layout(height=700, showlegend=False)
            st.plotly_chart(fig, use_container_width=True)
        
        # ==================== TAB 3: VELOCIDADE E ACELERAÇÃO ====================
        with tab3:
            st.subheader("Análise de Velocidade e Aceleração")
            
            col1, col2 = st.columns(2)
            
            with col1:
                if 'Velocity' in df_filtered.columns:
                    fig_hist = px.histogram(df_filtered, x='Velocity', nbins=40,
                                           title="Distribuição de Velocidades",
                                           labels={'Velocity': 'Velocidade (km/h)'},
                                           color_discrete_sequence=['blue'])
                    fig_hist.add_vline(x=df_filtered['Velocity'].mean(), 
                                       line_dash="dash", line_color="red",
                                       annotation_text=f"Média: {df_filtered['Velocity'].mean():.1f} km/h")
                    st.plotly_chart(fig_hist, use_container_width=True)
            
            with col2:
                if 'Velocity' in df_filtered.columns and 'Seconds' in df_filtered.columns:
                    df_filtered['Percentil_Tempo'] = pd.cut(df_filtered['Seconds'], 
                                                            bins=10,
                                                            labels=[f'{i*10}-{(i+1)*10}%' for i in range(10)])
                    fig_box = px.box(df_filtered, x='Percentil_Tempo', y='Velocity',
                                    title="Velocidade por Percentil do Tempo")
                    st.plotly_chart(fig_box, use_container_width=True)
            
            if 'Acceleration' in df_filtered.columns and 'Seconds' in df_filtered.columns:
                st.subheader("Aceleração ao longo do tempo")
                fig_acc = go.Figure()
                fig_acc.add_trace(go.Scatter(x=df_filtered['Seconds'], y=df_filtered['Acceleration'],
                                             mode='lines', line=dict(color='purple', width=1.5)))
                fig_acc.add_hline(y=0, line_dash="dash", line_color="black", annotation_text="Repouso")
                fig_acc.update_layout(xaxis_title="Tempo (s)", yaxis_title="Aceleração (m/s²)", height=450)
                st.plotly_chart(fig_acc, use_container_width=True)
        
        # ==================== TAB 4: FREQUÊNCIA CARDÍACA ====================
        with tab4:
            st.subheader("Análise da Frequência Cardíaca")
            
            if 'HeartRate' in df_filtered.columns:
                col1, col2 = st.columns(2)
                
                with col1:
                    fig_hr_hist = px.histogram(df_filtered, x='HeartRate', nbins=30,
                                               title="Distribuição da Frequência Cardíaca",
                                               labels={'HeartRate': 'Frequência Cardíaca (bpm)'},
                                               color_discrete_sequence=['red'])
                    fig_hr_hist.add_vline(x=df_filtered['HeartRate'].mean(), 
                                          line_dash="dash", line_color="blue",
                                          annotation_text=f"Média: {df_filtered['HeartRate'].mean():.0f} bpm")
                    st.plotly_chart(fig_hr_hist, use_container_width=True)
                
                with col2:
                    if 'Seconds' in df_filtered.columns:
                        fig_hr_time = go.Figure()
                        fig_hr_time.add_trace(go.Scatter(x=df_filtered['Seconds'], y=df_filtered['HeartRate'],
                                                         mode='lines', line=dict(color='red', width=2)))
                        
                        fc_max = df_filtered['HeartRate'].max()
                        fig_hr_time.add_hrect(y0=0, y1=fc_max*0.6, fillcolor="lightgreen", opacity=0.3)
                        fig_hr_time.add_hrect(y0=fc_max*0.6, y1=fc_max*0.75, fillcolor="yellow", opacity=0.3)
                        fig_hr_time.add_hrect(y0=fc_max*0.75, y1=fc_max*0.9, fillcolor="orange", opacity=0.3)
                        fig_hr_time.add_hrect(y0=fc_max*0.9, y1=fc_max, fillcolor="red", opacity=0.3)
                        
                        fig_hr_time.update_layout(xaxis_title="Tempo (s)", yaxis_title="FC (bpm)", height=450)
                        st.plotly_chart(fig_hr_time, use_container_width=True)
                
                st.markdown("### 📊 Análise por Zona de Intensidade")
                fc_max = df_filtered['HeartRate'].max()
                df_filtered['Zona_FC'] = pd.cut(df_filtered['HeartRate'], 
                                                bins=[0, fc_max*0.6, fc_max*0.75, fc_max*0.9, fc_max],
                                                labels=['Recuperação', 'Aeróbica', 'Anaeróbica', 'Máximo'])
                
                zona_stats = df_filtered.groupby('Zona_FC', observed=True).agg({
                    'HeartRate': ['count', 'mean', 'min', 'max'],
                    'Velocity': 'mean' if 'Velocity' in df_filtered.columns else lambda x: 0
                }).round(0)
                
                zona_stats.columns = ['Contagem', 'FC Média', 'FC Mín', 'FC Máx', 'Velocidade Média']
                zona_stats['% do Tempo'] = (zona_stats['Contagem'] / len(df_filtered) * 100).round(1).astype(str) + '%'
                st.dataframe(zona_stats, use_container_width=True)
        
        # ==================== TAB 5: ENTROPIA OTIMIZADA COM BOTÃO ====================
        with tab5:
            st.subheader("📊 Análise de Entropia - Complexidade dos Dados")
            st.markdown("""
            A entropia mede a **complexidade e imprevisibilidade** dos dados. 
            - **Maior entropia** = maior variabilidade e complexidade do movimento
            - **Menor entropia** = padrões mais regulares e previsíveis
            """)
            
            # Seleção da variável
            var_entropy_options = {}
            if 'Velocity' in df_filtered.columns:
                var_entropy_options['Velocity'] = 'Velocidade (km/h)'
            if 'HeartRate' in df_filtered.columns:
                var_entropy_options['HeartRate'] = 'Frequência Cardíaca (bpm)'
            if 'Acceleration' in df_filtered.columns:
                var_entropy_options['Acceleration'] = 'Aceleração (m/s²)'
            if 'PlayerLoad' in df_filtered.columns:
                var_entropy_options['PlayerLoad'] = 'Carga do Jogador'
            
            selected_entropy_var = st.selectbox(
                "Selecione a variável para análise de entropia",
                options=list(var_entropy_options.keys()),
                format_func=lambda x: var_entropy_options[x],
                key="entropy_var"
            )
            
            # Parâmetros
            col1, col2, col3 = st.columns(3)
            with col1:
                m_value = st.selectbox("Comprimento da sequência (m)", [1, 2, 3], index=1,
                                      help="2 é o padrão para análise de movimento")
            with col2:
                r_value = st.slider("Tolerância (r)", 0.1, 0.5, 0.2, step=0.05,
                                   help="0.2 é o valor padrão para dados esportivos")
            with col3:
                use_rolling = st.checkbox("Análise por janela deslizante", value=True,
                                         help="Mostra evolução temporal da entropia")
            
            # Botão para processar
            process_button = st.button("🚀 Processar Análise de Entropia", type="primary", use_container_width=True)
            
            if process_button:
                # Obter dados
                data_series = df_filtered[selected_entropy_var].dropna().values
                
                if len(data_series) > 30:
                    with st.spinner("Calculando entropia... Isso pode levar alguns segundos"):
                        # Calcular entropias globais
                        shannon_val = compute_entropy_fast(data_series, bins=30)
                        sample_val = sample_entropy_fast(data_series, m=m_value, r=r_value)
                        cv_val = np.std(data_series) / np.mean(data_series) if np.mean(data_series) > 0 else 0
                    
                    # Mostrar métricas
                    st.markdown("### 📈 Métricas de Entropia Global")
                    col_a, col_b, col_c = st.columns(3)
                    
                    with col_a:
                        st.metric("Entropia de Shannon", f"{shannon_val:.4f}", 
                                 help="Mede a diversidade dos valores. Valores típicos: 2-5")
                    with col_b:
                        st.metric("Entropia Amostral", f"{sample_val:.4f}" if not np.isnan(sample_val) else "N/A",
                                 help="Mede a regularidade temporal")
                    with col_c:
                        st.metric("Coeficiente de Variação", f"{cv_val:.3f}",
                                 help="Variabilidade relativa dos dados")
                    
                    # Classificação
                    st.markdown("**🏷️ Classificação da Complexidade:**")
                    if shannon_val < 2.5:
                        st.info("🔵 **Baixa complexidade** - Movimento padronizado e repetitivo")
                    elif shannon_val < 4.0:
                        st.success("🟢 **Média complexidade** - Variabilidade normal do esporte")
                    else:
                        st.warning("🟠 **Alta complexidade** - Grande variabilidade e imprevisibilidade")
                    
                    # Série temporal
                    st.markdown("### 📉 Série Temporal")
                    
                    fig_entropy_time = go.Figure()
                    time_x = df_filtered['Seconds'].values if 'Seconds' in df_filtered.columns else range(len(data_series))
                    
                    fig_entropy_time.add_trace(
                        go.Scatter(x=time_x, y=data_series, mode='lines', 
                                  line=dict(color='blue', width=2),
                                  name=var_entropy_options[selected_entropy_var])
                    )
                    
                    mean_val = np.mean(data_series)
                    fig_entropy_time.add_hline(y=mean_val, line_dash="dash", line_color="green", 
                                              annotation_text=f"Média: {mean_val:.2f}")
                    
                    fig_entropy_time.update_layout(
                        title=f"Série Temporal - {var_entropy_options[selected_entropy_var]}",
                        xaxis_title="Tempo (s)",
                        yaxis_title=var_entropy_options[selected_entropy_var],
                        height=400
                    )
                    st.plotly_chart(fig_entropy_time, use_container_width=True)
                    
                    # Análise por janela deslizante
                    if use_rolling and len(data_series) > 100:
                        with st.spinner("Calculando entropia por janela deslizante..."):
                            window_size = min(100, len(data_series) // 5)
                            positions, rolling_ent = rolling_entropy_fast(data_series, 
                                                                          window_size=window_size, 
                                                                          step=window_size//3,
                                                                          m=m_value, r=r_value)
                        
                        if len(rolling_ent) > 0:
                            st.markdown("### 📊 Evolução da Entropia ao Longo do Tempo")
                            
                            fig_rolling = go.Figure()
                            
                            if 'Seconds' in df_filtered.columns:
                                time_positions = np.interp(positions, range(len(data_series)), time_x)
                            else:
                                time_positions = positions
                            
                            fig_rolling.add_trace(
                                go.Scatter(x=time_positions, y=rolling_ent, mode='lines+markers',
                                          line=dict(color='red', width=2), marker=dict(size=6),
                                          name="Entropia Amostral")
                            )
                            fig_rolling.add_hline(y=np.nanmean(rolling_ent), line_dash="dash", 
                                                  line_color="orange", annotation_text=f"Média: {np.nanmean(rolling_ent):.3f}")
                            
                            fig_rolling.update_layout(
                                title="Entropia Amostral por Janela Deslizante",
                                xaxis_title="Tempo (s)",
                                yaxis_title="Entropia Amostral",
                                height=400
                            )
                            st.plotly_chart(fig_rolling, use_container_width=True)
                            
                            # Interpretação da tendência
                            if len(rolling_ent) > 3:
                                trend = rolling_ent[-1] - rolling_ent[0]
                                if trend > 0.1:
                                    st.info("📈 **Tendência: Entropia aumentando** - Atleta está se movendo com maior variabilidade")
                                elif trend < -0.1:
                                    st.warning("📉 **Tendência: Entropia diminuindo** - Possível fadiga ou padrões mais repetitivos")
                                else:
                                    st.success("➡️ **Tendência: Entropia estável** - Comportamento consistente")
                    
                    # Histograma
                    st.markdown("### 📊 Distribuição dos Valores")
                    fig_hist_entropy = px.histogram(x=data_series, nbins=30, 
                                                    title="Histograma",
                                                    labels={'x': var_entropy_options[selected_entropy_var]},
                                                    color_discrete_sequence=['steelblue'])
                    st.plotly_chart(fig_hist_entropy, use_container_width=True)
                    
                    # Valores de referência
                    with st.expander("📌 Valores de Referência para Entropia"):
                        st.markdown("""
                        | Categoria | Entropia Shannon | Entropia Amostral | Interpretação |
                        |-----------|-----------------|------------------|---------------|
                        | **Baixa** | < 2.5 | < 0.5 | Movimento repetitivo, padrões regulares |
                        | **Média** | 2.5 - 4.0 | 0.5 - 1.2 | Variabilidade normal do esporte |
                        | **Alta** | > 4.0 | > 1.2 | Alta complexidade, movimentos variados |
                        
                        **Aplicações no Esporte:**
                        - Entropia baixa pode indicar **fadiga** ou movimentos muito padronizados
                        - Entropia alta pode indicar **alta variabilidade** e adaptabilidade
                        - Acompanhar a evolução pode ajudar a **prevenir lesões**
                        """)
                
                else:
                    st.warning(f"⚠️ Dados insuficientes para análise de entropia. São necessários pelo menos 30 pontos. Atualmente: {len(data_series)} pontos.")
            else:
                # Estado inicial antes do botão
                st.info("👆 **Clique no botão acima para iniciar a análise de entropia.**")
                st.markdown("""
                ### 📝 Sobre a Análise de Entropia:
                
                A análise de entropia avalia a **complexidade** dos movimentos do atleta:
                
                - **Entropia de Shannon**: Mede a diversidade de valores na série temporal
                - **Entropia Amostral**: Mede a regularidade dos padrões temporais
                
                **Para melhores resultados:**
                1. Selecione a variável que deseja analisar (Velocidade, FC, etc.)
                2. Ajuste os parâmetros conforme necessário
                3. Clique em "Processar Análise de Entropia"
                4. Aguarde o processamento (alguns segundos)
                
                **Dica:** Quanto mais dados, mais precisa será a análise!
                """)
        
        # Botão de download
        st.markdown("---")
        csv_data = df_filtered.to_csv(index=False)
        st.download_button(
            label="📥 Exportar dados filtrados (CSV)",
            data=csv_data,
            file_name=f"dados_filtrados_{atleta.replace(' ', '_')}.csv",
            mime="text/csv"
        )
        
        st.markdown("---")
        st.markdown(f"**📊 Resumo da análise:** {len(df_filtered)} registros")
        if 'Seconds' in df_filtered.columns:
            st.markdown(f"**⏱️ Período:** {time_range[0]:.0f}s - {time_range[1]:.0f}s")
    
    else:
        st.error("❌ Erro ao processar o arquivo. Verifique se o formato está correto.")

else:
    # Tela inicial
    st.markdown("""
    ### 👋 Bem-vindo ao Analisador de Percurso do Atleta!
    
    Esta ferramenta permite visualizar e analisar os dados de posicionamento e desempenho de atletas.
    
    ### 🚀 Como usar:
    1. **Faça upload** do arquivo CSV na barra lateral esquerda
    2. Aguarde o processamento automático
    3. Explore as visualizações nas abas
    
    ---
    **👈 Clique em "Browse files" na barra lateral para começar!**
    """)
    
    st.info("ℹ️ Aguardando upload do arquivo...")