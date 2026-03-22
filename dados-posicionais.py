import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import numpy as np
from io import StringIO
from scipy.stats import entropy
from scipy.signal import find_peaks
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

# Função para calcular Entropia de Shannon
def shannon_entropy(data, bins=20):
    """Calcula a entropia de Shannon de uma série de dados"""
    hist, _ = np.histogram(data, bins=bins, density=True)
    # Remove zeros para evitar log(0)
    hist = hist[hist > 0]
    return entropy(hist, base=2)

# Função para calcular Entropia Amostral (Sample Entropy)
def sample_entropy(data, m=2, r=0.2):
    """
    Calcula a entropia amostral (Sample Entropy)
    m: comprimento das sequências a serem comparadas
    r: tolerância (geralmente 0.1 a 0.25 do desvio padrão)
    """
    data = np.array(data)
    N = len(data)
    
    # Se não há dados suficientes
    if N < m + 1:
        return np.nan
    
    # Normalizar pelo desvio padrão
    r = r * np.std(data)
    if r == 0:
        r = 0.01  # Evita divisão por zero
    
    # Função para contar matches
    def _maxdist(xi, xj):
        return np.max(np.abs(xi - xj))
    
    # Para m e m+1
    def _phi(m):
        count = 0
        total = 0
        for i in range(N - m):
            template = data[i:i+m]
            for j in range(i+1, N - m + 1):
                if _maxdist(template, data[j:j+m]) <= r:
                    count += 1
                total += 1
        return count / total if total > 0 else 0
    
    phi_m = _phi(m)
    phi_m1 = _phi(m+1)
    
    if phi_m == 0 or phi_m1 == 0:
        return 0
    
    return -np.log(phi_m1 / phi_m)

# Função para Entropia por janela deslizante
def rolling_entropy(data, window_size=30, step=5):
    """Calcula entropia em janelas deslizantes"""
    entropies = []
    positions = []
    
    for i in range(0, len(data) - window_size + 1, step):
        window = data[i:i+window_size]
        if len(window) > 10 and np.std(window) > 0:
            ent = sample_entropy(window, m=2, r=0.2)
            entropies.append(ent)
            positions.append(i + window_size//2)
    
    return np.array(positions), np.array(entropies)

# Função para carregar e processar os dados
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

# Sidebar
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

# Processar arquivo quando enviado
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
        
        # Criar abas (agora com 5 abas)
        tab1, tab2, tab3, tab4, tab5 = st.tabs(["🗺️ Mapa do Percurso", "📈 Gráficos de Desempenho", 
                                                  "⚡ Velocidade e Aceleração", "❤️ Frequência Cardíaca",
                                                  "📊 Análise de Entropia"])
        
        # Tab 1: Mapa do Percurso
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
                text=[f"<b>Tempo:</b> {t:.1f}s<br><b>Velocidade:</b> {v:.1f} km/h<br><b>FC:</b> {h:.0f} bpm" 
                      for t, v, h in zip(df_filtered['Seconds'] if 'Seconds' in df_filtered.columns else range(len(df_filtered)), 
                                        df_filtered['Velocity'] if 'Velocity' in df_filtered.columns else [0]*len(df_filtered),
                                        df_filtered['HeartRate'] if 'HeartRate' in df_filtered.columns else [0]*len(df_filtered))],
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
        
        # Tab 2: Gráficos de Desempenho
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
        
        # Tab 3: Velocidade e Aceleração
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
        
        # Tab 4: Frequência Cardíaca
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
        
        # Tab 5: Análise de Entropia (NOVA ABA)
        with tab5:
            st.subheader("📊 Análise de Entropia - Complexidade dos Dados")
            st.markdown("""
            A entropia mede a **complexidade e imprevisibilidade** dos dados. 
            - **Maior entropia** = maior variabilidade e complexidade do movimento
            - **Menor entropia** = padrões mais regulares e previsíveis
            """)
            
            # Seleção da variável para análise de entropia
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
            
            # Parâmetros para entropia amostral
            col1, col2, col3 = st.columns(3)
            with col1:
                m_value = st.slider("Comprimento da sequência (m)", 1, 5, 2, 
                                   help="Valor menor = mais sensível a ruído, maior = captura padrões mais longos")
            with col2:
                r_value = st.slider("Tolerância (r)", 0.1, 0.5, 0.2, step=0.05,
                                   help="Proporção do desvio padrão. Menor = mais sensível")
            with col3:
                window_size = st.slider("Janela deslizante (pontos)", 20, 100, 50,
                                       help="Tamanho da janela para análise temporal")
            
            # Obter os dados da variável selecionada
            data_series = df_filtered[selected_entropy_var].dropna().values
            
            if len(data_series) > 50:
                # Calcular entropia de Shannon global
                shannon_val = shannon_entropy(data_series, bins=30)
                
                # Calcular entropia amostral global
                try:
                    sample_val = sample_entropy(data_series, m=m_value, r=r_value)
                except:
                    sample_val = np.nan
                
                # Métricas de entropia
                st.markdown("### 📈 Métricas de Entropia Global")
                col_a, col_b, col_c = st.columns(3)
                
                with col_a:
                    st.metric("Entropia de Shannon", f"{shannon_val:.4f}", 
                             help="Mede a diversidade dos valores. Valores típicos: 2-5 para dados esportivos")
                with col_b:
                    st.metric("Entropia Amostral", f"{sample_val:.4f}" if not np.isnan(sample_val) else "N/A",
                             help="Mede a regularidade temporal. Valores baixos = padrões repetitivos")
                with col_c:
                    cv_value = np.std(data_series) / np.mean(data_series) if np.mean(data_series) > 0 else 0
                    st.metric("Coeficiente de Variação", f"{cv_value:.3f}",
                             help="Variabilidade relativa dos dados")
                
                # Gráfico da série temporal com destaque
                st.markdown("### 📉 Série Temporal e Complexidade")
                
                fig_entropy_time = make_subplots(rows=2, cols=1, 
                                                  subplot_titles=("Série Temporal - " + var_entropy_options[selected_entropy_var],
                                                                "Entropia Amostral por Janela Deslizante"),
                                                  row_heights=[0.5, 0.5])
                
                # Série temporal
                time_x = df_filtered['Seconds'].values if 'Seconds' in df_filtered.columns else range(len(data_series))
                fig_entropy_time.add_trace(
                    go.Scatter(x=time_x, y=data_series, mode='lines', 
                              line=dict(color='blue', width=2),
                              name=var_entropy_options[selected_entropy_var]),
                    row=1, col=1
                )
                
                # Adicionar média e desvio padrão
                mean_val = np.mean(data_series)
                std_val = np.std(data_series)
                fig_entropy_time.add_hline(y=mean_val, line_dash="dash", line_color="green", 
                                          annotation_text=f"Média: {mean_val:.2f}", row=1, col=1)
                fig_entropy_time.add_hrect(y0=mean_val-std_val, y1=mean_val+std_val, 
                                          fillcolor="lightblue", opacity=0.3, line_width=0, row=1, col=1)
                
                # Entropia por janela deslizante
                positions, rolling_ent = rolling_entropy(data_series, window_size=window_size, step=5)
                if len(rolling_ent) > 0 and len(positions) > 0:
                    # Ajustar posições de tempo
                    if 'Seconds' in df_filtered.columns:
                        time_positions = np.interp(positions, range(len(data_series)), time_x)
                    else:
                        time_positions = positions
                    
                    fig_entropy_time.add_trace(
                        go.Scatter(x=time_positions, y=rolling_ent, mode='lines+markers',
                                  line=dict(color='red', width=2), marker=dict(size=6),
                                  name="Entropia Amostral"),
                        row=2, col=1
                    )
                    fig_entropy_time.add_hline(y=np.nanmean(rolling_ent), line_dash="dash", 
                                              line_color="orange", row=2, col=1,
                                              annotation_text=f"Média: {np.nanmean(rolling_ent):.3f}")
                
                fig_entropy_time.update_layout(height=600, showlegend=False)
                fig_entropy_time.update_xaxes(title_text="Tempo (s)", row=2, col=1)
                fig_entropy_time.update_yaxes(title_text=var_entropy_options[selected_entropy_var], row=1, col=1)
                fig_entropy_time.update_yaxes(title_text="Entropia Amostral", row=2, col=1)
                
                st.plotly_chart(fig_entropy_time, use_container_width=True)
                
                # Histograma com análise de distribuição
                st.markdown("### 📊 Distribuição dos Valores")
                col_hist, col_ref = st.columns([2, 1])
                
                with col_hist:
                    fig_hist_entropy = go.Figure()
                    fig_hist_entropy.add_trace(go.Histogram(x=data_series, nbinsx=40, 
                                                            marker_color='steelblue', opacity=0.7))
                    fig_hist_entropy.update_layout(title="Histograma e Densidade",
                                                   xaxis_title=var_entropy_options[selected_entropy_var],
                                                   yaxis_title="Frequência")
                    st.plotly_chart(fig_hist_entropy, use_container_width=True)
                
                with col_ref:
                    st.markdown("**📌 Valores de Referência para Entropia**")
                    st.markdown("""
                    | Categoria | Entropia Shannon | Entropia Amostral |
                    |-----------|-----------------|------------------|
                    | **Baixa** | < 2.5 | < 0.5 |
                    | **Média** | 2.5 - 4.0 | 0.5 - 1.2 |
                    | **Alta** | > 4.0 | > 1.2 |
                    
                    **Interpretação:**
                    - **Baixa entropia**: Movimento repetitivo, padrões regulares
                    - **Média entropia**: Variabilidade normal do esporte
                    - **Alta entropia**: Alta complexidade, movimentos variados
                    """)
                    
                    # Classificação atual
                    st.markdown("**🏷️ Classificação Atual:**")
                    if shannon_val < 2.5:
                        st.info("🔵 **Baixa complexidade** - Movimento padronizado")
                    elif shannon_val < 4.0:
                        st.success("🟢 **Média complexidade** - Variabilidade normal")
                    else:
                        st.warning("🟠 **Alta complexidade** - Grande variabilidade")
                
                # Análise por segmentos do jogo
                st.markdown("### 🎯 Análise por Segmentos do Jogo")
                
                if 'Seconds' in df_filtered.columns:
                    num_segments = 4
                    df_filtered['Segmento_Jogo'] = pd.cut(df_filtered['Seconds'], 
                                                          bins=num_segments,
                                                          labels=['Início', '1º Quarto', '2º Quarto', 'Final'])
                    
                    segment_entropy = []
                    segment_names = []
                    
                    for segment in df_filtered['Segmento_Jogo'].unique():
                        segment_data = df_filtered[df_filtered['Segmento_Jogo'] == segment][selected_entropy_var].dropna().values
                        if len(segment_data) > 20:
                            seg_ent = sample_entropy(segment_data, m=m_value, r=r_value)
                            segment_entropy.append(seg_ent if not np.isnan(seg_ent) else 0)
                            segment_names.append(segment)
                    
                    if segment_entropy:
                        fig_segment = go.Figure(data=[
                            go.Bar(x=segment_names, y=segment_entropy, 
                                  marker_color=['#2ecc71', '#3498db', '#e74c3c', '#f39c12'],
                                  text=[f'{e:.3f}' for e in segment_entropy], textposition='auto')
                        ])
                        fig_segment.update_layout(title="Entropia Amostral por Segmento do Jogo",
                                                  xaxis_title="Segmento", yaxis_title="Entropia Amostral",
                                                  height=400)
                        st.plotly_chart(fig_segment, use_container_width=True)
                        
                        st.markdown("""
                        **💡 Interpretação:**
                        - **Entropia crescente**: Atleta aumenta variabilidade de movimentos ao longo do jogo
                        - **Entropia decrescente**: Atleta se torna mais regular/padronizado
                        - **Entropia estável**: Comportamento consistente durante toda a partida
                        """)
                
            else:
                st.warning("⚠️ Dados insuficientes para análise de entropia. São necessários pelo menos 50 pontos.")
        
        # Botão para download
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
        st.info("💡 Dica: O arquivo deve ser exportado pelo sistema OpenField no formato CSV.")

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
    
    with st.expander("🎯 Funcionalidades da Análise de Entropia"):
        st.markdown("""
        **O que é Entropia?**
        
        A entropia mede a **complexidade** e **imprevisibilidade** dos dados de movimento:
        
        - **Entropia de Shannon**: Mede a diversidade de valores (quão distribuídos estão os dados)
        - **Entropia Amostral**: Mede a regularidade temporal (quão repetitivos são os padrões)
        
        **Aplicações no Esporte:**
        - Identificar fadiga (redução da entropia)
        - Avaliar variabilidade de movimento
        - Comparar padrões entre diferentes momentos do jogo
        - Detectar mudanças no comportamento do atleta
        """)