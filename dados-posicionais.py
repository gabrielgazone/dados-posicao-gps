import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import numpy as np
from io import StringIO
import warnings
import re
from datetime import datetime, timedelta
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

# ==================== FUNÇÕES DE ENTROPIA AMOSTRAL OTIMIZADA ====================

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
def rolling_sample_entropy(data, window_size=50, step=10, m=2, r=0.2):
    """
    Entropia Amostral por janela deslizante otimizada
    """
    N = len(data)
    entropies = []
    positions = []
    
    num_windows = min(30, (N - window_size) // step + 1)
    
    for i in range(0, N - window_size + 1, max(step, (N - window_size) // num_windows)):
        window = data[i:i+window_size]
        if len(window) > 10 and np.std(window) > 0.01:
            ent = sample_entropy_fast(window, m=m, r=r)
            if not np.isnan(ent):
                entropies.append(ent)
                positions.append(i + window_size // 2)
    
    return np.array(positions), np.array(entropies)

# ==================== FUNÇÃO DE CONVERSÃO DE SEGUNDOS PARA HORÁRIO ====================

def seconds_to_time_str(seconds, start_datetime):
    """Converte segundos desde o início para horário HH:MM:SS"""
    if start_datetime is None:
        return f"{int(seconds // 3600):02d}:{int((seconds % 3600) // 60):02d}:{int(seconds % 60):02d}"
    target_time = start_datetime + timedelta(seconds=seconds)
    return target_time.strftime("%H:%M:%S")

def time_str_to_seconds(time_str, start_datetime):
    """Converte horário HH:MM:SS para segundos desde o início"""
    try:
        target_time = datetime.strptime(time_str, "%H:%M:%S")
        start_time = start_datetime.time()
        # Cria um datetime combinando a data de início com o horário alvo
        target_full = datetime.combine(start_datetime.date(), target_time.time())
        seconds = (target_full - start_datetime).total_seconds()
        return max(0, seconds)
    except:
        return 0

# ==================== FUNÇÃO DE CARREGAMENTO DE DADOS CORRIGIDA ====================

def extract_athlete_from_line8(content):
    """Extrai o nome do atleta especificamente da linha 8 do arquivo"""
    try:
        lines = content.split('\n')
        # Verifica se tem pelo menos 8 linhas
        if len(lines) >= 8:
            line8 = lines[7]  # linha 8 (índice 7, pois começa em 0)
            # Busca o padrão # Athlete: "NOME"
            if '# Athlete:' in line8:
                # Procura por texto entre aspas
                match = re.search(r'"([^"]*)"', line8)
                if match:
                    return match.group(1).strip()
                # Se não encontrar aspas, tenta após os dois pontos
                parts = line8.split(':')
                if len(parts) > 1:
                    nome = parts[1].split(';')[0].strip().strip('"')
                    if nome:
                        return nome
        return None
    except Exception as e:
        return None

@st.cache_data
def load_data(uploaded_file):
    """Carrega e processa os dados do arquivo CSV enviado com extração correta do nome do atleta da linha 8"""
    try:
        content = uploaded_file.getvalue().decode('utf-8')
        
        # Extrair nome do atleta especificamente da linha 8
        atleta = extract_athlete_from_line8(content)
        if atleta is None:
            atleta = "Não identificado"
        
        # Extrair período (pode estar em qualquer linha)
        periodo = "Não identificado"
        lines = content.split('\n')
        
        for line in lines[:15]:
            if '# Period:' in line or '# Periodo:' in line:
                try:
                    match = re.search(r'"([^"]*)"', line)
                    if match:
                        periodo = match.group(1).strip()
                    else:
                        periodo = line.split(':')[1].strip().strip('"').strip(';')
                except:
                    periodo = "Não identificado"
                break
        
        # Encontrar onde os dados começam (após os cabeçalhos)
        data_start = 0
        for i, line in enumerate(lines):
            if not line.startswith('#') and 'Timestamp' in line:
                data_start = i
                break
        
        # Ler os dados
        df = pd.read_csv(StringIO(content), skiprows=data_start, sep=';')
        
        # Renomear colunas
        col_names = ['Timestamp', 'Seconds', 'Velocity', 'Acceleration', 'Odometer', 
                     'Latitude', 'Longitude', 'HeartRate', 'PlayerLoad', 
                     'PositionalQuality', 'HDOP', 'Sats']
        
        # Se o número de colunas não corresponder, tentar usar os nomes originais
        if len(df.columns) == len(col_names):
            df.columns = col_names
        else:
            # Usar os primeiros nomes disponíveis
            for i, col in enumerate(df.columns[:len(col_names)]):
                df.rename(columns={col: col_names[i]}, inplace=True)
        
        # Converter colunas numéricas
        numeric_cols = ['Seconds', 'Velocity', 'Acceleration', 'Odometer', 
                        'Latitude', 'Longitude', 'HeartRate', 'PlayerLoad', 
                        'PositionalQuality', 'HDOP', 'Sats']
        
        for col in numeric_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col].astype(str).str.replace(',', '.'), errors='coerce')
        
        # Tratamento flexível para o Timestamp
        start_datetime = None
        if 'Timestamp' in df.columns:
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
            
            # Obter o timestamp inicial
            start_datetime = df['Timestamp'].min() if not df['Timestamp'].isna().all() else None
        
        # Remover linhas com dados essenciais faltando
        df = df.dropna(subset=['Latitude', 'Longitude', 'Velocity', 'HeartRate'])
        
        # Adicionar coluna de arquivo origem e timestamp inicial
        df['arquivo_origem'] = uploaded_file.name
        df['start_datetime'] = start_datetime
        
        return df, atleta, periodo, start_datetime
    except Exception as e:
        st.error(f"Erro ao carregar arquivo {uploaded_file.name}: {e}")
        return None, None, None, None

# ==================== FUNÇÃO PARA FORMATAR HORÁRIO ====================

def format_time_range(start_seconds, end_seconds, start_datetime):
    """Formata o intervalo de tempo em horários HH:MM:SS"""
    if start_datetime is None:
        return f"{int(start_seconds // 3600):02d}:{int((start_seconds % 3600) // 60):02d}:{int(start_seconds % 60):02d} - {int(end_seconds // 3600):02d}:{int((end_seconds % 3600) // 60):02d}:{int(end_seconds % 60):02d}"
    
    start_time = start_datetime + timedelta(seconds=start_seconds)
    end_time = start_datetime + timedelta(seconds=end_seconds)
    return f"{start_time.strftime('%H:%M:%S')} - {end_time.strftime('%H:%M:%S')}"

# ==================== SIDEBAR ====================

st.sidebar.header("📁 Upload de Arquivos")
st.sidebar.markdown("""
### Instruções:
1. Clique em "Browse files"
2. Selecione **um ou mais** arquivos CSV
3. Aguarde o processamento automático
""")

uploaded_files = st.sidebar.file_uploader(
    "Escolha os arquivos CSV",
    type=['csv'],
    accept_multiple_files=True,
    help="Arquivos exportados pelo sistema OpenField no formato CSV"
)

with st.sidebar.expander("📋 Exemplo de formato esperado"):
    st.markdown("""
    O arquivo deve conter as seguintes colunas:
    - **Timestamp**: Data/hora da leitura
    - **Seconds**: Tempo em segundos
    - **Velocity**: Velocidade (km/h)
    - **Acceleration**: Aceleração (m/s²)
    - **Latitude**: Coordenada de latitude
    - **Longitude**: Coordenada de longitude
    - **HeartRate**: Frequência cardíaca (bpm)
    
    **Nome do atleta** é extraído da linha 8:
    `# Athlete: "L.GAZAL";;;;;;;;;;;`
    """)

# ==================== PROCESSAMENTO PRINCIPAL ====================

if uploaded_files:
    st.sidebar.success(f"✅ {len(uploaded_files)} arquivo(s) carregado(s)")
    
    # Carregar todos os arquivos
    all_data = []
    all_atletas = []
    all_periodos = []
    all_start_datetimes = []
    
    progress_bar = st.sidebar.progress(0)
    for idx, file in enumerate(uploaded_files):
        st.sidebar.info(f"📊 Processando: {file.name}")
        df, atleta, periodo, start_datetime = load_data(file)
        if df is not None and len(df) > 0:
            all_data.append(df)
            all_atletas.append(atleta)
            all_periodos.append(periodo)
            all_start_datetimes.append(start_datetime)
        progress_bar.progress((idx + 1) / len(uploaded_files))
    
    if all_data:
        # Seleção de atletas (múltiplos)
        st.sidebar.markdown("---")
        st.sidebar.subheader("🎯 Selecionar Atleta(s)")
        
        # Criar lista de opções para seleção múltipla
        atleta_options = []
        for atleta, periodo, arquivo in zip(all_atletas, all_periodos, [f.name for f in uploaded_files]):
            display_name = f"{atleta} - {periodo}" if periodo != "Não identificado" else atleta
            atleta_options.append(display_name)
        
        selected_atletas_indices = st.sidebar.multiselect(
            "Escolha os atletas para análise",
            options=range(len(atleta_options)),
            format_func=lambda x: atleta_options[x],
            default=[0] if len(atleta_options) > 0 else []
        )
        
        # Determinar atletas selecionados
        selected_indices = selected_atletas_indices
        selected_atletas = [all_atletas[i] for i in selected_indices]
        selected_data = [all_data[i] for i in selected_indices]
        selected_periodos = [all_periodos[i] for i in selected_indices]
        selected_start_datetimes = [all_start_datetimes[i] for i in selected_indices]
        
        # Usar o timestamp do primeiro atleta como referência
        reference_datetime = selected_start_datetimes[0] if selected_start_datetimes[0] is not None else None
        
        # Filtro de tempo global na sidebar (em horário HH:MM:SS)
        st.sidebar.markdown("---")
        st.sidebar.subheader("⏱️ Filtro Temporal Global")
        
        # Encontrar o tempo máximo entre todos os atletas selecionados
        max_time_global = 0
        for df in selected_data:
            if 'Seconds' in df.columns and df['Seconds'].max() > max_time_global:
                max_time_global = df['Seconds'].max()
        
        if max_time_global > 0 and reference_datetime is not None:
            # Criar opções de horário para o slider
            start_time_default = reference_datetime
            end_time_default = reference_datetime + timedelta(seconds=max_time_global)
            
            # Usar dois seletores de horário (início e fim)
            col_time1, col_time2 = st.sidebar.columns(2)
            
            with col_time1:
                start_time_str = st.time_input(
                    "Horário de início",
                    value=start_time_default.time(),
                    key="start_time_global"
                )
            
            with col_time2:
                end_time_str = st.time_input(
                    "Horário de fim",
                    value=end_time_default.time(),
                    key="end_time_global"
                )
            
            # Converter horários para segundos desde o início
            start_datetime_global = datetime.combine(reference_datetime.date(), start_time_str)
            end_datetime_global = datetime.combine(reference_datetime.date(), end_time_str)
            
            start_time_global = max(0, (start_datetime_global - reference_datetime).total_seconds())
            end_time_global = min(max_time_global, (end_datetime_global - reference_datetime).total_seconds())
            
            # Garantir que start < end
            if start_time_global >= end_time_global:
                st.sidebar.warning("⚠️ Horário de início deve ser anterior ao horário de fim")
                end_time_global = start_time_global + 1
            
            # Exibir o intervalo selecionado
            st.sidebar.info(f"📅 Intervalo: {start_time_str.strftime('%H:%M:%S')} - {end_time_str.strftime('%H:%M:%S')}")
            st.sidebar.info(f"⏱️ Duração: {int(end_time_global - start_time_global)} segundos")
            
        else:
            # Fallback para segundos se não houver timestamp
            start_time_global = 0.0
            end_time_global = float(max_time_global) if max_time_global > 0 else 100
            st.sidebar.info(f"⏱️ Intervalo: {start_time_global:.0f}s - {end_time_global:.0f}s")
        
        # Filtro de velocidade
        st.sidebar.markdown("---")
        st.sidebar.subheader("⚡ Filtro por Velocidade")
        
        # Encontrar velocidade máxima entre todos os atletas selecionados
        max_speed_global = 0
        min_speed_global = 0
        for df in selected_data:
            if 'Velocity' in df.columns:
                max_speed_global = max(max_speed_global, df['Velocity'].max())
                min_speed_global = min(min_speed_global, df['Velocity'].min())
        
        if max_speed_global > 0:
            speed_range = st.sidebar.slider(
                "Filtro por velocidade (km/h)",
                min_value=float(min_speed_global),
                max_value=float(max_speed_global),
                value=(float(min_speed_global), float(max_speed_global)),
                step=0.5
            )
        else:
            speed_range = (0, 100)
        
        # Opções de visualização
        st.sidebar.markdown("---")
        st.sidebar.subheader("🎨 Opções de Visualização")
        show_field = st.sidebar.checkbox("🗺️ Mostrar campo de futebol", value=True)
        map_style = st.sidebar.selectbox(
            "Estilo do mapa base",
            ["open-street-map", "carto-positron", "carto-darkmatter", "stamen-terrain"],
            index=0
        )
        show_heatmap = st.sidebar.checkbox("🌡️ Mostrar mapa de calor de velocidade", value=False)
        
        # Processar dados de cada atleta selecionado (aplicando filtros globais)
        dfs_filtered = []
        for df, atleta, periodo, start_dt in zip(selected_data, selected_atletas, selected_periodos, selected_start_datetimes):
            # Aplicar filtros de tempo e velocidade
            time_filter = (df['Seconds'] >= start_time_global) & (df['Seconds'] <= end_time_global)
            speed_filter = (df['Velocity'] >= speed_range[0]) & (df['Velocity'] <= speed_range[1])
            df_filtered = df[time_filter & speed_filter].copy()
            df_filtered['Atleta'] = atleta
            df_filtered['Periodo'] = periodo
            df_filtered['start_datetime'] = start_dt
            dfs_filtered.append(df_filtered)
        
        # Combinar dados filtrados para análises
        if len(dfs_filtered) > 0:
            df_combined = pd.concat(dfs_filtered, ignore_index=True)
        else:
            df_combined = pd.DataFrame()
        
        if len(df_combined) == 0:
            st.warning("⚠️ Nenhum dado encontrado com os filtros selecionados.")
            st.stop()
        
        # Exibir atletas selecionados e filtros aplicados
        st.sidebar.markdown("---")
        st.sidebar.subheader("🏅 Atletas em Análise")
        for atleta, periodo in zip(selected_atletas, selected_periodos):
            st.sidebar.write(f"**{atleta}** - {periodo}")
        
        # Exibir resumo dos filtros no corpo principal
        st.markdown("### 📊 Filtros Aplicados")
        col_f1, col_f2, col_f3 = st.columns(3)
        with col_f1:
            st.info(f"**⏱️ Intervalo de tempo:** {format_time_range(start_time_global, end_time_global, reference_datetime)}")
        with col_f2:
            st.info(f"**⚡ Filtro de velocidade:** {speed_range[0]:.1f} - {speed_range[1]:.1f} km/h")
        with col_f3:
            total_registros = sum(len(df) for df in dfs_filtered)
            st.info(f"**📊 Total de registros:** {total_registros:,}")
        
        # Métricas principais (para o primeiro atleta selecionado)
        df_main = dfs_filtered[0]
        atleta_main = selected_atletas[0]
        periodo_main = selected_periodos[0]
        start_dt_main = selected_start_datetimes[0]
        
        st.markdown("### 📈 Métricas de Desempenho")
        col1, col2, col3, col4, col5 = st.columns(5)
        
        with col1:
            dist_total = df_main['Odometer'].max() if 'Odometer' in df_main.columns else 0
            st.metric("Distância total", f"{dist_total:.0f} m")
        with col2:
            vel_max = df_main['Velocity'].max() if 'Velocity' in df_main.columns else 0
            st.metric("Velocidade máxima", f"{vel_max:.1f} km/h")
        with col3:
            vel_media = df_main['Velocity'].mean() if 'Velocity' in df_main.columns else 0
            st.metric("Velocidade média", f"{vel_media:.1f} km/h")
        with col4:
            fc_media = df_main['HeartRate'].mean() if 'HeartRate' in df_main.columns else 0
            st.metric("FC média", f"{fc_media:.0f} bpm")
        with col5:
            fc_max = df_main['HeartRate'].max() if 'HeartRate' in df_main.columns else 0
            st.metric("FC máxima", f"{fc_max:.0f} bpm")
        
        # Criar abas
        tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
            "🗺️ Mapa do Percurso", 
            "📈 Gráficos de Desempenho", 
            "⚡ Velocidade e Aceleração",
            "❤️ Frequência Cardíaca",
            "🔄 Aceleração vs Velocidade", 
            "📊 Entropia Amostral"
        ])
        
        # ==================== TAB 1: MAPA COM CAMPO DE FUTEBOL ====================
        with tab1:
            st.subheader("Percurso do Atleta no Campo de Futebol")
            
            # Slider para selecionar o atleta para visualização no mapa
            if len(selected_atletas) > 1:
                atleta_mapa = st.selectbox(
                    "Selecione o atleta para visualizar no mapa",
                    options=selected_atletas,
                    index=0,
                    key="mapa_atleta_select"
                )
                idx_mapa = selected_atletas.index(atleta_mapa)
                df_mapa = dfs_filtered[idx_mapa].copy()
                atleta_mapa_nome = atleta_mapa
                periodo_mapa = selected_periodos[idx_mapa]
                start_dt_mapa = selected_start_datetimes[idx_mapa]
            else:
                df_mapa = df_main.copy()
                atleta_mapa_nome = atleta_main
                periodo_mapa = periodo_main
                start_dt_mapa = start_dt_main
            
            # Já temos o filtro global aplicado, então usamos os dados filtrados
            df_mapa_filtrado = df_mapa.copy()
            
            st.info(f"📊 Mostrando percurso de {atleta_mapa_nome} no intervalo {format_time_range(start_time_global, end_time_global, reference_datetime)} | {len(df_mapa_filtrado)} pontos")
            
            # Calcular centro do mapa
            center_lat = df_mapa_filtrado['Latitude'].mean() if len(df_mapa_filtrado) > 0 else -23.5505
            center_lon = df_mapa_filtrado['Longitude'].mean() if len(df_mapa_filtrado) > 0 else -46.6333
            
            # Criar figura do mapa
            fig_map = go.Figure()
            
            # Adicionar campo de futebol como shapes (se selecionado)
            if show_field:
                # Adicionar shapes do campo
                fig_map.add_shape(
                    type="rect",
                    x0=center_lon - 0.0005, x1=center_lon + 0.0005,
                    y0=center_lat - 0.0003, y1=center_lat + 0.0003,
                    line=dict(color="white", width=2),
                    fillcolor="rgba(34, 139, 34, 0.2)",
                    layer="below"
                )
            
            # Adicionar trilha do percurso
            if len(df_mapa_filtrado) > 1:
                if show_heatmap:
                    # Mapa de calor de velocidade
                    fig_map.add_trace(go.Densitymapbox(
                        lat=df_mapa_filtrado['Latitude'],
                        lon=df_mapa_filtrado['Longitude'],
                        z=df_mapa_filtrado['Velocity'],
                        radius=10,
                        colorscale='Viridis',
                        showscale=True,
                        name='Velocidade',
                        colorbar=dict(title="Velocidade<br>(km/h)")
                    ))
                else:
                    # Pontos coloridos por velocidade
                    # Criar texto com horário formatado
                    hover_texts = []
                    for _, row in df_mapa_filtrado.iterrows():
                        time_str = seconds_to_time_str(row['Seconds'], start_dt_mapa)
                        hover_texts.append(f"<b>Tempo:</b> {time_str}<br><b>Velocidade:</b> {row['Velocity']:.1f} km/h<br><b>FC:</b> {row['HeartRate']:.0f} bpm")
                    
                    fig_map.add_trace(go.Scattermapbox(
                        lat=df_mapa_filtrado['Latitude'],
                        lon=df_mapa_filtrado['Longitude'],
                        mode='markers',
                        marker=dict(
                            size=6,
                            color=df_mapa_filtrado['Velocity'],
                            colorscale='Viridis',
                            showscale=True,
                            colorbar=dict(title="Velocidade<br>(km/h)", x=1.02),
                            cmin=df_mapa_filtrado['Velocity'].min(),
                            cmax=df_mapa_filtrado['Velocity'].max()
                        ),
                        text=hover_texts,
                        hoverinfo='text',
                        name='Posições'
                    ))
            
            # Adicionar marcadores de início e fim
            if len(df_mapa_filtrado) > 0:
                start_time_str = seconds_to_time_str(start_time_global, start_dt_mapa)
                end_time_str = seconds_to_time_str(end_time_global, start_dt_mapa)
                
                fig_map.add_trace(go.Scattermapbox(
                    lat=[df_mapa_filtrado['Latitude'].iloc[0]],
                    lon=[df_mapa_filtrado['Longitude'].iloc[0]],
                    mode='markers',
                    marker=dict(size=16, color='green', symbol='circle'),
                    text=[f"🏁 INÍCIO<br>Horário: {start_time_str}<br>Vel: {df_mapa_filtrado['Velocity'].iloc[0]:.1f} km/h"],
                    hoverinfo='text',
                    name='Início'
                ))
                
                fig_map.add_trace(go.Scattermapbox(
                    lat=[df_mapa_filtrado['Latitude'].iloc[-1]],
                    lon=[df_mapa_filtrado['Longitude'].iloc[-1]],
                    mode='markers',
                    marker=dict(size=16, color='red', symbol='circle'),
                    text=[f"🏁 FIM<br>Horário: {end_time_str}<br>Vel: {df_mapa_filtrado['Velocity'].iloc[-1]:.1f} km/h"],
                    hoverinfo='text',
                    name='Fim'
                ))
            
            # Configurar layout do mapa
            mapbox_center = dict(lat=center_lat, lon=center_lon)
            
            # Ajustar zoom
            if len(df_mapa_filtrado) > 1:
                lat_range = df_mapa_filtrado['Latitude'].max() - df_mapa_filtrado['Latitude'].min()
                lon_range = df_mapa_filtrado['Longitude'].max() - df_mapa_filtrado['Longitude'].min()
                zoom_level = 18 - min(lat_range * 100, lon_range * 100)
                zoom_level = max(14, min(zoom_level, 18))
            else:
                zoom_level = 15
            
            fig_map.update_layout(
                mapbox=dict(
                    style=map_style,
                    center=mapbox_center,
                    zoom=zoom_level
                ),
                margin=dict(l=0, r=0, t=30, b=0),
                height=700,
                title={
                    'text': f"Trajetória de {atleta_mapa_nome} - {periodo_mapa} | {format_time_range(start_time_global, end_time_global, reference_datetime)}",
                    'x': 0.5,
                    'xanchor': 'center'
                },
                hovermode='closest'
            )
            
            st.plotly_chart(fig_map, use_container_width=True)
            
            # Estatísticas do percurso
            with st.expander("📊 Estatísticas do Percurso Selecionado"):
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Distância percorrida", f"{df_mapa_filtrado['Odometer'].max() - df_mapa_filtrado['Odometer'].min():.0f} m")
                with col2:
                    st.metric("Velocidade média", f"{df_mapa_filtrado['Velocity'].mean():.1f} km/h")
                with col3:
                    st.metric("Velocidade máxima", f"{df_mapa_filtrado['Velocity'].max():.1f} km/h")
                with col4:
                    st.metric("FC média", f"{df_mapa_filtrado['HeartRate'].mean():.0f} bpm")
        
        # ==================== TAB 2: GRÁFICOS DE DESEMPENHO COM SOBREPOSIÇÃO ====================
        with tab2:
            st.subheader("Análise de Desempenho ao Longo do Tempo")
            st.markdown("### 📈 Velocidade e Frequência Cardíaca - Análise Temporal")
            
            # Selecionar atleta para esta análise
            if len(selected_atletas) > 1:
                atleta_temporal = st.selectbox(
                    "Selecione o atleta para análise temporal",
                    options=selected_atletas,
                    index=0,
                    key="temporal_atleta_select"
                )
                idx_temporal = selected_atletas.index(atleta_temporal)
                df_temporal = dfs_filtered[idx_temporal].copy()
                atleta_temporal_nome = atleta_temporal
                start_dt_temporal = selected_start_datetimes[idx_temporal]
            else:
                df_temporal = df_main.copy()
                atleta_temporal_nome = atleta_main
                start_dt_temporal = start_dt_main
            
            # Criar coluna de tempo formatado para o eixo X
            df_temporal['Tempo_Horario'] = df_temporal['Seconds'].apply(
                lambda x: seconds_to_time_str(x, start_dt_temporal)
            )
            
            # Criar gráfico com sobreposição de velocidade e frequência cardíaca
            fig_temporal = make_subplots(specs=[[{"secondary_y": True}]])
            
            # Adicionar velocidade
            fig_temporal.add_trace(
                go.Scatter(
                    x=df_temporal['Seconds'],
                    y=df_temporal['Velocity'],
                    mode='lines',
                    name='Velocidade',
                    line=dict(color='#3498db', width=2),
                    fill='tozeroy',
                    fillcolor='rgba(52, 152, 219, 0.1)'
                ),
                secondary_y=False
            )
            
            # Adicionar frequência cardíaca
            fig_temporal.add_trace(
                go.Scatter(
                    x=df_temporal['Seconds'],
                    y=df_temporal['HeartRate'],
                    mode='lines',
                    name='Frequência Cardíaca',
                    line=dict(color='#e74c3c', width=2),
                    fill='tozeroy',
                    fillcolor='rgba(231, 76, 60, 0.1)'
                ),
                secondary_y=True
            )
            
            # Adicionar linha da média de velocidade
            vel_mean = df_temporal['Velocity'].mean()
            fig_temporal.add_hline(
                y=vel_mean, line_dash="dash", line_color="#3498db",
                annotation_text=f"Vel Média: {vel_mean:.1f} km/h",
                annotation_position="top right",
                secondary_y=False
            )
            
            # Adicionar linha da média de FC
            hr_mean = df_temporal['HeartRate'].mean()
            fig_temporal.add_hline(
                y=hr_mean, line_dash="dash", line_color="#e74c3c",
                annotation_text=f"FC Média: {hr_mean:.0f} bpm",
                annotation_position="bottom right",
                secondary_y=True
            )
            
            # Adicionar zonas de intensidade (FC)
            fc_max = df_temporal['HeartRate'].max()
            fig_temporal.add_hrect(
                y0=0, y1=fc_max*0.6, 
                fillcolor="lightgreen", opacity=0.2,
                line_width=0, secondary_y=True
            )
            fig_temporal.add_hrect(
                y0=fc_max*0.6, y1=fc_max*0.75, 
                fillcolor="yellow", opacity=0.2,
                line_width=0, secondary_y=True
            )
            fig_temporal.add_hrect(
                y0=fc_max*0.75, y1=fc_max*0.9, 
                fillcolor="orange", opacity=0.2,
                line_width=0, secondary_y=True
            )
            fig_temporal.add_hrect(
                y0=fc_max*0.9, y1=fc_max, 
                fillcolor="red", opacity=0.2,
                line_width=0, secondary_y=True
            )
            
            # Configurar layout
            fig_temporal.update_layout(
                title=f"Velocidade e Frequência Cardíaca - {atleta_temporal_nome}",
                xaxis_title="Tempo (segundos)",
                height=500,
                hovermode='x unified',
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
            )
            
            fig_temporal.update_yaxes(title_text="Velocidade (km/h)", secondary_y=False, color="#3498db")
            fig_temporal.update_yaxes(title_text="Frequência Cardíaca (bpm)", secondary_y=True, color="#e74c3c")
            
            st.plotly_chart(fig_temporal, use_container_width=True)
            
            # Gráfico com horário no eixo X
            st.markdown("### 📅 Visualização por Horário")
            
            fig_time = make_subplots(specs=[[{"secondary_y": True}]])
            
            fig_time.add_trace(
                go.Scatter(
                    x=df_temporal['Tempo_Horario'],
                    y=df_temporal['Velocity'],
                    mode='lines',
                    name='Velocidade',
                    line=dict(color='#3498db', width=2)
                ),
                secondary_y=False
            )
            
            fig_time.add_trace(
                go.Scatter(
                    x=df_temporal['Tempo_Horario'],
                    y=df_temporal['HeartRate'],
                    mode='lines',
                    name='Frequência Cardíaca',
                    line=dict(color='#e74c3c', width=2)
                ),
                secondary_y=True
            )
            
            fig_time.update_layout(
                title=f"Velocidade e Frequência Cardíaca - {atleta_temporal_nome} (por Horário)",
                xaxis_title="Horário",
                height=450,
                hovermode='x unified'
            )
            fig_time.update_yaxes(title_text="Velocidade (km/h)", secondary_y=False, color="#3498db")
            fig_time.update_yaxes(title_text="Frequência Cardíaca (bpm)", secondary_y=True, color="#e74c3c")
            
            st.plotly_chart(fig_time, use_container_width=True)
            
            # Adicionar gráfico de dispersão Velocidade vs FC
            st.markdown("### 🎯 Relação Velocidade vs Frequência Cardíaca")
            
            fig_scatter = px.scatter(
                df_temporal,
                x='Velocity',
                y='HeartRate',
                color='Seconds',
                color_continuous_scale='Viridis',
                title="Relação Velocidade vs Frequência Cardíaca",
                labels={'Velocity': 'Velocidade (km/h)', 'HeartRate': 'Frequência Cardíaca (bpm)'}
            )
            fig_scatter.update_layout(height=450)
            st.plotly_chart(fig_scatter, use_container_width=True)
            
            # Comparação entre atletas (se mais de um selecionado)
            if len(selected_atletas) > 1:
                st.markdown("### 👥 Comparação entre Atletas")
                
                fig_comp = go.Figure()
                for df, atleta, start_dt in zip(dfs_filtered, selected_atletas, selected_start_datetimes):
                    fig_comp.add_trace(go.Scatter(
                        x=df['Seconds'],
                        y=df['Velocity'],
                        mode='lines',
                        name=f'{atleta} - Velocidade',
                        line=dict(width=2)
                    ))
                
                fig_comp.update_layout(
                    title="Comparação de Velocidade entre Atletas",
                    xaxis_title="Tempo (segundos)",
                    yaxis_title="Velocidade (km/h)",
                    height=450,
                    hovermode='x unified'
                )
                st.plotly_chart(fig_comp, use_container_width=True)
        
        # ==================== TAB 3: VELOCIDADE E ACELERAÇÃO ====================
        with tab3:
            st.subheader("Análise de Velocidade e Aceleração")
            
            if len(selected_atletas) > 1:
                atleta_vel = st.selectbox(
                    "Selecione o atleta para análise de velocidade",
                    options=selected_atletas,
                    index=0,
                    key="vel_atleta_select"
                )
                idx_vel = selected_atletas.index(atleta_vel)
                df_vel = dfs_filtered[idx_vel].copy()
            else:
                df_vel = df_main.copy()
            
            col1, col2 = st.columns(2)
            
            with col1:
                fig_hist = px.histogram(df_vel, x='Velocity', nbins=40,
                                       title="Distribuição de Velocidades",
                                       labels={'Velocity': 'Velocidade (km/h)'},
                                       color_discrete_sequence=['blue'])
                fig_hist.add_vline(x=df_vel['Velocity'].mean(), 
                                   line_dash="dash", line_color="red",
                                   annotation_text=f"Média: {df_vel['Velocity'].mean():.1f} km/h")
                st.plotly_chart(fig_hist, use_container_width=True)
            
            with col2:
                df_vel['Percentil_Tempo'] = pd.cut(df_vel['Seconds'], 
                                                   bins=10,
                                                   labels=[f'{i*10}-{(i+1)*10}%' for i in range(10)])
                fig_box = px.box(df_vel, x='Percentil_Tempo', y='Velocity',
                                title="Velocidade por Percentil do Tempo",
                                labels={'Velocity': 'Velocidade (km/h)', 
                                       'Percentil_Tempo': 'Percentil do Jogo'})
                st.plotly_chart(fig_box, use_container_width=True)
            
            if 'Acceleration' in df_vel.columns:
                st.subheader("Aceleração ao longo do tempo")
                fig_acc = go.Figure()
                fig_acc.add_trace(go.Scatter(x=df_vel['Seconds'], y=df_vel['Acceleration'],
                                             mode='lines', name='Aceleração',
                                             line=dict(color='purple', width=1.5)))
                fig_acc.add_hline(y=0, line_dash="dash", line_color="black", 
                                 annotation_text="Repouso", annotation_position="top right")
                fig_acc.update_layout(xaxis_title="Tempo (s)", yaxis_title="Aceleração (m/s²)", height=450)
                st.plotly_chart(fig_acc, use_container_width=True)
        
        # ==================== TAB 4: FREQUÊNCIA CARDÍACA ====================
        with tab4:
            st.subheader("Análise da Frequência Cardíaca")
            
            if len(selected_atletas) > 1:
                atleta_fc = st.selectbox(
                    "Selecione o atleta para análise de FC",
                    options=selected_atletas,
                    index=0,
                    key="fc_atleta_select"
                )
                idx_fc = selected_atletas.index(atleta_fc)
                df_fc = dfs_filtered[idx_fc].copy()
            else:
                df_fc = df_main.copy()
            
            if 'HeartRate' in df_fc.columns:
                col1, col2 = st.columns(2)
                
                with col1:
                    fig_hr_hist = px.histogram(df_fc, x='HeartRate', nbins=30,
                                               title="Distribuição da Frequência Cardíaca",
                                               labels={'HeartRate': 'Frequência Cardíaca (bpm)'},
                                               color_discrete_sequence=['red'])
                    fig_hr_hist.add_vline(x=df_fc['HeartRate'].mean(), 
                                          line_dash="dash", line_color="blue",
                                          annotation_text=f"Média: {df_fc['HeartRate'].mean():.0f} bpm")
                    st.plotly_chart(fig_hr_hist, use_container_width=True)
                
                with col2:
                    fig_hr_time = go.Figure()
                    fig_hr_time.add_trace(go.Scatter(x=df_fc['Seconds'], y=df_fc['HeartRate'],
                                                     mode='lines', name='FC',
                                                     line=dict(color='red', width=2)))
                    
                    fc_max = df_fc['HeartRate'].max()
                    fig_hr_time.add_hrect(y0=0, y1=fc_max*0.6, fillcolor="lightgreen", opacity=0.3,
                                         annotation_text="Recuperação", annotation_position="top left")
                    fig_hr_time.add_hrect(y0=fc_max*0.6, y1=fc_max*0.75, fillcolor="yellow", opacity=0.3,
                                         annotation_text="Aeróbica", annotation_position="top left")
                    fig_hr_time.add_hrect(y0=fc_max*0.75, y1=fc_max*0.9, fillcolor="orange", opacity=0.3,
                                         annotation_text="Anaeróbica", annotation_position="top left")
                    fig_hr_time.add_hrect(y0=fc_max*0.9, y1=fc_max, fillcolor="red", opacity=0.3,
                                         annotation_text="Máximo", annotation_position="top left")
                    
                    fig_hr_time.update_layout(xaxis_title="Tempo (s)", yaxis_title="FC (bpm)", height=450)
                    st.plotly_chart(fig_hr_time, use_container_width=True)
                
                st.markdown("### 📊 Análise por Zona de Intensidade")
                fc_max = df_fc['HeartRate'].max()
                df_fc['Zona_FC'] = pd.cut(df_fc['HeartRate'], 
                                          bins=[0, fc_max*0.6, fc_max*0.75, fc_max*0.9, fc_max],
                                          labels=['Recuperação', 'Aeróbica', 'Anaeróbica', 'Máximo'])
                
                zona_stats = df_fc.groupby('Zona_FC', observed=True).agg({
                    'HeartRate': ['count', 'mean', 'min', 'max'],
                    'Velocity': 'mean' if 'Velocity' in df_fc.columns else lambda x: 0
                }).round(0)
                
                zona_stats.columns = ['Contagem', 'FC Média', 'FC Mín', 'FC Máx', 'Velocidade Média']
                zona_stats['% do Tempo'] = (zona_stats['Contagem'] / len(df_fc) * 100).round(1).astype(str) + '%'
                st.dataframe(zona_stats, use_container_width=True)
        
        # ==================== TAB 5: ACELERAÇÃO VS VELOCIDADE COM QUADRANTES ====================
        with tab5:
            st.subheader("🔄 Relação Aceleração vs Velocidade")
            st.markdown("""
            O gráfico abaixo mostra a relação entre **Aceleração (m/s²)** e **Velocidade (km/h)** para cada ponto de registro.
            Os quadrantes ajudam a identificar os padrões de movimento do atleta.
            """)
            
            if len(selected_atletas) > 1:
                atleta_acc = st.selectbox(
                    "Selecione o atleta para análise de aceleração",
                    options=selected_atletas,
                    index=0,
                    key="acc_atleta_select"
                )
                idx_acc = selected_atletas.index(atleta_acc)
                df_acc = dfs_filtered[idx_acc].copy()
            else:
                df_acc = df_main.copy()
            
            if 'Acceleration' in df_acc.columns and 'Velocity' in df_acc.columns:
                # Preparar dados
                acc_data = df_acc['Acceleration'].values
                vel_data = df_acc['Velocity'].values
                
                # Calcular médias para os quadrantes
                mean_acc = np.mean(acc_data)
                mean_vel = np.mean(vel_data)
                
                # Classificar quadrantes
                quadrantes = []
                for acc, vel in zip(acc_data, vel_data):
                    if acc >= mean_acc and vel >= mean_vel:
                        quadrantes.append('Q1 - Alta Vel + Alta Acel')
                    elif acc >= mean_acc and vel < mean_vel:
                        quadrantes.append('Q2 - Baixa Vel + Alta Acel')
                    elif acc < mean_acc and vel < mean_vel:
                        quadrantes.append('Q3 - Baixa Vel + Baixa Acel')
                    else:
                        quadrantes.append('Q4 - Alta Vel + Baixa Acel')
                
                df_acc['Quadrante'] = quadrantes
                
                # Cores para cada quadrante
                cores_quadrantes = {
                    'Q1 - Alta Vel + Alta Acel': '#e74c3c',
                    'Q2 - Baixa Vel + Alta Acel': '#f39c12',
                    'Q3 - Baixa Vel + Baixa Acel': '#2ecc71',
                    'Q4 - Alta Vel + Baixa Acel': '#3498db'
                }
                
                # Criar figura
                fig_acc_vel = go.Figure()
                
                for quadrante, cor in cores_quadrantes.items():
                    mask = df_acc['Quadrante'] == quadrante
                    fig_acc_vel.add_trace(go.Scatter(
                        x=df_acc[mask]['Velocity'],
                        y=df_acc[mask]['Acceleration'],
                        mode='markers',
                        name=quadrante,
                        marker=dict(size=8, color=cor, opacity=0.6, symbol='circle'),
                        text=[f"<b>Tempo:</b> {t:.1f}s<br><b>Vel:</b> {v:.1f} km/h<br><b>Acel:</b> {a:.2f} m/s²"
                              for t, v, a in zip(df_acc[mask]['Seconds'] if 'Seconds' in df_acc.columns else range(len(df_acc[mask])),
                                                df_acc[mask]['Velocity'],
                                                df_acc[mask]['Acceleration'])],
                        hoverinfo='text'
                    ))
                
                # Adicionar linhas das médias
                fig_acc_vel.add_vline(x=mean_vel, line_dash="dash", line_color="gray", 
                                     annotation_text=f"Média Vel: {mean_vel:.1f} km/h", 
                                     annotation_position="top")
                fig_acc_vel.add_hline(y=mean_acc, line_dash="dash", line_color="gray", 
                                     annotation_text=f"Média Acel: {mean_acc:.2f} m/s²", 
                                     annotation_position="right")
                
                fig_acc_vel.update_layout(
                    title="Relação Aceleração vs Velocidade",
                    xaxis_title="Velocidade (km/h)",
                    yaxis_title="Aceleração (m/s²)",
                    height=600,
                    legend_title="Quadrantes",
                    hovermode='closest'
                )
                
                st.plotly_chart(fig_acc_vel, use_container_width=True)
                
                # Estatísticas por quadrante
                st.markdown("### 📊 Estatísticas por Quadrante")
                
                quadrant_stats = df_acc.groupby('Quadrante').agg({
                    'Velocity': ['count', 'mean', 'min', 'max', 'std'],
                    'Acceleration': ['mean', 'min', 'max', 'std']
                }).round(2)
                
                quadrant_stats.columns = ['Contagem', 'Vel Média', 'Vel Mín', 'Vel Máx', 'Vel Std', 
                                         'Acel Média', 'Acel Mín', 'Acel Máx', 'Acel Std']
                quadrant_stats['% do Tempo'] = (quadrant_stats['Contagem'] / len(df_acc) * 100).round(1).astype(str) + '%'
                
                st.dataframe(quadrant_stats, use_container_width=True)
                
                # Interpretação
                with st.expander("📖 Interpretação dos Quadrantes"):
                    st.markdown("""
                    | Quadrante | Significado | Interpretação no Esporte |
                    |-----------|-------------|--------------------------|
                    | **Q1 - Alta Vel + Alta Acel** | Alta velocidade com aceleração positiva | **Esforço máximo** - Sprints, arrancadas |
                    | **Q2 - Baixa Vel + Alta Acel** | Baixa velocidade com aceleração positiva | **Partidas** - Saídas de posição parada |
                    | **Q3 - Baixa Vel + Baixa Acel** | Baixa velocidade com desaceleração | **Recuperação** - Movimentos de baixa intensidade |
                    | **Q4 - Alta Vel + Baixa Acel** | Alta velocidade com desaceleração | **Frenagem** - Desaceleração após sprint |
                    """)
                
                # Gráfico de pizza
                st.markdown("### 🎯 Distribuição do Tempo por Quadrante")
                fig_pie = px.pie(
                    quadrant_stats, 
                    values='Contagem', 
                    names=quadrant_stats.index,
                    title="Proporção de Tempo em Cada Quadrante",
                    color_discrete_sequence=['#e74c3c', '#f39c12', '#2ecc71', '#3498db']
                )
                fig_pie.update_traces(textposition='inside', textinfo='percent+label')
                st.plotly_chart(fig_pie, use_container_width=True)
        
        # ==================== TAB 6: ENTROPIA AMOSTRAL ====================
        with tab6:
            st.subheader("📊 Entropia Amostral - Regularidade dos Movimentos")
            st.markdown("""
            A **Entropia Amostral (Sample Entropy)** mede a **regularidade e previsibilidade** dos dados.
            - **Maior entropia** = maior complexidade, movimentos variados e imprevisíveis
            - **Menor entropia** = padrões mais regulares, repetitivos e previsíveis
            """)
            
            if len(selected_atletas) > 1:
                atleta_entropy = st.selectbox(
                    "Selecione o atleta para análise de entropia",
                    options=selected_atletas,
                    index=0,
                    key="entropy_atleta_select"
                )
                idx_entropy = selected_atletas.index(atleta_entropy)
                df_entropy = dfs_filtered[idx_entropy].copy()
            else:
                df_entropy = df_main.copy()
            
            var_entropy_options = {}
            if 'Velocity' in df_entropy.columns:
                var_entropy_options['Velocity'] = 'Velocidade (km/h)'
            if 'HeartRate' in df_entropy.columns:
                var_entropy_options['HeartRate'] = 'Frequência Cardíaca (bpm)'
            if 'Acceleration' in df_entropy.columns:
                var_entropy_options['Acceleration'] = 'Aceleração (m/s²)'
            if 'PlayerLoad' in df_entropy.columns:
                var_entropy_options['PlayerLoad'] = 'Carga do Jogador'
            
            selected_entropy_var = st.selectbox(
                "Selecione a variável para análise de entropia",
                options=list(var_entropy_options.keys()),
                format_func=lambda x: var_entropy_options[x],
                key="entropy_var"
            )
            
            col1, col2, col3 = st.columns(3)
            with col1:
                m_value = st.selectbox("Comprimento da sequência (m)", [1, 2, 3], index=1)
            with col2:
                r_value = st.slider("Tolerância (r)", 0.1, 0.5, 0.2, step=0.05)
            with col3:
                use_rolling = st.checkbox("Análise por janela deslizante", value=True)
            
            process_button = st.button("🚀 Processar Análise de Entropia Amostral", type="primary", use_container_width=True)
            
            if process_button:
                data_series = df_entropy[selected_entropy_var].dropna().values
                
                if len(data_series) > 30:
                    with st.spinner("Calculando entropia amostral..."):
                        sample_val = sample_entropy_fast(data_series, m=m_value, r=r_value)
                        cv_val = np.std(data_series) / np.mean(data_series) if np.mean(data_series) > 0 else 0
                    
                    st.markdown("### 📈 Métrica de Entropia Amostral Global")
                    col_a, col_b = st.columns(2)
                    
                    with col_a:
                        st.metric("Entropia Amostral", f"{sample_val:.4f}" if not np.isnan(sample_val) else "N/A")
                    with col_b:
                        st.metric("Coeficiente de Variação", f"{cv_val:.3f}")
                    
                    st.markdown("**🏷️ Classificação:**")
                    if sample_val < 0.5:
                        st.info("🔵 **Baixa entropia** - Movimento padronizado e repetitivo")
                    elif sample_val < 1.2:
                        st.success("🟢 **Média entropia** - Variabilidade normal do esporte")
                    else:
                        st.warning("🟠 **Alta entropia** - Grande variabilidade e imprevisibilidade")
                    
                    # Série temporal
                    st.markdown("### 📉 Série Temporal")
                    fig_entropy_time = go.Figure()
                    time_x = df_entropy['Seconds'].values if 'Seconds' in df_entropy.columns else range(len(data_series))
                    
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
                        with st.spinner("Calculando entropia por janela..."):
                            window_size = min(100, len(data_series) // 5)
                            positions, rolling_ent = rolling_sample_entropy(data_series, 
                                                                            window_size=window_size, 
                                                                            step=window_size//3,
                                                                            m=m_value, r=r_value)
                        
                        if len(rolling_ent) > 0:
                            st.markdown("### 📊 Evolução da Entropia Amostral")
                            fig_rolling = go.Figure()
                            
                            if 'Seconds' in df_entropy.columns:
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
                            
                            if len(rolling_ent) > 3:
                                trend = rolling_ent[-1] - rolling_ent[0]
                                if trend > 0.1:
                                    st.info("📈 **Tendência: Entropia aumentando** - Movimentos mais variáveis")
                                elif trend < -0.1:
                                    st.warning("📉 **Tendência: Entropia diminuindo** - Possível fadiga")
                                else:
                                    st.success("➡️ **Tendência: Entropia estável** - Comportamento consistente")
                    
                    # Valores de referência
                    with st.expander("📌 Valores de Referência para Entropia Amostral"):
                        st.markdown("""
                        | Categoria | Entropia Amostral | Interpretação |
                        |-----------|------------------|---------------|
                        | **Baixa** | < 0.5 | Movimento repetitivo, padrões regulares |
                        | **Média** | 0.5 - 1.2 | Variabilidade normal do esporte |
                        | **Alta** | > 1.2 | Alta complexidade, movimentos variados |
                        
                        **Aplicações:**
                        - Entropia baixa pode indicar **fadiga**
                        - Entropia alta indica **alta variabilidade** e adaptabilidade
                        """)
                else:
                    st.warning(f"⚠️ Dados insuficientes. Necessários 30 pontos. Atual: {len(data_series)}")
            else:
                st.info("👆 **Clique no botão acima para iniciar a análise de entropia amostral.**")
        
        # Botão de download
        st.markdown("---")
        csv_data = df_combined.to_csv(index=False)
        st.download_button(
            label="📥 Exportar dados filtrados (CSV)",
            data=csv_data,
            file_name=f"dados_filtrados_{len(selected_atletas)}_atletas.csv",
            mime="text/csv"
        )
        
        st.markdown("---")
        st.markdown(f"**📊 Resumo da análise:** {len(df_combined)} registros | **Atletas:** {', '.join(selected_atletas)}")
        st.markdown(f"**⏱️ Período:** {format_time_range(start_time_global, end_time_global, reference_datetime)}")
        st.markdown(f"**⚡ Filtro de velocidade:** {speed_range[0]:.1f} - {speed_range[1]:.1f} km/h")
    
    else:
        st.error("❌ Nenhum arquivo válido foi processado. Verifique o formato dos arquivos.")

else:
    # Tela inicial
    st.markdown("""
    ### 👋 Bem-vindo ao Analisador de Percurso do Atleta!
    
    Esta ferramenta permite visualizar e analisar os dados de posicionamento e desempenho de atletas.
    
    ### 🚀 Como usar:
    1. **Faça upload** de um ou mais arquivos CSV na barra lateral esquerda
    2. Aguarde o processamento automático
    3. Selecione **um ou mais atletas** no menu
    4. Ajuste os **horários de início e fim** para selecionar o intervalo desejado
    5. Explore as visualizações nas abas
    
    ### ✨ Novas funcionalidades:
    - **Múltiplos arquivos**: Carregue dados de vários atletas simultaneamente
    - **Seleção múltipla de atletas**: Analise um ou mais atletas ao mesmo tempo
    - **Filtro temporal por horário**: Selecione o intervalo usando horários HH:MM:SS
    - **Campo de futebol**: Visualização com dimensões oficiais do campo
    - **Sobreposição de gráficos**: Velocidade e FC no mesmo gráfico
    
    ---
    **👈 Clique em "Browse files" na barra lateral para começar!**
    """)
    
    st.info("ℹ️ Aguardando upload do arquivo...")
    
    with st.expander("🎯 Funcionalidades do Aplicativo"):
        st.markdown("""
        **Abas disponíveis:**
        
        1. **🗺️ Mapa do Percurso** - Com campo de futebol e marcadores temporais (início/fim)
        2. **📈 Gráficos de Desempenho** - Análise de velocidade, FC e sobreposição temporal
        3. **⚡ Velocidade e Aceleração** - Distribuição e evolução temporal
        4. **❤️ Frequência Cardíaca** - Análise de zonas de intensidade
        5. **🔄 Aceleração vs Velocidade** - Relação com quadrantes
        6. **📊 Entropia Amostral** - Análise da regularidade dos movimentos
        
        **Múltiplos arquivos:**
        - Selecione vários arquivos CSV ao mesmo tempo
        - Compare dados de diferentes atletas ou jogos
        - Escolha quais atletas visualizar em cada aba
        """)