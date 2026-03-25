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
import sqlite3
import os
from pathlib import Path
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

# ==================== BANCO DE DADOS DE ESTÁDIOS ====================

DB_PATH = Path("estadios.db")

def init_database():
    """Inicializa o banco de dados SQLite com a tabela de estádios"""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS estadios (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            nome TEXT NOT NULL,
            cidade TEXT,
            pais TEXT,
            latitude_centro REAL,
            longitude_centro REAL,
            latitude_min REAL,
            latitude_max REAL,
            longitude_min REAL,
            longitude_max REAL,
            orientacao REAL DEFAULT 0,
            data_cadastro TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    
    # Verificar se já existem dados
    cursor.execute("SELECT COUNT(*) FROM estadios")
    count = cursor.fetchone()[0]
    
    # Inserir estádios de exemplo se não existirem
    if count == 0:
        estadios_exemplo = [
            ("Maracanã", "Rio de Janeiro", "Brasil", -22.912, -43.230, -22.915, -22.909, -43.233, -43.227, 0),
            ("Morumbi", "São Paulo", "Brasil", -23.600, -46.720, -23.603, -23.597, -46.723, -46.717, 0),
            ("Allianz Parque", "São Paulo", "Brasil", -23.527, -46.678, -23.530, -23.524, -46.681, -46.675, 0),
            ("Arena Corinthians", "São Paulo", "Brasil", -23.545, -46.475, -23.548, -23.542, -46.478, -46.472, 0),
            ("Mineirão", "Belo Horizonte", "Brasil", -19.866, -43.971, -19.869, -19.863, -43.974, -43.968, 0),
            ("Beira-Rio", "Porto Alegre", "Brasil", -30.065, -51.236, -30.068, -30.062, -51.239, -51.233, 0),
            ("Arena do Grêmio", "Porto Alegre", "Brasil", -29.974, -51.192, -29.977, -29.971, -51.195, -51.189, 0),
            ("Arena da Baixada", "Curitiba", "Brasil", -25.448, -49.277, -25.451, -25.445, -49.280, -49.274, 0),
            ("Nacional de Brasília", "Brasília", "Brasil", -15.783, -47.899, -15.786, -15.780, -47.902, -47.896, 0),
            ("Castelão", "Fortaleza", "Brasil", -3.807, -38.523, -3.810, -3.804, -38.526, -38.520, 0),
        ]
        
        for estadio in estadios_exemplo:
            cursor.execute('''
                INSERT INTO estadios (nome, cidade, pais, latitude_centro, longitude_centro, 
                                     latitude_min, latitude_max, longitude_min, longitude_max, orientacao)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', estadio)
        
        conn.commit()
    
    conn.close()

def carregar_estadios():
    """Carrega lista de estádios do banco de dados"""
    conn = sqlite3.connect(DB_PATH)
    df = pd.read_sql_query("SELECT id, nome, cidade, pais FROM estadios ORDER BY nome", conn)
    conn.close()
    return df

def obter_estadio(id_estadio):
    """Obtém dados de um estádio específico"""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute('''
        SELECT id, nome, cidade, pais, latitude_centro, longitude_centro,
               latitude_min, latitude_max, longitude_min, longitude_max, orientacao
        FROM estadios WHERE id = ?
    ''', (id_estadio,))
    resultado = cursor.fetchone()
    conn.close()
    
    if resultado:
        return {
            'id': resultado[0],
            'nome': resultado[1],
            'cidade': resultado[2],
            'pais': resultado[3],
            'centro': (resultado[4], resultado[5]),
            'bounds': (resultado[6], resultado[7], resultado[8], resultado[9]),
            'orientacao': resultado[10]
        }
    return None

def adicionar_estadio(nome, cidade, pais, dados_calibracao):
    """Adiciona novo estádio ao banco de dados"""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    cursor.execute('''
        INSERT INTO estadios (nome, cidade, pais, latitude_centro, longitude_centro,
                             latitude_min, latitude_max, longitude_min, longitude_max, orientacao)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    ''', (
        nome, cidade, pais,
        dados_calibracao['centro_lat'],
        dados_calibracao['centro_lon'],
        dados_calibracao['lat_min'],
        dados_calibracao['lat_max'],
        dados_calibracao['lon_min'],
        dados_calibracao['lon_max'],
        dados_calibracao.get('orientacao', 0)
    ))
    
    conn.commit()
    conn.close()

def calibrar_estadio_from_data(df):
    """Calibra automaticamente os limites do estádio baseado nos dados do atleta"""
    # Usar percentis para evitar outliers
    lat_min = df['Latitude'].quantile(0.01)
    lat_max = df['Latitude'].quantile(0.99)
    lon_min = df['Longitude'].quantile(0.01)
    lon_max = df['Longitude'].quantile(0.99)
    
    centro_lat = (lat_min + lat_max) / 2
    centro_lon = (lon_min + lon_max) / 2
    
    # Adicionar uma margem de 5% para não cortar os pontos
    lat_padding = (lat_max - lat_min) * 0.05
    lon_padding = (lon_max - lon_min) * 0.05
    
    return {
        'centro_lat': centro_lat,
        'centro_lon': centro_lon,
        'lat_min': lat_min - lat_padding,
        'lat_max': lat_max + lat_padding,
        'lon_min': lon_min - lon_padding,
        'lon_max': lon_max + lon_padding
    }

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

def seconds_to_datetime(seconds, start_datetime):
    """Converte segundos desde o início para objeto datetime"""
    if start_datetime is None:
        return None
    return start_datetime + timedelta(seconds=seconds)

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
        
        # Criar coluna de horário formatado
        if start_datetime is not None:
            df['Horario'] = df['Seconds'].apply(lambda x: seconds_to_time_str(x, start_datetime))
        
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

# Inicializar banco de dados
init_database()

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

# ==================== SIDEBAR - SELEÇÃO DE ESTÁDIO ====================

st.sidebar.markdown("---")
st.sidebar.subheader("🏟️ Configuração do Estádio")

# Carregar estádios cadastrados
df_estadios = carregar_estadios()

if len(df_estadios) > 0:
    # Opção de seleção de estádio
    opcoes_estadio = ["Detectar automaticamente"] + df_estadios['nome'].tolist() + ["Cadastrar novo estádio"]
    selecao_estadio = st.sidebar.selectbox(
        "Selecione o estádio ou modo de detecção",
        options=opcoes_estadio,
        index=0,
        help="Escolha um estádio cadastrado ou deixe o app detectar automaticamente"
    )
    
    estadio_selecionado = None
    if selecao_estadio != "Detectar automaticamente" and selecao_estadio != "Cadastrar novo estádio":
        idx_estadio = df_estadios[df_estadios['nome'] == selecao_estadio].index[0]
        estadio_selecionado = obter_estadio(df_estadios.loc[idx_estadio, 'id'])
        if estadio_selecionado:
            st.sidebar.success(f"✅ Estádio: {estadio_selecionado['nome']}")
            st.sidebar.info(f"📍 {estadio_selecionado['cidade']}, {estadio_selecionado['pais']}")
    
    # Interface para cadastro de novo estádio
    if selecao_estadio == "Cadastrar novo estádio":
        with st.sidebar.expander("📝 Cadastrar novo estádio"):
            st.markdown("### Preencha os dados do estádio")
            nome_novo = st.text_input("Nome do estádio")
            cidade_nova = st.text_input("Cidade")
            pais_novo = st.text_input("País")
            
            st.markdown("#### Coordenadas de referência")
            st.markdown("Você pode obter essas coordenadas no Google Maps")
            lat_centro = st.number_input("Latitude do centro", value=0.0, format="%.6f")
            lon_centro = st.number_input("Longitude do centro", value=0.0, format="%.6f")
            lat_min = st.number_input("Latitude mínima (limite inferior)", value=0.0, format="%.6f")
            lat_max = st.number_input("Latitude máxima (limite superior)", value=0.0, format="%.6f")
            lon_min = st.number_input("Longitude mínima (limite esquerdo)", value=0.0, format="%.6f")
            lon_max = st.number_input("Longitude máxima (limite direito)", value=0.0, format="%.6f")
            orientacao = st.number_input("Orientação (rotação em graus)", value=0.0, help="Rotação do campo em relação ao norte")
            
            if st.button("💾 Salvar estádio", use_container_width=True):
                if nome_novo and lat_centro != 0 and lon_centro != 0:
                    dados_calibracao = {
                        'centro_lat': lat_centro,
                        'centro_lon': lon_centro,
                        'lat_min': lat_min,
                        'lat_max': lat_max,
                        'lon_min': lon_min,
                        'lon_max': lon_max,
                        'orientacao': orientacao
                    }
                    adicionar_estadio(nome_novo, cidade_nova, pais_novo, dados_calibracao)
                    st.success(f"✅ Estádio {nome_novo} cadastrado com sucesso!")
                    st.rerun()
                else:
                    st.error("❌ Preencha pelo menos o nome do estádio e as coordenadas do centro")
else:
    st.sidebar.warning("⚠️ Nenhum estádio cadastrado. Os dados serão usados sem referência de campo.")
    selecao_estadio = "Detectar automaticamente"
    estadio_selecionado = None

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
        
        # Calibração do estádio baseado nos dados do primeiro atleta
        df_calibracao = selected_data[0]
        
        if selecao_estadio == "Detectar automaticamente" or estadio_selecionado is None:
            # Detecção automática baseada nos dados
            bounds_auto = calibrar_estadio_from_data(df_calibracao)
            bounds_estadio = (bounds_auto['lat_min'], bounds_auto['lat_max'], 
                            bounds_auto['lon_min'], bounds_auto['lon_max'])
            centro_estadio = (bounds_auto['centro_lat'], bounds_auto['centro_lon'])
            nome_estadio = "Detectado automaticamente"
            st.sidebar.info(f"🔍 Estádio detectado automaticamente com base nos dados")
        else:
            # Usar estádio selecionado
            bounds_estadio = estadio_selecionado['bounds']
            centro_estadio = estadio_selecionado['centro']
            nome_estadio = estadio_selecionado['nome']
            st.sidebar.success(f"🏟️ Usando referências do {nome_estadio}")
        
        # Filtro de tempo global na sidebar (SLIDER DUPLO COM HORÁRIOS)
        st.sidebar.markdown("---")
        st.sidebar.subheader("⏱️ Filtro Temporal Global")
        
        # Encontrar o tempo mínimo e máximo entre todos os atletas selecionados
        min_time_global = float('inf')
        max_time_global = 0
        for df in selected_data:
            if 'Seconds' in df.columns:
                min_time_global = min(min_time_global, df['Seconds'].min())
                max_time_global = max(max_time_global, df['Seconds'].max())
        
        if min_time_global == float('inf'):
            min_time_global = 0
        if max_time_global == 0:
            max_time_global = 100
        
        if reference_datetime is not None:
            # Calcular horários de início e fim do jogo
            start_jogo = reference_datetime + timedelta(seconds=min_time_global)
            end_jogo = reference_datetime + timedelta(seconds=max_time_global)
            
            # Criar slider duplo com valores em segundos, mas exibindo horários
            # Converter horários para segundos para o slider
            min_seconds = min_time_global
            max_seconds = max_time_global
            
            # Criar labels para o slider (a cada 5 minutos aproximadamente)
            num_ticks = min(20, int((max_seconds - min_seconds) / 300) + 1)
            tick_vals = np.linspace(min_seconds, max_seconds, num_ticks)
            tick_labels = [seconds_to_time_str(t, reference_datetime) for t in tick_vals]
            
            # Slider duplo
            selected_range = st.sidebar.slider(
                "Selecione o intervalo de tempo",
                min_value=float(min_seconds),
                max_value=float(max_seconds),
                value=(float(min_seconds), float(max_seconds)),
                step=1.0,
                format="%d",
                key="time_range_slider"
            )
            
            start_time_global, end_time_global = selected_range
            
            # Exibir os horários selecionados
            start_horario = seconds_to_time_str(start_time_global, reference_datetime)
            end_horario = seconds_to_time_str(end_time_global, reference_datetime)
            
            st.sidebar.info(f"📅 Intervalo selecionado:")
            st.sidebar.info(f"   Início: **{start_horario}**")
            st.sidebar.info(f"   Fim: **{end_horario}**")
            st.sidebar.info(f"   Duração: **{int(end_time_global - start_time_global)}** segundos")
            
            # Adicionar um slider visual com horários (usando markdown para mostrar os ticks)
            st.sidebar.markdown("---")
            st.sidebar.markdown("### 📍 Marcadores de Tempo")
            
            # Criar uma representação visual dos marcadores
            col_m1, col_m2, col_m3 = st.sidebar.columns(3)
            with col_m1:
                st.markdown(f"**Início**\n\n{start_horario}")
            with col_m2:
                st.markdown(f"**Duração**\n\n{int(end_time_global - start_time_global)}s")
            with col_m3:
                st.markdown(f"**Fim**\n\n{end_horario}")
            
        else:
            # Fallback para segundos se não houver timestamp
            start_time_global = float(min_time_global)
            end_time_global = float(max_time_global)
            st.sidebar.info(f"⏱️ Intervalo: {start_time_global:.0f}s - {end_time_global:.0f}s")
            start_horario = f"{int(start_time_global // 3600):02d}:{int((start_time_global % 3600) // 60):02d}:{int(start_time_global % 60):02d}"
            end_horario = f"{int(end_time_global // 3600):02d}:{int((end_time_global % 3600) // 60):02d}:{int(end_time_global % 60):02d}"
        
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
        col_f1, col_f2, col_f3, col_f4 = st.columns(4)
        with col_f1:
            st.info(f"**⏱️ Intervalo de tempo:** {start_horario} → {end_horario}")
        with col_f2:
            st.info(f"**⚡ Filtro de velocidade:** {speed_range[0]:.1f} - {speed_range[1]:.1f} km/h")
        with col_f3:
            total_registros = sum(len(df) for df in dfs_filtered)
            st.info(f"**📊 Total de registros:** {total_registros:,}")
        with col_f4:
            st.info(f"**🏟️ Estádio:** {nome_estadio}")
        
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
        tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
            "🗺️ Mapa do Percurso", 
            "📈 Gráficos de Desempenho", 
            "⚡ Velocidade e Aceleração",
            "❤️ Frequência Cardíaca",
            "🔄 Aceleração vs Velocidade", 
            "📊 Entropia Amostral",
            "📐 Análise Tática por Zonas"
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
            
            df_mapa_filtrado = df_mapa.copy()
            
            st.info(f"📊 Mostrando percurso de {atleta_mapa_nome} no intervalo {start_horario} → {end_horario} | {len(df_mapa_filtrado)} pontos")
            st.info(f"🏟️ Referência: {nome_estadio}")
            
            # Calcular centro do mapa baseado no estádio ou nos dados
            if estadio_selecionado:
                center_lat, center_lon = centro_estadio
            else:
                center_lat = df_mapa_filtrado['Latitude'].mean() if len(df_mapa_filtrado) > 0 else -23.5505
                center_lon = df_mapa_filtrado['Longitude'].mean() if len(df_mapa_filtrado) > 0 else -46.6333
            
            # Criar figura do mapa
            fig_map = go.Figure()
            
            # Adicionar campo de futebol como shapes (se selecionado)
            if show_field and estadio_selecionado:
                # Desenhar o retângulo do campo baseado nos bounds do estádio
                lat_min, lat_max, lon_min, lon_max = bounds_estadio
                
                fig_map.add_shape(
                    type="rect",
                    x0=lon_min, x1=lon_max,
                    y0=lat_min, y1=lat_max,
                    line=dict(color="white", width=2),
                    fillcolor="rgba(34, 139, 34, 0.2)",
                    layer="below"
                )
                
                # Adicionar linha do meio de campo
                mid_lon = (lon_min + lon_max) / 2
                fig_map.add_shape(
                    type="line",
                    x0=mid_lon, x1=mid_lon,
                    y0=lat_min, y1=lat_max,
                    line=dict(color="white", width=1, dash="dash"),
                    layer="below"
                )
            elif show_field:
                # Fallback: desenhar um retângulo baseado nos dados
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
                    hover_texts = []
                    for _, row in df_mapa_filtrado.iterrows():
                        time_str = seconds_to_time_str(row['Seconds'], start_dt_mapa)
                        hover_texts.append(f"<b>Horário:</b> {time_str}<br><b>Velocidade:</b> {row['Velocity']:.1f} km/h<br><b>FC:</b> {row['HeartRate']:.0f} bpm")
                    
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
            
            if estadio_selecionado:
                lat_range = bounds_estadio[1] - bounds_estadio[0]
                lon_range = bounds_estadio[3] - bounds_estadio[2]
                zoom_level = 18 - min(lat_range * 100, lon_range * 100)
                zoom_level = max(14, min(zoom_level, 18))
            elif len(df_mapa_filtrado) > 1:
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
                    'text': f"Trajetória de {atleta_mapa_nome} - {periodo_mapa} | {start_horario} → {end_horario} | {nome_estadio}",
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
            
            df_temporal['Horario'] = df_temporal['Seconds'].apply(
                lambda x: seconds_to_time_str(x, start_dt_temporal)
            )
            
            fig_temporal = make_subplots(specs=[[{"secondary_y": True}]])
            
            fig_temporal.add_trace(
                go.Scatter(
                    x=df_temporal['Horario'],
                    y=df_temporal['Velocity'],
                    mode='lines',
                    name='Velocidade',
                    line=dict(color='#3498db', width=2),
                    fill='tozeroy',
                    fillcolor='rgba(52, 152, 219, 0.1)'
                ),
                secondary_y=False
            )
            
            fig_temporal.add_trace(
                go.Scatter(
                    x=df_temporal['Horario'],
                    y=df_temporal['HeartRate'],
                    mode='lines',
                    name='Frequência Cardíaca',
                    line=dict(color='#e74c3c', width=2),
                    fill='tozeroy',
                    fillcolor='rgba(231, 76, 60, 0.1)'
                ),
                secondary_y=True
            )
            
            vel_mean = df_temporal['Velocity'].mean()
            fig_temporal.add_hline(
                y=vel_mean, line_dash="dash", line_color="#3498db",
                annotation_text=f"Vel Média: {vel_mean:.1f} km/h",
                annotation_position="top right",
                secondary_y=False
            )
            
            hr_mean = df_temporal['HeartRate'].mean()
            fig_temporal.add_hline(
                y=hr_mean, line_dash="dash", line_color="#e74c3c",
                annotation_text=f"FC Média: {hr_mean:.0f} bpm",
                annotation_position="bottom right",
                secondary_y=True
            )
            
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
            
            fig_temporal.update_layout(
                title=f"Velocidade e Frequência Cardíaca - {atleta_temporal_nome}",
                xaxis_title="Horário",
                height=500,
                hovermode='x unified',
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
            )
            
            fig_temporal.update_yaxes(title_text="Velocidade (km/h)", secondary_y=False, color="#3498db")
            fig_temporal.update_yaxes(title_text="Frequência Cardíaca (bpm)", secondary_y=True, color="#e74c3c")
            
            st.plotly_chart(fig_temporal, use_container_width=True)
            
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
            
            if len(selected_atletas) > 1:
                st.markdown("### 👥 Comparação entre Atletas")
                
                fig_comp = go.Figure()
                for df, atleta, start_dt in zip(dfs_filtered, selected_atletas, selected_start_datetimes):
                    df_comp_temp = df.copy()
                    df_comp_temp['Horario'] = df_comp_temp['Seconds'].apply(lambda x: seconds_to_time_str(x, start_dt))
                    fig_comp.add_trace(go.Scatter(
                        x=df_comp_temp['Horario'],
                        y=df_comp_temp['Velocity'],
                        mode='lines',
                        name=f'{atleta} - Velocidade',
                        line=dict(width=2)
                    ))
                
                fig_comp.update_layout(
                    title="Comparação de Velocidade entre Atletas",
                    xaxis_title="Horário",
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
                    df_fc['Horario'] = df_fc['Seconds'].apply(lambda x: seconds_to_time_str(x, reference_datetime))
                    fig_hr_time.add_trace(go.Scatter(x=df_fc['Horario'], y=df_fc['HeartRate'],
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
                    
                    fig_hr_time.update_layout(xaxis_title="Horário", yaxis_title="FC (bpm)", height=450)
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
                acc_data = df_acc['Acceleration'].values
                vel_data = df_acc['Velocity'].values
                
                mean_acc = np.mean(acc_data)
                mean_vel = np.mean(vel_data)
                
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
                
                cores_quadrantes = {
                    'Q1 - Alta Vel + Alta Acel': '#e74c3c',
                    'Q2 - Baixa Vel + Alta Acel': '#f39c12',
                    'Q3 - Baixa Vel + Baixa Acel': '#2ecc71',
                    'Q4 - Alta Vel + Baixa Acel': '#3498db'
                }
                
                fig_acc_vel = go.Figure()
                
                for quadrante, cor in cores_quadrantes.items():
                    mask = df_acc['Quadrante'] == quadrante
                    fig_acc_vel.add_trace(go.Scatter(
                        x=df_acc[mask]['Velocity'],
                        y=df_acc[mask]['Acceleration'],
                        mode='markers',
                        name=quadrante,
                        marker=dict(size=8, color=cor, opacity=0.6, symbol='circle'),
                        text=[f"<b>Horário:</b> {seconds_to_time_str(t, reference_datetime)}<br><b>Vel:</b> {v:.1f} km/h<br><b>Acel:</b> {a:.2f} m/s²"
                              for t, v, a in zip(df_acc[mask]['Seconds'] if 'Seconds' in df_acc.columns else range(len(df_acc[mask])),
                                                df_acc[mask]['Velocity'],
                                                df_acc[mask]['Acceleration'])],
                        hoverinfo='text'
                    ))
                
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
                
                st.markdown("### 📊 Estatísticas por Quadrante")
                
                quadrant_stats = df_acc.groupby('Quadrante').agg({
                    'Velocity': ['count', 'mean', 'min', 'max', 'std'],
                    'Acceleration': ['mean', 'min', 'max', 'std']
                }).round(2)
                
                quadrant_stats.columns = ['Contagem', 'Vel Média', 'Vel Mín', 'Vel Máx', 'Vel Std', 
                                         'Acel Média', 'Acel Mín', 'Acel Máx', 'Acel Std']
                quadrant_stats['% do Tempo'] = (quadrant_stats['Contagem'] / len(df_acc) * 100).round(1).astype(str) + '%'
                
                st.dataframe(quadrant_stats, use_container_width=True)
                
                with st.expander("📖 Interpretação dos Quadrantes"):
                    st.markdown("""
                    | Quadrante | Significado | Interpretação no Esporte |
                    |-----------|-------------|--------------------------|
                    | **Q1 - Alta Vel + Alta Acel** | Alta velocidade com aceleração positiva | **Esforço máximo** - Sprints, arrancadas |
                    | **Q2 - Baixa Vel + Alta Acel** | Baixa velocidade com aceleração positiva | **Partidas** - Saídas de posição parada |
                    | **Q3 - Baixa Vel + Baixa Acel** | Baixa velocidade com desaceleração | **Recuperação** - Movimentos de baixa intensidade |
                    | **Q4 - Alta Vel + Baixa Acel** | Alta velocidade com desaceleração | **Frenagem** - Desaceleração após sprint |
                    """)
                
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
                    
                    st.markdown("### 📉 Série Temporal")
                    fig_entropy_time = go.Figure()
                    time_x = df_entropy['Seconds'].values if 'Seconds' in df_entropy.columns else range(len(data_series))
                    horarios = [seconds_to_time_str(t, reference_datetime) for t in time_x]
                    
                    fig_entropy_time.add_trace(
                        go.Scatter(x=horarios, y=data_series, mode='lines', 
                                  line=dict(color='blue', width=2),
                                  name=var_entropy_options[selected_entropy_var])
                    )
                    
                    mean_val = np.mean(data_series)
                    fig_entropy_time.add_hline(y=mean_val, line_dash="dash", line_color="green", 
                                              annotation_text=f"Média: {mean_val:.2f}")
                    
                    fig_entropy_time.update_layout(
                        title=f"Série Temporal - {var_entropy_options[selected_entropy_var]}",
                        xaxis_title="Horário",
                        yaxis_title=var_entropy_options[selected_entropy_var],
                        height=400
                    )
                    st.plotly_chart(fig_entropy_time, use_container_width=True)
                    
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
                                horarios_pos = [seconds_to_time_str(t, reference_datetime) for t in time_positions]
                            else:
                                horarios_pos = positions
                            
                            fig_rolling.add_trace(
                                go.Scatter(x=horarios_pos, y=rolling_ent, mode='lines+markers',
                                          line=dict(color='red', width=2), marker=dict(size=6),
                                          name="Entropia Amostral")
                            )
                            fig_rolling.add_hline(y=np.nanmean(rolling_ent), line_dash="dash", 
                                                  line_color="orange", annotation_text=f"Média: {np.nanmean(rolling_ent):.3f}")
                            
                            fig_rolling.update_layout(
                                title="Entropia Amostral por Janela Deslizante",
                                xaxis_title="Horário",
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
        
        # ==================== TAB 7: ANÁLISE TÁTICA POR ZONAS ====================
        with tab7:
            st.subheader("📐 Análise Tática Integrada: Demanda Física por Zona do Campo")
            st.markdown("""
            Esta ferramenta permite analisar como o atleta distribui sua **demanda física** (tempo, distância, intensidade) 
            pelas **diferentes zonas do campo**, auxiliando na integração de dados físicos e táticos.
            """)
            
            if len(selected_atletas) > 1:
                atleta_tatico = st.selectbox(
                    "Selecione o atleta para análise tática",
                    options=selected_atletas,
                    index=0,
                    key="tatico_atleta_select"
                )
                idx_tatico = selected_atletas.index(atleta_tatico)
                df_tatico = dfs_filtered[idx_tatico].copy()
                atleta_tatico_nome = atleta_tatico
                start_dt_tatico = selected_start_datetimes[idx_tatico]
            else:
                df_tatico = df_main.copy()
                atleta_tatico_nome = atleta_main
                start_dt_tatico = start_dt_main
            
            st.markdown("#### 🎚️ Filtros Específicos para Análise Tática")
            col_filtro1, col_filtro2 = st.columns(2)
            
            with col_filtro1:
                use_custom_time = st.checkbox("Usar intervalo de tempo personalizado para análise tática", value=False)
                if use_custom_time:
                    min_sec_custom = float(df_tatico['Seconds'].min())
                    max_sec_custom = float(df_tatico['Seconds'].max())
                    custom_range = st.slider(
                        "Selecione o intervalo de tempo para análise tática",
                        min_value=min_sec_custom,
                        max_value=max_sec_custom,
                        value=(min_sec_custom, max_sec_custom),
                        step=1.0,
                        format="%d",
                        key="tatico_time_slider"
                    )
                    start_time_tatico, end_time_tatico = custom_range
                    time_filter_tatico = (df_tatico['Seconds'] >= start_time_tatico) & (df_tatico['Seconds'] <= end_time_tatico)
                    df_tatico = df_tatico[time_filter_tatico].copy()
                    st.info(f"Intervalo personalizado: {seconds_to_time_str(start_time_tatico, start_dt_tatico)} → {seconds_to_time_str(end_time_tatico, start_dt_tatico)}")
                else:
                    st.info(f"Usando filtro global: {start_horario} → {end_horario}")
            
            with col_filtro2:
                use_custom_speed = st.checkbox("Usar filtro de velocidade personalizado", value=False)
                if use_custom_speed:
                    min_speed_custom = float(df_tatico['Velocity'].min())
                    max_speed_custom = float(df_tatico['Velocity'].max())
                    custom_speed_range = st.slider(
                        "Filtro de velocidade (km/h)",
                        min_value=min_speed_custom,
                        max_value=max_speed_custom,
                        value=(min_speed_custom, max_speed_custom),
                        step=0.5,
                        key="tatico_speed_slider"
                    )
                    speed_filter_tatico = (df_tatico['Velocity'] >= custom_speed_range[0]) & (df_tatico['Velocity'] <= custom_speed_range[1])
                    df_tatico = df_tatico[speed_filter_tatico].copy()
                    st.info(f"Velocidade filtrada: {custom_speed_range[0]:.1f} - {custom_speed_range[1]:.1f} km/h")
                else:
                    st.info(f"Usando filtro global: {speed_range[0]:.1f} - {speed_range[1]:.1f} km/h")
            
            if len(df_tatico) == 0:
                st.warning("⚠️ Nenhum dado disponível com os filtros selecionados.")
                st.stop()
            
            st.markdown("---")
            
            st.markdown("#### 🧩 Configuração da Divisão do Campo")
            col_div1, col_div2 = st.columns(2)
            
            with col_div1:
                num_linhas = st.number_input("Número de linhas (divisão horizontal)", min_value=1, max_value=10, value=3, step=1,
                                            help="Divide o campo longitudinalmente (ex: 3 = terços defensivo, médio, ofensivo)")
                mostrar_linhas = st.checkbox("Mostrar divisão em linhas no gráfico", value=True)
            
            with col_div2:
                num_colunas = st.number_input("Número de colunas (divisão vertical)", min_value=1, max_value=10, value=3, step=1,
                                            help="Divide o campo transversalmente (ex: 3 = corredores esquerdo, central, direito)")
                mostrar_colunas = st.checkbox("Mostrar divisão em colunas no gráfico", value=True)
            
            if estadio_selecionado:
                lat_min, lat_max, lon_min, lon_max = bounds_estadio
                st.info(f"Usando limites do {nome_estadio} para zoneamento")
            else:
                lat_min, lat_max = df_tatico['Latitude'].min(), df_tatico['Latitude'].max()
                lon_min, lon_max = df_tatico['Longitude'].min(), df_tatico['Longitude'].max()
                st.info("Usando limites baseados nos dados do atleta")
            
            linhas_bins = np.linspace(lat_min, lat_max, num_linhas + 1)
            colunas_bins = np.linspace(lon_min, lon_max, num_colunas + 1)
            
            df_tatico['Zona_Linha'] = pd.cut(df_tatico['Latitude'], bins=linhas_bins, labels=[f'L{i+1}' for i in range(num_linhas)], include_lowest=True)
            df_tatico['Zona_Coluna'] = pd.cut(df_tatico['Longitude'], bins=colunas_bins, labels=[f'C{i+1}' for i in range(num_colunas)], include_lowest=True)
            df_tatico['Zona'] = df_tatico['Zona_Linha'].astype(str) + '-' + df_tatico['Zona_Coluna'].astype(str)
            
            st.markdown("#### 📊 Demanda Física por Zona")
            
            zona_metrics = df_tatico.groupby('Zona', observed=True).agg({
                'Seconds': 'count',
                'Velocity': ['mean', 'max'],
                'HeartRate': ['mean', 'max'],
            }).round(2)
            
            zona_metrics.columns = ['Contagem', 'Vel_Média', 'Vel_Máx', 'FC_Média', 'FC_Máx']
            
            if len(df_tatico) > 1:
                sample_rate = df_tatico['Seconds'].diff().median()
                zona_metrics['Tempo_Total(s)'] = zona_metrics['Contagem'] * sample_rate
                zona_metrics['Tempo_Total(min)'] = zona_metrics['Tempo_Total(s)'] / 60
            else:
                zona_metrics['Tempo_Total(s)'] = 0
                zona_metrics['Tempo_Total(min)'] = 0
            
            if 'Odometer' in df_tatico.columns:
                df_tatico = df_tatico.sort_values('Seconds')
                df_tatico['Zona_Anterior'] = df_tatico['Zona'].shift(1)
                df_tatico['Delta_Odometer'] = df_tatico['Odometer'].diff()
                df_tatico['Dist_Zona'] = df_tatico.apply(
                    lambda row: row['Delta_Odometer'] if row['Zona'] == row['Zona_Anterior'] else 0, axis=1
                )
                dist_por_zona = df_tatico.groupby('Zona')['Dist_Zona'].sum().round(0)
                zona_metrics['Distância(m)'] = dist_por_zona
            else:
                zona_metrics['Distância(m)'] = 0
            
            zona_metrics['Intensidade'] = (zona_metrics['Vel_Média'] * zona_metrics['Contagem']) / zona_metrics['Contagem'].sum() * 100
            
            st.dataframe(zona_metrics.style.format({
                'Contagem': '{:.0f}',
                'Vel_Média': '{:.1f}',
                'Vel_Máx': '{:.1f}',
                'FC_Média': '{:.0f}',
                'FC_Máx': '{:.0f}',
                'Tempo_Total(s)': '{:.1f}',
                'Tempo_Total(min)': '{:.1f}',
                'Distância(m)': '{:.0f}',
                'Intensidade': '{:.1f}%'
            }), use_container_width=True)
            
            st.markdown("#### 🗺️ Visualização Tática")
            
            viz_type = st.radio(
                "Tipo de visualização",
                options=["Trajetória com cores por zona", "Mapa de calor de tempo", "Mapa de calor de velocidade"],
                horizontal=True,
                key="tatico_viz_type"
            )
            
            fig_tatico = go.Figure()
            
            fig_tatico.add_shape(
                type="rect",
                x0=lon_min, x1=lon_max,
                y0=lat_min, y1=lat_max,
                line=dict(color="white", width=2),
                fillcolor="rgba(34, 139, 34, 0.2)",
                layer="below"
            )
            
            if mostrar_linhas:
                for linha in linhas_bins[1:-1]:
                    fig_tatico.add_shape(
                        type="line",
                        x0=lon_min, x1=lon_max,
                        y0=linha, y1=linha,
                        line=dict(color="rgba(255,255,255,0.5)", width=1, dash="dash"),
                        layer="below"
                    )
            
            if mostrar_colunas:
                for coluna in colunas_bins[1:-1]:
                    fig_tatico.add_shape(
                        type="line",
                        x0=coluna, x1=coluna,
                        y0=lat_min, y1=lat_max,
                        line=dict(color="rgba(255,255,255,0.5)", width=1, dash="dash"),
                        layer="below"
                    )
            
            if viz_type == "Trajetória com cores por zona":
                zonas_unicas = df_tatico['Zona'].unique()
                cores = px.colors.qualitative.Set3
                cor_por_zona = {zona: cores[i % len(cores)] for i, zona in enumerate(zonas_unicas)}
                
                for zona, group in df_tatico.groupby('Zona'):
                    fig_tatico.add_trace(go.Scatter(
                        x=group['Longitude'],
                        y=group['Latitude'],
                        mode='markers',
                        name=f'Zona {zona}',
                        marker=dict(size=4, opacity=0.7, color=cor_por_zona[zona]),
                        text=[f"<b>Zona:</b> {zona}<br><b>Horário:</b> {seconds_to_time_str(t, start_dt_tatico)}<br><b>Vel:</b> {v:.1f} km/h<br><b>FC:</b> {fc:.0f} bpm"
                              for t, v, fc in zip(group['Seconds'], group['Velocity'], group['HeartRate'])],
                        hoverinfo='text',
                        showlegend=True
                    ))
            
            elif viz_type == "Mapa de calor de tempo":
                heatmap_data = np.zeros((num_linhas, num_colunas))
                for i in range(num_linhas):
                    for j in range(num_colunas):
                        zona_label = f'L{i+1}-C{j+1}'
                        count = zona_metrics.loc[zona_label, 'Contagem'] if zona_label in zona_metrics.index else 0
                        heatmap_data[i, j] = count
                
                fig_tatico.add_trace(go.Heatmap(
                    x=colunas_bins,
                    y=linhas_bins,
                    z=heatmap_data,
                    colorscale='Hot',
                    opacity=0.7,
                    colorbar=dict(title="Tempo gasto<br>(nº registros)"),
                    name="Mapa de calor de tempo"
                ))
                
                fig_tatico.add_trace(go.Scatter(
                    x=df_tatico['Longitude'],
                    y=df_tatico['Latitude'],
                    mode='markers',
                    marker=dict(size=3, color='white', opacity=0.5),
                    name='Trajetória',
                    hoverinfo='skip'
                ))
            
            elif viz_type == "Mapa de calor de velocidade":
                heatmap_data_vel = np.zeros((num_linhas, num_colunas))
                for i in range(num_linhas):
                    for j in range(num_colunas):
                        zona_label = f'L{i+1}-C{j+1}'
                        vel_mean = zona_metrics.loc[zona_label, 'Vel_Média'] if zona_label in zona_metrics.index else 0
                        heatmap_data_vel[i, j] = vel_mean
                
                fig_tatico.add_trace(go.Heatmap(
                    x=colunas_bins,
                    y=linhas_bins,
                    z=heatmap_data_vel,
                    colorscale='Viridis',
                    opacity=0.7,
                    colorbar=dict(title="Velocidade média<br>(km/h)"),
                    name="Mapa de calor de velocidade"
                ))
                
                fig_tatico.add_trace(go.Scatter(
                    x=df_tatico['Longitude'],
                    y=df_tatico['Latitude'],
                    mode='markers',
                    marker=dict(size=3, color='white', opacity=0.5),
                    name='Trajetória',
                    hoverinfo='skip'
                ))
            
            fig_tatico.update_layout(
                title=f"Análise Tática - {atleta_tatico_nome}",
                xaxis_title="Longitude",
                yaxis_title="Latitude",
                height=700,
                hovermode='closest',
                legend=dict(
                    orientation="h",
                    yanchor="bottom",
                    y=1.02,
                    xanchor="right",
                    x=1
                )
            )
            
            fig_tatico.update_yaxes(
                scaleanchor="x",
                scaleratio=1
            )
            
            st.plotly_chart(fig_tatico, use_container_width=True)
            
            st.markdown("#### 📈 Comparação de Intensidade entre Zonas")
            
            top_zonas = zona_metrics.nlargest(8, 'Intensidade').reset_index()
            
            if len(top_zonas) > 0:
                fig_radar = go.Figure()
                
                fig_radar.add_trace(go.Bar(
                    x=top_zonas['Zona'],
                    y=top_zonas['Intensidade'],
                    marker_color='rgba(255, 99, 71, 0.7)',
                    name='Intensidade Relativa (%)',
                    text=top_zonas['Intensidade'].round(1),
                    textposition='auto'
                ))
                
                fig_radar.update_layout(
                    title="Top 8 Zonas com Maior Intensidade",
                    xaxis_title="Zona",
                    yaxis_title="Intensidade Relativa (%)",
                    height=500,
                    xaxis_tickangle=-45
                )
                
                st.plotly_chart(fig_radar, use_container_width=True)
            
            csv_tatico = zona_metrics.reset_index().to_csv(index=False)
            st.download_button(
                label="📥 Exportar análise tática (CSV)",
                data=csv_tatico,
                file_name=f"analise_tatica_{atleta_tatico_nome}.csv",
                mime="text/csv"
            )
        
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
        st.markdown(f"**⏱️ Período:** {start_horario} → {end_horario}")
        st.markdown(f"**⚡ Filtro de velocidade:** {speed_range[0]:.1f} - {speed_range[1]:.1f} km/h")
        st.markdown(f"**🏟️ Estádio:** {nome_estadio}")
    
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
    3. **Selecione o estádio** ou deixe o app detectar automaticamente
    4. Selecione **um ou mais atletas** no menu
    5. **Arraste os marcadores** na barra de rolagem para selecionar o intervalo de horário desejado
    6. Explore as visualizações nas abas
    
    ### ✨ Novas funcionalidades:
    - **🏟️ Banco de dados de estádios**: Selecione o estádio ou cadastre novos
    - **🔍 Detecção automática**: O app pode identificar os limites do campo baseado nos dados
    - **📝 Cadastro de estádios**: Contribua com novos estádios para a comunidade
    - **🎯 Calibração precisa**: Use coordenadas reais dos estádios para maior precisão
    - **Múltiplos arquivos**: Carregue dados de vários atletas simultaneamente
    - **Seleção múltipla de atletas**: Analise um ou mais atletas ao mesmo tempo
    - **Barra de rolagem com horários**: Arraste os marcadores para selecionar o intervalo exato
    - **Campo de futebol**: Visualização com dimensões oficiais do campo
    - **Análise Tática por Zonas**: Integração de dados físicos e táticos
    
    ---
    **👈 Clique em "Browse files" na barra lateral para começar!**
    """)
    
    st.info("ℹ️ Aguardando upload do arquivo...")
    
    with st.expander("🎯 Funcionalidades do Aplicativo"):
        st.markdown("""
        **Abas disponíveis:**
        
        1. **🗺️ Mapa do Percurso** - Com campo de futebol e marcadores temporais
        2. **📈 Gráficos de Desempenho** - Análise de velocidade, FC e sobreposição temporal
        3. **⚡ Velocidade e Aceleração** - Distribuição e evolução temporal
        4. **❤️ Frequência Cardíaca** - Análise de zonas de intensidade
        5. **🔄 Aceleração vs Velocidade** - Relação com quadrantes
        6. **📊 Entropia Amostral** - Análise da regularidade dos movimentos
        7. **📐 Análise Tática por Zonas** - Integração de dados físicos e táticos com divisão do campo
        
        **Sistema de Estádios:**
        - **Detecção automática**: O app calcula os limites do campo baseado nos dados
        - **Estádios pré-cadastrados**: Maracanã, Morumbi, Allianz Parque, etc.
        - **Cadastro de novos**: Adicione novos estádios com suas coordenadas
        - **Persistência**: Os dados são salvos em um banco SQLite local
        """)