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
import json
import requests
from scipy.optimize import curve_fit
warnings.filterwarnings('ignore')

# Configuração da página
st.set_page_config(
    page_title="⚽ ScoutLab - Análise de Performance Esportiva",
    page_icon="⚽",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS personalizado
st.markdown("""
<style>
    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(20px); }
        to { opacity: 1; transform: translateY(0); }
    }
    .metric-card {
        background: linear-gradient(135deg, rgba(26, 26, 46, 0.95) 0%, rgba(22, 33, 62, 0.95) 100%);
        border-radius: 20px;
        padding: 20px;
        text-align: center;
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.3);
        border: 1px solid rgba(0, 210, 255, 0.3);
        backdrop-filter: blur(10px);
        transition: all 0.3s ease;
        animation: fadeIn 0.5s ease-out;
    }
    .metric-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 12px 40px rgba(0, 210, 255, 0.2);
        border-color: rgba(0, 210, 255, 0.8);
    }
    .metric-value {
        font-size: 2.5rem;
        font-weight: bold;
        background: linear-gradient(135deg, #00d2ff 0%, #3a7bd5 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
    }
    .metric-label {
        font-size: 0.85rem;
        color: #aaa;
        text-transform: uppercase;
        letter-spacing: 2px;
        margin-top: 8px;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 12px;
        background: rgba(0, 0, 0, 0.3);
        padding: 10px;
        border-radius: 50px;
        backdrop-filter: blur(10px);
    }
    .stTabs [data-baseweb="tab"] {
        background: rgba(255, 255, 255, 0.05);
        border-radius: 40px;
        padding: 10px 24px;
        font-weight: 600;
        transition: all 0.3s ease;
        border: none;
    }
    .stTabs [data-baseweb="tab"]:hover {
        background: rgba(0, 210, 255, 0.2);
    }
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, #00d2ff 0%, #3a7bd5 100%);
        color: white;
        box-shadow: 0 0 20px rgba(0, 210, 255, 0.5);
    }
    .main-header {
        background: linear-gradient(135deg, #00d2ff 0%, #3a7bd5 50%, #00ff88 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        font-size: 2.8rem;
        font-weight: bold;
        text-align: center;
        margin-bottom: 20px;
        animation: fadeIn 0.8s ease-out;
    }
    .sub-header {
        text-align: center;
        color: #00d2ff;
        font-size: 1.1rem;
        margin-bottom: 30px;
        letter-spacing: 1px;
    }
    .stButton > button {
        background: linear-gradient(135deg, #00d2ff 0%, #3a7bd5 100%);
        border: none;
        border-radius: 50px;
        padding: 12px 24px;
        font-weight: bold;
        font-size: 1rem;
        transition: all 0.3s ease;
        box-shadow: 0 4px 15px rgba(0, 210, 255, 0.3);
    }
    .stButton > button:hover {
        transform: scale(1.02);
        box-shadow: 0 8px 25px rgba(0, 210, 255, 0.5);
    }
</style>
""", unsafe_allow_html=True)

# Título do app
st.markdown('<p class="main-header">⚽ ScoutLab | Análise de Performance Esportiva</p>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">Plataforma Inteligente para Análise de Movimento e Desempenho Atlético</p>', unsafe_allow_html=True)
st.markdown("---")

# ==================== CONSTANTES DO CAMPO ====================
CAMPO_COMPRIMENTO = 105
CAMPO_LARGURA = 68
X_MIN = -CAMPO_COMPRIMENTO / 2
X_MAX = CAMPO_COMPRIMENTO / 2
Y_MIN = -CAMPO_LARGURA / 2
Y_MAX = CAMPO_LARGURA / 2

# ==================== FUNÇÃO DO PERFIL ACELERAÇÃO-VELOCIDADE ====================

def fit_velocidade_aceleracao(velocidades, aceleracoes):
    """
    Ajusta a curva de relação Aceleração-Velocidade baseada no modelo linear:
    a(v) = a0 - (a0/v0) * v
    Retorna os coeficientes da regressão linear e o R²
    """
    # Remover valores inválidos
    mask = (velocidades > 0) & (aceleracoes > -5) & (aceleracoes < 10) & (~np.isnan(velocidades)) & (~np.isnan(aceleracoes))
    v_clean = velocidades[mask]
    a_clean = aceleracoes[mask]
    
    if len(v_clean) < 5:
        return None
    
    try:
        # Regressão linear: a = b + m * v
        # Onde m é negativo (a diminui com v)
        from scipy import stats
        slope, intercept, r_value, p_value, std_err = stats.linregress(v_clean, a_clean)
        
        # slope = -a0/v0, intercept = a0
        a0 = intercept
        v0 = -a0 / slope if slope != 0 else 0
        r2 = r_value ** 2
        
        p_max = a0 * v0 / 4
        v_max = np.max(v_clean)
        
        return {
            'a0': float(a0),
            'v0': float(v0),
            'slope': float(slope),
            'intercept': float(intercept),
            'r2': float(r2),
            'v_max': float(v_max),
            'p_max': float(p_max),
            'n_points': len(v_clean)
        }
    except:
        return None

def calcular_asp_metrics(df):
    df_sprints = df[(df['Acceleration'] > 0) & (df['Velocity'] > 0)].copy()
    if len(df_sprints) < 10:
        return None
    
    if df_sprints['Velocity'].max() < 50:
        v_ms = df_sprints['Velocity'].values
    else:
        v_ms = df_sprints['Velocity'].values / 3.6
    
    a_ms2 = df_sprints['Acceleration'].values
    result = fit_velocidade_aceleracao(v_ms, a_ms2)
    
    if result:
        result['v_medio'] = np.mean(v_ms)
        result['a_medio'] = np.mean(a_ms2)
        result['num_sprints'] = len(df_sprints)
    return result

# ==================== FUNÇÕES DE CONVERSÃO ====================

@st.cache_data(ttl=3600)
def converter_gps_para_campo_cached(lat, lon, bounds):
    lat_min, lat_max, lon_min, lon_max = bounds
    norm_x = (lon - lon_min) / (lon_max - lon_min)
    norm_y = (lat - lat_min) / (lat_max - lat_min)
    campo_x = (norm_x * CAMPO_COMPRIMENTO) - (CAMPO_COMPRIMENTO / 2)
    campo_y = (norm_y * CAMPO_LARGURA) - (CAMPO_LARGURA / 2)
    return campo_x, campo_y

def desenhar_campo_futebol():
    shapes = []
    shapes.append(go.layout.Shape(type="rect", x0=X_MIN, x1=X_MAX, y0=Y_MIN, y1=Y_MAX,
                                  line=dict(color="white", width=2), fillcolor="rgba(34,139,34,0.2)"))
    shapes.append(go.layout.Shape(type="line", x0=0, x1=0, y0=Y_MIN, y1=Y_MAX,
                                  line=dict(color="white", width=2)))
    shapes.append(go.layout.Shape(type="circle", x0=-9.15, x1=9.15, y0=-9.15, y1=9.15,
                                  line=dict(color="white", width=2), fillcolor="rgba(255,255,255,0)"))
    shapes.append(go.layout.Shape(type="circle", x0=-0.3, x1=0.3, y0=-0.3, y1=0.3,
                                  line=dict(color="white", width=1), fillcolor="white"))
    
    grande_area_prof, grande_area_larg = 16.5, 40.32
    shapes.append(go.layout.Shape(type="rect", x0=X_MAX - grande_area_prof, x1=X_MAX, y0=-grande_area_larg/2, y1=grande_area_larg/2,
                                  line=dict(color="white", width=2), fillcolor="rgba(255,255,255,0)"))
    shapes.append(go.layout.Shape(type="rect", x0=X_MIN, x1=X_MIN + grande_area_prof, y0=-grande_area_larg/2, y1=grande_area_larg/2,
                                  line=dict(color="white", width=2), fillcolor="rgba(255,255,255,0)"))
    
    pequena_area_prof, pequena_area_larg = 5.5, 18.32
    shapes.append(go.layout.Shape(type="rect", x0=X_MAX - pequena_area_prof, x1=X_MAX, y0=-pequena_area_larg/2, y1=pequena_area_larg/2,
                                  line=dict(color="white", width=2), fillcolor="rgba(255,255,255,0)"))
    shapes.append(go.layout.Shape(type="rect", x0=X_MIN, x1=X_MIN + pequena_area_prof, y0=-pequena_area_larg/2, y1=pequena_area_larg/2,
                                  line=dict(color="white", width=2), fillcolor="rgba(255,255,255,0)"))
    
    penalty_dist = 11
    shapes.append(go.layout.Shape(type="circle", x0=X_MAX - penalty_dist - 0.3, x1=X_MAX - penalty_dist + 0.3, y0=-0.3, y1=0.3,
                                  line=dict(color="white", width=1), fillcolor="white"))
    shapes.append(go.layout.Shape(type="circle", x0=X_MIN + penalty_dist - 0.3, x1=X_MIN + penalty_dist + 0.3, y0=-0.3, y1=0.3,
                                  line=dict(color="white", width=1), fillcolor="white"))
    return shapes

def desenhar_linhas_divisorias(num_linhas, num_colunas):
    shapes = []
    linhas_bins = np.linspace(X_MIN, X_MAX, num_linhas + 1)
    for linha in linhas_bins[1:-1]:
        shapes.append(go.layout.Shape(type="line", x0=linha, x1=linha, y0=Y_MIN, y1=Y_MAX,
                                      line=dict(color="rgba(255,255,255,0.7)", width=2, dash="dash")))
    colunas_bins = np.linspace(Y_MIN, Y_MAX, num_colunas + 1)
    for coluna in colunas_bins[1:-1]:
        shapes.append(go.layout.Shape(type="line", x0=X_MIN, x1=X_MAX, y0=coluna, y1=coluna,
                                      line=dict(color="rgba(255,255,255,0.7)", width=2, dash="dash")))
    return shapes, linhas_bins, colunas_bins

# ==================== BANCO DE DADOS ====================

DB_PATH = Path("estadios.db")

def init_database():
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS estadios (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            nome TEXT NOT NULL,
            cidade TEXT,
            pais TEXT,
            endereco TEXT,
            latitude_centro REAL,
            longitude_centro REAL,
            latitude_min REAL,
            latitude_max REAL,
            longitude_min REAL,
            longitude_max REAL,
            pontos_json TEXT,
            data_cadastro TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    cursor.execute("SELECT COUNT(*) FROM estadios")
    count = cursor.fetchone()[0]
    if count == 0:
        estadios_exemplo = [
            ("Maracanã", "Rio de Janeiro", "Brasil", "Maracanã, Rio de Janeiro", -22.912, -43.230, -22.915, -22.909, -43.233, -43.227, None),
            ("Morumbi", "São Paulo", "Brasil", "Estádio do Morumbi, São Paulo", -23.600, -46.720, -23.603, -23.597, -46.723, -46.717, None),
            ("Allianz Parque", "São Paulo", "Brasil", "Allianz Parque, São Paulo", -23.527, -46.678, -23.530, -23.524, -46.681, -46.675, None),
            ("Castelão", "Fortaleza", "Brasil", "Arena Castelão, Fortaleza", -3.807, -38.523, -3.810, -3.804, -38.526, -38.520, None),
        ]
        for estadio in estadios_exemplo:
            cursor.execute('''INSERT INTO estadios (nome, cidade, pais, endereco, latitude_centro, longitude_centro,
                             latitude_min, latitude_max, longitude_min, longitude_max, pontos_json)
                             VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)''', estadio)
        conn.commit()
    conn.close()

def carregar_estadios():
    conn = sqlite3.connect(DB_PATH)
    df = pd.read_sql_query("SELECT id, nome, cidade, pais, endereco FROM estadios ORDER BY nome", conn)
    conn.close()
    return df

def obter_estadio(id_estadio):
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute('''SELECT id, nome, cidade, pais, endereco, latitude_centro, longitude_centro,
               latitude_min, latitude_max, longitude_min, longitude_max, pontos_json
               FROM estadios WHERE id = ?''', (id_estadio,))
    resultado = cursor.fetchone()
    conn.close()
    if resultado:
        pontos = json.loads(resultado[10]) if resultado[10] else None
        return {'id': resultado[0], 'nome': resultado[1], 'cidade': resultado[2], 'pais': resultado[3],
                'endereco': resultado[4], 'centro': (resultado[5], resultado[6]),
                'bounds': (resultado[7], resultado[8], resultado[9], resultado[10]), 'pontos': pontos}
    return None

def adicionar_estadio(nome, cidade, pais, endereco, centro_lat, centro_lon, pontos):
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    if pontos:
        lats = [p['lat'] for p in pontos]
        lons = [p['lng'] for p in pontos]
        lat_min, lat_max, lon_min, lon_max = min(lats), max(lats), min(lons), max(lons)
    else:
        lat_min, lat_max = centro_lat - 0.003, centro_lat + 0.003
        lon_min, lon_max = centro_lon - 0.003, centro_lon + 0.003
    cursor.execute('''INSERT INTO estadios (nome, cidade, pais, endereco, latitude_centro, longitude_centro,
                     latitude_min, latitude_max, longitude_min, longitude_max, pontos_json)
                     VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)''',
                   (nome, cidade, pais, endereco, centro_lat, centro_lon, lat_min, lat_max, lon_min, lon_max,
                    json.dumps(pontos) if pontos else None))
    conn.commit()
    conn.close()

def geocodificar_endereco(endereco):
    try:
        url = "https://nominatim.openstreetmap.org/search"
        params = {'q': endereco, 'format': 'json', 'limit': 1, 'addressdetails': 1}
        headers = {'User-Agent': 'AnalisePercursoAtleta/1.0'}
        response = requests.get(url, params=params, headers=headers, timeout=10)
        if response.status_code == 200:
            dados = response.json()
            if dados:
                return float(dados[0]['lat']), float(dados[0]['lon']), dados[0].get('display_name', endereco)
        return None, None, None
    except Exception:
        return None, None, None

# ==================== FUNÇÕES AUXILIARES ====================

def seconds_to_time_str(seconds, start_datetime):
    if start_datetime is None:
        return f"{int(seconds // 3600):02d}:{int((seconds % 3600) // 60):02d}:{int(seconds % 60):02d}"
    target_time = start_datetime + timedelta(seconds=seconds)
    return target_time.strftime("%H:%M:%S")

def format_duration(seconds):
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"

def extract_athlete_from_line8(content):
    try:
        lines = content.split('\n')
        if len(lines) >= 8:
            line8 = lines[7]
            if '# Athlete:' in line8:
                match = re.search(r'"([^"]*)"', line8)
                if match:
                    return match.group(1).strip()
        return None
    except Exception:
        return None

@st.cache_data
def load_data(uploaded_file):
    try:
        content = uploaded_file.getvalue().decode('utf-8')
        atleta = extract_athlete_from_line8(content)
        if atleta is None or atleta == "":
            atleta = "Não identificado"
        
        periodo = "Não identificado"
        lines = content.split('\n')
        for line in lines[:15]:
            if '# Period:' in line or '# Periodo:' in line:
                try:
                    match = re.search(r'"([^"]*)"', line)
                    periodo = match.group(1).strip() if match else line.split(':')[1].strip().strip('"').strip(';')
                except:
                    periodo = "Não identificado"
                break
        
        data_start = 0
        for i, line in enumerate(lines):
            if not line.startswith('#') and 'Timestamp' in line:
                data_start = i
                break
        
        df = pd.read_csv(StringIO(content), skiprows=data_start, sep=';')
        col_names = ['Timestamp', 'Seconds', 'Velocity', 'Acceleration', 'Odometer', 
                     'Latitude', 'Longitude', 'HeartRate', 'PlayerLoad', 'PositionalQuality', 'HDOP', 'Sats']
        
        if len(df.columns) == len(col_names):
            df.columns = col_names
        else:
            for i, col in enumerate(df.columns[:len(col_names)]):
                df.rename(columns={col: col_names[i]}, inplace=True)
        
        numeric_cols = ['Seconds', 'Velocity', 'Acceleration', 'Odometer', 'Latitude', 'Longitude', 'HeartRate', 'PlayerLoad', 'PositionalQuality', 'HDOP', 'Sats']
        for col in numeric_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col].astype(str).str.replace(',', '.'), errors='coerce')
        
        start_datetime = None
        if 'Timestamp' in df.columns:
            df['Timestamp'] = df['Timestamp'].astype(str).str.strip()
            try:
                df['Timestamp'] = pd.to_datetime(df['Timestamp'], format='%d/%m/%Y %H:%M:%S.%f', errors='coerce')
            except:
                try:
                    df['Timestamp'] = pd.to_datetime(df['Timestamp'], format='%d/%m/%Y %H:%M:%S', errors='coerce')
                except:
                    df['Timestamp'] = pd.to_datetime(df['Timestamp'], dayfirst=True, errors='coerce')
            start_datetime = df['Timestamp'].min() if not df['Timestamp'].isna().all() else None
        
        df = df.dropna(subset=['Latitude', 'Longitude', 'Velocity', 'HeartRate'])
        df['arquivo_origem'] = uploaded_file.name
        df['start_datetime'] = start_datetime
        if start_datetime is not None:
            df['Horario'] = df['Seconds'].apply(lambda x: seconds_to_time_str(x, start_datetime))
        
        return df, atleta, periodo, start_datetime
    except Exception as e:
        st.error(f"Erro ao carregar arquivo {uploaded_file.name}: {e}")
        return None, None, None, None

# ==================== INICIALIZAÇÃO ====================

init_database()

st.sidebar.header("📁 1. Upload de Arquivos")
uploaded_files = st.sidebar.file_uploader("Escolha os arquivos CSV", type=['csv'], accept_multiple_files=True)

# ==================== CARREGAMENTO DOS DADOS ====================

if uploaded_files:
    # Carregar todos os dados para o session_state
    if 'dados_carregados' not in st.session_state or st.session_state.get('arquivos_anteriores') != [f.name for f in uploaded_files]:
        with st.spinner("Carregando arquivos..."):
            all_data = []
            all_atletas = []
            all_periodos = []
            all_start_datetimes = []
            
            for file in uploaded_files:
                df, atleta, periodo, start_datetime = load_data(file)
                if df is not None and len(df) > 0:
                    all_data.append(df)
                    all_atletas.append(atleta)
                    all_periodos.append(periodo)
                    all_start_datetimes.append(start_datetime)
            
            if all_data:
                st.session_state.dados_carregados = all_data
                st.session_state.atletas = all_atletas
                st.session_state.periodos_orig = all_periodos
                st.session_state.start_datetimes = all_start_datetimes
                st.session_state.reference_datetime = all_start_datetimes[0] if all_start_datetimes[0] is not None else None
                st.session_state.arquivos_anteriores = [f.name for f in uploaded_files]
                st.session_state.dados_prontos = True
            else:
                st.session_state.dados_prontos = False
                st.error("❌ Nenhum arquivo válido processado.")
    
    if st.session_state.get('dados_prontos', False):
        # ==================== CONFIGURAÇÃO (SIDEBAR) ====================
        
        # 2. Configuração do Estádio
        st.sidebar.markdown("---")
        st.sidebar.header("🏟️ 2. Configuração do Estádio")
        
        df_estadios = carregar_estadios()
        bounds_estadio = None
        centro_estadio = None
        nome_estadio = "Não selecionado"
        
        if len(df_estadios) > 0:
            opcoes_estadio = ["Detectar automaticamente"] + df_estadios['nome'].tolist() + ["Cadastrar novo estádio"]
            selecao_estadio = st.sidebar.selectbox("Selecione o estádio ou modo de detecção", options=opcoes_estadio, index=0, key="selecao_estadio")
            
            if selecao_estadio != "Detectar automaticamente" and selecao_estadio != "Cadastrar novo estádio":
                idx_estadio = df_estadios[df_estadios['nome'] == selecao_estadio].index[0]
                estadio = obter_estadio(df_estadios.loc[idx_estadio, 'id'])
                if estadio:
                    bounds_estadio = estadio['bounds']
                    centro_estadio = estadio['centro']
                    nome_estadio = estadio['nome']
                    st.session_state.bounds_estadio = bounds_estadio
                    st.session_state.centro_estadio = centro_estadio
                    st.session_state.nome_estadio = nome_estadio
                    st.sidebar.success(f"✅ Estádio: {nome_estadio}")
            elif selecao_estadio == "Detectar automaticamente":
                df_calibracao = st.session_state.dados_carregados[0]
                lat_min = df_calibracao['Latitude'].quantile(0.01)
                lat_max = df_calibracao['Latitude'].quantile(0.99)
                lon_min = df_calibracao['Longitude'].quantile(0.01)
                lon_max = df_calibracao['Longitude'].quantile(0.99)
                centro_lat, centro_lon = (lat_min + lat_max) / 2, (lon_min + lon_max) / 2
                bounds_estadio = (lat_min, lat_max, lon_min, lon_max)
                centro_estadio = (centro_lat, centro_lon)
                nome_estadio = "Detectado automaticamente"
                st.session_state.bounds_estadio = bounds_estadio
                st.session_state.centro_estadio = centro_estadio
                st.session_state.nome_estadio = nome_estadio
                st.sidebar.info(f"🔍 Estádio detectado automaticamente")
            
            if selecao_estadio == "Cadastrar novo estádio":
                with st.sidebar.expander("📝 Cadastrar novo estádio", expanded=False):
                    st.markdown("### Dados básicos")
                    col1, col2 = st.columns(2)
                    with col1:
                        nome_novo = st.text_input("Nome do estádio*", placeholder="Ex: Castelão")
                    with col2:
                        cidade_nova = st.text_input("Cidade", placeholder="Ex: Fortaleza")
                    pais_novo = st.text_input("País", placeholder="Ex: Brasil")
                    
                    st.markdown("---")
                    st.markdown("### 🔍 Localizar estádio")
                    endereco_busca = st.text_input("Buscar local", placeholder="Ex: Arena Castelão, Fortaleza")
                    if endereco_busca:
                        with st.spinner("Buscando localização..."):
                            lat, lon, nome_encontrado = geocodificar_endereco(endereco_busca)
                            if lat and lon:
                                st.success(f"📍 Local encontrado: {nome_encontrado}")
                                st.info(f"Coordenadas: {lat:.6f}, {lon:.6f}")
                                if 'centro_lat_temp' not in st.session_state:
                                    st.session_state.centro_lat_temp = lat
                                    st.session_state.centro_lon_temp = lon
                            else:
                                st.error("❌ Local não encontrado.")
                    
                    st.markdown("---")
                    st.markdown("### 🗺️ Limites do campo")
                    center_lat = st.session_state.centro_lat_temp if 'centro_lat_temp' in st.session_state else -23.5505
                    center_lon = st.session_state.centro_lon_temp if 'centro_lon_temp' in st.session_state else -46.6333
                    
                    col_lim1, col_lim2 = st.columns(2)
                    with col_lim1:
                        lat_min = st.number_input("Latitude mínima (Sul)", value=center_lat - 0.002, format="%.8f")
                        lat_max = st.number_input("Latitude máxima (Norte)", value=center_lat + 0.002, format="%.8f")
                    with col_lim2:
                        lon_min = st.number_input("Longitude mínima (Oeste)", value=center_lon - 0.002, format="%.8f")
                        lon_max = st.number_input("Longitude máxima (Leste)", value=center_lon + 0.002, format="%.8f")
                    
                    centro_lat, centro_lon = (lat_min + lat_max) / 2, (lon_min + lon_max) / 2
                    
                    if st.button("💾 Salvar estádio", type="primary", use_container_width=True):
                        if nome_novo:
                            conn = sqlite3.connect(DB_PATH)
                            cursor = conn.cursor()
                            cursor.execute("SELECT id FROM estadios WHERE nome = ?", (nome_novo,))
                            existe = cursor.fetchone()
                            conn.close()
                            if existe:
                                st.error(f"❌ Estádio '{nome_novo}' já existe!")
                            else:
                                pontos = [{'lat': lat_max, 'lng': lon_min, 'nome': 'NO'}, {'lat': lat_max, 'lng': lon_max, 'nome': 'NE'},
                                          {'lat': lat_min, 'lng': lon_min, 'nome': 'SO'}, {'lat': lat_min, 'lng': lon_max, 'nome': 'SE'}]
                                endereco = f"{nome_novo}, {cidade_nova}, {pais_novo}"
                                adicionar_estadio(nome_novo, cidade_nova, pais_novo, endereco, centro_lat, centro_lon, pontos)
                                st.success(f"✅ Estádio {nome_novo} cadastrado!")
                                import time; time.sleep(1); st.rerun()
                        else:
                            st.error("❌ Nome do estádio é obrigatório!")
        else:
            st.sidebar.warning("Nenhum estádio cadastrado")
            selecao_estadio = "Detectar automaticamente"
        
        # 3. Divisão Temporal
        st.sidebar.markdown("---")
        st.sidebar.header("⏱️ 3. Divisão Temporal do Jogo")
        
        if 'periodos_config' not in st.session_state:
            st.session_state.periodos_config = [{"nome": "1º Tempo", "inicio": 0, "fim": 45}]
        
        if st.sidebar.button("➕ Adicionar período", use_container_width=True):
            st.session_state.periodos_config.append({"nome": f"Período {len(st.session_state.periodos_config) + 1}", "inicio": 0, "fim": 45})
            st.rerun()
        
        reference_dt = st.session_state.get('reference_datetime', None)
        
        periodos_para_remover = []
        for i, periodo in enumerate(st.session_state.periodos_config):
            with st.sidebar.expander(f"📅 {periodo['nome']}", expanded=False):
                col_nome, col_remover = st.columns([3, 1])
                with col_nome:
                    novo_nome = st.text_input("Nome", value=periodo['nome'], key=f"config_nome_{i}")
                with col_remover:
                    if i > 0:
                        if st.button("🗑️", key=f"config_remover_{i}"):
                            periodos_para_remover.append(i)
                
                col_ini, col_fim = st.columns(2)
                with col_ini:
                    novo_inicio = st.number_input("Início (min)", value=float(periodo['inicio']), step=1.0, key=f"config_inicio_{i}")
                with col_fim:
                    novo_fim = st.number_input("Fim (min)", value=float(periodo['fim']), step=1.0, key=f"config_fim_{i}")
                
                st.session_state.periodos_config[i] = {"nome": novo_nome, "inicio": novo_inicio, "fim": novo_fim}
                
                if reference_dt:
                    inicio_horario = seconds_to_time_str(novo_inicio * 60, reference_dt)
                    fim_horario = seconds_to_time_str(novo_fim * 60, reference_dt)
                    duracao_seg = (novo_fim - novo_inicio) * 60
                    duracao_str = format_duration(duracao_seg)
                    st.caption(f"🕐 {inicio_horario} → {fim_horario}  |  ⏱️ Duração: {duracao_str}")
                else:
                    st.caption("⏳ Aguardando horário de referência")
        
        for i in sorted(periodos_para_remover, reverse=True):
            st.session_state.periodos_config.pop(i)
            st.rerun()
        
        # 4. Filtros
        st.sidebar.markdown("---")
        st.sidebar.header("⚡ 4. Filtros")
        
        min_time = float('inf')
        max_time = 0
        for df in st.session_state.dados_carregados:
            min_time = min(min_time, df['Seconds'].min())
            max_time = max(max_time, df['Seconds'].max())
        
        min_time_min, max_time_min = min_time / 60, max_time / 60
        
        if reference_dt:
            tempo_range = st.sidebar.slider("Intervalo de tempo (minutos)", 
                                            min_value=float(min_time_min),
                                            max_value=float(max_time_min), 
                                            value=(float(min_time_min), float(max_time_min)), 
                                            step=0.5,
                                            key="tempo_global")
            start_time_min, end_time_min = tempo_range
            start_time, end_time = start_time_min * 60, end_time_min * 60
            start_horario_global = seconds_to_time_str(start_time, reference_dt)
            end_horario_global = seconds_to_time_str(end_time, reference_dt)
            st.sidebar.caption(f"🕐 {start_horario_global} → {end_horario_global}")
        else:
            tempo_range = st.sidebar.slider("Intervalo de tempo (minutos)", 
                                            min_value=float(min_time_min),
                                            max_value=float(max_time_min), 
                                            value=(float(min_time_min), float(max_time_min)), 
                                            step=0.5,
                                            key="tempo_global")
            start_time_min, end_time_min = tempo_range
            start_time, end_time = start_time_min * 60, end_time_min * 60
        
        max_speed = max([df['Velocity'].max() for df in st.session_state.dados_carregados])
        speed_range = st.sidebar.slider("Velocidade (km/h)", 
                                        min_value=0.0, 
                                        max_value=float(max_speed),
                                        value=(0.0, float(max_speed)), 
                                        step=0.5,
                                        key="velocidade_filtro")
        
        # 5. Seleção de Atletas
        st.sidebar.markdown("---")
        st.sidebar.header("🏅 5. Selecionar Atleta(s)")
        
        atleta_options = []
        for atleta, periodo in zip(st.session_state.atletas, st.session_state.periodos_orig):
            display_name = f"{atleta} - {periodo}" if periodo != "Não identificado" else atleta
            atleta_options.append(display_name)
        
        selected_indices = st.sidebar.multiselect(
            "Escolha os atletas",
            options=range(len(atleta_options)),
            format_func=lambda x: atleta_options[x],
            default=[0] if len(atleta_options) > 0 else []
        )
        
        # 6. Opções de Visualização
        st.sidebar.markdown("---")
        st.sidebar.header("🎨 6. Opções de Visualização")
        show_field = st.sidebar.checkbox("Mostrar campo", value=True, key="show_field")
        
        # 7. Seleção de Períodos para Análise
        st.sidebar.markdown("---")
        st.sidebar.header("📊 7. Selecionar Períodos para Análise")
        
        opcoes_periodos_analise = ["Todos os períodos"] + [p['nome'] for p in st.session_state.periodos_config]
        periodos_analise_indices = st.sidebar.multiselect(
            "Períodos para análise",
            options=range(len(opcoes_periodos_analise)),
            format_func=lambda x: opcoes_periodos_analise[x],
            default=[0],
            key="periodos_analise"
        )
        
        # ==================== BOTÃO DE PROCESSAMENTO ====================
        st.sidebar.markdown("---")
        processar = st.sidebar.button("🚀 PROCESSAR ANÁLISE", type="primary", use_container_width=True)
        
        # ==================== PROCESSAMENTO DAS ANÁLISES ====================
        if processar:
            if not selected_indices:
                st.warning("⚠️ Selecione pelo menos um atleta para análise.")
                st.stop()
            
            selected_atletas = [st.session_state.atletas[i] for i in selected_indices]
            selected_data = [st.session_state.dados_carregados[i] for i in selected_indices]
            selected_periodos_orig = [st.session_state.periodos_orig[i] for i in selected_indices]
            selected_start_datetimes = [st.session_state.start_datetimes[i] for i in selected_indices]
            
            # Filtrar dados
            dfs_por_periodo = {}
            df_combinado_total = pd.DataFrame()
            distancias_por_periodo = {}
            tempos_por_periodo = {}
            
            for periodo_idx in periodos_analise_indices:
                if periodo_idx == 0:
                    periodo_nome = "Todos os períodos"
                    dfs_periodo = []
                    distancia_total = 0
                    tempo_total = 0
                    for df, atleta, periodo_orig, start_dt in zip(selected_data, selected_atletas, selected_periodos_orig, selected_start_datetimes):
                        time_filter = (df['Seconds'] >= start_time) & (df['Seconds'] <= end_time)
                        speed_filter = (df['Velocity'] >= speed_range[0]) & (df['Velocity'] <= speed_range[1])
                        df_filtered = df[time_filter & speed_filter].copy()
                        df_filtered['Atleta'] = atleta
                        df_filtered['Periodo'] = periodo_orig
                        df_filtered['Periodo_Analise'] = periodo_nome
                        df_filtered['start_datetime'] = start_dt
                        dfs_periodo.append(df_filtered)
                        if 'Odometer' in df_filtered.columns:
                            distancia_total += df_filtered['Odometer'].max() - df_filtered['Odometer'].min()
                        if len(df_filtered) > 1:
                            sample_rate = df_filtered['Seconds'].diff().median()
                            tempo_total += len(df_filtered) * sample_rate
                    if dfs_periodo:
                        df_temp = pd.concat(dfs_periodo, ignore_index=True)
                        dfs_por_periodo[periodo_nome] = df_temp
                        df_combinado_total = pd.concat([df_combinado_total, df_temp], ignore_index=True) if not df_combinado_total.empty else df_temp
                        distancias_por_periodo[periodo_nome] = distancia_total
                        tempos_por_periodo[periodo_nome] = tempo_total
                else:
                    periodo = st.session_state.periodos_config[periodo_idx - 1]
                    periodo_nome = periodo['nome']
                    periodo_inicio = periodo['inicio'] * 60
                    periodo_fim = periodo['fim'] * 60
                    
                    dfs_periodo = []
                    distancia_total = 0
                    tempo_total = 0
                    for df, atleta, periodo_orig, start_dt in zip(selected_data, selected_atletas, selected_periodos_orig, selected_start_datetimes):
                        time_filter = (df['Seconds'] >= max(start_time, periodo_inicio)) & (df['Seconds'] <= min(end_time, periodo_fim))
                        speed_filter = (df['Velocity'] >= speed_range[0]) & (df['Velocity'] <= speed_range[1])
                        df_filtered = df[time_filter & speed_filter].copy()
                        df_filtered['Atleta'] = atleta
                        df_filtered['Periodo'] = periodo_orig
                        df_filtered['Periodo_Analise'] = periodo_nome
                        df_filtered['start_datetime'] = start_dt
                        dfs_periodo.append(df_filtered)
                        if 'Odometer' in df_filtered.columns:
                            distancia_total += df_filtered['Odometer'].max() - df_filtered['Odometer'].min()
                        if len(df_filtered) > 1:
                            sample_rate = df_filtered['Seconds'].diff().median()
                            tempo_total += len(df_filtered) * sample_rate
                    
                    if dfs_periodo:
                        df_temp = pd.concat(dfs_periodo, ignore_index=True)
                        dfs_por_periodo[periodo_nome] = df_temp
                        df_combinado_total = pd.concat([df_combinado_total, df_temp], ignore_index=True) if not df_combinado_total.empty else df_temp
                        distancias_por_periodo[periodo_nome] = distancia_total
                        tempos_por_periodo[periodo_nome] = tempo_total
            
            if not dfs_por_periodo:
                st.warning("⚠️ Nenhum dado encontrado nos períodos selecionados.")
                st.stop()
            
            if len(periodos_analise_indices) > 1 and 0 not in periodos_analise_indices:
                dfs_por_periodo["Todos períodos combinados"] = df_combinado_total
                distancias_por_periodo["Todos períodos combinados"] = sum(distancias_por_periodo.get(p, 0) for p in dfs_por_periodo.keys() if p != "Todos períodos combinados")
                tempos_por_periodo["Todos períodos combinados"] = sum(tempos_por_periodo.get(p, 0) for p in dfs_por_periodo.keys() if p != "Todos períodos combinados")
            
            st.session_state.dfs_por_periodo = dfs_por_periodo
            st.session_state.distancias_por_periodo = distancias_por_periodo
            st.session_state.tempos_por_periodo = tempos_por_periodo
            st.session_state.selected_atletas = selected_atletas
            st.session_state.analise_processada = True
        
        # ==================== EXIBIÇÃO DAS ANÁLISES ====================
        if st.session_state.get('analise_processada', False):
            dfs_por_periodo = st.session_state.dfs_por_periodo
            distancias_por_periodo = st.session_state.distancias_por_periodo
            tempos_por_periodo = st.session_state.tempos_por_periodo
            selected_atletas = st.session_state.selected_atletas
            
            # Métricas principais
            st.markdown("### 📊 Métricas de Desempenho por Período")
            
            for periodo_nome, df_periodo in dfs_por_periodo.items():
                col1, col2, col3, col4, col5 = st.columns(5)
                
                dist_corrigida = distancias_por_periodo.get(periodo_nome, 0)
                tempo_corrigido = tempos_por_periodo.get(periodo_nome, 0)
                tempo_min = tempo_corrigido / 60
                
                with col1:
                    st.markdown(f"""
                    <div class="metric-card">
                        <div class="metric-value">{dist_corrigida:.0f}</div>
                        <div class="metric-label">Distância (m)</div>
                    </div>
                    """, unsafe_allow_html=True)
                with col2:
                    st.markdown(f"""
                    <div class="metric-card">
                        <div class="metric-value">{df_periodo['Velocity'].max():.1f}</div>
                        <div class="metric-label">Vel Máx (km/h)</div>
                    </div>
                    """, unsafe_allow_html=True)
                with col3:
                    st.markdown(f"""
                    <div class="metric-card">
                        <div class="metric-value">{df_periodo['Velocity'].mean():.1f}</div>
                        <div class="metric-label">Vel Média (km/h)</div>
                    </div>
                    """, unsafe_allow_html=True)
                with col4:
                    st.markdown(f"""
                    <div class="metric-card">
                        <div class="metric-value">{df_periodo['HeartRate'].mean():.0f}</div>
                        <div class="metric-label">FC Média (bpm)</div>
                    </div>
                    """, unsafe_allow_html=True)
                with col5:
                    st.markdown(f"""
                    <div class="metric-card">
                        <div class="metric-value">{df_periodo['HeartRate'].max():.0f}</div>
                        <div class="metric-label">FC Máx (bpm)</div>
                    </div>
                    """, unsafe_allow_html=True)
                st.markdown(f"**{periodo_nome}** | Tempo: {tempo_min:.1f} min | {len(df_periodo):,} registros")
                st.markdown("---")
            
            # ==================== ABAS ====================
            tab1, tab2, tab3, tab4, tab5 = st.tabs([
                "🗺️ Mapa Tático", 
                "📐 Análise por Zonas", 
                "⚡ Perfil Aceleração-Velocidade",
                "❤️ Performance Cardíaca",
                "📊 Comparação Esportiva"
            ])
            
            # TAB 1: MAPA TÁTICO
            with tab1:
                st.subheader("🗺️ Mapa Tático de Posicionamento")
                
                periodo_mapa = st.selectbox(
                    "Selecionar período", 
                    options=list(dfs_por_periodo.keys()),
                    key="periodo_mapa_select"
                )
                df_mapa = dfs_por_periodo[periodo_mapa]
                
                bounds = st.session_state.get('bounds_estadio', None)
                centro = st.session_state.get('centro_estadio', None)
                
                if bounds:
                    center_lat, center_lon = centro
                else:
                    center_lat, center_lon = df_mapa['Latitude'].mean(), df_mapa['Longitude'].mean()
                
                fig_map = go.Figure()
                if show_field and bounds:
                    lat_min, lat_max, lon_min, lon_max = bounds
                    fig_map.add_shape(type="rect", x0=lon_min, x1=lon_max, y0=lat_min, y1=lat_max,
                                      line=dict(color="white", width=2), fillcolor="rgba(34,139,34,0.2)")
                    fig_map.add_shape(type="line", x0=(lon_min+lon_max)/2, x1=(lon_min+lon_max)/2, y0=lat_min, y1=lat_max,
                                      line=dict(color="white", width=1, dash="dash"))
                
                if len(df_mapa) > 5000:
                    df_mapa_plot = df_mapa.sample(5000, random_state=42)
                else:
                    df_mapa_plot = df_mapa
                
                hover_texts = [f"<b>{seconds_to_time_str(row['Seconds'], row['start_datetime'])}</b><br>Vel: {row['Velocity']:.1f} km/h<br>FC: {row['HeartRate']:.0f} bpm" for _, row in df_mapa_plot.iterrows()]
                fig_map.add_trace(go.Scattermapbox(lat=df_mapa_plot['Latitude'], lon=df_mapa_plot['Longitude'], mode='markers',
                                                   marker=dict(size=4, color=df_mapa_plot['Velocity'], colorscale='Viridis', showscale=True,
                                                              colorbar=dict(title="Velocidade")), text=hover_texts, hoverinfo='text', name='Percurso'))
                
                if len(df_mapa_plot) > 0:
                    fig_map.add_trace(go.Scattermapbox(lat=[df_mapa_plot['Latitude'].iloc[0]], lon=[df_mapa_plot['Longitude'].iloc[0]],
                                                       mode='markers', marker=dict(size=16, color='green'), name='Início'))
                    fig_map.add_trace(go.Scattermapbox(lat=[df_mapa_plot['Latitude'].iloc[-1]], lon=[df_mapa_plot['Longitude'].iloc[-1]],
                                                       mode='markers', marker=dict(size=16, color='red'), name='Fim'))
                
                zoom = 18 if bounds else 15
                fig_map.update_layout(mapbox=dict(style="open-street-map", center=dict(lat=center_lat, lon=center_lon), zoom=zoom),
                                      height=600, margin=dict(l=0, r=0, t=30, b=0),
                                      title=f"Trajetória de {selected_atletas[0]} - {periodo_mapa} - {st.session_state.get('nome_estadio', 'Campo')}")
                st.plotly_chart(fig_map, use_container_width=True)
            
            # TAB 2: ANÁLISE POR ZONAS
            with tab2:
                st.subheader("📐 Análise Tática por Zonas do Campo")
                st.markdown(f"Campo com dimensões oficiais: **{CAMPO_COMPRIMENTO}m x {CAMPO_LARGURA}m**")
                
                periodo_tatica = st.selectbox(
                    "Selecionar período", 
                    options=list(dfs_por_periodo.keys()),
                    key="periodo_tatica_select"
                )
                df_tat = dfs_por_periodo[periodo_tatica]
                start_dt_tat = df_tat['start_datetime'].iloc[0] if len(df_tat) > 0 else None
                
                st.markdown(f"### 📊 Métricas do Período: {periodo_tatica}")
                col_metric1, col_metric2, col_metric3, col_metric4, col_metric5 = st.columns(5)
                with col_metric1:
                    dist = distancias_por_periodo.get(periodo_tatica, 0)
                    st.metric("Distância", f"{dist:.0f} m")
                with col_metric2:
                    st.metric("Velocidade Máxima", f"{df_tat['Velocity'].max():.1f} km/h")
                with col_metric3:
                    st.metric("Velocidade Média", f"{df_tat['Velocity'].mean():.1f} km/h")
                with col_metric4:
                    st.metric("FC Média", f"{df_tat['HeartRate'].mean():.0f} bpm")
                with col_metric5:
                    st.metric("FC Máxima", f"{df_tat['HeartRate'].max():.0f} bpm")
                st.markdown("---")
                
                bounds = st.session_state.get('bounds_estadio', None)
                if bounds:
                    if len(df_tat) > 3000:
                        df_tat_sample = df_tat.sample(3000, random_state=42)
                        st.caption(f"📊 Usando amostra de 3.000 pontos (total: {len(df_tat):,})")
                    else:
                        df_tat_sample = df_tat
                    
                    campo_x, campo_y = [], []
                    for _, row in df_tat_sample.iterrows():
                        x, y = converter_gps_para_campo_cached(row['Latitude'], row['Longitude'], bounds)
                        campo_x.append(x); campo_y.append(y)
                    df_tat_sample['campo_x'], df_tat_sample['campo_y'] = campo_x, campo_y
                else:
                    st.warning("⚠️ Limites do estádio não definidos.")
                    st.stop()
                
                col_lin, col_col = st.columns(2)
                with col_lin:
                    num_linhas = st.number_input("Linhas (divisão horizontal)", 1, 8, 3, key="num_linhas")
                with col_col:
                    num_colunas = st.number_input("Colunas (divisão vertical)", 1, 8, 3, key="num_colunas")
                
                linhas_bins = np.linspace(X_MIN, X_MAX, num_linhas + 1)
                colunas_bins = np.linspace(Y_MIN, Y_MAX, num_colunas + 1)
                
                df_tat_sample['Zona_Linha'] = pd.cut(df_tat_sample['campo_x'], bins=linhas_bins, labels=[f'L{i+1}' for i in range(num_linhas)], include_lowest=True)
                df_tat_sample['Zona_Coluna'] = pd.cut(df_tat_sample['campo_y'], bins=colunas_bins, labels=[f'C{i+1}' for i in range(num_colunas)], include_lowest=True)
                df_tat_sample['Zona'] = df_tat_sample['Zona_Linha'].astype(str) + '-' + df_tat_sample['Zona_Coluna'].astype(str)
                
                zona_metrics = df_tat_sample.groupby('Zona', observed=True).agg({
                    'Seconds': 'count',
                    'Velocity': ['mean', 'max'],
                    'HeartRate': ['mean', 'max']
                }).round(2)
                zona_metrics.columns = ['Contagem', 'Vel_Média', 'Vel_Máx', 'FC_Média', 'FC_Máx']
                
                if len(df_tat_sample) > 1:
                    sample_rate = df_tat_sample['Seconds'].diff().median()
                    zona_metrics['Tempo(s)'] = zona_metrics['Contagem'] * sample_rate
                    zona_metrics['Tempo(min)'] = zona_metrics['Tempo(s)'] / 60
                else:
                    zona_metrics['Tempo(s)'] = zona_metrics['Tempo(min)'] = 0
                
                total_contagem = zona_metrics['Contagem'].sum()
                zona_metrics['% Frequência'] = (zona_metrics['Contagem'] / total_contagem * 100).round(1) if total_contagem > 0 else 0
                zona_metrics = zona_metrics.sort_values('% Frequência', ascending=False)
                zona_metrics['% Acumulada'] = zona_metrics['% Frequência'].cumsum().round(1)
                zona_metrics = zona_metrics.sort_index()
                
                total_vel_peso = (zona_metrics['Vel_Média'] * zona_metrics['Contagem']).sum()
                zona_metrics['Intensidade (%)'] = ((zona_metrics['Vel_Média'] * zona_metrics['Contagem']) / total_vel_peso * 100).round(1) if total_vel_peso > 0 else 0
                
                st.markdown("### 🗺️ Visualização Tática")
                viz_type = st.radio("Tipo de visualização", ["Trajetória por zona", "Mapa de calor - Tempo", "Mapa de calor - Velocidade"], horizontal=True)
                
                fig_tat = go.Figure()
                for shape in desenhar_campo_futebol():
                    fig_tat.add_shape(shape)
                
                shapes_div, linhas_bins_plot, colunas_bins_plot = desenhar_linhas_divisorias(num_linhas, num_colunas)
                for shape in shapes_div:
                    fig_tat.add_shape(shape)
                
                if viz_type == "Trajetória por zona":
                    cores = px.colors.qualitative.Set3
                    for i, (zona, group) in enumerate(df_tat_sample.groupby('Zona')):
                        fig_tat.add_trace(go.Scatter(x=group['campo_x'], y=group['campo_y'], mode='markers', name=f'Zona {zona}',
                                                     marker=dict(size=4, color=cores[i % len(cores)], opacity=0.7),
                                                     text=[f"Zona: {zona}<br>Vel: {v:.1f} km/h" for v in group['Velocity']],
                                                     hoverinfo='text'))
                elif viz_type == "Mapa de calor - Tempo":
                    heatmap = np.zeros((num_linhas, num_colunas))
                    for i in range(num_linhas):
                        for j in range(num_colunas):
                            zona = f'L{i+1}-C{j+1}'
                            if zona in zona_metrics.index:
                                heatmap[i, j] = zona_metrics.loc[zona, 'Contagem']
                    fig_tat.add_trace(go.Heatmap(x=linhas_bins_plot, y=colunas_bins_plot, z=heatmap.T, colorscale='Hot', opacity=0.7,
                                                 colorbar=dict(title="Tempo gasto")))
                    for i in range(num_linhas):
                        for j in range(num_colunas):
                            zona = f'L{i+1}-C{j+1}'
                            if zona in zona_metrics.index:
                                pct = zona_metrics.loc[zona, '% Frequência']
                                centro_x, centro_y = (linhas_bins_plot[i] + linhas_bins_plot[i+1]) / 2, (colunas_bins_plot[j] + colunas_bins_plot[j+1]) / 2
                                fig_tat.add_annotation(x=centro_x, y=centro_y, text=f"{pct:.1f}%", showarrow=False,
                                                       font=dict(color="white", size=12),
                                                       bgcolor="rgba(0,0,0,0.6)", borderpad=2)
                    fig_tat.add_trace(go.Scatter(x=df_tat_sample['campo_x'], y=df_tat_sample['campo_y'], mode='markers',
                                                 marker=dict(size=2, color='white', opacity=0.5), name='Trajetória', hoverinfo='skip'))
                else:
                    heatmap = np.zeros((num_linhas, num_colunas))
                    for i in range(num_linhas):
                        for j in range(num_colunas):
                            zona = f'L{i+1}-C{j+1}'
                            if zona in zona_metrics.index:
                                heatmap[i, j] = zona_metrics.loc[zona, 'Vel_Média']
                    fig_tat.add_trace(go.Heatmap(x=linhas_bins_plot, y=colunas_bins_plot, z=heatmap.T, colorscale='Viridis', opacity=0.7,
                                                 colorbar=dict(title="Velocidade média")))
                    for i in range(num_linhas):
                        for j in range(num_colunas):
                            zona = f'L{i+1}-C{j+1}'
                            if zona in zona_metrics.index:
                                vel = zona_metrics.loc[zona, 'Vel_Média']
                                centro_x, centro_y = (linhas_bins_plot[i] + linhas_bins_plot[i+1]) / 2, (colunas_bins_plot[j] + colunas_bins_plot[j+1]) / 2
                                fig_tat.add_annotation(x=centro_x, y=centro_y, text=f"{vel:.1f}", showarrow=False,
                                                       font=dict(color="white", size=10),
                                                       bgcolor="rgba(0,0,0,0.6)", borderpad=2)
                    fig_tat.add_trace(go.Scatter(x=df_tat_sample['campo_x'], y=df_tat_sample['campo_y'], mode='markers',
                                                 marker=dict(size=2, color='white', opacity=0.5), name='Trajetória', hoverinfo='skip'))
                
                fig_tat.update_layout(title=f"Análise Tática - {selected_atletas[0]} - {periodo_tatica}",
                                      xaxis_title="Posição (m) - Comprimento", yaxis_title="Posição (m) - Largura",
                                      height=600, xaxis=dict(scaleanchor="y", scaleratio=1, range=[X_MIN-2, X_MAX+2]),
                                      yaxis=dict(range=[Y_MIN-2, Y_MAX+2]), plot_bgcolor='rgba(34,139,34,0.2)')
                st.plotly_chart(fig_tat, use_container_width=True)
                
                with st.expander("📖 **O que é o índice de Intensidade?**"):
                    st.markdown("""
                    O **Índice de Intensidade** combina **velocidade média** e **tempo de permanência** em cada zona.
                    **Fórmula:** `Intensidade = (Vel_Média × Contagem) / Σ(Vel_Média × Contagem) × 100`
                    **Interpretação:** >70%: esforço máximo, 30-70%: moderado, <30%: recuperação
                    """)
                
                st.markdown("### 📊 Demanda Física por Zona")
                st.dataframe(zona_metrics[['Contagem', '% Frequência', '% Acumulada', 'Vel_Média', 'Vel_Máx', 'FC_Média', 'FC_Máx', 'Intensidade (%)']].style.format({
                    'Contagem': '{:.0f}', '% Frequência': '{:.1f}%', '% Acumulada': '{:.1f}%',
                    'Vel_Média': '{:.1f}', 'Vel_Máx': '{:.1f}', 'FC_Média': '{:.0f}', 'FC_Máx': '{:.0f}', 'Intensidade (%)': '{:.1f}%'
                }), use_container_width=True)
                
                csv_tatico = zona_metrics.reset_index().to_csv(index=False)
                st.download_button("📥 Exportar análise tática", csv_tatico, f"analise_tatica_{selected_atletas[0]}_{periodo_tatica}.csv")
            
            # TAB 3: PERFIL ACELERAÇÃO-VELOCIDADE (CORRIGIDO - COMO NA IMAGEM)
            with tab3:
                st.subheader("⚡ Perfil Aceleração-Velocidade (Acceleration-Speed Profile)")
                
                with st.expander("📄 **Referência Científica**"):
                    st.markdown("""
                    **Alonso-Callejo, A., et al. (2024).** Validity and reliability of the acceleration-speed profile...  
                    *Journal of Strength and Conditioning Research, 38*(3), 563-570.
                    
                    **Interpretação do ASP:**
                    - **Aceleração Máxima (a₀)**: Capacidade de gerar força nos primeiros passos
                    - **Velocidade Máxima (v₀)**: Capacidade de atingir altas velocidades
                    - **R²**: Qualidade do ajuste do modelo (>0.7 indica boa confiabilidade)
                    """)
                
                periodo_asp = st.selectbox("Selecionar período", options=list(dfs_por_periodo.keys()), key="asp_periodo_select")
                df_asp_atual = dfs_por_periodo[periodo_asp]
                
                # Calcular ASP para todos os períodos
                asp_results = {}
                with st.spinner("Calculando perfil ASP..."):
                    for nome, df_periodo in dfs_por_periodo.items():
                        asp_metrics = calcular_asp_metrics(df_periodo)
                        if asp_metrics:
                            asp_results[nome] = asp_metrics
                
                if not asp_results:
                    st.warning("⚠️ Dados insuficientes para calcular o Perfil Aceleração-Velocidade. São necessários pelo menos 10 pontos com aceleração positiva e velocidade > 0.")
                else:
                    st.markdown("### 📈 Curva Aceleração-Velocidade (ASP)")
                    st.markdown("**Relação entre Velocidade (m/s) e Aceleração (m/s²) - Modelo Linear**")
                    
                    # Selecionar período atual
                    metrics = asp_results.get(periodo_asp)
                    if metrics:
                        df_sprints = dfs_por_periodo[periodo_asp]
                        df_sprints = df_sprints[(df_sprints['Acceleration'] > 0) & (df_sprints['Velocity'] > 0)].copy()
                        
                        # Converter velocidade para m/s
                        if df_sprints['Velocity'].max() < 50:
                            v_ms = df_sprints['Velocity'].values
                        else:
                            v_ms = df_sprints['Velocity'].values / 3.6
                        
                        a_ms2 = df_sprints['Acceleration'].values
                        
                        # Criar figura
                        fig_asp = go.Figure()
                        
                        # 1. Plotar todos os pontos (como na imagem)
                        fig_asp.add_trace(go.Scatter(
                            x=v_ms, y=a_ms2,
                            mode='markers',
                            name='In situ raw data, all points',
                            marker=dict(
                                size=6,
                                color='rgba(0, 210, 255, 0.6)',
                                symbol='circle',
                                line=dict(width=1, color='white')
                            ),
                            hovertemplate='Velocidade: %{x:.2f} m/s<br>Aceleração: %{y:.2f} m/s²<extra></extra>'
                        ))
                        
                        # 2. Curva de regressão linear
                        v_fit = np.linspace(0, max(v_ms) * 1.05, 100)
                        a_fit = metrics['intercept'] + metrics['slope'] * v_fit
                        
                        fig_asp.add_trace(go.Scatter(
                            x=v_fit, y=a_fit,
                            mode='lines',
                            name='Acceleration-speed profile',
                            line=dict(color='red', width=3, dash='solid'),
                            hovertemplate='Modelo: a = {intercept:.2f} + {slope:.2f}·v<br>v: %{{x:.2f}} m/s<br>a: %{{y:.2f}} m/s²<extra></extra>'.format(
                                intercept=metrics['intercept'], slope=metrics['slope']
                            )
                        ))
                        
                        # 3. Destacar ponto A₀ (aceleração máxima)
                        fig_asp.add_trace(go.Scatter(
                            x=[0], y=[metrics['a0']],
                            mode='markers+text',
                            marker=dict(size=14, color='green', symbol='circle', line=dict(width=2, color='white')),
                            text=[f"A₀ = {metrics['a0']:.2f} m/s²"],
                            textposition='top right',
                            textfont=dict(size=12, color='white', family='Arial Black'),
                            name='A₀ - Aceleração Máxima',
                            hovertemplate='Aceleração Máxima: %{y:.2f} m/s²<extra></extra>'
                        ))
                        
                        # 4. Adicionar equação da reta e R²
                        equation_text = f"y = {metrics['slope']:.3f}x + {metrics['intercept']:.2f}<br>R² = {metrics['r2']:.3f}"
                        
                        fig_asp.add_annotation(
                            x=0.95, y=0.95,
                            xref="paper", yref="paper",
                            text=equation_text,
                            showarrow=False,
                            font=dict(size=14, color='white', family='monospace'),
                            bgcolor='rgba(0,0,0,0.6)',
                            borderpad=10,
                            bordercolor='white',
                            borderwidth=1
                        )
                        
                        # Configurar layout
                        fig_asp.update_layout(
                            title=dict(
                                text=f"<b>Perfil Aceleração-Velocidade - {periodo_asp}</b>",
                                x=0.5,
                                font=dict(size=18)
                            ),
                            xaxis=dict(
                                title="<b>Running Speed (m/s)</b>",
                                title_font=dict(size=14),
                                gridcolor='rgba(255,255,255,0.1)',
                                zerolinecolor='rgba(255,255,255,0.2)',
                                range=[0, max(v_ms) * 1.05],
                                dtick=1
                            ),
                            yaxis=dict(
                                title="<b>Acceleration (m/s²)</b>",
                                title_font=dict(size=14),
                                gridcolor='rgba(255,255,255,0.1)',
                                zerolinecolor='rgba(255,255,255,0.2)',
                                range=[0, max(a_ms2) * 1.1],
                                dtick=1
                            ),
                            height=550,
                            legend=dict(
                                orientation="h",
                                yanchor="bottom",
                                y=1.02,
                                xanchor="center",
                                x=0.5,
                                bgcolor='rgba(0,0,0,0.5)',
                                font=dict(size=11)
                            ),
                            plot_bgcolor='rgba(0,0,0,0.2)',
                            paper_bgcolor='rgba(0,0,0,0)',
                            hovermode='closest'
                        )
                        
                        st.plotly_chart(fig_asp, use_container_width=True)
                        
                        # Tabela de métricas ASP
                        st.markdown("### 📊 Métricas do Perfil Aceleração-Velocidade")
                        
                        asp_df = pd.DataFrame(asp_results).T
                        asp_df = asp_df[['a0', 'v0', 'v_max', 'p_max', 'r2', 'slope']].round(3)
                        asp_df.columns = ['A₀ (m/s²)', 'V₀ (m/s)', 'Vₘₐₓ (m/s)', 'Pₘₐₓ (W/kg)', 'R²', 'Slope']
                        
                        # Adicionar indicador de qualidade
                        def qualidade_r2(r2):
                            if r2 >= 0.8:
                                return '🟢 Excelente'
                            elif r2 >= 0.6:
                                return '🟡 Bom'
                            elif r2 >= 0.4:
                                return '🟠 Moderado'
                            else:
                                return '🔴 Baixo'
                        
                        asp_df['Qualidade'] = asp_df['R²'].apply(qualidade_r2)
                        
                        st.dataframe(asp_df, use_container_width=True)
                        
                        # Cards de interpretação
                        st.markdown("### 🎯 Interpretação dos Resultados")
                        col_asp1, col_asp2, col_asp3 = st.columns(3)
                        
                        with col_asp1:
                            a0_medio = asp_df['A₀ (m/s²)'].mean()
                            st.markdown(f"""
                            <div class="metric-card">
                                <div class="metric-value">{a0_medio:.2f}</div>
                                <div class="metric-label">Aceleração Média (m/s²)</div>
                                <div style="font-size:0.7rem; margin-top:8px;">⚡ Capacidade de aceleração inicial</div>
                            </div>
                            """, unsafe_allow_html=True)
                        
                        with col_asp2:
                            v0_medio = asp_df['V₀ (m/s)'].mean()
                            st.markdown(f"""
                            <div class="metric-card">
                                <div class="metric-value">{v0_medio:.2f}</div>
                                <div class="metric-label">Velocidade Teórica (m/s)</div>
                                <div style="font-size:0.7rem; margin-top:8px;">🏃 Capacidade de alta velocidade</div>
                            </div>
                            """, unsafe_allow_html=True)
                        
                        with col_asp3:
                            p_max_medio = asp_df['Pₘₐₓ (W/kg)'].mean()
                            st.markdown(f"""
                            <div class="metric-card">
                                <div class="metric-value">{p_max_medio:.2f}</div>
                                <div class="metric-label">Potência Máx (W/kg)</div>
                                <div style="font-size:0.7rem; margin-top:8px;">💪 Explosividade muscular</div>
                            </div>
                            """, unsafe_allow_html=True)
            
            # TAB 4: ANÁLISE DE PERFORMANCE CARDÍACA
            with tab4:
                st.subheader("❤️ Análise de Performance Cardíaca")
                
                periodo_fc = st.selectbox("Selecionar período", options=list(dfs_por_periodo.keys()), key="fc_periodo_select")
                df_fc = dfs_por_periodo[periodo_fc]
                start_dt_fc = df_fc['start_datetime'].iloc[0] if len(df_fc) > 0 else None
                
                if len(df_fc) > 2000:
                    df_fc_plot = df_fc.sample(2000, random_state=42).sort_values('Seconds')
                    st.caption(f"📊 Usando amostra de 2.000 pontos (total: {len(df_fc):,})")
                else:
                    df_fc_plot = df_fc
                
                df_fc_plot['Horario'] = df_fc_plot['Seconds'].apply(lambda x: seconds_to_time_str(x, start_dt_fc))
                
                fig_fc_acc = make_subplots(specs=[[{"secondary_y": True}]])
                
                fig_fc_acc.add_trace(
                    go.Scatter(x=df_fc_plot['Horario'], y=df_fc_plot['HeartRate'], mode='lines', name='FC',
                              line=dict(color='#ff6b6b', width=2), fill='tozeroy', fillcolor='rgba(255,107,107,0.1)'),
                    secondary_y=False
                )
                
                fig_fc_acc.add_trace(
                    go.Scatter(x=df_fc_plot['Horario'], y=df_fc_plot['Acceleration'], mode='lines', name='Aceleração',
                              line=dict(color='#4ecdc4', width=1.5), fill='tozeroy', fillcolor='rgba(78,205,196,0.1)'),
                    secondary_y=True
                )
                
                fc_max = df_fc_plot['HeartRate'].max()
                limiar_anaerobico = fc_max * 0.85
                limiar_aerobico = fc_max * 0.75
                
                fig_fc_acc.add_hrect(y0=limiar_aerobico, y1=limiar_anaerobico, fillcolor="rgba(255,215,0,0.2)", line_width=0, secondary_y=False,
                                     annotation_text="Zona Anaeróbica", annotation_position="bottom right")
                fig_fc_acc.add_hrect(y0=limiar_anaerobico, y1=fc_max, fillcolor="rgba(231,76,60,0.2)", line_width=0, secondary_y=False,
                                     annotation_text="Zona Máxima", annotation_position="top right")
                
                fig_fc_acc.update_layout(
                    title=f"FC vs Aceleração - {periodo_fc}",
                    height=450,
                    hovermode='x unified',
                    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5)
                )
                fig_fc_acc.update_yaxes(title_text="Frequência Cardíaca (bpm)", secondary_y=False, color="#ff6b6b")
                fig_fc_acc.update_yaxes(title_text="Aceleração (m/s²)", secondary_y=True, color="#4ecdc4")
                st.plotly_chart(fig_fc_acc, use_container_width=True)
                
                if 'Odometer' in df_fc_plot.columns:
                    st.markdown("### 📈 Distância Acumulada vs FC")
                    
                    df_fc_plot['Distancia_Acumulada'] = df_fc_plot['Odometer'] - df_fc_plot['Odometer'].min()
                    
                    fig_dist_fc = make_subplots(specs=[[{"secondary_y": True}]])
                    
                    fig_dist_fc.add_trace(
                        go.Scatter(x=df_fc_plot['Horario'], y=df_fc_plot['Distancia_Acumulada'], mode='lines', name='Distância',
                                  line=dict(color='#2ecc71', width=2), fill='tozeroy', fillcolor='rgba(46,204,113,0.1)'),
                        secondary_y=False
                    )
                    
                    fig_dist_fc.add_trace(
                        go.Scatter(x=df_fc_plot['Horario'], y=df_fc_plot['HeartRate'], mode='lines', name='FC',
                                  line=dict(color='#ff6b6b', width=2)),
                        secondary_y=True
                    )
                    
                    fig_dist_fc.update_layout(title=f"Distância vs FC - {periodo_fc}", height=400)
                    fig_dist_fc.update_yaxes(title_text="Distância (m)", secondary_y=False, color="#2ecc71")
                    fig_dist_fc.update_yaxes(title_text="FC (bpm)", secondary_y=True, color="#ff6b6b")
                    st.plotly_chart(fig_dist_fc, use_container_width=True)
                
                st.markdown("### 📊 Zonas de Intensidade Cardíaca")
                
                df_fc_plot['Zona_FC'] = pd.cut(df_fc_plot['HeartRate'], 
                                          bins=[0, fc_max*0.6, fc_max*0.75, fc_max*0.9, fc_max],
                                          labels=['Recuperação (<60%)', 'Aeróbica (60-75%)', 'Anaeróbica (75-90%)', 'Máximo (>90%)'])
                
                zona_stats = df_fc_plot.groupby('Zona_FC', observed=True).size().reset_index(name='Contagem')
                zona_stats['% do Tempo'] = (zona_stats['Contagem'] / len(df_fc_plot) * 100).round(1)
                zona_stats['Tempo (min)'] = (zona_stats['Contagem'] * df_fc_plot['Seconds'].diff().median() / 60).round(1)
                
                fig_zona_bar = px.bar(zona_stats, x='Zona_FC', y='% do Tempo', 
                                      title="Distribuição do Tempo por Zona de Intensidade",
                                      text=zona_stats['% do Tempo'].apply(lambda x: f'{x}%'),
                                      color='Zona_FC',
                                      color_discrete_sequence=['#2ecc71', '#f1c40f', '#e67e22', '#e74c3c'])
                fig_zona_bar.update_layout(showlegend=False, height=400)
                st.plotly_chart(fig_zona_bar, use_container_width=True)
                
                col_pie1, col_pie2 = st.columns(2)
                with col_pie1:
                    fig_pie = px.pie(zona_stats, values='Contagem', names='Zona_FC', title="Distribuição por Zona de Intensidade",
                                     color_discrete_sequence=['#2ecc71', '#f1c40f', '#e67e22', '#e74c3c'])
                    st.plotly_chart(fig_pie, use_container_width=True)
                
                with col_pie2:
                    st.markdown("### 📈 Resumo Cardíaco")
                    st.markdown(f"""
                    <div style="background: rgba(0,0,0,0.3); border-radius: 15px; padding: 20px;">
                        <p><b>❤️ FC Máxima:</b> {fc_max:.0f} bpm</p>
                        <p><b>📊 FC Média:</b> {df_fc_plot['HeartRate'].mean():.0f} bpm</p>
                        <p><b>⚡ Limiar Anaeróbico:</b> {limiar_anaerobico:.0f} bpm</p>
                        <p><b>🏃 Tempo em Zona Máxima:</b> {zona_stats[zona_stats['Zona_FC']=='Máximo (>90%)']['% do Tempo'].values[0] if 'Máximo (>90%)' in zona_stats['Zona_FC'].values else 0}%</p>
                    </div>
                    """, unsafe_allow_html=True)
            
            # TAB 5: COMPARAÇÃO ESPORTIVA
            with tab5:
                st.subheader("📊 Comparação Esportiva entre Períodos")
                
                var_selecionadas = st.multiselect(
                    "Selecione as variáveis para comparação",
                    options=["Velocidade Média (km/h)", "Velocidade Máxima (km/h)", 
                             "Frequência Cardíaca Média (bpm)", "Frequência Cardíaca Máxima (bpm)", 
                             "Distância Total (m)", "Tempo Total (min)"],
                    default=["Velocidade Média (km/h)", "Distância Total (m)", "Frequência Cardíaca Média (bpm)"],
                    key="var_comparacao"
                )
                
                if var_selecionadas:
                    comparacao_data = []
                    for periodo_nome, df_periodo in dfs_por_periodo.items():
                        row = {"Período": periodo_nome}
                        for var in var_selecionadas:
                            if var == "Velocidade Média (km/h)":
                                row[var] = df_periodo['Velocity'].mean()
                            elif var == "Velocidade Máxima (km/h)":
                                row[var] = df_periodo['Velocity'].max()
                            elif var == "Frequência Cardíaca Média (bpm)":
                                row[var] = df_periodo['HeartRate'].mean()
                            elif var == "Frequência Cardíaca Máxima (bpm)":
                                row[var] = df_periodo['HeartRate'].max()
                            elif var == "Distância Total (m)":
                                row[var] = distancias_por_periodo.get(periodo_nome, 0)
                            elif var == "Tempo Total (min)":
                                row[var] = tempos_por_periodo.get(periodo_nome, 0) / 60
                        comparacao_data.append(row)
                    
                    df_comp = pd.DataFrame(comparacao_data)
                    
                    for var in var_selecionadas:
                        fig_bar = px.bar(df_comp, x='Período', y=var, title=f"<b>{var}</b>", 
                                         text_auto='.1f', color='Período',
                                         color_discrete_sequence=px.colors.qualitative.Set2)
                        fig_bar.update_layout(height=400, showlegend=False)
                        st.plotly_chart(fig_bar, use_container_width=True)
                    
                    if len(df_comp) >= 2:
                        st.markdown("### 🎯 Perfil de Desempenho (Radar)")
                        
                        df_radar = df_comp.copy()
                        for var in var_selecionadas:
                            max_val = df_radar[var].max()
                            if max_val > 0:
                                df_radar[var + " (%)"] = (df_radar[var] / max_val * 100).round(1)
                        
                        fig_radar = go.Figure()
                        radar_cores = ['#00d2ff', '#ff6b6b', '#4ecdc4', '#ffe66d', '#9b59b6', '#e74c3c']
                        
                        for idx, (_, row) in enumerate(df_radar.iterrows()):
                            valores = [row[var + " (%)"] for var in var_selecionadas]
                            fig_radar.add_trace(go.Scatterpolar(
                                r=valores,
                                theta=var_selecionadas,
                                fill='toself',
                                name=row['Período'],
                                line=dict(width=3, color=radar_cores[idx % len(radar_cores)]),
                                fillcolor=f'rgba({int(radar_cores[idx % len(radar_cores)][1:3], 16)}, {int(radar_cores[idx % len(radar_cores)][3:5], 16)}, {int(radar_cores[idx % len(radar_cores)][5:7], 16)}, 0.3)'
                            ))
                        
                        fig_radar.update_layout(
                            polar=dict(
                                radialaxis=dict(
                                    visible=True,
                                    range=[0, 100],
                                    tickvals=[0, 25, 50, 75, 100],
                                    ticktext=['0%', '25%', '50%', '75%', '100%'],
                                    gridcolor='rgba(255,255,255,0.2)',
                                    linecolor='rgba(255,255,255,0.3)'
                                ),
                                angularaxis=dict(
                                    tickfont=dict(size=12, color='white'),
                                    gridcolor='rgba(255,255,255,0.2)'
                                ),
                                bgcolor='rgba(0,0,0,0.2)'
                            ),
                            title=dict(
                                text="<b>Perfil de Desempenho Normalizado</b><br><sup>Comparação entre períodos (quanto maior a área, melhor o desempenho)</sup>",
                                x=0.5,
                                font=dict(size=16)
                            ),
                            height=600,
                            legend=dict(
                                orientation="h",
                                yanchor="bottom",
                                y=1.02,
                                xanchor="center",
                                x=0.5,
                                bgcolor='rgba(0,0,0,0.5)',
                                font=dict(size=12)
                            ),
                            paper_bgcolor='rgba(0,0,0,0)',
                            plot_bgcolor='rgba(0,0,0,0)'
                        )
                        st.plotly_chart(fig_radar, use_container_width=True)
                        
                        variacao_data = []
                        for i, row in df_comp.iterrows():
                            if i == 0:
                                variacao_data.append({"Período": row['Período'], "Status": "📌 Referência"})
                            else:
                                ref = df_comp.iloc[0]
                                var_dict = {"Período": row['Período']}
                                for var in var_selecionadas:
                                    if var in row and var in ref:
                                        diff = row[var] - ref[var]
                                        pct = (diff / ref[var] * 100) if ref[var] != 0 else 0
                                        var_dict[var] = f"{'+' if diff > 0 else ''}{pct:.1f}%"
                                variacao_data.append(var_dict)
                        
                        df_variacao = pd.DataFrame(variacao_data)
                        st.dataframe(df_variacao, use_container_width=True)
                        
                        csv_comp = df_comp.to_csv(index=False)
                        st.download_button(
                            "📥 Exportar comparação (CSV)",
                            csv_comp,
                            "comparacao_periodos.csv",
                            use_container_width=True
                        )
        
        else:
            st.info("👈 Configure os filtros na barra lateral e clique em **PROCESSAR ANÁLISE** para visualizar os resultados.")

else:
    st.markdown("""
    <div style="text-align: center; padding: 50px;">
        <h2>⚽ ScoutLab - Plataforma de Análise de Performance</h2>
        <p style="font-size: 1.2rem; color: #aaa;">Carregue os arquivos CSV na barra lateral para iniciar a análise</p>
        <p style="margin-top: 30px;">📊 Análise de posicionamento | ⚡ Perfil de aceleração | ❤️ Monitoramento cardíaco | 🎯 Zonas de intensidade</p>
    </div>
    """, unsafe_allow_html=True)