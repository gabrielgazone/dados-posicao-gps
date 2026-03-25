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
    page_title="Análise de Percurso do Atleta",
    page_icon="🏃",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Título do app
st.title("🏃 Análise de Percurso do Atleta durante o Jogo")
st.markdown("---")

# ==================== CONSTANTES DO CAMPO ====================
CAMPO_COMPRIMENTO = 105
CAMPO_LARGURA = 68
X_MIN = -CAMPO_COMPRIMENTO / 2
X_MAX = CAMPO_COMPRIMENTO / 2
Y_MIN = -CAMPO_LARGURA / 2
Y_MAX = CAMPO_LARGURA / 2

# ==================== FUNÇÃO DO PERFIL ACELERAÇÃO-VELOCIDADE ====================

@st.cache_data(ttl=3600)
def fit_velocidade_aceleracao_cached(velocidades_tuple, aceleracoes_tuple):
    velocidades = np.array(velocidades_tuple)
    aceleracoes = np.array(aceleracoes_tuple)
    
    mask = (velocidades > 0) & (aceleracoes > -5) & (aceleracoes < 10) & (~np.isnan(velocidades)) & (~np.isnan(aceleracoes))
    v_clean = velocidades[mask]
    a_clean = aceleracoes[mask]
    
    if len(v_clean) < 5:
        return None
    
    try:
        def modelo_linear(v, a0, v0):
            return a0 - (a0 / v0) * v
        
        popt, _ = curve_fit(modelo_linear, v_clean, a_clean, p0=[5, 8], maxfev=5000)
        a0, v0 = popt
        
        p_max = a0 * v0 / 4
        v_max = np.max(v_clean)
        
        residuos = a_clean - modelo_linear(v_clean, a0, v0)
        ss_res = np.sum(residuos**2)
        ss_tot = np.sum((a_clean - np.mean(a_clean))**2)
        r2 = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
        
        return {
            'a0': float(a0),
            'v0': float(v0),
            'v_max': float(v_max),
            'p_max': float(p_max),
            'r2': float(r2),
            'n_points': len(v_clean),
            'slope': -a0 / v0 if v0 > 0 else 0
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
    result = fit_velocidade_aceleracao_cached(tuple(v_ms), tuple(a_ms2))
    
    if result:
        result['v_medio'] = np.mean(v_ms)
        result['a_medio'] = np.mean(a_ms2)
        result['num_sprints'] = len(df_sprints)
        result['f0'] = result['a0']
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

# ==================== SIDEBAR ====================

init_database()

st.sidebar.header("📁 Upload de Arquivos")
uploaded_files = st.sidebar.file_uploader("Escolha os arquivos CSV", type=['csv'], accept_multiple_files=True)

# ==================== SIDEBAR - SELEÇÃO DE ESTÁDIO ====================

st.sidebar.markdown("---")
st.sidebar.subheader("🏟️ Configuração do Estádio")

df_estadios = carregar_estadios()
estadio_selecionado = None
bounds_estadio = None
centro_estadio = None
nome_estadio = "Não selecionado"

if len(df_estadios) > 0:
    opcoes_estadio = ["Detectar automaticamente"] + df_estadios['nome'].tolist() + ["Cadastrar novo estádio"]
    selecao_estadio = st.sidebar.selectbox("Selecione o estádio ou modo de detecção", options=opcoes_estadio, index=0, key="selecao_estadio")
    
    if selecao_estadio != "Detectar automaticamente" and selecao_estadio != "Cadastrar novo estádio":
        idx_estadio = df_estadios[df_estadios['nome'] == selecao_estadio].index[0]
        estadio_selecionado = obter_estadio(df_estadios.loc[idx_estadio, 'id'])
        if estadio_selecionado:
            bounds_estadio = estadio_selecionado['bounds']
            centro_estadio = estadio_selecionado['centro']
            nome_estadio = estadio_selecionado['nome']
            st.sidebar.success(f"✅ Estádio: {nome_estadio}")
    
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

# ==================== SIDEBAR - DIVISÃO TEMPORAL ====================

st.sidebar.markdown("---")
st.sidebar.subheader("⏱️ Divisão Temporal do Jogo")

# Inicializar períodos
if 'periodos' not in st.session_state:
    st.session_state.periodos = [{"nome": "1º Tempo", "inicio": 0, "fim": 45}]

# Função callback para atualizar a prova real
def atualizar_prova_real():
    """Callback para atualizar a exibição da prova real"""
    pass

if st.sidebar.button("➕ Adicionar período", use_container_width=True):
    st.session_state.periodos.append({"nome": f"Período {len(st.session_state.periodos) + 1}", "inicio": 0, "fim": 45})
    st.rerun()

def get_horario_formatado(minutos, start_dt):
    if start_dt:
        segundos = minutos * 60
        return seconds_to_time_str(segundos, start_dt)
    return f"{int(minutos)}:00"

# Função para obter o horário de referência (se já houver arquivo carregado)
def get_reference_datetime():
    if 'reference_datetime_temp' in st.session_state:
        return st.session_state.reference_datetime_temp
    return None

# Exibir períodos com prova real (mesmo antes do processamento)
st.sidebar.markdown("### Períodos configurados:")

periodos_para_remover = []
for i, periodo in enumerate(st.session_state.periodos):
    with st.sidebar.expander(f"📅 {periodo['nome']}", expanded=False):
        col_nome, col_remover = st.columns([3, 1])
        with col_nome:
            novo_nome = st.text_input("Nome", value=periodo['nome'], key=f"nome_{i}")
        with col_remover:
            if i > 0:
                if st.button("🗑️", key=f"remover_{i}"):
                    periodos_para_remover.append(i)
        
        col_ini, col_fim = st.columns(2)
        with col_ini:
            novo_inicio = st.number_input("Início (min)", value=float(periodo['inicio']), step=1.0, key=f"inicio_{i}")
        with col_fim:
            novo_fim = st.number_input("Fim (min)", value=float(periodo['fim']), step=1.0, key=f"fim_{i}")
        
        st.session_state.periodos[i] = {"nome": novo_nome, "inicio": novo_inicio, "fim": novo_fim}
        
        # PROVA REAL - Exibe horário e duração ANTES do processamento
        reference_dt = get_reference_datetime()
        if reference_dt:
            inicio_horario = get_horario_formatado(novo_inicio, reference_dt)
            fim_horario = get_horario_formatado(novo_fim, reference_dt)
            duracao_seg = (novo_fim - novo_inicio) * 60
            duracao_str = format_duration(duracao_seg)
            st.caption(f"🕐 {inicio_horario} → {fim_horario}  |  ⏱️ Duração: {duracao_str}")
        else:
            st.caption("⏳ Aguardando upload do arquivo para exibir horários reais")

for i in sorted(periodos_para_remover, reverse=True):
    st.session_state.periodos.pop(i)
    st.rerun()

# ==================== BOTÃO DE PROCESSAMENTO ====================

st.sidebar.markdown("---")
processar = st.sidebar.button("🚀 PROCESSAR ANÁLISE", type="primary", use_container_width=True)

# ==================== PROCESSAMENTO PRINCIPAL ====================

if uploaded_files and processar:
    with st.spinner("Carregando e processando arquivos..."):
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
            selected_indices = st.sidebar.multiselect(
                "Escolha os atletas",
                options=range(len(all_atletas)),
                format_func=lambda x: f"{all_atletas[x]} - {all_periodos[x]}" if all_periodos[x] != "Não identificado" else all_atletas[x],
                default=[0] if len(all_atletas) > 0 else []
            )
            
            if not selected_indices:
                st.warning("⚠️ Selecione pelo menos um atleta para análise.")
                st.stop()
            
            selected_atletas = [all_atletas[i] for i in selected_indices]
            selected_data = [all_data[i] for i in selected_indices]
            selected_periodos = [all_periodos[i] for i in selected_indices]
            selected_start_datetimes = [all_start_datetimes[i] for i in selected_indices]
            
            reference_datetime = selected_start_datetimes[0] if selected_start_datetimes[0] is not None else None
            st.session_state.reference_datetime_temp = reference_datetime
            
            df_calibracao = selected_data[0]
            
            if selecao_estadio == "Detectar automaticamente" or estadio_selecionado is None:
                lat_min = df_calibracao['Latitude'].quantile(0.01)
                lat_max = df_calibracao['Latitude'].quantile(0.99)
                lon_min = df_calibracao['Longitude'].quantile(0.01)
                lon_max = df_calibracao['Longitude'].quantile(0.99)
                centro_lat, centro_lon = (lat_min + lat_max) / 2, (lon_min + lon_max) / 2
                bounds_estadio = (lat_min, lat_max, lon_min, lon_max)
                centro_estadio = (centro_lat, centro_lon)
                nome_estadio = "Detectado automaticamente"
            
            st.sidebar.markdown("---")
            st.sidebar.subheader("⏱️ Filtro Temporal Global")
            
            min_time = float('inf')
            max_time = 0
            for df in selected_data:
                min_time = min(min_time, df['Seconds'].min())
                max_time = max(max_time, df['Seconds'].max())
            
            min_time_min, max_time_min = min_time / 60, max_time / 60
            
            if reference_datetime:
                selected_range = st.sidebar.slider("Intervalo de tempo (minutos)", min_value=float(min_time_min),
                                                   max_value=float(max_time_min), value=(float(min_time_min), float(max_time_min)), step=0.5)
                start_time_min, end_time_min = selected_range
                start_time, end_time = start_time_min * 60, end_time_min * 60
                start_horario = seconds_to_time_str(start_time, reference_datetime)
                end_horario = seconds_to_time_str(end_time, reference_datetime)
            else:
                start_time, end_time = min_time, max_time
                start_horario = f"{int(start_time//3600):02d}:{int((start_time%3600)//60):02d}:{int(start_time%60):02d}"
                end_horario = f"{int(end_time//3600):02d}:{int((end_time%3600)//60):02d}:{int(end_time%60):02d}"
            
            st.sidebar.markdown("---")
            st.sidebar.subheader("⚡ Filtro Velocidade")
            
            max_speed = max([df['Velocity'].max() for df in selected_data])
            speed_range = st.sidebar.slider("Velocidade (km/h)", min_value=0.0, max_value=float(max_speed),
                                            value=(0.0, float(max_speed)), step=0.5)
            
            st.sidebar.markdown("---")
            st.sidebar.subheader("🎨 Opções")
            show_field = st.sidebar.checkbox("Mostrar campo", value=True)
            
            # Opção "Todos os períodos"
            opcoes_periodos = ["Todos os períodos"] + [p['nome'] for p in st.session_state.periodos]
            periodos_selecionados = st.sidebar.multiselect(
                "Selecionar períodos para análise",
                options=range(len(opcoes_periodos)),
                format_func=lambda x: opcoes_periodos[x],
                default=[0]
            )
            
            # Filtrar dados por período selecionado
            dfs_por_periodo = {}
            df_combinado_total = pd.DataFrame()
            
            for periodo_idx in periodos_selecionados:
                if periodo_idx == 0:
                    periodo_nome = "Todos os períodos"
                    dfs_periodo = []
                    for df, atleta, periodo_nome_orig, start_dt in zip(selected_data, selected_atletas, selected_periodos, selected_start_datetimes):
                        time_filter = (df['Seconds'] >= start_time) & (df['Seconds'] <= end_time)
                        speed_filter = (df['Velocity'] >= speed_range[0]) & (df['Velocity'] <= speed_range[1])
                        df_filtered = df[time_filter & speed_filter].copy()
                        df_filtered['Atleta'] = atleta
                        df_filtered['Periodo'] = periodo_nome_orig
                        df_filtered['Periodo_Analise'] = periodo_nome
                        df_filtered['start_datetime'] = start_dt
                        dfs_periodo.append(df_filtered)
                    if dfs_periodo:
                        df_temp = pd.concat(dfs_periodo, ignore_index=True)
                        dfs_por_periodo[periodo_nome] = df_temp
                        df_combinado_total = pd.concat([df_combinado_total, df_temp], ignore_index=True) if not df_combinado_total.empty else df_temp
                else:
                    periodo = st.session_state.periodos[periodo_idx - 1]
                    periodo_nome = periodo['nome']
                    periodo_inicio = periodo['inicio'] * 60
                    periodo_fim = periodo['fim'] * 60
                    
                    dfs_periodo = []
                    for df, atleta, periodo_nome_orig, start_dt in zip(selected_data, selected_atletas, selected_periodos, selected_start_datetimes):
                        time_filter = (df['Seconds'] >= max(start_time, periodo_inicio)) & (df['Seconds'] <= min(end_time, periodo_fim))
                        speed_filter = (df['Velocity'] >= speed_range[0]) & (df['Velocity'] <= speed_range[1])
                        df_filtered = df[time_filter & speed_filter].copy()
                        df_filtered['Atleta'] = atleta
                        df_filtered['Periodo'] = periodo_nome_orig
                        df_filtered['Periodo_Analise'] = periodo_nome
                        df_filtered['start_datetime'] = start_dt
                        dfs_periodo.append(df_filtered)
                    
                    if dfs_periodo:
                        df_temp = pd.concat(dfs_periodo, ignore_index=True)
                        dfs_por_periodo[periodo_nome] = df_temp
                        df_combinado_total = pd.concat([df_combinado_total, df_temp], ignore_index=True) if not df_combinado_total.empty else df_temp
            
            if not dfs_por_periodo:
                st.warning("⚠️ Nenhum dado encontrado nos períodos selecionados.")
                st.stop()
            
            if len(dfs_por_periodo) > 1:
                dfs_por_periodo["Todos períodos combinados"] = df_combinado_total
            
            primeiro_periodo = list(dfs_por_periodo.keys())[0]
            df_main = dfs_por_periodo[primeiro_periodo]
            atleta_main = selected_atletas[0]
            
            st.markdown("### 📊 Filtros Aplicados")
            col_f1, col_f2, col_f3, col_f4 = st.columns(4)
            with col_f1:
                st.info(f"⏱️ {start_horario} → {end_horario}")
            with col_f2:
                st.info(f"⚡ {speed_range[0]:.1f} - {speed_range[1]:.1f} km/h")
            with col_f3:
                total_registros = sum(len(df) for df in dfs_por_periodo.values())
                st.info(f"📊 {total_registros:,} registros")
            with col_f4:
                st.info(f"🏟️ {nome_estadio}")
            
            st.markdown("### 📈 Métricas de Desempenho por Período")
            
            for periodo_nome, df_periodo in dfs_por_periodo.items():
                col1, col2, col3, col4, col5 = st.columns(5)
                with col1:
                    dist = df_periodo['Odometer'].max() - df_periodo['Odometer'].min() if 'Odometer' in df_periodo.columns else 0
                    st.metric(f"{periodo_nome} - Distância", f"{dist:.0f} m")
                with col2:
                    st.metric(f"{periodo_nome} - Vel Máx", f"{df_periodo['Velocity'].max():.1f} km/h")
                with col3:
                    st.metric(f"{periodo_nome} - Vel Média", f"{df_periodo['Velocity'].mean():.1f} km/h")
                with col4:
                    st.metric(f"{periodo_nome} - FC Média", f"{df_periodo['HeartRate'].mean():.0f} bpm")
                with col5:
                    st.metric(f"{periodo_nome} - FC Máx", f"{df_periodo['HeartRate'].max():.0f} bpm")
                st.markdown("---")
            
            # ==================== ABAS ====================
            tab1, tab2, tab3, tab4, tab5 = st.tabs([
                "🗺️ Mapa do Percurso", 
                "📐 Análise Tática por Zonas", 
                "⚡ Perfil Aceleração-Velocidade (ASP)",
                "❤️ Análise de Performance Cardíaca",
                "📊 Comparação Esportiva"
            ])
            
            # TAB 1: MAPA
            with tab1:
                st.subheader("Percurso no Campo de Futebol")
                opcoes_mapa = list(dfs_por_periodo.keys())
                periodo_selecionado_mapa = st.selectbox("Selecionar período para visualizar", options=opcoes_mapa, key="mapa_periodo_select")
                df_mapa = dfs_por_periodo[periodo_selecionado_mapa]
                
                if bounds_estadio:
                    center_lat, center_lon = centro_estadio
                else:
                    center_lat, center_lon = df_mapa['Latitude'].mean(), df_mapa['Longitude'].mean()
                
                fig_map = go.Figure()
                if show_field and bounds_estadio:
                    lat_min, lat_max, lon_min, lon_max = bounds_estadio
                    fig_map.add_shape(type="rect", x0=lon_min, x1=lon_max, y0=lat_min, y1=lat_max,
                                      line=dict(color="white", width=2), fillcolor="rgba(34,139,34,0.2)")
                    fig_map.add_shape(type="line", x0=(lon_min+lon_max)/2, x1=(lon_min+lon_max)/2, y0=lat_min, y1=lat_max,
                                      line=dict(color="white", width=1, dash="dash"))
                
                if len(df_mapa) > 5000:
                    df_mapa_plot = df_mapa.sample(5000, random_state=42)
                    st.caption(f"📊 Exibindo amostra de 5.000 pontos (total: {len(df_mapa):,})")
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
                
                zoom = 18 if bounds_estadio else 15
                fig_map.update_layout(mapbox=dict(style="open-street-map", center=dict(lat=center_lat, lon=center_lon), zoom=zoom),
                                      height=600, margin=dict(l=0, r=0, t=30, b=0),
                                      title=f"Trajetória de {selected_atletas[0]} - {periodo_selecionado_mapa} - {nome_estadio}")
                st.plotly_chart(fig_map, use_container_width=True)
            
            # TAB 2: ANÁLISE TÁTICA
            with tab2:
                st.subheader("Análise Tática - Posicionamento no Campo")
                st.markdown(f"Campo com dimensões oficiais: **{CAMPO_COMPRIMENTO}m x {CAMPO_LARGURA}m**")
                
                opcoes_tatica = list(dfs_por_periodo.keys())
                periodo_selecionado_tatica = st.selectbox("Selecionar período para análise tática", options=opcoes_tatica, key="tatica_periodo_select")
                df_tat = dfs_por_periodo[periodo_selecionado_tatica]
                start_dt_tat = df_tat['start_datetime'].iloc[0] if len(df_tat) > 0 else None
                
                # CARDS DE MÉTRICAS DO PERÍODO
                st.markdown(f"### 📊 Métricas do Período: {periodo_selecionado_tatica}")
                col_metric1, col_metric2, col_metric3, col_metric4, col_metric5 = st.columns(5)
                with col_metric1:
                    dist = df_tat['Odometer'].max() - df_tat['Odometer'].min() if 'Odometer' in df_tat.columns else 0
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
                
                if bounds_estadio:
                    if len(df_tat) > 3000:
                        df_tat_sample = df_tat.sample(3000, random_state=42)
                        st.caption(f"📊 Usando amostra de 3.000 pontos (total: {len(df_tat):,})")
                    else:
                        df_tat_sample = df_tat
                    
                    campo_x, campo_y = [], []
                    for _, row in df_tat_sample.iterrows():
                        x, y = converter_gps_para_campo_cached(row['Latitude'], row['Longitude'], bounds_estadio)
                        campo_x.append(x); campo_y.append(y)
                    df_tat_sample['campo_x'], df_tat_sample['campo_y'] = campo_x, campo_y
                else:
                    st.warning("⚠️ Limites do estádio não definidos.")
                    st.stop()
                
                col_lin, col_col = st.columns(2)
                with col_lin:
                    num_linhas = st.number_input("Número de linhas (divisão horizontal)", 1, 8, 3)
                with col_col:
                    num_colunas = st.number_input("Número de colunas (divisão vertical)", 1, 8, 3)
                
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
                viz_type = st.radio("Tipo de visualização", ["Trajetória com cores por zona", "Mapa de calor de tempo", "Mapa de calor de velocidade"], horizontal=True)
                
                fig_tat = go.Figure()
                for shape in desenhar_campo_futebol():
                    fig_tat.add_shape(shape)
                
                shapes_div, linhas_bins_plot, colunas_bins_plot = desenhar_linhas_divisorias(num_linhas, num_colunas)
                for shape in shapes_div:
                    fig_tat.add_shape(shape)
                
                if viz_type == "Trajetória com cores por zona":
                    cores = px.colors.qualitative.Set3
                    for i, (zona, group) in enumerate(df_tat_sample.groupby('Zona')):
                        fig_tat.add_trace(go.Scatter(x=group['campo_x'], y=group['campo_y'], mode='markers', name=f'Zona {zona}',
                                                     marker=dict(size=4, color=cores[i % len(cores)], opacity=0.7),
                                                     text=[f"Zona: {zona}<br>Vel: {v:.1f} km/h" for v in group['Velocity']],
                                                     hoverinfo='text'))
                elif viz_type == "Mapa de calor de tempo":
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
                
                fig_tat.update_layout(title=f"Análise Tática - {selected_atletas[0]} - {periodo_selecionado_tatica}",
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
                st.download_button("📥 Exportar análise tática", csv_tatico, f"analise_tatica_{selected_atletas[0]}_{periodo_selecionado_tatica}.csv")
            
            # TAB 3: PERFIL ACELERAÇÃO-VELOCIDADE
            with tab3:
                st.subheader("⚡ Perfil Aceleração-Velocidade (Acceleration-Speed Profile)")
                
                with st.expander("📄 **Referência Científica**"):
                    st.markdown("""
                    **Alonso-Callejo, A., et al. (2024).** Validity and reliability of the acceleration-speed profile...  
                    *Journal of Strength and Conditioning Research, 38*(3), 563-570.
                    """)
                
                opcoes_asp = list(dfs_por_periodo.keys())
                periodo_selecionado_asp = st.selectbox("Selecionar período para análise ASP", options=opcoes_asp, key="asp_periodo_select")
                
                asp_results = {}
                with st.spinner("Calculando perfil ASP..."):
                    for nome, df_periodo in dfs_por_periodo.items():
                        asp_metrics = calcular_asp_metrics(df_periodo)
                        if asp_metrics:
                            asp_results[nome] = asp_metrics
                
                if not asp_results:
                    st.warning("⚠️ Dados insuficientes para calcular o Perfil Aceleração-Velocidade.")
                else:
                    st.markdown("### 📈 Curva Aceleração-Velocidade (ASP)")
                    
                    fig_asp = go.Figure()
                    
                    for nome, metrics in asp_results.items():
                        df_periodo = dfs_por_periodo[nome]
                        df_sprints = df_periodo[(df_periodo['Acceleration'] > 0) & (df_periodo['Velocity'] > 0)].copy()
                        
                        if df_sprints['Velocity'].max() < 50:
                            v_ms = df_sprints['Velocity'].values
                        else:
                            v_ms = df_sprints['Velocity'].values / 3.6
                        
                        a_ms2 = df_sprints['Acceleration'].values
                        
                        if len(v_ms) > 1000:
                            idx = np.random.choice(len(v_ms), 1000, replace=False)
                            v_ms = v_ms[idx]
                            a_ms2 = a_ms2[idx]
                        
                        fig_asp.add_trace(go.Scatter(
                            x=a_ms2, y=v_ms,
                            mode='markers',
                            name=f'{nome} (dados)',
                            marker=dict(size=4, opacity=0.5),
                            hoverinfo='skip'
                        ))
                        
                        a_curve = np.linspace(0, metrics['a0'], 50)
                        v_curve = metrics['v0'] * (1 - a_curve / metrics['a0'])
                        
                        fig_asp.add_trace(go.Scatter(
                            x=a_curve, y=v_curve,
                            mode='lines',
                            name=f'{nome} (ajuste)',
                            line=dict(width=2, dash='dash')
                        ))
                    
                    fig_asp.update_layout(
                        title="Relação Aceleração-Velocidade (ASP) - Eixos invertidos",
                        xaxis_title="Aceleração (m/s²)",
                        yaxis_title="Velocidade (m/s)",
                        height=500,
                        legend=dict(orientation="h", yanchor="bottom", y=1.02)
                    )
                    st.plotly_chart(fig_asp, use_container_width=True)
                    
                    asp_df = pd.DataFrame(asp_results).T
                    asp_df = asp_df[['a0', 'v0', 'v_max', 'p_max', 'r2']].round(2)
                    asp_df.columns = ['Aceleração Máx (m/s²)', 'Velocidade Teórica (m/s)', 'Velocidade Máx (m/s)', 'Potência Máx (W/kg)', 'R²']
                    st.dataframe(asp_df, use_container_width=True)
            
            # TAB 4: ANÁLISE DE PERFORMANCE CARDÍACA
            with tab4:
                st.subheader("❤️ Análise de Performance Cardíaca")
                
                opcoes_fc = list(dfs_por_periodo.keys())
                periodo_selecionado_fc = st.selectbox("Selecionar período para análise cardíaca", options=opcoes_fc, key="fc_periodo_select")
                df_fc = dfs_por_periodo[periodo_selecionado_fc]
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
                              line=dict(color='red', width=1.5)),
                    secondary_y=False
                )
                
                fig_fc_acc.add_trace(
                    go.Scatter(x=df_fc_plot['Horario'], y=df_fc_plot['Acceleration'], mode='lines', name='Aceleração',
                              line=dict(color='blue', width=1)),
                    secondary_y=True
                )
                
                fc_max = df_fc_plot['HeartRate'].max()
                limiar = fc_max * 0.85
                fig_fc_acc.add_hline(y=limiar, line_dash="dash", line_color="orange",
                                     annotation_text=f"Limiar: {limiar:.0f} bpm", secondary_y=False)
                
                fig_fc_acc.update_layout(title=f"FC vs Aceleração - {periodo_selecionado_fc}", height=400, hovermode='x unified')
                fig_fc_acc.update_yaxes(title_text="FC (bpm)", secondary_y=False)
                fig_fc_acc.update_yaxes(title_text="Aceleração (m/s²)", secondary_y=True)
                st.plotly_chart(fig_fc_acc, use_container_width=True)
                
                if 'Odometer' in df_fc_plot.columns:
                    st.markdown("### 📈 Distância Acumulada vs FC")
                    
                    df_fc_plot['Distancia_Acumulada'] = df_fc_plot['Odometer'] - df_fc_plot['Odometer'].min()
                    
                    fig_dist_fc = make_subplots(specs=[[{"secondary_y": True}]])
                    
                    fig_dist_fc.add_trace(
                        go.Scatter(x=df_fc_plot['Horario'], y=df_fc_plot['Distancia_Acumulada'], mode='lines', name='Distância',
                                  line=dict(color='green', width=1.5)),
                        secondary_y=False
                    )
                    
                    fig_dist_fc.add_trace(
                        go.Scatter(x=df_fc_plot['Horario'], y=df_fc_plot['HeartRate'], mode='lines', name='FC',
                                  line=dict(color='red', width=1.5)),
                        secondary_y=True
                    )
                    
                    fig_dist_fc.update_layout(title=f"Distância vs FC - {periodo_selecionado_fc}", height=400)
                    fig_dist_fc.update_yaxes(title_text="Distância (m)", secondary_y=False)
                    fig_dist_fc.update_yaxes(title_text="FC (bpm)", secondary_y=True)
                    st.plotly_chart(fig_dist_fc, use_container_width=True)
                
                st.markdown("### 📊 Zonas de Intensidade Cardíaca")
                
                df_fc_plot['Zona_FC'] = pd.cut(df_fc_plot['HeartRate'], 
                                          bins=[0, fc_max*0.6, fc_max*0.75, fc_max*0.9, fc_max],
                                          labels=['Recuperação', 'Aeróbica', 'Anaeróbica', 'Máximo'])
                
                zona_stats = df_fc_plot.groupby('Zona_FC', observed=True).size().reset_index(name='Contagem')
                zona_stats['% do Tempo'] = (zona_stats['Contagem'] / len(df_fc_plot) * 100).round(1)
                st.dataframe(zona_stats, use_container_width=True)
                
                fig_pie = px.pie(zona_stats, values='Contagem', names='Zona_FC', title="Distribuição por Zona de Intensidade")
                st.plotly_chart(fig_pie, use_container_width=True)
            
            # TAB 5: COMPARAÇÃO ESPORTIVA
            with tab5:
                st.subheader("Comparação Esportiva entre Períodos")
                
                var_selecionadas = st.multiselect(
                    "Selecione as variáveis para comparação",
                    options=["Velocidade Média (km/h)", "Velocidade Máxima (km/h)", 
                             "Frequência Cardíaca Média (bpm)", "Frequência Cardíaca Máxima (bpm)", 
                             "Distância Total (m)", "Tempo Total (min)"],
                    default=["Velocidade Média (km/h)", "Distância Total (m)"]
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
                                if 'Odometer' in df_periodo.columns:
                                    row[var] = df_periodo['Odometer'].max() - df_periodo['Odometer'].min()
                            elif var == "Tempo Total (min)":
                                if len(df_periodo) > 1:
                                    sample_rate = df_periodo['Seconds'].diff().median()
                                    row[var] = (len(df_periodo) * sample_rate) / 60
                        comparacao_data.append(row)
                    
                    df_comp = pd.DataFrame(comparacao_data)
                    st.dataframe(df_comp.round(1), use_container_width=True)
                    
                    for var in var_selecionadas:
                        fig_bar = px.bar(df_comp, x='Período', y=var, title=var, text_auto='.1f', color='Período')
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
                        for _, row in df_radar.iterrows():
                            valores = [row[var + " (%)"] for var in var_selecionadas]
                            fig_radar.add_trace(go.Scatterpolar(
                                r=valores,
                                theta=var_selecionadas,
                                fill='toself',
                                name=row['Período'],
                                line=dict(width=2)
                            ))
                        
                        fig_radar.update_layout(
                            polar=dict(radialaxis=dict(visible=True, range=[0, 100])),
                            title="Perfil de Desempenho Normalizado (0-100%)",
                            height=500,
                            showlegend=True
                        )
                        st.plotly_chart(fig_radar, use_container_width=True)
                    
                    csv_comp = df_comp.to_csv(index=False)
                    st.download_button("📥 Exportar comparação", csv_comp, "comparacao_periodos.csv")
        
        else:
            st.error("❌ Nenhum arquivo válido processado.")

else:
    st.markdown("""
    ### 👋 Bem-vindo ao Analisador de Percurso do Atleta!
    
    ### 🚀 Como usar:
    1. **Faça upload** de arquivos CSV na barra lateral
    2. **Selecione o estádio** ou use detecção automática
    3. **Configure os períodos** de análise (horários e duração serão exibidos automaticamente)
    4. **Escolha os atletas** e períodos
    5. **Clique em PROCESSAR ANÁLISE** para iniciar
    
    ### ✨ Funcionalidades:
    - 🏟️ **Campo retangular** com dimensões oficiais (105m x 68m)
    - 📐 **Divisões iguais em metros** - zonas com tamanhos exatos
    - ⏱️ **Períodos personalizados** com horários e duração (prova real antes do processamento)
    - ⚡ **Perfil Aceleração-Velocidade (ASP)** com eixos invertidos
    - ❤️ **Análise de Performance Cardíaca** com zonas de intensidade
    - 📊 **Comparação Esportiva** com radar, correlação e variação percentual
    - 🎯 **Cards de métricas** específicas do período selecionado
    
    ---
    **👈 Configure os filtros e clique em PROCESSAR ANÁLISE!**
    """)