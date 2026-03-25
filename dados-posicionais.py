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
    layout="wide"
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

def fit_velocidade_aceleracao(velocidades, aceleracoes):
    """
    Ajusta a curva de relação Aceleração-Velocidade baseada no modelo linear:
    a(v) = a0 - (a0/v0) * v
    Onde:
    - a0: aceleração máxima (em t=0)
    - v0: velocidade máxima teórica (onde a=0)
    """
    # Remover valores inválidos
    mask = (velocidades > 0) & (aceleracoes > -5) & (aceleracoes < 10)
    v_clean = velocidades[mask]
    a_clean = aceleracoes[mask]
    
    if len(v_clean) < 5:
        return None
    
    try:
        # Modelo linear: a = a0 - (a0/v0) * v
        # Equivalente: a = b - c * v, onde b = a0, c = a0/v0
        def modelo_linear(v, a0, v0):
            return a0 - (a0 / v0) * v
        
        # Ajuste da curva
        popt, _ = curve_fit(modelo_linear, v_clean, a_clean, p0=[5, 8])
        a0, v0 = popt
        
        # Calcular outras métricas
        p_max = a0 * v0 / 4  # Potência máxima teórica
        v_max = np.max(v_clean)  # Velocidade máxima observada
        
        return {
            'a0': a0,  # Aceleração máxima (m/s²)
            'v0': v0,  # Velocidade máxima teórica (m/s)
            'v_max': v_max,  # Velocidade máxima observada (m/s)
            'p_max': p_max,  # Potência máxima (W/kg)
            'r2': np.corrcoef(a_clean, modelo_linear(v_clean, a0, v0))[0, 1]**2,
            'n_points': len(v_clean)
        }
    except:
        return None

def calcular_asp_metrics(df):
    """
    Calcula as métricas do Perfil Aceleração-Velocidade (ASP)
    baseado na metodologia do artigo de referência
    """
    # Filtrar sprints (acelerações positivas e velocidades > 0)
    df_sprints = df[(df['Acceleration'] > 0) & (df['Velocity'] > 0)].copy()
    
    if len(df_sprints) < 10:
        return None
    
    # Converter velocidade para m/s (se estiver em km/h)
    if df_sprints['Velocity'].max() < 50:  # Provavelmente está em m/s
        v_ms = df_sprints['Velocity'].values
    else:
        v_ms = df_sprints['Velocity'].values / 3.6
    
    a_ms2 = df_sprints['Acceleration'].values
    
    # Ajuste da curva
    result = fit_velocidade_aceleracao(v_ms, a_ms2)
    
    if result:
        # Adicionar métricas complementares
        result['v_medio'] = np.mean(v_ms)
        result['a_medio'] = np.mean(a_ms2)
        result['num_sprints'] = len(df_sprints)
        
        # Força relativa (F = m*a, assumindo massa = 1)
        result['f0'] = result['a0']  # Força máxima em N/kg
        
        # Slope do perfil (declividade)
        result['slope'] = -result['a0'] / result['v0'] if result['v0'] > 0 else 0
        
    return result

# ==================== FUNÇÕES DE CONVERSÃO ====================

def converter_gps_para_campo(lat, lon, bounds):
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
        response = requests.get(url, params=params, headers=headers)
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
    """Formata duração em HH:MM:SS"""
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
        with st.sidebar.expander("📝 Cadastrar novo estádio", expanded=True):
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

if 'periodos' not in st.session_state:
    st.session_state.periodos = [{"nome": "1º Tempo", "inicio": 0, "fim": 45}]

if st.sidebar.button("➕ Adicionar período", use_container_width=True):
    st.session_state.periodos.append({"nome": f"Período {len(st.session_state.periodos) + 1}", "inicio": 0, "fim": 45})
    st.rerun()

def get_horario_formatado(minutos, start_dt):
    if start_dt:
        segundos = minutos * 60
        return seconds_to_time_str(segundos, start_dt)
    return f"{int(minutos)}:00"

periodos_para_remover = []
for i, periodo in enumerate(st.session_state.periodos):
    with st.sidebar.expander(f"📅 {periodo['nome']}", expanded=(i == 0)):
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
        
        if 'reference_datetime' in st.session_state and st.session_state.reference_datetime:
            inicio_horario = get_horario_formatado(novo_inicio, st.session_state.reference_datetime)
            fim_horario = get_horario_formatado(novo_fim, st.session_state.reference_datetime)
            duracao_seg = (novo_fim - novo_inicio) * 60
            duracao_str = format_duration(duracao_seg)
            st.caption(f"🕐 {inicio_horario} → {fim_horario}  |  ⏱️ Duração: {duracao_str}")

for i in sorted(periodos_para_remover, reverse=True):
    st.session_state.periodos.pop(i)
    st.rerun()

# Opção "Todos os períodos"
opcoes_periodos = ["Todos os períodos"] + [p['nome'] for p in st.session_state.periodos]
periodos_selecionados = st.sidebar.multiselect(
    "Selecionar períodos para análise",
    options=range(len(opcoes_periodos)),
    format_func=lambda x: opcoes_periodos[x],
    default=[0]
)

# ==================== PROCESSAMENTO PRINCIPAL ====================

if uploaded_files:
    st.sidebar.success(f"✅ {len(uploaded_files)} arquivo(s) carregado(s)")
    
    all_data = []
    all_atletas = []
    all_periodos = []
    all_start_datetimes = []
    
    progress_bar = st.sidebar.progress(0)
    for idx, file in enumerate(uploaded_files):
        df, atleta, periodo, start_datetime = load_data(file)
        if df is not None and len(df) > 0:
            all_data.append(df)
            all_atletas.append(atleta)
            all_periodos.append(periodo)
            all_start_datetimes.append(start_datetime)
            st.sidebar.write(f"📊 {file.name}: {atleta}")
        progress_bar.progress((idx + 1) / len(uploaded_files))
    
    if all_data:
        selected_indices = st.sidebar.multiselect(
            "Escolha os atletas",
            options=range(len(all_atletas)),
            format_func=lambda x: f"{all_atletas[x]} - {all_periodos[x]}" if all_periodos[x] != "Não identificado" else all_atletas[x],
            default=[0] if len(all_atletas) > 0 else []
        )
        
        selected_atletas = [all_atletas[i] for i in selected_indices]
        selected_data = [all_data[i] for i in selected_indices]
        selected_periodos = [all_periodos[i] for i in selected_indices]
        selected_start_datetimes = [all_start_datetimes[i] for i in selected_indices]
        
        reference_datetime = selected_start_datetimes[0] if selected_start_datetimes[0] is not None else None
        st.session_state.reference_datetime = reference_datetime
        
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
            st.sidebar.info(f"🔍 Estádio detectado automaticamente")
        
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
        
        # Adicionar análise combinada
        if len(dfs_por_periodo) > 1:
            dfs_por_periodo["Todos períodos combinados"] = df_combinado_total
        
        # Métricas principais
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
        
        # TAB 1: MAPA (mantido igual)
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
            
            hover_texts = [f"<b>{seconds_to_time_str(row['Seconds'], row['start_datetime'])}</b><br>Vel: {row['Velocity']:.1f} km/h<br>FC: {row['HeartRate']:.0f} bpm" for _, row in df_mapa.iterrows()]
            fig_map.add_trace(go.Scattermapbox(lat=df_mapa['Latitude'], lon=df_mapa['Longitude'], mode='markers',
                                               marker=dict(size=6, color=df_mapa['Velocity'], colorscale='Viridis', showscale=True,
                                                          colorbar=dict(title="Velocidade")), text=hover_texts, hoverinfo='text', name='Percurso'))
            fig_map.add_trace(go.Scattermapbox(lat=[df_mapa['Latitude'].iloc[0]], lon=[df_mapa['Longitude'].iloc[0]],
                                               mode='markers', marker=dict(size=16, color='green'), name='Início'))
            fig_map.add_trace(go.Scattermapbox(lat=[df_mapa['Latitude'].iloc[-1]], lon=[df_mapa['Longitude'].iloc[-1]],
                                               mode='markers', marker=dict(size=16, color='red'), name='Fim'))
            
            zoom = 18 if bounds_estadio else 15
            fig_map.update_layout(mapbox=dict(style="open-street-map", center=dict(lat=center_lat, lon=center_lon), zoom=zoom),
                                  height=700, margin=dict(l=0, r=0, t=30, b=0),
                                  title=f"Trajetória de {selected_atletas[0]} - {periodo_selecionado_mapa} - {nome_estadio}")
            st.plotly_chart(fig_map, use_container_width=True)
        
        # TAB 2: ANÁLISE TÁTICA (mantido igual)
        with tab2:
            st.subheader("Análise Tática - Posicionamento no Campo")
            st.markdown(f"Campo com dimensões oficiais: **{CAMPO_COMPRIMENTO}m x {CAMPO_LARGURA}m**")
            
            opcoes_tatica = list(dfs_por_periodo.keys())
            periodo_selecionado_tatica = st.selectbox("Selecionar período para análise tática", options=opcoes_tatica, key="tatica_periodo_select")
            df_tat = dfs_por_periodo[periodo_selecionado_tatica]
            start_dt_tat = df_tat['start_datetime'].iloc[0] if len(df_tat) > 0 else None
            
            if bounds_estadio:
                campo_x, campo_y = [], []
                for _, row in df_tat.iterrows():
                    x, y = converter_gps_para_campo(row['Latitude'], row['Longitude'], bounds_estadio)
                    campo_x.append(x); campo_y.append(y)
                df_tat['campo_x'], df_tat['campo_y'] = campo_x, campo_y
                st.success(f"✅ Usando limites do {nome_estadio} para conversão")
            else:
                st.warning("⚠️ Limites do estádio não definidos.")
                st.stop()
            
            col_lin, col_col = st.columns(2)
            with col_lin:
                num_linhas = st.number_input("Número de linhas (divisão horizontal)", 1, 10, 3)
                st.info(f"📏 Cada linha terá **{CAMPO_COMPRIMENTO / num_linhas:.1f}m** de comprimento")
            with col_col:
                num_colunas = st.number_input("Número de colunas (divisão vertical)", 1, 10, 3)
                st.info(f"📏 Cada coluna terá **{CAMPO_LARGURA / num_colunas:.1f}m** de largura")
            
            linhas_bins = np.linspace(X_MIN, X_MAX, num_linhas + 1)
            colunas_bins = np.linspace(Y_MIN, Y_MAX, num_colunas + 1)
            
            df_tat['Zona_Linha'] = pd.cut(df_tat['campo_x'], bins=linhas_bins, labels=[f'L{i+1}' for i in range(num_linhas)], include_lowest=True)
            df_tat['Zona_Coluna'] = pd.cut(df_tat['campo_y'], bins=colunas_bins, labels=[f'C{i+1}' for i in range(num_colunas)], include_lowest=True)
            df_tat['Zona'] = df_tat['Zona_Linha'].astype(str) + '-' + df_tat['Zona_Coluna'].astype(str)
            
            zona_metrics = df_tat.groupby('Zona', observed=True).agg({'Seconds': 'count', 'Velocity': ['mean', 'max'], 'HeartRate': ['mean', 'max']}).round(2)
            zona_metrics.columns = ['Contagem', 'Vel_Média', 'Vel_Máx', 'FC_Média', 'FC_Máx']
            
            if len(df_tat) > 1:
                sample_rate = df_tat['Seconds'].diff().median()
                zona_metrics['Tempo(s)'] = zona_metrics['Contagem'] * sample_rate
                zona_metrics['Tempo(min)'] = zona_metrics['Tempo(s)'] / 60
            else:
                zona_metrics['Tempo(s)'] = zona_metrics['Tempo(min)'] = 0
            
            if 'Odometer' in df_tat.columns:
                df_tat_sorted = df_tat.sort_values('Seconds')
                df_tat_sorted['Zona_Ant'] = df_tat_sorted['Zona'].shift(1)
                df_tat_sorted['Delta_Odometer'] = df_tat_sorted['Odometer'].diff()
                df_tat_sorted['Dist_Zona'] = df_tat_sorted.apply(lambda r: r['Delta_Odometer'] if r['Zona'] == r['Zona_Ant'] else 0, axis=1)
                dist_por_zona = df_tat_sorted.groupby('Zona')['Dist_Zona'].sum().round(0)
                zona_metrics['Distância(m)'] = dist_por_zona
            else:
                zona_metrics['Distância(m)'] = 0
            
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
                for i, (zona, group) in enumerate(df_tat.groupby('Zona')):
                    fig_tat.add_trace(go.Scatter(x=group['campo_x'], y=group['campo_y'], mode='markers', name=f'Zona {zona}',
                                                 marker=dict(size=5, color=cores[i % len(cores)], opacity=0.7),
                                                 text=[f"Zona: {zona}<br>Horário: {seconds_to_time_str(t, start_dt_tat)}<br>Vel: {v:.1f} km/h<br>FC: {fc:.0f} bpm"
                                                       for t, v, fc in zip(group['Seconds'], group['Velocity'], group['HeartRate'])], hoverinfo='text'))
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
                                                   font=dict(color="white", size=14, family="Arial Black"),
                                                   bgcolor="rgba(0,0,0,0.7)", bordercolor="white", borderwidth=1, borderpad=4)
                fig_tat.add_trace(go.Scatter(x=df_tat['campo_x'], y=df_tat['campo_y'], mode='markers',
                                             marker=dict(size=3, color='white', opacity=0.5), name='Trajetória', hoverinfo='skip'))
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
                            fig_tat.add_annotation(x=centro_x, y=centro_y, text=f"{vel:.1f}<br>km/h", showarrow=False,
                                                   font=dict(color="white", size=12, family="Arial Black"),
                                                   bgcolor="rgba(0,0,0,0.7)", bordercolor="white", borderwidth=1, borderpad=4)
                fig_tat.add_trace(go.Scatter(x=df_tat['campo_x'], y=df_tat['campo_y'], mode='markers',
                                             marker=dict(size=3, color='white', opacity=0.5), name='Trajetória', hoverinfo='skip'))
            
            fig_tat.update_layout(title=f"Análise Tática - {selected_atletas[0]} - {periodo_selecionado_tatica}",
                                  xaxis_title="Posição (m) - Comprimento", yaxis_title="Posição (m) - Largura",
                                  height=700, xaxis=dict(scaleanchor="y", scaleratio=1, range=[X_MIN-2, X_MAX+2]),
                                  yaxis=dict(range=[Y_MIN-2, Y_MAX+2]), plot_bgcolor='rgba(34,139,34,0.2)')
            st.plotly_chart(fig_tat, use_container_width=True)
            
            with st.expander("📖 **O que é o índice de Intensidade?**"):
                st.markdown("""
                O **Índice de Intensidade** combina **velocidade média** e **tempo de permanência** em cada zona.
                **Fórmula:** `Intensidade = (Vel_Média × Contagem) / Σ(Vel_Média × Contagem) × 100`
                **Interpretação:** >70%: esforço máximo, 30-70%: moderado, <30%: recuperação
                """)
            
            st.markdown("### 📊 Demanda Física por Zona")
            st.dataframe(zona_metrics[['Contagem', '% Frequência', '% Acumulada', 'Tempo(s)', 'Tempo(min)', 'Distância(m)',
                                        'Vel_Média', 'Vel_Máx', 'FC_Média', 'FC_Máx', 'Intensidade (%)']].style.format({
                'Contagem': '{:.0f}', '% Frequência': '{:.1f}%', '% Acumulada': '{:.1f}%', 'Tempo(s)': '{:.1f}',
                'Tempo(min)': '{:.1f}', 'Distância(m)': '{:.0f}', 'Vel_Média': '{:.1f}', 'Vel_Máx': '{:.1f}',
                'FC_Média': '{:.0f}', 'FC_Máx': '{:.0f}', 'Intensidade (%)': '{:.1f}%'
            }), use_container_width=True)
            
            csv_tatico = zona_metrics.reset_index().to_csv(index=False)
            st.download_button("📥 Exportar análise tática", csv_tatico, f"analise_tatica_{selected_atletas[0]}_{periodo_selecionado_tatica}.csv")
        
        # TAB 3: PERFIL ACELERAÇÃO-VELOCIDADE (ASP) - NOVA
        with tab3:
            st.subheader("⚡ Perfil Aceleração-Velocidade (Acceleration-Speed Profile)")
            st.markdown("""
            Baseado no estudo de **Alonso-Callejo et al. (2024)** - *Validity and Reliability of the Acceleration-Speed Profile*  
            Publicado no *Journal of Strength and Conditioning Research*.
            
            O **Perfil Aceleração-Velocidade (ASP)** é um método confiável para avaliar as variáveis mecânicas 
            da cinemática de corrida em jogadores de futebol profissionais.
            """)
            
            # Referência do artigo
            with st.expander("📄 **Referência Científica**"):
                st.markdown("""
                **Alonso-Callejo, A., García-Unanue, J., Guitart-Trench, M., Majano, C., Gallardo, L., & Felipe, J. (2024).**  
                Validity and reliability of the acceleration-speed profile for assessing running kinematics' variables derived 
                from the force-velocity profile in professional soccer players.  
                *Journal of Strength and Conditioning Research, 38*(3), 563-570.  
                DOI: 10.1519/JSC.0000000000004637
                
                **Principais achados:**
                - O ASP, quando calculado com dados de uma semana competitiva completa, é um método confiável
                - As variáveis mais confiáveis são relacionadas à velocidade (V₀, S₀, Vₘₐₓ)
                - Permite analisar a relação entre aceleração e velocidade em sprints
                - Pode ser usado para prescrição de treinamento de resistência, avaliação de desempenho e gestão de retorno ao jogo
                """)
            
            # Selecionar período para análise
            opcoes_asp = list(dfs_por_periodo.keys())
            periodo_selecionado_asp = st.selectbox("Selecionar período para análise ASP", options=opcoes_asp, key="asp_periodo_select")
            df_asp = dfs_por_periodo[periodo_selecionado_asp]
            
            # Calcular ASP metrics
            asp_results = {}
            for nome, df_periodo in dfs_por_periodo.items():
                asp_metrics = calcular_asp_metrics(df_periodo)
                if asp_metrics:
                    asp_results[nome] = asp_metrics
            
            if not asp_results:
                st.warning("⚠️ Dados insuficientes para calcular o Perfil Aceleração-Velocidade. São necessários pelo menos 10 pontos com aceleração positiva e velocidade > 0.")
            else:
                # Gráfico principal: Curva Aceleração-Velocidade
                st.markdown("### 📈 Curva Aceleração-Velocidade (ASP)")
                
                fig_asp = go.Figure()
                
                # Plotar pontos e curvas para cada período
                for nome, metrics in asp_results.items():
                    df_periodo = dfs_por_periodo[nome]
                    
                    # Filtrar sprints
                    df_sprints = df_periodo[(df_periodo['Acceleration'] > 0) & (df_periodo['Velocity'] > 0)].copy()
                    
                    # Converter velocidade para m/s
                    if df_sprints['Velocity'].max() < 50:
                        v_ms = df_sprints['Velocity'].values
                    else:
                        v_ms = df_sprints['Velocity'].values / 3.6
                    
                    a_ms2 = df_sprints['Acceleration'].values
                    
                    # Pontos observados
                    fig_asp.add_trace(go.Scatter(
                        x=v_ms, y=a_ms2,
                        mode='markers',
                        name=f'{nome} (dados)',
                        marker=dict(size=6, opacity=0.6),
                        text=[f"Vel: {v:.2f} m/s<br>Acel: {a:.2f} m/s²" for v, a in zip(v_ms, a_ms2)],
                        hoverinfo='text'
                    ))
                    
                    # Curva ajustada
                    v_curve = np.linspace(0, metrics['v0'], 50)
                    a_curve = metrics['a0'] - (metrics['a0'] / metrics['v0']) * v_curve
                    
                    fig_asp.add_trace(go.Scatter(
                        x=v_curve, y=a_curve,
                        mode='lines',
                        name=f'{nome} (ajuste)',
                        line=dict(width=2, dash='dash'),
                        text=[f"Modelo: a = {metrics['a0']:.2f} - {abs(metrics['slope']):.2f}·v" for _ in v_curve],
                        hoverinfo='text'
                    ))
                    
                    # Ponto de velocidade máxima
                    fig_asp.add_trace(go.Scatter(
                        x=[metrics['v_max']], y=[0],
                        mode='markers+text',
                        marker=dict(size=12, color='red', symbol='x'),
                        text=[f"Vₘₐₓ: {metrics['v_max']:.2f} m/s"],
                        textposition='top center',
                        name=f'{nome} - Vₘₐₓ',
                        showlegend=False
                    ))
                
                fig_asp.update_layout(
                    title="Relação Aceleração-Velocidade (ASP)",
                    xaxis_title="Velocidade (m/s)",
                    yaxis_title="Aceleração (m/s²)",
                    height=500,
                    hovermode='closest',
                    legend=dict(orientation="h", yanchor="bottom", y=1.02)
                )
                st.plotly_chart(fig_asp, use_container_width=True)
                
                # Tabela comparativa de métricas ASP
                st.markdown("### 📊 Métricas do Perfil Aceleração-Velocidade")
                
                asp_df = pd.DataFrame(asp_results).T
                asp_df = asp_df[['a0', 'v0', 'v_max', 'p_max', 'r2', 'n_points', 'slope']].round(3)
                asp_df.columns = ['Aceleração Máx (m/s²)', 'Velocidade Teórica (m/s)', 'Velocidade Máx (m/s)', 
                                  'Potência Máx (W/kg)', 'R²', 'Nº Sprints', 'Slope (1/s)']
                
                st.dataframe(asp_df.style.format({
                    'Aceleração Máx (m/s²)': '{:.2f}',
                    'Velocidade Teórica (m/s)': '{:.2f}',
                    'Velocidade Máx (m/s)': '{:.2f}',
                    'Potência Máx (W/kg)': '{:.2f}',
                    'R²': '{:.3f}',
                    'Nº Sprints': '{:.0f}',
                    'Slope (1/s)': '{:.3f}'
                }), use_container_width=True)
                
                # Interpretação
                st.markdown("### 📖 Interpretação das Métricas ASP")
                col_asp1, col_asp2, col_asp3 = st.columns(3)
                with col_asp1:
                    st.metric("Aceleração Máxima (a₀)", 
                              f"{asp_df['Aceleração Máx (m/s²)'].iloc[0]:.2f} m/s²",
                              help="Capacidade de aceleração inicial. Valores > 5 m/s² indicam boa capacidade de sprint")
                with col_asp2:
                    st.metric("Velocidade Teórica (v₀)", 
                              f"{asp_df['Velocidade Teórica (m/s)'].iloc[0]:.2f} m/s",
                              help="Velocidade máxima teórica. Valores > 8 m/s são excelentes")
                with col_asp3:
                    st.metric("Potência Máxima (Pₘₐₓ)", 
                              f"{asp_df['Potência Máx (W/kg)'].iloc[0]:.2f} W/kg",
                              help="Potência mecânica. Valores > 10 W/kg são considerados de elite")
                
                st.markdown("""
                **Interpretação Clínica e Esportiva:**
                - **a₀ (Aceleração Máxima)**: Capacidade de gerar força nos primeiros passos. Valores mais altos indicam melhor capacidade de aceleração.
                - **v₀ (Velocidade Teórica)**: Capacidade de atingir altas velocidades. Correlaciona-se com desempenho em sprints longos.
                - **R²**: Qualidade do ajuste do modelo. Valores > 0.7 indicam boa confiabilidade do perfil.
                - **Slope**: Taxa de declínio da aceleração com a velocidade. Valores mais negativos indicam maior fadiga em sprints.
                
                **Aplicações Práticas (conforme Alonso-Callejo et al., 2024):**
                - Prescrição de treinamento de resistência
                - Avaliação de desempenho
                - Gestão de retorno ao jogo (return-to-play)
                - Identificação de assimetrias entre membros
                """)
                
                # Gráfico comparativo entre períodos
                if len(asp_df) > 1:
                    st.markdown("### 📊 Comparação do Perfil ASP entre Períodos")
                    
                    fig_comp_asp = go.Figure()
                    metricas_para_plotar = ['Aceleração Máx (m/s²)', 'Velocidade Teórica (m/s)', 'Potência Máx (W/kg)']
                    
                    for metrica in metricas_para_plotar:
                        fig_comp_asp.add_trace(go.Bar(
                            x=asp_df.index,
                            y=asp_df[metrica],
                            name=metrica,
                            text=asp_df[metrica].round(2),
                            textposition='outside'
                        ))
                    
                    fig_comp_asp.update_layout(
                        title="Comparação de Métricas ASP entre Períodos",
                        xaxis_title="Período",
                        yaxis_title="Valor",
                        height=450,
                        barmode='group'
                    )
                    st.plotly_chart(fig_comp_asp, use_container_width=True)
        
        # TAB 4: ANÁLISE DE PERFORMANCE CARDÍACA - NOVA
        with tab4:
            st.subheader("❤️ Análise de Performance Cardíaca")
            st.markdown("Análise integrada da frequência cardíaca, aceleração e distância percorrida.")
            
            opcoes_fc = list(dfs_por_periodo.keys())
            periodo_selecionado_fc = st.selectbox("Selecionar período para análise cardíaca", options=opcoes_fc, key="fc_periodo_select")
            df_fc = dfs_por_periodo[periodo_selecionado_fc]
            start_dt_fc = df_fc['start_datetime'].iloc[0] if len(df_fc) > 0 else None
            
            # Criar coluna de horário
            df_fc['Horario'] = df_fc['Seconds'].apply(lambda x: seconds_to_time_str(x, start_dt_fc))
            
            # Gráfico 1: FC e Aceleração sobrepostos
            st.markdown("### 📈 Frequência Cardíaca e Aceleração")
            
            fig_fc_acc = make_subplots(specs=[[{"secondary_y": True}]])
            
            # FC (eixo primário)
            fig_fc_acc.add_trace(
                go.Scatter(x=df_fc['Horario'], y=df_fc['HeartRate'], mode='lines', name='FC',
                          line=dict(color='red', width=2), fill='tozeroy', fillcolor='rgba(231,76,60,0.2)'),
                secondary_y=False
            )
            
            # Aceleração (eixo secundário)
            fig_fc_acc.add_trace(
                go.Scatter(x=df_fc['Horario'], y=df_fc['Acceleration'], mode='lines', name='Aceleração',
                          line=dict(color='blue', width=1.5), fill='tozeroy', fillcolor='rgba(52,152,219,0.1)'),
                secondary_y=True
            )
            
            # Linha de limiar anaeróbico (85% da FC máxima)
            fc_max = df_fc['HeartRate'].max()
            limiar_anaerobico = fc_max * 0.85
            fig_fc_acc.add_hline(y=limiar_anaerobico, line_dash="dash", line_color="orange",
                                 annotation_text=f"Limiar Anaeróbico: {limiar_anaerobico:.0f} bpm",
                                 annotation_position="top right", secondary_y=False)
            
            fig_fc_acc.update_layout(
                title=f"Frequência Cardíaca vs Aceleração - {periodo_selecionado_fc}",
                xaxis_title="Horário",
                height=450,
                hovermode='x unified'
            )
            fig_fc_acc.update_yaxes(title_text="Frequência Cardíaca (bpm)", secondary_y=False, color="red")
            fig_fc_acc.update_yaxes(title_text="Aceleração (m/s²)", secondary_y=True, color="blue")
            st.plotly_chart(fig_fc_acc, use_container_width=True)
            
            # Gráfico 2: Distância acumulada e FC sobrepostos
            st.markdown("### 📈 Distância Acumulada vs Frequência Cardíaca")
            
            if 'Odometer' in df_fc.columns:
                # Calcular distância acumulada normalizada
                df_fc['Distancia_Acumulada'] = df_fc['Odometer'] - df_fc['Odometer'].min()
                df_fc['Distancia_Normalizada'] = df_fc['Distancia_Acumulada'] / df_fc['Distancia_Acumulada'].max() * 100 if df_fc['Distancia_Acumulada'].max() > 0 else 0
                df_fc['FC_Normalizada'] = df_fc['HeartRate'] / fc_max * 100 if fc_max > 0 else 0
                
                fig_dist_fc = make_subplots(specs=[[{"secondary_y": True}]])
                
                # Distância (eixo primário)
                fig_dist_fc.add_trace(
                    go.Scatter(x=df_fc['Horario'], y=df_fc['Distancia_Acumulada'], mode='lines', name='Distância Acumulada',
                              line=dict(color='green', width=2), fill='tozeroy', fillcolor='rgba(46,204,113,0.2)'),
                    secondary_y=False
                )
                
                # FC (eixo secundário)
                fig_dist_fc.add_trace(
                    go.Scatter(x=df_fc['Horario'], y=df_fc['HeartRate'], mode='lines', name='FC',
                              line=dict(color='red', width=2)),
                    secondary_y=True
                )
                
                fig_dist_fc.update_layout(
                    title=f"Distância Acumulada vs Frequência Cardíaca - {periodo_selecionado_fc}",
                    xaxis_title="Horário",
                    height=450,
                    hovermode='x unified'
                )
                fig_dist_fc.update_yaxes(title_text="Distância (m)", secondary_y=False, color="green")
                fig_dist_fc.update_yaxes(title_text="FC (bpm)", secondary_y=True, color="red")
                st.plotly_chart(fig_dist_fc, use_container_width=True)
                
                # Gráfico 3: Relação FC vs Distância (scatter)
                st.markdown("### 🎯 Relação FC vs Distância Acumulada")
                
                fig_scatter_fc = px.scatter(
                    df_fc, x='Distancia_Acumulada', y='HeartRate', color='Velocity',
                    title="Relação entre Distância Percorrida e Frequência Cardíaca",
                    labels={'Distancia_Acumulada': 'Distância Acumulada (m)', 'HeartRate': 'FC (bpm)', 'Velocity': 'Velocidade (km/h)'},
                    color_continuous_scale='Viridis'
                )
                
                # Adicionar linha de tendência
                z = np.polyfit(df_fc['Distancia_Acumulada'], df_fc['HeartRate'], 1)
                p = np.poly1d(z)
                x_trend = np.linspace(df_fc['Distancia_Acumulada'].min(), df_fc['Distancia_Acumulada'].max(), 100)
                fig_scatter_fc.add_trace(go.Scatter(
                    x=x_trend, y=p(x_trend),
                    mode='lines',
                    name=f'Tendência: {z[0]:.2f} m/bpm',
                    line=dict(color='red', dash='dash')
                ))
                
                st.plotly_chart(fig_scatter_fc, use_container_width=True)
                
                st.markdown(f"""
                **Análise da Relação FC-Distância:**
                - **Coeficiente angular da tendência:** {z[0]:.2f} bpm/m
                - **Interpretação:** Quanto maior o coeficiente, mais rapidamente a FC aumenta com a distância percorrida
                - **Limiar anaeróbico atingido em:** {df_fc[df_fc['HeartRate'] >= limiar_anaerobico]['Distancia_Acumulada'].min():.0f} m (se aplicável)
                """)
            
            # Zonas de intensidade cardíaca
            st.markdown("### 📊 Distribuição por Zonas de Intensidade Cardíaca")
            
            fc_max = df_fc['HeartRate'].max()
            df_fc['Zona_FC'] = pd.cut(df_fc['HeartRate'], 
                                      bins=[0, fc_max*0.6, fc_max*0.75, fc_max*0.9, fc_max],
                                      labels=['Recuperação (<60%)', 'Aeróbica (60-75%)', 'Anaeróbica (75-90%)', 'Máximo (>90%)'])
            
            zona_stats = df_fc.groupby('Zona_FC', observed=True).agg({
                'HeartRate': 'count',
                'Seconds': 'count',
                'Velocity': 'mean'
            }).round(0)
            
            zona_stats.columns = ['Contagem', 'Tempo (samples)', 'Vel Média (km/h)']
            zona_stats['% do Tempo'] = (zona_stats['Contagem'] / len(df_fc) * 100).round(1).astype(str) + '%'
            
            # Calcular tempo real
            if len(df_fc) > 1:
                sample_rate = df_fc['Seconds'].diff().median()
                zona_stats['Tempo Real (min)'] = (zona_stats['Contagem'] * sample_rate / 60).round(1)
            else:
                zona_stats['Tempo Real (min)'] = 0
            
            st.dataframe(zona_stats[['Contagem', '% do Tempo', 'Tempo Real (min)', 'Vel Média (km/h)']], use_container_width=True)
            
            # Gráfico de pizza das zonas
            fig_pie_fc = px.pie(
                zona_stats, values='Contagem', names=zona_stats.index,
                title="Distribuição do Tempo por Zona de Intensidade Cardíaca",
                color_discrete_sequence=['#2ecc71', '#f1c40f', '#e67e22', '#e74c3c']
            )
            st.plotly_chart(fig_pie_fc, use_container_width=True)
        
        # TAB 5: COMPARAÇÃO ESPORTIVA (mantido)
        with tab4:  # Será tab5 na interface
            st.subheader("Comparação Esportiva entre Períodos")
            st.markdown("Análise detalhada das diferenças de desempenho entre os períodos selecionados.")
            
            var_selecionadas = st.multiselect(
                "Selecione as variáveis para análise comparativa",
                options=["Velocidade Média (km/h)", "Velocidade Máxima (km/h)", 
                         "Frequência Cardíaca Média (bpm)", "Frequência Cardíaca Máxima (bpm)", 
                         "Distância Total (m)", "Tempo Total (min)", "Intensidade Média"],
                default=["Velocidade Média (km/h)", "Frequência Cardíaca Média (bpm)", "Distância Total (m)"],
                key="var_comparacao"
            )
            
            if not var_selecionadas:
                st.warning("Selecione pelo menos uma variável para análise.")
            else:
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
                            else:
                                row[var] = 0
                        elif var == "Tempo Total (min)":
                            if len(df_periodo) > 1:
                                sample_rate = df_periodo['Seconds'].diff().median()
                                row[var] = (len(df_periodo) * sample_rate) / 60
                            else:
                                row[var] = 0
                        elif var == "Intensidade Média":
                            vel_max_total = max([df['Velocity'].max() for df in dfs_por_periodo.values()])
                            if vel_max_total > 0:
                                row[var] = (df_periodo['Velocity'].mean() / vel_max_total * 100)
                            else:
                                row[var] = 0
                    
                    comparacao_data.append(row)
                
                df_comparacao = pd.DataFrame(comparacao_data)
                
                st.markdown("### 📊 Tabela Comparativa de Desempenho")
                styled_df = df_comparacao.copy()
                for col in var_selecionadas:
                    if col in styled_df.columns:
                        styled_df[col] = styled_df[col].round(1)
                st.dataframe(styled_df, use_container_width=True)
                
                # Gráfico de barras
                st.markdown("### 📈 Comparação Visual")
                for var in var_selecionadas:
                    st.markdown(f"#### {var}")
                    fig_comp = go.Figure()
                    valores = df_comparacao[var].values
                    periodos = df_comparacao['Período'].values
                    max_val = max(valores)
                    cores = ['#2ecc71' if v == max_val else '#3498db' for v in valores]
                    
                    fig_comp.add_trace(go.Bar(
                        x=periodos, y=valores,
                        marker_color=cores,
                        text=[f"{v:.1f}" for v in valores],
                        textposition='outside'
                    ))
                    
                    media = np.mean(valores)
                    fig_comp.add_hline(y=media, line_dash="dash", line_color="orange",
                                      annotation_text=f"Média: {media:.1f}", annotation_position="top right")
                    
                    unidade = var.split('(')[-1].replace(')', '') if '(' in var else ''
                    fig_comp.update_layout(
                        title=f"{var} por Período",
                        xaxis_title="Período",
                        yaxis_title=unidade if unidade else var,
                        height=400,
                        showlegend=False
                    )
                    st.plotly_chart(fig_comp, use_container_width=True)
                
                # Radar Chart
                if len(df_comparacao) >= 2:
                    st.markdown("### 🎯 Perfil de Desempenho (Radar)")
                    
                    df_radar = df_comparacao.copy()
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
                        height=600,
                        showlegend=True
                    )
                    st.plotly_chart(fig_radar, use_container_width=True)
                
                csv_comparacao = df_comparacao.to_csv(index=False)
                st.download_button("📥 Exportar análise comparativa", csv_comparacao, f"comparacao_periodos_{selected_atletas[0]}.csv")
        
        st.markdown("---")
        st.download_button("📥 Exportar dados filtrados", pd.concat(dfs_por_periodo.values()).to_csv(index=False), "dados_filtrados.csv")
        st.markdown(f"**Períodos analisados:** {', '.join(dfs_por_periodo.keys())}")
    
    else:
        st.error("❌ Nenhum arquivo válido processado.")

else:
    st.markdown("""
    ### 👋 Bem-vindo ao Analisador de Percurso do Atleta!
    
    ### 🚀 Como usar:
    1. **Faça upload** de arquivos CSV na barra lateral
    2. **Selecione o estádio** ou use detecção automática
    3. **Escolha os atletas** para análise
    4. **Configure os períodos** de análise
    5. **Ajuste os filtros** de tempo e velocidade
    6. **Explore as 5 abas** de análise
    
    ### ✨ Funcionalidades:
    - 🏟️ **Campo retangular** com dimensões oficiais (105m x 68m)
    - 📐 **Divisões iguais em metros** - zonas com tamanhos exatos
    - ⏱️ **Períodos personalizados** com horários e duração
    - ⚡ **Perfil Aceleração-Velocidade (ASP)** baseado em pesquisa científica
    - ❤️ **Análise de Performance Cardíaca** com zonas de intensidade
    - 📊 **Comparação esportiva** com radar, correlação e variação percentual
    
    ---
    **👈 Faça upload para começar!**
    """)