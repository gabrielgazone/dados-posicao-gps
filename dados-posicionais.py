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
# Dimensões oficiais do campo (em metros)
CAMPO_COMPRIMENTO = 105  # metros (comprimento)
CAMPO_LARGURA = 68       # metros (largura)

# ==================== FUNÇÕES DE CONVERSÃO DE COORDENADAS ====================

def converter_gps_para_campo(lat, lon, bounds):
    """
    Converte coordenadas GPS para coordenadas do campo (x, y) em metros
    bounds = (lat_min, lat_max, lon_min, lon_max)
    """
    lat_min, lat_max, lon_min, lon_max = bounds
    
    # Normalizar para 0-1
    norm_x = (lon - lon_min) / (lon_max - lon_min)
    norm_y = (lat - lat_min) / (lat_max - lat_min)
    
    # Converter para coordenadas do campo (-52.5 a 52.5 no X, -34 a 34 no Y)
    campo_x = (norm_x * CAMPO_COMPRIMENTO) - (CAMPO_COMPRIMENTO / 2)
    campo_y = (norm_y * CAMPO_LARGURA) - (CAMPO_LARGURA / 2)
    
    return campo_x, campo_y

def desenhar_campo_futebol():
    """
    Desenha todas as linhas do campo de futebol (retangular, proporção correta)
    """
    shapes = []
    
    # Dimensões do campo
    comprimento = CAMPO_COMPRIMENTO
    largura = CAMPO_LARGURA
    x_min = -comprimento / 2
    x_max = comprimento / 2
    y_min = -largura / 2
    y_max = largura / 2
    
    # 1. Contorno do campo
    shapes.append(go.layout.Shape(
        type="rect",
        x0=x_min, x1=x_max,
        y0=y_min, y1=y_max,
        line=dict(color="white", width=2),
        fillcolor="rgba(34, 139, 34, 0.3)",
        layer="below"
    ))
    
    # 2. Linha do meio de campo
    shapes.append(go.layout.Shape(
        type="line",
        x0=0, x1=0,
        y0=y_min, y1=y_max,
        line=dict(color="white", width=2),
        layer="below"
    ))
    
    # 3. Círculo central (raio 9.15m)
    shapes.append(go.layout.Shape(
        type="circle",
        x0=-9.15, x1=9.15,
        y0=-9.15, y1=9.15,
        line=dict(color="white", width=2),
        fillcolor="rgba(255,255,255,0)",
        layer="below"
    ))
    
    # 4. Ponto central
    shapes.append(go.layout.Shape(
        type="circle",
        x0=-0.3, x1=0.3,
        y0=-0.3, y1=0.3,
        line=dict(color="white", width=1),
        fillcolor="white",
        layer="below"
    ))
    
    # 5. Grandes áreas (16.5m da linha de fundo)
    grande_area_prof = 16.5
    grande_area_larg = 40.3  # 16.5 + 16.5 + 7.32? Na verdade 40.3m de largura
    
    # Grande área direita
    shapes.append(go.layout.Shape(
        type="rect",
        x0=x_max - grande_area_prof, x1=x_max,
        y0=-grande_area_larg/2, y1=grande_area_larg/2,
        line=dict(color="white", width=2),
        fillcolor="rgba(255,255,255,0)",
        layer="below"
    ))
    
    # Grande área esquerda
    shapes.append(go.layout.Shape(
        type="rect",
        x0=x_min, x1=x_min + grande_area_prof,
        y0=-grande_area_larg/2, y1=grande_area_larg/2,
        line=dict(color="white", width=2),
        fillcolor="rgba(255,255,255,0)",
        layer="below"
    ))
    
    # 6. Pequenas áreas (5.5m da linha de fundo)
    pequena_area_prof = 5.5
    pequena_area_larg = 18.32  # 5.5 + 5.5 + 7.32 = 18.32m
    
    # Pequena área direita
    shapes.append(go.layout.Shape(
        type="rect",
        x0=x_max - pequena_area_prof, x1=x_max,
        y0=-pequena_area_larg/2, y1=pequena_area_larg/2,
        line=dict(color="white", width=2),
        fillcolor="rgba(255,255,255,0)",
        layer="below"
    ))
    
    # Pequena área esquerda
    shapes.append(go.layout.Shape(
        type="rect",
        x0=x_min, x1=x_min + pequena_area_prof,
        y0=-pequena_area_larg/2, y1=pequena_area_larg/2,
        line=dict(color="white", width=2),
        fillcolor="rgba(255,255,255,0)",
        layer="below"
    ))
    
    # 7. Marcas de pênalti (11m da linha de fundo)
    penalty_dist = 11
    # Pênalti direita
    shapes.append(go.layout.Shape(
        type="circle",
        x0=x_max - penalty_dist - 0.3, x1=x_max - penalty_dist + 0.3,
        y0=-0.3, y1=0.3,
        line=dict(color="white", width=1),
        fillcolor="white",
        layer="below"
    ))
    # Pênalti esquerda
    shapes.append(go.layout.Shape(
        type="circle",
        x0=x_min + penalty_dist - 0.3, x1=x_min + penalty_dist + 0.3,
        y0=-0.3, y1=0.3,
        line=dict(color="white", width=1),
        fillcolor="white",
        layer="below"
    ))
    
    # 8. Arcos da grande área (opcional)
    arc_radius = 9.15
    # Arco direito
    shapes.append(go.layout.Shape(
        type="path",
        path=f"M {x_max - 11} 0 A {arc_radius} {arc_radius} 0 0 1 {x_max - 11 + arc_radius * 0.5} {arc_radius * 0.5}",
        line=dict(color="white", width=1),
        layer="below"
    ))
    # Arco esquerdo
    shapes.append(go.layout.Shape(
        type="path",
        path=f"M {x_min + 11} 0 A {arc_radius} {arc_radius} 0 0 0 {x_min + 11 - arc_radius * 0.5} {arc_radius * 0.5}",
        line=dict(color="white", width=1),
        layer="below"
    ))
    
    return shapes

# ==================== BANCO DE DADOS DE ESTÁDIOS ====================

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
            cursor.execute('''
                INSERT INTO estadios (nome, cidade, pais, endereco, latitude_centro, longitude_centro,
                                     latitude_min, latitude_max, longitude_min, longitude_max, pontos_json)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', estadio)
        
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
    cursor.execute('''
        SELECT id, nome, cidade, pais, endereco, latitude_centro, longitude_centro,
               latitude_min, latitude_max, longitude_min, longitude_max, pontos_json
        FROM estadios WHERE id = ?
    ''', (id_estadio,))
    resultado = cursor.fetchone()
    conn.close()
    
    if resultado:
        pontos = None
        if resultado[10]:
            pontos = json.loads(resultado[10])
        
        return {
            'id': resultado[0],
            'nome': resultado[1],
            'cidade': resultado[2],
            'pais': resultado[3],
            'endereco': resultado[4],
            'centro': (resultado[5], resultado[6]),
            'bounds': (resultado[7], resultado[8], resultado[9], resultado[10]),
            'pontos': pontos
        }
    return None

def adicionar_estadio(nome, cidade, pais, endereco, centro_lat, centro_lon, pontos):
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    if pontos:
        lats = [p['lat'] for p in pontos]
        lons = [p['lng'] for p in pontos]
        lat_min = min(lats)
        lat_max = max(lats)
        lon_min = min(lons)
        lon_max = max(lons)
    else:
        lat_min = centro_lat - 0.003
        lat_max = centro_lat + 0.003
        lon_min = centro_lon - 0.003
        lon_max = centro_lon + 0.003
    
    cursor.execute('''
        INSERT INTO estadios (nome, cidade, pais, endereco, latitude_centro, longitude_centro,
                             latitude_min, latitude_max, longitude_min, longitude_max, pontos_json)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    ''', (
        nome, cidade, pais, endereco,
        centro_lat, centro_lon,
        lat_min, lat_max, lon_min, lon_max,
        json.dumps(pontos) if pontos else None
    ))
    
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
                lat = float(dados[0]['lat'])
                lon = float(dados[0]['lon'])
                display_name = dados[0].get('display_name', endereco)
                return lat, lon, display_name
        return None, None, None
    except Exception as e:
        return None, None, None

# ==================== FUNÇÕES AUXILIARES ====================

def seconds_to_time_str(seconds, start_datetime):
    if start_datetime is None:
        return f"{int(seconds // 3600):02d}:{int((seconds % 3600) // 60):02d}:{int(seconds % 60):02d}"
    target_time = start_datetime + timedelta(seconds=seconds)
    return target_time.strftime("%H:%M:%S")

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
                    if match:
                        periodo = match.group(1).strip()
                    else:
                        periodo = line.split(':')[1].strip().strip('"').strip(';')
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
                     'Latitude', 'Longitude', 'HeartRate', 'PlayerLoad', 
                     'PositionalQuality', 'HDOP', 'Sats']
        
        if len(df.columns) == len(col_names):
            df.columns = col_names
        else:
            for i, col in enumerate(df.columns[:len(col_names)]):
                df.rename(columns={col: col_names[i]}, inplace=True)
        
        numeric_cols = ['Seconds', 'Velocity', 'Acceleration', 'Odometer', 
                        'Latitude', 'Longitude', 'HeartRate', 'PlayerLoad', 
                        'PositionalQuality', 'HDOP', 'Sats']
        
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
uploaded_files = st.sidebar.file_uploader(
    "Escolha os arquivos CSV",
    type=['csv'],
    accept_multiple_files=True
)

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
    selecao_estadio = st.sidebar.selectbox(
        "Selecione o estádio ou modo de detecção",
        options=opcoes_estadio,
        index=0,
        key="selecao_estadio"
    )
    
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
            
            if 'centro_lat_temp' in st.session_state:
                center_lat = st.session_state.centro_lat_temp
                center_lon = st.session_state.centro_lon_temp
            else:
                center_lat = -23.5505
                center_lon = -46.6333
            
            col_lim1, col_lim2 = st.columns(2)
            with col_lim1:
                lat_min = st.number_input("Latitude mínima (Sul)", value=center_lat - 0.002, format="%.8f")
                lat_max = st.number_input("Latitude máxima (Norte)", value=center_lat + 0.002, format="%.8f")
            with col_lim2:
                lon_min = st.number_input("Longitude mínima (Oeste)", value=center_lon - 0.002, format="%.8f")
                lon_max = st.number_input("Longitude máxima (Leste)", value=center_lon + 0.002, format="%.8f")
            
            centro_lat = (lat_min + lat_max) / 2
            centro_lon = (lon_min + lon_max) / 2
            
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
                        pontos = [
                            {'lat': lat_max, 'lng': lon_min, 'nome': 'NO'},
                            {'lat': lat_max, 'lng': lon_max, 'nome': 'NE'},
                            {'lat': lat_min, 'lng': lon_min, 'nome': 'SO'},
                            {'lat': lat_min, 'lng': lon_max, 'nome': 'SE'}
                        ]
                        endereco = f"{nome_novo}, {cidade_nova}, {pais_novo}"
                        adicionar_estadio(nome_novo, cidade_nova, pais_novo, endereco, centro_lat, centro_lon, pontos)
                        st.success(f"✅ Estádio {nome_novo} cadastrado!")
                        import time; time.sleep(1); st.rerun()
                else:
                    st.error("❌ Nome do estádio é obrigatório!")
else:
    st.sidebar.warning("Nenhum estádio cadastrado")
    selecao_estadio = "Detectar automaticamente"

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
        
        df_calibracao = selected_data[0]
        
        if selecao_estadio == "Detectar automaticamente" or estadio_selecionado is None:
            lat_min = df_calibracao['Latitude'].quantile(0.01)
            lat_max = df_calibracao['Latitude'].quantile(0.99)
            lon_min = df_calibracao['Longitude'].quantile(0.01)
            lon_max = df_calibracao['Longitude'].quantile(0.99)
            centro_lat = (lat_min + lat_max) / 2
            centro_lon = (lon_min + lon_max) / 2
            bounds_estadio = (lat_min, lat_max, lon_min, lon_max)
            centro_estadio = (centro_lat, centro_lon)
            nome_estadio = "Detectado automaticamente"
            st.sidebar.info(f"🔍 Estádio detectado automaticamente")
        
        st.sidebar.markdown("---")
        st.sidebar.subheader("⏱️ Filtro Temporal")
        
        min_time = float('inf')
        max_time = 0
        for df in selected_data:
            min_time = min(min_time, df['Seconds'].min())
            max_time = max(max_time, df['Seconds'].max())
        
        if reference_datetime:
            selected_range = st.sidebar.slider(
                "Intervalo de tempo",
                min_value=float(min_time),
                max_value=float(max_time),
                value=(float(min_time), float(max_time)),
                step=1.0
            )
            start_time, end_time = selected_range
            start_horario = seconds_to_time_str(start_time, reference_datetime)
            end_horario = seconds_to_time_str(end_time, reference_datetime)
        else:
            start_time, end_time = min_time, max_time
            start_horario = f"{int(start_time//3600):02d}:{int((start_time%3600)//60):02d}:{int(start_time%60):02d}"
            end_horario = f"{int(end_time//3600):02d}:{int((end_time%3600)//60):02d}:{int(end_time%60):02d}"
        
        st.sidebar.markdown("---")
        st.sidebar.subheader("⚡ Filtro Velocidade")
        
        max_speed = 0
        for df in selected_data:
            max_speed = max(max_speed, df['Velocity'].max())
        
        speed_range = st.sidebar.slider(
            "Velocidade (km/h)",
            min_value=0.0,
            max_value=float(max_speed),
            value=(0.0, float(max_speed)),
            step=0.5
        )
        
        st.sidebar.markdown("---")
        st.sidebar.subheader("🎨 Opções")
        show_field = st.sidebar.checkbox("Mostrar campo", value=True)
        
        dfs_filtered = []
        for df, atleta, periodo, start_dt in zip(selected_data, selected_atletas, selected_periodos, selected_start_datetimes):
            time_filter = (df['Seconds'] >= start_time) & (df['Seconds'] <= end_time)
            speed_filter = (df['Velocity'] >= speed_range[0]) & (df['Velocity'] <= speed_range[1])
            df_filtered = df[time_filter & speed_filter].copy()
            df_filtered['Atleta'] = atleta
            df_filtered['Periodo'] = periodo
            df_filtered['start_datetime'] = start_dt
            dfs_filtered.append(df_filtered)
        
        df_combined = pd.concat(dfs_filtered, ignore_index=True) if dfs_filtered else pd.DataFrame()
        
        if len(df_combined) == 0:
            st.warning("⚠️ Nenhum dado encontrado.")
            st.stop()
        
        df_main = dfs_filtered[0]
        atleta_main = selected_atletas[0]
        start_dt_main = selected_start_datetimes[0]
        
        st.markdown("### 📊 Filtros Aplicados")
        col_f1, col_f2, col_f3, col_f4 = st.columns(4)
        with col_f1:
            st.info(f"⏱️ {start_horario} → {end_horario}")
        with col_f2:
            st.info(f"⚡ {speed_range[0]:.1f} - {speed_range[1]:.1f} km/h")
        with col_f3:
            st.info(f"📊 {len(df_combined):,} registros")
        with col_f4:
            st.info(f"🏟️ {nome_estadio}")
        
        st.markdown("### 📈 Métricas de Desempenho")
        col1, col2, col3, col4, col5 = st.columns(5)
        with col1:
            dist = df_main['Odometer'].max() if 'Odometer' in df_main.columns else 0
            st.metric("Distância", f"{dist:.0f} m")
        with col2:
            st.metric("Vel Máx", f"{df_main['Velocity'].max():.1f} km/h")
        with col3:
            st.metric("Vel Média", f"{df_main['Velocity'].mean():.1f} km/h")
        with col4:
            st.metric("FC Média", f"{df_main['HeartRate'].mean():.0f} bpm")
        with col5:
            st.metric("FC Máx", f"{df_main['HeartRate'].max():.0f} bpm")
        
        # ==================== ABAS ====================
        tab1, tab2 = st.tabs(["🗺️ Mapa do Percurso", "📐 Análise Tática por Zonas"])
        
        # TAB 1: MAPA (coordenadas GPS)
        with tab1:
            st.subheader("Percurso no Campo de Futebol")
            
            if len(selected_atletas) > 1:
                atleta_mapa = st.selectbox("Atleta", selected_atletas, key="mapa_select")
                idx_mapa = selected_atletas.index(atleta_mapa)
                df_mapa = dfs_filtered[idx_mapa].copy()
                start_dt_mapa = selected_start_datetimes[idx_mapa]
                atleta_mapa_nome = atleta_mapa
            else:
                df_mapa = df_main.copy()
                start_dt_mapa = start_dt_main
                atleta_mapa_nome = atleta_main
            
            if bounds_estadio:
                center_lat, center_lon = centro_estadio
            else:
                center_lat = df_mapa['Latitude'].mean()
                center_lon = df_mapa['Longitude'].mean()
            
            fig_map = go.Figure()
            
            if show_field and bounds_estadio:
                lat_min, lat_max, lon_min, lon_max = bounds_estadio
                fig_map.add_shape(type="rect", x0=lon_min, x1=lon_max, y0=lat_min, y1=lat_max,
                                  line=dict(color="white", width=2), fillcolor="rgba(34,139,34,0.2)")
                fig_map.add_shape(type="line", x0=(lon_min+lon_max)/2, x1=(lon_min+lon_max)/2,
                                  y0=lat_min, y1=lat_max, line=dict(color="white", width=1, dash="dash"))
            
            hover_texts = []
            for _, row in df_mapa.iterrows():
                time_str = seconds_to_time_str(row['Seconds'], start_dt_mapa)
                hover_texts.append(f"<b>{time_str}</b><br>Vel: {row['Velocity']:.1f} km/h<br>FC: {row['HeartRate']:.0f} bpm")
            
            fig_map.add_trace(go.Scattermapbox(
                lat=df_mapa['Latitude'], lon=df_mapa['Longitude'],
                mode='markers', marker=dict(size=6, color=df_mapa['Velocity'], colorscale='Viridis', showscale=True,
                                           colorbar=dict(title="Velocidade")),
                text=hover_texts, hoverinfo='text', name='Percurso'
            ))
            
            fig_map.add_trace(go.Scattermapbox(
                lat=[df_mapa['Latitude'].iloc[0]], lon=[df_mapa['Longitude'].iloc[0]],
                mode='markers', marker=dict(size=16, color='green'), name='Início'
            ))
            fig_map.add_trace(go.Scattermapbox(
                lat=[df_mapa['Latitude'].iloc[-1]], lon=[df_mapa['Longitude'].iloc[-1]],
                mode='markers', marker=dict(size=16, color='red'), name='Fim'
            ))
            
            zoom = 18 if bounds_estadio else 15
            fig_map.update_layout(
                mapbox=dict(style="open-street-map", center=dict(lat=center_lat, lon=center_lon), zoom=zoom),
                height=700, margin=dict(l=0, r=0, t=30, b=0),
                title=f"Trajetória de {atleta_mapa_nome} - {nome_estadio}"
            )
            st.plotly_chart(fig_map, use_container_width=True)
        
        # TAB 2: ANÁLISE TÁTICA (coordenadas do campo)
        with tab2:
            st.subheader("Análise Tática - Posicionamento no Campo")
            st.markdown("O campo é representado em metros com as dimensões oficiais (105m x 68m)")
            
            if len(selected_atletas) > 1:
                atleta_tat = st.selectbox("Atleta", selected_atletas, key="tat_select")
                idx_tat = selected_atletas.index(atleta_tat)
                df_tat = dfs_filtered[idx_tat].copy()
                start_dt_tat = selected_start_datetimes[idx_tat]
                atleta_tat_nome = atleta_tat
            else:
                df_tat = df_main.copy()
                start_dt_tat = start_dt_main
                atleta_tat_nome = atleta_main
            
            # Converter coordenadas GPS para coordenadas do campo
            if bounds_estadio:
                campo_x = []
                campo_y = []
                for _, row in df_tat.iterrows():
                    x, y = converter_gps_para_campo(row['Latitude'], row['Longitude'], bounds_estadio)
                    campo_x.append(x)
                    campo_y.append(y)
                df_tat['campo_x'] = campo_x
                df_tat['campo_y'] = campo_y
                st.success(f"✅ Usando limites do {nome_estadio} para conversão")
            else:
                st.warning("⚠️ Limites do estádio não definidos. Use a detecção automática ou cadastre um estádio.")
                st.stop()
            
            # Configuração da divisão em zonas
            col_lin, col_col = st.columns(2)
            with col_lin:
                num_linhas = st.number_input("Número de linhas (divisão horizontal)", min_value=1, max_value=10, value=5,
                                            help="Divide o campo longitudinalmente (ex: 5 = quintos defensivo, médio, ofensivo)")
            with col_col:
                num_colunas = st.number_input("Número de colunas (divisão vertical)", min_value=1, max_value=10, value=3,
                                            help="Divide o campo transversalmente (ex: 3 = corredores esquerdo, central, direito)")
            
            # Criar bins para as zonas baseado nas dimensões reais do campo
            x_min = -CAMPO_COMPRIMENTO / 2
            x_max = CAMPO_COMPRIMENTO / 2
            y_min = -CAMPO_LARGURA / 2
            y_max = CAMPO_LARGURA / 2
            
            linhas_bins = np.linspace(y_min, y_max, num_linhas + 1)  # Y é latitude (vertical)
            colunas_bins = np.linspace(x_min, x_max, num_colunas + 1)  # X é longitude (horizontal)
            
            # Atribuir zona para cada ponto
            df_tat['Zona_Linha'] = pd.cut(df_tat['campo_y'], bins=linhas_bins, labels=[f'L{i+1}' for i in range(num_linhas)], include_lowest=True)
            df_tat['Zona_Coluna'] = pd.cut(df_tat['campo_x'], bins=colunas_bins, labels=[f'C{i+1}' for i in range(num_colunas)], include_lowest=True)
            df_tat['Zona'] = df_tat['Zona_Linha'].astype(str) + '-' + df_tat['Zona_Coluna'].astype(str)
            
            # Métricas por zona
            st.markdown("### 📊 Demanda Física por Zona")
            
            zona_metrics = df_tat.groupby('Zona', observed=True).agg({
                'Seconds': 'count',
                'Velocity': ['mean', 'max'],
                'HeartRate': ['mean', 'max']
            }).round(2)
            zona_metrics.columns = ['Contagem', 'Vel_Média', 'Vel_Máx', 'FC_Média', 'FC_Máx']
            
            if len(df_tat) > 1:
                sample_rate = df_tat['Seconds'].diff().median()
                zona_metrics['Tempo(s)'] = zona_metrics['Contagem'] * sample_rate
                zona_metrics['Tempo(min)'] = zona_metrics['Tempo(s)'] / 60
            
            if 'Odometer' in df_tat.columns:
                df_tat_sorted = df_tat.sort_values('Seconds')
                df_tat_sorted['Zona_Ant'] = df_tat_sorted['Zona'].shift(1)
                df_tat_sorted['Delta_Odometer'] = df_tat_sorted['Odometer'].diff()
                df_tat_sorted['Dist_Zona'] = df_tat_sorted.apply(
                    lambda r: r['Delta_Odometer'] if r['Zona'] == r['Zona_Ant'] else 0, axis=1
                )
                dist_por_zona = df_tat_sorted.groupby('Zona')['Dist_Zona'].sum().round(0)
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
                'Tempo(s)': '{:.1f}',
                'Tempo(min)': '{:.1f}',
                'Distância(m)': '{:.0f}',
                'Intensidade': '{:.1f}%'
            }), use_container_width=True)
            
            # Visualização do campo
            st.markdown("### 🗺️ Visualização Tática")
            
            viz_type = st.radio(
                "Tipo de visualização",
                options=["Trajetória com cores por zona", "Mapa de calor de tempo", "Mapa de calor de velocidade"],
                horizontal=True
            )
            
            fig_tat = go.Figure()
            
            # Desenhar o campo de futebol completo
            shapes = desenhar_campo_futebol()
            for shape in shapes:
                fig_tat.add_shape(shape)
            
            # Adicionar linhas divisórias das zonas
            for linha in linhas_bins[1:-1]:
                fig_tat.add_shape(
                    type="line",
                    x0=x_min, x1=x_max,
                    y0=linha, y1=linha,
                    line=dict(color="rgba(255,255,255,0.3)", width=1, dash="dash"),
                    layer="above"
                )
            
            for coluna in colunas_bins[1:-1]:
                fig_tat.add_shape(
                    type="line",
                    x0=coluna, x1=coluna,
                    y0=y_min, y1=y_max,
                    line=dict(color="rgba(255,255,255,0.3)", width=1, dash="dash"),
                    layer="above"
                )
            
            # Plotar os pontos
            if viz_type == "Trajetória com cores por zona":
                cores = px.colors.qualitative.Set3
                for i, (zona, group) in enumerate(df_tat.groupby('Zona')):
                    fig_tat.add_trace(go.Scatter(
                        x=group['campo_x'], y=group['campo_y'],
                        mode='markers',
                        name=f'Zona {zona}',
                        marker=dict(size=5, color=cores[i % len(cores)], opacity=0.7),
                        text=[f"Zona {zona}<br>Horário: {seconds_to_time_str(t, start_dt_tat)}<br>Vel: {v:.1f} km/h<br>FC: {fc:.0f} bpm"
                              for t, v, fc in zip(group['Seconds'], group['Velocity'], group['HeartRate'])],
                        hoverinfo='text'
                    ))
            
            elif viz_type == "Mapa de calor de tempo":
                heatmap = np.zeros((num_linhas, num_colunas))
                for i in range(num_linhas):
                    for j in range(num_colunas):
                        zona = f'L{i+1}-C{j+1}'
                        if zona in zona_metrics.index:
                            heatmap[i, j] = zona_metrics.loc[zona, 'Contagem']
                
                fig_tat.add_trace(go.Heatmap(
                    x=colunas_bins,
                    y=linhas_bins,
                    z=heatmap,
                    colorscale='Hot',
                    opacity=0.6,
                    colorbar=dict(title="Tempo gasto<br>(nº registros)"),
                    name="Mapa de calor de tempo"
                ))
                
                fig_tat.add_trace(go.Scatter(
                    x=df_tat['campo_x'], y=df_tat['campo_y'],
                    mode='markers',
                    marker=dict(size=3, color='white', opacity=0.5),
                    name='Trajetória',
                    hoverinfo='skip'
                ))
            
            else:
                heatmap = np.zeros((num_linhas, num_colunas))
                for i in range(num_linhas):
                    for j in range(num_colunas):
                        zona = f'L{i+1}-C{j+1}'
                        if zona in zona_metrics.index:
                            heatmap[i, j] = zona_metrics.loc[zona, 'Vel_Média']
                
                fig_tat.add_trace(go.Heatmap(
                    x=colunas_bins,
                    y=linhas_bins,
                    z=heatmap,
                    colorscale='Viridis',
                    opacity=0.6,
                    colorbar=dict(title="Velocidade média<br>(km/h)"),
                    name="Mapa de calor de velocidade"
                ))
                
                fig_tat.add_trace(go.Scatter(
                    x=df_tat['campo_x'], y=df_tat['campo_y'],
                    mode='markers',
                    marker=dict(size=3, color='white', opacity=0.5),
                    name='Trajetória',
                    hoverinfo='skip'
                ))
            
            # Configurar layout
            fig_tat.update_layout(
                title=f"Análise Tática - {atleta_tat_nome}",
                xaxis_title="Posição no campo (metros) - Comprimento",
                yaxis_title="Posição no campo (metros) - Largura",
                height=700,
                hovermode='closest',
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                xaxis=dict(scaleanchor="y", scaleratio=1, range=[-55, 55]),
                yaxis=dict(range=[-38, 38]),
                plot_bgcolor='rgba(34, 139, 34, 0.2)',
                paper_bgcolor='rgba(0,0,0,0)'
            )
            
            st.plotly_chart(fig_tat, use_container_width=True)
            
            # Top zonas
            st.markdown("### 📈 Comparação de Intensidade entre Zonas")
            
            top_zonas = zona_metrics.nlargest(8, 'Intensidade').reset_index()
            
            if len(top_zonas) > 0:
                fig_bar = px.bar(
                    top_zonas, 
                    x='Zona', 
                    y='Intensidade', 
                    title="Top 8 Zonas com Maior Intensidade",
                    text=top_zonas['Intensidade'].round(1),
                    color='Intensidade',
                    color_continuous_scale='Viridis'
                )
                fig_bar.update_traces(textposition='outside')
                fig_bar.update_layout(height=500)
                st.plotly_chart(fig_bar, use_container_width=True)
            
            # Exportar
            csv_tatico = zona_metrics.reset_index().to_csv(index=False)
            st.download_button(
                "📥 Exportar análise tática (CSV)",
                csv_tatico,
                f"analise_tatica_{atleta_tat_nome}.csv"
            )
        
        st.markdown("---")
        st.download_button("📥 Exportar dados filtrados", df_combined.to_csv(index=False), "dados_filtrados.csv")
        st.markdown(f"**Resumo:** {len(df_combined)} registros | **Atletas:** {', '.join(selected_atletas)}")
    
    else:
        st.error("❌ Nenhum arquivo válido processado.")

else:
    st.markdown("""
    ### 👋 Bem-vindo ao Analisador de Percurso do Atleta!
    
    ### 🚀 Como usar:
    1. **Faça upload** de arquivos CSV na barra lateral
    2. **Selecione o estádio** ou use detecção automática
    3. **Escolha os atletas** para análise
    4. **Ajuste os filtros** de tempo e velocidade
    5. **Explore as 2 abas** de análise
    
    ### ✨ Funcionalidades:
    - 🏟️ **Campo retangular** com dimensões oficiais (105m x 68m)
    - 📐 **Todas as linhas do campo**: meio-campo, círculo central, grandes áreas, pequenas áreas, marcas de pênalti
    - 🗺️ **Mapa interativo** com trajetória do atleta
    - 📊 **Conversão precisa** de coordenadas GPS para campo
    - 📐 **Análise Tática** com divisão do campo em zonas (linhas e colunas)
    
    ---
    **👈 Faça upload para começar!**
    """)