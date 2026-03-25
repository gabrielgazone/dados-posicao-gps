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
    
    cursor.execute("SELECT COUNT(*) FROM estadios")
    count = cursor.fetchone()[0]
    
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

def validar_dimensoes_campo(lat_min, lat_max, lon_min, lon_max):
    """Valida se as dimensões são compatíveis com um campo de futebol"""
    if lat_min >= lat_max or lon_min >= lon_max:
        return False, "Limites inválidos"
    
    lat_center = (lat_min + lat_max) / 2
    largura_m = (lat_max - lat_min) * 111000
    comprimento_m = (lon_max - lon_min) * 111000 * np.cos(np.radians(lat_center))
    
    if comprimento_m < 50 or comprimento_m > 150:
        return False, f"Comprimento ({comprimento_m:.0f}m) fora do esperado (50-150m)"
    if largura_m < 40 or largura_m > 100:
        return False, f"Largura ({largura_m:.0f}m) fora do esperado (40-100m)"
    
    ratio = comprimento_m / largura_m
    if ratio < 1.2 or ratio > 2.0:
        return False, f"Proporção ({ratio:.2f}) fora do esperado (1.2-2.0)"
    
    return True, f"Dimensões válidas: {comprimento_m:.0f}m x {largura_m:.0f}m"

def parse_coordenada_string(coord_str):
    """Converte string de coordenada no formato do Google Maps para float
    Aceita formatos:
    -3.8067363941105574, -38.5226464606229
    -3.8067363941105574,-38.5226464606229
    """
    try:
        # Remove espaços extras
        coord_str = coord_str.strip()
        # Separa por vírgula
        partes = coord_str.split(',')
        if len(partes) >= 2:
            lat = float(partes[0].strip())
            lon = float(partes[1].strip())
            return lat, lon
    except:
        pass
    return None, None

# ==================== FUNÇÕES AUXILIARES ====================

def seconds_to_time_str(seconds, start_datetime):
    """Converte segundos desde o início para horário HH:MM:SS"""
    if start_datetime is None:
        return f"{int(seconds // 3600):02d}:{int((seconds % 3600) // 60):02d}:{int(seconds % 60):02d}"
    target_time = start_datetime + timedelta(seconds=seconds)
    return target_time.strftime("%H:%M:%S")

def extract_athlete_from_line8(content):
    """Extrai o nome do atleta da linha 8 do arquivo
    Exemplo: # Athlete: "L. SASHA";;;;;;;;;;;
    Retorna: L. SASHA
    """
    try:
        lines = content.split('\n')
        if len(lines) >= 8:
            line8 = lines[7]  # linha 8 (índice 7)
            # Procura pelo padrão # Athlete:
            if '# Athlete:' in line8:
                # Procura por texto entre aspas
                match = re.search(r'"([^"]*)"', line8)
                if match:
                    nome = match.group(1).strip()
                    if nome:
                        return nome
                # Se não encontrar aspas, tenta extrair após os dois pontos
                parts = line8.split(':')
                if len(parts) > 1:
                    # Pega a parte após os dois pontos e antes do primeiro ponto e vírgula
                    nome = parts[1].split(';')[0].strip().strip('"')
                    if nome:
                        return nome
        return None
    except Exception:
        return None

@st.cache_data
def load_data(uploaded_file):
    """Carrega e processa os dados do arquivo CSV enviado"""
    try:
        content = uploaded_file.getvalue().decode('utf-8')
        
        # Extrair nome do atleta da linha 8
        atleta = extract_athlete_from_line8(content)
        if atleta is None or atleta == "":
            atleta = "Não identificado"
        
        # Extrair período
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
        
        # Encontrar início dos dados
        data_start = 0
        for i, line in enumerate(lines):
            if not line.startswith('#') and 'Timestamp' in line:
                data_start = i
                break
        
        # Ler dados
        df = pd.read_csv(StringIO(content), skiprows=data_start, sep=';')
        
        # Renomear colunas
        col_names = ['Timestamp', 'Seconds', 'Velocity', 'Acceleration', 'Odometer', 
                     'Latitude', 'Longitude', 'HeartRate', 'PlayerLoad', 
                     'PositionalQuality', 'HDOP', 'Sats']
        
        if len(df.columns) == len(col_names):
            df.columns = col_names
        else:
            for i, col in enumerate(df.columns[:len(col_names)]):
                df.rename(columns={col: col_names[i]}, inplace=True)
        
        # Converter colunas numéricas
        numeric_cols = ['Seconds', 'Velocity', 'Acceleration', 'Odometer', 
                        'Latitude', 'Longitude', 'HeartRate', 'PlayerLoad', 
                        'PositionalQuality', 'HDOP', 'Sats']
        
        for col in numeric_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col].astype(str).str.replace(',', '.'), errors='coerce')
        
        # Processar timestamp
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
        
        # Remover linhas com dados essenciais faltando
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

# Inicializar banco de dados
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

# Carregar estádios cadastrados
df_estadios = carregar_estadios()

# Inicializar variáveis globais
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
            st.sidebar.info(f"📍 {estadio_selecionado['cidade']}, {estadio_selecionado['pais']}")
    
    # INTERFACE DE CADASTRO
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
            st.markdown("### 🗺️ Pontos de referência do campo")
            st.markdown("""
            **Como obter as coordenadas:**
            1. Abra o [Google Maps](https://www.google.com/maps) em outra aba
            2. Localize o estádio e dê zoom máximo no campo
            3. Clique com botão direito nos cantos e selecione "O que há aqui?"
            4. Copie a coordenada (ex: `-3.8067363941105574, -38.5226464606229`)
            5. Cole nos campos abaixo
            """)
            
            # Layout dos 4 cantos
            col_esq, col_dir = st.columns(2)
            
            with col_esq:
                st.markdown("**🏁 Lateral Esquerda**")
                coord_no = st.text_area("Canto Superior Esquerdo (NO)", 
                                        value="-3.806736, -38.524000",
                                        key="coord_no",
                                        height=68)
                coord_so = st.text_area("Canto Inferior Esquerdo (SO)", 
                                        value="-3.808800, -38.524000",
                                        key="coord_so",
                                        height=68)
            
            with col_dir:
                st.markdown("**🏁 Lateral Direita**")
                coord_ne = st.text_area("Canto Superior Direito (NE)", 
                                        value="-3.806736, -38.521500",
                                        key="coord_ne",
                                        height=68)
                coord_se = st.text_area("Canto Inferior Direito (SE)", 
                                        value="-3.808800, -38.521500",
                                        key="coord_se",
                                        height=68)
            
            # Processar coordenadas
            lat_no, lon_no = parse_coordenada_string(coord_no)
            lat_ne, lon_ne = parse_coordenada_string(coord_ne)
            lat_so, lon_so = parse_coordenada_string(coord_so)
            lat_se, lon_se = parse_coordenada_string(coord_se)
            
            # Verificar se todas as coordenadas foram processadas
            if all([lat_no is not None, lat_ne is not None, lat_so is not None, lat_se is not None]):
                lat_min = min(lat_no, lat_ne, lat_so, lat_se)
                lat_max = max(lat_no, lat_ne, lat_so, lat_se)
                lon_min = min(lon_no, lon_ne, lon_so, lon_se)
                lon_max = max(lon_no, lon_ne, lon_so, lon_se)
                centro_lat = (lat_min + lat_max) / 2
                centro_lon = (lon_min + lon_max) / 2
                
                # Visualização
                fig_preview = go.Figure()
                fig_preview.add_shape(type="rect", x0=lon_min, x1=lon_max, y0=lat_min, y1=lat_max,
                                      line=dict(color="green", width=3), fillcolor="rgba(34,139,34,0.2)")
                
                pontos = [(lon_no, lat_no, "NO"), (lon_ne, lat_ne, "NE"), (lon_so, lat_so, "SO"), (lon_se, lat_se, "SE")]
                for lon, lat, nome in pontos:
                    fig_preview.add_trace(go.Scatter(x=[lon], y=[lat], mode='markers+text', text=[nome],
                                                     marker=dict(size=12, color='blue'), textposition="top center"))
                
                fig_preview.add_trace(go.Scatter(x=[centro_lon], y=[centro_lat], mode='markers',
                                                 marker=dict(size=15, color='red', symbol='x'), name='Centro'))
                
                fig_preview.update_layout(title="Visualização do campo", height=400, xaxis_title="Longitude", yaxis_title="Latitude")
                st.plotly_chart(fig_preview, use_container_width=True)
                
                valido, msg = validar_dimensoes_campo(lat_min, lat_max, lon_min, lon_max)
                if valido:
                    st.success(f"✅ {msg}")
                else:
                    st.warning(f"⚠️ {msg}")
                
                st.markdown("---")
                orientacao = st.number_input("Orientação (graus)", value=0.0, step=1.0)
                
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
                            dados = {'centro_lat': centro_lat, 'centro_lon': centro_lon,
                                    'lat_min': lat_min, 'lat_max': lat_max,
                                    'lon_min': lon_min, 'lon_max': lon_max, 'orientacao': orientacao}
                            adicionar_estadio(nome_novo, cidade_nova, pais_novo, dados)
                            st.success(f"✅ Estádio {nome_novo} cadastrado!")
                            import time; time.sleep(1); st.rerun()
                    else:
                        st.error("❌ Nome do estádio é obrigatório!")
            else:
                st.info("👆 Cole as coordenadas dos 4 cantos do campo")
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
        # Seleção de atletas
        st.sidebar.markdown("---")
        st.sidebar.subheader("🎯 Selecionar Atleta(s)")
        
        atleta_options = []
        for atleta, periodo in zip(all_atletas, all_periodos):
            display_name = f"{atleta} - {periodo}" if periodo != "Não identificado" else atleta
            atleta_options.append(display_name)
        
        selected_indices = st.sidebar.multiselect(
            "Escolha os atletas",
            options=range(len(atleta_options)),
            format_func=lambda x: atleta_options[x],
            default=[0] if len(atleta_options) > 0 else []
        )
        
        selected_atletas = [all_atletas[i] for i in selected_indices]
        selected_data = [all_data[i] for i in selected_indices]
        selected_periodos = [all_periodos[i] for i in selected_indices]
        selected_start_datetimes = [all_start_datetimes[i] for i in selected_indices]
        
        reference_datetime = selected_start_datetimes[0] if selected_start_datetimes[0] is not None else None
        
        # Calibração do estádio baseado nos dados do primeiro atleta (se necessário)
        df_calibracao = selected_data[0]
        
        # Se não tiver estádio selecionado e estiver em modo automático, detectar
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
        else:
            # Usar estádio selecionado (já definido acima)
            pass
        
        # Filtros
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
        map_style = st.sidebar.selectbox("Estilo do mapa", ["open-street-map", "carto-positron", "carto-darkmatter"], index=0)
        
        # Filtrar dados
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
        
        # Métricas
        df_main = dfs_filtered[0]
        atleta_main = selected_atletas[0]
        periodo_main = selected_periodos[0]
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
        
        # ==================== ABAS (APENAS 1 E 6) ====================
        tab1, tab6 = st.tabs([
            "🗺️ Mapa do Percurso",
            "📐 Análise Tática por Zonas"
        ])
        
        # TAB 1: MAPA
        with tab1:
            st.subheader("Percurso no Campo")
            
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
            elif show_field:
                # Fallback: desenhar um retângulo baseado nos dados
                fig_map.add_shape(type="rect", 
                                  x0=center_lon - 0.0005, x1=center_lon + 0.0005,
                                  y0=center_lat - 0.0003, y1=center_lat + 0.0003,
                                  line=dict(color="white", width=2), fillcolor="rgba(34,139,34,0.2)")
            
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
                mapbox=dict(style=map_style, center=dict(lat=center_lat, lon=center_lon), zoom=zoom),
                height=700, margin=dict(l=0, r=0, t=30, b=0),
                title=f"Trajetória de {atleta_mapa_nome} - {nome_estadio}"
            )
            st.plotly_chart(fig_map, use_container_width=True)
        
        # TAB 6: ANÁLISE TÁTICA POR ZONAS (COM LIMITES DO ESTÁDIO)
        with tab6:
            st.subheader("Análise Tática Integrada")
            st.markdown("""
            Esta ferramenta permite analisar como o atleta distribui sua **demanda física** (tempo, distância, intensidade) 
            pelas **diferentes zonas do campo**, auxiliando na integração de dados físicos e táticos.
            """)
            
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
            
            col_lin, col_col = st.columns(2)
            with col_lin:
                num_linhas = st.number_input("Número de linhas (divisão horizontal)", min_value=1, max_value=10, value=3,
                                            help="Divide o campo longitudinalmente (ex: 3 = terços defensivo, médio, ofensivo)")
            with col_col:
                num_colunas = st.number_input("Número de colunas (divisão vertical)", min_value=1, max_value=10, value=3,
                                            help="Divide o campo transversalmente (ex: 3 = corredores esquerdo, central, direito)")
            
            # Usar os limites do estádio se disponíveis
            if bounds_estadio:
                lat_min, lat_max, lon_min, lon_max = bounds_estadio
                st.info(f"✅ Usando limites do {nome_estadio} para zoneamento")
                st.info(f"   Latitude: {lat_min:.6f} → {lat_max:.6f}")
                st.info(f"   Longitude: {lon_min:.6f} → {lon_max:.6f}")
            else:
                lat_min = df_tat['Latitude'].min()
                lat_max = df_tat['Latitude'].max()
                lon_min = df_tat['Longitude'].min()
                lon_max = df_tat['Longitude'].max()
                st.info("📊 Usando limites baseados nos dados do atleta")
            
            # Criar bins para as zonas
            linhas_bins = np.linspace(lat_min, lat_max, num_linhas + 1)
            colunas_bins = np.linspace(lon_min, lon_max, num_colunas + 1)
            
            # Atribuir zona para cada ponto
            df_tat['Zona_Linha'] = pd.cut(df_tat['Latitude'], bins=linhas_bins, labels=[f'L{i+1}' for i in range(num_linhas)], include_lowest=True)
            df_tat['Zona_Coluna'] = pd.cut(df_tat['Longitude'], bins=colunas_bins, labels=[f'C{i+1}' for i in range(num_colunas)], include_lowest=True)
            df_tat['Zona'] = df_tat['Zona_Linha'].astype(str) + '-' + df_tat['Zona_Coluna'].astype(str)
            
            # Calcular métricas por zona
            st.markdown("### 📊 Demanda Física por Zona")
            
            zona_metrics = df_tat.groupby('Zona', observed=True).agg({
                'Seconds': 'count',
                'Velocity': ['mean', 'max'],
                'HeartRate': ['mean', 'max']
            }).round(2)
            zona_metrics.columns = ['Contagem', 'Vel_Média', 'Vel_Máx', 'FC_Média', 'FC_Máx']
            
            # Calcular tempo total
            if len(df_tat) > 1:
                sample_rate = df_tat['Seconds'].diff().median()
                zona_metrics['Tempo(s)'] = zona_metrics['Contagem'] * sample_rate
                zona_metrics['Tempo(min)'] = zona_metrics['Tempo(s)'] / 60
            else:
                zona_metrics['Tempo(s)'] = 0
                zona_metrics['Tempo(min)'] = 0
            
            # Calcular distância percorrida por zona
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
            
            # Calcular intensidade
            zona_metrics['Intensidade'] = (zona_metrics['Vel_Média'] * zona_metrics['Contagem']) / zona_metrics['Contagem'].sum() * 100
            
            # Exibir tabela
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
            
            # Visualização do campo com zonas
            st.markdown("### 🗺️ Visualização Tática")
            
            viz_type = st.radio(
                "Tipo de visualização",
                options=["Trajetória com cores por zona", "Mapa de calor de tempo", "Mapa de calor de velocidade"],
                horizontal=True,
                key="tatico_viz_type"
            )
            
            fig_tat = go.Figure()
            
            # Adicionar retângulo do campo
            fig_tat.add_shape(
                type="rect",
                x0=lon_min, x1=lon_max,
                y0=lat_min, y1=lat_max,
                line=dict(color="white", width=2),
                fillcolor="rgba(34, 139, 34, 0.2)",
                layer="below"
            )
            
            # Adicionar linhas divisórias
            for linha in linhas_bins[1:-1]:
                fig_tat.add_shape(
                    type="line",
                    x0=lon_min, x1=lon_max,
                    y0=linha, y1=linha,
                    line=dict(color="rgba(255,255,255,0.5)", width=1, dash="dash"),
                    layer="below"
                )
            
            for coluna in colunas_bins[1:-1]:
                fig_tat.add_shape(
                    type="line",
                    x0=coluna, x1=coluna,
                    y0=lat_min, y1=lat_max,
                    line=dict(color="rgba(255,255,255,0.5)", width=1, dash="dash"),
                    layer="below"
                )
            
            if viz_type == "Trajetória com cores por zona":
                # Criar mapa de cores para as zonas
                zonas_unicas = df_tat['Zona'].unique()
                cores = px.colors.qualitative.Set3
                cor_por_zona = {zona: cores[i % len(cores)] for i, zona in enumerate(zonas_unicas)}
                
                for zona, group in df_tat.groupby('Zona'):
                    fig_tat.add_trace(go.Scatter(
                        x=group['Longitude'],
                        y=group['Latitude'],
                        mode='markers',
                        name=f'Zona {zona}',
                        marker=dict(size=4, opacity=0.7, color=cor_por_zona[zona]),
                        text=[f"<b>Zona:</b> {zona}<br><b>Horário:</b> {seconds_to_time_str(t, start_dt_tat)}<br><b>Vel:</b> {v:.1f} km/h<br><b>FC:</b> {fc:.0f} bpm"
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
                
                fig_tat.add_trace(go.Heatmap(
                    x=colunas_bins,
                    y=linhas_bins,
                    z=heatmap_data.T,
                    colorscale='Hot',
                    opacity=0.7,
                    colorbar=dict(title="Tempo gasto<br>(nº registros)"),
                    name="Mapa de calor de tempo"
                ))
                
                fig_tat.add_trace(go.Scatter(
                    x=df_tat['Longitude'],
                    y=df_tat['Latitude'],
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
                
                fig_tat.add_trace(go.Heatmap(
                    x=colunas_bins,
                    y=linhas_bins,
                    z=heatmap_data_vel.T,
                    colorscale='Viridis',
                    opacity=0.7,
                    colorbar=dict(title="Velocidade média<br>(km/h)"),
                    name="Mapa de calor de velocidade"
                ))
                
                fig_tat.add_trace(go.Scatter(
                    x=df_tat['Longitude'],
                    y=df_tat['Latitude'],
                    mode='markers',
                    marker=dict(size=3, color='white', opacity=0.5),
                    name='Trajetória',
                    hoverinfo='skip'
                ))
            
            fig_tat.update_layout(
                title=f"Análise Tática - {atleta_tat_nome}",
                xaxis_title="Longitude",
                yaxis_title="Latitude",
                height=700,
                hovermode='closest',
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
            )
            
            fig_tat.update_yaxes(scaleanchor="x", scaleratio=1)
            st.plotly_chart(fig_tat, use_container_width=True)
            
            # Gráfico de barras das zonas com maior intensidade
            st.markdown("### 📈 Comparação de Intensidade entre Zonas")
            
            top_zonas = zona_metrics.nlargest(8, 'Intensidade').reset_index()
            
            if len(top_zonas) > 0:
                fig_bar = go.Figure()
                fig_bar.add_trace(go.Bar(
                    x=top_zonas['Zona'],
                    y=top_zonas['Intensidade'],
                    marker_color='rgba(255, 99, 71, 0.7)',
                    name='Intensidade Relativa (%)',
                    text=top_zonas['Intensidade'].round(1),
                    textposition='auto'
                ))
                
                fig_bar.update_layout(
                    title="Top 8 Zonas com Maior Intensidade",
                    xaxis_title="Zona",
                    yaxis_title="Intensidade Relativa (%)",
                    height=500,
                    xaxis_tickangle=-45
                )
                
                st.plotly_chart(fig_bar, use_container_width=True)
            
            # Exportar dados
            csv_tatico = zona_metrics.reset_index().to_csv(index=False)
            st.download_button(
                label="📥 Exportar análise tática (CSV)",
                data=csv_tatico,
                file_name=f"analise_tatica_{atleta_tat_nome}.csv",
                mime="text/csv"
            )
        
        # Download dados filtrados
        st.markdown("---")
        st.download_button(
            label="📥 Exportar dados filtrados (CSV)",
            data=df_combined.to_csv(index=False),
            file_name=f"dados_filtrados.csv",
            mime="text/csv"
        )
        
        st.markdown(f"**📊 Resumo:** {len(df_combined)} registros | **Atletas:** {', '.join(selected_atletas)}")
        st.markdown(f"**🏟️ Estádio:** {nome_estadio}")
    
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
    - 🏟️ **Cadastro de estádios** - Cole coordenadas do Google Maps (formato: `-3.806736, -38.522646`)
    - 🗺️ **Mapa interativo** com trajetória do atleta
    - 📐 **Análise Tática** com divisão do campo em zonas (linhas e colunas)
    - 📊 **Métricas por zona**: tempo, distância, velocidade, FC e intensidade
    - 🎯 **Mapas de calor** de tempo e velocidade
    
    ---
    **👈 Faça upload para começar!**
    """)