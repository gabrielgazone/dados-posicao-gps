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
    conn = sqlite3.connect(DB_PATH)
    df = pd.read_sql_query("SELECT id, nome, cidade, pais FROM estadios ORDER BY nome", conn)
    conn.close()
    return df

def obter_estadio(id_estadio):
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
    if lat_min >= lat_max or lon_min >= lon_max:
        return False, "Limites inválidos"
    
    lat_center = (lat_min + lat_max) / 2
    largura_m = (lat_max - lat_min) * 111000
    comprimento_m = (lon_max - lon_min) * 111000 * np.cos(np.radians(lat_center))
    
    if comprimento_m < 50 or comprimento_m > 150:
        return False, f"Comprimento ({comprimento_m:.0f}m) fora do esperado"
    if largura_m < 40 or largura_m > 100:
        return False, f"Largura ({largura_m:.0f}m) fora do esperado"
    
    ratio = comprimento_m / largura_m
    if ratio < 1.2 or ratio > 2.0:
        return False, f"Proporção ({ratio:.2f}) fora do esperado"
    
    return True, f"Dimensões válidas: {comprimento_m:.0f}m x {largura_m:.0f}m"

def parse_coordenada_string(coord_str):
    """Converte string de coordenada no formato do Google Maps para float"""
    try:
        # Remove aspas, espaços e parênteses
        coord_str = coord_str.strip().strip('"').strip("'").strip('()')
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
    if start_datetime is None:
        return f"{int(seconds // 3600):02d}:{int((seconds % 3600) // 60):02d}:{int(seconds % 60):02d}"
    target_time = start_datetime + timedelta(seconds=seconds)
    return target_time.strftime("%H:%M:%S")

def extract_athlete_from_line8(content):
    """Extrai o nome do atleta da linha 8 do arquivo - texto entre aspas após # Athlete:"""
    try:
        lines = content.split('\n')
        if len(lines) >= 8:
            line8 = lines[7]  # linha 8 (índice 7)
            # Procura pelo padrão # Athlete: "NOME"
            if '# Athlete:' in line8:
                # Procura por texto entre aspas
                match = re.search(r'"([^"]*)"', line8)
                if match:
                    return match.group(1).strip()
                # Se não encontrar aspas, tenta extrair após os dois pontos até o ponto e vírgula
                parts = line8.split(':')
                if len(parts) > 1:
                    nome = parts[1].split(';')[0].strip().strip('"')
                    if nome:
                        return nome
        return None
    except Exception:
        return None

@st.cache_data
def load_data(uploaded_file):
    try:
        content = uploaded_file.getvalue().decode('utf-8')
        
        atleta = extract_athlete_from_line8(content)
        if atleta is None:
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

if len(df_estadios) > 0:
    opcoes_estadio = ["Detectar automaticamente"] + df_estadios['nome'].tolist() + ["Cadastrar novo estádio"]
    selecao_estadio = st.sidebar.selectbox(
        "Selecione o estádio ou modo de detecção",
        options=opcoes_estadio,
        index=0
    )
    
    estadio_selecionado = None
    if selecao_estadio != "Detectar automaticamente" and selecao_estadio != "Cadastrar novo estádio":
        idx_estadio = df_estadios[df_estadios['nome'] == selecao_estadio].index[0]
        estadio_selecionado = obter_estadio(df_estadios.loc[idx_estadio, 'id'])
        if estadio_selecionado:
            st.sidebar.success(f"✅ {estadio_selecionado['nome']}")
    
    # INTERFACE DE CADASTRO MELHORADA COM ENTRADA SIMPLIFICADA
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
            4. Copie a coordenada que aparece (ex: `-3.8067363941105574, -38.5226464606229`)
            5. Cole nos campos abaixo - o sistema vai extrair latitude e longitude automaticamente
            """)
            
            # Layout dos 4 cantos com entrada simplificada
            col_esq, col_dir = st.columns(2)
            
            with col_esq:
                st.markdown("**🏁 Lateral Esquerda**")
                
                with st.expander("📍 Canto Superior Esquerdo (NO)", expanded=True):
                    coord_no = st.text_input("Coordenada (cole do Google Maps)", 
                                            value="-3.8067363941105574, -38.5226464606229",
                                            key="coord_no",
                                            help="Exemplo: -3.8067363941105574, -38.5226464606229")
                
                with st.expander("📍 Canto Inferior Esquerdo (SO)", expanded=True):
                    coord_so = st.text_input("Coordenada (cole do Google Maps)", 
                                            value="-3.8076924786839124, -38.52277184997347",
                                            key="coord_so")
            
            with col_dir:
                st.markdown("**🏁 Lateral Direita**")
                
                with st.expander("📍 Canto Superior Direito (NE)", expanded=True):
                    coord_ne = st.text_input("Coordenada (cole do Google Maps)", 
                                            value="-3.8067363941105574, -38.5226464606229",
                                            key="coord_ne")
                
                with st.expander("📍 Canto Inferior Direito (SE)", expanded=True):
                    coord_se = st.text_input("Coordenada (cole do Google Maps)", 
                                            value="-3.8076924786839124, -38.52277184997347",
                                            key="coord_se")
            
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
                st.info("👆 Cole as coordenadas dos 4 cantos do campo (formato: latitude, longitude)")
else:
    st.sidebar.warning("Nenhum estádio cadastrado")
    selecao_estadio = "Detectar automaticamente"
    estadio_selecionado = None

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
        
        # Calibração do estádio
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
        else:
            bounds_estadio = estadio_selecionado['bounds']
            centro_estadio = estadio_selecionado['centro']
            nome_estadio = estadio_selecionado['nome']
        
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
        
        # ==================== ABAS (SEM ENTROPIA) ====================
        tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
            "🗺️ Mapa", "📈 Desempenho", "⚡ Velocidade", "❤️ FC", "🔄 Aceleração", "📐 Tática"
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
            
            if estadio_selecionado:
                center_lat, center_lon = centro_estadio
            else:
                center_lat = df_mapa['Latitude'].mean()
                center_lon = df_mapa['Longitude'].mean()
            
            fig_map = go.Figure()
            
            if show_field and estadio_selecionado:
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
            
            zoom = 18 if estadio_selecionado else 15
            fig_map.update_layout(
                mapbox=dict(style=map_style, center=dict(lat=center_lat, lon=center_lon), zoom=zoom),
                height=700, margin=dict(l=0, r=0, t=30, b=0),
                title=f"Trajetória de {atleta_mapa_nome} - {nome_estadio}"
            )
            st.plotly_chart(fig_map, use_container_width=True)
        
        # TAB 2: GRÁFICOS DE DESEMPENHO
        with tab2:
            st.subheader("Velocidade e Frequência Cardíaca")
            
            if len(selected_atletas) > 1:
                atleta_temp = st.selectbox("Atleta", selected_atletas, key="temp_select")
                idx_temp = selected_atletas.index(atleta_temp)
                df_temp = dfs_filtered[idx_temp].copy()
                start_dt_temp = selected_start_datetimes[idx_temp]
                atleta_temp_nome = atleta_temp
            else:
                df_temp = df_main.copy()
                start_dt_temp = start_dt_main
                atleta_temp_nome = atleta_main
            
            df_temp['Horario'] = df_temp['Seconds'].apply(lambda x: seconds_to_time_str(x, start_dt_temp))
            
            fig_temp = make_subplots(specs=[[{"secondary_y": True}]])
            fig_temp.add_trace(go.Scatter(x=df_temp['Horario'], y=df_temp['Velocity'], mode='lines', name='Velocidade',
                                          line=dict(color='#3498db', width=2), fill='tozeroy'), secondary_y=False)
            fig_temp.add_trace(go.Scatter(x=df_temp['Horario'], y=df_temp['HeartRate'], mode='lines', name='FC',
                                          line=dict(color='#e74c3c', width=2), fill='tozeroy'), secondary_y=True)
            
            vel_mean = df_temp['Velocity'].mean()
            fig_temp.add_hline(y=vel_mean, line_dash="dash", line_color="#3498db",
                               annotation_text=f"Vel Média: {vel_mean:.1f} km/h", secondary_y=False)
            hr_mean = df_temp['HeartRate'].mean()
            fig_temp.add_hline(y=hr_mean, line_dash="dash", line_color="#e74c3c",
                               annotation_text=f"FC Média: {hr_mean:.0f} bpm", secondary_y=True)
            
            fc_max = df_temp['HeartRate'].max()
            fig_temp.add_hrect(y0=0, y1=fc_max*0.6, fillcolor="lightgreen", opacity=0.2, line_width=0, secondary_y=True)
            fig_temp.add_hrect(y0=fc_max*0.6, y1=fc_max*0.75, fillcolor="yellow", opacity=0.2, line_width=0, secondary_y=True)
            fig_temp.add_hrect(y0=fc_max*0.75, y1=fc_max*0.9, fillcolor="orange", opacity=0.2, line_width=0, secondary_y=True)
            fig_temp.add_hrect(y0=fc_max*0.9, y1=fc_max, fillcolor="red", opacity=0.2, line_width=0, secondary_y=True)
            
            fig_temp.update_layout(title=f"{atleta_temp_nome}", xaxis_title="Horário", height=500, hovermode='x unified')
            fig_temp.update_yaxes(title_text="Velocidade (km/h)", secondary_y=False)
            fig_temp.update_yaxes(title_text="FC (bpm)", secondary_y=True)
            st.plotly_chart(fig_temp, use_container_width=True)
            
            # Gráfico de dispersão Velocidade vs FC
            st.markdown("### 🎯 Relação Velocidade vs FC")
            fig_scatter = px.scatter(df_temp, x='Velocity', y='HeartRate', color='Seconds',
                                     color_continuous_scale='Viridis', title="Velocidade vs Frequência Cardíaca")
            st.plotly_chart(fig_scatter, use_container_width=True)
            
            # Comparação entre atletas
            if len(selected_atletas) > 1:
                st.markdown("### 👥 Comparação entre Atletas")
                fig_comp = go.Figure()
                for df, atleta, start_dt in zip(dfs_filtered, selected_atletas, selected_start_datetimes):
                    df_comp = df.copy()
                    df_comp['Horario'] = df_comp['Seconds'].apply(lambda x: seconds_to_time_str(x, start_dt))
                    fig_comp.add_trace(go.Scatter(x=df_comp['Horario'], y=df_comp['Velocity'], mode='lines', name=atleta))
                fig_comp.update_layout(title="Comparação de Velocidade", xaxis_title="Horário", height=450)
                st.plotly_chart(fig_comp, use_container_width=True)
        
        # TAB 3: VELOCIDADE E ACELERAÇÃO
        with tab3:
            st.subheader("Análise de Velocidade e Aceleração")
            
            if len(selected_atletas) > 1:
                atleta_vel = st.selectbox("Atleta", selected_atletas, key="vel_select")
                idx_vel = selected_atletas.index(atleta_vel)
                df_vel = dfs_filtered[idx_vel].copy()
            else:
                df_vel = df_main.copy()
            
            col1, col2 = st.columns(2)
            with col1:
                fig_hist = px.histogram(df_vel, x='Velocity', nbins=40, title="Distribuição de Velocidades")
                fig_hist.add_vline(x=df_vel['Velocity'].mean(), line_dash="dash", line_color="red",
                                   annotation_text=f"Média: {df_vel['Velocity'].mean():.1f} km/h")
                st.plotly_chart(fig_hist, use_container_width=True)
            
            with col2:
                df_vel['Percentil'] = pd.cut(df_vel['Seconds'], bins=10, labels=[f'{i*10}-{(i+1)*10}%' for i in range(10)])
                fig_box = px.box(df_vel, x='Percentil', y='Velocity', title="Velocidade por Percentil do Tempo")
                st.plotly_chart(fig_box, use_container_width=True)
            
            if 'Acceleration' in df_vel.columns:
                st.subheader("Aceleração ao longo do tempo")
                fig_acc = go.Figure()
                fig_acc.add_trace(go.Scatter(x=df_vel['Seconds'], y=df_vel['Acceleration'], mode='lines', name='Aceleração'))
                fig_acc.add_hline(y=0, line_dash="dash", line_color="black", annotation_text="Repouso")
                fig_acc.update_layout(xaxis_title="Tempo (s)", yaxis_title="Aceleração (m/s²)", height=450)
                st.plotly_chart(fig_acc, use_container_width=True)
        
        # TAB 4: FREQUÊNCIA CARDÍACA
        with tab4:
            st.subheader("Análise da Frequência Cardíaca")
            
            if len(selected_atletas) > 1:
                atleta_fc = st.selectbox("Atleta", selected_atletas, key="fc_select")
                idx_fc = selected_atletas.index(atleta_fc)
                df_fc = dfs_filtered[idx_fc].copy()
            else:
                df_fc = df_main.copy()
            
            col1, col2 = st.columns(2)
            with col1:
                fig_hr_hist = px.histogram(df_fc, x='HeartRate', nbins=30, title="Distribuição da FC")
                fig_hr_hist.add_vline(x=df_fc['HeartRate'].mean(), line_dash="dash", line_color="blue",
                                      annotation_text=f"Média: {df_fc['HeartRate'].mean():.0f} bpm")
                st.plotly_chart(fig_hr_hist, use_container_width=True)
            
            with col2:
                df_fc['Horario'] = df_fc['Seconds'].apply(lambda x: seconds_to_time_str(x, reference_datetime))
                fig_hr_time = go.Figure()
                fig_hr_time.add_trace(go.Scatter(x=df_fc['Horario'], y=df_fc['HeartRate'], mode='lines', name='FC',
                                                 line=dict(color='red', width=2)))
                fc_max = df_fc['HeartRate'].max()
                fig_hr_time.add_hrect(y0=0, y1=fc_max*0.6, fillcolor="lightgreen", opacity=0.3, annotation_text="Recuperação")
                fig_hr_time.add_hrect(y0=fc_max*0.6, y1=fc_max*0.75, fillcolor="yellow", opacity=0.3, annotation_text="Aeróbica")
                fig_hr_time.add_hrect(y0=fc_max*0.75, y1=fc_max*0.9, fillcolor="orange", opacity=0.3, annotation_text="Anaeróbica")
                fig_hr_time.add_hrect(y0=fc_max*0.9, y1=fc_max, fillcolor="red", opacity=0.3, annotation_text="Máximo")
                fig_hr_time.update_layout(xaxis_title="Horário", yaxis_title="FC (bpm)", height=450)
                st.plotly_chart(fig_hr_time, use_container_width=True)
            
            # Zonas de intensidade
            st.markdown("### 📊 Análise por Zona de Intensidade")
            fc_max = df_fc['HeartRate'].max()
            df_fc['Zona'] = pd.cut(df_fc['HeartRate'], bins=[0, fc_max*0.6, fc_max*0.75, fc_max*0.9, fc_max],
                                   labels=['Recuperação', 'Aeróbica', 'Anaeróbica', 'Máximo'])
            zona_stats = df_fc.groupby('Zona', observed=True).agg({'HeartRate': 'count', 'Velocity': 'mean'}).round(0)
            zona_stats.columns = ['Tempo (samples)', 'Vel Média']
            zona_stats['% Tempo'] = (zona_stats['Tempo (samples)'] / len(df_fc) * 100).round(1).astype(str) + '%'
            st.dataframe(zona_stats, use_container_width=True)
        
        # TAB 5: ACELERAÇÃO VS VELOCIDADE COM QUADRANTES
        with tab5:
            st.subheader("Relação Aceleração vs Velocidade")
            
            if len(selected_atletas) > 1:
                atleta_acc = st.selectbox("Atleta", selected_atletas, key="acc_select")
                idx_acc = selected_atletas.index(atleta_acc)
                df_acc = dfs_filtered[idx_acc].copy()
            else:
                df_acc = df_main.copy()
            
            if 'Acceleration' in df_acc.columns:
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
                
                cores = {'Q1 - Alta Vel + Alta Acel': '#e74c3c', 'Q2 - Baixa Vel + Alta Acel': '#f39c12',
                         'Q3 - Baixa Vel + Baixa Acel': '#2ecc71', 'Q4 - Alta Vel + Baixa Acel': '#3498db'}
                
                fig_acc = go.Figure()
                for quad, cor in cores.items():
                    mask = df_acc['Quadrante'] == quad
                    fig_acc.add_trace(go.Scatter(x=df_acc[mask]['Velocity'], y=df_acc[mask]['Acceleration'],
                                                 mode='markers', name=quad, marker=dict(size=8, color=cor, opacity=0.6)))
                
                fig_acc.add_vline(x=mean_vel, line_dash="dash", line_color="gray",
                                  annotation_text=f"Vel Média: {mean_vel:.1f}")
                fig_acc.add_hline(y=mean_acc, line_dash="dash", line_color="gray",
                                  annotation_text=f"Acel Média: {mean_acc:.2f}")
                fig_acc.update_layout(title="Aceleração vs Velocidade", xaxis_title="Velocidade (km/h)",
                                      yaxis_title="Aceleração (m/s²)", height=600)
                st.plotly_chart(fig_acc, use_container_width=True)
                
                # Estatísticas por quadrante
                quad_stats = df_acc.groupby('Quadrante').agg({'Velocity': ['count', 'mean'], 'Acceleration': 'mean'}).round(2)
                quad_stats.columns = ['Contagem', 'Vel Média', 'Acel Média']
                quad_stats['% Tempo'] = (quad_stats['Contagem'] / len(df_acc) * 100).round(1).astype(str) + '%'
                st.dataframe(quad_stats, use_container_width=True)
                
                # Gráfico de pizza
                fig_pie = px.pie(quad_stats, values='Contagem', names=quad_stats.index, title="Distribuição por Quadrante")
                st.plotly_chart(fig_pie, use_container_width=True)
        
        # TAB 6: ANÁLISE TÁTICA POR ZONAS
        with tab6:
            st.subheader("Análise Tática Integrada")
            
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
                num_linhas = st.number_input("Linhas (horizontal)", 1, 10, 3)
            with col_col:
                num_colunas = st.number_input("Colunas (vertical)", 1, 10, 3)
            
            if estadio_selecionado:
                lat_min, lat_max, lon_min, lon_max = bounds_estadio
            else:
                lat_min, lat_max = df_tat['Latitude'].min(), df_tat['Latitude'].max()
                lon_min, lon_max = df_tat['Longitude'].min(), df_tat['Longitude'].max()
            
            linhas_bins = np.linspace(lat_min, lat_max, num_linhas + 1)
            colunas_bins = np.linspace(lon_min, lon_max, num_colunas + 1)
            
            df_tat['Zona'] = (pd.cut(df_tat['Latitude'], bins=linhas_bins, labels=[f'L{i+1}' for i in range(num_linhas)]).astype(str) + '-' +
                              pd.cut(df_tat['Longitude'], bins=colunas_bins, labels=[f'C{i+1}' for i in range(num_colunas)]).astype(str))
            
            # Métricas por zona
            zona_metrics = df_tat.groupby('Zona').agg({
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
                df_tat_sorted['Dist'] = df_tat_sorted['Odometer'].diff()
                df_tat_sorted['Dist_Zona'] = df_tat_sorted.apply(lambda r: r['Dist'] if r['Zona'] == r['Zona_Ant'] else 0, axis=1)
                dist_por_zona = df_tat_sorted.groupby('Zona')['Dist_Zona'].sum().round(0)
                zona_metrics['Distância(m)'] = dist_por_zona
            
            zona_metrics['Intensidade'] = (zona_metrics['Vel_Média'] * zona_metrics['Contagem']) / zona_metrics['Contagem'].sum() * 100
            st.dataframe(zona_metrics.style.format({
                'Contagem': '{:.0f}', 'Vel_Média': '{:.1f}', 'Vel_Máx': '{:.1f}',
                'FC_Média': '{:.0f}', 'FC_Máx': '{:.0f}', 'Tempo(s)': '{:.1f}',
                'Tempo(min)': '{:.1f}', 'Distância(m)': '{:.0f}', 'Intensidade': '{:.1f}%'
            }), use_container_width=True)
            
            # Visualização
            viz_type = st.radio("Visualização", ["Trajetória por zona", "Mapa de calor tempo", "Mapa de calor velocidade"], horizontal=True)
            
            fig_tat = go.Figure()
            fig_tat.add_shape(type="rect", x0=lon_min, x1=lon_max, y0=lat_min, y1=lat_max,
                              line=dict(color="white", width=2), fillcolor="rgba(34,139,34,0.2)")
            
            for linha in linhas_bins[1:-1]:
                fig_tat.add_shape(type="line", x0=lon_min, x1=lon_max, y0=linha, y1=linha,
                                  line=dict(color="white", width=1, dash="dash"))
            for coluna in colunas_bins[1:-1]:
                fig_tat.add_shape(type="line", x0=coluna, x1=coluna, y0=lat_min, y1=lat_max,
                                  line=dict(color="white", width=1, dash="dash"))
            
            if viz_type == "Trajetória por zona":
                cores = px.colors.qualitative.Set3
                for i, (zona, group) in enumerate(df_tat.groupby('Zona')):
                    fig_tat.add_trace(go.Scatter(x=group['Longitude'], y=group['Latitude'], mode='markers',
                                                 name=f'Zona {zona}', marker=dict(size=4, color=cores[i % len(cores)], opacity=0.7),
                                                 text=[f"Zona {zona}<br>Horário: {seconds_to_time_str(t, start_dt_tat)}<br>Vel: {v:.1f}<br>FC: {fc:.0f}"
                                                       for t, v, fc in zip(group['Seconds'], group['Velocity'], group['HeartRate'])],
                                                 hoverinfo='text'))
            
            elif viz_type == "Mapa de calor tempo":
                heatmap = np.zeros((num_linhas, num_colunas))
                for i in range(num_linhas):
                    for j in range(num_colunas):
                        zona = f'L{i+1}-C{j+1}'
                        if zona in zona_metrics.index:
                            heatmap[i, j] = zona_metrics.loc[zona, 'Contagem']
                fig_tat.add_trace(go.Heatmap(x=colunas_bins, y=linhas_bins, z=heatmap.T,
                                             colorscale='Hot', opacity=0.7, colorbar=dict(title="Tempo")))
                fig_tat.add_trace(go.Scatter(x=df_tat['Longitude'], y=df_tat['Latitude'],
                                             mode='markers', marker=dict(size=3, color='white', opacity=0.5), name='Trajetória'))
            
            else:
                heatmap = np.zeros((num_linhas, num_colunas))
                for i in range(num_linhas):
                    for j in range(num_colunas):
                        zona = f'L{i+1}-C{j+1}'
                        if zona in zona_metrics.index:
                            heatmap[i, j] = zona_metrics.loc[zona, 'Vel_Média']
                fig_tat.add_trace(go.Heatmap(x=colunas_bins, y=linhas_bins, z=heatmap.T,
                                             colorscale='Viridis', opacity=0.7, colorbar=dict(title="Velocidade Média")))
                fig_tat.add_trace(go.Scatter(x=df_tat['Longitude'], y=df_tat['Latitude'],
                                             mode='markers', marker=dict(size=3, color='white', opacity=0.5), name='Trajetória'))
            
            fig_tat.update_layout(title=f"Análise Tática - {atleta_tat_nome}", xaxis_title="Longitude", yaxis_title="Latitude", height=700)
            st.plotly_chart(fig_tat, use_container_width=True)
            
            # Top zonas
            top_zonas = zona_metrics.nlargest(8, 'Intensidade').reset_index()
            if len(top_zonas) > 0:
                fig_bar = px.bar(top_zonas, x='Zona', y='Intensidade', title="Top 8 Zonas por Intensidade",
                                 text=top_zonas['Intensidade'].round(1))
                fig_bar.update_traces(textposition='outside')
                st.plotly_chart(fig_bar, use_container_width=True)
            
            # Exportar
            csv_tat = zona_metrics.reset_index().to_csv(index=False)
            st.download_button("📥 Exportar análise tática", csv_tat, f"analise_tatica_{atleta_tat_nome}.csv")
        
        # Download dados filtrados
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
    5. **Explore as 6 abas** de análise
    
    ### ✨ Funcionalidades:
    - 🏟️ **Cadastro de estádios** por pontos de referência (cole coordenadas do Google Maps)
    - 🗺️ **Mapa interativo** com trajetória
    - 📈 **Gráficos sobrepostos** (Velocidade + FC)
    - ⚡ **Distribuição de velocidades** e boxplot por percentil
    - ❤️ **Zonas de frequência cardíaca**
    - 🔄 **Quadrantes de aceleração vs velocidade**
    - 📐 **Análise tática** com divisão do campo em zonas
    
    ---
    **👈 Faça upload para começar!**
    """)