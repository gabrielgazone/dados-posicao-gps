import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import numpy as np
from io import StringIO, BytesIO
import warnings
import re
from datetime import datetime, timedelta
import sqlite3
import os
from pathlib import Path
import json
import requests
from scipy.optimize import curve_fit
from scipy import stats
import matplotlib.pyplot as plt
from sklearn.linear_model import RANSACRegressor
from sklearn.preprocessing import StandardScaler
from reportlab.lib import colors
from reportlab.lib.pagesizes import letter, A4
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.lib.enums import TA_CENTER
import base64
import tempfile
warnings.filterwarnings('ignore')

# Configuração da página
st.set_page_config(
    page_title="⚽ ScoutLab - Análise de Performance Esportiva",
    page_icon="⚽",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS personalizado - VISUAL INOVADOR
st.markdown("""
<style>
    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(20px); }
        to { opacity: 1; transform: translateY(0); }
    }
    @keyframes glow {
        0% { box-shadow: 0 0 5px rgba(0, 210, 255, 0.3); }
        50% { box-shadow: 0 0 20px rgba(0, 210, 255, 0.6); }
        100% { box-shadow: 0 0 5px rgba(0, 210, 255, 0.3); }
    }
    @keyframes slideIn {
        from { opacity: 0; transform: translateX(-20px); }
        to { opacity: 1; transform: translateX(0); }
    }
    .stApp {
        background: linear-gradient(135deg, #0a0a1a 0%, #0f0f2a 50%, #1a1a3a 100%);
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
        transform: translateY(-5px) scale(1.02);
        box-shadow: 0 12px 40px rgba(0, 210, 255, 0.3);
        border-color: rgba(0, 210, 255, 0.8);
        animation: glow 1.5s infinite;
    }
    .metric-value {
        font-size: 2.5rem;
        font-weight: bold;
        background: linear-gradient(135deg, #00d2ff 0%, #3a7bd5 50%, #00ff88 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        text-shadow: 0 0 20px rgba(0, 210, 255, 0.3);
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
        background: rgba(0, 0, 0, 0.5);
        padding: 10px;
        border-radius: 50px;
        backdrop-filter: blur(10px);
        border: 1px solid rgba(0, 210, 255, 0.2);
    }
    .stTabs [data-baseweb="tab"] {
        background: rgba(255, 255, 255, 0.05);
        border-radius: 40px;
        padding: 10px 24px;
        font-weight: 600;
        transition: all 0.3s ease;
        border: none;
        color: #ccc;
    }
    .stTabs [data-baseweb="tab"]:hover {
        background: rgba(0, 210, 255, 0.2);
        transform: translateY(-2px);
    }
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, #00d2ff 0%, #3a7bd5 100%);
        color: white;
        box-shadow: 0 0 20px rgba(0, 210, 255, 0.5);
        animation: glow 2s infinite;
    }
    .main-header {
        background: linear-gradient(135deg, #00d2ff 0%, #3a7bd5 50%, #00ff88 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        font-size: 3rem;
        font-weight: bold;
        text-align: center;
        margin-bottom: 20px;
        animation: fadeIn 0.8s ease-out;
        text-shadow: 0 0 30px rgba(0, 210, 255, 0.5);
    }
    .sub-header {
        text-align: center;
        background: linear-gradient(135deg, #00d2ff 0%, #3a7bd5 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        font-size: 1.2rem;
        margin-bottom: 30px;
        letter-spacing: 2px;
        animation: slideIn 0.6s ease-out;
    }
    .stButton > button {
        background: linear-gradient(135deg, #00d2ff 0%, #3a7bd5 100%);
        border: none;
        border-radius: 50px;
        padding: 12px 28px;
        font-weight: bold;
        font-size: 1rem;
        transition: all 0.3s ease;
        box-shadow: 0 4px 15px rgba(0, 210, 255, 0.3);
        animation: fadeIn 0.5s ease-out;
    }
    .stButton > button:hover {
        transform: scale(1.05) translateY(-2px);
        box-shadow: 0 8px 30px rgba(0, 210, 255, 0.6);
    }
    .zone-selector {
        background: linear-gradient(135deg, rgba(0, 0, 0, 0.5) 0%, rgba(26, 26, 46, 0.5) 100%);
        border-radius: 15px;
        padding: 15px;
        margin-bottom: 20px;
        border: 1px solid rgba(0, 210, 255, 0.3);
        backdrop-filter: blur(10px);
    }
    .warning-card {
        background: rgba(255, 100, 100, 0.2);
        border-left: 4px solid #ff6b6b;
        padding: 10px;
        border-radius: 8px;
        margin: 10px 0;
        animation: slideIn 0.3s ease-out;
    }
    .info-card {
        background: rgba(0, 210, 255, 0.1);
        border-left: 4px solid #00d2ff;
        padding: 10px;
        border-radius: 8px;
        margin: 10px 0;
        animation: slideIn 0.3s ease-out;
    }
    .success-card {
        background: linear-gradient(135deg, rgba(0, 255, 136, 0.1) 0%, rgba(0, 210, 255, 0.1) 100%);
        border-left: 4px solid #00ff88;
        padding: 10px;
        border-radius: 8px;
        margin: 10px 0;
        animation: slideIn 0.3s ease-out;
    }
    .stat-badge {
        background: linear-gradient(135deg, #00d2ff20 0%, #3a7bd520 100%);
        border-radius: 10px;
        padding: 8px 15px;
        text-align: center;
        border: 1px solid rgba(0, 210, 255, 0.3);
        transition: all 0.3s ease;
    }
    .stat-badge:hover {
        transform: scale(1.05);
        border-color: rgba(0, 210, 255, 0.8);
    }
    .glass-card {
        background: rgba(255, 255, 255, 0.05);
        backdrop-filter: blur(10px);
        border-radius: 20px;
        padding: 20px;
        border: 1px solid rgba(255, 255, 255, 0.1);
        transition: all 0.3s ease;
    }
    .glass-card:hover {
        border-color: rgba(0, 210, 255, 0.5);
        box-shadow: 0 8px 32px rgba(0, 210, 255, 0.1);
    }
    div[data-testid="stSelectbox"] > div {
        background: rgba(0, 0, 0, 0.3);
        border-radius: 10px;
        border: 1px solid rgba(0, 210, 255, 0.3);
    }
    div[data-testid="stMultiSelect"] > div {
        background: rgba(0, 0, 0, 0.3);
        border-radius: 10px;
        border: 1px solid rgba(0, 210, 255, 0.3);
    }
    .stSlider > div > div {
        background: linear-gradient(135deg, #00d2ff 0%, #3a7bd5 100%);
    }
    hr {
        background: linear-gradient(90deg, transparent, #00d2ff, transparent);
        height: 2px;
        border: none;
        margin: 20px 0;
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

# ==================== CONSTANTES FISIOLÓGICAS ====================
VEL_MAX_HUMANA_MS = 12.0  # 43.2 km/h
VEL_MIN_HUMANA_MS = 0.0
ACEL_MAX_HUMANA_MS2 = 7.0
DESACEL_MAX_HUMANA_MS2 = -8.5

# ==================== FUNÇÃO DE VALIDAÇÃO E FILTRAGEM ====================

def validar_e_filtrar_dados(df):
    """Valida e filtra dados com base em limites fisiológicos"""
    df_filtrado = df.copy()
    alertas = []
    
    if 'Velocity' in df_filtrado.columns:
        v_ms = df_filtrado['Velocity'].values
        mask_vel_valida = (v_ms >= VEL_MIN_HUMANA_MS) & (v_ms <= VEL_MAX_HUMANA_MS)
        num_removidos_vel = (~mask_vel_valida).sum()
        
        if num_removidos_vel > 0:
            alertas.append(f"⚠️ {num_removidos_vel} registros com velocidade fora do limite humano (0-{VEL_MAX_HUMANA_MS*3.6:.0f} km/h) foram removidos")
            df_filtrado = df_filtrado[mask_vel_valida]
    
    if 'Acceleration' in df_filtrado.columns:
        a_ms2 = df_filtrado['Acceleration'].values
        mask_acc_valida = (a_ms2 >= DESACEL_MAX_HUMANA_MS2) & (a_ms2 <= ACEL_MAX_HUMANA_MS2)
        num_removidos_acc = (~mask_acc_valida).sum()
        
        if num_removidos_acc > 0:
            alertas.append(f"⚠️ {num_removidos_acc} registros com aceleração fora do limite humano ({DESACEL_MAX_HUMANA_MS2:.0f}-{ACEL_MAX_HUMANA_MS2:.0f} m/s²) foram removidos")
            df_filtrado = df_filtrado[mask_acc_valida]
    
    return df_filtrado, alertas

# ==================== FUNÇÕES DE EXTRAÇÃO DE DADOS DO CABEÇALHO ====================

def extract_athlete_from_line8(content):
    """
    Extrai o nome do atleta da linha 8 do arquivo CSV.
    Formato esperado: # Athlete: "NOME" ou # Athlete: NOME; ou # Athlete: NOME
    """
    try:
        lines = content.split('\n')
        if len(lines) >= 8:
            line8 = lines[7]  # Linha 8 (índice 7)
            
            if '# Athlete:' in line8 or '#Atleta:' in line8:
                # Encontra a posição após os dois pontos
                if ':' in line8:
                    after_colon = line8.split(':', 1)[1].strip()
                    
                    # Tenta extrair entre aspas primeiro
                    match_quotes = re.search(r'"([^"]*)"', after_colon)
                    if match_quotes:
                        athlete = match_quotes.group(1).strip()
                        if athlete:
                            return athlete
                    
                    # Se não houver aspas, pega até o primeiro ';' ou fim da linha
                    if ';' in after_colon:
                        athlete = after_colon.split(';')[0].strip()
                    else:
                        athlete = after_colon.strip()
                    
                    # Remove caracteres indesejados
                    athlete = athlete.strip('"').strip("'").strip()
                    
                    # Se o nome estiver vazio ou for apenas ';', retorna None
                    if athlete and athlete != '' and not athlete.startswith('#') and athlete != ';':
                        return athlete
        return None
    except Exception:
        return None


def extract_period_from_content(content):
    """
    Extrai o período do arquivo CSV.
    Formato esperado: # Period: "NOME" ou # Period: NOME; ou # Period: NOME
    """
    try:
        lines = content.split('\n')
        for line in lines[:15]:
            if '# Period:' in line or '# Periodo:' in line or '#Período:' in line:
                if ':' in line:
                    after_colon = line.split(':', 1)[1].strip()
                    
                    # Tenta extrair entre aspas
                    match_quotes = re.search(r'"([^"]*)"', after_colon)
                    if match_quotes:
                        periodo = match_quotes.group(1).strip()
                        if periodo:
                            return periodo
                    
                    # Se não houver aspas, pega até o primeiro ';' ou fim da linha
                    if ';' in after_colon:
                        periodo = after_colon.split(';')[0].strip()
                    else:
                        periodo = after_colon.strip()
                    
                    periodo = periodo.strip('"').strip("'").strip()
                    
                    if periodo and periodo != '' and not periodo.startswith('#') and periodo != ';':
                        return periodo
        return "Não identificado"
    except Exception:
        return "Não identificado"


def format_athlete_name(athlete):
    """
    Formata o nome do atleta para exibição consistente.
    Ex: "L.SASHA" -> "L. SASHA"
        "L.SASHA;" -> "L. SASHA"
        "L. SASHA" -> "L. SASHA"
    """
    if not athlete:
        return athlete
    
    # Remove caracteres especiais no final
    athlete = athlete.rstrip(';').strip()
    
    # Adiciona espaço após ponto se não houver
    if '.' in athlete and ' ' not in athlete:
        parts = athlete.split('.')
        if len(parts) > 1:
            athlete = parts[0] + '. ' + ''.join(parts[1:])
    
    return athlete

# ==================== FUNÇÃO DO PERFIL ACELERAÇÃO-VELOCIDADE ====================

def fit_velocidade_aceleracao(velocidades, aceleracoes):
    """
    Ajusta a curva de relação Aceleração-Velocidade baseada no modelo físico correto:
    a(v) = a0 * (1 - v/v0)
    """
    mask = (velocidades > 0) & (aceleracoes > -5) & (aceleracoes < 10) & (~np.isnan(velocidades)) & (~np.isnan(aceleracoes))
    v_clean = velocidades[mask]
    a_clean = aceleracoes[mask]
    
    if len(v_clean) < 10:
        return None
    
    # Filtrar outliers usando IQR
    q1_a, q3_a = np.percentile(a_clean, [25, 75])
    iqr_a = q3_a - q1_a
    lower_bound_a = q1_a - 1.5 * iqr_a
    upper_bound_a = q3_a + 1.5 * iqr_a
    
    q1_v, q3_v = np.percentile(v_clean, [25, 75])
    iqr_v = q3_v - q1_v
    lower_bound_v = q1_v - 1.5 * iqr_v
    upper_bound_v = q3_v + 1.5 * iqr_v
    
    mask_outliers = (a_clean >= lower_bound_a) & (a_clean <= upper_bound_a) & \
                    (v_clean >= lower_bound_v) & (v_clean <= upper_bound_v)
    
    v_filtered = v_clean[mask_outliers]
    a_filtered = a_clean[mask_outliers]
    
    if len(v_filtered) < 8:
        v_filtered = v_clean
        a_filtered = a_clean
    
    try:
        X = v_filtered.reshape(-1, 1)
        y = a_filtered
        
        scaler_X = StandardScaler()
        scaler_y = StandardScaler()
        X_scaled = scaler_X.fit_transform(X)
        y_scaled = scaler_y.fit_transform(y.reshape(-1, 1)).ravel()
        
        ransac = RANSACRegressor(min_samples=0.5, residual_threshold=np.std(y_scaled) * 1.5, max_trials=100, random_state=42)
        ransac.fit(X_scaled, y_scaled)
        
        slope_scaled = ransac.estimator_.coef_[0]
        intercept_scaled = ransac.estimator_.intercept_
        
        slope = slope_scaled * (scaler_y.scale_ / scaler_X.scale_)
        intercept = scaler_y.mean_ + intercept_scaled * scaler_y.scale_ - slope * scaler_X.mean_
        
        y_pred = slope * v_filtered + intercept
        ss_res = np.sum((a_filtered - y_pred) ** 2)
        ss_tot = np.sum((a_filtered - np.mean(a_filtered)) ** 2)
        r2 = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
        
        a0 = intercept
        if a0 <= 0 or a0 > 8:
            a0 = np.percentile(a_filtered[a_filtered > 0], 85) if np.any(a_filtered > 0) else 2.5
            a0 = np.clip(a0, 1.5, 6.0)
        
        if slope < 0:
            v0 = -a0 / slope
        else:
            a_positivas = a_filtered[a_filtered > 0.1]
            v_correspondentes = v_filtered[a_filtered > 0.1]
            if len(v_correspondentes) > 0:
                v0 = np.percentile(v_correspondentes, 90) * 1.2
            else:
                v0 = np.max(v_filtered) * 1.1
            v0 = np.clip(v0, 6.0, VEL_MAX_HUMANA_MS)
            slope = -a0 / v0
        
        v0 = np.clip(v0, 6.0, VEL_MAX_HUMANA_MS)
        p_max = a0 * v0 / 4
        p_max = np.clip(p_max, 4.0, 20.0)
        
        return {
            'a0': float(a0),
            'v0': float(v0),
            'slope': float(slope),
            'intercept': float(intercept),
            'r2': float(r2),
            'v_max': float(np.max(v_filtered)),
            'p_max': float(p_max),
            'n_points': len(v_filtered)
        }
    except Exception:
        try:
            z_scores = np.abs(stats.zscore(a_clean))
            mask_z = z_scores < 2.5
            v_z = v_clean[mask_z]
            a_z = a_clean[mask_z]
            
            if len(v_z) >= 5:
                slope, intercept, r_value, p_value, std_err = stats.linregress(v_z, a_z)
                
                a0 = intercept
                if a0 <= 0:
                    a0 = np.percentile(a_z[a_z > 0], 75) if np.any(a_z > 0) else 2.5
                    a0 = np.clip(a0, 1.5, 6.0)
                
                if slope < 0:
                    v0 = -a0 / slope
                else:
                    v0 = np.percentile(v_z, 90) * 1.2
                
                v0 = np.clip(v0, 6.0, VEL_MAX_HUMANA_MS)
                slope = -a0 / v0
                p_max = a0 * v0 / 4
                p_max = np.clip(p_max, 4.0, 20.0)
                r2 = r_value ** 2
                
                return {
                    'a0': float(a0),
                    'v0': float(v0),
                    'slope': float(slope),
                    'intercept': float(intercept),
                    'r2': float(r2),
                    'v_max': float(np.max(v_z)),
                    'p_max': float(p_max),
                    'n_points': len(v_z)
                }
        except:
            pass
        
        return None

def calcular_asp_metrics(df):
    """Calcula as métricas do Perfil Aceleração-Velocidade"""
    df_sprints = df[(df['Acceleration'] > 0) & (df['Velocity'] > 0)].copy()
    if len(df_sprints) < 10:
        return None
    
    v_ms = df_sprints['Velocity'].values
    a_ms2 = df_sprints['Acceleration'].values
    v_ms = np.clip(v_ms, 0, VEL_MAX_HUMANA_MS)
    
    result = fit_velocidade_aceleracao(v_ms, a_ms2)
    
    if result:
        result['v_medio'] = np.mean(v_ms)
        result['a_medio'] = np.mean(a_ms2)
        result['num_sprints'] = len(df_sprints)
    
    return result

# ==================== FUNÇÕES PARA TRIMP ====================

def calcular_trimp_edwards(df, fc_max=None):
    """Calcula o TRIMP usando o método de Edwards"""
    if fc_max is None:
        fc_max = df['HeartRate'].max()
    
    fc_percent = (df['HeartRate'] / fc_max) * 100
    
    if len(df) > 1:
        sample_rate = df['Seconds'].diff().median()
    else:
        sample_rate = 1
    
    trimp_total = 0
    zonas = []
    
    zonas_config = [
        (50, 60, 1, 'Z1 (50-60%)'),
        (60, 70, 2, 'Z2 (60-70%)'),
        (70, 80, 3, 'Z3 (70-80%)'),
        (80, 90, 4, 'Z4 (80-90%)'),
        (90, 100, 5, 'Z5 (90-100%)')
    ]
    
    for min_pct, max_pct, fator, nome in zonas_config:
        mask = (fc_percent >= min_pct) & (fc_percent < max_pct)
        tempo = mask.sum() * sample_rate / 60
        trimp_zona = tempo * fator
        trimp_total += trimp_zona
        zonas.append({'Zona': nome, 'Tempo (min)': tempo, 'Fator': fator, 'TRIMP': trimp_zona})
    
    return trimp_total, pd.DataFrame(zonas)

# ==================== FUNÇÕES DE CONVERSÃO E CAMPO ====================

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

# ==================== NOVAS FUNÇÕES PARA VISUALIZAÇÕES AVANÇADAS ====================

def calcular_metrica_esforco_acumulado(df):
    """Calcula métricas de esforço acumulado por zona"""
    if len(df) < 2:
        return None
    
    # Calcular aceleração instantânea como derivada da velocidade
    df = df.sort_values('Seconds').copy()
    df['Accel_deriv'] = df['Velocity'].diff() / df['Seconds'].diff()
    
    # Calcular potência (simplificada: velocidade * aceleração)
    df['Potencia'] = df['Velocity'] * df['Acceleration'].clip(lower=0)
    
    # Calcular carga metabólica (simplificada)
    df['Carga_Metabolica'] = df['Velocity'] ** 2 * df['Acceleration'].clip(lower=0)
    
    # Identificar sprints (>25 km/h)
    df['Is_Sprint'] = df['Velocity'] * 3.6 > 25
    
    # Identificar mudanças de direção (desaceleração seguida de aceleração)
    df['Mudanca_Direcao'] = ((df['Acceleration'] < -2) & (df['Acceleration'].shift(-1) > 2)) | \
                             ((df['Acceleration'] > 2) & (df['Acceleration'].shift(1) < -2))
    
    return df

def criar_visualizacoes_avancadas_campo(df_tat_sample, bounds, periodo_nome, atleta_nome):
    """Cria visualizações avançadas no campo"""
    
    if len(df_tat_sample) < 10:
        return []
    
    figuras = []
    
    # 1. Mapa de Calor de Acelerações (Onde o atleta mais acelera)
    fig_acel = go.Figure()
    for shape in desenhar_campo_futebol():
        fig_acel.add_shape(shape)
    
    acel_z = df_tat_sample['Acceleration'].clip(lower=0)
    fig_acel.add_trace(go.Scatter(
        x=df_tat_sample['campo_x'], 
        y=df_tat_sample['campo_y'],
        mode='markers',
        marker=dict(
            size=8,
            color=acel_z,
            colorscale='RdYlGn',
            showscale=True,
            colorbar=dict(title="Aceleração (m/s²)"),
            opacity=0.6
        ),
        text=[f"Acel: {a:.2f} m/s²<br>Vel: {v*3.6:.1f} km/h" for a, v in zip(acel_z, df_tat_sample['Velocity'])],
        hoverinfo='text',
        name='Acelerações'
    ))
    
    fig_acel.update_layout(
        title=f"⚡ Mapa de Acelerações - {atleta_nome} - {periodo_nome}",
        xaxis_title="Posição (m) - Comprimento",
        yaxis_title="Posição (m) - Largura",
        height=500,
        xaxis=dict(scaleanchor="y", scaleratio=1, range=[X_MIN-2, X_MAX+2]),
        yaxis=dict(range=[Y_MIN-2, Y_MAX+2]),
        plot_bgcolor='rgba(34,139,34,0.2)'
    )
    figuras.append(("⚡ Mapa de Acelerações", fig_acel))
    
    # 2. Mapa de Potência (Velocidade x Aceleração)
    fig_pot = go.Figure()
    for shape in desenhar_campo_futebol():
        fig_pot.add_shape(shape)
    
    potencia = df_tat_sample['Velocity'] * df_tat_sample['Acceleration'].clip(lower=0)
    
    fig_pot.add_trace(go.Scatter(
        x=df_tat_sample['campo_x'], 
        y=df_tat_sample['campo_y'],
        mode='markers',
        marker=dict(
            size=8,
            color=potencia,
            colorscale='Plasma',
            showscale=True,
            colorbar=dict(title="Potência (m²/s³)"),
            opacity=0.6
        ),
        text=[f"Pot: {p:.2f}<br>Vel: {v*3.6:.1f} km/h<br>Acel: {a:.2f}" 
              for p, v, a in zip(potencia, df_tat_sample['Velocity'], df_tat_sample['Acceleration'])],
        hoverinfo='text',
        name='Potência'
    ))
    
    fig_pot.update_layout(
        title=f"💪 Mapa de Potência - {atleta_nome} - {periodo_nome}",
        xaxis_title="Posição (m) - Comprimento",
        yaxis_title="Posição (m) - Largura",
        height=500,
        xaxis=dict(scaleanchor="y", scaleratio=1, range=[X_MIN-2, X_MAX+2]),
        yaxis=dict(range=[Y_MIN-2, Y_MAX+2]),
        plot_bgcolor='rgba(34,139,34,0.2)'
    )
    figuras.append(("💪 Mapa de Potência", fig_pot))
    
    # 3. Mapa de Sprints (Zonas onde ocorrem sprints)
    fig_sprints = go.Figure()
    for shape in desenhar_campo_futebol():
        fig_sprints.add_shape(shape)
    
    df_sprints = df_tat_sample[df_tat_sample['Velocity'] * 3.6 > 25].copy()
    
    if len(df_sprints) > 0:
        fig_sprints.add_trace(go.Scatter(
            x=df_sprints['campo_x'], 
            y=df_sprints['campo_y'],
            mode='markers',
            marker=dict(
                size=12,
                color='red',
                symbol='star',
                opacity=0.8,
                line=dict(width=1, color='white')
            ),
            text=[f"Sprint!<br>Vel: {v*3.6:.1f} km/h<br>Acel: {a:.2f}" 
                  for v, a in zip(df_sprints['Velocity'], df_sprints['Acceleration'])],
            hoverinfo='text',
            name='Sprints'
        ))
    
    fig_sprints.update_layout(
        title=f"🏃 Sprints (>{25} km/h) - {atleta_nome} - {periodo_nome}",
        xaxis_title="Posição (m) - Comprimento",
        yaxis_title="Posição (m) - Largura",
        height=500,
        xaxis=dict(scaleanchor="y", scaleratio=1, range=[X_MIN-2, X_MAX+2]),
        yaxis=dict(range=[Y_MIN-2, Y_MAX+2]),
        plot_bgcolor='rgba(34,139,34,0.2)'
    )
    figuras.append(("🏃 Mapa de Sprints", fig_sprints))
    
    # 4. Mapa de Frequência Cardíaca (sobreposto no campo)
    fig_fc_campo = go.Figure()
    for shape in desenhar_campo_futebol():
        fig_fc_campo.add_shape(shape)
    
    fig_fc_campo.add_trace(go.Scatter(
        x=df_tat_sample['campo_x'], 
        y=df_tat_sample['campo_y'],
        mode='markers',
        marker=dict(
            size=8,
            color=df_tat_sample['HeartRate'],
            colorscale='Hot',
            showscale=True,
            colorbar=dict(title="FC (bpm)"),
            opacity=0.6
        ),
        text=[f"FC: {fc:.0f} bpm<br>Vel: {v*3.6:.1f} km/h" 
              for fc, v in zip(df_tat_sample['HeartRate'], df_tat_sample['Velocity'])],
        hoverinfo='text',
        name='Frequência Cardíaca'
    ))
    
    fig_fc_campo.update_layout(
        title=f"❤️ Mapa de Frequência Cardíaca - {atleta_nome} - {periodo_nome}",
        xaxis_title="Posição (m) - Comprimento",
        yaxis_title="Posição (m) - Largura",
        height=500,
        xaxis=dict(scaleanchor="y", scaleratio=1, range=[X_MIN-2, X_MAX+2]),
        yaxis=dict(range=[Y_MIN-2, Y_MAX+2]),
        plot_bgcolor='rgba(34,139,34,0.2)'
    )
    figuras.append(("❤️ Mapa de Frequência Cardíaca", fig_fc_campo))
    
    # 5. Mapa de Eficiência (Relação Velocidade/FC)
    fig_eficiencia = go.Figure()
    for shape in desenhar_campo_futebol():
        fig_eficiencia.add_shape(shape)
    
    eficiencia = df_tat_sample['Velocity'] / (df_tat_sample['HeartRate'] / 100)
    eficiencia = eficiencia.clip(upper=10)
    
    fig_eficiencia.add_trace(go.Scatter(
        x=df_tat_sample['campo_x'], 
        y=df_tat_sample['campo_y'],
        mode='markers',
        marker=dict(
            size=8,
            color=eficiencia,
            colorscale='Viridis',
            showscale=True,
            colorbar=dict(title="Eficiência (m/s por 100bpm)"),
            opacity=0.6
        ),
        text=[f"Eficiência: {e:.2f}<br>Vel: {v*3.6:.1f} km/h<br>FC: {fc:.0f}" 
              for e, v, fc in zip(eficiencia, df_tat_sample['Velocity'], df_tat_sample['HeartRate'])],
        hoverinfo='text',
        name='Eficiência'
    ))
    
    fig_eficiencia.update_layout(
        title=f"🎯 Mapa de Eficiência (Vel/FC) - {atleta_nome} - {periodo_nome}",
        xaxis_title="Posição (m) - Comprimento",
        yaxis_title="Posição (m) - Largura",
        height=500,
        xaxis=dict(scaleanchor="y", scaleratio=1, range=[X_MIN-2, X_MAX+2]),
        yaxis=dict(range=[Y_MIN-2, Y_MAX+2]),
        plot_bgcolor='rgba(34,139,34,0.2)'
    )
    figuras.append(("🎯 Mapa de Eficiência", fig_eficiencia))
    
    # 6. Análise de Mudanças de Direção
    df_tat_sample['Accel_change'] = df_tat_sample['Acceleration'].diff()
    df_direcao = df_tat_sample[abs(df_tat_sample['Accel_change']) > 3].copy()
    
    fig_direcao = go.Figure()
    for shape in desenhar_campo_futebol():
        fig_direcao.add_shape(shape)
    
    if len(df_direcao) > 0:
        fig_direcao.add_trace(go.Scatter(
            x=df_direcao['campo_x'], 
            y=df_direcao['campo_y'],
            mode='markers',
            marker=dict(
                size=10,
                color='orange',
                symbol='triangle-up',
                opacity=0.8,
                line=dict(width=1, color='white')
            ),
            text=[f"Mudança de Direção<br>ΔAcel: {ac:.2f}<br>Vel: {v*3.6:.1f} km/h" 
                  for ac, v in zip(df_direcao['Accel_change'], df_direcao['Velocity'])],
            hoverinfo='text',
            name='Mudanças de Direção'
        ))
    
    fig_direcao.update_layout(
        title=f"🔄 Mudanças de Direção - {atleta_nome} - {periodo_nome}",
        xaxis_title="Posição (m) - Comprimento",
        yaxis_title="Posição (m) - Largura",
        height=500,
        xaxis=dict(scaleanchor="y", scaleratio=1, range=[X_MIN-2, X_MAX+2]),
        yaxis=dict(range=[Y_MIN-2, Y_MAX+2]),
        plot_bgcolor='rgba(34,139,34,0.2)'
    )
    figuras.append(("🔄 Mudanças de Direção", fig_direcao))
    
    return figuras

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
        headers = {'User-Agent': 'ScoutLab/1.0'}
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

@st.cache_data
def load_data(uploaded_file):
    try:
        content = uploaded_file.getvalue().decode('utf-8')
        
        # Extrai e formata o nome do atleta
        atleta_raw = extract_athlete_from_line8(content)
        if atleta_raw is None or atleta_raw == "":
            atleta = "Não identificado"
        else:
            atleta = format_athlete_name(atleta_raw)
        
        # Extrai o período
        periodo = extract_period_from_content(content)
        
        data_start = 0
        lines = content.split('\n')
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
        
        df_filtrado, alertas = validar_e_filtrar_dados(df)
        
        for alerta in alertas:
            st.warning(alerta)
        
        df_filtrado['arquivo_origem'] = uploaded_file.name
        df_filtrado['start_datetime'] = start_datetime
        if start_datetime is not None:
            df_filtrado['Horario'] = df_filtrado['Seconds'].apply(lambda x: seconds_to_time_str(x, start_datetime))
        
        return df_filtrado, atleta, periodo, start_datetime
    except Exception as e:
        st.error(f"Erro ao carregar arquivo {uploaded_file.name}: {e}")
        return None, None, None, None

# ==================== FUNÇÃO PARA GERAR PDF ====================

def gerar_relatorio_pdf(dfs_por_periodo, distancias_por_periodo, tempos_por_periodo, selected_atletas, periodo_atual, selected_atletas_nomes):
    buffer = BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=A4, rightMargin=72, leftMargin=72, topMargin=72, bottomMargin=72)
    styles = getSampleStyleSheet()
    story = []
    
    title_style = ParagraphStyle('CustomTitle', parent=styles['Heading1'], fontSize=24,
                                  textColor=colors.HexColor('#00d2ff'), alignment=TA_CENTER, spaceAfter=30)
    story.append(Paragraph("ScoutLab - Relatório de Performance Esportiva", title_style))
    story.append(Spacer(1, 12))
    
    atletas_str = ", ".join(selected_atletas_nomes) if selected_atletas_nomes else "Não identificado"
    story.append(Paragraph(f"Atleta(s): {atletas_str}", styles['Heading2']))
    story.append(Paragraph(f"Período analisado: {periodo_atual}", styles['Normal']))
    story.append(Paragraph(f"Data do relatório: {datetime.now().strftime('%d/%m/%Y %H:%M')}", styles['Normal']))
    story.append(Spacer(1, 20))
    
    story.append(Paragraph("Métricas de Desempenho", styles['Heading2']))
    
    df_metricas = dfs_por_periodo[periodo_atual]
    dist_total = distancias_por_periodo.get(periodo_atual, 0)
    tempo_total = tempos_por_periodo.get(periodo_atual, 0) / 60
    
    metrics_data = [
        ['Métrica', 'Valor'],
        ['Distância Total', f'{dist_total:.0f} m'],
        ['Tempo Total', f'{tempo_total:.1f} min'],
        ['Velocidade Máxima', f'{df_metricas["Velocity"].max() * 3.6:.1f} km/h'],
        ['Velocidade Média', f'{df_metricas["Velocity"].mean() * 3.6:.1f} km/h'],
        ['FC Máxima', f'{df_metricas["HeartRate"].max():.0f} bpm'],
        ['FC Média', f'{df_metricas["HeartRate"].mean():.0f} bpm'],
    ]
    
    metrics_table = Table(metrics_data, colWidths=[200, 200])
    metrics_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#00d2ff')),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, 0), 12),
        ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
        ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
        ('GRID', (0, 0), (-1, -1), 1, colors.black)
    ]))
    story.append(metrics_table)
    story.append(Spacer(1, 20))
    
    fc_max = df_metricas['HeartRate'].max()
    trimp_total, zonas_trimp = calcular_trimp_edwards(df_metricas, fc_max)
    
    story.append(Paragraph("Carga Interna (TRIMP - Edwards)", styles['Heading2']))
    story.append(Paragraph(f"TRIMP Total: {trimp_total:.1f} unidades", styles['Normal']))
    story.append(Spacer(1, 10))
    
    trimp_data = [['Zona', 'Tempo (min)', 'Fator', 'TRIMP']]
    for _, row in zonas_trimp.iterrows():
        trimp_data.append([row['Zona'], f"{row['Tempo (min)']:.1f}", str(row['Fator']), f"{row['TRIMP']:.1f}"])
    
    trimp_table = Table(trimp_data, colWidths=[150, 100, 80, 100])
    trimp_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#00d2ff')),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('GRID', (0, 0), (-1, -1), 1, colors.black)
    ]))
    story.append(trimp_table)
    
    doc.build(story)
    buffer.seek(0)
    return buffer

# ==================== INICIALIZAÇÃO ====================

init_database()

st.sidebar.header("📁 1. Upload de Arquivos")
uploaded_files = st.sidebar.file_uploader("Escolha os arquivos CSV", type=['csv'], accept_multiple_files=True)

# ==================== CARREGAMENTO DOS DADOS ====================

if uploaded_files:
    if 'dados_carregados' not in st.session_state or st.session_state.get('arquivos_anteriores') != [f.name for f in uploaded_files]:
        with st.spinner("Carregando e validando arquivos..."):
            all_data = []
            all_atletas = []
            all_periodos = []
            all_start_datetimes = []
            all_filenames = []
            
            for file in uploaded_files:
                df, atleta, periodo, start_datetime = load_data(file)
                if df is not None and len(df) > 0:
                    all_data.append(df)
                    all_atletas.append(atleta)
                    all_periodos.append(periodo)
                    all_start_datetimes.append(start_datetime)
                    all_filenames.append(file.name)
            
            if all_data:
                st.session_state.dados_carregados = all_data
                st.session_state.atletas = all_atletas
                st.session_state.periodos_orig = all_periodos
                st.session_state.start_datetimes = all_start_datetimes
                st.session_state.filenames = all_filenames
                st.session_state.reference_datetime = all_start_datetimes[0] if all_start_datetimes[0] is not None else None
                st.session_state.arquivos_anteriores = [f.name for f in uploaded_files]
                st.session_state.dados_prontos = True
                
                st.markdown('<div class="success-card">', unsafe_allow_html=True)
                st.success(f"✅ {len(all_data)} arquivos carregados com sucesso. Dados validados com limites fisiológicos (velocidade máxima: {VEL_MAX_HUMANA_MS*3.6:.0f} km/h, aceleração: {ACEL_MAX_HUMANA_MS2:.0f} m/s²)")
                st.markdown('</div>', unsafe_allow_html=True)
                
                # Mostrar atletas carregados
                atletas_list = ", ".join([f"{a} ({p})" for a, p in zip(all_atletas, all_periodos)])
                st.markdown(f'<div class="info-card">📋 Atletas carregados: {atletas_list}</div>', unsafe_allow_html=True)
            else:
                st.session_state.dados_prontos = False
                st.error("❌ Nenhum arquivo válido processado.")
    
    if st.session_state.get('dados_prontos', False):
        # ==================== CONFIGURAÇÃO (SIDEBAR) ====================
        
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
        
        # 4. Filtros - Zonas de Velocidade Customizáveis
        st.sidebar.markdown("---")
        st.sidebar.header("⚡ 4. Zonas de Velocidade (km/h)")
        
        st.sidebar.markdown('<div class="zone-selector">', unsafe_allow_html=True)
        
        if 'velocidade_zonas' not in st.session_state:
            st.session_state.velocidade_zonas = {
                'Z1 (Caminhada)': (0.0, 7.0),
                'Z2 (Trote)': (7.0, 15.0),
                'Z3 (Corrida Moderada)': (15.0, 21.0),
                'Z4 (Alta Intensidade)': (21.0, 25.0),
                'Z5 (Sprint)': (25.0, 100.0)
            }
        
        st.sidebar.markdown("**Defina os limites das zonas (km/h):**")
        
        zonas_editadas = {}
        for zona, (min_val, max_val) in st.session_state.velocidade_zonas.items():
            col1, col2 = st.sidebar.columns(2)
            with col1:
                novo_min = st.number_input(f"{zona} - min", value=float(min_val), step=0.5, key=f"zona_min_{zona}", format="%.1f")
            with col2:
                novo_max = st.number_input(f"{zona} - max", value=float(max_val), step=0.5, key=f"zona_max_{zona}", format="%.1f")
            zonas_editadas[zona] = (novo_min, novo_max)
        
        st.session_state.velocidade_zonas = zonas_editadas
        
        zonas_selecionadas = st.sidebar.multiselect(
            "Selecionar zonas para análise",
            options=list(zonas_editadas.keys()),
            default=list(zonas_editadas.keys()),
            key="zonas_selecionadas"
        )
        
        st.sidebar.markdown('</div>', unsafe_allow_html=True)
        
        # Filtro de tempo global
        st.sidebar.markdown("---")
        st.sidebar.header("⏰ 5. Filtro Temporal")
        
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
        
        # 6. Seleção de Atletas - MULTIPLA SELEÇÃO MELHORADA
        st.sidebar.markdown("---")
        st.sidebar.header("🏅 6. Selecionar Atleta(s)")
        
        # Criar opções de seleção com "Todos os atletas" como opção especial
        atleta_options_display = ["🏆 Todos os atletas"]
        atleta_mapping = []  # Mapeamento para os índices reais
        
        for i, (atleta, periodo) in enumerate(zip(st.session_state.atletas, st.session_state.periodos_orig)):
            if atleta and atleta != "Não identificado" and not atleta.startswith('#'):
                display_name = f"{atleta} - {periodo}" if periodo and periodo != "Não identificado" and not periodo.startswith('#') else atleta
                atleta_options_display.append(display_name)
                atleta_mapping.append(i)
            elif periodo and periodo != "Não identificado" and not periodo.startswith('#'):
                display_name = f"Atleta {i+1} - {periodo}"
                atleta_options_display.append(display_name)
                atleta_mapping.append(i)
            else:
                display_name = f"Atleta {i+1}"
                atleta_options_display.append(display_name)
                atleta_mapping.append(i)
        
        # Seleção múltipla de atletas
        selected_atleta_indices_display = st.sidebar.multiselect(
            "Escolha os atletas",
            options=range(len(atleta_options_display)),
            format_func=lambda x: atleta_options_display[x],
            default=[0] if len(atleta_options_display) > 0 else [],
            help="Selecione 'Todos os atletas' para visualizar todos ou escolha atletas específicos"
        )
        
        # Processar a seleção
        if 0 in selected_atleta_indices_display:
            # Selecionou "Todos os atletas"
            selected_original_indices = list(range(len(st.session_state.atletas)))
            selected_atletas_nomes = st.session_state.atletas
            st.sidebar.success(f"✅ {len(selected_original_indices)} atletas selecionados")
        else:
            # Selecionou atletas específicos
            selected_original_indices = [atleta_mapping[i-1] for i in selected_atleta_indices_display if i > 0]
            selected_atletas_nomes = [st.session_state.atletas[i] for i in selected_original_indices]
            st.sidebar.info(f"✅ {len(selected_original_indices)} atleta(s) selecionado(s)")
        
        # 7. Opções de Visualização
        st.sidebar.markdown("---")
        st.sidebar.header("🎨 7. Opções de Visualização")
        show_field = st.sidebar.checkbox("Mostrar campo", value=True, key="show_field")
        
        # 8. Seleção de Períodos para Análise
        st.sidebar.markdown("---")
        st.sidebar.header("📊 8. Selecionar Períodos para Análise")
        
        opcoes_periodos_analise = ["📅 Todos os períodos"] + [p['nome'] for p in st.session_state.periodos_config]
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
            if not selected_original_indices:
                st.warning("⚠️ Selecione pelo menos um atleta para análise.")
                st.stop()
            
            selected_atletas = [st.session_state.atletas[i] for i in selected_original_indices]
            selected_data = [st.session_state.dados_carregados[i] for i in selected_original_indices]
            selected_periodos_orig = [st.session_state.periodos_orig[i] for i in selected_original_indices]
            selected_start_datetimes = [st.session_state.start_datetimes[i] for i in selected_original_indices]
            
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
                        
                        speed_filter = pd.Series([False] * len(df))
                        for zona in zonas_selecionadas:
                            min_v_kmh, max_v_kmh = st.session_state.velocidade_zonas[zona]
                            min_v_ms = min_v_kmh / 3.6
                            max_v_ms = max_v_kmh / 3.6
                            zona_filter = (df['Velocity'] >= min_v_ms) & (df['Velocity'] <= max_v_ms)
                            speed_filter = speed_filter | zona_filter
                        
                        df_filtered = df[time_filter & speed_filter].copy()
                        df_filtered['Atleta'] = atleta
                        df_filtered['Periodo'] = periodo_orig
                        df_filtered['Periodo_Analise'] = periodo_nome
                        df_filtered['start_datetime'] = start_dt
                        dfs_periodo.append(df_filtered)
                        if 'Odometer' in df_filtered.columns and len(df_filtered) > 0:
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
                        
                        speed_filter = pd.Series([False] * len(df))
                        for zona in zonas_selecionadas:
                            min_v_kmh, max_v_kmh = st.session_state.velocidade_zonas[zona]
                            min_v_ms = min_v_kmh / 3.6
                            max_v_ms = max_v_kmh / 3.6
                            zona_filter = (df['Velocity'] >= min_v_ms) & (df['Velocity'] <= max_v_ms)
                            speed_filter = speed_filter | zona_filter
                        
                        df_filtered = df[time_filter & speed_filter].copy()
                        df_filtered['Atleta'] = atleta
                        df_filtered['Periodo'] = periodo_orig
                        df_filtered['Periodo_Analise'] = periodo_nome
                        df_filtered['start_datetime'] = start_dt
                        dfs_periodo.append(df_filtered)
                        if 'Odometer' in df_filtered.columns and len(df_filtered) > 0:
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
            st.session_state.selected_atletas_nomes = selected_atletas_nomes
            st.session_state.analise_processada = True
        
        # ==================== EXIBIÇÃO DAS ANÁLISES ====================
        if st.session_state.get('analise_processada', False):
            dfs_por_periodo = st.session_state.dfs_por_periodo
            distancias_por_periodo = st.session_state.distancias_por_periodo
            tempos_por_periodo = st.session_state.tempos_por_periodo
            selected_atletas = st.session_state.selected_atletas
            selected_atletas_nomes = st.session_state.selected_atletas_nomes
            
            # Botão para gerar relatório PDF
            col_pdf1, col_pdf2, col_pdf3 = st.columns([1, 2, 1])
            with col_pdf2:
                if st.button("📄 GERAR RELATÓRIO PDF", type="primary", use_container_width=True):
                    periodo_atual = list(dfs_por_periodo.keys())[0]
                    pdf_buffer = gerar_relatorio_pdf(dfs_por_periodo, distancias_por_periodo, tempos_por_periodo, selected_atletas, periodo_atual, selected_atletas_nomes)
                    atletas_str = "_".join(selected_atletas_nomes[:3])
                    st.download_button(
                        label="📥 BAIXAR RELATÓRIO PDF",
                        data=pdf_buffer,
                        file_name=f"scoutlab_relatorio_{atletas_str}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf",
                        mime="application/pdf",
                        use_container_width=True
                    )
            st.markdown("---")
            
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
                        <div class="metric-value">{df_periodo['Velocity'].max() * 3.6:.1f}</div>
                        <div class="metric-label">Vel Máx (km/h)</div>
                    </div>
                    """, unsafe_allow_html=True)
                with col3:
                    st.markdown(f"""
                    <div class="metric-card">
                        <div class="metric-value">{df_periodo['Velocity'].mean() * 3.6:.1f}</div>
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
                
                # Exibir quais atletas estão no período
                atletas_no_periodo = df_periodo['Atleta'].unique()
                atletas_str = ", ".join(atletas_no_periodo[:3])
                if len(atletas_no_periodo) > 3:
                    atletas_str += f" +{len(atletas_no_periodo)-3}"
                st.markdown(f"**{periodo_nome}** | Tempo: {tempo_min:.1f} min | {len(df_periodo):,} registros | 👥 {atletas_str}")
                st.markdown("---")
            
            # ==================== ABAS ====================
            tab1, tab2, tab3, tab4, tab5 = st.tabs([
                "🗺️ Mapa Tático", 
                "📐 Análise por Zonas", 
                "⚡ Perfil Aceleração-Velocidade",
                "❤️ Performance Cardíaca",
                "📊 Comparação Esportiva"
            ])
            
            # TAB 1: MAPA TÁTICO (SEM 3D)
            with tab1:
                st.subheader("🗺️ Mapa Tático de Posicionamento")
                
                col_map1, col_map2 = st.columns([2, 1])
                with col_map1:
                    periodo_mapa = st.selectbox("Selecionar período", options=list(dfs_por_periodo.keys()), key="periodo_mapa_select")
                with col_map2:
                    atleta_mapa = st.selectbox("Selecionar atleta", options=["Todos"] + list(dfs_por_periodo[periodo_mapa]['Atleta'].unique()), key="atleta_mapa_select")
                
                df_mapa = dfs_por_periodo[periodo_mapa]
                
                if atleta_mapa != "Todos":
                    df_mapa = df_mapa[df_mapa['Atleta'] == atleta_mapa]
                
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
                
                # Plotar trajetórias por atleta com cores diferentes
                if atleta_mapa == "Todos":
                    atletas_unicos = df_mapa['Atleta'].unique()
                    cores = px.colors.qualitative.Plotly
                    for idx, atleta in enumerate(atletas_unicos):
                        df_atleta = df_mapa[df_mapa['Atleta'] == atleta]
                        if len(df_atleta) > 5000:
                            df_atleta = df_atleta.sample(5000, random_state=42)
                        
                        hover_texts = [f"<b>{atleta}</b><br>{seconds_to_time_str(row['Seconds'], row['start_datetime'])}<br>Vel: {row['Velocity'] * 3.6:.1f} km/h<br>FC: {row['HeartRate']:.0f} bpm" for _, row in df_atleta.iterrows()]
                        fig_map.add_trace(go.Scattermapbox(lat=df_atleta['Latitude'], lon=df_atleta['Longitude'], mode='markers',
                                                           marker=dict(size=4, color=cores[idx % len(cores)], symbol='circle'),
                                                           text=hover_texts, hoverinfo='text', name=atleta))
                else:
                    if len(df_mapa) > 5000:
                        df_mapa_plot = df_mapa.sample(5000, random_state=42)
                    else:
                        df_mapa_plot = df_mapa
                    
                    hover_texts = [f"<b>{seconds_to_time_str(row['Seconds'], row['start_datetime'])}</b><br>Vel: {row['Velocity'] * 3.6:.1f} km/h<br>FC: {row['HeartRate']:.0f} bpm" for _, row in df_mapa_plot.iterrows()]
                    fig_map.add_trace(go.Scattermapbox(lat=df_mapa_plot['Latitude'], lon=df_mapa_plot['Longitude'], mode='markers',
                                                       marker=dict(size=4, color=df_mapa_plot['Velocity'] * 3.6, colorscale='Viridis', showscale=True,
                                                                  colorbar=dict(title="Velocidade (km/h)")), text=hover_texts, hoverinfo='text', name='Percurso'))
                    
                    if len(df_mapa_plot) > 0:
                        fig_map.add_trace(go.Scattermapbox(lat=[df_mapa_plot['Latitude'].iloc[0]], lon=[df_mapa_plot['Longitude'].iloc[0]],
                                                           mode='markers', marker=dict(size=16, color='green'), name='Início'))
                        fig_map.add_trace(go.Scattermapbox(lat=[df_mapa_plot['Latitude'].iloc[-1]], lon=[df_mapa_plot['Longitude'].iloc[-1]],
                                                           mode='markers', marker=dict(size=16, color='red'), name='Fim'))
                
                zoom = 18 if bounds else 15
                fig_map.update_layout(mapbox=dict(style="open-street-map", center=dict(lat=center_lat, lon=center_lon), zoom=zoom),
                                      height=600, margin=dict(l=0, r=0, t=30, b=0),
                                      title=f"Trajetória - {atleta_mapa if atleta_mapa != 'Todos' else 'Múltiplos atletas'} - {periodo_mapa} - {st.session_state.get('nome_estadio', 'Campo')}")
                st.plotly_chart(fig_map, use_container_width=True)
            
            # TAB 2: ANÁLISE POR ZONAS (COM NOVAS VISUALIZAÇÕES)
            with tab2:
                st.subheader("📐 Análise Tática por Zonas do Campo")
                st.markdown(f"Campo com dimensões oficiais: **{CAMPO_COMPRIMENTO}m x {CAMPO_LARGURA}m**")
                
                col_zone1, col_zone2 = st.columns(2)
                with col_zone1:
                    periodo_tatica = st.selectbox("Selecionar período", options=list(dfs_por_periodo.keys()), key="periodo_tatica_select")
                with col_zone2:
                    atleta_tatica = st.selectbox("Selecionar atleta", options=["Todos"] + list(dfs_por_periodo[periodo_tatica]['Atleta'].unique()), key="atleta_tatica_select")
                
                df_tat = dfs_por_periodo[periodo_tatica]
                
                if atleta_tatica != "Todos":
                    df_tat = df_tat[df_tat['Atleta'] == atleta_tatica]
                
                start_dt_tat = df_tat['start_datetime'].iloc[0] if len(df_tat) > 0 else None
                
                st.markdown(f"### 📊 Métricas do Período: {periodo_tatica} - Atleta: {atleta_tatica if atleta_tatica != 'Todos' else 'Todos combinados'}")
                col_metric1, col_metric2, col_metric3, col_metric4, col_metric5 = st.columns(5)
                with col_metric1:
                    dist = distancias_por_periodo.get(periodo_tatica, 0)
                    st.metric("Distância", f"{dist:.0f} m")
                with col_metric2:
                    st.metric("Velocidade Máxima", f"{df_tat['Velocity'].max() * 3.6:.1f} km/h")
                with col_metric3:
                    st.metric("Velocidade Média", f"{df_tat['Velocity'].mean() * 3.6:.1f} km/h")
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
                
                # Visualizações avançadas
                st.markdown("### 🎯 Visualizações Avançadas de Performance")
                visualizacoes_avancadas = criar_visualizacoes_avancadas_campo(df_tat_sample, bounds, periodo_tatica, atleta_tatica)
                
                for titulo, fig in visualizacoes_avancadas:
                    with st.expander(f"{titulo}", expanded=False):
                        st.plotly_chart(fig, use_container_width=True)
                
                # Análise de zonas tradicional
                st.markdown("### 🗺️ Análise por Zonas do Campo")
                
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
                
                # Converter velocidades para km/h
                zona_metrics['Vel_Média'] = zona_metrics['Vel_Média'] * 3.6
                zona_metrics['Vel_Máx'] = zona_metrics['Vel_Máx'] * 3.6
                
                viz_type = st.radio("Tipo de visualização", 
                                   ["Trajetória por zona", "Mapa de calor - Tempo", "Mapa de calor - Velocidade", "Mapa de calor - Densidade (KDE)"], 
                                   horizontal=True)
                
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
                                                     text=[f"Zona: {zona}<br>Vel: {v * 3.6:.1f} km/h" for v in group['Velocity']],
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
                
                elif viz_type == "Mapa de calor - Velocidade":
                    heatmap = np.zeros((num_linhas, num_colunas))
                    for i in range(num_linhas):
                        for j in range(num_colunas):
                            zona = f'L{i+1}-C{j+1}'
                            if zona in zona_metrics.index:
                                heatmap[i, j] = zona_metrics.loc[zona, 'Vel_Média']
                    fig_tat.add_trace(go.Heatmap(x=linhas_bins_plot, y=colunas_bins_plot, z=heatmap.T, colorscale='Viridis', opacity=0.7,
                                                 colorbar=dict(title="Velocidade média (km/h)")))
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
                
                else:
                    from scipy.stats import gaussian_kde
                    if len(df_tat_sample) > 1:
                        xy = np.vstack([df_tat_sample['campo_x'], df_tat_sample['campo_y']])
                        z = gaussian_kde(xy)(xy)
                        fig_tat.add_trace(go.Scatter(x=df_tat_sample['campo_x'], y=df_tat_sample['campo_y'], mode='markers',
                                                     marker=dict(size=4, color=z, colorscale='Hot', showscale=True,
                                                                colorbar=dict(title="Densidade")),
                                                     name='Densidade', hoverinfo='skip'))
                    fig_tat.add_trace(go.Scatter(x=df_tat_sample['campo_x'], y=df_tat_sample['campo_y'], mode='markers',
                                                 marker=dict(size=1, color='white', opacity=0.3), name='Trajetória', hoverinfo='skip'))
                
                fig_tat.update_layout(title=f"Análise Tática - {atleta_tatica if atleta_tatica != 'Todos' else 'Múltiplos atletas'} - {periodo_tatica}",
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
                st.download_button("📥 Exportar análise tática", csv_tatico, f"analise_tatica_{atleta_tatica}_{periodo_tatica}.csv")
            
                        # TAB 3: PERFIL ACELERAÇÃO-VELOCIDADE (COM LINHA DE ACELERAÇÃO MÁXIMA REAL)
            with tab3:
                st.subheader("⚡ Perfil Aceleração-Velocidade (Acceleration-Speed Profile)")
                
                with st.expander("📄 **Referência Científica**"):
                    st.markdown("""
                    **Modelo Físico:** a(v) = a₀ × (1 - v/v₀)
                    
                    **Interpretação:**
                    - **a₀ (m/s²)**: Aceleração máxima teórica (capacidade de aceleração inicial)
                    - **v₀ (m/s)**: Velocidade máxima teórica (onde a aceleração se anula)
                    - **Pₘₐₓ (W/kg)**: Potência máxima = a₀ × v₀ / 4
                    - **R²**: Qualidade do ajuste (>0.7 = excelente)
                    
                    **Valores de referência para atletas de futebol:**
                    - a₀: 2.5-4.5 m/s²
                    - v₀: 8.5-11.5 m/s (30.6-41.4 km/h)
                    - Pₘₐₓ: 5-12 W/kg
                    """)
                
                col_asp1, col_asp2 = st.columns(2)
                with col_asp1:
                    periodo_asp = st.selectbox("Selecionar período", options=list(dfs_por_periodo.keys()), key="asp_periodo_select")
                with col_asp2:
                    atleta_asp = st.selectbox("Selecionar atleta", options=["Todos"] + list(dfs_por_periodo[periodo_asp]['Atleta'].unique()), key="atleta_asp_select")
                
                df_asp_atual = dfs_por_periodo[periodo_asp]
                
                if atleta_asp != "Todos":
                    df_asp_atual = df_asp_atual[df_asp_atual['Atleta'] == atleta_asp]
                
                asp_results = {}
                with st.spinner("Calculando perfil ASP com tratamento estatístico robusto..."):
                    # Agrupar por atleta se necessário
                    if atleta_asp == "Todos":
                        for atleta in df_asp_atual['Atleta'].unique():
                            df_atleta = df_asp_atual[df_asp_atual['Atleta'] == atleta]
                            asp_metrics = calcular_asp_metrics(df_atleta)
                            if asp_metrics:
                                asp_results[atleta] = asp_metrics
                    else:
                        asp_metrics = calcular_asp_metrics(df_asp_atual)
                        if asp_metrics:
                            asp_results[atleta_asp] = asp_metrics
                
                if not asp_results:
                    st.warning("⚠️ Dados insuficientes para calcular o Perfil Aceleração-Velocidade. São necessários pelo menos 10 pontos com aceleração positiva e velocidade > 0.")
                else:
                    st.markdown("### 📈 Curva Aceleração-Velocidade (ASP)")
                    st.markdown("**Relação entre Velocidade (m/s) e Aceleração (m/s²) - Modelo Físico**")
                    
                    fig_asp = go.Figure()
                    
                    cores_asp = px.colors.qualitative.Plotly
                    for idx, (atleta, metrics) in enumerate(asp_results.items()):
                        df_sprints = dfs_por_periodo[periodo_asp]
                        if atleta != "Todos":
                            df_sprints = df_sprints[df_sprints['Atleta'] == atleta]
                        df_sprints = df_sprints[(df_sprints['Acceleration'] > 0) & (df_sprints['Velocity'] > 0)].copy()
                        
                        if len(df_sprints) > 0:
                            v_ms = df_sprints['Velocity'].values
                            a_ms2 = df_sprints['Acceleration'].values
                            
                            # Encontrar os valores reais máximos do atleta (não os teóricos do modelo)
                            v_max_real = np.max(v_ms)
                            a_max_real = np.max(a_ms2)
                            
                            # Dados brutos
                            fig_asp.add_trace(go.Scatter(
                                x=v_ms, y=a_ms2,
                                mode='markers',
                                name=f'{atleta} - Dados',
                                marker=dict(size=6, color=cores_asp[idx % len(cores_asp)], opacity=0.6),
                                hovertemplate=f'{atleta}<br>Vel: %{{x:.2f}} m/s (%{{x*3.6:.1f}} km/h)<br>Acel: %{{y:.2f}} m/s²<extra></extra>'
                            ))
                            
                            # Curva do modelo ASP
                            v_fit = np.linspace(0, min(metrics['v0'], metrics['v_max'] * 1.05), 100)
                            a_fit = metrics['a0'] * (1 - v_fit / metrics['v0'])
                            
                            fig_asp.add_trace(go.Scatter(
                                x=v_fit, y=a_fit,
                                mode='lines',
                                name=f'{atleta} - Modelo',
                                line=dict(color=cores_asp[idx % len(cores_asp)], width=3, dash='dash'),
                                hovertemplate=f'{atleta}<br>a = {metrics["a0"]:.2f} × (1 - v/{metrics["v0"]:.2f})<br>v: %{{x:.2f}} m/s<br>a: %{{y:.2f}} m/s²<extra></extra>'
                            ))
                            
                            # Linha horizontal de aceleração máxima REAL (do eixo Y até o ponto de velocidade máxima real)
                            fig_asp.add_trace(go.Scatter(
                                x=[0, v_max_real], 
                                y=[a_max_real, a_max_real],
                                mode='lines',
                                name=f'{atleta} - Aceleração Máxima Real',
                                line=dict(color='red', width=3, dash='dot'),
                                hovertemplate=f'Aceleração Máxima Real: {a_max_real:.2f} m/s²<br>Velocidade Máxima: %{{x:.2f}} m/s<extra></extra>'
                            ))
                            
                            # Linha vertical de velocidade máxima REAL (do eixo X até o ponto de aceleração máxima real)
                            fig_asp.add_trace(go.Scatter(
                                x=[v_max_real, v_max_real], 
                                y=[0, a_max_real],
                                mode='lines',
                                name=f'{atleta} - Velocidade Máxima Real',
                                line=dict(color='orange', width=3, dash='dot'),
                                hovertemplate=f'Velocidade Máxima Real: {v_max_real:.2f} m/s ({v_max_real*3.6:.1f} km/h)<extra></extra>'
                            ))
                            
                            # Ponto de interseção (velocidade máxima real, aceleração máxima real)
                            fig_asp.add_trace(go.Scatter(
                                x=[v_max_real], 
                                y=[a_max_real],
                                mode='markers',
                                name=f'{atleta} - Ponto de Performance Máxima Real',
                                marker=dict(size=16, color='gold', symbol='star', line=dict(width=2, color='white')),
                                hovertemplate=f'{atleta}<br><b>Velocidade Máxima Real:</b> {v_max_real:.2f} m/s ({v_max_real*3.6:.1f} km/h)<br><b>Aceleração Máxima Real:</b> {a_max_real:.2f} m/s²<br><b>Ponto de Máxima Performance</b><extra></extra>'
                            ))
                            
                            # Adicionar anotação para o valor de aceleração máxima real no eixo Y
                            fig_asp.add_annotation(
                                x=0,
                                y=a_max_real,
                                xref="x",
                                yref="y",
                                text=f"aₘₐₓ real = {a_max_real:.2f} m/s²",
                                showarrow=True,
                                arrowhead=2,
                                arrowsize=1,
                                arrowwidth=2,
                                arrowcolor="red",
                                ax=-60,
                                ay=0,
                                font=dict(size=10, color="red", weight="bold"),
                                bgcolor="rgba(0,0,0,0.6)",
                                borderpad=4
                            )
                            
                            # Adicionar anotação para o valor de velocidade máxima real no eixo X
                            fig_asp.add_annotation(
                                x=v_max_real,
                                y=0,
                                xref="x",
                                yref="y",
                                text=f"vₘₐₓ real = {v_max_real:.2f} m/s<br>({v_max_real*3.6:.1f} km/h)",
                                showarrow=True,
                                arrowhead=2,
                                arrowsize=1,
                                arrowwidth=2,
                                arrowcolor="orange",
                                ax=0,
                                ay=-50,
                                font=dict(size=10, color="orange", weight="bold"),
                                bgcolor="rgba(0,0,0,0.6)",
                                borderpad=4
                            )
                            
                            # Adicionar também as métricas teóricas para referência
                            a0_teorico = metrics['a0']
                            v0_teorico = metrics['v0']
                            
                            fig_asp.add_annotation(
                                x=v0_teorico * 0.7,
                                y=a0_teorico * 0.8,
                                xref="x",
                                yref="y",
                                text=f"Modelo: a₀ = {a0_teorico:.2f} m/s²<br>v₀ = {v0_teorico:.2f} m/s ({v0_teorico*3.6:.1f} km/h)",
                                showarrow=False,
                                font=dict(size=9, color="white"),
                                bgcolor="rgba(0,0,0,0.5)",
                                borderpad=4,
                                bordercolor=cores_asp[idx % len(cores_asp)],
                                borderwidth=1
                            )
                    
                    # Ajustar os limites dos eixos para garantir que as linhas sejam totalmente visíveis
                    max_a0_real = 0
                    max_v_real = 0
                    for atleta, metrics in asp_results.items():
                        df_atleta = dfs_por_periodo[periodo_asp]
                        if atleta != "Todos":
                            df_atleta = df_atleta[df_atleta['Atleta'] == atleta]
                        df_sprints = df_atleta[(df_atleta['Acceleration'] > 0) & (df_atleta['Velocity'] > 0)]
                        if len(df_sprints) > 0:
                            max_a0_real = max(max_a0_real, df_sprints['Acceleration'].max())
                            max_v_real = max(max_v_real, df_sprints['Velocity'].max())
                    
                    fig_asp.update_layout(
                        title=f"<b>Perfil Aceleração-Velocidade - {periodo_asp}</b>",
                        xaxis=dict(
                            title="<b>Velocidade (m/s)</b>", 
                            gridcolor='rgba(255,255,255,0.1)',
                            range=[-0.5, max_v_real * 1.15]
                        ),
                        yaxis=dict(
                            title="<b>Aceleração (m/s²)</b>", 
                            gridcolor='rgba(255,255,255,0.1)',
                            range=[-0.5, max_a0_real * 1.15]
                        ),
                        height=600,
                        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5),
                        plot_bgcolor='rgba(0,0,0,0.2)',
                        paper_bgcolor='rgba(0,0,0,0)',
                        hovermode='closest'
                    )
                    
                    st.plotly_chart(fig_asp, use_container_width=True)
                    
                    st.markdown("### 📊 Métricas do Perfil Aceleração-Velocidade")
                    
                    # Criar DataFrame com métricas reais e teóricas
                    dados_asp = []
                    for atleta, metrics in asp_results.items():
                        df_atleta = dfs_por_periodo[periodo_asp]
                        if atleta != "Todos":
                            df_atleta = df_atleta[df_atleta['Atleta'] == atleta]
                        df_atleta_sprints = df_atleta[(df_atleta['Acceleration'] > 0) & (df_atleta['Velocity'] > 0)]
                        
                        dados_asp.append({
                            'Atleta': atleta,
                            'a₀ Teórico (m/s²)': metrics['a0'],
                            'v₀ Teórico (m/s)': metrics['v0'],
                            'v₀ Teórico (km/h)': metrics['v0'] * 3.6,
                            'aₘₐₓ Real (m/s²)': df_atleta_sprints['Acceleration'].max() if len(df_atleta_sprints) > 0 else 0,
                            'vₘₐₓ Real (m/s)': df_atleta_sprints['Velocity'].max() if len(df_atleta_sprints) > 0 else 0,
                            'vₘₐₓ Real (km/h)': df_atleta_sprints['Velocity'].max() * 3.6 if len(df_atleta_sprints) > 0 else 0,
                            'Pₘₐₓ (W/kg)': metrics['p_max'],
                            'R²': metrics['r2']
                        })
                    
                    asp_df = pd.DataFrame(dados_asp).round(3)
                    
                    def qualidade_r2(r2):
                        if r2 >= 0.8: return '🟢 Excelente'
                        elif r2 >= 0.6: return '🟡 Bom'
                        elif r2 >= 0.4: return '🟠 Moderado'
                        else: return '🔴 Baixo'
                    
                    asp_df['Qualidade'] = asp_df['R²'].apply(qualidade_r2)
                    
                    st.dataframe(asp_df, use_container_width=True)
                    
                    st.markdown("### 🎯 Interpretação dos Resultados")
                    col_asp_metrics1, col_asp_metrics2, col_asp_metrics3, col_asp_metrics4 = st.columns(4)
                    
                    with col_asp_metrics1:
                        a0_medio = asp_df['a₀ Teórico (m/s²)'].mean()
                        st.markdown(f"""
                        <div class="metric-card">
                            <div class="metric-value">{a0_medio:.2f}</div>
                            <div class="metric-label">a₀ Teórico (m/s²)</div>
                            <div style="font-size:0.7rem">⚡ Capacidade teórica de aceleração</div>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    with col_asp_metrics2:
                        a_max_real = asp_df['aₘₐₓ Real (m/s²)'].mean()
                        st.markdown(f"""
                        <div class="metric-card">
                            <div class="metric-value">{a_max_real:.2f}</div>
                            <div class="metric-label">aₘₐₓ Real (m/s²)</div>
                            <div style="font-size:0.7rem">💪 Aceleração máxima atingida</div>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    with col_asp_metrics3:
                        v0_medio = asp_df['v₀ Teórico (km/h)'].mean()
                        st.markdown(f"""
                        <div class="metric-card">
                            <div class="metric-value">{v0_medio:.1f}</div>
                            <div class="metric-label">v₀ Teórico (km/h)</div>
                            <div style="font-size:0.7rem">🏃 Velocidade teórica máxima</div>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    with col_asp_metrics4:
                        v_max_real = asp_df['vₘₐₓ Real (km/h)'].mean()
                        st.markdown(f"""
                        <div class="metric-card">
                            <div class="metric-value">{v_max_real:.1f}</div>
                            <div class="metric-label">vₘₐₓ Real (km/h)</div>
                            <div style="font-size:0.7rem">⚡ Velocidade máxima atingida</div>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    st.markdown('<div class="info-card">', unsafe_allow_html=True)
                    st.info("📊 **Tratamento Estatístico Aplicado:** Filtro IQR para remoção de outliers, regressão robusta RANSAC, limites fisiológicos (a₀: 1.5-6.0 m/s², v₀: 6.0-13.0 m/s, Pₘₐₓ: 4-20 W/kg)\n\n📌 **Interpretação:** A linha vermelha representa a aceleração máxima REAL do atleta, partindo do eixo Y (x=0) no valor da aceleração máxima atingida e indo até a velocidade máxima real. A linha laranja representa a velocidade máxima real do atleta.")
                    st.markdown('</div>', unsafe_allow_html=True)
            
            # TAB 4: PERFORMANCE CARDÍACA
            with tab4:
                st.subheader("❤️ Análise de Performance Cardíaca")
                
                col_fc1, col_fc2 = st.columns(2)
                with col_fc1:
                    periodo_fc = st.selectbox("Selecionar período", options=list(dfs_por_periodo.keys()), key="fc_periodo_select")
                with col_fc2:
                    atleta_fc = st.selectbox("Selecionar atleta", options=["Todos"] + list(dfs_por_periodo[periodo_fc]['Atleta'].unique()), key="atleta_fc_select")
                
                df_fc = dfs_por_periodo[periodo_fc]
                
                if atleta_fc != "Todos":
                    df_fc = df_fc[df_fc['Atleta'] == atleta_fc]
                
                start_dt_fc = df_fc['start_datetime'].iloc[0] if len(df_fc) > 0 else None
                
                fc_max = df_fc['HeartRate'].max()
                trimp_total, zonas_trimp = calcular_trimp_edwards(df_fc, fc_max)
                
                st.markdown("### 📊 Carga Interna (TRIMP)")
                col_trimp1, col_trimp2, col_trimp3 = st.columns(3)
                
                with col_trimp1:
                    st.markdown(f"""
                    <div class="metric-card">
                        <div class="metric-value">{trimp_total:.1f}</div>
                        <div class="metric-label">TRIMP Total</div>
                        <div style="font-size:0.7rem; margin-top:8px;">❤️ Carga interna total</div>
                    </div>
                    """, unsafe_allow_html=True)
                
                with col_trimp2:
                    st.markdown(f"""
                    <div class="metric-card">
                        <div class="metric-value">{fc_max:.0f}</div>
                        <div class="metric-label">FC Máxima (bpm)</div>
                    </div>
                    """, unsafe_allow_html=True)
                
                with col_trimp3:
                    st.markdown(f"""
                    <div class="metric-card">
                        <div class="metric-value">{df_fc['HeartRate'].mean():.0f}</div>
                        <div class="metric-label">FC Média (bpm)</div>
                    </div>
                    """, unsafe_allow_html=True)
                
                st.markdown("---")
                
                st.markdown("### 📈 Distribuição do TRIMP por Zona de Intensidade")
                
                fig_trimp_bar = px.bar(zonas_trimp, x='Zona', y='TRIMP', 
                                       title="TRIMP por Zona de Frequência Cardíaca",
                                       text=zonas_trimp['TRIMP'].apply(lambda x: f'{x:.1f}'),
                                       color='Zona',
                                       color_discrete_sequence=['#2ecc71', '#f1c40f', '#e67e22', '#e74c3c', '#e84393'])
                fig_trimp_bar.update_layout(showlegend=False, height=400)
                st.plotly_chart(fig_trimp_bar, use_container_width=True)
                
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
                
                limiar_anaerobico = fc_max * 0.85
                limiar_aerobico = fc_max * 0.75
                
                fig_fc_acc.add_hrect(y0=limiar_aerobico, y1=limiar_anaerobico, fillcolor="rgba(255,215,0,0.2)", line_width=0, secondary_y=False,
                                     annotation_text="Zona Anaeróbica", annotation_position="bottom right")
                fig_fc_acc.add_hrect(y0=limiar_anaerobico, y1=fc_max, fillcolor="rgba(231,76,60,0.2)", line_width=0, secondary_y=False,
                                     annotation_text="Zona Máxima", annotation_position="top right")
                
                fig_fc_acc.update_layout(
                    title=f"FC vs Aceleração - {atleta_fc if atleta_fc != 'Todos' else 'Múltiplos atletas'} - {periodo_fc}",
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
                    
                    fig_dist_fc.update_layout(title=f"Distância vs FC - {atleta_fc if atleta_fc != 'Todos' else 'Múltiplos atletas'} - {periodo_fc}", height=400)
                    fig_dist_fc.update_yaxes(title_text="Distância (m)", secondary_y=False, color="#2ecc71")
                    fig_dist_fc.update_yaxes(title_text="FC (bpm)", secondary_y=True, color="#ff6b6b")
                    st.plotly_chart(fig_dist_fc, use_container_width=True)
                
                st.markdown("### 📊 Zonas de Intensidade Cardíaca")
                
                df_fc_plot['Zona_FC'] = pd.cut(df_fc_plot['HeartRate'], 
                                          bins=[0, fc_max*0.6, fc_max*0.75, fc_max*0.9, fc_max],
                                          labels=['Recuperação (<60%)', 'Aeróbica (60-75%)', 'Anaeróbica (75-90%)', 'Máximo (>90%)'])
                
                zona_stats = df_fc_plot.groupby('Zona_FC', observed=True).size().reset_index(name='Contagem')
                zona_stats['% do Tempo'] = (zona_stats['Contagem'] / len(df_fc_plot) * 100).round(1)
                if len(df_fc_plot) > 1:
                    sample_rate = df_fc_plot['Seconds'].diff().median()
                    zona_stats['Tempo (min)'] = (zona_stats['Contagem'] * sample_rate / 60).round(1)
                else:
                    zona_stats['Tempo (min)'] = 0
                
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
                    tempo_maximo = zona_stats[zona_stats['Zona_FC'] == 'Máximo (>90%)']['% do Tempo'].values[0] if 'Máximo (>90%)' in zona_stats['Zona_FC'].values else 0
                    st.markdown(f"""
                    <div class="glass-card">
                        <p><b>❤️ FC Máxima:</b> {fc_max:.0f} bpm</p>
                        <p><b>📊 FC Média:</b> {df_fc_plot['HeartRate'].mean():.0f} bpm</p>
                        <p><b>⚡ Limiar Anaeróbico:</b> {limiar_anaerobico:.0f} bpm</p>
                        <p><b>🏃 Tempo em Zona Máxima:</b> {tempo_maximo}%</p>
                        <p><b>📈 TRIMP Total:</b> {trimp_total:.1f} unidades</p>
                    </div>
                    """, unsafe_allow_html=True)
                
                csv_trimp = zonas_trimp.to_csv(index=False)
                st.download_button("📥 Exportar dados de TRIMP", csv_trimp, f"trimp_{atleta_fc}_{periodo_fc}.csv")
            
            # TAB 5: COMPARAÇÃO ESPORTIVA
            with tab5:
                st.subheader("📊 Comparação Esportiva entre Períodos")
                
                col_comp1, col_comp2 = st.columns(2)
                with col_comp1:
                    atleta_comparacao = st.selectbox("Selecionar atleta para comparação", 
                                                     options=["Todos"] + list(dfs_por_periodo[list(dfs_por_periodo.keys())[0]]['Atleta'].unique()), 
                                                     key="atleta_comparacao_select")
                
                var_selecionadas = st.multiselect(
                    "Selecione as variáveis para comparação",
                    options=["Velocidade Média (km/h)", "Velocidade Máxima (km/h)", 
                             "Frequência Cardíaca Média (bpm)", "Frequência Cardíaca Máxima (bpm)", 
                             "Distância Total (m)", "Tempo Total (min)", "TRIMP Total"],
                    default=["Velocidade Média (km/h)", "Distância Total (m)", "Frequência Cardíaca Média (bpm)"],
                    key="var_comparacao"
                )
                
                if var_selecionadas:
                    comparacao_data = []
                    for periodo_nome, df_periodo in dfs_por_periodo.items():
                        df_temp = df_periodo
                        if atleta_comparacao != "Todos":
                            df_temp = df_temp[df_temp['Atleta'] == atleta_comparacao]
                        
                        if len(df_temp) > 0:
                            row = {"Período": periodo_nome, "Atleta": atleta_comparacao if atleta_comparacao != "Todos" else "Todos"}
                            for var in var_selecionadas:
                                if var == "Velocidade Média (km/h)":
                                    row[var] = df_temp['Velocity'].mean() * 3.6
                                elif var == "Velocidade Máxima (km/h)":
                                    row[var] = df_temp['Velocity'].max() * 3.6
                                elif var == "Frequência Cardíaca Média (bpm)":
                                    row[var] = df_temp['HeartRate'].mean()
                                elif var == "Frequência Cardíaca Máxima (bpm)":
                                    row[var] = df_temp['HeartRate'].max()
                                elif var == "Distância Total (m)":
                                    row[var] = distancias_por_periodo.get(periodo_nome, 0)
                                elif var == "Tempo Total (min)":
                                    row[var] = tempos_por_periodo.get(periodo_nome, 0) / 60
                                elif var == "TRIMP Total":
                                    fc_max = df_temp['HeartRate'].max()
                                    trimp_total, _ = calcular_trimp_edwards(df_temp, fc_max)
                                    row[var] = trimp_total
                            comparacao_data.append(row)
                    
                    df_comp = pd.DataFrame(comparacao_data)
                    
                    for var in var_selecionadas:
                        fig_bar = px.bar(df_comp, x='Período', y=var, title=f"<b>{var} - {atleta_comparacao if atleta_comparacao != 'Todos' else 'Todos os atletas'}</b>", 
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
                                text=f"<b>Perfil de Desempenho Normalizado - {atleta_comparacao if atleta_comparacao != 'Todos' else 'Todos os atletas'}</b><br><sup>Comparação entre períodos (quanto maior a área, melhor o desempenho)</sup>",
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
                            f"comparacao_periodos_{atleta_comparacao}.csv",
                            use_container_width=True
                        )
        
        else:
            st.info("👈 Configure os filtros na barra lateral e clique em **PROCESSAR ANÁLISE** para visualizar os resultados.")

else:
    st.markdown("""
    <div style="text-align: center; padding: 50px;">
        <div class="glass-card" style="max-width: 800px; margin: 0 auto;">
            <h2>⚽ ScoutLab - Plataforma de Análise de Performance</h2>
            <p style="font-size: 1.2rem; color: #aaa; margin-top: 20px;">Carregue os arquivos CSV na barra lateral para iniciar a análise</p>
            <div style="display: flex; justify-content: center; gap: 20px; margin-top: 40px;">
                <div class="stat-badge">📊 Análise de posicionamento</div>
                <div class="stat-badge">⚡ Perfil de aceleração (ASP)</div>
                <div class="stat-badge">❤️ Monitoramento cardíaco</div>
                <div class="stat-badge">🎯 Zonas de intensidade</div>
            </div>
            <p style="margin-top: 30px; font-size: 0.9rem; color: #00d2ff;">✅ Dados validados com limites fisiológicos: velocidade máxima 43.2 km/h, aceleração máxima 7.0 m/s²</p>
        </div>
    </div>
    """, unsafe_allow_html=True)