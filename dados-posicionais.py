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
    .zone-selector {
        background: rgba(0, 0, 0, 0.3);
        border-radius: 15px;
        padding: 15px;
        margin-bottom: 20px;
    }
    .warning-card {
        background: rgba(255, 100, 100, 0.2);
        border-left: 4px solid #ff6b6b;
        padding: 10px;
        border-radius: 8px;
        margin: 10px 0;
    }
    .info-card {
        background: rgba(0, 210, 255, 0.1);
        border-left: 4px solid #00d2ff;
        padding: 10px;
        border-radius: 8px;
        margin: 10px 0;
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

# ==================== FUNÇÃO CORRIGIDA DO PERFIL ACELERAÇÃO-VELOCIDADE ====================

def fit_velocidade_aceleracao(velocidades, aceleracoes):
    """
    Ajusta a curva de relação Aceleração-Velocidade baseada no modelo físico correto:
    a(v) = a0 * (1 - v/v0)
    
    Onde:
    - a0: aceleração máxima (intercept)
    - v0: velocidade máxima teórica (onde a aceleração = 0)
    - A inclinação deve ser negativa (a0/v0)
    
    Parâmetros fisiológicos esperados:
    - a0: 2-5 m/s² (aceleração máxima)
    - v0: 8-12 m/s (28.8-43.2 km/h)
    - Pmax = a0 * v0 / 4: 4-15 W/kg
    """
    # Remover valores inválidos
    mask = (velocidades > 0) & (aceleracoes > -5) & (aceleracoes < 10) & (~np.isnan(velocidades)) & (~np.isnan(aceleracoes))
    v_clean = velocidades[mask]
    a_clean = aceleracoes[mask]
    
    if len(v_clean) < 10:
        return None
    
    # Filtrar outliers usando IQR (intervalo interquartil)
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
        # Usar regressão linear robusta com RANSAC para eliminar outliers
        X = v_filtered.reshape(-1, 1)
        y = a_filtered
        
        # Padronizar os dados
        scaler_X = StandardScaler()
        scaler_y = StandardScaler()
        X_scaled = scaler_X.fit_transform(X)
        y_scaled = scaler_y.fit_transform(y.reshape(-1, 1)).ravel()
        
        # RANSAC com parâmetros conservadores
        ransac = RANSACRegressor(
            min_samples=0.5,
            residual_threshold=np.std(y_scaled) * 1.5,
            max_trials=100,
            random_state=42
        )
        ransac.fit(X_scaled, y_scaled)
        
        # Obter coeficientes na escala original
        slope_scaled = ransac.estimator_.coef_[0]
        intercept_scaled = ransac.estimator_.intercept_
        
        # Converter de volta para escala original
        slope = slope_scaled * (scaler_y.scale_ / scaler_X.scale_)
        intercept = scaler_y.mean_ + intercept_scaled * scaler_y.scale_ - slope * scaler_X.mean_
        
        # Calcular R² usando a regressão RANSAC
        y_pred = slope * v_filtered + intercept
        ss_res = np.sum((a_filtered - y_pred) ** 2)
        ss_tot = np.sum((a_filtered - np.mean(a_filtered)) ** 2)
        r2 = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
        
        # Modelo físico: a = a0 - (a0/v0) * v
        # slope = -a0/v0, intercept = a0
        a0 = intercept
        
        # Garantir que a0 seja positivo e fisiológico
        if a0 <= 0 or a0 > 8:
            # Se a0 for inválido, usar o valor máximo de aceleração observado
            a0 = np.percentile(a_filtered[a_filtered > 0], 85) if np.any(a_filtered > 0) else 2.5
            a0 = np.clip(a0, 1.5, 6.0)
        
        # v0 = -a0 / slope (slope deve ser negativo)
        if slope < 0:
            v0 = -a0 / slope
        else:
            # Se a inclinação for positiva (fisicamente inválida), estimar v0 baseado nos dados
            a_positivas = a_filtered[a_filtered > 0.1]
            v_correspondentes = v_filtered[a_filtered > 0.1]
            if len(v_correspondentes) > 0:
                v0 = np.percentile(v_correspondentes, 90) * 1.2
            else:
                v0 = np.max(v_filtered) * 1.1
            
            v0 = np.clip(v0, 6.0, 13.0)
            slope = -a0 / v0
        
        # Limitar v0 a valores fisiológicos
        v0 = np.clip(v0, 6.0, 13.0)
        
        # Calcular potência máxima (Pmax = a0 * v0 / 4)
        p_max = a0 * v0 / 4
        p_max = np.clip(p_max, 4.0, 20.0)
        
        # Velocidade máxima observada
        v_max = np.max(v_filtered)
        
        return {
            'a0': float(a0),
            'v0': float(v0),
            'slope': float(slope),
            'intercept': float(intercept),
            'r2': float(r2),
            'v_max': float(v_max),
            'p_max': float(p_max),
            'n_points': len(v_filtered),
            'se': float(np.sqrt(ss_res / (len(v_filtered) - 2)) if len(v_filtered) > 2 else 0)
        }
    except Exception as e:
        # Fallback para método alternativo
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
                
                v0 = np.clip(v0, 6.0, 13.0)
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
                    'n_points': len(v_z),
                    'se': float(std_err)
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
    
    # Limitar velocidade máxima a valores fisiológicos
    v_max_fisio = 13.0
    v_ms = np.clip(v_ms, 0, v_max_fisio)
    
    result = fit_velocidade_aceleracao(v_ms, a_ms2)
    
    if result:
        result['v_medio'] = np.mean(v_ms)
        result['a_medio'] = np.mean(a_ms2)
        result['num_sprints'] = len(df_sprints)
    
    return result

# ==================== FUNÇÕES DE VALIDAÇÃO ESTATÍSTICA ====================

def validar_dados_gps(df):
    """Valida a integridade e consistência dos dados GPS"""
    validacao = {'alertas': [], 'valores': {}}
    
    if len(df) > 1:
        sample_rates = df['Seconds'].diff().dropna()
        taxa_media = sample_rates.median()
        taxa_std = sample_rates.std()
        validacao['valores']['taxa_amostragem_media'] = taxa_media
        validacao['valores']['taxa_amostragem_std'] = taxa_std
        validacao['valores']['taxa_amostragem_cv'] = taxa_std / taxa_media if taxa_media > 0 else 0
        
        if validacao['valores']['taxa_amostragem_cv'] > 0.1:
            validacao['alertas'].append(f"Alta variabilidade na taxa de amostragem (CV={validacao['valores']['taxa_amostragem_cv']:.2f})")
    
    v_ms = df['Velocity'].values
    v_max_kmh = v_ms.max() * 3.6
    
    validacao['valores']['velocidade_max_kmh'] = v_max_kmh
    validacao['valores']['velocidade_media_kmh'] = np.mean(v_ms) * 3.6
    
    if v_max_kmh > 36:
        validacao['alertas'].append(f"Velocidade máxima ({v_max_kmh:.1f} km/h) acima do limite fisiológico típico (36 km/h)")
    elif v_max_kmh < 20:
        validacao['alertas'].append(f"Velocidade máxima ({v_max_kmh:.1f} km/h) abaixo do esperado para um atleta de futebol")
    
    a_ms2 = df['Acceleration'].values
    a_max = a_ms2.max()
    a_min = a_ms2.min()
    
    validacao['valores']['aceleracao_max'] = a_max
    validacao['valores']['desaceleracao_max'] = a_min
    
    if a_max > 6:
        validacao['alertas'].append(f"Aceleração máxima ({a_max:.2f} m/s²) acima do limite fisiológico típico (6 m/s²)")
    if a_min < -8:
        validacao['alertas'].append(f"Desaceleração máxima ({a_min:.2f} m/s²) abaixo do limite fisiológico típico (-8 m/s²)")
    
    if 'Odometer' in df.columns:
        odometer_diff = df['Odometer'].diff().dropna()
        odometer_negativos = (odometer_diff < -0.1).sum()
        validacao['valores']['odometer_negativos'] = odometer_negativos
        
        if odometer_negativos > 0:
            validacao['alertas'].append(f"{odometer_negativos} leituras do odômetro com valores decrescentes (inconsistência)")
        
        distancia_odometer = df['Odometer'].max() - df['Odometer'].min()
        if len(df) > 1:
            sample_rates = df['Seconds'].diff().median()
            distancia_estimada = np.sum(v_ms * sample_rates)
        else:
            distancia_estimada = 0
        
        validacao['valores']['distancia_odometer'] = distancia_odometer
        validacao['valores']['distancia_estimada'] = distancia_estimada
        validacao['valores']['diferenca_distancia'] = abs(distancia_odometer - distancia_estimada)
        
        if distancia_odometer > 0 and validacao['valores']['diferenca_distancia'] / distancia_odometer > 0.05:
            validacao['alertas'].append(f"Diferença entre odômetro e distância estimada: {validacao['valores']['diferenca_distancia']:.0f}m")
    
    return validacao

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

# ==================== FUNÇÕES PARA SPRINTS ====================

def detectar_sprints(df, limiar_velocidade=25, duracao_minima=1):
    """Detecta sprints baseado em limiar de velocidade e duração mínima"""
    v_ms = df['Velocity'].values
    limiar_ms = limiar_velocidade / 3.6
    
    acima_limiar = v_ms >= limiar_ms
    
    if acima_limiar.sum() == 0:
        return []
    
    sprints = []
    i = 0
    while i < len(acima_limiar):
        if acima_limiar[i]:
            inicio = i
            fim = i
            while fim + 1 < len(acima_limiar) and acima_limiar[fim + 1]:
                fim += 1
            
            duracao = (df['Seconds'].iloc[fim] - df['Seconds'].iloc[inicio])
            
            if duracao >= duracao_minima:
                velocidades_sprint = v_ms[inicio:fim+1]
                aceleracoes_sprint = df['Acceleration'].iloc[inicio:fim+1].values
                
                if 'Odometer' in df.columns:
                    distancia = df['Odometer'].iloc[fim] - df['Odometer'].iloc[inicio]
                else:
                    velocidade_media_ms = np.mean(velocidades_sprint)
                    distancia = velocidade_media_ms * duracao
                
                sprints.append({
                    'inicio_idx': inicio, 'fim_idx': fim,
                    'inicio_time': df['Seconds'].iloc[inicio],
                    'fim_time': df['Seconds'].iloc[fim],
                    'duracao': duracao,
                    'vel_max': np.max(velocidades_sprint) * 3.6,
                    'vel_media': np.mean(velocidades_sprint) * 3.6,
                    'acel_max': np.max(aceleracoes_sprint),
                    'distancia': distancia
                })
            
            i = fim + 1
        else:
            i += 1
    
    return sprints

def detectar_esforcos_alta_intensidade(df, limiar_velocidade=20, duracao_minima=3):
    """Detecta esforços de alta intensidade"""
    v_ms = df['Velocity'].values
    limiar_ms = limiar_velocidade / 3.6
    
    acima_limiar = v_ms >= limiar_ms
    
    if acima_limiar.sum() == 0:
        return []
    
    esforcos = []
    i = 0
    while i < len(acima_limiar):
        if acima_limiar[i]:
            inicio = i
            fim = i
            while fim + 1 < len(acima_limiar) and acima_limiar[fim + 1]:
                fim += 1
            
            duracao = (df['Seconds'].iloc[fim] - df['Seconds'].iloc[inicio])
            
            if duracao >= duracao_minima:
                velocidades = v_ms[inicio:fim+1]
                
                if 'Odometer' in df.columns:
                    distancia = df['Odometer'].iloc[fim] - df['Odometer'].iloc[inicio]
                else:
                    distancia = np.mean(velocidades) * duracao
                
                esforcos.append({
                    'inicio_time': df['Seconds'].iloc[inicio],
                    'fim_time': df['Seconds'].iloc[fim],
                    'duracao': duracao,
                    'vel_max': np.max(velocidades) * 3.6,
                    'vel_media': np.mean(velocidades) * 3.6,
                    'distancia': distancia
                })
            
            i = fim + 1
        else:
            i += 1
    
    return esforcos

# ==================== FUNÇÕES PARA ACELERAÇÕES, DESACELERAÇÕES ====================

def detectar_aceleracoes(df, limiar_aceleracao=2.5, duracao_minima=0.5):
    """Detecta acelerações (aceleração positiva acima do limiar)"""
    aceleracoes = df['Acceleration'].values
    eventos = []
    i = 0
    
    while i < len(aceleracoes):
        if aceleracoes[i] >= limiar_aceleracao:
            inicio = i
            fim = i
            while fim + 1 < len(aceleracoes) and aceleracoes[fim + 1] >= limiar_aceleracao:
                fim += 1
            
            duracao = (df['Seconds'].iloc[fim] - df['Seconds'].iloc[inicio])
            
            if duracao >= duracao_minima:
                eventos.append({
                    'inicio_time': df['Seconds'].iloc[inicio],
                    'fim_time': df['Seconds'].iloc[fim],
                    'duracao': duracao,
                    'pico_aceleracao': np.max(aceleracoes[inicio:fim+1]),
                    'vel_inicio': df['Velocity'].iloc[inicio] * 3.6,
                    'vel_fim': df['Velocity'].iloc[fim] * 3.6
                })
            
            i = fim + 1
        else:
            i += 1
    
    return eventos

def detectar_desaceleracoes(df, limiar_desaceleracao=-2.5, duracao_minima=0.5):
    """Detecta desacelerações (aceleração negativa abaixo do limiar)"""
    aceleracoes = df['Acceleration'].values
    eventos = []
    i = 0
    
    while i < len(aceleracoes):
        if aceleracoes[i] <= limiar_desaceleracao:
            inicio = i
            fim = i
            while fim + 1 < len(aceleracoes) and aceleracoes[fim + 1] <= limiar_desaceleracao:
                fim += 1
            
            duracao = (df['Seconds'].iloc[fim] - df['Seconds'].iloc[inicio])
            
            if duracao >= duracao_minima:
                eventos.append({
                    'inicio_time': df['Seconds'].iloc[inicio],
                    'fim_time': df['Seconds'].iloc[fim],
                    'duracao': duracao,
                    'pico_desaceleracao': np.min(aceleracoes[inicio:fim+1]),
                    'vel_inicio': df['Velocity'].iloc[inicio] * 3.6,
                    'vel_fim': df['Velocity'].iloc[fim] * 3.6
                })
            
            i = fim + 1
        else:
            i += 1
    
    return eventos

def calcular_mudancas_direcao(df, angulo_limiar=45):
    """Calcula mudanças de direção baseado no ângulo entre vetores de movimento"""
    if len(df) < 3 or 'campo_x' not in df.columns or 'campo_y' not in df.columns:
        return []
    
    dx = np.diff(df['campo_x'].values)
    dy = np.diff(df['campo_y'].values)
    mudancas = []
    
    for i in range(1, len(dx) - 1):
        v1 = np.array([dx[i-1], dy[i-1]])
        v2 = np.array([dx[i], dy[i]])
        
        if np.linalg.norm(v1) > 0 and np.linalg.norm(v2) > 0:
            cos_angulo = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
            cos_angulo = np.clip(cos_angulo, -1, 1)
            angulo = np.arccos(cos_angulo) * 180 / np.pi
            
            if angulo >= angulo_limiar:
                mudancas.append({
                    'tempo': df['Seconds'].iloc[i],
                    'angulo': angulo,
                    'velocidade': df['Velocity'].iloc[i] * 3.6
                })
    
    return mudancas

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

# ==================== FUNÇÃO PARA GERAR PDF ====================

def gerar_relatorio_pdf(dfs_por_periodo, distancias_por_periodo, tempos_por_periodo, selected_atletas, periodo_atual):
    buffer = BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=A4, rightMargin=72, leftMargin=72, topMargin=72, bottomMargin=72)
    styles = getSampleStyleSheet()
    story = []
    
    title_style = ParagraphStyle('CustomTitle', parent=styles['Heading1'], fontSize=24,
                                  textColor=colors.HexColor('#00d2ff'), alignment=TA_CENTER, spaceAfter=30)
    story.append(Paragraph("ScoutLab - Relatório de Performance Esportiva", title_style))
    story.append(Spacer(1, 12))
    
    story.append(Paragraph(f"Atleta: {selected_atletas[0] if selected_atletas else 'Não identificado'}", styles['Heading2']))
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
    story.append(Spacer(1, 20))
    
    story.append(Paragraph("Sprints e Esforços de Alta Intensidade", styles['Heading2']))
    sprints = detectar_sprints(df_metricas, limiar_velocidade=25, duracao_minima=1)
    esforcos_ai = detectar_esforcos_alta_intensidade(df_metricas, limiar_velocidade=20, duracao_minima=3)
    
    sprint_data = [
        ['Métrica', 'Valor'],
        ['Número de Sprints', str(len(sprints))],
        ['Distância em Sprints', f"{sum(s['distancia'] for s in sprints):.0f} m"],
        ['Velocidade Máxima em Sprint', f"{max([s['vel_max'] for s in sprints], default=0):.1f} km/h"],
        ['Número de Esforços AI', str(len(esforcos_ai))],
        ['Distância em Esforços AI', f"{sum(e['distancia'] for e in esforcos_ai):.0f} m"],
    ]
    
    sprint_table = Table(sprint_data, colWidths=[200, 200])
    sprint_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#00d2ff')),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('GRID', (0, 0), (-1, -1), 1, colors.black)
    ]))
    story.append(sprint_table)
    story.append(Spacer(1, 20))
    
    story.append(Paragraph("Acelerações e Desacelerações", styles['Heading2']))
    aceleracoes = detectar_aceleracoes(df_metricas, limiar_aceleracao=2.5, duracao_minima=0.5)
    desaceleracoes = detectar_desaceleracoes(df_metricas, limiar_desaceleracao=-2.5, duracao_minima=0.5)
    
    acc_data = [
        ['Métrica', 'Valor'],
        ['Número de Acelerações', str(len(aceleracoes))],
        ['Pico de Aceleração', f"{max([a['pico_aceleracao'] for a in aceleracoes], default=0):.2f} m/s²"],
        ['Número de Desacelerações', str(len(desaceleracoes))],
        ['Pico de Desaceleração', f"{min([d['pico_desaceleracao'] for d in desaceleracoes], default=0):.2f} m/s²"],
    ]
    
    acc_table = Table(acc_data, colWidths=[200, 200])
    acc_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#00d2ff')),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('GRID', (0, 0), (-1, -1), 1, colors.black)
    ]))
    story.append(acc_table)
    
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
        st.sidebar.header("⚡ 4. Zonas de Velocidade")
        
        st.sidebar.markdown('<div class="zone-selector">', unsafe_allow_html=True)
        
        if 'velocidade_zonas' not in st.session_state:
            st.session_state.velocidade_zonas = {
                'Z1 (Caminhada)': (0, 7),
                'Z2 (Trote)': (7, 15),
                'Z3 (Corrida Moderada)': (15, 21),
                'Z4 (Alta Intensidade)': (21, 25),
                'Z5 (Sprint)': (25, 100)
            }
        
        st.sidebar.markdown("**Defina os limites das zonas (km/h):**")
        
        zonas_editadas = {}
        for zona, (min_val, max_val) in st.session_state.velocidade_zonas.items():
            col1, col2 = st.sidebar.columns(2)
            with col1:
                novo_min = st.number_input(f"{zona} - min", value=float(min_val), step=1.0, key=f"zona_min_{zona}", format="%.0f")
            with col2:
                novo_max = st.number_input(f"{zona} - max", value=float(max_val), step=1.0, key=f"zona_max_{zona}", format="%.0f")
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
        
        # 6. Seleção de Atletas
        st.sidebar.markdown("---")
        st.sidebar.header("🏅 6. Selecionar Atleta(s)")
        
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
        
        # 7. Opções de Visualização
        st.sidebar.markdown("---")
        st.sidebar.header("🎨 7. Opções de Visualização")
        show_field = st.sidebar.checkbox("Mostrar campo", value=True, key="show_field")
        
        # 8. Seleção de Períodos para Análise
        st.sidebar.markdown("---")
        st.sidebar.header("📊 8. Selecionar Períodos para Análise")
        
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
            st.session_state.analise_processada = True
        
        # ==================== EXIBIÇÃO DAS ANÁLISES ====================
        if st.session_state.get('analise_processada', False):
            dfs_por_periodo = st.session_state.dfs_por_periodo
            distancias_por_periodo = st.session_state.distancias_por_periodo
            tempos_por_periodo = st.session_state.tempos_por_periodo
            selected_atletas = st.session_state.selected_atletas
            
            # Botão para gerar relatório PDF
            col_pdf1, col_pdf2, col_pdf3 = st.columns([1, 2, 1])
            with col_pdf2:
                if st.button("📄 GERAR RELATÓRIO PDF", type="primary", use_container_width=True):
                    periodo_atual = list(dfs_por_periodo.keys())[0]
                    pdf_buffer = gerar_relatorio_pdf(dfs_por_periodo, distancias_por_periodo, tempos_por_periodo, selected_atletas, periodo_atual)
                    st.download_button(
                        label="📥 BAIXAR RELATÓRIO PDF",
                        data=pdf_buffer,
                        file_name=f"scoutlab_relatorio_{selected_atletas[0]}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf",
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
                st.markdown(f"**{periodo_nome}** | Tempo: {tempo_min:.1f} min | {len(df_periodo):,} registros")
                st.markdown("---")
            
            # ==================== ABAS ====================
            tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
                "🗺️ Mapa Tático", 
                "📐 Análise por Zonas", 
                "⚡ Perfil Aceleração-Velocidade",
                "❤️ Performance Cardíaca",
                "🚀 Sprints e Alta Intensidade",
                "🔄 Acelerações, Desacelerações e Mudanças de Direção",
                "📊 Comparação Esportiva"
            ])
            
            # TAB 1: Mapa Tático (simplificado por limite de espaço)
            with tab1:
                st.subheader("🗺️ Mapa Tático de Posicionamento")
                periodo_mapa = st.selectbox("Selecionar período", options=list(dfs_por_periodo.keys()), key="periodo_mapa_select")
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
                                      title=f"Trajetória de {selected_atletas[0]} - {periodo_mapa}")
                st.plotly_chart(fig_map, use_container_width=True)
            
            # TAB 2: Análise por Zonas (simplificado)
            with tab2:
                st.subheader("📐 Análise Tática por Zonas do Campo")
                st.markdown(f"Campo com dimensões oficiais: **{CAMPO_COMPRIMENTO}m x {CAMPO_LARGURA}m**")
                
                periodo_tatica = st.selectbox("Selecionar período", options=list(dfs_por_periodo.keys()), key="periodo_tatica_select")
                df_tat = dfs_por_periodo[periodo_tatica]
                
                bounds = st.session_state.get('bounds_estadio', None)
                if not bounds:
                    st.warning("⚠️ Limites do estádio não definidos.")
                    st.stop()
                
                if len(df_tat) > 3000:
                    df_tat_sample = df_tat.sample(3000, random_state=42)
                else:
                    df_tat_sample = df_tat
                
                campo_x, campo_y = [], []
                for _, row in df_tat_sample.iterrows():
                    x, y = converter_gps_para_campo_cached(row['Latitude'], row['Longitude'], bounds)
                    campo_x.append(x); campo_y.append(y)
                df_tat_sample['campo_x'], df_tat_sample['campo_y'] = campo_x, campo_y
                
                col_lin, col_col = st.columns(2)
                with col_lin:
                    num_linhas = st.number_input("Linhas", 1, 8, 3, key="num_linhas")
                with col_col:
                    num_colunas = st.number_input("Colunas", 1, 8, 3, key="num_colunas")
                
                linhas_bins = np.linspace(X_MIN, X_MAX, num_linhas + 1)
                colunas_bins = np.linspace(Y_MIN, Y_MAX, num_colunas + 1)
                
                df_tat_sample['Zona_Linha'] = pd.cut(df_tat_sample['campo_x'], bins=linhas_bins, labels=[f'L{i+1}' for i in range(num_linhas)], include_lowest=True)
                df_tat_sample['Zona_Coluna'] = pd.cut(df_tat_sample['campo_y'], bins=colunas_bins, labels=[f'C{i+1}' for i in range(num_colunas)], include_lowest=True)
                df_tat_sample['Zona'] = df_tat_sample['Zona_Linha'].astype(str) + '-' + df_tat_sample['Zona_Coluna'].astype(str)
                
                zona_metrics = df_tat_sample.groupby('Zona', observed=True).agg({
                    'Seconds': 'count', 'Velocity': ['mean', 'max']
                }).round(2)
                zona_metrics.columns = ['Contagem', 'Vel_Média', 'Vel_Máx']
                zona_metrics['Vel_Média'] = zona_metrics['Vel_Média'] * 3.6
                zona_metrics['Vel_Máx'] = zona_metrics['Vel_Máx'] * 3.6
                
                total_contagem = zona_metrics['Contagem'].sum()
                zona_metrics['% Frequência'] = (zona_metrics['Contagem'] / total_contagem * 100).round(1) if total_contagem > 0 else 0
                
                fig_tat = go.Figure()
                for shape in desenhar_campo_futebol():
                    fig_tat.add_shape(shape)
                
                shapes_div, linhas_bins_plot, colunas_bins_plot = desenhar_linhas_divisorias(num_linhas, num_colunas)
                for shape in shapes_div:
                    fig_tat.add_shape(shape)
                
                heatmap = np.zeros((num_linhas, num_colunas))
                for i in range(num_linhas):
                    for j in range(num_colunas):
                        zona = f'L{i+1}-C{j+1}'
                        if zona in zona_metrics.index:
                            heatmap[i, j] = zona_metrics.loc[zona, 'Contagem']
                
                fig_tat.add_trace(go.Heatmap(x=linhas_bins_plot, y=colunas_bins_plot, z=heatmap.T, colorscale='Hot', opacity=0.7,
                                             colorbar=dict(title="Frequência")))
                fig_tat.add_trace(go.Scatter(x=df_tat_sample['campo_x'], y=df_tat_sample['campo_y'], mode='markers',
                                             marker=dict(size=2, color='white', opacity=0.5), name='Trajetória'))
                
                fig_tat.update_layout(title=f"Análise Tática - {selected_atletas[0]}", height=600,
                                      xaxis_title="Posição (m)", yaxis_title="Posição (m)",
                                      xaxis=dict(scaleanchor="y", scaleratio=1, range=[X_MIN-2, X_MAX+2]),
                                      yaxis=dict(range=[Y_MIN-2, Y_MAX+2]), plot_bgcolor='rgba(34,139,34,0.2)')
                st.plotly_chart(fig_tat, use_container_width=True)
                
                st.markdown("### 📊 Demanda Física por Zona")
                st.dataframe(zona_metrics[['Contagem', '% Frequência', 'Vel_Média', 'Vel_Máx']].style.format({
                    'Contagem': '{:.0f}', '% Frequência': '{:.1f}%', 'Vel_Média': '{:.1f}', 'Vel_Máx': '{:.1f}'
                }), use_container_width=True)
            
            # TAB 3: Perfil Aceleração-Velocidade (CORRIGIDO)
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
                
                periodo_asp = st.selectbox("Selecionar período", options=list(dfs_por_periodo.keys()), key="asp_periodo_select")
                df_asp_atual = dfs_por_periodo[periodo_asp]
                
                asp_results = {}
                with st.spinner("Calculando perfil ASP com tratamento estatístico robusto..."):
                    for nome, df_periodo in dfs_por_periodo.items():
                        asp_metrics = calcular_asp_metrics(df_periodo)
                        if asp_metrics:
                            asp_results[nome] = asp_metrics
                
                if not asp_results:
                    st.warning("⚠️ Dados insuficientes para calcular o Perfil Aceleração-Velocidade. São necessários pelo menos 10 pontos com aceleração positiva e velocidade > 0.")
                else:
                    st.markdown("### 📈 Curva Aceleração-Velocidade (ASP)")
                    st.markdown("**Relação entre Velocidade (m/s) e Aceleração (m/s²) - Modelo Físico**")
                    
                    metrics = asp_results.get(periodo_asp)
                    if metrics:
                        df_sprints = dfs_por_periodo[periodo_asp]
                        df_sprints = df_sprints[(df_sprints['Acceleration'] > 0) & (df_sprints['Velocity'] > 0)].copy()
                        v_ms = df_sprints['Velocity'].values
                        a_ms2 = df_sprints['Acceleration'].values
                        
                        fig_asp = go.Figure()
                        
                        fig_asp.add_trace(go.Scatter(
                            x=v_ms, y=a_ms2,
                            mode='markers',
                            name='Dados brutos',
                            marker=dict(size=6, color='rgba(0, 210, 255, 0.6)', symbol='circle', line=dict(width=1, color='white')),
                            hovertemplate='Vel: %{x:.2f} m/s<br>Acel: %{y:.2f} m/s²<extra></extra>'
                        ))
                        
                        v_fit = np.linspace(0, min(metrics['v0'], metrics['v_max'] * 1.1), 100)
                        a_fit = metrics['a0'] * (1 - v_fit / metrics['v0'])
                        
                        fig_asp.add_trace(go.Scatter(
                            x=v_fit, y=a_fit,
                            mode='lines',
                            name='Modelo ASP',
                            line=dict(color='red', width=3),
                            hovertemplate='Modelo: a = {a0:.2f} × (1 - v/{v0:.2f})<br>v: %{{x:.2f}} m/s<br>a: %{{y:.2f}} m/s²<extra></extra>'.format(
                                a0=metrics['a0'], v0=metrics['v0']
                            )
                        ))
                        
                        fig_asp.add_trace(go.Scatter(
                            x=[0], y=[metrics['a0']],
                            mode='markers+text',
                            marker=dict(size=14, color='green', symbol='circle', line=dict(width=2, color='white')),
                            text=[f"a₀ = {metrics['a0']:.2f} m/s²"],
                            textposition='top right',
                            name='Aceleração Máxima'
                        ))
                        
                        fig_asp.add_trace(go.Scatter(
                            x=[metrics['v0']], y=[0],
                            mode='markers+text',
                            marker=dict(size=14, color='orange', symbol='circle', line=dict(width=2, color='white')),
                            text=[f"v₀ = {metrics['v0']:.2f} m/s"],
                            textposition='bottom right',
                            name='Velocidade Máxima Teórica'
                        ))
                        
                        equation_text = f"a(v) = {metrics['a0']:.2f} × (1 - v/{metrics['v0']:.2f})<br>R² = {metrics['r2']:.3f}<br>Pₘₐₓ = {metrics['p_max']:.2f} W/kg"
                        
                        fig_asp.add_annotation(
                            x=0.95, y=0.95, xref="paper", yref="paper",
                            text=equation_text, showarrow=False,
                            font=dict(size=12, color='white', family='monospace'),
                            bgcolor='rgba(0,0,0,0.6)', borderpad=10, bordercolor='white', borderwidth=1
                        )
                        
                        fig_asp.update_layout(
                            title=f"<b>Perfil Aceleração-Velocidade - {periodo_asp}</b>",
                            xaxis=dict(title="<b>Velocidade (m/s)</b>", gridcolor='rgba(255,255,255,0.1)', range=[0, metrics['v0'] * 1.05]),
                            yaxis=dict(title="<b>Aceleração (m/s²)</b>", gridcolor='rgba(255,255,255,0.1)', range=[0, metrics['a0'] * 1.1]),
                            height=550, legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5),
                            plot_bgcolor='rgba(0,0,0,0.2)', paper_bgcolor='rgba(0,0,0,0)'
                        )
                        
                        st.plotly_chart(fig_asp, use_container_width=True)
                        
                        st.markdown("### 📊 Métricas do Perfil Aceleração-Velocidade")
                        
                        asp_df = pd.DataFrame(asp_results).T
                        asp_df = asp_df[['a0', 'v0', 'v_max', 'p_max', 'r2']].round(3)
                        asp_df.columns = ['a₀ (m/s²)', 'v₀ (m/s)', 'vₘₐₓ (m/s)', 'Pₘₐₓ (W/kg)', 'R²']
                        asp_df['v₀ (km/h)'] = (asp_df['v₀ (m/s)'] * 3.6).round(1)
                        asp_df['vₘₐₓ (km/h)'] = (asp_df['vₘₐₓ (m/s)'] * 3.6).round(1)
                        
                        def qualidade_r2(r2):
                            if r2 >= 0.8: return '🟢 Excelente'
                            elif r2 >= 0.6: return '🟡 Bom'
                            elif r2 >= 0.4: return '🟠 Moderado'
                            else: return '🔴 Baixo'
                        
                        asp_df['Qualidade'] = asp_df['R²'].apply(qualidade_r2)
                        
                        st.dataframe(asp_df[['a₀ (m/s²)', 'v₀ (m/s)', 'v₀ (km/h)', 'vₘₐₓ (km/h)', 'Pₘₐₓ (W/kg)', 'R²', 'Qualidade']], use_container_width=True)
                        
                        # Cards de interpretação
                        st.markdown("### 🎯 Interpretação dos Resultados")
                        col_asp1, col_asp2, col_asp3 = st.columns(3)
                        
                        with col_asp1:
                            a0_medio = asp_df['a₀ (m/s²)'].mean()
                            st.markdown(f"""
                            <div class="metric-card">
                                <div class="metric-value">{a0_medio:.2f}</div>
                                <div class="metric-label">Aceleração Máxima (m/s²)</div>
                                <div style="font-size:0.7rem">⚡ Capacidade de aceleração inicial</div>
                            </div>
                            """, unsafe_allow_html=True)
                        
                        with col_asp2:
                            v0_medio = asp_df['v₀ (m/s)'].mean()
                            st.markdown(f"""
                            <div class="metric-card">
                                <div class="metric-value">{v0_medio:.2f}</div>
                                <div class="metric-label">Velocidade Máx Teórica (m/s)</div>
                                <div style="font-size:0.7rem">🏃 {v0_medio * 3.6:.1f} km/h</div>
                            </div>
                            """, unsafe_allow_html=True)
                        
                        with col_asp3:
                            p_max_medio = asp_df['Pₘₐₓ (W/kg)'].mean()
                            st.markdown(f"""
                            <div class="metric-card">
                                <div class="metric-value">{p_max_medio:.2f}</div>
                                <div class="metric-label">Potência Máxima (W/kg)</div>
                                <div style="font-size:0.7rem">💪 Explosividade muscular</div>
                            </div>
                            """, unsafe_allow_html=True)
                        
                        # Info sobre correções aplicadas
                        st.markdown('<div class="info-card">', unsafe_allow_html=True)
                        st.info("📊 **Tratamento Estatístico Aplicado:** Filtro IQR para remoção de outliers, regressão robusta RANSAC, limites fisiológicos (a₀: 1.5-6.0 m/s², v₀: 6.0-13.0 m/s, Pₘₐₓ: 4-20 W/kg)")
                        st.markdown('</div>', unsafe_allow_html=True)
            
            # TAB 4: Performance Cardíaca com TRIMP
            with tab4:
                st.subheader("❤️ Análise de Performance Cardíaca")
                
                periodo_fc = st.selectbox("Selecionar período", options=list(dfs_por_periodo.keys()), key="fc_periodo_select")
                df_fc = dfs_por_periodo[periodo_fc]
                start_dt_fc = df_fc['start_datetime'].iloc[0] if len(df_fc) > 0 else None
                
                fc_max = df_fc['HeartRate'].max()
                trimp_total, zonas_trimp = calcular_trimp_edwards(df_fc, fc_max)
                
                st.markdown("### 📊 Carga Interna (TRIMP - Edwards)")
                col_trimp1, col_trimp2, col_trimp3 = st.columns(3)
                with col_trimp1:
                    st.metric("TRIMP Total", f"{trimp_total:.1f}")
                with col_trimp2:
                    st.metric("FC Máxima", f"{fc_max:.0f} bpm")
                with col_trimp3:
                    st.metric("FC Média", f"{df_fc['HeartRate'].mean():.0f} bpm")
                
                fig_trimp_bar = px.bar(zonas_trimp, x='Zona', y='TRIMP', title="TRIMP por Zona de Frequência Cardíaca",
                                       text=zonas_trimp['TRIMP'].apply(lambda x: f'{x:.1f}'), color='Zona')
                fig_trimp_bar.update_layout(showlegend=False, height=400)
                st.plotly_chart(fig_trimp_bar, use_container_width=True)
                
                if len(df_fc) > 2000:
                    df_fc_plot = df_fc.sample(2000, random_state=42).sort_values('Seconds')
                else:
                    df_fc_plot = df_fc
                
                df_fc_plot['Horario'] = df_fc_plot['Seconds'].apply(lambda x: seconds_to_time_str(x, start_dt_fc))
                
                fig_fc = go.Figure()
                fig_fc.add_trace(go.Scatter(x=df_fc_plot['Horario'], y=df_fc_plot['HeartRate'], mode='lines', name='FC',
                                            line=dict(color='#ff6b6b', width=2), fill='tozeroy'))
                
                limiar_anaerobico = fc_max * 0.85
                limiar_aerobico = fc_max * 0.75
                
                fig_fc.add_hrect(y0=limiar_aerobico, y1=limiar_anaerobico, fillcolor="rgba(255,215,0,0.2)", line_width=0,
                                 annotation_text="Zona Anaeróbica", annotation_position="bottom right")
                fig_fc.add_hrect(y0=limiar_anaerobico, y1=fc_max, fillcolor="rgba(231,76,60,0.2)", line_width=0,
                                 annotation_text="Zona Máxima", annotation_position="top right")
                
                fig_fc.update_layout(title=f"FC durante o período - {periodo_fc}", height=450, hovermode='x unified')
                fig_fc.update_yaxes(title_text="Frequência Cardíaca (bpm)")
                st.plotly_chart(fig_fc, use_container_width=True)
                
                st.markdown("### 📊 Zonas de Intensidade Cardíaca")
                
                df_fc_plot['Zona_FC'] = pd.cut(df_fc_plot['HeartRate'], 
                                          bins=[0, fc_max*0.6, fc_max*0.75, fc_max*0.9, fc_max],
                                          labels=['Recuperação', 'Aeróbica', 'Anaeróbica', 'Máximo'])
                
                zona_stats = df_fc_plot.groupby('Zona_FC', observed=True).size().reset_index(name='Contagem')
                zona_stats['% do Tempo'] = (zona_stats['Contagem'] / len(df_fc_plot) * 100).round(1)
                
                fig_zona_bar = px.bar(zona_stats, x='Zona_FC', y='% do Tempo', title="Distribuição do Tempo por Zona",
                                      text=zona_stats['% do Tempo'].apply(lambda x: f'{x}%'), color='Zona_FC')
                fig_zona_bar.update_layout(showlegend=False, height=400)
                st.plotly_chart(fig_zona_bar, use_container_width=True)
                
                csv_trimp = zonas_trimp.to_csv(index=False)
                st.download_button("📥 Exportar dados de TRIMP", csv_trimp, f"trimp_{selected_atletas[0]}_{periodo_fc}.csv")
            
            # TAB 5: Sprints e Alta Intensidade
            with tab5:
                st.subheader("🚀 Sprints e Esforços de Alta Intensidade")
                
                periodo_sprint = st.selectbox("Selecionar período", options=list(dfs_por_periodo.keys()), key="periodo_sprint_select")
                df_sprint = dfs_por_periodo[periodo_sprint]
                start_dt_sprint = df_sprint['start_datetime'].iloc[0] if len(df_sprint) > 0 else None
                
                validacao = validar_dados_gps(df_sprint)
                if validacao['alertas']:
                    for alerta in validacao['alertas']:
                        st.warning(f"⚠️ {alerta}")
                
                col_param1, col_param2 = st.columns(2)
                with col_param1:
                    limiar_sprint = st.number_input("Limiar para Sprint (km/h)", value=25.0, step=1.0, min_value=10.0, max_value=40.0, key="limiar_sprint")
                    duracao_min_sprint = st.number_input("Duração mínima do Sprint (s)", value=1.0, step=0.5, min_value=0.5, key="duracao_sprint")
                with col_param2:
                    limiar_ai = st.number_input("Limiar para Alta Intensidade (km/h)", value=20.0, step=1.0, min_value=10.0, max_value=35.0, key="limiar_ai")
                    duracao_min_ai = st.number_input("Duração mínima AI (s)", value=3.0, step=0.5, min_value=1.0, key="duracao_ai")
                
                v_max_kmh = df_sprint['Velocity'].max() * 3.6
                if limiar_sprint > v_max_kmh:
                    st.warning(f"⚠️ Limiar de sprint ({limiar_sprint} km/h) > velocidade máxima ({v_max_kmh:.1f} km/h). Nenhum sprint será detectado.")
                if limiar_ai > v_max_kmh:
                    st.warning(f"⚠️ Limiar de AI ({limiar_ai} km/h) > velocidade máxima ({v_max_kmh:.1f} km/h). Nenhum esforço será detectado.")
                
                with st.spinner("Detectando sprints..."):
                    sprints = detectar_sprints(df_sprint, limiar_sprint, duracao_min_sprint)
                    esforcos_ai = detectar_esforcos_alta_intensidade(df_sprint, limiar_ai, duracao_min_ai)
                
                col_s1, col_s2, col_s3, col_s4 = st.columns(4)
                with col_s1:
                    st.metric("Total Sprints", len(sprints))
                with col_s2:
                    st.metric("Distância Sprints", f"{sum(s['distancia'] for s in sprints):.0f} m")
                with col_s3:
                    st.metric("Esforços AI", len(esforcos_ai))
                with col_s4:
                    st.metric("Distância AI", f"{sum(e['distancia'] for e in esforcos_ai):.0f} m")
                
                if sprints:
                    st.markdown("### 📋 Detalhamento dos Sprints")
                    df_sprints = pd.DataFrame(sprints)
                    df_sprints['inicio_time'] = df_sprints['inicio_time'].apply(lambda x: seconds_to_time_str(x, start_dt_sprint))
                    df_sprints['fim_time'] = df_sprints['fim_time'].apply(lambda x: seconds_to_time_str(x, start_dt_sprint))
                    st.dataframe(df_sprints[['inicio_time', 'fim_time', 'duracao', 'vel_max', 'vel_media', 'distancia']], use_container_width=True)
                    
                    fig_sprint = go.Figure()
                    fig_sprint.add_trace(go.Scatter(x=df_sprint['Seconds'], y=df_sprint['Velocity'] * 3.6, mode='lines', name='Velocidade', line=dict(color='#00d2ff')))
                    for sprint in sprints:
                        fig_sprint.add_vrect(x0=sprint['inicio_time'], x1=sprint['fim_time'], fillcolor="rgba(255,0,0,0.3)", line_width=0)
                    fig_sprint.add_hline(y=limiar_sprint, line_dash="dash", line_color="red")
                    fig_sprint.update_layout(title="Perfil de Velocidade com Sprints", height=500)
                    st.plotly_chart(fig_sprint, use_container_width=True)
                
                if sprints:
                    csv_sprints = pd.DataFrame(sprints).to_csv(index=False)
                    st.download_button("📥 Exportar Sprints", csv_sprints, f"sprints_{selected_atletas[0]}_{periodo_sprint}.csv")
            
            # TAB 6: Acelerações, Desacelerações e Mudanças de Direção
            with tab6:
                st.subheader("🔄 Acelerações, Desacelerações e Mudanças de Direção")
                
                periodo_acc = st.selectbox("Selecionar período", options=list(dfs_por_periodo.keys()), key="periodo_acc_select")
                df_acc = dfs_por_periodo[periodo_acc]
                start_dt_acc = df_acc['start_datetime'].iloc[0] if len(df_acc) > 0 else None
                
                col_acc1, col_acc2, col_acc3 = st.columns(3)
                with col_acc1:
                    limiar_aceleracao = st.number_input("Limiar Aceleração (m/s²)", value=2.5, step=0.5, key="limiar_aceleracao")
                with col_acc2:
                    limiar_desaceleracao = st.number_input("Limiar Desaceleração (m/s²)", value=-2.5, step=0.5, key="limiar_desaceleracao")
                with col_acc3:
                    angulo_limiar = st.number_input("Ângulo Mudança Direção (°)", value=45, step=5, key="angulo_limiar")
                
                aceleracoes = detectar_aceleracoes(df_acc, limiar_aceleracao, duracao_minima=0.5)
                desaceleracoes = detectar_desaceleracoes(df_acc, limiar_desaceleracao, duracao_minima=0.5)
                
                bounds = st.session_state.get('bounds_estadio', None)
                if bounds and 'campo_x' not in df_acc.columns:
                    campo_x, campo_y = [], []
                    for _, row in df_acc.iterrows():
                        x, y = converter_gps_para_campo_cached(row['Latitude'], row['Longitude'], bounds)
                        campo_x.append(x); campo_y.append(y)
                    df_acc = df_acc.copy()
                    df_acc['campo_x'] = campo_x
                    df_acc['campo_y'] = campo_y
                
                mudancas_direcao = calcular_mudancas_direcao(df_acc, angulo_limiar) if bounds else []
                
                col_a1, col_a2, col_a3 = st.columns(3)
                with col_a1:
                    st.metric("Acelerações", len(aceleracoes))
                with col_a2:
                    st.metric("Desacelerações", len(desaceleracoes))
                with col_a3:
                    st.metric("Mudanças Direção", len(mudancas_direcao))
                
                fig_acc = go.Figure()
                fig_acc.add_trace(go.Scatter(x=df_acc['Seconds'], y=df_acc['Acceleration'], mode='lines', name='Aceleração', line=dict(color='#4ecdc4')))
                fig_acc.add_hline(y=limiar_aceleracao, line_dash="dash", line_color="green")
                fig_acc.add_hline(y=limiar_desaceleracao, line_dash="dash", line_color="red")
                fig_acc.update_layout(title="Perfil de Aceleração/Desaceleração", height=500)
                st.plotly_chart(fig_acc, use_container_width=True)
                
                if aceleracoes:
                    df_aceleracoes = pd.DataFrame(aceleracoes)
                    df_aceleracoes['inicio_time'] = df_aceleracoes['inicio_time'].apply(lambda x: seconds_to_time_str(x, start_dt_acc))
                    st.dataframe(df_aceleracoes[['inicio_time', 'duracao', 'pico_aceleracao', 'vel_inicio', 'vel_fim']], use_container_width=True)
                
                if desaceleracoes:
                    df_desaceleracoes = pd.DataFrame(desaceleracoes)
                    df_desaceleracoes['inicio_time'] = df_desaceleracoes['inicio_time'].apply(lambda x: seconds_to_time_str(x, start_dt_acc))
                    st.dataframe(df_desaceleracoes[['inicio_time', 'duracao', 'pico_desaceleracao', 'vel_inicio', 'vel_fim']], use_container_width=True)
            
            # TAB 7: Comparação Esportiva
            with tab7:
                st.subheader("📊 Comparação Esportiva entre Períodos")
                
                var_selecionadas = st.multiselect(
                    "Selecione as variáveis para comparação",
                    options=["Velocidade Média (km/h)", "Velocidade Máxima (km/h)", 
                             "Frequência Cardíaca Média (bpm)", "Frequência Cardíaca Máxima (bpm)", 
                             "Distância Total (m)", "TRIMP Total"],
                    default=["Velocidade Média (km/h)", "Distância Total (m)", "TRIMP Total"]
                )
                
                if var_selecionadas:
                    comparacao_data = []
                    for periodo_nome, df_periodo in dfs_por_periodo.items():
                        row = {"Período": periodo_nome}
                        for var in var_selecionadas:
                            if var == "Velocidade Média (km/h)":
                                row[var] = df_periodo['Velocity'].mean() * 3.6
                            elif var == "Velocidade Máxima (km/h)":
                                row[var] = df_periodo['Velocity'].max() * 3.6
                            elif var == "Frequência Cardíaca Média (bpm)":
                                row[var] = df_periodo['HeartRate'].mean()
                            elif var == "Frequência Cardíaca Máxima (bpm)":
                                row[var] = df_periodo['HeartRate'].max()
                            elif var == "Distância Total (m)":
                                row[var] = distancias_por_periodo.get(periodo_nome, 0)
                            elif var == "TRIMP Total":
                                trimp_total, _ = calcular_trimp_edwards(df_periodo, df_periodo['HeartRate'].max())
                                row[var] = trimp_total
                        comparacao_data.append(row)
                    
                    df_comp = pd.DataFrame(comparacao_data)
                    
                    for var in var_selecionadas:
                        fig_bar = px.bar(df_comp, x='Período', y=var, title=f"<b>{var}</b>", text_auto='.1f')
                        fig_bar.update_layout(height=400, showlegend=False)
                        st.plotly_chart(fig_bar, use_container_width=True)
                    
                    csv_comp = df_comp.to_csv(index=False)
                    st.download_button("📥 Exportar comparação", csv_comp, "comparacao_periodos.csv")
        
        else:
            st.info("👈 Configure os filtros na barra lateral e clique em **PROCESSAR ANÁLISE** para visualizar os resultados.")

else:
    st.markdown("""
    <div style="text-align: center; padding: 50px;">
        <h2>⚽ ScoutLab - Plataforma de Análise de Performance</h2>
        <p style="font-size: 1.2rem; color: #aaa;">Carregue os arquivos CSV na barra lateral para iniciar a análise</p>
        <p style="margin-top: 30px;">📊 Análise de posicionamento | ⚡ Perfil de aceleração (ASP corrigido) | ❤️ Monitoramento cardíaco | 🎯 Zonas de intensidade | 🚀 Detecção de sprints | 🔄 Acelerações/Desacelerações</p>
        <p style="margin-top: 15px; font-size: 0.9rem; color: #00d2ff;">✅ Tratamento estatístico robusto com RANSAC e limites fisiológicos</p>
    </div>
    """, unsafe_allow_html=True)