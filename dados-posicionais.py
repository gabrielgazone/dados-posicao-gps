import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import numpy as np
from io import StringIO

# Configuração da página
st.set_page_config(
    page_title="Análise de Percurso do Atleta",
    page_icon="🏃",
    layout="wide"
)

# Título do app
st.title("🏃 Análise de Percurso do Atleta durante o Jogo")
st.markdown("---")

# Função para carregar e processar os dados (VERSÃO CORRIGIDA)
@st.cache_data
def load_data(uploaded_file):
    """Carrega e processa os dados do arquivo CSV enviado"""
    try:
        content = uploaded_file.getvalue().decode('utf-8')
        df = pd.read_csv(StringIO(content), skiprows=7, sep=';')
        
        df.columns = ['Timestamp', 'Seconds', 'Velocity', 'Acceleration', 'Odometer', 
                      'Latitude', 'Longitude', 'HeartRate', 'PlayerLoad', 
                      'PositionalQuality', 'HDOP', 'Sats']
        
        numeric_cols = ['Seconds', 'Velocity', 'Acceleration', 'Odometer', 
                        'Latitude', 'Longitude', 'HeartRate', 'PlayerLoad', 
                        'PositionalQuality', 'HDOP', 'Sats']
        
        for col in numeric_cols:
            df[col] = pd.to_numeric(df[col].astype(str).str.replace(',', '.'), errors='coerce')
        
        # CORREÇÃO: Tratamento flexível para o Timestamp
        # Remove espaços extras e converte para string
        df['Timestamp'] = df['Timestamp'].astype(str).str.strip()
        
        # Tenta diferentes formatos de data
        try:
            # Primeiro tenta com o formato com microssegundos
            df['Timestamp'] = pd.to_datetime(df['Timestamp'], format='%d/%m/%Y %H:%M:%S.%f', errors='coerce')
        except:
            try:
                # Tenta com formato sem microssegundos
                df['Timestamp'] = pd.to_datetime(df['Timestamp'], format='%d/%m/%Y %H:%M:%S', errors='coerce')
            except:
                try:
                    # Tenta com formato americano
                    df['Timestamp'] = pd.to_datetime(df['Timestamp'], format='%Y-%m-%d %H:%M:%S', errors='coerce')
                except:
                    try:
                        # Tenta com formato misto (pandas infere)
                        df['Timestamp'] = pd.to_datetime(df['Timestamp'], dayfirst=True, errors='coerce')
                    except:
                        # Última tentativa: deixa o pandas tentar inferir
                        df['Timestamp'] = pd.to_datetime(df['Timestamp'], errors='coerce')
        
        # Se a conversão falhou para todas as linhas, mantém como string
        if df['Timestamp'].isna().all():
            st.warning("⚠️ Não foi possível converter a coluna de Timestamp. Mantendo como texto.")
            df['Timestamp'] = df['Timestamp'].astype(str)
        
        # Remove linhas com dados essenciais ausentes
        df = df.dropna(subset=['Latitude', 'Longitude', 'Velocity', 'HeartRate'])
        
        return df
    except Exception as e:
        st.error(f"Erro ao carregar arquivo: {e}")
        return None

# Sidebar
st.sidebar.header("📁 Upload do Arquivo")
st.sidebar.markdown("""
### Instruções:
1. Clique em "Browse files"
2. Selecione o arquivo CSV exportado pelo OpenField
3. Aguarde o processamento automático
""")

uploaded_file = st.sidebar.file_uploader(
    "Escolha o arquivo CSV",
    type=['csv'],
    help="Arquivo exportado pelo sistema OpenField no formato CSV"
)

with st.sidebar.expander("📋 Exemplo de formato esperado"):
    st.markdown("""
    O arquivo deve conter as seguintes colunas:
    - **Timestamp**: Data/hora da leitura
    - **Seconds**: Tempo em segundos
    - **Velocity**: Velocidade (km/h)
    - **Latitude**: Coordenada de latitude
    - **Longitude**: Coordenada de longitude
    - **HeartRate**: Frequência cardíaca (bpm)
    """)

# Processar arquivo quando enviado
if uploaded_file is not None:
    # Mostrar informações do arquivo
    st.sidebar.success(f"✅ Arquivo carregado: {uploaded_file.name}")
    st.sidebar.info(f"📊 Tamanho: {uploaded_file.size / 1024:.1f} KB")
    
    # Carregar dados
    with st.spinner('Processando dados...'):
        df = load_data(uploaded_file)
    
    if df is not None and len(df) > 0:
        # Extrair informações do cabeçalho do arquivo
        content = uploaded_file.getvalue().decode('utf-8')
        lines = content.split('\n')
        
        # Tentar extrair período e atleta
        periodo = "Não identificado"
        atleta = "Não identificado"
        for line in lines[:10]:
            if 'Period:' in line:
                try:
                    periodo = line.split('"')[1] if '"' in line else line.split(':')[1].strip().strip('"')
                except:
                    periodo = "Não identificado"
            if 'Athlete:' in line:
                try:
                    atleta = line.split('"')[1] if '"' in line else line.split(':')[1].strip().strip('"')
                except:
                    atleta = "Não identificado"
        
        # Mostrar informações do atleta
        st.sidebar.markdown("---")
        st.sidebar.subheader("🏅 Informações do Atleta")
        st.sidebar.write(f"**Atleta:** {atleta}")
        st.sidebar.write(f"**Jogo:** {periodo}")
        
        # Informações gerais
        st.sidebar.markdown("---")
        st.sidebar.subheader("📊 Estatísticas")
        st.sidebar.write(f"Registros: {len(df):,}")
        if 'Seconds' in df.columns and df['Seconds'].max() > 0:
            st.sidebar.write(f"Duração: {df['Seconds'].max():.0f} seg")
        if 'Odometer' in df.columns and df['Odometer'].max() > 0:
            st.sidebar.write(f"Distância: {df['Odometer'].max():.0f} m")
        
        # Filtros na sidebar
        st.sidebar.markdown("---")
        st.sidebar.subheader("⚙️ Filtros")
        
        # Filtro de tempo (apenas se Seconds existir)
        if 'Seconds' in df.columns and df['Seconds'].max() > 0:
            max_time = df['Seconds'].max()
            time_range = st.sidebar.slider(
                "Intervalo de tempo (segundos)",
                min_value=0.0,
                max_value=float(max_time),
                value=(0.0, float(max_time)),
                step=5.0
            )
            time_filter = (df['Seconds'] >= time_range[0]) & (df['Seconds'] <= time_range[1])
        else:
            time_filter = pd.Series([True] * len(df))
            time_range = (0, df['Seconds'].max() if 'Seconds' in df.columns else 100)
        
        # Filtro de velocidade
        if 'Velocity' in df.columns and df['Velocity'].max() > 0:
            max_speed = df['Velocity'].max()
            min_speed = df['Velocity'].min()
            speed_range = st.sidebar.slider(
                "Filtro por velocidade (km/h)",
                min_value=float(min_speed),
                max_value=float(max_speed),
                value=(float(min_speed), float(max_speed)),
                step=0.5
            )
            speed_filter = (df['Velocity'] >= speed_range[0]) & (df['Velocity'] <= speed_range[1])
        else:
            speed_filter = pd.Series([True] * len(df))
            speed_range = (0, 100)
        
        # Aplicar filtros
        df_filtered = df[time_filter & speed_filter].copy()
        
        if len(df_filtered) == 0:
            st.warning("⚠️ Nenhum dado encontrado com os filtros selecionados. Ajuste os intervalos.")
            st.stop()
        
        # Métricas principais
        st.markdown("### 📈 Métricas de Desempenho")
        col1, col2, col3, col4, col5 = st.columns(5)
        
        with col1:
            dist_total = df_filtered['Odometer'].max() if 'Odometer' in df_filtered.columns else 0
            st.metric("Distância total", f"{dist_total:.0f} m")
        with col2:
            vel_max = df_filtered['Velocity'].max() if 'Velocity' in df_filtered.columns else 0
            st.metric("Velocidade máxima", f"{vel_max:.1f} km/h")
        with col3:
            vel_media = df_filtered['Velocity'].mean() if 'Velocity' in df_filtered.columns else 0
            st.metric("Velocidade média", f"{vel_media:.1f} km/h")
        with col4:
            fc_media = df_filtered['HeartRate'].mean() if 'HeartRate' in df_filtered.columns else 0
            st.metric("FC média", f"{fc_media:.0f} bpm")
        with col5:
            fc_max = df_filtered['HeartRate'].max() if 'HeartRate' in df_filtered.columns else 0
            st.metric("FC máxima", f"{fc_max:.0f} bpm")
        
        # Criar abas para diferentes visualizações
        tab1, tab2, tab3, tab4 = st.tabs(["🗺️ Mapa do Percurso", "📈 Gráficos de Desempenho", 
                                           "⚡ Velocidade e Aceleração", "❤️ Frequência Cardíaca"])
        
        # Tab 1: Mapa do Percurso
        with tab1:
            st.subheader("Percurso do Atleta no Campo")
            
            # Calcular centro do mapa
            center_lat = df_filtered['Latitude'].mean() if 'Latitude' in df_filtered.columns else -23.5505
            center_lon = df_filtered['Longitude'].mean() if 'Longitude' in df_filtered.columns else -46.6333
            
            # Opções de estilo de mapa
            map_style = st.selectbox(
                "Estilo do mapa",
                ["open-street-map", "carto-positron", "carto-darkmatter", "stamen-terrain"],
                index=0
            )
            
            # Criar figura do mapa
            fig_map = go.Figure()
            
            # Adicionar linha do percurso
            fig_map.add_trace(go.Scattermapbox(
                lat=df_filtered['Latitude'],
                lon=df_filtered['Longitude'],
                mode='lines+markers',
                line=dict(width=3, color='red'),
                marker=dict(
                    size=6,
                    color=df_filtered['Velocity'] if 'Velocity' in df_filtered.columns else 'blue',
                    colorscale='Viridis',
                    showscale=True,
                    colorbar=dict(title="Velocidade<br>(km/h)", x=1.02),
                    cmin=df_filtered['Velocity'].min() if 'Velocity' in df_filtered.columns else 0,
                    cmax=df_filtered['Velocity'].max() if 'Velocity' in df_filtered.columns else 10
                ),
                text=[f"<b>Tempo:</b> {t:.1f}s<br><b>Velocidade:</b> {v:.1f} km/h<br><b>FC:</b> {h:.0f} bpm" 
                      for t, v, h in zip(df_filtered['Seconds'] if 'Seconds' in df_filtered.columns else range(len(df_filtered)), 
                                        df_filtered['Velocity'] if 'Velocity' in df_filtered.columns else [0]*len(df_filtered),
                                        df_filtered['HeartRate'] if 'HeartRate' in df_filtered.columns else [0]*len(df_filtered))],
                hoverinfo='text',
                name='Percurso'
            ))
            
            # Adicionar marcadores para início e fim
            fig_map.add_trace(go.Scattermapbox(
                lat=[df_filtered['Latitude'].iloc[0], df_filtered['Latitude'].iloc[-1]],
                lon=[df_filtered['Longitude'].iloc[0], df_filtered['Longitude'].iloc[-1]],
                mode='markers',
                marker=dict(size=12, color=['green', 'red'], symbol=['marker', 'marker']),
                text=['🏁 Início', '🏁 Fim'],
                hoverinfo='text',
                name='Pontos'
            ))
            
            # Configurar layout do mapa
            fig_map.update_layout(
                mapbox=dict(
                    style=map_style,
                    center=dict(lat=center_lat, lon=center_lon),
                    zoom=15
                ),
                margin=dict(l=0, r=0, t=30, b=0),
                height=650,
                title={
                    'text': f"Trajetória de {atleta} - {periodo}",
                    'x': 0.5,
                    'xanchor': 'center'
                },
                hovermode='closest'
            )
            
            st.plotly_chart(fig_map, use_container_width=True)
        
        # Tab 2: Gráficos de Desempenho
        with tab2:
            st.subheader("Análise de Desempenho ao Longo do Tempo")
            
            # Seleção de variáveis para gráfico
            col1, col2 = st.columns([1, 3])
            with col1:
                var_options = {}
                if 'Velocity' in df_filtered.columns:
                    var_options['Velocity'] = 'Velocidade (km/h)'
                if 'HeartRate' in df_filtered.columns:
                    var_options['HeartRate'] = 'Frequência Cardíaca (bpm)'
                if 'Acceleration' in df_filtered.columns:
                    var_options['Acceleration'] = 'Aceleração (m/s²)'
                if 'PlayerLoad' in df_filtered.columns:
                    var_options['PlayerLoad'] = 'Carga do Jogador'
                
                selected_var = st.selectbox(
                    "Selecionar variável",
                    options=list(var_options.keys()),
                    format_func=lambda x: var_options[x]
                )
            
            # Gráfico principal
            fig = make_subplots(rows=2, cols=2, 
                               subplot_titles=(var_options[selected_var], 
                                             "Velocidade vs FC", 
                                             "Distância Acumulada",
                                             "Mapa de Calor de Velocidade"),
                               specs=[[{"type": "scatter"}, {"type": "scatter"}],
                                      [{"type": "scatter"}, {"type": "heatmap"}]])
            
            # Gráfico 1: Variável selecionada vs tempo
            if 'Seconds' in df_filtered.columns:
                fig.add_trace(
                    go.Scatter(x=df_filtered['Seconds'], y=df_filtered[selected_var],
                              mode='lines', name=var_options[selected_var],
                              line=dict(width=2), fill='tozeroy'),
                    row=1, col=1
                )
                fig.update_xaxes(title_text="Tempo (s)", row=1, col=1)
            
            # Gráfico 2: Velocidade vs FC
            if 'Velocity' in df_filtered.columns and 'HeartRate' in df_filtered.columns:
                fig.add_trace(
                    go.Scatter(x=df_filtered['Velocity'], y=df_filtered['HeartRate'],
                              mode='markers', name='Dados',
                              marker=dict(size=5, color=df_filtered['Seconds'] if 'Seconds' in df_filtered.columns else 'blue', 
                                        colorscale='Viridis', showscale=True,
                                        colorbar=dict(title="Tempo (s)", x=1.02))),
                    row=1, col=2
                )
                fig.update_xaxes(title_text="Velocidade (km/h)", row=1, col=2)
                fig.update_yaxes(title_text="Frequência Cardíaca (bpm)", row=1, col=2)
            
            # Gráfico 3: Distância acumulada
            if 'Odometer' in df_filtered.columns and 'Seconds' in df_filtered.columns:
                fig.add_trace(
                    go.Scatter(x=df_filtered['Seconds'], y=df_filtered['Odometer'],
                              mode='lines', name='Distância',
                              line=dict(color='green', width=2), fill='tozeroy'),
                    row=2, col=1
                )
                fig.update_xaxes(title_text="Tempo (s)", row=2, col=1)
                fig.update_yaxes(title_text="Distância (m)", row=2, col=1)
            
            # Gráfico 4: Mapa de calor de velocidade
            if 'Velocity' in df_filtered.columns and 'Seconds' in df_filtered.columns:
                segments = min(50, len(df_filtered))
                if segments > 1:
                    df_filtered['Segment'] = pd.cut(df_filtered['Seconds'], bins=segments, labels=False)
                    heat_data = df_filtered.groupby('Segment')['Velocity'].mean().values.reshape(-1, 1)
                    
                    fig.add_trace(
                        go.Heatmap(z=heat_data, 
                                  colorscale='Viridis',
                                  showscale=True,
                                  colorbar=dict(title="Velocidade<br>média (km/h)")),
                        row=2, col=2
                    )
            
            fig.update_layout(height=700, showlegend=False)
            st.plotly_chart(fig, use_container_width=True)
        
        # Tab 3: Velocidade e Aceleração
        with tab3:
            st.subheader("Análise de Velocidade e Aceleração")
            
            col1, col2 = st.columns(2)
            
            with col1:
                if 'Velocity' in df_filtered.columns:
                    fig_hist = px.histogram(df_filtered, x='Velocity', nbins=40,
                                           title="Distribuição de Velocidades",
                                           labels={'Velocity': 'Velocidade (km/h)'},
                                           color_discrete_sequence=['blue'])
                    fig_hist.add_vline(x=df_filtered['Velocity'].mean(), 
                                       line_dash="dash", line_color="red",
                                       annotation_text=f"Média: {df_filtered['Velocity'].mean():.1f} km/h")
                    st.plotly_chart(fig_hist, use_container_width=True)
            
            with col2:
                if 'Velocity' in df_filtered.columns and 'Seconds' in df_filtered.columns:
                    df_filtered['Percentil_Tempo'] = pd.cut(df_filtered['Seconds'], 
                                                            bins=10,
                                                            labels=[f'{i*10}-{(i+1)*10}%' for i in range(10)])
                    fig_box = px.box(df_filtered, x='Percentil_Tempo', y='Velocity',
                                    title="Velocidade por Percentil do Tempo",
                                    labels={'Velocity': 'Velocidade (km/h)', 
                                           'Percentil_Tempo': 'Percentil do Jogo'})
                    st.plotly_chart(fig_box, use_container_width=True)
            
            # Gráfico de aceleração
            if 'Acceleration' in df_filtered.columns and 'Seconds' in df_filtered.columns:
                st.subheader("Aceleração ao longo do tempo")
                
                fig_acc = go.Figure()
                fig_acc.add_trace(go.Scatter(x=df_filtered['Seconds'], y=df_filtered['Acceleration'],
                                             mode='lines', name='Aceleração',
                                             line=dict(color='purple', width=1.5)))
                fig_acc.add_hline(y=0, line_dash="dash", line_color="black", 
                                 annotation_text="Repouso", annotation_position="top right")
                
                fig_acc.update_layout(
                    xaxis_title="Tempo (s)",
                    yaxis_title="Aceleração (m/s²)",
                    height=450,
                    hovermode='x unified'
                )
                st.plotly_chart(fig_acc, use_container_width=True)
        
        # Tab 4: Frequência Cardíaca
        with tab4:
            st.subheader("Análise da Frequência Cardíaca")
            
            if 'HeartRate' in df_filtered.columns:
                col1, col2 = st.columns(2)
                
                with col1:
                    fig_hr_hist = px.histogram(df_filtered, x='HeartRate', nbins=30,
                                               title="Distribuição da Frequência Cardíaca",
                                               labels={'HeartRate': 'Frequência Cardíaca (bpm)'},
                                               color_discrete_sequence=['red'])
                    fig_hr_hist.add_vline(x=df_filtered['HeartRate'].mean(), 
                                          line_dash="dash", line_color="blue",
                                          annotation_text=f"Média: {df_filtered['HeartRate'].mean():.0f} bpm")
                    st.plotly_chart(fig_hr_hist, use_container_width=True)
                
                with col2:
                    if 'Seconds' in df_filtered.columns:
                        fig_hr_time = go.Figure()
                        fig_hr_time.add_trace(go.Scatter(x=df_filtered['Seconds'], y=df_filtered['HeartRate'],
                                                         mode='lines', name='FC',
                                                         line=dict(color='red', width=2)))
                        
                        fc_max = df_filtered['HeartRate'].max()
                        fig_hr_time.add_hrect(y0=0, y1=fc_max*0.6, line_width=0, 
                                              fillcolor="lightgreen", opacity=0.3,
                                              annotation_text="Recuperação", annotation_position="top left")
                        fig_hr_time.add_hrect(y0=fc_max*0.6, y1=fc_max*0.75, line_width=0, 
                                              fillcolor="yellow", opacity=0.3,
                                              annotation_text="Aeróbica", annotation_position="top left")
                        fig_hr_time.add_hrect(y0=fc_max*0.75, y1=fc_max*0.9, line_width=0, 
                                              fillcolor="orange", opacity=0.3,
                                              annotation_text="Anaeróbica", annotation_position="top left")
                        fig_hr_time.add_hrect(y0=fc_max*0.9, y1=fc_max, line_width=0, 
                                              fillcolor="red", opacity=0.3,
                                              annotation_text="Máximo", annotation_position="top left")
                        
                        fig_hr_time.update_layout(xaxis_title="Tempo (s)", 
                                                 yaxis_title="Frequência Cardíaca (bpm)",
                                                 height=450,
                                                 hovermode='x unified')
                        st.plotly_chart(fig_hr_time, use_container_width=True)
                
                # Análise por zona de intensidade
                st.markdown("### 📊 Análise por Zona de Intensidade")
                
                fc_max = df_filtered['HeartRate'].max()
                df_filtered['Zona_FC'] = pd.cut(df_filtered['HeartRate'], 
                                                bins=[0, fc_max*0.6, fc_max*0.75, fc_max*0.9, fc_max],
                                                labels=['Recuperação', 'Aeróbica', 'Anaeróbica', 'Máximo'])
                
                zona_stats = df_filtered.groupby('Zona_FC', observed=True).agg({
                    'HeartRate': ['count', 'mean', 'min', 'max'],
                    'Velocity': 'mean' if 'Velocity' in df_filtered.columns else lambda x: 0
                }).round(0)
                
                zona_stats.columns = ['Contagem', 'FC Média', 'FC Mín', 'FC Máx', 'Velocidade Média']
                zona_stats['% do Tempo'] = (zona_stats['Contagem'] / len(df_filtered) * 100).round(1).astype(str) + '%'
                
                st.dataframe(zona_stats, use_container_width=True)
                
                # Gráfico de pizza das zonas
                fig_pie = px.pie(zona_stats, values='Contagem', names=zona_stats.index,
                                title="Distribuição por Zonas de Intensidade",
                                color_discrete_sequence=['lightgreen', 'yellow', 'orange', 'red'])
                st.plotly_chart(fig_pie, use_container_width=True)
        
        # Botão para download dos dados filtrados
        st.markdown("---")
        csv_data = df_filtered.to_csv(index=False)
        st.download_button(
            label="📥 Exportar dados filtrados (CSV)",
            data=csv_data,
            file_name=f"dados_filtrados_{atleta.replace(' ', '_')}.csv",
            mime="text/csv"
        )
        
        # Rodapé com informações
        st.markdown("---")
        st.markdown(f"**📊 Resumo da análise:** {len(df_filtered)} registros")
        if 'Seconds' in df_filtered.columns:
            st.markdown(f"**⏱️ Período:** {time_range[0]:.0f}s - {time_range[1]:.0f}s")
        if 'Velocity' in df_filtered.columns:
            st.markdown(f"**⚡ Filtro de velocidade:** {speed_range[0]:.1f} - {speed_range[1]:.1f} km/h")
    
    else:
        st.error("❌ Erro ao processar o arquivo. Verifique se o formato está correto.")
        st.info("💡 Dica: O arquivo deve ser exportado pelo sistema OpenField no formato CSV.")
        st.info("📝 Verifique se o arquivo contém as colunas: Timestamp, Seconds, Velocity, Latitude, Longitude, HeartRate")

else:
    # Tela inicial quando nenhum arquivo foi carregado
    st.markdown("""
    ### 👋 Bem-vindo ao Analisador de Percurso do Atleta!
    
    Esta ferramenta permite visualizar e analisar os dados de posicionamento e desempenho de atletas durante partidas.
    
    ### 🚀 Como usar:
    1. **Faça upload** do arquivo CSV exportado pelo sistema OpenField na barra lateral esquerda
    2. Aguarde o processamento automático dos dados
    3. Explore as diferentes visualizações nas abas
    
    ### 📋 Formatos suportados:
    - Arquivos CSV exportados pelo sistema OpenField
    - Deve conter colunas de latitude, longitude, velocidade e frequência cardíaca
    
    ### 💡 Dicas:
    - Use os filtros na barra lateral para focar em períodos específicos
    - Passe o mouse sobre os gráficos para ver detalhes
    - Os dados são processados localmente no seu navegador
    
    ---
    **👈 Clique em "Browse files" na barra lateral para começar!**
    """)
    
    st.info("ℹ️ Aguardando upload do arquivo...")
    
    with st.expander("🎯 Exemplo de análise que você poderá fazer"):
        st.markdown("""
        Com os dados carregados, você poderá:
        - Visualizar a trajetória completa do atleta no campo
        - Identificar picos de velocidade e aceleração
        - Analisar a distribuição da frequência cardíaca
        - Comparar desempenho em diferentes momentos do jogo
        - Exportar dados filtrados para análises adicionais
        
        **O mapa interativo mostrará:**
        - Percurso colorido por velocidade
        - Pontos de início e fim destacados
        - Informações detalhadas ao passar o mouse
        - Diferentes estilos de mapa para melhor visualização
        """)