import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
import io
import re
from scipy import stats
from scipy.signal import savgol_filter
import warnings
warnings.filterwarnings('ignore')

# Page configuration
st.set_page_config(page_title="GPS Soccer Analytics - Complete Dashboard", layout="wide", initial_sidebar_state="expanded")

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        background: linear-gradient(90deg, #1E3A8A, #3B82F6);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 1rem;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 1rem;
        color: white;
        text-align: center;
    }
    .info-text {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Title
st.markdown('<div class="main-header">⚽ GPS Soccer Analytics - Complete Performance Dashboard</div>', unsafe_allow_html=True)
st.markdown("---")

# Helper function to parse athlete name from file header
def parse_athlete_name(file_content):
    """Extract athlete name from the file header"""
    try:
        lines = file_content.split('\n')
        for line in lines[:20]:
            if '# Athlete:' in line:
                match = re.search(r'"([^"]*)"', line)
                if match:
                    return match.group(1).strip()
        return None
    except:
        return None

# Function to load and parse GPS data
@st.cache_data
def load_gps_data(uploaded_file):
    """Load and parse the GPS data file"""
    try:
        content = uploaded_file.getvalue().decode('utf-8')
        athlete_name = parse_athlete_name(content)
        
        lines = content.split('\n')
        data_start = 0
        for i, line in enumerate(lines):
            if not line.startswith('#') and 'Timestamp' in line:
                data_start = i
                break
        
        data = pd.read_csv(io.StringIO(content), skiprows=data_start, delimiter=';')
        data.columns = data.columns.str.strip()
        
        numeric_columns = ['Seconds', 'Velocity', 'Acceleration', 'Odometer', 'Heart Rate', 
                          'Player Load', 'Positional Quality (%)', 'HDOP', '#Sats']
        for col in numeric_columns:
            if col in data.columns:
                data[col] = data[col].astype(str).str.replace(',', '.').astype(float)
        
        if 'Timestamp' in data.columns:
            data['Timestamp'] = pd.to_datetime(data['Timestamp'], format='%d/%m/%Y %H:%M:%S.%f')
        
        if 'Timestamp' in data.columns:
            data['Elapsed_Time'] = (data['Timestamp'] - data['Timestamp'].iloc[0]).dt.total_seconds()
        else:
            data['Elapsed_Time'] = data['Seconds']
        
        # Calculate additional metrics
        data['Speed_kmh'] = data['Velocity']
        data['Speed_ms'] = data['Velocity'] / 3.6
        
        # Acceleration zones
        data['Accel_Zone'] = pd.cut(data['Acceleration'], 
                                     bins=[-float('inf'), -2, -1, 0, 1, 2, float('inf')],
                                     labels=['Hard Decel', 'Moderate Decel', 'Light Decel', 
                                            'Light Accel', 'Moderate Accel', 'Hard Accel'])
        
        return data, athlete_name
        
    except Exception as e:
        st.error(f"Error loading file: {str(e)}")
        return None, None

# Function to filter data by time range
def filter_data_by_time(data, start_time, end_time):
    """Filter data between start and end time"""
    if data is None or data.empty:
        return data
    
    if 'Elapsed_Time' in data.columns:
        mask = (data['Elapsed_Time'] >= start_time) & (data['Elapsed_Time'] <= end_time)
        return data[mask].copy()
    return data

# Function to create football field overlay
def create_football_field():
    """Create a football field outline for the map"""
    field_length = 105
    field_width = 68
    
    field_coords = {
        'outline': {'x': [0, field_length, field_length, 0, 0], 'y': [0, 0, field_width, field_width, 0]},
        'center_line': {'x': [field_length/2, field_length/2], 'y': [0, field_width]},
        'center_circle': {'x': field_length/2 + 9.15 * np.cos(np.linspace(0, 2*np.pi, 100)),
                         'y': field_width/2 + 9.15 * np.sin(np.linspace(0, 2*np.pi, 100))},
        'penalty_areas': [
            {'x': [0, 16.5, 16.5, 0], 'y': [field_width/2 - 20.16, field_width/2 - 20.16, field_width/2 + 20.16, field_width/2 + 20.16]},
            {'x': [field_length, field_length - 16.5, field_length - 16.5, field_length], 'y': [field_width/2 - 20.16, field_width/2 - 20.16, field_width/2 + 20.16, field_width/2 + 20.16]}
        ],
        'goal_areas': [
            {'x': [0, 5.5, 5.5, 0], 'y': [field_width/2 - 9.16, field_width/2 - 9.16, field_width/2 + 9.16, field_width/2 + 9.16]},
            {'x': [field_length, field_length - 5.5, field_length - 5.5, field_length], 'y': [field_width/2 - 9.16, field_width/2 - 9.16, field_width/2 + 9.16, field_width/2 + 9.16]}
        ]
    }
    return field_coords

# Function to create trajectory map
def create_trajectory_map(data, athlete_name, show_field=True, show_heatmap=False):
    """Create interactive trajectory map with football field overlay"""
    if data is None or data.empty:
        return None
    
    fig = go.Figure()
    
    if show_field:
        field_coords = create_football_field()
        
        fig.add_trace(go.Scatter(
            x=field_coords['outline']['x'],
            y=field_coords['outline']['y'],
            mode='lines',
            line=dict(color='white', width=2),
            showlegend=False,
            hoverinfo='skip'
        ))
        
        fig.add_trace(go.Scatter(
            x=field_coords['center_line']['x'],
            y=field_coords['center_line']['y'],
            mode='lines',
            line=dict(color='white', width=1, dash='dash'),
            showlegend=False,
            hoverinfo='skip'
        ))
        
        fig.add_trace(go.Scatter(
            x=field_coords['center_circle']['x'],
            y=field_coords['center_circle']['y'],
            mode='lines',
            line=dict(color='white', width=1),
            fill='toself',
            fillcolor='rgba(255,255,255,0.1)',
            showlegend=False,
            hoverinfo='skip'
        ))
        
        for area in field_coords['penalty_areas']:
            fig.add_trace(go.Scatter(
                x=area['x'],
                y=area['y'],
                mode='lines',
                line=dict(color='white', width=1),
                fill='toself',
                fillcolor='rgba(255,255,255,0.1)',
                showlegend=False,
                hoverinfo='skip'
            ))
        
        for area in field_coords['goal_areas']:
            fig.add_trace(go.Scatter(
                x=area['x'],
                y=area['y'],
                mode='lines',
                line=dict(color='white', width=1),
                fill='toself',
                fillcolor='rgba(255,255,255,0.1)',
                showlegend=False,
                hoverinfo='skip'
            ))
    
    fig.add_trace(go.Scattermapbox(
        lat=data['Latitude'],
        lon=data['Longitude'],
        mode='lines',
        line=dict(width=3, color='red'),
        name='Trajectory',
        showlegend=False
    ))
    
    if show_heatmap:
        fig.add_trace(go.Densitymapbox(
            lat=data['Latitude'],
            lon=data['Longitude'],
            z=data['Velocity'],
            radius=10,
            colorscale='Viridis',
            showscale=True,
            name='Heatmap',
            colorbar=dict(title="Velocity (km/h)")
        ))
    else:
        fig.add_trace(go.Scattermapbox(
            lat=data['Latitude'],
            lon=data['Longitude'],
            mode='markers',
            marker=dict(size=5, color=data['Velocity'], 
                       colorscale='Viridis', 
                       colorbar=dict(title="Velocity (km/h)"),
                       showscale=True),
            text=[f"Time: {t:.1f}s<br>Speed: {v:.1f} km/h<br>HR: {hr:.0f} bpm<br>Accel: {a:.2f} m/s²" 
                  for t, v, hr, a in zip(data['Elapsed_Time'], data['Velocity'], data['Heart Rate'], data['Acceleration'])],
            hoverinfo='text',
            name='Positions'
        ))
    
    start_point = data.iloc[0]
    end_point = data.iloc[-1]
    
    fig.add_trace(go.Scattermapbox(
        lat=[start_point['Latitude']],
        lon=[start_point['Longitude']],
        mode='markers',
        marker=dict(size=14, color='green', symbol='circle'),
        text=[f"🏁 START<br>Time: {start_point['Elapsed_Time']:.1f}s<br>Speed: {start_point['Velocity']:.1f} km/h"],
        hoverinfo='text',
        name='Start'
    ))
    
    fig.add_trace(go.Scattermapbox(
        lat=[end_point['Latitude']],
        lon=[end_point['Longitude']],
        mode='markers',
        marker=dict(size=14, color='red', symbol='circle'),
        text=[f"🏁 END<br>Time: {end_point['Elapsed_Time']:.1f}s<br>Speed: {end_point['Velocity']:.1f} km/h"],
        hoverinfo='text',
        name='End'
    ))
    
    center_lat = data['Latitude'].mean()
    center_lon = data['Longitude'].mean()
    
    fig.update_layout(
        mapbox=dict(
            style="open-street-map",
            center=dict(lat=center_lat, lon=center_lon),
            zoom=15
        ),
        margin=dict(l=0, r=0, t=30, b=0),
        height=650,
        title=f"📍 {athlete_name} - Movement Trajectory"
    )
    
    return fig

# Function to calculate high-intensity metrics
def calculate_high_intensity_metrics(data):
    """Calculate high-intensity running and sprinting metrics"""
    if data is None or data.empty:
        return {}
    
    # Speed zones (km/h)
    high_intensity_threshold = 19.8  # >19.8 km/h is high intensity
    sprint_threshold = 25.2  # >25.2 km/h is sprint
    
    high_intensity_mask = data['Velocity'] >= high_intensity_threshold
    sprint_mask = data['Velocity'] >= sprint_threshold
    
    # Calculate distances in each zone
    time_diff = np.diff(data['Elapsed_Time'], prepend=data['Elapsed_Time'].iloc[0])
    distance_per_point = data['Speed_ms'] * time_diff
    
    total_distance = distance_per_point.sum()
    high_intensity_distance = distance_per_point[high_intensity_mask].sum()
    sprint_distance = distance_per_point[sprint_mask].sum()
    
    # Count high-intensity efforts
    high_intensity_efforts = 0
    sprint_efforts = 0
    in_high_intensity = False
    in_sprint = False
    
    for i in range(len(data)):
        if data['Velocity'].iloc[i] >= high_intensity_threshold and not in_high_intensity:
            high_intensity_efforts += 1
            in_high_intensity = True
        elif data['Velocity'].iloc[i] < high_intensity_threshold:
            in_high_intensity = False
        
        if data['Velocity'].iloc[i] >= sprint_threshold and not in_sprint:
            sprint_efforts += 1
            in_sprint = True
        elif data['Velocity'].iloc[i] < sprint_threshold:
            in_sprint = False
    
    return {
        'Total Distance': total_distance,
        'High Intensity Distance': high_intensity_distance,
        'Sprint Distance': sprint_distance,
        '% High Intensity': (high_intensity_distance / total_distance * 100) if total_distance > 0 else 0,
        '% Sprint': (sprint_distance / total_distance * 100) if total_distance > 0 else 0,
        'High Intensity Efforts': high_intensity_efforts,
        'Sprint Efforts': sprint_efforts
    }

# Function to create performance dashboard
def create_performance_dashboard(data, athlete_name):
    """Create comprehensive performance metrics dashboard"""
    if data is None or data.empty:
        return None, None
    
    # Calculate metrics
    total_distance = data['Odometer'].max() / 1000
    max_speed = data['Velocity'].max()
    avg_speed = data['Velocity'].mean()
    total_duration = data['Elapsed_Time'].max()
    avg_hr = data['Heart Rate'].mean()
    max_hr = data['Heart Rate'].max()
    
    # High intensity metrics
    hi_metrics = calculate_high_intensity_metrics(data)
    
    # Speed zones
    speed_zones = {
        'Walking (0-7 km/h)': ((data['Velocity'] >= 0) & (data['Velocity'] < 7)).sum(),
        'Jogging (7-12 km/h)': ((data['Velocity'] >= 7) & (data['Velocity'] < 12)).sum(),
        'Running (12-18 km/h)': ((data['Velocity'] >= 12) & (data['Velocity'] < 18)).sum(),
        'High Intensity (18-25 km/h)': ((data['Velocity'] >= 18) & (data['Velocity'] < 25)).sum(),
        'Sprinting (>25 km/h)': (data['Velocity'] >= 25).sum()
    }
    
    # Heart rate zones
    hr_zones = {
        'Recovery (<120 bpm)': (data['Heart Rate'] < 120).sum(),
        'Aerobic (120-150 bpm)': ((data['Heart Rate'] >= 120) & (data['Heart Rate'] < 150)).sum(),
        'Anaerobic (150-170 bpm)': ((data['Heart Rate'] >= 150) & (data['Heart Rate'] < 170)).sum(),
        'Maximal (>170 bpm)': (data['Heart Rate'] >= 170).sum()
    }
    
    # Acceleration zones
    accel_counts = data['Accel_Zone'].value_counts()
    
    # Create metrics row with better visualization
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        st.metric("🏃 Total Distance", f"{total_distance:.2f} km", 
                 delta=f"{hi_metrics['High Intensity Distance']/1000:.2f} km HI")
    with col2:
        st.metric("⚡ Max Speed", f"{max_speed:.1f} km/h")
    with col3:
        st.metric("📊 Avg Speed", f"{avg_speed:.1f} km/h")
    with col4:
        st.metric("⏱️ Duration", f"{total_duration/60:.1f} min")
    with col5:
        st.metric("❤️ Avg HR", f"{avg_hr:.0f} bpm", delta=f"Max: {max_hr:.0f}")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("### 🏃‍♂️ High Intensity Metrics")
        hi_df = pd.DataFrame({
            'Metric': ['High Intensity Distance', 'Sprint Distance', '% High Intensity', '% Sprint', 
                      'HI Efforts', 'Sprint Efforts'],
            'Value': [f"{hi_metrics['High Intensity Distance']/1000:.2f} km", 
                     f"{hi_metrics['Sprint Distance']/1000:.2f} km",
                     f"{hi_metrics['% High Intensity']:.1f}%",
                     f"{hi_metrics['% Sprint']:.1f}%",
                     hi_metrics['High Intensity Efforts'],
                     hi_metrics['Sprint Efforts']]
        })
        st.dataframe(hi_df, use_container_width=True, hide_index=True)
    
    with col2:
        st.markdown("### ⚡ Speed Zones")
        fig_speed = go.Figure(data=[go.Pie(
            labels=list(speed_zones.keys()),
            values=list(speed_zones.values()),
            hole=0.3,
            marker=dict(colors=['#2ecc71', '#3498db', '#f39c12', '#e74c3c', '#c0392b'])
        )])
        fig_speed.update_layout(height=350, margin=dict(t=0, b=0))
        st.plotly_chart(fig_speed, use_container_width=True)
    
    with col3:
        st.markdown("### ❤️ Heart Rate Zones")
        fig_hr = go.Figure(data=[go.Pie(
            labels=list(hr_zones.keys()),
            values=list(hr_zones.values()),
            hole=0.3,
            marker=dict(colors=['#2ecc71', '#3498db', '#f39c12', '#e74c3c'])
        )])
        fig_hr.update_layout(height=350, margin=dict(t=0, b=0))
        st.plotly_chart(fig_hr, use_container_width=True)
    
    # Speed and HR over time
    fig_over_time = make_subplots(specs=[[{"secondary_y": True}]])
    
    fig_over_time.add_trace(
        go.Scatter(x=data['Elapsed_Time']/60, y=data['Velocity'], mode='lines', 
                  name='Speed', line=dict(color='#3498db', width=2)),
        secondary_y=False
    )
    
    fig_over_time.add_trace(
        go.Scatter(x=data['Elapsed_Time']/60, y=data['Heart Rate'], mode='lines',
                  name='Heart Rate', line=dict(color='#e74c3c', width=2)),
        secondary_y=True
    )
    
    fig_over_time.update_layout(
        title="Speed and Heart Rate Over Time",
        xaxis=dict(title="Time (minutes)"),
        height=450,
        hovermode='x unified'
    )
    fig_over_time.update_yaxes(title_text="Speed (km/h)", secondary_y=False)
    fig_over_time.update_yaxes(title_text="Heart Rate (bpm)", secondary_y=True)
    
    st.plotly_chart(fig_over_time, use_container_width=True)
    
    # Acceleration profile
    fig_accel = go.Figure()
    
    # Acceleration histogram
    fig_accel.add_trace(go.Histogram(
        x=data['Acceleration'],
        nbinsx=50,
        name='Acceleration',
        marker_color='#9b59b6'
    ))
    
    fig_accel.update_layout(
        title="Acceleration Profile",
        xaxis=dict(title="Acceleration (m/s²)"),
        yaxis=dict(title="Frequency"),
        height=400
    )
    
    st.plotly_chart(fig_accel, use_container_width=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### 📊 Acceleration Zones")
        if not accel_counts.empty:
            fig_accel_zones = go.Figure(data=[go.Bar(
                x=accel_counts.index,
                y=accel_counts.values,
                marker_color='#9b59b6'
            )])
            fig_accel_zones.update_layout(height=350, xaxis=dict(title="Zone"), yaxis=dict(title="Count"))
            st.plotly_chart(fig_accel_zones, use_container_width=True)
    
    with col2:
        st.markdown("### 📈 Speed Distribution")
        fig_speed_dist = go.Figure(data=[go.Histogram(
            x=data['Velocity'],
            nbinsx=30,
            marker_color='#3498db'
        )])
        fig_speed_dist.update_layout(
            height=350,
            xaxis=dict(title="Speed (km/h)"),
            yaxis=dict(title="Frequency")
        )
        st.plotly_chart(fig_speed_dist, use_container_width=True)
    
    return speed_zones, hr_zones

# Function to create advanced comparative analysis
def create_advanced_comparison(data_dict, selected_athletes, start_time, end_time):
    """Create advanced comparative analysis between athletes"""
    
    # Prepare data for comparison
    comparison_data = []
    
    for athlete in selected_athletes:
        data = data_dict[athlete]
        filtered_data = filter_data_by_time(data, start_time, end_time)
        
        if not filtered_data.empty:
            hi_metrics = calculate_high_intensity_metrics(filtered_data)
            
            comparison_data.append({
                'Athlete': athlete,
                'Distance (km)': filtered_data['Odometer'].max() / 1000,
                'Max Speed (km/h)': filtered_data['Velocity'].max(),
                'Avg Speed (km/h)': filtered_data['Velocity'].mean(),
                'Avg HR (bpm)': filtered_data['Heart Rate'].mean(),
                'Max HR (bpm)': filtered_data['Heart Rate'].max(),
                'Duration (min)': filtered_data['Elapsed_Time'].max() / 60,
                'HI Distance (km)': hi_metrics['High Intensity Distance'] / 1000,
                'Sprint Distance (km)': hi_metrics['Sprint Distance'] / 1000,
                '% HI': hi_metrics['% High Intensity'],
                '% Sprint': hi_metrics['% Sprint'],
                'HI Efforts': hi_metrics['High Intensity Efforts'],
                'Sprint Efforts': hi_metrics['Sprint Efforts']
            })
    
    if not comparison_data:
        return None
    
    df_comp = pd.DataFrame(comparison_data)
    
    # Display comparison table
    st.markdown("### 📊 Comparative Metrics Table")
    st.dataframe(df_comp.style.highlight_max(axis=0, color='lightgreen').highlight_min(axis=0, color='lightcoral'), 
                 use_container_width=True)
    
    # Speed comparison chart
    fig_speed_comp = go.Figure()
    for athlete in selected_athletes:
        data = data_dict[athlete]
        filtered_data = filter_data_by_time(data, start_time, end_time)
        if not filtered_data.empty:
            # Smooth the speed data for better visualization
            if len(filtered_data) > 10:
                smoothed_speed = savgol_filter(filtered_data['Velocity'], min(11, len(filtered_data)-1 if len(filtered_data)%2==0 else len(filtered_data)), 3)
            else:
                smoothed_speed = filtered_data['Velocity']
            
            fig_speed_comp.add_trace(go.Scatter(
                x=filtered_data['Elapsed_Time'] / 60,
                y=smoothed_speed,
                mode='lines',
                name=athlete,
                line=dict(width=2)
            ))
    
    fig_speed_comp.update_layout(
        title="Speed Comparison Over Time",
        xaxis=dict(title="Time (minutes)"),
        yaxis=dict(title="Speed (km/h)"),
        height=500,
        hovermode='x unified'
    )
    st.plotly_chart(fig_speed_comp, use_container_width=True)
    
    # Heart rate comparison
    fig_hr_comp = go.Figure()
    for athlete in selected_athletes:
        data = data_dict[athlete]
        filtered_data = filter_data_by_time(data, start_time, end_time)
        if not filtered_data.empty:
            fig_hr_comp.add_trace(go.Scatter(
                x=filtered_data['Elapsed_Time'] / 60,
                y=filtered_data['Heart Rate'],
                mode='lines',
                name=athlete,
                line=dict(width=2)
            ))
    
    fig_hr_comp.update_layout(
        title="Heart Rate Comparison Over Time",
        xaxis=dict(title="Time (minutes)"),
        yaxis=dict(title="Heart Rate (bpm)"),
        height=500,
        hovermode='x unified'
    )
    st.plotly_chart(fig_hr_comp, use_container_width=True)
    
    # Radar chart for overall performance
    metrics_for_radar = ['Distance (km)', 'Max Speed (km/h)', 'Avg Speed (km/h)', 
                         'Avg HR (bpm)', 'HI Distance (km)', '% HI']
    
    fig_radar = go.Figure()
    
    for athlete in selected_athletes:
        athlete_data = df_comp[df_comp['Athlete'] == athlete].iloc[0]
        values = [athlete_data[m] for m in metrics_for_radar]
        
        # Normalize values for radar chart
        max_values = df_comp[metrics_for_radar].max()
        normalized_values = [values[i] / max_values[i] if max_values[i] > 0 else 0 for i in range(len(values))]
        
        fig_radar.add_trace(go.Scatterpolar(
            r=normalized_values,
            theta=metrics_for_radar,
            fill='toself',
            name=athlete
        ))
    
    fig_radar.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 1]
            )),
        title="Performance Radar Chart (Normalized)",
        height=500
    )
    st.plotly_chart(fig_radar, use_container_width=True)
    
    # Bar chart comparison
    metrics_to_plot = ['Distance (km)', 'Max Speed (km/h)', 'Avg Speed (km/h)', 
                      'HI Distance (km)', 'Sprint Distance (km)']
    
    for metric in metrics_to_plot:
        fig_bar = px.bar(
            df_comp,
            x='Athlete',
            y=metric,
            title=f'{metric} Comparison',
            color='Athlete',
            text_auto='.2f'
        )
        fig_bar.update_layout(height=400, showlegend=False)
        st.plotly_chart(fig_bar, use_container_width=True)
    
    return df_comp

# Main app
def main():
    st.sidebar.markdown("## 📁 File Upload")
    uploaded_files = st.sidebar.file_uploader(
        "Choose GPS data files", 
        type=['csv'], 
        accept_multiple_files=True,
        help="Upload one or more CSV files from OpenField export"
    )
    
    if not uploaded_files:
        st.markdown("""
        <div class="info-text">
        <h3>👈 Welcome to GPS Soccer Analytics!</h3>
        <p>This comprehensive dashboard allows you to:</p>
        <ul>
            <li>📊 <strong>Analyze performance metrics</strong> - Distance, speed, heart rate, acceleration zones</li>
            <li>🗺️ <strong>Visualize movement trajectories</strong> - Interactive maps with football field overlay</li>
            <li>⚡ <strong>Track high-intensity metrics</strong> - HI running, sprints, and acceleration/deceleration profiles</li>
            <li>📈 <strong>Compare multiple athletes</strong> - Side-by-side performance analysis</li>
            <li>⏱️ <strong>Select time ranges</strong> - Focus on specific moments of the game</li>
        </ul>
        <p><strong>Getting Started:</strong> Upload one or more CSV files from your OpenField export to begin analysis.</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Example preview
        st.markdown("### 📋 File Format Example")
        st.code("""
# OpenField Export : 28/02/2026 16:26:07
# Reference time : 24/02/2026 23:51:52 UTC
# Athlete: "L.SASHA"
Timestamp;Seconds;Velocity;Acceleration;Odometer;Latitude;Longitude;Heart Rate;...
24/02/2026 20:51:52.09;0;1,03;0,059;0;-3,8070812;-38,5228445;92;...
        """, language="csv")
        return
    
    # Load all data files
    all_data = {}
    for file in uploaded_files:
        with st.spinner(f'Loading {file.name}...'):
            data, athlete_name = load_gps_data(file)
            if data is not None:
                if athlete_name:
                    all_data[athlete_name] = data
                else:
                    all_data[file.name] = data
                    st.sidebar.warning(f"Could not extract athlete name from {file.name}")
    
    if not all_data:
        st.error("No valid data files could be loaded")
        return
    
    # Sidebar - Athlete selection
    st.sidebar.markdown("## 👥 Athlete Selection")
    athlete_names = list(all_data.keys())
    
    selected_athletes = st.sidebar.multiselect(
        "Select athletes to analyze",
        athlete_names,
        default=[athlete_names[0]] if athlete_names else []
    )
    
    if not selected_athletes:
        st.warning("Please select at least one athlete")
        return
    
    # Sidebar - Time range selection
    st.sidebar.markdown("## ⏱️ Time Range Selection")
    
    first_athlete = selected_athletes[0]
    first_data = all_data[first_athlete]
    total_duration = first_data['Elapsed_Time'].max()
    
    # Get min and max times from all selected athletes
    all_min_times = []
    all_max_times = []
    for athlete in selected_athletes:
        data = all_data[athlete]
        all_min_times.append(data['Elapsed_Time'].min())
        all_max_times.append(data['Elapsed_Time'].max())
    
    global_min_time = min(all_min_times)
    global_max_time = max(all_max_times)
    
    col1, col2 = st.sidebar.columns(2)
    with col1:
        start_time = st.number_input(
            "Start Time (s)",
            min_value=float(global_min_time),
            max_value=float(global_max_time),
            value=float(global_min_time),
            step=1.0,
            format="%.1f"
        )
    with col2:
        end_time = st.number_input(
            "End Time (s)",
            min_value=float(global_min_time),
            max_value=float(global_max_time),
            value=float(global_max_time),
            step=1.0,
            format="%.1f"
        )
    
    if start_time >= end_time:
        st.sidebar.error("Start time must be less than end time")
        end_time = start_time + 1
    
    # Show time range info
    duration_seconds = end_time - start_time
    st.sidebar.info(f"Selected duration: {duration_seconds:.0f}s ({duration_seconds/60:.1f} min)")
    
    # Sidebar - Visualization options
    st.sidebar.markdown("## 🎨 Visualization Options")
    show_football_field = st.sidebar.checkbox("Show Football Field on Map", value=True)
    show_heatmap = st.sidebar.checkbox("Show Heatmap (instead of points)", value=False)
    
    # Sidebar - Advanced filters
    st.sidebar.markdown("## 🔍 Advanced Filters")
    min_speed = st.sidebar.slider("Minimum Speed (km/h)", 0.0, 30.0, 0.0, 1.0)
    max_speed = st.sidebar.slider("Maximum Speed (km/h)", 0.0, 30.0, 30.0, 1.0)
    
    # Main content - Tabs
    tab1, tab2, tab3, tab4 = st.tabs(["📊 Performance Metrics", "🗺️ Movement Trajectory", "📈 Comparative Analysis", "📋 Raw Data"])
    
    with tab1:
        st.header("Performance Metrics Dashboard")
        
        if len(selected_athletes) > 1:
            selected_for_details = st.selectbox(
                "Select athlete for detailed metrics",
                selected_athletes,
                key="details_select"
            )
        else:
            selected_for_details = selected_athletes[0]
        
        data = all_data[selected_for_details]
        
        # Apply speed filters
        filtered_data = filter_data_by_time(data, start_time, end_time)
        filtered_data = filtered_data[(filtered_data['Velocity'] >= min_speed) & 
                                       (filtered_data['Velocity'] <= max_speed)]
        
        if filtered_data.empty:
            st.warning("No data available for selected filters")
        else:
            st.info(f"📊 Showing data for: **{selected_for_details}** | Time range: **{start_time:.1f}s - {end_time:.1f}s** | Speed filter: **{min_speed:.0f}-{max_speed:.0f} km/h**")
            create_performance_dashboard(filtered_data, selected_for_details)
    
    with tab2:
        st.header("Movement Trajectory Analysis")
        
        if len(selected_athletes) > 1:
            map_athletes = st.multiselect(
                "Select athletes to display on map",
                selected_athletes,
                default=[selected_athletes[0]],
                key="map_select"
            )
        else:
            map_athletes = selected_athletes
        
        if map_athletes:
            for athlete in map_athletes:
                data = all_data[athlete]
                filtered_data = filter_data_by_time(data, start_time, end_time)
                filtered_data = filtered_data[(filtered_data['Velocity'] >= min_speed) & 
                                               (filtered_data['Velocity'] <= max_speed)]
                
                if not filtered_data.empty:
                    fig = create_trajectory_map(filtered_data, athlete, show_football_field, show_heatmap)
                    if fig:
                        st.plotly_chart(fig, use_container_width=True)
                    
                    # Add statistics for the trajectory
                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        st.metric("Points in trajectory", len(filtered_data))
                    with col2:
                        st.metric("Avg Speed", f"{filtered_data['Velocity'].mean():.1f} km/h")
                    with col3:
                        st.metric("Max Speed", f"{filtered_data['Velocity'].max():.1f} km/h")
                    with col4:
                        st.metric("Duration", f"{filtered_data['Elapsed_Time'].max() - filtered_data['Elapsed_Time'].min():.1f}s")
                else:
                    st.warning(f"No data available for {athlete} in selected time range")
        else:
            st.warning("No athletes selected for map display")
    
    with tab3:
        st.header("Comparative Analysis")
        
        if len(selected_athletes) < 2:
            st.info("👥 Select at least 2 athletes for comparative analysis")
        else:
            create_advanced_comparison(all_data, selected_athletes, start_time, end_time)
    
    with tab4:
        st.header("Raw Data Explorer")
        
        if len(selected_athletes) > 1:
            data_select = st.selectbox("Select athlete to view data", selected_athletes, key="raw_select")
        else:
            data_select = selected_athletes[0]
        
        data = all_data[data_select]
        filtered_data = filter_data_by_time(data, start_time, end_time)
        filtered_data = filtered_data[(filtered_data['Velocity'] >= min_speed) & 
                                       (filtered_data['Velocity'] <= max_speed)]
        
        st.dataframe(filtered_data, use_container_width=True, height=400)
        
        # Download button
        csv = filtered_data.to_csv(index=False, sep=';', decimal=',')
        st.download_button(
            label="📥 Download filtered data as CSV",
            data=csv,
            file_name=f"{data_select}_filtered_data.csv",
            mime="text/csv"
        )

if __name__ == "__main__":
    main()