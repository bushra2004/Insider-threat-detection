import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import os
import plotly.graph_objects as go
from plotly.subplots import make_subplots



def create_realtime_monitoring():
    """Create real-time monitoring dashboard"""
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=('Real-time Anomalies', 'Risk Score Distribution', 
                       'User Activity Heatmap', 'Network Traffic'),
        specs=[[{"type": "scatter"}, {"type": "histogram"}],
               [{"type": "heatmap"}, {"type": "scatter"}]]
    )
    
    # Add real-time data here from WebSocket/Kafka
    return fig

# Page config
st.set_page_config(
    page_title="Insider Threat Detection System",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {font-size: 3rem; color: #1f77b4;}
    .metric-card {background-color: #f0f2f6; padding: 20px; border-radius: 10px;}
    .anomaly-high {color: #ff4b4b; font-weight: bold;}
    .anomaly-medium {color: #ffa14b; font-weight: bold;}
    .anomaly-low {color: #00cc96; font-weight: bold;}
</style>
""", unsafe_allow_html=True)

# Header
st.markdown('<h1 class="main-header">🔍 Insider Threat Detection System</h1>', unsafe_allow_html=True)
st.markdown("**Real-time Behavioral Analytics & Anomaly Detection**")

# Load data function
@st.cache_data
def load_data():
    try:
        features = pd.read_csv('data/processed/behavioral_features.csv')
        predictions = pd.read_csv('data/processed/advanced_predictions.csv')
        return features, predictions, "advanced"
    except:
        try:
            predictions = pd.read_csv('data/processed/predictions.csv')
            return features, predictions, "basic"
        except:
            return None, None, "none"

# Load data
features, predictions, data_type = load_data()

# Sidebar
st.sidebar.header("🔧 Control Panel")
st.sidebar.markdown("---")

if features is not None and predictions is not None:
    # User filter
    users = sorted(predictions['user'].unique())
    selected_user = st.sidebar.selectbox("👤 Select User", ["All Users"] + users)
    
    # Date filter
    date_range = st.sidebar.date_input("📅 Date Range", [])
    
    # Risk threshold
    risk_threshold = st.sidebar.slider("⚡ Risk Threshold", 0, 100, 70)
    
    # Filter data
    if selected_user != "All Users":
        filtered_pred = predictions[predictions['user'] == selected_user]
        filtered_feat = features[features['user'] == selected_user]
    else:
        filtered_pred = predictions.copy()
        filtered_feat = features.copy()

        

# Main content
if features is not None and predictions is not None:
    # Metrics row
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        total_users = filtered_pred['user'].nunique()
        st.markdown(f'<div class="metric-card"><h3>👥 Total Users</h3><h2>{total_users}</h2></div>', unsafe_allow_html=True)
    
    with col2:
        anomalies = filtered_pred['final_anomaly'].sum() if 'final_anomaly' in filtered_pred.columns else filtered_pred['is_anomaly'].sum()
        st.markdown(f'<div class="metric-card"><h3>⚠️ Anomalies</h3><h2>{anomalies}</h2></div>', unsafe_allow_html=True)
    
    with col3:
        total_records = len(filtered_pred)
        risk_score = (anomalies / total_records) * 100 if total_records > 0 else 0
        risk_class = "anomaly-high" if risk_score > risk_threshold else "anomaly-medium" if risk_score > 30 else "anomaly-low"
        st.markdown(f'<div class="metric-card"><h3>📊 Risk Score</h3><h2 class="{risk_class}">{risk_score:.1f}%</h2></div>', unsafe_allow_html=True)
    
    with col4:
        high_risk_users = len(filtered_pred[filtered_pred['final_anomaly'] == 1]['user'].unique()) if 'final_anomaly' in filtered_pred.columns else len(filtered_pred[filtered_pred['is_anomaly'] == 1]['user'].unique())
        st.markdown(f'<div class="metric-card"><h3>🚨 High Risk Users</h3><h2>{high_risk_users}</h2></div>', unsafe_allow_html=True)

    # Charts
    tab1, tab2, tab3, tab4 = st.tabs(["📈 Risk Overview", "👤 User Analysis", "🕒 Time Patterns", "📋 Raw Data"])
    
    with tab1:
        col1, col2 = st.columns(2)
        
        with col1:
            # Risk distribution
            fig = px.histogram(filtered_pred, x='iso_forest_score' if 'iso_forest_score' in filtered_pred.columns else 'anomaly_score', 
                             title="Anomaly Score Distribution", nbins=30)
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Anomaly by user
            user_anomalies = filtered_pred.groupby('user')['final_anomaly' if 'final_anomaly' in filtered_pred.columns else 'is_anomaly'].sum().reset_index()
            fig = px.bar(user_anomalies, x='user', y='final_anomaly' if 'final_anomaly' in user_anomalies.columns else 'is_anomaly',
                       title="Anomalies per User", color='final_anomaly' if 'final_anomaly' in user_anomalies.columns else 'is_anomaly')
            st.plotly_chart(fig, use_container_width=True)
    
    with tab2:
        if selected_user != "All Users":
            user_data = filtered_pred[filtered_pred['user'] == selected_user]
            st.subheader(f"Detailed Analysis for {selected_user}")
            
            col1, col2 = st.columns(2)
            
            with col1:
                fig = px.scatter(user_data, x='date', y='iso_forest_score' if 'iso_forest_score' in user_data.columns else 'anomaly_score',
                               title=f"Risk Score Timeline for {selected_user}", color='final_anomaly' if 'final_anomaly' in user_data.columns else 'is_anomaly')
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                metrics = ['login_count', 'failed_logins', 'file_access_count', 'late_logins']
                avg_values = [user_data[metric].mean() for metric in metrics]
                fig = px.bar(x=metrics, y=avg_values, title="Average Behavioral Metrics")
                st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Select a specific user from the sidebar to see detailed analysis")
    
    with tab3:
        st.subheader("Temporal Patterns")
        time_analysis = filtered_pred.groupby('date').agg({
            'login_count': 'sum',
            'failed_logins': 'sum',
            'final_anomaly' if 'final_anomaly' in filtered_pred.columns else 'is_anomaly': 'sum'
        }).reset_index()
        
        fig = px.line(time_analysis, x='date', y=['login_count', 'failed_logins'], 
                    title="Login Patterns Over Time", labels={'value': 'Count', 'variable': 'Metric'})
        st.plotly_chart(fig, use_container_width=True)
    
    with tab4:
        st.subheader("Processed Features")
        st.dataframe(filtered_feat, use_container_width=True)
        
        st.subheader("Anomaly Predictions")
        st.dataframe(filtered_pred, use_container_width=True)

else:
    # Setup instructions
    st.warning("📋 Setup Required")
    st.info("""
    To use this dashboard, please run the following commands in your terminal:
    
    ```bash
    # 1. Preprocess data
    python src\\preprocess.py
    
    # 2. Train machine learning model
    python src\\train_baseline.py
    
    # 3. (Optional) Advanced training
    python src\\train_advanced.py
    ```
    """)
    
    # Quick demo with sample data
    st.markdown("---")
    st.subheader("🎯 Sample Demo Data")
    
    sample_data = {
        'user': ['user1', 'user2', 'user3', 'user4', 'user5'],
        'login_count': [45, 32, 78, 28, 35],
        'failed_logins': [2, 1, 15, 1, 3],
        'file_access': [120, 85, 250, 70, 95],
        'after_hours': [2, 1, 18, 0, 3],
        'risk_score': [15, 10, 85, 5, 20],
        'status': ['Low', 'Low', 'High', 'Low', 'Medium']
    }
    
    sample_df = pd.DataFrame(sample_data)
    st.dataframe(sample_df)
    
    fig = px.bar(sample_df, x='user', y='risk_score', color='status',
                 title="Sample Risk Scores by User", 
                 color_discrete_map={'Low': 'green', 'Medium': 'orange', 'High': 'red'})
    st.plotly_chart(fig, use_container_width=True)

# Footer
st.markdown("---")
st.markdown("**Insider Threat Detection System**")