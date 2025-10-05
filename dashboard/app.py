# Add these imports at the very top
import pandas as pd
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
import os

# Only try to import psycopg2 if it's available
try:
    import psycopg2
    DB_AVAILABLE = True
except ImportError:
    DB_AVAILABLE = False
    st.warning("PostgreSQL not available. Using demo mode.")

# Only define database functions if psycopg2 is available
if DB_AVAILABLE:
    DB_CONFIG = {
        'host': 'localhost',
        'port': '5432',
        'database': 'postgres',
        'user': 'postgres',
        'password': 'password'
    }

    def init_database():
        try:
            conn = psycopg2.connect(**DB_CONFIG)
            cursor = conn.cursor()
            # ... your database code here ...
            conn.close()
        except Exception as e:
            st.error(f"Database connection failed: {e}")

# Then later in your code, only call if available
if DB_AVAILABLE:
    init_database()
else:
    st.info("Running in demo mode without database")



# app.py
import os
import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from streamlit_autorefresh import st_autorefresh
from alert_system import AlertSystem
from dashboard_ui import render_dashboard

# =======================
# SETTINGS / ENV VARS
# =======================
SENDER_EMAIL = os.getenv("ALERT_SENDER_EMAIL")
SENDER_PASSWORD = os.getenv("ALERT_SENDER_PASSWORD")
RAW_RECIPIENTS = os.getenv("ALERT_RECIPIENTS", "")
RECIPIENTS = [x.strip() for x in RAW_RECIPIENTS.split(",") if x.strip()]
missing_creds = not (SENDER_EMAIL and SENDER_PASSWORD and RECIPIENTS)

alert_system = AlertSystem(sender_email=SENDER_EMAIL, password=SENDER_PASSWORD)

# =======================
# SIMULATED LIVE DATA
# =======================
def generate_live_data(existing=None, events=8):
    users = [f"user{i}" for i in range(1, 7)]
    new = pd.DataFrame({
        "user": np.random.choice(users, events),
        "date": [datetime.now() + timedelta(seconds=i) for i in range(events)],
        "login_count": np.random.randint(1, 10, events),
        "failed_logins": np.random.randint(0, 5, events),
        "file_access": np.random.randint(1, 60, events),
        "is_anomaly": np.random.choice([0,1], events, p=[0.75,0.25]),
        "anomaly_score": np.round(np.random.uniform(-1,1,events), 3)
    })
    if existing is not None:
        return pd.concat([existing, new]).tail(200).reset_index(drop=True)
    return new

if "live_data" not in st.session_state:
    st.session_state.live_data = generate_live_data()
else:
    st.session_state.live_data = generate_live_data(st.session_state.live_data)

df = st.session_state.live_data.copy()

# =======================
# RENDER UI
# =======================
render_dashboard(df, missing_creds, alert_system)



# Database configuration
DB_CONFIG = {
    'host': 'localhost',
    'port': '5432',
    'database': 'postgres',
    'user': 'postgres',
    'password': 'password'
}

REDIS_CONFIG = {
    'host': 'localhost',
    'port': 6379,
    'db': 0
}

# Add this to your app to create tables
def init_database():
    conn = psycopg2.connect(**DB_CONFIG)
    cursor = conn.cursor()
    
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS anomalies (
            id SERIAL PRIMARY KEY,
            user_id VARCHAR(50),
            anomaly_score FLOAT,
            timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            details TEXT,
            severity VARCHAR(20)
        )
    ''')
    
    conn.commit()
    conn.close()

# Call this when your app starts
init_database()
