# dashboard_ui.py
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
from datetime import datetime

def render_dashboard(df, missing_creds, alert_system):
    # =======================
    # PAGE CONFIG
    # =======================
    st.set_page_config(
        page_title="Insider Threat Detection Dashboard",
        page_icon="🛡",
        layout="wide"
    )

    # =======================
    # THEME TOGGLE
    # =======================
    if "dark_mode" not in st.session_state:
        st.session_state.dark_mode = False

    dark_mode = st.sidebar.toggle("🌙 Dark Mode", value=st.session_state.dark_mode)
    st.session_state.dark_mode = dark_mode

    if dark_mode:
        bg_color = "#0E1117"
        text_color = "#FAFAFA"
        card_bg = "#1E222A"
    else:
        bg_color = "#FFFFFF"
        text_color = "#000000"
        card_bg = "#F8F9FA"

    st.markdown(
        f"""
        <style>
        body {{
            background-color: {bg_color};
            color: {text_color};
        }}
        .stApp {{
            background-color: {bg_color};
        }}
        .stMetric, .stDataFrame, .stTable {{
            background-color: {card_bg};
            color: {text_color};
            border-radius: 8px;
            padding: 8px;
        }}
        </style>
        """,
        unsafe_allow_html=True
    )

    # =======================
    # HEADER
    # =======================
    st.markdown(
        f"<h1 style='color:#1f77b4;'>🛡 Insider Threat Detection Dashboard</h1>",
        unsafe_allow_html=True
    )
    st.markdown("Real-time Monitoring • Behavioral Analytics • Automated Alerts")

    from streamlit_autorefresh import st_autorefresh
    st_autorefresh(interval=15_000, limit=None, key="auto_refresh")

    # =======================
    # SIDEBAR
    # =======================
    st.sidebar.header("⚙️ Control Panel")
    if missing_creds:
        st.sidebar.warning("Email alerts not configured")
    else:
        st.sidebar.success("✅ Alerts active")

    st.sidebar.text(f"Last refresh: {datetime.now().strftime('%H:%M:%S')}")

    users = sorted(df["user"].unique())
    selected_user = st.sidebar.selectbox("👤 Select user", ["All Users"] + users)
    threshold = st.sidebar.slider("⚡ Risk threshold (%)", 0, 100, 60)

    if selected_user != "All Users":
        df = df[df["user"] == selected_user]

    # =======================
    # METRICS ROW
    # =======================
    col1, col2, col3, col4 = st.columns(4)
    total_users = df["user"].nunique()
    anomalies = int(df["is_anomaly"].sum())
    risk_pct = (anomalies / len(df) * 100) if len(df) else 0
    high_risk_users = df[df["is_anomaly"]==1]["user"].nunique()

    col1.metric("👥 Users", total_users)
    col2.metric("⚠️ Anomalies", anomalies)
    col3.metric("📊 Risk Level", f"{risk_pct:.1f}%")
    col4.metric("🚨 High Risk", high_risk_users)

    # =======================
    # TABS
    # =======================
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📊 Overview", "🧑‍💻 User Analysis", "⏳ Time Trends", "🗂 Raw Logs", "🛠 Remediation"
    ])

    # ---- OVERVIEW ----
    with tab1:
        st.subheader("Threat Overview")
        colA, colB = st.columns([2,1])

        with colA:
            fig1 = px.histogram(
                df, x="failed_logins", color="is_anomaly",
                nbins=10, title="Failed Logins Distribution",
                color_discrete_map={0:"green",1:"red"}
            )
            st.plotly_chart(fig1, use_container_width=True)

        with colB:
            pie = px.pie(
                df, names="is_anomaly", hole=0.4,
                color="is_anomaly", title="Anomaly Split",
                color_discrete_map={0:"green",1:"red"}
            )
            pie.update_traces(textinfo="percent+label")
            st.plotly_chart(pie, use_container_width=True)

        gauge = go.Figure(go.Indicator(
            mode="gauge+number+delta",
            value=max(0, 100 - risk_pct),
            title={"text":"Security Health"},
            delta={"reference":100, "increasing":{"color":"red"}},
            gauge={
                "axis":{"range":[0,100]},
                "steps":[
                    {"range":[0,40],"color":"#ff4d4d"},
                    {"range":[40,70],"color":"#ffa64d"},
                    {"range":[70,100],"color":"#85e085"}
                ],
                "bar":{"color":"darkblue"}
            }
        ))
        st.plotly_chart(gauge, use_container_width=True)

    # ---- USER ANALYSIS ----
    with tab2:
        st.subheader("User Analysis")
        user_summary = df.groupby("user")["is_anomaly"].sum().reset_index().sort_values("is_anomaly", ascending=False)
        st.dataframe(user_summary, use_container_width=True)

        bar = px.bar(user_summary, x="user", y="is_anomaly",
                     color="is_anomaly", title="Anomalies by User",
                     color_continuous_scale="Reds")
        st.plotly_chart(bar, use_container_width=True)

    # ---- TIME TRENDS ----
    with tab3:
        st.subheader("Time Trends")
        ts = df.groupby(pd.Grouper(key="date", freq="1min")).agg({"login_count":"sum","failed_logins":"sum"}).reset_index()
        if not ts.empty:
            line = px.line(ts, x="date", y=["login_count","failed_logins"], markers=True,
                           title="Activity Over Time")
            st.plotly_chart(line, use_container_width=True)
        else:
            st.info("Not enough data yet.")

    # ---- RAW LOGS ----
    with tab4:
        st.subheader("Latest Events")
        def highlight(row):
            return ["background-color: #ffcccc" if row.is_anomaly==1 else "" for _ in row]
        st.dataframe(df.tail(25).style.apply(highlight, axis=1), use_container_width=True)

    # ---- REMEDIATION ----
    with tab5:
        st.subheader("Automated Recommendations")
        def remediation(row):
            tips=[]
            if row["failed_logins"]>3: tips.append("🔑 Reset password & enable MFA")
            if row["file_access"]>40: tips.append("📂 Review file permissions")
            if row["anomaly_score"]<-0.5: tips.append("🚨 Escalate investigation")
            return " | ".join(tips) if tips else "✅ No immediate action"
        latest = df.tail(10).copy()
        latest["recommendation"] = latest.apply(remediation, axis=1)
        st.table(latest[["user","date","is_anomaly","anomaly_score","recommendation"]])

    # =======================
    # ALERT HISTORY
    # =======================
    st.markdown("### 🔔 Alert Log")
    if alert_system.alert_history:
        log_df = pd.DataFrame(alert_system.alert_history)
        st.dataframe(log_df.tail(10), use_container_width=True)
    else:
        st.info("No alerts sent yet.")
