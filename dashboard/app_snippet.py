import streamlit as st
import requests
import numpy as np
import pandas as pd

st.title("Insider Threat — Anomaly Scoring Demo")
api_url = st.text_input("API URL", "http://localhost:8000/score")

uploaded = st.file_uploader(
    "Upload sample feature windows (.npy or .csv)", type=["npy", "csv"]
)

if uploaded:
    if uploaded.name.endswith(".npy"):
        arr = np.load(uploaded)
        st.write("Loaded NumPy array with shape", arr.shape)
    else:
        df = pd.read_csv(uploaded)
        st.write("Loaded CSV — ensure it represents flattened windows or valid shape.")
        arr = df.values

    try:
        payload = {"sequences": arr.tolist()}
        r = requests.post(api_url, json=payload, timeout=30)
        r.raise_for_status()
        resp = r.json()
        scores = np.array(resp["scores"])
        st.write("Suggested threshold:", resp.get("suggested_threshold"))
        st.line_chart(scores)
        st.dataframe(pd.DataFrame({"score": scores}))
    except Exception as e:
        st.error(f"Error calling API: {e}")
else:
    st.info("Upload an .npy with shape (n_samples, seq_len, n_features) to test.")
