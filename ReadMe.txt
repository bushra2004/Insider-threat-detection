
cd insider-threat-detection

streamlit run dashboard\app.py

----- API services are running on this ---------- 
run : uvicorn src.api_server:app --reload --host 0.0.0.0 --port 8000
open browser : http://localhost:8000/docs or http://localhost:8000/redoc

