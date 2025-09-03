---------- To run project -----
step1 -
cd insider-threat-detection
insider-threat-env\Scripts\activate

step2 -
# Generate sample data and features
python src\preprocess.py

# Train basic ML model
python src\train_baseline.py

# Train advanced LSTM model
python src\train_advanced.py

cd insider-threat-detection
insider-threat-env\Scripts\activate
streamlit run dashboard\app.py

----- API services are running on this ---------- 
run : uvicorn src.api_server:app --reload --host 0.0.0.0 --port 8000
open browser : http://localhost:8000/docs or http://localhost:8000/redoc

