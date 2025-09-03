from fastapi import FastAPI, Depends, HTTPException, Security
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
import pandas as pd
import numpy as np
import joblib
from datetime import datetime, timedelta
import jwt
from jwt.exceptions import InvalidTokenError
import uvicorn

# JWT Configuration
SECRET_KEY = "your-super-secret-jwt-key-here"
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 30

# FastAPI app
app = FastAPI(title="Insider Threat Detection API", version="1.0.0")

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Security
security = HTTPBearer()

# Pydantic models
class UserLogin(BaseModel):
    username: str
    password: str

class AnomalyDetectionRequest(BaseModel):
    user_data: List[dict]
    model_type: str = "isolation_forest"

class AnomalyDetectionResponse(BaseModel):
    predictions: List[dict]
    anomaly_count: int
    risk_score: float
    model_used: str

class ThreatAlert(BaseModel):
    user_id: str
    severity: str
    description: str
    timestamp: datetime
    confidence: float

# Mock user database (in production, use real database)
users_db = {
    "admin": {"password": "admin123", "role": "admin"},
    "analyst": {"password": "analyst123", "role": "analyst"}
}

# Load ML models
try:
    iso_forest = joblib.load('models/iso_forest.pkl')
    scaler = joblib.load('models/scaler.pkl')
    models_loaded = True
except:
    models_loaded = False

# JWT functions
def create_access_token(data: dict):
    to_encode = data.copy()
    expire = datetime.utcnow() + timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt

def verify_token(credentials: HTTPAuthorizationCredentials = Security(security)):
    try:
        payload = jwt.decode(credentials.credentials, SECRET_KEY, algorithms=[ALGORITHM])
        return payload
    except InvalidTokenError:
        raise HTTPException(status_code=401, detail="Invalid token")

# API endpoints
@app.post("/auth/login")
async def login(user_login: UserLogin):
    if user_login.username not in users_db:
        raise HTTPException(status_code=401, detail="Invalid credentials")
    
    if users_db[user_login.username]["password"] != user_login.password:
        raise HTTPException(status_code=401, detail="Invalid credentials")
    
    access_token = create_access_token(
        data={"sub": user_login.username, "role": users_db[user_login.username]["role"]}
    )
    
    return {"access_token": access_token, "token_type": "bearer"}

@app.post("/detect/anomalies", response_model=AnomalyDetectionResponse)
async def detect_anomalies(
    request: AnomalyDetectionRequest,
    token_payload: dict = Depends(verify_token)
):
    if not models_loaded:
        raise HTTPException(status_code=503, detail="ML models not loaded")
    
    # Convert to DataFrame
    df = pd.DataFrame(request.user_data)
    
    # Prepare features
    features = ['login_count', 'failed_logins', 'file_access_count', 
                'avg_login_time', 'late_logins', 'failed_login_ratio']
    
    X = df[features].fillna(0)
    X_scaled = scaler.transform(X)
    
    # Predict anomalies
    predictions = iso_forest.predict(X_scaled)
    scores = iso_forest.decision_function(X_scaled)
    
    # Prepare response
    results = []
    for i, (_, row) in enumerate(df.iterrows()):
        results.append({
            "user": row['user'] if 'user' in row else f"user_{i}",
            "anomaly_score": float(scores[i]),
            "is_anomaly": bool(predictions[i] == -1),
            "features": row.to_dict()
        })
    
    anomaly_count = sum(1 for r in results if r['is_anomaly'])
    risk_score = (anomaly_count / len(results)) * 100
    
    return AnomalyDetectionResponse(
        predictions=results,
        anomaly_count=anomaly_count,
        risk_score=risk_score,
        model_used=request.model_type
    )

@app.get("/alerts/recent")
async def get_recent_alerts(
    limit: int = 10,
    token_payload: dict = Depends(verify_token)
):
    # Mock recent alerts (in production, fetch from database)
    mock_alerts = [
        ThreatAlert(
            user_id=f"user{np.random.randint(1, 10)}",
            severity=np.random.choice(["high", "medium", "low"]),
            description="Suspicious after-hours access",
            timestamp=datetime.now() - timedelta(hours=np.random.randint(1, 24)),
            confidence=round(np.random.uniform(0.7, 0.95), 2)
        ) for _ in range(limit)
    ]
    
    return mock_alerts

@app.get("/system/health")
async def system_health():
    return {
        "status": "healthy" if models_loaded else "degraded",
        "models_loaded": models_loaded,
        "timestamp": datetime.now().isoformat()
    }

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)