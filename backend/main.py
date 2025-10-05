# backend/main.py

from fastapi import FastAPI, Depends
from sqlalchemy.orm import Session
from backend import db_models
from backend.database import engine, get_db

# Create tables in Postgres (only first run)
db_models.Base.metadata.create_all(bind=engine)

app = FastAPI(title="Insider Threat Backend")

# Health check endpoint
@app.get("/health")
def health_check():
    return {"status": "ok"}

# List users
@app.get("/users")
def get_users(db: Session = Depends(get_db)):
    users = db.query(db_models.User).all()
    return users

# List logs
@app.get("/logs")
def get_logs(db: Session = Depends(get_db)):
    logs = db.query(db_models.Log).all()
    return logs

# List alerts
@app.get("/alerts")
def get_alerts(db: Session = Depends(get_db)):
    alerts = db.query(db_models.Alert).all()
    return alerts

# List anomalies
@app.get("/anomalies")
def get_anomalies(db: Session = Depends(get_db)):
    anomalies = db.query(db_models.Anomaly).all()
    return anomalies
