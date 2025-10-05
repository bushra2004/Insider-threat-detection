# backend/db_models.py

from sqlalchemy import Column, Integer, String, DateTime, Float, Boolean, ForeignKey, Text
from sqlalchemy.orm import declarative_base, relationship
from datetime import datetime

Base = declarative_base()

# ===================
# Users Table
# ===================
class User(Base):
    __tablename__ = "users"

    id = Column(Integer, primary_key=True, index=True)
    username = Column(String(50), unique=True, index=True, nullable=False)
    email = Column(String(100), unique=True, nullable=False)
    role = Column(String(20), default="analyst")  # admin / analyst
    created_at = Column(DateTime, default=datetime.utcnow)

    logs = relationship("Log", back_populates="user")
    alerts = relationship("Alert", back_populates="user")


# ===================
# Logs Table
# ===================
class Log(Base):
    __tablename__ = "logs"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"))
    activity = Column(Text, nullable=False)
    risk_score = Column(Float, default=0.0)
    timestamp = Column(DateTime, default=datetime.utcnow)

    user = relationship("User", back_populates="logs")


# ===================
# Alerts Table
# ===================
class Alert(Base):
    __tablename__ = "alerts"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"))
    message = Column(Text, nullable=False)
    severity = Column(String(20), default="low")  # low, medium, high
    is_resolved = Column(Boolean, default=False)
    timestamp = Column(DateTime, default=datetime.utcnow)

    user = relationship("User", back_populates="alerts")


# ===================
# Anomalies Table
# ===================
class Anomaly(Base):
    __tablename__ = "anomalies"

    id = Column(Integer, primary_key=True, index=True)
    description = Column(Text, nullable=False)
    detected_at = Column(DateTime, default=datetime.utcnow)
    confidence = Column(Float, default=0.0)  # ML model confidence
