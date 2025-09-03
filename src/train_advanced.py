import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.cluster import DBSCAN
from sklearn.svm import OneClassSVM
from sklearn.preprocessing import StandardScaler
import joblib
import os

print("=== Advanced Model Training ===")

# Load data
df = pd.read_csv('data/processed/behavioral_features.csv')
print(f"Loaded {len(df)} records")

# Prepare features
features = ['login_count', 'failed_logins', 'file_access_count', 
            'avg_login_time', 'late_logins', 'failed_login_ratio']
X = df[features].fillna(0)

# Scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train multiple models
print("Training multiple anomaly detection models...")

# 1. Isolation Forest
iso_forest = IsolationForest(contamination=0.1, random_state=42)
iso_predictions = iso_forest.fit_predict(X_scaled)
iso_scores = iso_forest.decision_function(X_scaled)

# 2. One-Class SVM
svm = OneClassSVM(nu=0.1, kernel='rbf')
svm_predictions = svm.fit_predict(X_scaled)
svm_scores = svm.decision_function(X_scaled)

# 3. DBSCAN (clustering-based)
dbscan = DBSCAN(eps=0.5, min_samples=5)
dbscan_predictions = dbscan.fit_predict(X_scaled)

# Combine results
df['iso_forest_score'] = iso_scores
df['iso_forest_anomaly'] = np.where(iso_predictions == -1, 1, 0)

df['svm_score'] = svm_scores
df['svm_anomaly'] = np.where(svm_predictions == -1, 1, 0)

df['dbscan_anomaly'] = np.where(dbscan_predictions == -1, 1, 0)

# Ensemble voting
df['ensemble_anomaly'] = (
    df['iso_forest_anomaly'] + 
    df['svm_anomaly'] + 
    df['dbscan_anomaly']
)
df['final_anomaly'] = np.where(df['ensemble_anomaly'] >= 2, 1, 0)

# Save results
os.makedirs('models', exist_ok=True)
joblib.dump(iso_forest, 'models/iso_forest.pkl')
joblib.dump(svm, 'models/oneclass_svm.pkl')
joblib.dump(scaler, 'models/scaler.pkl')

df.to_csv('data/processed/advanced_predictions.csv', index=False)

# Results
anomaly_count = df['final_anomaly'].sum()
print(f"Detected {anomaly_count} anomalies ({anomaly_count/len(df)*100:.1f}%)")
print("Models saved: iso_forest.pkl, oneclass_svm.pkl, scaler.pkl")
print("Predictions saved: data/processed/advanced_predictions.csv")