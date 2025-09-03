import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
import joblib
import os

print("=== Training Baseline Model ===")

# Load the processed data
def load_data():
    df = pd.read_csv('data/processed/behavioral_features.csv')
    print(f"Loaded {len(df)} records")
    return df

# Prepare features for model
def prepare_features(df):
    features = ['login_count', 'failed_logins', 'file_access_count', 
                'avg_login_time', 'late_logins', 'failed_login_ratio']
    X = df[features].fillna(0)
    return X

# Train Isolation Forest model
def train_model(X):
    print("Training Isolation Forest model...")
    model = IsolationForest(contamination=0.1, random_state=42)
    model.fit(X)
    return model

# Evaluate and save results
def save_results(model, X, df):
    # Get predictions
    predictions = model.predict(X)
    scores = model.decision_function(X)
    
    # Add to dataframe
    df['anomaly_score'] = scores
    df['is_anomaly'] = np.where(predictions == -1, 1, 0)
    
    # Save results
    os.makedirs('models', exist_ok=True)
    joblib.dump(model, 'models/isolation_forest.pkl')
    df.to_csv('data/processed/predictions.csv', index=False)
    
    # Show results
    anomaly_count = df['is_anomaly'].sum()
    print(f"Detected {anomaly_count} anomalies ({anomaly_count/len(df)*100:.1f}%)")
    
    return df

if __name__ == "__main__":
    # Load and prepare data
    df = load_data()
    X = prepare_features(df)
    
    # Train model
    model = train_model(X)
    
    # Save results
    results_df = save_results(model, X, df)
    
    print("=== Model Training Complete ===")
    print("Model saved: models/isolation_forest.pkl")
    print("Predictions saved: data/processed/predictions.csv")