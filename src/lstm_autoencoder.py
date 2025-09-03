import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import LSTM, Dense, Input, RepeatVector, TimeDistributed
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_curve, auc
import json
import os

class AdvancedLSTMAutoencoder:
    def __init__(self, time_steps=10, n_features=8, hidden_units=64):
        self.time_steps = time_steps
        self.n_features = n_features
        self.hidden_units = hidden_units
        self.model = self._build_advanced_model()
        self.scaler = StandardScaler()
        
    def _build_advanced_model(self):
        """Build advanced LSTM Autoencoder with attention mechanism"""
        # Encoder
        inputs = Input(shape=(self.time_steps, self.n_features))
        
        # Bidirectional LSTM layers
        encoded = tf.keras.layers.Bidirectional(
            LSTM(self.hidden_units, activation='relu', return_sequences=True)
        )(inputs)
        
        encoded = tf.keras.layers.Bidirectional(
            LSTM(self.hidden_units//2, activation='relu', return_sequences=False)
        )(encoded)
        
        # Attention mechanism
        attention = Dense(1, activation='tanh')(encoded)
        attention = tf.keras.layers.Flatten()(attention)
        attention = tf.keras.layers.Activation('softmax')(attention)
        attention = tf.keras.layers.RepeatVector(self.hidden_units)(attention)
        attention = tf.keras.layers.Permute([2, 1])(attention)
        
        # Decoder
        decoded = RepeatVector(self.time_steps)(encoded)
        decoded = LSTM(self.hidden_units//2, activation='relu', return_sequences=True)(decoded)
        decoded = LSTM(self.hidden_units, activation='relu', return_sequences=True)(decoded)
        decoded = TimeDistributed(Dense(self.n_features))(decoded)
        
        autoencoder = Model(inputs, decoded)
        autoencoder.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), 
                           loss='mse', metrics=['mae'])
        
        return autoencoder
    
    def create_sequences(self, data, sequence_length=10):
        """Create overlapping sequences for time series analysis"""
        sequences = []
        for i in range(len(data) - sequence_length + 1):
            sequence = data[i:i+sequence_length]
            sequences.append(sequence)
        return np.array(sequences)
    
    def train(self, X_train, epochs=100, batch_size=32, validation_split=0.2):
        """Train the advanced autoencoder"""
        # Scale data
        X_scaled = self.scaler.fit_transform(X_train)
        
        # Create sequences
        X_seq = self.create_sequences(X_scaled, self.time_steps)
        
        # Train-validation split
        split_idx = int(len(X_seq) * (1 - validation_split))
        X_train_seq, X_val_seq = X_seq[:split_idx], X_seq[split_idx:]
        
        callbacks = [
            EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True),
            ModelCheckpoint('models/lstm_autoencoder_advanced.h5', 
                           monitor='val_loss', save_best_only=True)
        ]
        
        history = self.model.fit(
            X_train_seq, X_train_seq,
            epochs=epochs,
            batch_size=batch_size,
            validation_data=(X_val_seq, X_val_seq),
            callbacks=callbacks,
            verbose=1,
            shuffle=True
        )
        
        return history
    
    def detect_anomalies(self, X_test, threshold_percentile=95):
        """Detect anomalies using reconstruction error"""
        X_scaled = self.scaler.transform(X_test)
        X_seq = self.create_sequences(X_scaled, self.time_steps)
        
        # Predict and calculate reconstruction error
        reconstructions = self.model.predict(X_seq)
        mse = np.mean(np.square(X_seq - reconstructions), axis=(1, 2))
        
        # Dynamic threshold based on percentile
        threshold = np.percentile(mse, threshold_percentile)
        anomalies = mse > threshold
        
        return mse, anomalies, threshold
    
    def evaluate_model(self, X_test, y_true=None):
        """Comprehensive model evaluation"""
        mse, anomalies, threshold = self.detect_anomalies(X_test)
        
        results = {
            'reconstruction_error': mse.tolist(),
            'anomalies': anomalies.astype(int).tolist(),
            'threshold': float(threshold),
            'anomaly_rate': float(np.mean(anomalies))
        }
        
        if y_true is not None:
            # Calculate precision-recall curve
            precision, recall, _ = precision_recall_curve(y_true, mse)
            pr_auc = auc(recall, precision)
            results['pr_auc'] = float(pr_auc)
        
        return results

# Advanced feature engineering
class AdvancedFeatureEngineer:
    def __init__(self):
        self.feature_scalers = {}
        
    def extract_advanced_features(self, df):
        """Extract sophisticated behavioral features"""
        features = []
        
        # Time-based features
        df['hour'] = pd.to_datetime(df['timestamp']).dt.hour
        df['day_of_week'] = pd.to_datetime(df['timestamp']).dt.dayofweek
        df['is_weekend'] = df['day_of_week'].isin([5, 6]).astype(int)
        
        # Advanced behavioral metrics
        for user, user_data in df.groupby('user'):
            user_data = user_data.sort_values('timestamp')
            
            # Session-based features
            user_data['time_since_last'] = user_data['timestamp'].diff().dt.total_seconds().fillna(0)
            
            # Rolling statistics
            user_data['login_rolling_avg'] = user_data['login_count'].rolling(window=3).mean().fillna(0)
            user_data['failed_rolling_avg'] = user_data['failed_logins'].rolling(window=3).mean().fillna(0)
            
            # Behavioral patterns
            user_data['access_intensity'] = user_data['login_count'] + user_data['file_access_count']
            user_data['suspicion_score'] = (
                user_data['failed_logins'] * 0.4 +
                user_data['late_logins'] * 0.3 +
                (user_data['file_access_count'] / user_data['login_count'].clip(lower=1)) * 0.3
            )
            
            features.append(user_data)
        
        features_df = pd.concat(features, ignore_index=True)
        return features_df

if __name__ == "__main__":
    # Load and preprocess data
    df = pd.read_csv('data/processed/behavioral_features.csv')
    
    # Advanced feature engineering
    engineer = AdvancedFeatureEngineer()
    advanced_features = engineer.extract_advanced_features(df)
    
    # Train LSTM Autoencoder
    lstm_ae = AdvancedLSTMAutoencoder(time_steps=10, n_features=12, hidden_units=128)
    
    # Select features for training
    feature_columns = ['login_count', 'failed_logins', 'file_access_count', 
                      'avg_login_time', 'late_logins', 'failed_login_ratio',
                      'hour', 'is_weekend', 'time_since_last', 'login_rolling_avg',
                      'access_intensity', 'suspicion_score']
    
    X_train = advanced_features[feature_columns].fillna(0).values
    
    # Train model
    history = lstm_ae.train(X_train, epochs=50, batch_size=32)
    
    # Detect anomalies
    mse, anomalies, threshold = lstm_ae.detect_anomalies(X_train)
    
    # Save results
    advanced_features['lstm_anomaly_score'] = mse
    advanced_features['lstm_anomaly'] = anomalies.astype(int)
    advanced_features.to_csv('data/processed/advanced_lstm_predictions.csv', index=False)
    
    print(f"Advanced LSTM training complete. Anomaly rate: {np.mean(anomalies):.3f}")