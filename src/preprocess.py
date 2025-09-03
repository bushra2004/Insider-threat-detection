import pandas as pd
import numpy as np
import os

print("=== Insider Threat Data Preprocessing ===")
print("Creating sample data...")

# Create simple sample data without complex datetime operations
data = []
for i in range(300):
    user = f"user{np.random.randint(1, 6)}"
    
    # Create simple timestamp (avoiding complex datetime operations)
    day = np.random.randint(1, 8)
    hour = np.random.randint(0, 24)
    
    # Normal activities
    action = np.random.choice(['login', 'file_access', 'logout', 'email'])
    status = 'success' if np.random.random() > 0.1 else 'failure'
    
    # Add some suspicious activity for user3 (after hours)
    if user == 'user3' and np.random.random() > 0.7:
        hour = np.random.randint(18, 24)  # After hours access
    
    timestamp = f"2023-01-{day:02d} {hour:02d}:00:00"
    
    data.append({
        'timestamp': timestamp,
        'user': user,
        'action': action,
        'status': status,
        'resource': f'resource_{np.random.randint(1, 10)}'
    })

# Create DataFrame
df = pd.DataFrame(data)
os.makedirs('data/raw', exist_ok=True)
df.to_csv('data/raw/sample_logs.csv', index=False)
print("Sample data created: data/raw/sample_logs.csv")

# Process the data
print("Processing data...")
df['timestamp'] = pd.to_datetime(df['timestamp'])
df['date'] = df['timestamp'].dt.date
df['hour'] = df['timestamp'].dt.hour

# Create behavioral features
features = []
for user, user_data in df.groupby('user'):
    for date, daily_data in user_data.groupby('date'):
        # Count activities
        login_count = len(daily_data[daily_data['action'] == 'login'])
        failed_logins = len(daily_data[(daily_data['action'] == 'login') & (daily_data['status'] == 'failure')])
        file_access = len(daily_data[daily_data['action'] == 'file_access'])
        
        # Time analysis
        login_hours = daily_data[daily_data['action'] == 'login']['hour']
        avg_login_time = login_hours.mean() if not login_hours.empty else 0
        late_logins = len(login_hours[login_hours > 18])
        
        features.append({
            'user': user,
            'date': str(date),  # Convert to string to avoid serialization issues
            'login_count': login_count,
            'failed_logins': failed_logins,
            'file_access_count': file_access,
            'avg_login_time': avg_login_time,
            'late_logins': late_logins,
            'failed_login_ratio': failed_logins / max(login_count, 1) if login_count > 0 else 0
        })

# Save features
features_df = pd.DataFrame(features)
os.makedirs('data/processed', exist_ok=True)
features_df.to_csv('data/processed/behavioral_features.csv', index=False)
print("Features saved: data/processed/behavioral_features.csv")

print("=== Preprocessing Complete ===")
print(f"Processed {len(features_df)} user-day records")
print("You can now run: python src\\train_baseline.py")