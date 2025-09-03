import json
import time
import random
from kafka import KafkaProducer
from datetime import datetime, timedelta
import asyncio
import websockets
import threading

class RealTimeDataProducer:
    def __init__(self):
        self.producer = KafkaProducer(
            bootstrap_servers=['localhost:9092'],
            value_serializer=lambda v: json.dumps(v).encode('utf-8')
        )
        self.users = [f'user{i}' for i in range(1, 11)]
        self.actions = ['login', 'logout', 'file_access', 'email', 'db_query']
        self.resources = [f'resource_{i}' for i in range(1, 21)]
        
    def generate_log_event(self):
        """Generate realistic log data with anomalies"""
        user = random.choice(self.users)
        timestamp = datetime.now().isoformat()
        action = random.choice(self.actions)
        
        # Introduce anomalies for specific users
        is_anomalous = False
        if user in ['user3', 'user7'] and random.random() < 0.3:
            action = 'file_access'
            status = 'success'
            resource = 'confidential_file'
            is_anomalous = True
        else:
            status = 'success' if random.random() > 0.1 else 'failure'
            resource = random.choice(self.resources)
        
        event = {
            'timestamp': timestamp,
            'user': user,
            'action': action,
            'status': status,
            'resource': resource,
            'is_anomalous': is_anomalous,
            'source_ip': f'192.168.1.{random.randint(1, 100)}'
        }
        return event
    
    def start_production(self, topic='user-logs'):
        """Start producing real-time data"""
        print("Starting real-time data production...")
        while True:
            event = self.generate_log_event()
            self.producer.send(topic, value=event)
            print(f"Produced: {event}")
            time.sleep(random.uniform(0.1, 1.0))  # Real-time interval

# WebSocket server for real-time dashboard updates
class WebSocketServer:
    async def send_updates(self, websocket, path):
        producer = RealTimeDataProducer()
        while True:
            event = producer.generate_log_event()
            await websocket.send(json.dumps(event))
            await asyncio.sleep(1)

def start_websocket_server():
    server = WebSocketServer()
    start_server = websockets.serve(server.send_updates, "localhost", 8765)
    asyncio.get_event_loop().run_until_complete(start_server)
    asyncio.get_event_loop().run_forever()

if __name__ == "__main__":
    # Start WebSocket server in background thread
    ws_thread = threading.Thread(target=start_websocket_server, daemon=True)
    ws_thread.start()
    
    # Start Kafka producer
    producer = RealTimeDataProducer()
    producer.start_production()