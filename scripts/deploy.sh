#!/bin/bash
set -e

# Pull latest images
docker-compose pull

# Deploy with zero downtime
docker-compose up -d --scale threat-detection-app=3 --no-recreate

# Health check
echo "🩺 Performing health check..."
sleep 30
curl -f http://localhost:8501/_stcore/health || exit 1

echo "✅ Deployment successful!"