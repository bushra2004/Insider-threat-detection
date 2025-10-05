#!/bin/bash
set -e

# Build the Docker image
docker build -t threat-detection-app:latest .

# Tag for registry
docker tag threat-detection-app:latest your-registry/threat-detection-app:latest

# Push to registry
docker push your-registry/threat-detection-app:latest

echo "✅ Image built and pushed successfully"