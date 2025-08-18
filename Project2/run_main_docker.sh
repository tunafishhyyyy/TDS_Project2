#!/bin/bash

# Enhanced TDS Data Analysis API system with main_app.py
echo "🚀 Building and running TDS Data Analysis API..."

# Stop any existing container
docker stop main-analysis-api 2>/dev/null || true
docker rm main-analysis-api 2>/dev/null || true

# Build the Docker image
echo "📦 Building Docker image..."
docker build -t main-analysis-api .

# Run the container on port 8001
echo "🏃 Starting container on port 8001..."
docker run -d \
  --name main-analysis-api \
  --env-file .env \
  -p 8001:8001 \
  -e PORT=8001 \
  main-analysis-api \
  python run_main_server.py

echo "✅ TDS Data Analysis API running at http://localhost:8001"
echo "📊 Multi-modal analysis system ready"
echo "🔧 Test with: curl 'http://localhost:8001/api/' -F 'questions.txt=@test_wikipedia.txt'"
echo "🩺 Diagnostics at: curl 'http://localhost:8001/summary'"
