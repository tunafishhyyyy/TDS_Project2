#!/bin/bash

# Enhanced system with temp.py
echo "🚀 Building and running enhanced data analysis system..."

# Stop any existing container
docker stop temp-analysis-api 2>/dev/null || true
docker rm temp-analysis-api 2>/dev/null || true

# Build the Docker image
echo "📦 Building Docker image..."
docker build -t temp-analysis-api .

# Run the container on port 8001
echo "🏃 Starting container on port 8001..."
docker run -d \
  --name temp-analysis-api \
  --env-file .env \
  -p 8001:8001 \
  -e PORT=8001 \
  temp-analysis-api \
  python run_temp_server.py

echo "✅ Enhanced system running at http://localhost:8001"
echo "📊 Main system still at http://localhost:8000"
echo "🔧 Test with: curl 'http://localhost:8001/api/' -F 'questions.txt=@test_wikipedia.txt'"
echo "🩺 Diagnostics at: curl 'http://localhost:8001/summary'"
