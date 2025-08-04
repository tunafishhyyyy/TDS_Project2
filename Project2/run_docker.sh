#!/bin/bash
# Build and run the Data Analysis API Docker container

IMAGE_NAME="data-analysis-api"
CONTAINER_NAME="data-analysis-api-container"
PORT=8000

# Build the Docker image

echo "Building Docker image: $IMAGE_NAME"
docker build -t $IMAGE_NAME .

# Remove any existing container with the same name
if [ $(docker ps -aq -f name=$CONTAINER_NAME) ]; then
    echo "Removing existing container: $CONTAINER_NAME"
    docker rm -f $CONTAINER_NAME
fi

# Run the Docker container

echo "Starting Docker container: $CONTAINER_NAME"
docker run -d --name $CONTAINER_NAME -p $PORT:80 --env-file .env $IMAGE_NAME

echo "Container started. API is available at http://localhost:$PORT/"
