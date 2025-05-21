#!/bin/bash

# ANSI color codes
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

echo -e "${BLUE}=========================================================${NC}"
echo -e "${BLUE}     Starting Image-Text Ranking MLOps System           ${NC}"
echo -e "${BLUE}=========================================================${NC}"

# Check if Docker is running
if ! docker info > /dev/null 2>&1; then
    echo -e "${RED}Error: Docker is not running. Please start Docker and try again.${NC}"
    exit 1
fi

# Create data directories if they don't exist
echo -e "${YELLOW}Creating necessary directories...${NC}"
mkdir -p dags logs plugins data/images models

# Make sure we have a model file
if [ ! -f "models/best_model.pth" ]; then
    echo -e "${YELLOW}No model file found in models/best_model.pth${NC}"
    echo -e "${YELLOW}You may need to train a model first or place an existing model in the models directory${NC}"
fi

# Start the services
echo -e "${YELLOW}Starting services with Docker Compose...${NC}"
docker-compose up -d

# Wait for services to be ready
echo -e "${YELLOW}Waiting for services to start...${NC}"
echo -e "${YELLOW}This may take a few minutes...${NC}"

# Check if API is ready
MAX_RETRIES=30
RETRY_INTERVAL=10
RETRIES=0

echo -e "${YELLOW}Checking if API is ready...${NC}"
while [ $RETRIES -lt $MAX_RETRIES ]; do
    if curl -s http://localhost:8000/health > /dev/null; then
        echo -e "${GREEN}API is ready!${NC}"
        break
    else
        echo -e "${YELLOW}API not ready yet, waiting...${NC}"
        RETRIES=$((RETRIES+1))
        sleep $RETRY_INTERVAL
    fi
done

if [ $RETRIES -eq $MAX_RETRIES ]; then
    echo -e "${RED}API did not become ready in the expected time.${NC}"
    echo -e "${RED}Check docker-compose logs with: docker-compose logs api${NC}"
fi

# Display service URLs
echo -e "${GREEN}=========================================================${NC}"
echo -e "${GREEN}     Services are now running!                          ${NC}"
echo -e "${GREEN}=========================================================${NC}"
echo -e "${GREEN}FastAPI: http://localhost:8000${NC}"
echo -e "${GREEN}MinIO Console: http://localhost:9001 (login: minioadmin/minioadmin)${NC}"
echo -e "${GREEN}Airflow: http://localhost:8080 (login: airflow/airflow)${NC}"
echo -e "${GREEN}=========================================================${NC}"

# Usage instructions
echo -e "${BLUE}To run the demo:${NC}"
echo -e "${YELLOW}python demo.py --images_dir data/images --query \"a dog running on the beach\" --top_k 5${NC}"
echo -e ""
echo -e "${BLUE}To stop the services:${NC}"
echo -e "${YELLOW}docker-compose down${NC}"
echo -e ""
echo -e "${BLUE}To view logs:${NC}"
echo -e "${YELLOW}docker-compose logs -f api${NC}" 