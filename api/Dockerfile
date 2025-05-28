FROM python:3.12-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

# Copy model directory first (contains dependencies)
COPY model/ /app/model/

# Install model dependencies
WORKDIR /app/model
RUN uv pip install --system -e .

# Copy API code
WORKDIR /app
COPY api/ /app/

# Copy data flows utils for MinIO client
COPY data_flows/dags/utils/ /app/data_flows/dags/utils/

# Install API dependencies
RUN uv sync

# Create directories for data and models
RUN mkdir -p /app/data/processed /app/models /tmp/uploads

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Run the API
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000", "--reload"] 