# MLOps Image-Text Ranking System

A complete MLOps pipeline for image-text ranking with Airflow, MinIO object storage, and FastAPI serving.

## 🚀 Features

- **Complete MLOps Pipeline**: Data ingestion → Processing → Training → Serving
- **FastAPI REST API**: Production-ready API for model inference and training
- **MinIO Object Storage**: S3-compatible storage for datasets and models
- **Apache Airflow**: Automated data pipeline orchestration
- **Docker Compose**: One-command deployment
- **Model Training API**: Train models via REST API
- **Real-time Inference**: Image-text ranking with sub-500ms latency

## 📋 System Architecture

```
┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│                 │     │                 │     │                 │
│  Airflow DAGs   │────►│  MinIO Storage  │────►│   FastAPI       │
│  (Data Pipeline)│     │  (Object Store) │     │   (Serving)     │
│                 │     │                 │     │                 │
└─────────────────┘     └─────────────────┘     └─────────────────┘
        │                       │                       │
        ▼                       ▼                       ▼
┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│                 │     │                 │     │                 │
│ Flickr8K Data   │     │ Processed Data  │     │ Training API    │
│ Download &      │     │ & Model         │     │ & Inference     │
│ Processing      │     │ Artifacts       │     │ Endpoints       │
│                 │     │                 │     │                 │
└─────────────────┘     └─────────────────┘     └─────────────────┘
```

## 🛠 Quick Start

### Prerequisites

- Docker and Docker Compose
- 8GB+ RAM
- 10GB+ disk space

### 1. Clone and Start Services

```bash
git clone <repository>
cd <project-directory>

# Start all services
docker-compose up -d

# Check service status
docker-compose ps
```

### 2. Access Services

- **FastAPI Documentation**: http://localhost:8000/docs
- **Airflow UI**: http://localhost:8080 (admin/admin)
- **MinIO Console**: http://localhost:9001 (minioadmin/minioadmin)

### 3. Set Up Data Pipeline

```bash
# Set up Airflow variables
docker-compose exec airflow-scheduler python /opt/airflow/dags/setup_airflow_vars.py setup

# Trigger data pipeline
curl -X POST "http://localhost:8080/api/v1/dags/flickr8k_with_minio/dagRuns" \
  -H "Content-Type: application/json" \
  -H "Authorization: Basic YWRtaW46YWRtaW4=" \
  -d '{"dag_run_id": "manual_'$(date +%Y%m%d_%H%M%S)'", "conf": {}}'
```

### 4. Train a Model

```bash
# Start training via API
curl -X POST "http://localhost:8000/training/start" \
  -H "Content-Type: application/json" \
  -d '{
    "epochs": 5,
    "batch_size": 32,
    "learning_rate": 0.001,
    "data_source": "minio"
  }'

# Check training status
curl http://localhost:8000/training/status
```

### 5. Test Inference

```bash
# Get sample queries
curl http://localhost:8000/demo/queries

# Test text-to-image search
curl http://localhost:8000/demo/search?query="a%20dog%20playing"&top_k=5
```

## 📡 API Endpoints

### Health & Status
- `GET /health` - Service health check
- `GET /health/detailed` - Detailed service status
- `GET /model/info` - Model information
- `GET /model/status` - Model loading status

### Inference
- `POST /inference/text-to-image` - Rank images by text query
- `POST /inference/image-to-text` - Rank texts by image
- `POST /inference/batch-ranking` - Batch ranking operations
- `POST /inference/upload-image` - Upload and rank image

### Training
- `POST /training/start` - Start model training
- `GET /training/status` - Get training status
- `GET /training/history` - Training history
- `POST /training/stop` - Stop current training

### Data Management
- `GET /data/info` - Dataset information
- `POST /data/download` - Download data from MinIO
- `GET /data/samples` - Get sample data

### Demo
- `GET /demo/queries` - Sample queries for testing
- `GET /demo/search` - Demo search with default query

## 🔧 Configuration

### Environment Variables

#### API Service
```bash
MINIO_ENDPOINT=minio:9000
MINIO_ACCESS_KEY=minioadmin
MINIO_SECRET_KEY=minioadmin
MINIO_BUCKET=image-text-data
MINIO_SECURE=false
```

#### Airflow Variables
```bash
# Set via setup script or Airflow UI
flickr8k_dataset_size=full
flickr8k_sample_ratio=1.0
minio_endpoint=minio:9000
minio_bucket=image-text-data
use_object_storage=true
```

### Model Configuration
Located in `model/src/config.py`:
- Image encoder: MobileNetV3 Small
- Text encoder: LSTM-based
- Embedding dimension: 256
- Target device: Auto-detect (MPS/CUDA/CPU)

## 📊 Model Performance

### Target Metrics
- **Recall@10**: > 40%
- **Inference Latency**: < 500ms
- **Training Time**: < 2 hours

### Model Architecture
- **Image Encoder**: MobileNetV3 Small (lightweight, mobile-optimized)
- **Text Encoder**: LSTM with attention
- **Loss Function**: Contrastive loss
- **Optimization**: AdamW with warmup

## 🗄 Data Pipeline

### Airflow DAGs

1. **`flickr8k_data_pipeline`**: Basic data download and processing
2. **`flickr8k_with_sync`**: Enhanced pipeline with model synchronization
3. **`flickr8k_with_minio`**: Full pipeline with MinIO integration

### Pipeline Steps
1. **Download**: Fetch Flickr8K from Hugging Face
2. **Process**: Resize images, create captions JSON
3. **Split**: Train/Val/Test splits (60/20/20)
4. **Validate**: Data integrity checks
5. **Upload**: Store in MinIO object storage
6. **Sync**: Synchronize with model training directory

## 🚀 Development

### Local Development

```bash
# Install dependencies
cd model
uv pip install -e .

# Run API locally
cd api
uvicorn app:app --reload --port 8000

# Run training
cd model
python train.py --epochs 5
```

### Testing

```bash
# Test model setup
cd model
python test_setup.py

# Test Hugging Face dataset
python test_hf_dataset.py

# Test inference
python inference_demo.py --model-path models/final_model.pt
```

### Adding New Features

1. **New API Endpoints**: Add to `api/app.py`
2. **New Services**: Create in `api/services/`
3. **New Models**: Add to `api/models/schemas.py`
4. **New DAGs**: Add to `data_flows/dags/`

## 📁 Project Structure

```
├── api/                    # FastAPI application
│   ├── app.py             # Main application
│   ├── services/          # Business logic services
│   ├── models/            # Pydantic schemas
│   └── Dockerfile         # API container
├── data_flows/            # Airflow pipeline
│   ├── dags/              # Airflow DAGs
│   └── utils/             # Utilities (MinIO client)
├── model/                 # ML model code
│   ├── src/               # Source code
│   ├── train.py           # Training script
│   └── inference_demo.py  # Inference demo
├── compose.yml            # Docker Compose configuration
└── README.md              # This file
```

## 🔍 Monitoring & Debugging

### Logs
```bash
# API logs
docker-compose logs api

# Airflow logs
docker-compose logs airflow-scheduler

# MinIO logs
docker-compose logs minio
```

### Health Checks
```bash
# Check all services
curl http://localhost:8000/health/detailed

# Check MinIO
curl http://localhost:9000/minio/health/live

# Check Airflow
curl http://localhost:8080/health
```

### Troubleshooting

#### Model Not Found
1. Check if training completed successfully
2. Verify model files in `/app/models/`
3. Check API logs for path issues

#### MinIO Connection Issues
1. Verify MinIO service is running
2. Check network connectivity between services
3. Validate credentials and bucket name

#### Training Failures
1. Check GPU/MPS availability
2. Verify data availability
3. Monitor memory usage

## 🚦 Deployment

### Production Considerations

1. **Security**:
   - Change default MinIO credentials
   - Use secrets management
   - Enable HTTPS/TLS

2. **Scaling**:
   - Use external database for Airflow
   - Scale API with load balancer
   - Use distributed MinIO setup

3. **Monitoring**:
   - Add Prometheus metrics
   - Set up log aggregation
   - Configure alerting

### Cloud Deployment

The system can be deployed on:
- AWS (ECS/EKS + S3)
- GCP (GKE + Cloud Storage)
- Azure (AKS + Blob Storage)

## 📝 License

This project is licensed under the MIT License.

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make changes and test
4. Submit a pull request

## 📞 Support

For issues and questions:
1. Check the troubleshooting section
2. Review logs for error messages
3. Open an issue with detailed information 