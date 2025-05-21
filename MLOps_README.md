# Image-Text Ranking MLOps System

This repository contains a complete MLOps system for an image-text ranking model, which enables searching images by text descriptions. The system includes a trained model, FastAPI wrapper, data storage in MinIO, Airflow for data ingestion, and Docker containerization.

## System Architecture

The system consists of the following components:

1. **Baseline Model**: A dual-encoder architecture using MobileNetV3-small for images and DistilBERT for text, with a shared embedding space.
2. **FastAPI Service**: REST API for model inference, providing endpoints for encoding images, encoding text, and searching images by text.
3. **MinIO Storage**: S3-compatible object storage for storing training and inference data.
4. **Airflow Workflow**: DAG for regularly ingesting new data into MinIO storage.
5. **Docker Containers**: All components are containerized using Docker and orchestrated with Docker Compose.

## Setup and Installation

### Prerequisites

- Docker and Docker Compose
- Python 3.8+

### Quick Start

1. Clone the repository:
   ```bash
   git clone <repository-url>
   cd <repository-name>
   ```

2. Start the services:
   ```bash
   docker-compose up -d
   ```

3. Wait for all services to start (this may take a few minutes).

4. Access the services:
   - FastAPI: http://localhost:8000
   - MinIO Console: http://localhost:9001 (login with minioadmin/minioadmin)
   - Airflow: http://localhost:8080 (login with airflow/airflow)

## Using the API

The API provides the following endpoints:

- `GET /health`: Check the health of the API
- `POST /encode_text`: Encode a text query into an embedding vector
- `POST /encode_image`: Encode an image into an embedding vector
- `POST /search_images_by_text`: Search for images using a text query

### Demo

A demo script is provided to showcase the API functionality:

```bash
python demo.py --images_dir data/images --query "a dog running on the beach" --top_k 5
```

The script will:
1. Check the API health
2. Encode images from the specified directory
3. Search for images matching the text query
4. Display the top-k matching images with similarity scores

## Data Management

### Data Storage

The system uses MinIO, an S3-compatible object storage, to store the Flickr8K dataset. Data is organized into:
- `flickr8k/images/`: Contains image files
- `flickr8k/text/`: Contains text annotations

### Data Ingestion

An Airflow DAG is set up to regularly ingest new data into MinIO. The DAG:
1. Downloads the Flickr8K dataset
2. Processes the data
3. Uploads it to the appropriate MinIO buckets

To manually trigger the data ingestion:
1. Access the Airflow UI at http://localhost:8080
2. Navigate to the DAGs page
3. Find the `flickr8k_data_ingestion` DAG
4. Click on the "Trigger DAG" button

### Data Loader

A custom dataset loader (`MinIOFlickr8kDataset`) is provided to:
1. Connect to MinIO
2. Load images and captions
3. Transform data for model training

## Model Training

The model is trained using:
- MobileNetV3-small for image encoding
- DistilBERT for text encoding
- InfoNCE contrastive loss
- Support for GPU, MPS (M1 Mac), or CPU

To train the model:

```bash
python train.py --batch_size 32 --epochs 20
```

## Docker Containers

The system includes the following containers:
- `api`: FastAPI application
- `minio`: MinIO object storage
- `minio-setup`: Sets up MinIO buckets
- `airflow-webserver`: Airflow web UI
- `airflow-scheduler`: Airflow scheduler
- `postgres`: Database for Airflow

## Customization

### Environment Variables

The services can be customized using environment variables:
- `MODEL_PATH`: Path to the model file
- `EMBEDDING_DIM`: Dimension of the embedding space
- `MINIO_ENDPOINT`: MinIO server endpoint
- `MINIO_ACCESS_KEY`: MinIO access key
- `MINIO_SECRET_KEY`: MinIO secret key
- `MINIO_BUCKET`: MinIO bucket name

## Troubleshooting

### Common Issues

- **API Health Check Failing**: Ensure the model is properly loaded in the API container
- **MinIO Connection Issues**: Check MinIO credentials and endpoint
- **Airflow DAG Failing**: Check the Airflow logs for detailed error messages

### Logs

To view container logs:
```bash
docker-compose logs -f api
docker-compose logs -f airflow-webserver
```

## License

This project is licensed under the MIT License. 