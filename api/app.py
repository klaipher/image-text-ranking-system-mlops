"""
FastAPI application for Image-Text Ranking Model.

This API provides endpoints for:
- Image-text ranking inference
- Model training
- Data management
- Health checks
"""

import logging
import os
import sys
from contextlib import asynccontextmanager
from pathlib import Path

import uvicorn
from fastapi import BackgroundTasks, FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
#
# sys.path.append('/app/')

from services.model_service import ModelService
from services.training_service import TrainingService
from services.data_service import DataService

from api_models.schemas import (
    TextToImageRequest, ImageToTextRequest, BatchRankingRequest,
    TrainingRequest, TrainingStatus, ModelInfo, HealthCheck
)

# Configure logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# Global services
model_service = ModelService()
training_service = TrainingService()
data_service = DataService()


@asynccontextmanager
async def lifespan(_app: FastAPI):
    """Application lifespan manager."""

    logger.info("Starting Image-Text Ranking API...")

    try:
        # Load model if available
        await model_service.load_model()

        logger.info("✅ API services initialized successfully")

    except Exception as e:
        logger.error(f"❌ Failed to initialize services: {e}")
        raise

    yield

    # Cleanup
    logger.info("Shutting down API services...")


# Create FastAPI app
app = FastAPI(
    title="Image-Text Ranking API",
    description="API for image-text ranking model serving and training",
    version="1.0.0",
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Health Check Endpoints
@app.get("/health", response_model=HealthCheck)
async def health_check():
    """Health check endpoint."""
    return HealthCheck(
        status="healthy",
        message="Image-Text Ranking API is running",
        model_loaded=model_service.is_model_loaded(),
        training_active=training_service.is_training_active() if training_service else False
    )


@app.get("/health/detailed")
async def detailed_health_check():
    """Detailed health check with service status."""
    return {
        "status": "healthy",
        "services": {
            "model_service": {
                "status": "ready" if model_service and model_service.is_model_loaded() else "not_ready",
                "model_path": model_service.get_model_info().model_path if model_service else None
            },
            "training_service": {
                "status": "ready" if training_service else "not_ready",
                "active_training": training_service.is_training_active() if training_service else False
            },
            "data_service": {
                "status": "ready" if data_service else "not_ready",
                "minio_connected": await data_service.check_connection() if data_service else False
            }
        },
        "environment": {
            "python_version": sys.version,
            "working_directory": os.getcwd()
        }
    }


# Model Information Endpoints
@app.get("/model/info", response_model=ModelInfo)
async def get_model_info():
    """Get information about the loaded model."""
    if not model_service or not model_service.is_model_loaded():
        raise HTTPException(status_code=404, detail="No model loaded")

    return model_service.get_model_info()


@app.get("/model/status")
async def get_model_status():
    """Get detailed model status."""
    if not model_service:
        raise HTTPException(status_code=503, detail="Model service not available")

    return {
        "loaded": model_service.is_model_loaded(),
        "info": model_service.get_model_info() if model_service.is_model_loaded() else None,
        "device": model_service.get_device() if model_service.is_model_loaded() else None
    }


# Inference Endpoints
@app.post("/inference/text-to-image")
async def text_to_image_ranking(request: TextToImageRequest):
    """Rank images based on text query."""
    if not model_service or not model_service.is_model_loaded():
        raise HTTPException(status_code=404, detail="No model loaded")

    try:
        results = await model_service.rank_images(
            text_query=request.text_query,
            image_paths=request.image_paths,
            top_k=request.top_k
        )

        return {
            "query": request.text_query,
            "results": results,
            "total_images": len(request.image_paths)
        }

    except Exception as e:
        logger.error(f"Text-to-image ranking failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/inference/image-to-text")
async def image_to_text_ranking(request: ImageToTextRequest):
    """Rank texts based on image query."""
    if not model_service or not model_service.is_model_loaded():
        raise HTTPException(status_code=404, detail="No model loaded")

    try:
        results = await model_service.rank_texts(
            image_path=request.image_path,
            texts=request.texts,
            top_k=request.top_k
        )

        return {
            "image_path": request.image_path,
            "results": results,
            "total_texts": len(request.texts)
        }

    except Exception as e:
        logger.error(f"Image-to-text ranking failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/inference/batch-ranking")
async def batch_ranking(request: BatchRankingRequest):
    """Perform batch ranking for multiple queries."""
    if not model_service or not model_service.is_model_loaded():
        raise HTTPException(status_code=404, detail="No model loaded")

    try:
        results = await model_service.batch_ranking(
            text_queries=request.text_queries,
            image_paths=request.image_paths,
            top_k=request.top_k
        )

        return {
            "queries": request.text_queries,
            "results": results,
            "total_images": len(request.image_paths)
        }

    except Exception as e:
        logger.error(f"Batch ranking failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/inference/upload-image")
async def upload_and_rank(
        file: UploadFile = File(...),
        text_query: str = None,
        top_k: int = 10
):
    """Upload an image and perform ranking."""
    if not model_service or not model_service.is_model_loaded():
        raise HTTPException(status_code=404, detail="No model loaded")

    # Validate file type
    if not file.content_type.startswith('image/'):
        raise HTTPException(status_code=400, detail="File must be an image")

    try:
        # Save uploaded file temporarily
        upload_dir = Path("/tmp/uploads")
        upload_dir.mkdir(exist_ok=True)

        file_path = upload_dir / file.filename
        with open(file_path, "wb") as f:
            content = await file.read()
            f.write(content)

        if text_query:
            # Text-to-image: rank this image against the text query
            results = await model_service.rank_images(
                text_query=text_query,
                image_paths=[str(file_path)],
                top_k=1
            )

            response = {
                "type": "text_to_image",
                "text_query": text_query,
                "uploaded_file": file.filename,
                "similarity_score": results[0].score if results else 0.0
            }
        else:
            # Get available dataset images for ranking
            dataset_images = await data_service.get_sample_images(limit=100)

            if not dataset_images:
                raise HTTPException(status_code=404, detail="No dataset images available")

            # Image-to-image: find similar images
            results = await model_service.find_similar_images(
                query_image_path=str(file_path),
                candidate_image_paths=dataset_images,
                top_k=top_k
            )

            response = {
                "type": "image_to_image",
                "uploaded_file": file.filename,
                "results": results,
                "total_candidates": len(dataset_images)
            }

        # Clean up uploaded file
        file_path.unlink()

        return response

    except Exception as e:
        logger.error(f"Upload and rank failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# Training Endpoints
@app.post("/training/start")
async def start_training(request: TrainingRequest, background_tasks: BackgroundTasks):
    """Start model training."""
    if not training_service:
        raise HTTPException(status_code=503, detail="Training service not available")

    if training_service.is_training_active():
        raise HTTPException(status_code=409, detail="Training already in progress")

    try:
        # Start training in background
        training_id = await training_service.start_training(
            config=request.dict(),
            background_tasks=background_tasks
        )

        return {
            "message": "Training started successfully",
            "training_id": training_id,
            "status": "started"
        }

    except Exception as e:
        logger.error(f"Failed to start training: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/training/status", response_model=TrainingStatus)
async def get_training_status():
    """Get current training status."""
    if not training_service:
        raise HTTPException(status_code=503, detail="Training service not available")

    return training_service.get_training_status()


@app.get("/training/history")
async def get_training_history():
    """Get training history."""
    if not training_service:
        raise HTTPException(status_code=503, detail="Training service not available")

    return training_service.get_training_history()


@app.post("/training/stop")
async def stop_training():
    """Stop current training."""
    if not training_service:
        raise HTTPException(status_code=503, detail="Training service not available")

    if not training_service.is_training_active():
        raise HTTPException(status_code=404, detail="No active training to stop")

    try:
        await training_service.stop_training()
        return {"message": "Training stopped successfully"}

    except Exception as e:
        logger.error(f"Failed to stop training: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# Data Management Endpoints
@app.get("/data/info")
async def get_data_info():
    """Get information about available datasets."""
    if not data_service:
        raise HTTPException(status_code=503, detail="Data service not available")

    try:
        return await data_service.get_data_info()

    except Exception as e:
        logger.error(f"Failed to get data info: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/data/download")
async def download_data_from_minio(background_tasks: BackgroundTasks):
    """Download latest dataset from MinIO."""
    if not data_service:
        raise HTTPException(status_code=503, detail="Data service not available")

    try:
        # Start download in background
        download_id = await data_service.download_latest_dataset(background_tasks)

        return {
            "message": "Dataset download started",
            "download_id": download_id,
            "status": "started"
        }

    except Exception as e:
        logger.error(f"Failed to start data download: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/data/samples")
async def get_sample_data(limit: int = 10):
    """Get sample images and captions from the dataset."""
    if not data_service:
        raise HTTPException(status_code=503, detail="Data service not available")

    try:
        samples = await data_service.get_sample_data(limit=limit)
        return {
            "samples": samples,
            "total_returned": len(samples)
        }

    except Exception as e:
        logger.error(f"Failed to get sample data: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# Model Management Endpoints
@app.post("/model/reload")
async def reload_model():
    """Reload the model from disk."""
    if not model_service:
        raise HTTPException(status_code=503, detail="Model service not available")

    try:
        await model_service.reload_model()
        return {
            "message": "Model reloaded successfully",
            "info": model_service.get_model_info()
        }

    except Exception as e:
        logger.error(f"Failed to reload model: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/model/load")
async def load_model(model_path: str = None):
    """Load a specific model."""
    if not model_service:
        raise HTTPException(status_code=503, detail="Model service not available")

    try:
        await model_service.load_model(model_path=model_path)
        return {
            "message": "Model loaded successfully",
            "info": model_service.get_model_info()
        }

    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# Demo Endpoints
@app.get("/demo/queries")
async def get_demo_queries():
    """Get sample queries for demo purposes."""
    return {
        "text_queries": [
            "a dog playing in the park",
            "people walking on the street",
            "a beautiful sunset over the ocean",
            "children playing football",
            "a cat sitting on a chair",
            "cars parked on the road",
            "birds flying in the sky",
            "flowers blooming in a garden"
        ],
        "sample_images": await data_service.get_sample_images(limit=20) if data_service else []
    }


@app.get("/demo/search")
async def demo_search(query: str = "a dog playing", top_k: int = 5):
    """Demo search endpoint with default query."""
    if not model_service or not model_service.is_model_loaded():
        raise HTTPException(status_code=404, detail="No model loaded")

    # Get sample images for demo
    sample_images = await data_service.get_sample_images(limit=50) if data_service else []

    if not sample_images:
        raise HTTPException(status_code=404, detail="No sample images available")

    try:
        results = await model_service.rank_images(
            text_query=query,
            image_paths=sample_images,
            top_k=top_k
        )

        return {
            "query": query,
            "results": results,
            "total_images": len(sample_images),
            "demo": True
        }

    except Exception as e:
        logger.error(f"Demo search failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    uvicorn.run(
        "app:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )
