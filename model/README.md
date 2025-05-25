# Image-Text Ranking Baseline Model

This repository contains the baseline model for the image-text ranking system using a dual-encoder architecture. The model learns to embed images and text into a shared space where similarity can be computed for ranking tasks.

## Overview

The system implements a dual-encoder model with:
- **Image Encoder**: MobileNetV3 Small for efficient image feature extraction
- **Text Encoder**: LSTM-based encoder for text sequence processing
- **Contrastive Learning**: Temperature-scaled cosine similarity for training
- **MLOps Integration**: MLflow for experiment tracking and model versioning

## Architecture

```
Text Input → Tokenization → Embedding → LSTM → Projection → L2 Norm → Text Embedding
                                                                           ↓
                                                                    Cosine Similarity
                                                                           ↓
Image Input → MobileNetV3 → Global Pool → Projection → L2 Norm → Image Embedding
```

### Key Features

- **Lightweight Design**: Optimized for M1 MacBook Pro with MPS acceleration
- **Scalable Inference**: Pre-computed image databases for fast search
- **Comprehensive Metrics**: Recall@K, MRR, and similarity statistics
- **Reproducible Training**: Fixed seeds and deterministic operations
- **Production Ready**: Clean API for integration with serving systems

## Installation

### Prerequisites

- Python 3.12
- uv package manager (0.7+)

### Setup

1. **Install dependencies:**
   ```bash
   cd model
   uv sync
   ```

2. **Activate virtual environment:**
   ```bash
   source .venv/bin/activate
   ```

3. **Dataset will be downloaded automatically:**
   ```bash
   # No setup required! Dataset downloads from Hugging Face automatically
   # during first training run
   ```

## Dataset

The model uses the **Flickr8K dataset from Hugging Face** containing:
- 8,000 images with 5 captions each
- Training: 60% (4,800 images, 24,000 pairs)
- Validation: 20% (1,600 images, 8,000 pairs)  
- Test: 20% (1,600 images, 8,000 pairs)

Data is automatically downloaded from Hugging Face and processed on first run.

## Usage

### Training

1. **Basic training:**
   ```bash
   python train.py
   ```

2. **Custom parameters:**
   ```bash
   python train.py --epochs 15 --batch-size 64 --lr 2e-4 --device mps
   ```

3. **Resume from checkpoint:**
   ```bash
   python train.py --resume models/checkpoint_epoch_5.pt
   ```

### Training Configuration

Key hyperparameters can be modified in `src/config.py`:

```python
@dataclass
class ModelConfig:
    # Model architecture
    embedding_dim: int = 256
    vocab_size: int = 10000
    temperature: float = 0.07
    
    # Training
    batch_size: int = 32
    learning_rate: float = 1e-4
    num_epochs: int = 10
```

### Inference

1. **Quick demo:**
   ```bash
   python inference_demo.py
   ```

2. **Custom inference:**
   ```bash
   python inference_demo.py --image-dir /path/to/images --model-path models/final_model.pt
   ```

3. **Programmatic usage:**
   ```python
   from src.inference import create_predictor
   
   # Create predictor
   predictor = create_predictor()
   
   # Text-to-image search
   results = predictor.rank_images(
       text_query="a dog playing in the park",
       image_paths=["image1.jpg", "image2.jpg"],
       top_k=5
   )
   
   # Build image database for fast search
   predictor.build_image_database(all_image_paths)
   results = predictor.search_database("beautiful sunset", top_k=10)
   ```

## Model Performance

### Target Metrics (from specs)
- **Recall@10**: > 40%
- **Inference Latency**: < 500ms on M1 MacBook Pro
- **Training Time**: < 2 hours

### Evaluation Metrics

The model is evaluated using:
- **Recall@K** (K=1,5,10): Percentage of queries where correct image appears in top-K
- **Mean Reciprocal Rank (MRR)**: Average reciprocal rank of correct images
- **Bidirectional Evaluation**: Both text-to-image and image-to-text retrieval

## Project Structure

```
model/
├── src/
│   ├── __init__.py
│   ├── config.py              # Configuration classes
│   │   ├── __init__.py
│   │   └── dataset.py         # Flickr8K dataset handling
│   ├── models/
│   │   ├── __init__.py
│   │   └── encoders.py        # Dual-encoder architecture
│   ├── training/
│   │   ├── __init__.py
│   │   ├── trainer.py         # Training loop and checkpoints
│   │   └── metrics.py         # Evaluation metrics
│   └── inference/
│       ├── __init__.py
│       └── predictor.py       # Inference and ranking
├── train.py                   # Main training script
├── inference_demo.py          # Demo and testing script
├── pyproject.toml            # Dependencies and project config
└── README.md                 # This file
```

## Configuration

### Model Configuration

```python
# Image processing
image_size: (224, 224)
image_mean: (0.485, 0.456, 0.406)  # ImageNet normalization
image_std: (0.229, 0.224, 0.225)

# Text processing  
max_text_length: 64
vocab_size: 10000

# Architecture
embedding_dim: 256
text_encoder_dim: 512
dropout_rate: 0.1
```

### Hardware Optimization

- **M1 MacBook Pro**: Uses MPS acceleration when available
- **Memory Efficient**: Gradient accumulation and mixed precision ready
- **Batch Processing**: Configurable batch sizes for memory constraints

## MLflow Integration

The training automatically tracks:
- **Hyperparameters**: Learning rate, batch size, architecture settings
- **Metrics**: Loss, recall@K, MRR per epoch  
- **Model Artifacts**: Best model, final model, checkpoints
- **Training History**: Complete training logs and curves

Access MLflow UI:
```bash
mlflow ui --backend-store-uri file:./mlruns
```

## API Integration

The model is designed for easy integration with serving APIs:

```python
# For FastAPI integration
from src.inference import create_predictor

predictor = create_predictor()

@app.post("/search")
async def search_images(query: str):
    results = predictor.search_database(query, top_k=10)
    return {"results": results}
```

## Development

### Code Quality

```bash
# Format code
black src/
isort src/

# Type checking
mypy src/

# Testing
pytest tests/
```

### Adding New Features

1. **New Model Architecture**: Extend `src/models/encoders.py`
2. **Custom Metrics**: Add to `src/training/metrics.py`
3. **Data Augmentation**: Modify transforms in `src/data/dataset.py`
4. **Inference Methods**: Extend `src/inference/predictor.py`

## Troubleshooting

### Common Issues

1. **MPS Device Error**: Fallback to CPU if MPS is not available
   ```python
   device = "cpu"  # In config.py
   ```

2. **Memory Issues**: Reduce batch size
   ```bash
   python train.py --batch-size 16
   ```

3. **Dataset Download**: Manual download if automatic download fails
   ```bash
   # Download from https://huggingface.co/datasets/jxie/flickr8k
   # Place in data/processed/
   ```

4. **Vocabulary Errors**: Clear processed data and rebuild
   ```bash
   rm -rf data/processed/
   python train.py  # Will rebuild dataset
   ```

## Performance Optimization

### Training Speed
- Use larger batch sizes if memory allows
- Enable mixed precision training (future enhancement)
- Use data loading with multiple workers

### Inference Speed
- Pre-build image databases for frequently searched collections
- Batch process multiple queries together
- Use quantized models for production (future enhancement)

## Future Enhancements

1. **Model Improvements**:
   - Vision Transformer (ViT) image encoder
   - Transformer-based text encoder
   - Cross-modal attention mechanisms

2. **Training Enhancements**:
   - Hard negative mining
   - Curriculum learning
   - Multi-scale training

3. **Production Features**:
   - Model quantization
   - ONNX export
   - TensorRT optimization
   - Distributed inference

## Contributing

1. Follow the existing code structure
2. Add tests for new functionality
3. Update documentation
4. Ensure backwards compatibility

## License

This project is part of the MLOps coursework and follows the specifications outlined in `specs.md`.
