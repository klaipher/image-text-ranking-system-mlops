# Quick Start Guide

This guide will help you get the image-text ranking baseline model up and running quickly.

## Prerequisites

- Python 3.12
- uv package manager (0.7+)
- M1 MacBook Pro (or compatible hardware)

## Setup

1. **Navigate to the model directory:**
   ```bash
   cd model
   ```

2. **Install dependencies:**
   ```bash
   uv sync
   ```

3. **Verify setup:**
   ```bash
   uv run python test_setup.py
   ```

## Training the Model

### Option 1: Quick Training (Small Dataset)
```bash
# Train with default settings (10 epochs, batch size 32)
uv run python train.py --epochs 5 --batch-size 16
```

### Option 2: Full Training
```bash
# Train with full settings
uv run python train.py --epochs 15 --batch-size 32 --lr 1e-4
```

### Training with Custom Parameters
```bash
uv run python train.py \
    --epochs 20 \
    --batch-size 64 \
    --lr 2e-4 \
    --device mps \
    --experiment-name "my_experiment" \
    --run-name "baseline_v1"
```

## Dataset Setup

The model uses the Flickr8K dataset from Hugging Face. You have two options:

### Option 1: Automatic Download (Recommended)
The dataset will be downloaded automatically from Hugging Face during the first training run.
No API keys or credentials required!

### Option 2: Manual Download
1. Download from [Hugging Face Flickr8K Dataset](https://huggingface.co/datasets/jxie/flickr8k)
2. Place in `data/processed/` directory
3. Ensure the structure is:
   ```
   data/processed/
   ├── images/
   │   ├── image_000001.jpg
   │   └── ...
   └── captions.txt
   ```

## Inference and Demo

### Quick Demo (After Training)
```bash
uv run python inference_demo.py
```

### Custom Inference
```bash
# Demo with custom image directory
uv run python inference_demo.py --image-dir /path/to/images

# Demo with specific model
uv run python inference_demo.py --model-path models/final_model
```

### Programmatic Usage
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

print(results)
```

## Monitoring Training

### MLflow UI
```bash
# Start MLflow UI (in a separate terminal)
uv run mlflow ui --backend-store-uri file:./mlruns
```

Then open http://localhost:5000 in your browser.

### Training Logs
- Models saved to: `models/`
- Training history: `models/training_history.json`
- MLflow logs: `mlruns/`

## Expected Performance

Based on the specifications:
- **Target Recall@10**: > 40%
- **Training Time**: < 2 hours on M1 MacBook Pro
- **Inference Latency**: < 500ms per query
- **Model Size**: ~10M parameters

## Troubleshooting

### Common Issues

1. **Memory Issues**:
   ```bash
   # Reduce batch size
   uv run python train.py --batch-size 16
   ```

2. **Device Issues**:
   ```bash
   # Force CPU if MPS has issues
   uv run python train.py --device cpu
   ```

3. **Dataset Issues**:
   ```bash
   # Clear and rebuild dataset
   rm -rf data/processed/
   uv run python train.py
   ```

### Getting Help

1. Check the full README.md for detailed documentation
2. Run the test setup to verify installation:
   ```bash
   uv run python test_setup.py
   ```

## Next Steps

After training the baseline model:

1. **Evaluate Performance**: Check the final test metrics
2. **Experiment**: Try different hyperparameters
3. **API Integration**: Use the model in your API (see main project)
4. **Improvements**: Consider model architecture enhancements

## File Structure

```
model/
├── src/                    # Source code
│   ├── config.py          # Configuration
│   ├── data/              # Data processing
│   ├── models/            # Model architectures
│   ├── training/          # Training logic
│   └── inference/         # Inference logic
├── train.py               # Training script
├── inference_demo.py      # Demo script
├── test_setup.py          # Setup verification
├── pyproject.toml         # Dependencies
├── README.md              # Full documentation
└── QUICKSTART.md          # This file
``` 