# Image-Text Ranking Model

This project implements a ranking model for searching images based on text prompts using the Flickr8K dataset and MobileNetV3-small architecture on PyTorch. The model is optimized to run on MacBook Pro with M1 chip.

## Features

- Text-to-image retrieval using a dual-encoder approach
- MobileNetV3-small as image encoder for efficiency
- DistilBERT as text encoder
- Contrastive learning with InfoNCE loss
- Optimized for running on M1 Mac (uses MPS acceleration when available)
- Evaluation metrics: Recall@K, Mean Rank, Median Rank

## Setup

1. Clone this repository:
```bash
git clone <repository-url>
cd <repository-name>
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Download and preprocess the Flickr8K dataset:
```bash
python preprocess_flickr8k.py --download
```

## Usage

### Training

To train the model:
```bash
python train.py --batch_size 32 --epochs 20
```

Additional training options:
```
--data_dir: Path to data directory (default: data/flickr8k)
--model_dir: Path to save models (default: models)
--embedding_dim: Dimension of the embedding space (default: 512)
--temperature: Temperature parameter for softmax (default: 0.07)
--batch_size: Batch size (default: 32)
--lr: Learning rate (default: 5e-5)
--weight_decay: Weight decay (default: 1e-4)
--epochs: Number of epochs (default: 20)
--save_every: Save model every n epochs (default: 5)
--num_workers: Number of workers for data loading (default: 4)
```

### Evaluation

To evaluate the model:
```bash
python evaluate.py
```

### Image Search

To search for images based on a text query:
```bash
python search.py --query "your text query" --top_k 5 --show
```

Search options:
```
--query: Text query for image search (required)
--top_k: Number of top images to retrieve (default: 9)
--use_test_set: Use test set for search (default: False)
--output_dir: Directory to save results (default: results)
--show: Show the results (default: False)
```

### Run the entire pipeline

To run the entire pipeline (preprocessing, training, evaluation, and search):
```bash
./run.sh
```

To download the dataset automatically:
```bash
./run.sh --download
```

## Model Architecture

The model consists of two encoders:

1. **Image Encoder**: MobileNetV3-small pretrained on ImageNet, with a projection layer to the embedding space
2. **Text Encoder**: DistilBERT with a projection layer to the embedding space

During training, the model learns to maximize the similarity between matching image-text pairs and minimize the similarity between non-matching pairs.

## Performance Considerations for M1 Mac

- The model uses MPS (Metal Performance Shaders) acceleration when available
- Batch size and number of workers can be adjusted based on available memory
- DistilBERT is used instead of BERT for efficiency
- MobileNetV3-small is used for efficient image encoding

## Results

The model is evaluated using the following metrics:
- Recall@1, Recall@5, Recall@10: Percentage of queries where the correct image is in the top k results
- Mean Rank: Average rank of the correct image
- Median Rank: Median rank of the correct image

## License

This project is licensed under the MIT License - see the LICENSE file for details. 