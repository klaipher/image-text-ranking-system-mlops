# Image-Text Ranking System: MLOps Design Doc

## 1. Overview

This document outlines the design for an image-text ranking system that enables searching for relevant images based on text queries. The system will be implemented as a complete MLOps pipeline, demonstrating model training, deployment, retraining, and monitoring. The solution will run both locally and potentially on an orchestration platform (to be decided) on an M1 MacBook Pro.

## 2. Motivation

Image search by text description is a foundational capability in modern content management systems. This project creates an educational MLOps implementation that demonstrates the complete lifecycle of a machine learning model, focusing on building a system that can improve over time as new data becomes available. This project is timely as it addresses practical MLOps skills required in industry while remaining feasible to implement on local hardware.

## 3. Success Metrics

### Business Metrics
- Demonstrable end-to-end MLOps workflow
- Educational value in showcasing MLOps best practices
- User-friendly API that enables natural language image search
- Documentation that enables others to understand and build upon the system

### Technical Metrics
- Recall@10 > 40% on test set
- Inference latency < 500ms on M1 MacBook Pro
- Complete retraining pipeline triggered by data volume
- System running successfully on both local environment and chosen orchestration platform

## 4. Requirements & Constraints

### Functional Requirements
- System must rank images based on text query relevance
- System must support retraining when new data becomes available
- System must provide API endpoints for searching images
- System must track model versions and performance

### Non-functional Requirements
- All services must run on M1 MacBook Pro with 32GB RAM
- Inference latency must be < 500ms
- Training completion time must be < 2 hours
- System must support containerization and orchestration via a platform to be selected (options include Kubeflow, SageMaker, Databricks)

### Constraints
- Limited to publicly available datasets (Flickr8K)
- Must use lightweight models suitable for M1 architecture
- Limited to open-source tools and frameworks

### 4.1 What's in-scope & out-of-scope?

**In-scope:**
- Image-text ranking model development
- Basic MLOps pipeline (training, deployment, retraining)
- REST API for model queries
- Simple data versioning and model tracking

**Out-of-scope:**
- Advanced UI for end users
- Multi-modal search capabilities beyond text-to-image
- Distributed training across multiple machines
- Advanced security features and user authentication
- High availability and fault tolerance

## 5. Methodology

### 5.1 Problem Statement

The problem is framed as a dual-encoder ranking task where image and text representations are projected into the same embedding space. Similarity between image and text embeddings determines the ranking of images for a given text query.

### 5.2 Data

**Primary Dataset:** Flickr8K, containing 8,000 images with 5 captions each
- Initial Training: 4,000 images (20,000 image-caption pairs)
- Later Training: Additional 2,000 images to simulate new data
- Validation: 1,000 images (5,000 image-caption pairs)
- Testing: 1,000 images (5,000 image-caption pairs)

### 5.3 Techniques

**Model Architecture:**
- Image Encoder: MobileNetV3 Small (lightweight and suitable for mobile devices)
- Text Encoder: Simple text embedding model
- Similarity Calculation: Cosine similarity between image and text embeddings

**Data Preparation:**
- Images: Resize to 224x224 pixels, normalize using ImageNet statistics
- Text: Basic tokenization, padding/truncation to fixed length

### 5.4 Experimentation & Validation

**Offline Evaluation:**
- Recall@K (K=1,5,10): Percentage of queries where the correct image appears in top K results
- Mean Reciprocal Rank (MRR): Average of reciprocal ranks of correct images
- Cross-validation on the training set to determine optimal hyperparameters

**Validation Strategy:**
- Regular evaluation on a held-out validation set
- Performance comparison between model versions after retraining

### 5.5 Human-in-the-loop

The initial system will not include human-in-the-loop components, but the architecture allows for future extension to incorporate user feedback to improve model performance.

## 6. Implementation

### 6.1 High-level Design

```
┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│                 │     │                 │     │                 │
│  Data Pipeline  │────►│ Model Pipeline  │────►│   Serving API   │
│                 │     │                 │     │                 │
└─────────────────┘     └─────────────────┘     └─────────────────┘
        │                       │                       │
        ▼                       ▼                       ▼
┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│                 │     │                 │     │                 │
│ Data Versioning │     │Model Versioning │     │  Monitoring &   │
│    (DVC/MinIO)  │     │   (MLflow)      │     │    Logging      │
│                 │     │                 │     │                 │
└─────────────────┘     └─────────────────┘     └─────────────────┘
        │                       │                       │
        │                       │                       │
        └───────────────────────┼───────────────────────┘
                                │
                                ▼
                        ┌─────────────────┐
                        │                 │
                        │   Orchestration │
                        │    Platform     │
                        │                 │
                        └─────────────────┘
```

### 6.2 Infra

The system will use a hybrid approach:
- Local development environment for initial development and testing
- Container orchestration platform for the final deployment (options include):
  - Kubeflow (local on M1)
  - Cloud-based platforms (SageMaker, Databricks)
  - Custom orchestration solution
- Container images optimized for M1 architecture

### 6.3 Performance (Throughput, Latency)

- Inference Service will support concurrent requests (up to 5 simultaneous users)
- Response time scaling linearly with query batch size
- Low memory footprint (< 2GB RAM per instance)
- Horizontal scaling via Kubernetes if needed

### 6.4 Security

Basic security measures will be implemented:
- API rate limiting to prevent abuse
- Input validation to prevent injection attacks
- Containerization to isolate components

### 6.5 Data Privacy

Since the project uses public datasets, no special data privacy measures are required beyond:
- Keeping data within the local environment
- Implementing proper access controls to MinIO storage

### 6.6 Monitoring & Alarms

The Monitoring Service will:
- Track inference latency, throughput, and errors
- Monitor model performance metrics
- Dashboard update frequency: every 30 seconds
- Alert when retraining is triggered (< 1 minute latency)

### 6.7 Cost

Cost is limited to development time as the system runs locally on owned hardware.
Estimated resource utilization:
- Storage: < 5GB for dataset, models, and artifacts
- Memory: < 16GB during training, < 8GB during inference
- CPU/GPU: Utilization of M1 GPU cores during training

### 6.8 Integration Points

- Data Processing Service: Interfaces with local filesystem and object storage
- Model Training Service: Interfaces with MLflow for experiment tracking
- Inference Service: REST API endpoints for external consumers
- Orchestration: Integration with selected workflow management platform

### 6.9 Risks & Uncertainties

- Performance of MobileNetV3 may be insufficient for high-quality image ranking
- M1 compatibility with some ML libraries may require workarounds
- Orchestration platform selection and setup on M1 architecture could be challenging
- Dataset size might be too small for robust generalization

## 7. Appendix

### 7.1 Alternatives

**Alternative Model Architectures:**
- CLIP (OpenAI): Better performance but significantly larger model size
- EfficientNet + DistilBERT: More accurate but higher computational requirements
- ResNet50 + Word2Vec: More established but less efficient on mobile hardware

### 7.2 Retraining Strategy

**Initial Training:**
- Train the initial model on 50% of the Flickr8K dataset
- Establish baseline performance metrics

**Retraining Triggers:**
- Data Volume Trigger: Retrain when new image-text pairs exceed 10% of current dataset size
- Manual Trigger: Allow manual initiation of retraining process for testing

**Retraining Workflow:**
1. Monitor data volume
2. When new data threshold is reached, initiate retraining pipeline
3. Train model on updated dataset
4. Evaluate new model against validation set
5. If improved, deploy updated model
6. Log retraining event and metrics

### 7.3 Microservices Breakdown

**Data Processing Service:**
- Process Flickr8K dataset into training/validation/test splits
- Track when new data is added
- Should handle the full Flickr8K dataset processing in < 30 minutes
- Storage requirements: < 2GB for processed dataset

**Model Training Service:**
- Train the dual-encoder model
- Support incremental retraining
- Evaluate model performance
- Support for batch sizes up to 64 images
- Efficient GPU utilization on M1 architecture
- Checkpointing to resume interrupted training

**Model Registry Service:**
- Store trained model versions
- Track model metadata
- Storage capacity for at least 10 model versions
- Retrieval time for model artifacts < 5 seconds

**Inference Service:**
- Provide endpoints for text-to-image ranking
- Serve latest model version
- Support concurrent requests (up to 5 simultaneous users)
- Response time scaling linearly with query batch size
- Low memory footprint (< 2GB RAM per instance)

**Monitoring Service:**
- Monitor model performance metrics
- Trigger retraining when data volume threshold is exceeded
- Dashboard update frequency: every 30 seconds
- Alert latency < 1 minute for retraining triggers

### 7.4 Tools and Technologies

**Infrastructure:**
- Docker: Containerization
- Orchestration options:
  - Kubeflow: ML workflow orchestration (local)
  - SageMaker: AWS managed ML platform
  - Databricks: Unified analytics platform
- MinIO: Object storage

**ML Framework:**
- PyTorch: Core ML framework
- Torchvision: Image processing

**MLOps:**
- MLflow: Experiment tracking
- DVC: Basic data versioning
- FastAPI: API development