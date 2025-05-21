import os
import io
import torch
import numpy as np
from PIL import Image
from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from transformers import DistilBertTokenizer
from torchvision import transforms
from model import ImageTextRankingModel
from typing import List, Dict, Any, Optional
import base64

# Initialize FastAPI app
app = FastAPI(
    title="Image-Text Ranking API",
    description="API for image-text retrieval using a dual-encoder model",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Constants
MODEL_PATH = os.environ.get("MODEL_PATH", "models/best_model.pth")
EMBEDDING_DIM = int(os.environ.get("EMBEDDING_DIM", "512"))
DEVICE = torch.device("cuda" if torch.cuda.is_available() else 
                     "mps" if torch.backends.mps.is_available() else 
                     "cpu")

# Image transformation pipeline
transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Load model and tokenizer
@app.on_event("startup")
async def startup_event():
    global model, tokenizer
    
    try:
        # Initialize tokenizer
        tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
        
        # Initialize model
        model = ImageTextRankingModel(embedding_dim=EMBEDDING_DIM)
        
        # Load model weights
        if os.path.exists(MODEL_PATH):
            checkpoint = torch.load(MODEL_PATH, map_location=DEVICE)
            model.load_state_dict(checkpoint['model_state_dict'])
            model.to(DEVICE)
            model.eval()
            print(f"Model loaded from {MODEL_PATH}")
        else:
            raise FileNotFoundError(f"Model file not found at {MODEL_PATH}")
    
    except Exception as e:
        print(f"Error loading model: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to load model: {str(e)}")

@app.get("/")
async def root():
    return {"message": "Image-Text Ranking API"}

@app.get("/health")
async def health_check():
    if 'model' not in globals():
        return JSONResponse(
            status_code=503,
            content={"status": "error", "message": "Model not loaded"}
        )
    return {"status": "healthy", "model_loaded": True}

@app.post("/encode_text")
async def encode_text(text: str = Form(...)):
    """
    Encode a text query into an embedding vector
    """
    try:
        # Tokenize the text
        tokens = tokenizer(
            text,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=128
        )
        
        # Move tokens to the appropriate device
        input_ids = tokens['input_ids'].to(DEVICE)
        attention_mask = tokens['attention_mask'].to(DEVICE)
        
        # Get text embedding
        with torch.no_grad():
            embedding = model.get_text_embeddings(input_ids, attention_mask)
        
        # Convert embedding to list for JSON serialization
        embedding_list = embedding.cpu().numpy().tolist()[0]
        
        return {
            "text": text,
            "embedding": embedding_list
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/encode_image")
async def encode_image(image: UploadFile = File(...)):
    """
    Encode an image into an embedding vector
    """
    try:
        # Read image file
        image_content = await image.read()
        img = Image.open(io.BytesIO(image_content)).convert("RGB")
        
        # Apply transformations
        img_tensor = transform(img).unsqueeze(0).to(DEVICE)
        
        # Get image embedding
        with torch.no_grad():
            embedding = model.get_image_embeddings(img_tensor)
        
        # Convert embedding to list for JSON serialization
        embedding_list = embedding.cpu().numpy().tolist()[0]
        
        return {
            "filename": image.filename,
            "embedding": embedding_list
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/search_images_by_text")
async def search_images_by_text(
    text: str = Form(...),
    image_embeddings: List[Dict[str, Any]] = Form(...),
    top_k: Optional[int] = Form(5)
):
    """
    Search for images using a text query against a provided list of image embeddings
    """
    try:
        # Encode the text query
        text_response = await encode_text(text)
        text_embedding = np.array(text_response["embedding"])
        
        # Calculate similarity with all image embeddings
        similarities = []
        for img_data in image_embeddings:
            img_embedding = np.array(img_data["embedding"])
            similarity = np.dot(text_embedding, img_embedding)
            similarities.append({
                "image_id": img_data.get("image_id", ""),
                "filename": img_data.get("filename", ""),
                "similarity": float(similarity)
            })
        
        # Sort by similarity (highest first)
        sorted_results = sorted(similarities, key=lambda x: x["similarity"], reverse=True)
        
        # Return top k results
        return {
            "query": text,
            "results": sorted_results[:top_k]
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True) 