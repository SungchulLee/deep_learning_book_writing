"""
Module 64.05: FastAPI Deployment
=================================

This module demonstrates deploying models using FastAPI, a modern, high-performance
web framework for building APIs with Python. FastAPI offers automatic documentation,
request validation, and async support.

Learning Objectives:
-------------------
1. Build APIs with FastAPI and Pydantic
2. Implement async request handling
3. Use automatic API documentation (Swagger/OpenAPI)
4. Add request/response validation
5. Compare FastAPI vs Flask

Time Estimate: 90 minutes
Difficulty: Intermediate  
Prerequisites: Module 64.04 (Flask API)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from fastapi import FastAPI, File, UploadFile, HTTPException, BackgroundTasks
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, validator
from typing import List, Optional, Dict, Any
from PIL import Image
import io
import base64
import numpy as np
import time
import asyncio
from datetime import datetime


# ============================================================================
# PART 1: Pydantic Models for Request/Response Validation
# ============================================================================

class PredictionRequest(BaseModel):
    """
    Request model for base64-encoded image prediction.
    
    Pydantic provides automatic validation:
    - Type checking
    - Field constraints
    - Custom validators
    - Automatic documentation
    """
    image: str = Field(..., description="Base64-encoded image")
    top_k: Optional[int] = Field(3, ge=1, le=10, description="Number of top predictions to return")
    
    @validator('image')
    def validate_base64(cls, v):
        """Validate that image is valid base64."""
        try:
            base64.b64decode(v)
            return v
        except Exception:
            raise ValueError('Invalid base64 encoding')
    
    class Config:
        schema_extra = {
            "example": {
                "image": "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg==",
                "top_k": 3
            }
        }


class PredictionResult(BaseModel):
    """Single prediction result."""
    class_name: str = Field(..., description="Predicted class name")
    probability: float = Field(..., ge=0.0, le=1.0, description="Prediction confidence")
    

class PredictionResponse(BaseModel):
    """Response model for predictions."""
    success: bool
    top_prediction: str
    confidence: float
    predictions: List[PredictionResult]
    inference_time_ms: float
    timestamp: str
    
    class Config:
        schema_extra = {
            "example": {
                "success": True,
                "top_prediction": "3",
                "confidence": 0.95,
                "predictions": [
                    {"class_name": "3", "probability": 0.95},
                    {"class_name": "8", "probability": 0.03}
                ],
                "inference_time_ms": 12.5,
                "timestamp": "2024-01-15T10:30:00"
            }
        }


class BatchPredictionRequest(BaseModel):
    """Request for batch predictions."""
    images: List[str] = Field(..., max_items=32, description="List of base64-encoded images")
    top_k: Optional[int] = Field(3, ge=1, le=10)


class ModelInfo(BaseModel):
    """Model information response."""
    model_name: str
    version: str
    device: str
    num_parameters: int
    num_classes: int
    input_shape: List[int]
    supported_formats: List[str]


# ============================================================================
# PART 2: Model Manager (Same as Flask version)
# ============================================================================

class SimpleClassifier(nn.Module):
    """Simple CNN for demonstration."""
    def __init__(self, num_classes: int = 10):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, num_classes)
        
    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 64 * 7 * 7)
        x = F.relu(self.fc1(x))
        return self.fc2(x)


class ModelManager:
    """Manages model loading and inference."""
    
    def __init__(self):
        self.model = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.class_names = [str(i) for i in range(10)]
        self.request_count = 0
        
    def load_model(self):
        """Load model for inference."""
        self.model = SimpleClassifier()
        self.model.to(self.device)
        self.model.eval()
        
    async def predict_async(self, image_tensor: torch.Tensor, top_k: int = 3) -> Dict[str, Any]:
        """
        Async prediction method.
        
        This doesn't block other requests during inference.
        For CPU-bound operations (like model inference), this still
        blocks, but it allows the event loop to handle other I/O
        operations concurrently.
        """
        self.request_count += 1
        start_time = time.time()
        
        image_tensor = image_tensor.to(self.device)
        
        with torch.no_grad():
            outputs = self.model(image_tensor)
            probabilities = F.softmax(outputs, dim=1)
            
        probs, indices = torch.topk(probabilities, k=top_k)
        
        predictions = []
        for prob, idx in zip(probs[0], indices[0]):
            predictions.append({
                'class_name': self.class_names[idx.item()],
                'probability': float(prob.item())
            })
        
        inference_time = (time.time() - start_time) * 1000  # ms
        
        return {
            'predictions': predictions,
            'top_prediction': predictions[0]['class_name'],
            'confidence': predictions[0]['probability'],
            'inference_time_ms': inference_time,
            'timestamp': datetime.now().isoformat()
        }


# Global model manager
model_manager = ModelManager()


# ============================================================================
# PART 3: Utility Functions
# ============================================================================

async def preprocess_image_async(image_bytes: bytes) -> torch.Tensor:
    """
    Async image preprocessing.
    
    In a real application, you might offload this to a thread pool
    for true parallelism.
    """
    # Simulate async I/O (in reality, this is CPU-bound)
    await asyncio.sleep(0)  # Yield to event loop
    
    image = Image.open(io.BytesIO(image_bytes))
    if image.mode != 'L':
        image = image.convert('L')
    image = image.resize((28, 28))
    
    image_array = np.array(image, dtype=np.float32) / 255.0
    image_tensor = torch.from_numpy(image_array).unsqueeze(0).unsqueeze(0)
    
    return image_tensor


# ============================================================================
# PART 4: FastAPI Application
# ============================================================================

# Create FastAPI app with metadata
app = FastAPI(
    title="Deep Learning Model API",
    description="FastAPI-based REST API for image classification",
    version="1.0.0",
    docs_url="/docs",  # Swagger UI
    redoc_url="/redoc"  # ReDoc documentation
)


@app.on_event("startup")
async def startup_event():
    """
    Startup event handler.
    
    Called once when the application starts.
    Use for:
    - Loading models
    - Initializing database connections
    - Setting up caches
    """
    print("Starting up FastAPI application...")
    model_manager.load_model()
    print("Model loaded successfully")


@app.on_event("shutdown")
async def shutdown_event():
    """
    Shutdown event handler.
    
    Called when application shuts down.
    Use for cleanup operations.
    """
    print("Shutting down FastAPI application...")


# ============================================================================
# PART 5: API Endpoints
# ============================================================================

@app.get("/", tags=["General"])
async def root():
    """Root endpoint with API information."""
    return {
        "message": "Deep Learning Model API",
        "version": "1.0.0",
        "docs": "/docs",
        "health": "/health",
        "endpoints": {
            "predict": "/predict (POST)",
            "predict_file": "/predict/file (POST)",
            "batch_predict": "/predict/batch (POST)",
            "model_info": "/model/info (GET)",
            "stats": "/stats (GET)"
        }
    }


@app.get("/health", tags=["General"], response_model=Dict[str, Any])
async def health_check():
    """
    Health check endpoint.
    
    Used by load balancers and monitoring systems.
    """
    return {
        "status": "healthy",
        "model_loaded": model_manager.model is not None,
        "device": str(model_manager.device),
        "timestamp": datetime.now().isoformat()
    }


@app.post("/predict", 
          tags=["Prediction"],
          response_model=PredictionResponse,
          summary="Predict from base64 image")
async def predict(request: PredictionRequest):
    """
    Predict class from base64-encoded image.
    
    This endpoint demonstrates:
    - Pydantic request validation
    - Automatic request/response documentation
    - Type-safe request handling
    """
    try:
        # Decode base64 image
        image_bytes = base64.b64decode(request.image)
        
        # Preprocess
        image_tensor = await preprocess_image_async(image_bytes)
        
        # Predict
        result = await model_manager.predict_async(image_tensor, request.top_k)
        
        return PredictionResponse(
            success=True,
            top_prediction=result['top_prediction'],
            confidence=result['confidence'],
            predictions=[PredictionResult(**p) for p in result['predictions']],
            inference_time_ms=result['inference_time_ms'],
            timestamp=result['timestamp']
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/predict/file",
          tags=["Prediction"],
          response_model=PredictionResponse,
          summary="Predict from uploaded image file")
async def predict_file(file: UploadFile = File(...), top_k: int = 3):
    """
    Predict class from uploaded image file.
    
    Accepts multipart/form-data with image file.
    FastAPI handles file uploads elegantly with UploadFile.
    """
    try:
        # Read file contents
        contents = await file.read()
        
        # Validate file
        if len(contents) > 10 * 1024 * 1024:  # 10MB limit
            raise HTTPException(status_code=400, detail="File too large (max 10MB)")
        
        # Preprocess
        image_tensor = await preprocess_image_async(contents)
        
        # Predict
        result = await model_manager.predict_async(image_tensor, top_k)
        
        return PredictionResponse(
            success=True,
            top_prediction=result['top_prediction'],
            confidence=result['confidence'],
            predictions=[PredictionResult(**p) for p in result['predictions']],
            inference_time_ms=result['inference_time_ms'],
            timestamp=result['timestamp']
        )
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/predict/batch",
          tags=["Prediction"],
          summary="Batch prediction for multiple images")
async def batch_predict(request: BatchPredictionRequest):
    """
    Process multiple images in a single request.
    
    More efficient than individual requests when:
    - You have multiple images to process
    - Network latency is a concern
    - You want to batch on GPU
    """
    try:
        if len(request.images) == 0:
            raise HTTPException(status_code=400, detail="No images provided")
        
        # Process all images
        tensors = []
        for img_b64 in request.images:
            img_bytes = base64.b64decode(img_b64)
            tensor = await preprocess_image_async(img_bytes)
            tensors.append(tensor)
        
        # Batch inference
        batch_tensor = torch.cat(tensors, dim=0).to(model_manager.device)
        
        start_time = time.time()
        with torch.no_grad():
            outputs = model_manager.model(batch_tensor)
            probabilities = F.softmax(outputs, dim=1)
        inference_time = (time.time() - start_time) * 1000
        
        # Prepare results
        results = []
        for i in range(len(tensors)):
            probs, indices = torch.topk(probabilities[i:i+1], k=request.top_k)
            predictions = []
            for prob, idx in zip(probs[0], indices[0]):
                predictions.append({
                    'class_name': model_manager.class_names[idx.item()],
                    'probability': float(prob.item())
                })
            results.append({
                'image_index': i,
                'top_prediction': predictions[0]['class_name'],
                'confidence': predictions[0]['probability'],
                'predictions': predictions
            })
        
        return {
            'success': True,
            'num_images': len(results),
            'total_inference_time_ms': inference_time,
            'avg_inference_time_ms': inference_time / len(results),
            'results': results,
            'timestamp': datetime.now().isoformat()
        }
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/model/info",
         tags=["Model"],
         response_model=ModelInfo,
         summary="Get model information")
async def model_info():
    """Get detailed information about the loaded model."""
    if model_manager.model is None:
        raise HTTPException(status_code=500, detail="Model not loaded")
    
    total_params = sum(p.numel() for p in model_manager.model.parameters())
    
    return ModelInfo(
        model_name="SimpleClassifier",
        version="1.0.0",
        device=str(model_manager.device),
        num_parameters=total_params,
        num_classes=len(model_manager.class_names),
        input_shape=[1, 1, 28, 28],
        supported_formats=["PNG", "JPEG", "BMP"]
    )


@app.get("/stats", tags=["Monitoring"], summary="Get API statistics")
async def get_stats():
    """Get API usage statistics."""
    return {
        "total_requests": model_manager.request_count,
        "model_device": str(model_manager.device),
        "uptime": "N/A",  # Would track actual uptime in production
        "timestamp": datetime.now().isoformat()
    }


# ============================================================================
# PART 6: Background Tasks Example
# ============================================================================

async def log_prediction(result: Dict[str, Any]):
    """
    Background task to log prediction.
    
    Background tasks run after the response is sent,
    so they don't block the client.
    
    Use for:
    - Logging
    - Analytics
    - Sending emails
    - Database writes
    """
    await asyncio.sleep(0.1)  # Simulate async work
    print(f"[LOG] Prediction: {result['top_prediction']}, "
          f"Confidence: {result['confidence']:.2f}, "
          f"Time: {result['inference_time_ms']:.2f}ms")


@app.post("/predict/logged", tags=["Prediction"])
async def predict_with_logging(
    request: PredictionRequest,
    background_tasks: BackgroundTasks
):
    """
    Prediction with background logging.
    
    Demonstrates FastAPI's background tasks feature.
    """
    try:
        image_bytes = base64.b64decode(request.image)
        image_tensor = await preprocess_image_async(image_bytes)
        result = await model_manager.predict_async(image_tensor, request.top_k)
        
        # Add background task (runs after response is sent)
        background_tasks.add_task(log_prediction, result)
        
        return PredictionResponse(
            success=True,
            top_prediction=result['top_prediction'],
            confidence=result['confidence'],
            predictions=[PredictionResult(**p) for p in result['predictions']],
            inference_time_ms=result['inference_time_ms'],
            timestamp=result['timestamp']
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ============================================================================
# PART 7: Main Entry Point
# ============================================================================

def main():
    """
    Main demonstration function.
    """
    print("\n" + "="*70)
    print("MODULE 64.05: FASTAPI DEPLOYMENT")
    print("="*70)
    
    print("""
This module demonstrates FastAPI deployment for deep learning models.

To run the server:
    uvicorn 05_fastapi_deployment:app --reload --host 0.0.0.0 --port 8000

To view automatic documentation:
    Swagger UI: http://localhost:8000/docs
    ReDoc: http://localhost:8000/redoc

To test the API:
    # Health check
    curl http://localhost:8000/health
    
    # Model info
    curl http://localhost:8000/model/info
    
    # Prediction (file upload)
    curl -X POST http://localhost:8000/predict/file \\
         -F "file=@test.png" \\
         -F "top_k=3"

For production deployment:
    uvicorn 05_fastapi_deployment:app --workers 4 --host 0.0.0.0 --port 8000

Key Advantages over Flask:
✓ Automatic API documentation (Swagger/OpenAPI)
✓ Request/response validation with Pydantic
✓ Async support for better concurrency
✓ Type hints throughout
✓ Modern Python 3.7+ features
✓ Better performance
✓ Built-in dependency injection

FastAPI is recommended for new projects!
    """)


if __name__ == "__main__":
    import uvicorn
    print("Starting FastAPI server...")
    uvicorn.run(app, host="0.0.0.0", port=8000)
