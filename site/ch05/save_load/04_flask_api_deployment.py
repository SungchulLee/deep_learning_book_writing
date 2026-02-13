"""
Module 64.04: Flask API Deployment
==================================

This module demonstrates deploying deep learning models as REST APIs using Flask.
Students will learn to build production-ready APIs with proper error handling,
input validation, and preprocessing pipelines.

Learning Objectives:
-------------------
1. Build REST APIs with Flask
2. Handle image uploads and preprocessing
3. Implement proper error handling
4. Add request validation
5. Create production-ready endpoints

Time Estimate: 90 minutes
Difficulty: Intermediate
Prerequisites: Module 64.01-64.03, basic HTTP knowledge
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from flask import Flask, request, jsonify
from PIL import Image
import io
import base64
import numpy as np
from typing import Dict, Any, Tuple
import logging
from functools import wraps
import time


# ============================================================================
# PART 1: Model Definition and Loading
# ============================================================================

class ImageClassifier(nn.Module):
    """Simple CNN for image classification (28x28 grayscale images)."""
    
    def __init__(self, num_classes: int = 10):
        super(ImageClassifier, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, num_classes)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 64 * 7 * 7)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


class ModelManager:
    """
    Manages model loading, caching, and inference.
    
    Benefits of using a manager class:
    - Singleton pattern: Load model once
    - Easy to add model versioning
    - Centralized inference logic
    - Thread-safe if needed
    """
    
    def __init__(self):
        self.model = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.class_names = [str(i) for i in range(10)]  # MNIST digits
        
    def load_model(self, model_path: str = None):
        """
        Load the model for inference.
        
        In production, you would load from a saved checkpoint.
        For this demo, we'll create and initialize a new model.
        """
        print(f"Loading model on device: {self.device}")
        
        if model_path:
            # In production: load saved model
            self.model = ImageClassifier()
            self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        else:
            # Demo: create new model
            self.model = ImageClassifier()
            
        self.model.to(self.device)
        self.model.eval()  # Set to evaluation mode
        print("Model loaded successfully")
        
    def predict(self, image_tensor: torch.Tensor) -> Dict[str, Any]:
        """
        Run inference on preprocessed image tensor.
        
        Args:
            image_tensor: Preprocessed image (1, 1, 28, 28)
            
        Returns:
            Dictionary with predictions and probabilities
        """
        if self.model is None:
            raise RuntimeError("Model not loaded. Call load_model() first.")
            
        # Move to correct device
        image_tensor = image_tensor.to(self.device)
        
        # Run inference
        with torch.no_grad():
            outputs = self.model(image_tensor)
            probabilities = F.softmax(outputs, dim=1)
            
        # Get top predictions
        probs, indices = torch.topk(probabilities, k=3)
        
        predictions = []
        for prob, idx in zip(probs[0], indices[0]):
            predictions.append({
                'class': self.class_names[idx.item()],
                'probability': float(prob.item())
            })
            
        return {
            'predictions': predictions,
            'top_class': predictions[0]['class'],
            'confidence': predictions[0]['probability']
        }


# Global model manager instance
model_manager = ModelManager()


# ============================================================================
# PART 2: Image Preprocessing
# ============================================================================

def preprocess_image(image: Image.Image, target_size: Tuple[int, int] = (28, 28)) -> torch.Tensor:
    """
    Preprocess image for model input.
    
    Steps:
    1. Convert to grayscale
    2. Resize to target size
    3. Convert to tensor
    4. Normalize (if needed)
    5. Add batch dimension
    
    Args:
        image: PIL Image object
        target_size: (height, width) tuple
        
    Returns:
        Preprocessed tensor of shape (1, 1, H, W)
    """
    # Convert to grayscale if needed
    if image.mode != 'L':
        image = image.convert('L')
        
    # Resize
    image = image.resize(target_size)
    
    # Convert to tensor
    image_array = np.array(image, dtype=np.float32)
    image_array = image_array / 255.0  # Normalize to [0, 1]
    
    # Add channel and batch dimensions: (H, W) -> (1, 1, H, W)
    image_tensor = torch.from_numpy(image_array).unsqueeze(0).unsqueeze(0)
    
    return image_tensor


def validate_image(file_bytes: bytes, max_size_mb: int = 10) -> Tuple[bool, str]:
    """
    Validate uploaded image file.
    
    Checks:
    - File size
    - Valid image format
    - Image can be opened
    
    Args:
        file_bytes: Raw file bytes
        max_size_mb: Maximum allowed file size in MB
        
    Returns:
        (is_valid, error_message) tuple
    """
    # Check file size
    file_size_mb = len(file_bytes) / (1024 * 1024)
    if file_size_mb > max_size_mb:
        return False, f"File too large ({file_size_mb:.2f}MB). Max: {max_size_mb}MB"
    
    # Try to open as image
    try:
        image = Image.open(io.BytesIO(file_bytes))
        # Verify it's a real image by accessing size
        _ = image.size
        return True, ""
    except Exception as e:
        return False, f"Invalid image format: {str(e)}"


# ============================================================================
# PART 3: Flask Application Setup
# ============================================================================

# Create Flask application
app = Flask(__name__)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def timing_decorator(f):
    """Decorator to measure endpoint execution time."""
    @wraps(f)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = f(*args, **kwargs)
        duration = time.time() - start_time
        logger.info(f"{f.__name__} took {duration:.4f} seconds")
        return result
    return wrapper


# ============================================================================
# PART 4: API Endpoints
# ============================================================================

@app.route('/health', methods=['GET'])
def health_check():
    """
    Health check endpoint.
    
    Used by:
    - Load balancers to check if service is running
    - Monitoring systems for uptime tracking
    - Kubernetes liveness/readiness probes
    
    Returns:
        JSON with service status
    """
    return jsonify({
        'status': 'healthy',
        'model_loaded': model_manager.model is not None,
        'device': str(model_manager.device)
    }), 200


@app.route('/predict', methods=['POST'])
@timing_decorator
def predict():
    """
    Main prediction endpoint.
    
    Accepts:
    - Multipart form data with 'image' file
    - JSON with base64-encoded image
    
    Returns:
    - JSON with predictions and confidence scores
    
    Error codes:
    - 400: Bad request (no image, invalid format)
    - 500: Internal server error (model failure)
    """
    try:
        # Get image from request
        image_bytes = None
        
        # Method 1: Multipart form upload
        if 'image' in request.files:
            file = request.files['image']
            if file.filename == '':
                return jsonify({'error': 'No file selected'}), 400
            image_bytes = file.read()
            
        # Method 2: JSON with base64 image
        elif request.is_json:
            data = request.get_json()
            if 'image' not in data:
                return jsonify({'error': 'No image in JSON'}), 400
            try:
                image_bytes = base64.b64decode(data['image'])
            except Exception as e:
                return jsonify({'error': f'Invalid base64 encoding: {str(e)}'}), 400
        else:
            return jsonify({
                'error': 'Invalid request format. Use multipart/form-data or JSON with base64 image'
            }), 400
        
        # Validate image
        is_valid, error_msg = validate_image(image_bytes)
        if not is_valid:
            return jsonify({'error': error_msg}), 400
            
        # Preprocess image
        image = Image.open(io.BytesIO(image_bytes))
        image_tensor = preprocess_image(image)
        
        logger.info(f"Processing image of size {image.size}")
        
        # Run prediction
        result = model_manager.predict(image_tensor)
        
        # Add metadata to response
        response = {
            'success': True,
            'predictions': result['predictions'],
            'top_prediction': {
                'class': result['top_class'],
                'confidence': result['confidence']
            },
            'image_size': image.size,
            'preprocessing': {
                'grayscale': True,
                'resized_to': (28, 28),
                'normalized': True
            }
        }
        
        return jsonify(response), 200
        
    except Exception as e:
        logger.error(f"Prediction error: {str(e)}")
        return jsonify({
            'success': False,
            'error': 'Internal server error',
            'details': str(e)
        }), 500


@app.route('/batch_predict', methods=['POST'])
@timing_decorator
def batch_predict():
    """
    Batch prediction endpoint for multiple images.
    
    Accepts JSON with array of base64-encoded images.
    Processes all images in a single batch for efficiency.
    
    Example request:
    {
        "images": ["base64_1", "base64_2", ...]
    }
    """
    try:
        if not request.is_json:
            return jsonify({'error': 'Request must be JSON'}), 400
            
        data = request.get_json()
        if 'images' not in data:
            return jsonify({'error': 'No images in request'}), 400
            
        images_b64 = data['images']
        if not isinstance(images_b64, list):
            return jsonify({'error': 'images must be an array'}), 400
            
        if len(images_b64) == 0:
            return jsonify({'error': 'images array is empty'}), 400
            
        if len(images_b64) > 32:
            return jsonify({'error': 'Maximum 32 images per batch'}), 400
            
        # Process all images
        tensors = []
        for i, img_b64 in enumerate(images_b64):
            try:
                img_bytes = base64.b64decode(img_b64)
                is_valid, error_msg = validate_image(img_bytes)
                if not is_valid:
                    return jsonify({'error': f'Image {i}: {error_msg}'}), 400
                    
                image = Image.open(io.BytesIO(img_bytes))
                tensor = preprocess_image(image)
                tensors.append(tensor)
            except Exception as e:
                return jsonify({'error': f'Image {i} processing failed: {str(e)}'}), 400
        
        # Batch inference
        batch_tensor = torch.cat(tensors, dim=0)  # (N, 1, 28, 28)
        batch_tensor = batch_tensor.to(model_manager.device)
        
        with torch.no_grad():
            outputs = model_manager.model(batch_tensor)
            probabilities = F.softmax(outputs, dim=1)
            
        # Prepare results
        results = []
        for i in range(len(tensors)):
            probs, indices = torch.topk(probabilities[i:i+1], k=3)
            predictions = []
            for prob, idx in zip(probs[0], indices[0]):
                predictions.append({
                    'class': model_manager.class_names[idx.item()],
                    'probability': float(prob.item())
                })
            results.append({
                'image_index': i,
                'predictions': predictions,
                'top_class': predictions[0]['class'],
                'confidence': predictions[0]['probability']
            })
        
        return jsonify({
            'success': True,
            'num_images': len(results),
            'results': results
        }), 200
        
    except Exception as e:
        logger.error(f"Batch prediction error: {str(e)}")
        return jsonify({
            'success': False,
            'error': 'Internal server error',
            'details': str(e)
        }), 500


@app.route('/model_info', methods=['GET'])
def model_info():
    """
    Get information about the loaded model.
    
    Useful for:
    - Debugging
    - Monitoring
    - Client-side validation
    """
    if model_manager.model is None:
        return jsonify({'error': 'Model not loaded'}), 500
        
    # Count parameters
    total_params = sum(p.numel() for p in model_manager.model.parameters())
    trainable_params = sum(p.numel() for p in model_manager.model.parameters() if p.requires_grad)
    
    return jsonify({
        'model_type': type(model_manager.model).__name__,
        'device': str(model_manager.device),
        'parameters': {
            'total': total_params,
            'trainable': trainable_params
        },
        'classes': model_manager.class_names,
        'num_classes': len(model_manager.class_names),
        'input_shape': [1, 1, 28, 28],
        'preprocessing': {
            'grayscale': True,
            'resize': [28, 28],
            'normalization': '0-1'
        }
    }), 200


@app.errorhandler(404)
def not_found(error):
    """Handle 404 errors."""
    return jsonify({'error': 'Endpoint not found'}), 404


@app.errorhandler(500)
def internal_error(error):
    """Handle 500 errors."""
    logger.error(f"Internal error: {str(error)}")
    return jsonify({'error': 'Internal server error'}), 500


# ============================================================================
# PART 5: Application Initialization and Running
# ============================================================================

def initialize_app():
    """
    Initialize the application.
    
    This is where you would:
    - Load models
    - Set up database connections
    - Initialize caches
    - Configure monitoring
    """
    logger.info("Initializing Flask application...")
    
    # Load model
    model_manager.load_model()
    
    logger.info("Application initialized successfully")


def run_server(host: str = '0.0.0.0', port: int = 5000, debug: bool = False):
    """
    Run the Flask development server.
    
    Production note:
    - Don't use Flask's built-in server in production
    - Use WSGI server like gunicorn or uwsgi
    - Example: gunicorn -w 4 -b 0.0.0.0:5000 04_flask_api_deployment:app
    
    Args:
        host: Host to bind to
        port: Port to listen on
        debug: Enable debug mode (don't use in production!)
    """
    logger.info(f"Starting server on {host}:{port}")
    app.run(host=host, port=port, debug=debug)


# ============================================================================
# PART 6: Testing Functions
# ============================================================================

def test_api_locally():
    """
    Test the API endpoints locally.
    
    This demonstrates how to:
    1. Create test images
    2. Make requests to endpoints
    3. Validate responses
    """
    import requests
    
    print("\n" + "="*70)
    print("TESTING FLASK API")
    print("="*70)
    
    base_url = "http://localhost:5000"
    
    # Test 1: Health check
    print("\n1. Testing health check endpoint...")
    response = requests.get(f"{base_url}/health")
    print(f"   Status: {response.status_code}")
    print(f"   Response: {response.json()}")
    
    # Test 2: Model info
    print("\n2. Testing model info endpoint...")
    response = requests.get(f"{base_url}/model_info")
    print(f"   Status: {response.status_code}")
    print(f"   Response: {response.json()}")
    
    # Test 3: Single prediction
    print("\n3. Testing prediction endpoint...")
    # Create a dummy image
    from PIL import Image
    import io
    
    # Create random 28x28 grayscale image
    img = Image.fromarray(np.random.randint(0, 255, (28, 28), dtype=np.uint8), mode='L')
    img_bytes = io.BytesIO()
    img.save(img_bytes, format='PNG')
    img_bytes.seek(0)
    
    # Send as multipart form
    files = {'image': ('test.png', img_bytes, 'image/png')}
    response = requests.post(f"{base_url}/predict", files=files)
    print(f"   Status: {response.status_code}")
    print(f"   Response: {response.json()}")
    
    print("\n✓ All tests completed")


# ============================================================================
# PART 7: Main Entry Point
# ============================================================================

def main():
    """
    Main demonstration function.
    """
    print("\n" + "="*70)
    print("MODULE 64.04: FLASK API DEPLOYMENT")
    print("="*70)
    
    print("""
This module demonstrates Flask API deployment for deep learning models.

To run the server:
    python 04_flask_api_deployment.py

To test the API:
    # Health check
    curl http://localhost:5000/health
    
    # Model info
    curl http://localhost:5000/model_info
    
    # Prediction (with image file)
    curl -X POST http://localhost:5000/predict -F "image=@test_image.png"

For production deployment:
    gunicorn -w 4 -b 0.0.0.0:5000 04_flask_api_deployment:app

Key Features:
- Multiple input formats (multipart form, base64 JSON)
- Batch prediction support
- Comprehensive error handling
- Request validation
- Performance monitoring
- Health checks for load balancers
    """)
    
    # Initialize and run
    initialize_app()
    
    print("\n" + "="*70)
    print("Starting Flask server...")
    print("Press CTRL+C to stop")
    print("="*70)
    
    run_server(host='0.0.0.0', port=5000, debug=True)


if __name__ == "__main__":
    main()
