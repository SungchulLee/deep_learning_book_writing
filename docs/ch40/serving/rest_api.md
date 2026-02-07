# Model Serving Frameworks

## Introduction

Model serving frameworks provide the infrastructure to deploy machine learning models as production services. They handle concerns like request batching, model versioning, health monitoring, and horizontal scaling. This section covers serving options from simple REST APIs to enterprise-grade platforms.

## Serving Architecture Overview

A production model serving system typically includes:

```
┌─────────────────────────────────────────────────────────────────────┐
│                       Load Balancer                                  │
└───────────────────────────────────┬─────────────────────────────────┘
                                    │
        ┌───────────────────────────┼───────────────────────────────┐
        │                           │                               │
        ▼                           ▼                               ▼
┌───────────────┐           ┌───────────────┐           ┌───────────────┐
│   Replica 1   │           │   Replica 2   │           │   Replica 3   │
├───────────────┤           ├───────────────┤           ├───────────────┤
│ Preprocessing │           │ Preprocessing │           │ Preprocessing │
│      ↓        │           │      ↓        │           │      ↓        │
│   Inference   │           │   Inference   │           │   Inference   │
│      ↓        │           │      ↓        │           │      ↓        │
│ Postprocessing│           │ Postprocessing│           │ Postprocessing│
└───────────────┘           └───────────────┘           └───────────────┘
```

## Framework Comparison

| Feature | Flask | FastAPI | TorchServe | Triton |
|---------|-------|---------|------------|--------|
| Setup Complexity | Low | Low | Medium | High |
| Performance | Moderate | Good | Good | Excellent |
| Async Support | No | Yes | Limited | Yes |
| Auto Documentation | No | Yes | No | Yes |
| Multi-Model | Manual | Manual | Built-in | Built-in |
| Dynamic Batching | Manual | Manual | Built-in | Built-in |
| GPU Support | Manual | Manual | Built-in | Optimized |
| Monitoring | Manual | Manual | Built-in | Built-in |
| Best For | Prototyping | Modern APIs | PyTorch Enterprise | Max Performance |

## Option 1: Flask API (Simple)

Flask provides a minimal, synchronous serving solution ideal for prototyping:

### Basic Flask Server

```python
from flask import Flask, request, jsonify
import torch
import torch.nn as nn
from PIL import Image
import io
import numpy as np

app = Flask(__name__)

# Global model instance
model = None
device = None

class ImageClassifier(nn.Module):
    """Simple CNN classifier."""
    def __init__(self, num_classes=10):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, num_classes)
    
    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        x = x.view(-1, 64 * 7 * 7)
        x = torch.relu(self.fc1(x))
        return self.fc2(x)


def load_model():
    """Load model on startup."""
    global model, device
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = ImageClassifier()
    model.load_state_dict(torch.load('model.pt', map_location=device))
    model = model.to(device)
    model.eval()
    
    print(f"Model loaded on {device}")


def preprocess_image(image_bytes):
    """Preprocess image for inference."""
    image = Image.open(io.BytesIO(image_bytes)).convert('L')
    image = image.resize((28, 28))
    
    tensor = torch.tensor(np.array(image), dtype=torch.float32)
    tensor = tensor.unsqueeze(0).unsqueeze(0) / 255.0
    
    return tensor.to(device)


@app.route('/health', methods=['GET'])
def health():
    """Health check endpoint."""
    return jsonify({
        'status': 'healthy',
        'model_loaded': model is not None,
        'device': str(device)
    })


@app.route('/predict', methods=['POST'])
def predict():
    """Prediction endpoint."""
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No file provided'}), 400
        
        file = request.files['file']
        image_bytes = file.read()
        
        input_tensor = preprocess_image(image_bytes)
        
        with torch.no_grad():
            outputs = model(input_tensor)
            probabilities = torch.softmax(outputs, dim=1)
            predicted_class = torch.argmax(probabilities, dim=1).item()
            confidence = probabilities[0, predicted_class].item()
        
        return jsonify({
            'predicted_class': predicted_class,
            'confidence': confidence,
            'probabilities': probabilities[0].tolist()
        })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/predict/batch', methods=['POST'])
def predict_batch():
    """Batch prediction endpoint."""
    try:
        files = request.files.getlist('files')
        
        if not files:
            return jsonify({'error': 'No files provided'}), 400
        
        tensors = [preprocess_image(f.read()) for f in files]
        batch = torch.cat(tensors, dim=0)
        
        with torch.no_grad():
            outputs = model(batch)
            probabilities = torch.softmax(outputs, dim=1)
            predictions = torch.argmax(probabilities, dim=1)
        
        results = [
            {
                'file': files[i].filename,
                'predicted_class': predictions[i].item(),
                'confidence': probabilities[i, predictions[i]].item()
            }
            for i in range(len(files))
        ]
        
        return jsonify({'predictions': results})
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    load_model()
    app.run(host='0.0.0.0', port=5000, debug=False)
```

### Flask with Gunicorn (Production)

```bash
# Install gunicorn
pip install gunicorn

# Run with multiple workers
gunicorn -w 4 -b 0.0.0.0:5000 app:app
```

```python
# gunicorn.conf.py
import multiprocessing

bind = "0.0.0.0:5000"
workers = multiprocessing.cpu_count() * 2 + 1
worker_class = "sync"
timeout = 120
keepalive = 5
```

## Option 2: FastAPI (Async, Modern)

FastAPI provides async support, automatic documentation, and better performance:

### Basic FastAPI Server

```python
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel, validator
from typing import List, Optional
import torch
import torch.nn as nn
from PIL import Image
import io
import numpy as np
from contextlib import asynccontextmanager

# Pydantic models for request/response validation
class PredictionResponse(BaseModel):
    predicted_class: int
    confidence: float
    probabilities: List[float]

class BatchPredictionItem(BaseModel):
    filename: str
    predicted_class: int
    confidence: float

class BatchPredictionResponse(BaseModel):
    predictions: List[BatchPredictionItem]

class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    device: str


# Global state
class ModelState:
    model: Optional[nn.Module] = None
    device: Optional[torch.device] = None

state = ModelState()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifecycle manager for model loading."""
    # Startup
    state.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    state.model = torch.jit.load('model.pt', map_location=state.device)
    state.model = state.model.to(state.device)
    state.model.eval()
    print(f"Model loaded on {state.device}")
    
    yield
    
    # Shutdown
    state.model = None
    print("Model unloaded")


app = FastAPI(
    title="Image Classification API",
    description="Production-ready image classification service",
    version="1.0.0",
    lifespan=lifespan
)


async def preprocess_image(image_bytes: bytes) -> torch.Tensor:
    """Preprocess image for inference."""
    image = Image.open(io.BytesIO(image_bytes)).convert('L')
    image = image.resize((28, 28))
    
    tensor = torch.tensor(np.array(image), dtype=torch.float32)
    tensor = tensor.unsqueeze(0).unsqueeze(0) / 255.0
    
    return tensor.to(state.device)


@app.get("/health", response_model=HealthResponse)
async def health():
    """Health check endpoint."""
    return HealthResponse(
        status="healthy",
        model_loaded=state.model is not None,
        device=str(state.device)
    )


@app.post("/predict", response_model=PredictionResponse)
async def predict(file: UploadFile = File(...)):
    """
    Predict class for a single image.
    
    - **file**: Image file (JPEG, PNG)
    """
    try:
        image_bytes = await file.read()
        input_tensor = await preprocess_image(image_bytes)
        
        with torch.no_grad():
            outputs = state.model(input_tensor)
            probabilities = torch.softmax(outputs, dim=1)
            predicted_class = torch.argmax(probabilities, dim=1).item()
            confidence = probabilities[0, predicted_class].item()
        
        return PredictionResponse(
            predicted_class=predicted_class,
            confidence=confidence,
            probabilities=probabilities[0].tolist()
        )
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
```

### FastAPI with Dynamic Batching

```python
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
import torch
from PIL import Image
import io
import asyncio
from collections import deque
from dataclasses import dataclass
import time

app = FastAPI(title="Model Serving API", version="1.0")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

@dataclass
class InferenceRequest:
    """Pending inference request."""
    input_tensor: torch.Tensor
    future: asyncio.Future
    timestamp: float

class BatchingInferenceService:
    """Service with dynamic batching for improved throughput."""
    
    def __init__(self, model, max_batch_size=32, max_latency_ms=100):
        self.model = model
        self.max_batch_size = max_batch_size
        self.max_latency_ms = max_latency_ms
        self.queue: deque = deque()
        self.lock = asyncio.Lock()
        self._running = False
    
    async def start(self):
        """Start batch processing loop."""
        self._running = True
        asyncio.create_task(self._process_loop())
    
    async def predict(self, input_tensor: torch.Tensor) -> torch.Tensor:
        """Submit inference request and wait for result."""
        loop = asyncio.get_event_loop()
        future = loop.create_future()
        
        request = InferenceRequest(
            input_tensor=input_tensor,
            future=future,
            timestamp=time.time()
        )
        
        async with self.lock:
            self.queue.append(request)
        
        return await future
    
    async def _process_loop(self):
        """Continuous batch processing loop."""
        while self._running:
            await self._process_batch()
            await asyncio.sleep(0.001)
    
    async def _process_batch(self):
        """Process accumulated requests as batch."""
        async with self.lock:
            if not self.queue:
                return
            
            batch_requests = []
            current_time = time.time()
            
            while self.queue and len(batch_requests) < self.max_batch_size:
                request = self.queue[0]
                age_ms = (current_time - request.timestamp) * 1000
                
                if age_ms > self.max_latency_ms:
                    break
                
                batch_requests.append(self.queue.popleft())
        
        if not batch_requests:
            return
        
        # Stack inputs and run inference
        batch_input = torch.stack([r.input_tensor for r in batch_requests]).to(device)
        
        with torch.no_grad():
            batch_output = self.model(batch_input)
        
        # Distribute results
        for i, request in enumerate(batch_requests):
            request.future.set_result(batch_output[i])

# Initialize service
inference_service = None

@app.on_event("startup")
async def startup():
    """Initialize model and service on startup."""
    global inference_service
    
    model = torch.jit.load("model.pt")
    model = model.to(device).eval()
    
    inference_service = BatchingInferenceService(model)
    await inference_service.start()
    
    print(f"Model loaded on {device}")

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    """Run inference on uploaded image."""
    try:
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert('RGB')
        
        from torchvision import transforms
        transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        input_tensor = transform(image)
        
        output = await inference_service.predict(input_tensor)
        
        probabilities = torch.softmax(output, dim=0)
        top5_probs, top5_indices = torch.topk(probabilities, 5)
        
        return JSONResponse({
            "predictions": [
                {"class_id": int(idx), "probability": float(prob)}
                for idx, prob in zip(top5_indices, top5_probs)
            ]
        })
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health():
    """Health check endpoint."""
    return {"status": "healthy", "device": str(device)}
```

### Docker Deployment

```dockerfile
# Dockerfile
FROM pytorch/pytorch:2.0.0-cuda11.7-cudnn8-runtime

WORKDIR /app

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .

EXPOSE 8000

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
```

## Option 3: TorchServe (Enterprise PyTorch)

TorchServe is PyTorch's official model serving solution with built-in features for production.

### Installation

```bash
pip install torchserve torch-model-archiver torch-workflow-archiver
```

### Custom Handler

```python
# handler.py
import torch
import torch.nn as nn
from torchvision import transforms
from ts.torch_handler.base_handler import BaseHandler
from PIL import Image
import io

class CustomHandler(BaseHandler):
    """Custom TorchServe handler."""
    
    def initialize(self, context):
        """Load model and initialize transforms."""
        super().initialize(context)
        
        self.transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])
    
    def preprocess(self, data):
        """Preprocess input data."""
        images = []
        for row in data:
            image = row.get("data") or row.get("body")
            if isinstance(image, (bytes, bytearray)):
                image = Image.open(io.BytesIO(image)).convert('RGB')
            
            images.append(self.transform(image))
        
        return torch.stack(images)
    
    def inference(self, data):
        """Run inference."""
        self.model.eval()
        with torch.no_grad():
            output = self.model(data)
        return output
    
    def postprocess(self, inference_output):
        """Format output."""
        probabilities = torch.softmax(inference_output, dim=1)
        top5_probs, top5_indices = torch.topk(probabilities, 5)
        
        results = []
        for probs, indices in zip(top5_probs, top5_indices):
            result = [
                {"class": self.mapping[str(idx.item())], "probability": prob.item()}
                for prob, idx in zip(probs, indices)
            ]
            results.append(result)
        
        return results
```

### Create Model Archive

```bash
torch-model-archiver \
    --model-name resnet50 \
    --version 1.0 \
    --serialized-file model.pth \
    --handler image_classifier \
    --export-path model_store \
    --extra-files index_to_name.json
```

### Start Server

```bash
# Start TorchServe
torchserve --start --model-store model_store --models resnet50=resnet50.mar

# Check status
curl http://localhost:8080/ping

# Make prediction
curl -X POST http://localhost:8080/predictions/resnet50 -T image.jpg
```

### Configuration

```properties
# config.properties
inference_address=http://0.0.0.0:8080
management_address=http://0.0.0.0:8081
metrics_address=http://0.0.0.0:8082

# Model settings
default_workers_per_model=4
job_queue_size=100
max_request_size=6553500

# Batching
batch_delay=100
batch_size=32
max_batch_size=64

# GPU configuration
number_of_gpu=1

# Timeouts
default_response_timeout=300
```

### Model Management API

```bash
# List models
curl http://localhost:8081/models

# Register model
curl -X POST "http://localhost:8081/models?url=resnet50.mar&model_name=resnet50&initial_workers=2"

# Scale workers
curl -X PUT "http://localhost:8081/models/resnet50?min_worker=4&max_worker=8"

# Unregister model
curl -X DELETE "http://localhost:8081/models/resnet50"
```

## Option 4: NVIDIA Triton Inference Server

Triton supports multiple frameworks (PyTorch, TensorFlow, ONNX, TensorRT) with advanced features like dynamic batching, ensemble models, and GPU instance groups.

### Model Repository Structure

```
model_repository/
├── resnet50/
│   ├── config.pbtxt
│   ├── 1/
│   │   └── model.onnx
│   └── 2/
│       └── model.onnx
└── bert/
    ├── config.pbtxt
    └── 1/
        └── model.plan
```

### Model Configuration

```protobuf
# config.pbtxt
name: "resnet50"
platform: "onnxruntime_onnx"
max_batch_size: 64

input [
  {
    name: "input"
    data_type: TYPE_FP32
    dims: [ 3, 224, 224 ]
  }
]

output [
  {
    name: "output"
    data_type: TYPE_FP32
    dims: [ 1000 ]
  }
]

# Dynamic batching
dynamic_batching {
  preferred_batch_size: [ 8, 16, 32 ]
  max_queue_delay_microseconds: 100000
}

# Instance groups (GPU allocation)
instance_group [
  {
    count: 2
    kind: KIND_GPU
    gpus: [ 0 ]
  }
]
```

### Docker Deployment

```bash
# Pull Triton image
docker pull nvcr.io/nvidia/tritonserver:23.10-py3

# Start server
docker run --gpus all -p 8000:8000 -p 8001:8001 -p 8002:8002 \
    -v /path/to/model_repository:/models \
    nvcr.io/nvidia/tritonserver:23.10-py3 \
    tritonserver --model-repository=/models
```

### Python Client

```python
import tritonclient.http as httpclient
import numpy as np

def triton_inference(model_name, input_data):
    """Run inference on Triton server."""
    client = httpclient.InferenceServerClient(url="localhost:8000")
    
    # Check server health
    if not client.is_server_live():
        raise RuntimeError("Triton server is not live")
    
    # Prepare input
    inputs = [
        httpclient.InferInput("input", input_data.shape, "FP32")
    ]
    inputs[0].set_data_from_numpy(input_data)
    
    # Request output
    outputs = [
        httpclient.InferRequestedOutput("output")
    ]
    
    # Inference
    response = client.infer(model_name, inputs, outputs=outputs)
    
    return response.as_numpy("output")

# Usage
input_data = np.random.randn(1, 3, 224, 224).astype(np.float32)
output = triton_inference("resnet50", input_data)
```

### Ensemble Models

Chain multiple models for complex pipelines:

```protobuf
# ensemble_config.pbtxt
name: "pipeline"
platform: "ensemble"
max_batch_size: 32

input [
  { name: "raw_image" data_type: TYPE_UINT8 dims: [ -1, -1, 3 ] }
]

output [
  { name: "classification" data_type: TYPE_FP32 dims: [ 1000 ] }
]

ensemble_scheduling {
  step [
    {
      model_name: "preprocessing"
      model_version: -1
      input_map { key: "raw_image" value: "raw_image" }
      output_map { key: "processed_image" value: "preprocessed" }
    },
    {
      model_name: "resnet50"
      model_version: -1
      input_map { key: "preprocessed" value: "input" }
      output_map { key: "output" value: "classification" }
    }
  ]
}
```

## Cloud ML Platforms

### AWS SageMaker

```python
import sagemaker
from sagemaker.pytorch import PyTorchModel

pytorch_model = PyTorchModel(
    model_data='s3://bucket/model.tar.gz',
    role='arn:aws:iam::account:role/SageMakerRole',
    framework_version='2.0',
    py_version='py310',
    entry_point='inference.py'
)

predictor = pytorch_model.deploy(
    instance_type='ml.g4dn.xlarge',
    initial_instance_count=1,
    endpoint_name='my-endpoint'
)

result = predictor.predict(input_data)
predictor.delete_endpoint()
```

### Google Cloud Vertex AI

```python
from google.cloud import aiplatform

aiplatform.init(project='project-id', location='us-central1')

model = aiplatform.Model.upload(
    display_name='my-model',
    artifact_uri='gs://bucket/model/',
    serving_container_image_uri='us-docker.pkg.dev/vertex-ai/prediction/pytorch-gpu.1-13:latest'
)

endpoint = model.deploy(
    machine_type='n1-standard-4',
    accelerator_type='NVIDIA_TESLA_T4',
    accelerator_count=1,
    min_replica_count=1,
    max_replica_count=5
)

predictions = endpoint.predict(instances=[input_data])
```

### Azure ML

```python
from azureml.core import Workspace, Model
from azureml.core.webservice import AciWebservice
from azureml.core.model import InferenceConfig

ws = Workspace.from_config()

model = Model.register(
    workspace=ws,
    model_path='model/',
    model_name='my-model'
)

inference_config = InferenceConfig(
    entry_script='score.py',
    environment=myenv
)

deployment_config = AciWebservice.deploy_configuration(
    cpu_cores=2,
    memory_gb=4,
    auth_enabled=True
)

service = Model.deploy(
    workspace=ws,
    name='my-service',
    models=[model],
    inference_config=inference_config,
    deployment_config=deployment_config
)

service.wait_for_deployment()
```

## Performance Monitoring

### Prometheus Metrics

```python
from prometheus_client import Counter, Histogram, Gauge, start_http_server
import time

REQUEST_COUNT = Counter(
    'inference_requests_total', 
    'Total inference requests',
    ['model_name', 'status']
)
LATENCY = Histogram(
    'inference_latency_seconds', 
    'Inference latency',
    buckets=[0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0]
)
ERROR_COUNT = Counter('inference_errors_total', 'Total inference errors')
IN_PROGRESS = Gauge('inference_in_progress', 'Requests currently being processed')

def monitored_inference(model, input_data, model_name="default"):
    """Inference with Prometheus metrics."""
    REQUEST_COUNT.labels(model_name=model_name, status='received').inc()
    IN_PROGRESS.inc()
    
    start = time.time()
    try:
        output = model(input_data)
        latency = time.time() - start
        LATENCY.observe(latency)
        REQUEST_COUNT.labels(model_name=model_name, status='success').inc()
        return output
    except Exception as e:
        ERROR_COUNT.inc()
        REQUEST_COUNT.labels(model_name=model_name, status='error').inc()
        raise
    finally:
        IN_PROGRESS.dec()

# Start metrics server
start_http_server(9090)
```

## Best Practices

### 1. Model Loading

```python
# ✓ Load model once at startup
def startup():
    global model
    model = load_model()
    model.eval()

# ✗ Don't load model per request
def predict(data):
    model = load_model()  # Bad! Slow and memory intensive
    return model(data)
```

### 2. Input Validation

```python
from pydantic import BaseModel, validator

class PredictionRequest(BaseModel):
    image: str  # base64
    
    @validator('image')
    def validate_image(cls, v):
        if len(v) > 10_000_000:  # 10MB limit
            raise ValueError('Image too large')
        return v
```

### 3. Error Handling

```python
from fastapi import HTTPException

@app.post("/predict")
async def predict(file: UploadFile):
    try:
        # ... inference logic
        pass
    except torch.cuda.OutOfMemoryError:
        raise HTTPException(status_code=503, detail="GPU out of memory")
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Prediction failed: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")
```

### 4. Request Timeouts

```python
import asyncio
from fastapi import HTTPException

@app.post("/predict")
async def predict(file: UploadFile):
    try:
        result = await asyncio.wait_for(
            run_inference(file),
            timeout=30.0
        )
        return result
    except asyncio.TimeoutError:
        raise HTTPException(status_code=504, detail="Request timeout")
```

### 5. Health Checks

```python
@app.get("/health/ready")
async def readiness():
    """Readiness probe - can the service handle requests?"""
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    return {"status": "ready"}

@app.get("/health/live")
async def liveness():
    """Liveness probe - is the service running?"""
    return {"status": "alive"}
```

### 6. Graceful Shutdown

```python
import signal
import sys

def graceful_shutdown(signum, frame):
    """Handle SIGTERM for graceful shutdown."""
    print("Received shutdown signal, finishing in-flight requests...")
    # Complete pending requests
    # Clean up resources
    sys.exit(0)

signal.signal(signal.SIGTERM, graceful_shutdown)
```

## Choosing the Right Framework

### Use Flask When:
- Rapid prototyping
- Simple deployments
- Single model serving
- Team familiar with Flask

### Use FastAPI When:
- Need async support
- Want auto-generated docs
- Modern Python features desired
- Good performance required

### Use TorchServe When:
- Enterprise PyTorch deployment
- Multi-model serving
- Need versioning and A/B testing
- PyTorch-specific optimizations

### Use Triton When:
- Maximum performance required
- Multiple frameworks (ONNX, TensorRT, PyTorch)
- GPU-intensive workloads
- Enterprise infrastructure

## Summary

Model serving is critical for production ML:

1. **Flask**: Simple, synchronous, good for prototyping
2. **FastAPI**: Modern, async, automatic documentation
3. **TorchServe**: Enterprise PyTorch, built-in features
4. **Triton**: Maximum performance, multi-framework

Key considerations:
- Load models once at startup
- Validate inputs thoroughly
- Handle errors gracefully
- Monitor latency and throughput
- Plan for scaling

## References

1. TorchServe Documentation: https://pytorch.org/serve/
2. NVIDIA Triton: https://developer.nvidia.com/nvidia-triton-inference-server
3. FastAPI: https://fastapi.tiangolo.com/
4. Flask: https://flask.palletsprojects.com/
5. AWS SageMaker: https://docs.aws.amazon.com/sagemaker/
6. Google Vertex AI: https://cloud.google.com/vertex-ai/docs
7. Azure ML: https://docs.microsoft.com/en-us/azure/machine-learning/
