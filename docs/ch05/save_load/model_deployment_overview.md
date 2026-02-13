# Module 64: Model Deployment

## Overview
This module covers the complete pipeline of deploying PyTorch deep learning models to production environments. Students will learn model serialization, API development, containerization, and production best practices.

## Learning Objectives
By completing this module, students will be able to:
1. Save and load PyTorch models in multiple formats
2. Export models to ONNX and TorchScript for production
3. Build REST APIs for model serving
4. Implement batch inference pipelines
5. Containerize models with Docker
6. Monitor and log model performance
7. Apply optimization techniques for production deployment

## Prerequisites
- Module 18: Save and Load Models
- Module 20: Feedforward Networks
- Module 23: Convolutional Neural Networks
- Basic understanding of HTTP and REST APIs
- Familiarity with command-line tools

## Module Structure

### Part 1: Beginner Level (01-03)
**Focus: Model Serialization and Basic Inference**

#### 01_basic_model_saving.py
- Saving complete models vs state dictionaries
- Loading models for inference
- Checkpoint management
- Best practices for model persistence
- **Time estimate**: 45 minutes

#### 02_torchscript_basics.py
- Introduction to TorchScript
- Tracing vs scripting
- Converting models to TorchScript
- TorchScript inference
- **Time estimate**: 60 minutes

#### 03_onnx_export.py
- ONNX format introduction
- Exporting PyTorch models to ONNX
- ONNX Runtime inference
- Cross-framework compatibility
- **Time estimate**: 60 minutes

### Part 2: Intermediate Level (04-07)
**Focus: API Development and Serving**

#### 04_flask_api_deployment.py
- Building REST APIs with Flask
- Request/response handling
- Image preprocessing pipelines
- Error handling and validation
- **Time estimate**: 90 minutes

#### 05_fastapi_deployment.py
- FastAPI for high-performance serving
- Asynchronous request handling
- Automatic API documentation
- Request validation with Pydantic
- **Time estimate**: 90 minutes

#### 06_batch_inference.py
- Batch processing pipelines
- DataLoader for inference
- GPU utilization optimization
- Progress monitoring
- **Time estimate**: 60 minutes

#### 07_model_versioning.py
- Version control for models
- A/B testing infrastructure
- Gradual rollout strategies
- Rollback mechanisms
- **Time estimate**: 75 minutes

### Part 3: Advanced Level (08-12)
**Focus: Production Optimization and Monitoring**

#### 08_torchserve_deployment.py
- Introduction to TorchServe
- Model archiving (.mar files)
- Custom handlers
- Multi-model serving
- **Time estimate**: 90 minutes

#### 09_docker_deployment.py
- Docker containerization
- Creating efficient Docker images
- Multi-stage builds
- Container orchestration basics
- **Time estimate**: 90 minutes

#### 10_inference_optimization.py
- Model quantization (dynamic, static)
- Pruning and knowledge distillation
- CUDA optimization
- Mixed precision inference
- **Time estimate**: 120 minutes

#### 11_monitoring_logging.py
- Performance metrics tracking
- Latency monitoring
- Error logging
- Request/response logging
- Integration with monitoring tools
- **Time estimate**: 75 minutes

#### 12_production_best_practices.py
- Security considerations
- Rate limiting and throttling
- Caching strategies
- Load balancing
- Health checks and readiness probes
- **Time estimate**: 90 minutes

## Installation Requirements

```bash
# Core dependencies
pip install torch torchvision
pip install onnx onnxruntime
pip install flask fastapi uvicorn
pip install pydantic python-multipart
pip install requests pillow

# Optional for advanced modules
pip install torchserve torch-model-archiver
pip install docker
pip install prometheus-client
pip install tensorboard
```

## Hardware Requirements
- **Minimum**: CPU with 8GB RAM
- **Recommended**: NVIDIA GPU with 8GB+ VRAM, 16GB RAM
- **For production**: Multiple GPUs, load balancer, container orchestration

## Directory Structure
```
64_model_deployment/
├── README.md
├── requirements.txt
├── 01_basic_model_saving.py
├── 02_torchscript_basics.py
├── 03_onnx_export.py
├── 04_flask_api_deployment.py
├── 05_fastapi_deployment.py
├── 06_batch_inference.py
├── 07_model_versioning.py
├── 08_torchserve_deployment.py
├── 09_docker_deployment.py
├── 10_inference_optimization.py
├── 11_monitoring_logging.py
├── 12_production_best_practices.py
├── models/          # Saved model files
├── data/            # Sample data for testing
├── configs/         # Configuration files
└── tests/           # Unit tests
```

## Usage Examples

### Running Individual Modules
```bash
# Beginner level
python 01_basic_model_saving.py
python 02_torchscript_basics.py

# Intermediate level
python 04_flask_api_deployment.py
# In another terminal: curl -X POST http://localhost:5000/predict -F "image=@test.jpg"

# Advanced level
python 10_inference_optimization.py
```

### Testing API Deployments
```bash
# Test Flask API
python 04_flask_api_deployment.py &
curl -X POST http://localhost:5000/predict -F "image=@sample.jpg"

# Test FastAPI
python 05_fastapi_deployment.py &
curl -X POST http://localhost:8000/predict -F "file=@sample.jpg"
```

## Key Concepts

### Model Serialization Formats
1. **PyTorch (.pt, .pth)**: Native format, requires Python/PyTorch
2. **TorchScript (.pt)**: Optimized, language-independent representation
3. **ONNX (.onnx)**: Framework-agnostic, cross-platform compatibility

### Deployment Strategies
1. **Online Inference**: Real-time prediction APIs
2. **Batch Inference**: Bulk processing pipelines
3. **Edge Deployment**: On-device inference
4. **Cloud Deployment**: Scalable cloud-based serving

### Production Considerations
- **Latency**: Response time requirements (ms to seconds)
- **Throughput**: Requests per second capacity
- **Scalability**: Horizontal vs vertical scaling
- **Reliability**: Error handling, failover mechanisms
- **Monitoring**: Metrics, logging, alerting

## Common Pitfalls and Solutions

### 1. Model Loading Errors
**Problem**: Model fails to load due to version mismatch
**Solution**: Save model with version metadata, use compatible PyTorch versions

### 2. Memory Issues
**Problem**: Out-of-memory errors during inference
**Solution**: Implement batch size limiting, use model quantization

### 3. Slow Inference
**Problem**: High latency in production
**Solution**: Use TorchScript/ONNX, optimize preprocessing, enable GPU inference

### 4. Inconsistent Results
**Problem**: Different outputs in training vs production
**Solution**: Set model to eval mode, disable dropout/batchnorm training behavior

## Assessment and Practice

### Beginner Level
- Save a trained model and load it for inference
- Convert a simple model to TorchScript
- Export a model to ONNX format

### Intermediate Level
- Build a Flask API for image classification
- Implement batch inference with progress tracking
- Create a model versioning system

### Advanced Level
- Deploy a model with TorchServe
- Containerize an API with Docker
- Implement model quantization and measure speedup
- Set up comprehensive monitoring and logging

## Additional Resources

### Documentation
- [PyTorch Documentation](https://pytorch.org/docs/)
- [TorchScript Documentation](https://pytorch.org/docs/stable/jit.html)
- [ONNX Documentation](https://onnx.ai/)
- [TorchServe Documentation](https://pytorch.org/serve/)

### Tutorials
- PyTorch Production Tutorials
- FastAPI Documentation
- Docker Getting Started Guide

### Papers
- "Model Serving in Production: A Survey" (2020)
- "TorchServe: Serve PyTorch Models at Scale" (2020)
- "ONNX: Open Neural Network Exchange" (2019)

## Troubleshooting

### Issue: ImportError for specific modules
**Solution**: Ensure all dependencies are installed: `pip install -r requirements.txt`

### Issue: CUDA out of memory during deployment
**Solution**: Reduce batch size, use CPU inference, or implement model quantization

### Issue: API returns 500 errors
**Solution**: Check logs, validate input data format, ensure model is in eval mode

## Contributing
Students are encouraged to:
- Extend examples with additional models
- Add support for different deployment platforms
- Implement additional optimization techniques
- Create comprehensive test suites

## License
Educational use only. Models and data should respect original licenses.

## Contact
For questions or issues with this module, please contact your instructor.

---

**Total Module Time**: 14-16 hours
**Difficulty Progression**: Beginner → Intermediate → Advanced
**Prerequisites**: Modules 18, 20, 23
**Next Modules**: 65 (Model Compression), 66 (Efficient Inference)
