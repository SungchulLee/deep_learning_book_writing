# Quick Start Guide: Module 64 - Model Deployment

## Installation

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## Running Individual Modules

### Beginner Level

```bash
# 01: Basic model saving and loading
python 01_basic_model_saving.py

# 02: TorchScript conversion
python 02_torchscript_basics.py

# 03: ONNX export
python 03_onnx_export.py
```

### Intermediate Level

```bash
# 04: Flask API
python 04_flask_api_deployment.py
# Visit: http://localhost:5000/health

# 05: FastAPI (in another terminal)
uvicorn 05_fastapi_deployment:app --reload --port 8000
# Visit: http://localhost:8000/docs

# 06: Batch inference
python 06_batch_inference.py
```

### Advanced Level

```bash
# 10: Inference optimization
python 10_inference_optimization.py
```

## Testing APIs

### Flask API (Module 04)

```bash
# Start server
python 04_flask_api_deployment.py &

# Test endpoints
curl http://localhost:5000/health
curl http://localhost:5000/model_info

# Prediction (create a test image first)
curl -X POST http://localhost:5000/predict -F "image=@test_image.png"
```

### FastAPI (Module 05)

```bash
# Start server
uvicorn 05_fastapi_deployment:app --host 0.0.0.0 --port 8000 &

# Interactive documentation
# Open browser: http://localhost:8000/docs

# Test endpoints
curl http://localhost:8000/health
curl http://localhost:8000/model/info

# Prediction
curl -X POST http://localhost:8000/predict/file -F "file=@test_image.png"
```

## Module Progression

```
01 → 02 → 03       (Serialization fundamentals)
    ↓
04 → 05 → 06       (API development & batch processing)
    ↓
10                 (Optimization techniques)
```

## Expected Outputs

Each module creates output files in:
- `models/` - Saved models and checkpoints
- `results/` - Inference results and logs
- `data/` - Test data (auto-generated)

## Common Commands

```bash
# Check PyTorch installation
python -c "import torch; print(torch.__version__)"

# Check CUDA availability
python -c "import torch; print(torch.cuda.is_available())"

# Run with specific device
CUDA_VISIBLE_DEVICES=0 python <module_name>.py

# Install additional dependencies
pip install tensorboard prometheus-client
```

## Troubleshooting

**Import errors?**
```bash
pip install --upgrade torch torchvision
```

**Out of memory?**
```python
# In code, reduce batch size or use CPU
device = 'cpu'  # Instead of 'cuda'
```

**API not responding?**
```bash
# Check if port is in use
lsof -i :5000  # Flask
lsof -i :8000  # FastAPI
```

## Next Steps

1. Complete beginner modules (01-03)
2. Build your first API (04 or 05)
3. Try batch inference (06)
4. Optimize for production (10)
5. Explore remaining advanced modules

## Support

- Read `README.md` for detailed documentation
- Check `MODULE_INFO.txt` for module overview
- Review inline code comments
- Consult prerequisites if stuck

Happy deploying! 🚀
