# Module 34: Video Understanding - Usage Guide

## Quick Start

### Installation
```bash
pip install -r REQUIREMENTS.txt
```

### Running Examples

#### Beginner Level
```bash
# 1. Video basics - loading and processing
python 01_video_basics.py

# 2. 3D Convolutions
python 02_3d_convolution.py

# 3. Simple video classifier
python 03_simple_video_classifier.py

# 4. Data augmentation
python 04_data_augmentation_video.py
```

#### Intermediate Level
```bash
# 5. Two-stream networks
python 05_two_stream_network.py

# 6. Optical flow computation
python 06_optical_flow.py

# 7. CNN-LSTM architecture
python 07_cnn_lstm_video.py

# 8. Action recognition
python 08_action_recognition.py
```

#### Advanced Level
```bash
# 9. Temporal attention
python 09_temporal_attention.py

# 10. SlowFast networks
python 10_slowfast_network.py

# 11. Video transformers
python 11_video_transformer.py

# 12. Complete project
python 12_video_understanding_project.py
```

## Module Structure

### Beginner (01-04): Foundations
- **01_video_basics.py**: Loading, processing, sampling videos
- **02_3d_convolution.py**: 3D CNNs and C3D architecture
- **03_simple_video_classifier.py**: Complete training pipeline
- **04_data_augmentation_video.py**: Temporal and spatial augmentation

### Intermediate (05-08): Core Techniques
- **05_two_stream_network.py**: Spatial + temporal streams
- **06_optical_flow.py**: Dense optical flow computation
- **07_cnn_lstm_video.py**: Recurrent models for video
- **08_action_recognition.py**: End-to-end action recognition

### Advanced (09-12): Modern Architectures
- **09_temporal_attention.py**: Attention mechanisms for video
- **10_slowfast_network.py**: SlowFast dual-pathway network
- **11_video_transformer.py**: Vision transformers for video (TimeSformer)
- **12_video_understanding_project.py**: Complete production system

## Key Concepts by Difficulty

### Beginner Concepts
- Video as temporal sequence of frames
- 3D convolutions vs 2D convolutions
- Spatiotemporal feature extraction
- Basic video classification pipeline

### Intermediate Concepts
- Two-stream architectures
- Optical flow computation
- Recurrent models (CNN-LSTM)
- Motion vs appearance modeling
- Fusion strategies

### Advanced Concepts
- Temporal attention mechanisms
- Multi-scale temporal modeling (SlowFast)
- Video transformers
- Action localization
- Real-time inference optimization

## Common Use Cases

### 1. Action Recognition
```python
from 08_action_recognition import ActionRecognizer

# Load model
model = ActionRecognizer(num_classes=400)

# Predict
video = load_video("path/to/video.mp4")
action = model.predict(video)
```

### 2. Real-time Video Analysis
```python
from 12_video_understanding_project import VideoAnalyzer

analyzer = VideoAnalyzer()
results = analyzer.analyze_stream(video_source=0)  # Webcam
```

### 3. Custom Dataset Training
```python
from 03_simple_video_classifier import VideoClassifier

# Create dataset
dataset = CustomVideoDataset(video_dir, labels)

# Train
classifier = VideoClassifier(model)
classifier.train(train_loader, val_loader, epochs=50)
```

## Datasets

### Recommended Datasets
1. **UCF-101**: 101 action classes, ~13k videos
2. **Kinetics-400**: 400 action classes, ~300k videos  
3. **HMDB-51**: 51 action classes, ~7k videos
4. **Something-Something-v2**: Temporal reasoning, ~220k videos

### Dataset Download
```bash
# UCF-101
wget https://www.crcv.ucf.edu/data/UCF101/UCF101.rar

# Kinetics (requires kinetics-dataset package)
pip install kinetics-dataset
```

## Performance Tips

### Memory Optimization
```python
# Reduce batch size
batch_size = 4  # Instead of 16

# Reduce temporal resolution
num_frames = 8  # Instead of 16

# Use mixed precision training
scaler = torch.cuda.amp.GradScaler()
```

### Speed Optimization
```python
# Use efficient video loader
from decord import VideoReader

# Compile model (PyTorch 2.0+)
model = torch.compile(model)

# Reduce spatial resolution
resize_to = (112, 112)  # Instead of (224, 224)
```

## Troubleshooting

### Common Issues

**Issue**: Out of memory errors
**Solution**: Reduce batch size, number of frames, or spatial resolution

**Issue**: Slow training
**Solution**: Use more workers in DataLoader, enable pin_memory, use GPU

**Issue**: Poor accuracy
**Solution**: Check data preprocessing, try data augmentation, train longer

**Issue**: Optical flow computation is slow
**Solution**: Use GPU-based flow (RAFT), or compute offline and cache

## References

### Papers
1. **Two-Stream CNNs**: Simonyan & Zisserman (2014)
2. **C3D**: Tran et al. (2015)
3. **I3D**: Carreira & Zisserman (2017)
4. **SlowFast**: Feichtenhofer et al. (2019)
5. **TimeSformer**: Bertasius et al. (2021)

### Resources
- [PyTorch Video Models](https://github.com/pytorch/vision)
- [MMAction2](https://github.com/open-mmlab/mmaction2)
- [Video Understanding Survey](https://arxiv.org/abs/2012.06567)

## Contact & Support

For questions or issues with this module:
- Check the README.md for mathematical foundations
- Review code comments for implementation details
- Refer to papers for theoretical background
- Experiment with hyperparameters for your specific task

## License

Educational use only. Cite original papers when using architectures.
