# Module 34: Video Understanding

## Overview
This module covers deep learning techniques for understanding and analyzing video data, including temporal modeling, action recognition, and spatiotemporal feature extraction.

## Learning Objectives
- Understand the unique challenges of video data compared to images
- Learn 3D convolutions for spatiotemporal feature extraction
- Implement two-stream architectures for motion and appearance
- Build recurrent models for temporal sequence modeling
- Apply attention mechanisms to video understanding
- Understand modern video architectures (SlowFast, TimeSformer)
- Implement action recognition and video classification systems

## Mathematical Foundations

### 3D Convolution
For video input V ∈ ℝ^(T×H×W×C) (T temporal frames, H height, W width, C channels):

**3D Convolution:**
```
(V * K)(t,i,j) = ΣΣΣ Σ V(t+τ, i+m, j+n, c) · K(τ,m,n,c)
                  τ m n c
```
where K ∈ ℝ^(T_k×H_k×W_k×C) is the 3D kernel

**Key Insight:** 3D convolutions capture both spatial and temporal patterns simultaneously

### Two-Stream Network Architecture
- **Spatial Stream:** Processes single frames for appearance information
  - Input: RGB frame x_t ∈ ℝ^(H×W×3)
  - Features: f_spatial(x_t) using 2D CNN
  
- **Temporal Stream:** Processes optical flow for motion information
  - Input: Optical flow stack u_t ∈ ℝ^(H×W×2L) (L consecutive flows)
  - Features: f_temporal(u_t) using 2D CNN
  
- **Fusion:** Combined prediction
  ```
  p = softmax(α·f_spatial(x_t) + β·f_temporal(u_t))
  ```

### Recurrent Video Models
**CNN-LSTM Architecture:**
```
h_t = LSTM(CNN(x_t), h_{t-1})
```
- CNN extracts spatial features from each frame
- LSTM models temporal dependencies across frames

**Hidden State Update:**
```
i_t = σ(W_i·[h_{t-1}, x_t] + b_i)    [input gate]
f_t = σ(W_f·[h_{t-1}, x_t] + b_f)    [forget gate]
g_t = tanh(W_g·[h_{t-1}, x_t] + b_g)  [cell candidate]
o_t = σ(W_o·[h_{t-1}, x_t] + b_o)    [output gate]
c_t = f_t ⊙ c_{t-1} + i_t ⊙ g_t     [cell state]
h_t = o_t ⊙ tanh(c_t)                 [hidden state]
```

### Temporal Attention
**Self-Attention across frames:**
```
Attention(Q, K, V) = softmax(QK^T / √d_k)V
```
where:
- Q = query features from frame t
- K = key features from all frames
- V = value features from all frames
- d_k = dimension of key vectors

**Temporal pooling with attention:**
```
c = Σ α_t · h_t
where α_t = exp(score(h_t)) / Σ exp(score(h_j))
```

### Optical Flow
**Horn-Schunck Method:**
Minimize energy functional:
```
E = ∫∫ [(I_x·u + I_y·v + I_t)² + λ²(||∇u||² + ||∇v||²)] dx dy
```
where:
- (u, v) = optical flow vectors
- I_x, I_y, I_t = image derivatives (spatial and temporal)
- λ = smoothness weight

**Farnebäck Dense Optical Flow:**
Approximates neighborhoods with quadratic polynomials for robust flow estimation

## Module Structure

### Beginner Level (01-04)
1. **01_video_basics.py** - Loading and processing videos
2. **02_3d_convolution.py** - 3D CNNs for video
3. **03_simple_video_classifier.py** - Basic video classification
4. **04_data_augmentation_video.py** - Video augmentation techniques

### Intermediate Level (05-08)
5. **05_two_stream_network.py** - Two-stream architecture
6. **06_optical_flow.py** - Optical flow computation
7. **07_cnn_lstm_video.py** - Recurrent models for video
8. **08_action_recognition.py** - Action recognition system

### Advanced Level (09-12)
9. **09_temporal_attention.py** - Attention mechanisms for video
10. **10_slowfast_network.py** - SlowFast architecture
11. **11_video_transformer.py** - Vision Transformers for video
12. **12_video_understanding_project.py** - Complete video analysis system

## Prerequisites
- Module 23: Convolutional Neural Networks
- Module 25: Attention Mechanisms
- Module 26: Transformers (for advanced topics)
- Module 28: Recurrent Neural Networks
- Module 29: LSTM and GRU

## Key Concepts Covered
- Video data representation and preprocessing
- Spatiotemporal convolutions (3D CNNs)
- Temporal modeling with RNNs
- Two-stream architectures
- Optical flow computation
- Temporal attention mechanisms
- Action recognition
- Video classification
- Modern video architectures (SlowFast, TimeSformer)
- Video datasets (UCF-101, Kinetics, HMDB-51)

## Datasets Used
1. **UCF-101**: 101 action categories, ~13k videos
2. **Kinetics-400**: 400 human action classes, ~300k videos
3. **HMDB-51**: 51 action categories, ~7k videos
4. **Something-Something-v2**: Temporal reasoning dataset

## Key Applications
- Action recognition and classification
- Video surveillance and anomaly detection
- Sports analytics
- Human activity recognition
- Video captioning
- Temporal action localization
- Video retrieval and search

## Additional Resources
- Papers:
  - "Two-Stream Convolutional Networks for Action Recognition" (Simonyan & Zisserman, 2014)
  - "3D Convolutional Neural Networks for Human Action Recognition" (Ji et al., 2013)
  - "SlowFast Networks for Video Recognition" (Feichtenhofer et al., 2019)
  - "TimeSformer: Is Space-Time Attention All You Need for Video Understanding?" (2021)
- Datasets: UCF-101, Kinetics, HMDB-51
- Libraries: torchvision, OpenCV, decord

## Estimated Time
- Beginner: 4-5 hours
- Intermediate: 6-7 hours
- Advanced: 8-10 hours
- Total: 18-22 hours

## Tips for Teaching
1. Start with static frame analysis before temporal modeling
2. Visualize optical flow to understand motion representation
3. Compare performance with and without temporal information
4. Discuss computational efficiency (3D CNNs vs. 2D CNNs + RNNs)
5. Emphasize the importance of temporal modeling for understanding dynamics
6. Show failure cases where spatial-only models fail
7. Connect to human perception of motion and time
