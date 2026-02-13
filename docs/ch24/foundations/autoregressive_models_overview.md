# Autoregressive Models in PyTorch

## 📚 Educational Package for Undergraduates

This repository contains PyTorch implementations of various autoregressive models, designed for educational purposes. Each example is fully commented and includes detailed explanations.

## 🎯 What are Autoregressive Models?

**Autoregressive (AR) models** predict future values based on past values. The term "autoregressive" means the model regresses on itself—it uses its own previous outputs as inputs.

### Key Concept
In an autoregressive model, the probability of a sequence is decomposed as:

```
P(x₁, x₂, ..., xₙ) = P(x₁) × P(x₂|x₁) × P(x₃|x₁,x₂) × ... × P(xₙ|x₁,...,xₙ₋₁)
```

Each element is predicted conditioned on all previous elements.

## 📁 Repository Structure

```
autoregressive_models_pytorch/
├── README.md                          # This file
├── requirements.txt                   # Python dependencies
├── 1_time_series_ar/                  # Example 1: Time Series
│   ├── ar_model.py                    # AR(p) model implementation
│   ├── train.py                       # Training script
│   └── data.py                        # Data generation utilities
├── 2_char_language_model/             # Example 2: Language Model
│   ├── model.py                       # Character-level AR model
│   ├── train.py                       # Training script
│   └── data.py                        # Text data utilities
├── 3_image_generation/                # Example 3: Image Generation
│   ├── pixelcnn.py                    # Simplified PixelCNN
│   └── train.py                       # Training script
└── utils/                             # Shared utilities
    └── visualization.py               # Plotting functions
```

## 🚀 Quick Start

### Installation

1. **Clone or download this repository**

2. **Install dependencies:**
```bash
pip install -r requirements.txt
```

### Running the Examples

#### Example 1: Time Series Prediction
```bash
cd 1_time_series_ar
python train.py
```
This trains an AR(p) model on synthetic time series data (e.g., sine waves with noise).

#### Example 2: Character-Level Language Model
```bash
cd 2_char_language_model
python train.py
```
This trains a small autoregressive neural network to generate text character by character.

#### Example 3: Image Generation (PixelCNN)
```bash
cd 3_image_generation
python train.py
```
This trains a simplified PixelCNN to generate small images pixel by pixel.

## 📖 Detailed Example Descriptions

### 1. Time Series AR Model (`1_time_series_ar/`)

**What it does:** Implements a classical AR(p) model for time series prediction.

**Key Concepts:**
- Linear autoregression
- Order selection (choosing p)
- Forecasting future values

**Model Equation:**
```
X_t = c + φ₁X_{t-1} + φ₂X_{t-2} + ... + φₚX_{t-p} + ε_t
```

**Files:**
- `ar_model.py`: Contains the PyTorch AR model class
- `data.py`: Generates synthetic time series (sine waves, random walks)
- `train.py`: Training loop and visualization

### 2. Character-Level Language Model (`2_char_language_model/`)

**What it does:** Trains a neural network to predict the next character in a sequence.

**Key Concepts:**
- Sequential prediction
- Teacher forcing during training
- Autoregressive generation

**Architecture:**
- Embedding layer
- LSTM/GRU for sequential processing
- Linear layer for character prediction

**Files:**
- `model.py`: Neural AR language model
- `data.py`: Text processing and dataset creation
- `train.py`: Training and text generation

### 3. Image Generation (`3_image_generation/`)

**What it does:** Implements a simplified PixelCNN that generates images pixel by pixel.

**Key Concepts:**
- Masked convolutions (can only see previous pixels)
- Spatial autoregression
- Probabilistic image generation

**Files:**
- `pixelcnn.py`: Masked convolution and PixelCNN model
- `train.py`: Training on MNIST or simple patterns

## 🔑 Key Takeaways

### Advantages of Autoregressive Models:
1. **Flexible**: Can model complex distributions
2. **Tractable likelihood**: Easy to compute P(x)
3. **Natural for sequences**: Perfect for text, audio, time series

### Limitations:
1. **Sequential generation**: Slow (can't parallelize)
2. **Error accumulation**: Mistakes compound over long sequences
3. **Conditional independence assumptions**: May miss long-range dependencies

## 📚 Learning Resources

### Theory:
- **Time Series Analysis**: Box & Jenkins (ARIMA models)
- **Deep Learning**: Goodfellow et al. (Chapter on sequence models)
- **Probabilistic Models**: Bishop (Pattern Recognition and Machine Learning)

### Papers:
- **PixelCNN**: van den Oord et al. (2016) - "Pixel Recurrent Neural Networks"
- **WaveNet**: van den Oord et al. (2016) - "WaveNet: A Generative Model for Raw Audio"
- **GPT**: Radford et al. (2018) - "Improving Language Understanding by Generative Pre-Training"

## 🛠️ Modifying the Code

All code is designed to be educational and hackable:

1. **Change hyperparameters**: Look for clearly marked sections in `train.py` files
2. **Try different architectures**: Models are modular and easy to modify
3. **Use your own data**: Replace data loading functions with your datasets
4. **Experiment**: Comments guide you through each section

## 💡 Exercise Ideas

1. **Time Series**: Try forecasting real stock prices or weather data
2. **Language Model**: Train on different languages or code
3. **Image Generation**: Extend to color images or larger resolutions
4. **Hybrid Models**: Combine AR with other techniques (VAEs, diffusion)

## 🤝 Contributing

This is an educational resource. Feel free to:
- Add more examples
- Improve documentation
- Fix bugs
- Share your experiments

## 📧 Questions?

If you encounter issues or have questions:
1. Check the comments in the code—they're extensive!
2. Review the theory sections in this README
3. Experiment with simpler versions first

## 📄 License

This educational package is provided as-is for learning purposes.

---

**Happy Learning! 🎓**

Remember: The best way to learn is by doing. Run the code, break it, fix it, and make it your own!
