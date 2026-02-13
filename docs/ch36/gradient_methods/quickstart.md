# Quick Start Guide - Saliency Maps Tutorial

## 🚀 Getting Started in 5 Minutes

### Step 1: Installation
```bash
# Install required packages
pip install torch torchvision Pillow numpy matplotlib

# Or install from requirements file
pip install -r requirements.txt
```

### Step 2: Run Your First Example
```bash
# Start with the simplest method
python 01_vanilla_gradient_saliency.py

# Try a cleaner visualization
python 03_smoothgrad.py

# Or jump to the best method
python 07_guided_gradcam.py
```

### Step 3: Explore the Modules

**Beginner** (Start here!)
- `01_vanilla_gradient_saliency.py` - Basic gradient-based saliency
- `02_gradient_input_saliency.py` - Improved attribution
- `03_smoothgrad.py` - Noise reduction

**Intermediate** (Build understanding)
- `04_integrated_gradients.py` - Theoretically principled
- `05_gradcam.py` - Region-level localization
- `06_guided_backpropagation.py` - High-resolution details

**Advanced** (Master techniques)
- `07_guided_gradcam.py` - State-of-the-art visualization
- `08_comparative_analysis.py` - Method comparison
- `09_advanced_techniques.py` - Cutting-edge approaches

## 📝 Basic Usage Example

```python
from utils import load_pretrained_model, preprocess_image
import torch

# Load model
model = load_pretrained_model('resnet50')

# Load image
image = preprocess_image('your_image.jpg', requires_grad=True)

# Get prediction
output = model(image)
predicted_class = output.argmax(dim=1).item()

# Compute saliency (vanilla gradient)
output[0, predicted_class].backward()
saliency = image.grad.abs().max(dim=1)[0]

# Visualize
from utils import visualize_saliency
visualize_saliency(image, saliency, title="My First Saliency Map")
```

## 🎯 Which Module Should I Use?

| Goal | Module | Time |
|------|--------|------|
| Quick understanding | 01 | 5 min |
| Better attribution | 02 | 5 min |
| Clean visualization | 03 | 10 min |
| Theory-backed | 04 | 15 min |
| Localization | 05 | 10 min |
| Best quality | 07 | 15 min |
| Compare all methods | 08 | 20 min |

## 📚 Learning Path

### Week 1: Foundations (3-4 hours)
- Day 1: Modules 01-02 (understand gradients)
- Day 2: Module 03 (noise reduction)
- Day 3: Practice with your own images

### Week 2: Advanced Methods (4-5 hours)
- Day 1: Modules 04-05 (theory + localization)
- Day 2: Module 06-07 (high-res + combined)
- Day 3: Module 08 (compare everything)

### Week 3: Mastery (3-4 hours)
- Apply to your research/project
- Explore Module 09 (advanced techniques)
- Experiment with different architectures

## 🔧 Common Issues

### Issue: Out of Memory
```python
# Solution: Use smaller batch size or CPU
device = torch.device('cpu')  # Force CPU
```

### Issue: Gradients are None
```python
# Solution: Set requires_grad=True
image_tensor.requires_grad = True
```

### Issue: Model not found
```python
# Solution: Download manually or check internet connection
model = models.resnet50(pretrained=True)
```

## 💡 Pro Tips

1. **Start simple**: Begin with Module 01 before jumping to advanced methods
2. **GPU optional**: All examples work on CPU (GPU just faster)
3. **Experiment**: Try different images and model architectures
4. **Compare**: Use Module 08 to see differences between methods
5. **Read papers**: Check README.md for paper references

## 📖 Additional Resources

- **Utils.py**: Common functions for all modules
- **README.md**: Comprehensive mathematical foundations
- **Requirements.txt**: All dependencies

## 🎓 For Instructors

### Classroom Use (3-week module)

**Week 1: Gradient-Based Methods**
- Lecture: Modules 01-03
- Lab: Implement vanilla gradient
- Assignment: Compare methods on dataset

**Week 2: Advanced Techniques**
- Lecture: Modules 04-07
- Lab: Implement Grad-CAM
- Assignment: Create visualizations

**Week 3: Applications**
- Lecture: Modules 08-09
- Lab: Full pipeline
- Project: Apply to real problem

### Suggested Assessments
1. Implement a saliency method from scratch
2. Compare 3+ methods on misclassified examples
3. Final project: Interpretability pipeline for domain-specific model

## 🤝 Contributing

Found a bug? Have a suggestion?
- Add comments to code
- Create variants for different architectures
- Share interesting visualizations

## 📄 License

Educational use only. See individual files for details.

## ✅ Checklist

- [ ] Installed PyTorch and dependencies
- [ ] Ran first example (Module 01)
- [ ] Understood basic gradients
- [ ] Tried with own image
- [ ] Compared multiple methods
- [ ] Read mathematical foundations
- [ ] Applied to own model
- [ ] Explored advanced techniques

**Ready to dive deeper? Start with `01_vanilla_gradient_saliency.py`!**
