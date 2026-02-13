# Quick Start Guide

Welcome to the PyTorch Softmax Regression Tutorial Series!

## 🚀 Get Started in 3 Steps

### Step 1: Install Dependencies

```bash
pip install torch torchvision numpy matplotlib scikit-learn
```

Or use the requirements file:
```bash
pip install -r requirements.txt
```

### Step 2: Choose Your Starting Point

**Complete Beginner?**
```bash
python 01_fundamentals.py
```

**Some ML Experience?**
```bash
python 03_mnist.py
```

**Want Advanced Techniques?**
```bash
python 04_advanced.py
```

**Ready for Production Code?**
```bash
python 05_comprehensive.py
```

### Step 3: Learn and Experiment!

Each file is:
- ✅ Fully self-contained (no imports from other files)
- ✅ Extensively commented
- ✅ Ready to run immediately
- ✅ Includes educational explanations

## 📖 What's in Each File?

### Level 1: `01_fundamentals.py`
**Time:** 20-30 minutes  
**Learns:** Softmax, cross-entropy, PyTorch basics  
**Runs:** Instantly (no large downloads)

### Level 2: `02_simple_classifier.py`
**Time:** 30-45 minutes  
**Learns:** Build & train neural networks  
**Runs:** Instantly (synthetic data)

### Level 3: `03_mnist.py`
**Time:** 45-60 minutes  
**Learns:** Work with real datasets  
**Runs:** Downloads MNIST first time (~50MB)

### Level 4: `04_advanced.py`
**Time:** 60-90 minutes  
**Learns:** Advanced techniques  
**Runs:** Quick (uses synthetic & cached data)

### Level 5: `05_comprehensive.py`
**Time:** 90-120 minutes  
**Learns:** Production ML pipelines  
**Runs:** Downloads multiple datasets first time

## 💡 Tips

1. **Read the comments!** They explain what's happening and why.

2. **Run the code as-is first** before modifying.

3. **Uncomment visualization lines** to see plots:
   ```python
   # plt.show()  # ← Remove the # to see plots
   ```

4. **Start simple** - Don't jump to Level 5 immediately!

5. **Experiment!** Change hyperparameters and see what happens.

## 🎯 Learning Goals by Level

- **Level 1:** Understand the math
- **Level 2:** Build basic models
- **Level 3:** Train on real data
- **Level 4:** Master advanced techniques
- **Level 5:** Create production code

## 🆘 Common Issues

**Issue: "No module named torch"**
```bash
pip install torch torchvision
```

**Issue: "CUDA out of memory"**
- Reduce batch_size in the code
- Or use CPU: device = 'cpu'

**Issue: "Can't download dataset"**
- Check internet connection
- Datasets download automatically on first run

## 📊 What You'll Achieve

After completing this series, you'll be able to:
- Build multi-class classifiers from scratch
- Train on real image datasets
- Apply advanced ML techniques
- Create production-ready training pipelines

**Estimated Total Time:** 4-6 hours
**Skill Level Gained:** Intermediate to Advanced

## 🎓 Recommended Path

1. **Day 1:** Level 1 + Level 2 (1-2 hours)
2. **Day 2:** Level 3 (1 hour)
3. **Day 3:** Level 4 (1-2 hours)
4. **Day 4:** Level 5 (1-2 hours)

Or do it all in one sitting if you're ambitious! ☕

## 🌟 Ready to Start?

```bash
python 01_fundamentals.py
```

Happy learning! 🚀

For detailed documentation, see README.md
