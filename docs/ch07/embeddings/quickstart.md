# Quick Start Guide

## Get Started in 5 Minutes!

### Step 1: Setup (2 minutes)
```bash
# Extract the package
unzip word_embeddings_tutorial.zip
cd word_embeddings_tutorial

# Install dependencies
pip install -r requirements.txt
```

### Step 2: Run Your First Example (1 minute)
```bash
cd 01_basics
python 01_simple_embeddings.py
```

This introduces you to word embeddings with clear explanations!

### Step 3: Build Your First Model (2 minutes)
```bash
python 02_ngram_cross_entropy.py
```

You'll train a complete n-gram language model and see:
- Training progress
- Loss curves
- Learned embeddings

## Learning Path

**Beginner** (1-2 hours):
```
01_basics/
├── 01_simple_embeddings.py      ← Start here!
├── 02_ngram_cross_entropy.py    
├── 03_ngram_functional.py       
└── 04_ngram_nll_loss.py         
```

**Intermediate** (2-3 hours):
```
02_intermediate/
├── 01_loss_comparison.py        
├── 02_cbow_model.py             ← Important!
├── 03_skipgram_model.py         
└── 04_embedding_analysis.py     
```

**Advanced** (3-4 hours):
```
03_advanced/
├── 01_word2vec_full.py          
├── 02_negative_sampling.py      
├── 03_pretrained_embeddings.py  
└── 04_embedding_visualization.py ← Great finale!
```

## Tips for Success

1. **Read the comments** - Every line is explained
2. **Run before modifying** - Understand baseline first
3. **Experiment** - Change hyperparameters
4. **Visualize** - Use the plotting functions
5. **Take breaks** - Let concepts sink in

## Common Commands

```bash
# Run any tutorial
python <tutorial_name>.py

# Check if packages are installed
pip list | grep torch

# Re-install if needed
pip install -r requirements.txt --upgrade
```

## Need Help?

- Each tutorial has KEY TAKEAWAYS section
- README.md has detailed explanations
- Code has extensive comments
- Start with basics before advanced topics

## Next Steps After Completion

1. Try with your own text data
2. Implement word analogies (king - man + woman = queen)
3. Build a simple chatbot using embeddings
4. Explore pre-trained models (GloVe, FastText)
5. Apply to real NLP tasks (classification, sentiment analysis)

**Happy Learning! 🚀**
