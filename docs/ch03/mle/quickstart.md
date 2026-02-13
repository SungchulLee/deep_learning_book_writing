# Quick Start Guide

## Welcome to PyTorch MLE Tutorial! 🎉

This guide will get you up and running in 5 minutes.

## Step 1: Installation

```bash
# Create and activate virtual environment (recommended)
python -m venv mle_env
source mle_env/bin/activate  # On Windows: mle_env\Scripts\activate

# Install required packages
pip install -r requirements.txt
```

## Step 2: Run Your First Example

```bash
# Start with the coin flip example (easiest!)
python 01_easy/coin_flip_mle.py
```

This will:
- Generate synthetic coin flip data
- Estimate the probability using MLE
- Create beautiful visualizations
- Save a figure showing the results

## Step 3: Explore More Examples

### Easy Level (Start here!)
```bash
python 01_easy/coin_flip_mle.py       # Estimate biased coin probability
python 01_easy/dice_roll_mle.py      # Estimate loaded die probabilities
```

### Medium Level (When you're comfortable)
```bash
python 02_medium/linear_regression_mle.py      # Connect MLE to regression
python 02_medium/normal_distribution_mle.py    # Estimate mean and variance
python 02_medium/capture_recapture_mle.py      # Wildlife population estimation
```

### Advanced Level (For the ambitious!)
```bash
python 03_advanced/logistic_regression_mle.py    # Binary classification
python 03_advanced/mixture_of_gaussians_em.py    # EM algorithm
python 03_advanced/neural_network_mle.py         # Deep learning with uncertainty
```

## What to Expect

Each example will:
1. ✅ Explain the problem and theory
2. ✅ Generate or load data
3. ✅ Implement MLE (both analytical and numerical)
4. ✅ Create visualizations
5. ✅ Save figures to your directory
6. ✅ Provide exercises to extend your learning

## Customization

Most examples accept command-line arguments. For example:

```bash
python 01_easy/coin_flip_mle.py --n_flips 1000 --true_p 0.3
```

Check the code for available options!

## Getting Help

- **Read the comments**: Every line is documented
- **Check the docstrings**: Functions explain their purpose
- **Try the exercises**: Found at the end of each script
- **Modify the code**: Change parameters and see what happens!

## Learning Path

```
Day 1: coin_flip_mle.py + dice_roll_mle.py
Day 2: linear_regression_mle.py + normal_distribution_mle.py
Day 3: capture_recapture_mle.py
Day 4: logistic_regression_mle.py
Day 5: mixture_of_gaussians_em.py
Day 6: neural_network_mle.py
```

## Tips for Success

1. **Start simple**: Don't skip the easy examples!
2. **Read the code**: Understanding is more important than running
3. **Do the exercises**: True learning comes from doing
4. **Experiment**: Change parameters and see what happens
5. **Be patient**: MLE is powerful but takes time to master

## Troubleshooting

### Import Error
```
pip install --upgrade torch numpy matplotlib scipy scikit-learn
```

### Plot Not Showing
- Add `plt.show()` if needed
- Check if running in correct environment
- Try saving the figure instead: already done automatically!

### Slow Execution
- Reduce `n_iterations` or `n_samples` in the code
- Normal for advanced examples with larger networks

## Next Steps

After completing the tutorials:
- Read the main README.md for deeper theory
- Try the exercises at the end of each script
- Apply MLE to your own datasets
- Explore PyTorch documentation for more advanced features

## Happy Learning! 🚀

Remember: The best way to learn MLE is by implementing it yourself. 
This tutorial gives you the tools – now go build something amazing!
