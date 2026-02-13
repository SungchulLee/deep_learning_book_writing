# Online vs Offline Evaluation

## Overview

Recommender systems can be evaluated offline (using historical data) or online (with live users). The two approaches measure different things and can give contradictory results.

## Offline Evaluation

### Setup
Split historical interaction data into train and test sets. Train the model on the train set and evaluate predictions on the test set.

### Temporal Split
For time-sensitive recommendations, split by time: train on data before time $T$, test on data after $T$. This is more realistic than random splitting.

### Leave-One-Out
For each user, hold out one interaction for testing. Common for implicit feedback datasets.

### Limitations
- **Missing data bias**: users only interacted with items they were shown; items never shown are not necessarily irrelevant
- **Popularity bias**: offline metrics favor popular items because they have more test interactions
- **No novelty credit**: offline metrics cannot measure user satisfaction with novel recommendations

## Online Evaluation (A/B Testing)

### Setup
Deploy multiple recommendation algorithms simultaneously. Randomly assign users to treatment groups. Measure engagement metrics over a fixed period.

### Metrics
- Click-through rate (CTR)
- Conversion rate
- Session duration
- Return visits
- Revenue per user

### Statistical Significance
Use appropriate statistical tests (t-test, bootstrap) with multiple comparison correction. Minimum detectable effect size determines required sample size.

## Bridging the Gap

Several techniques improve offline-online correlation:

- **Counterfactual evaluation**: use propensity scores to correct for selection bias in offline data
- **Replay evaluation**: simulate online behavior using logged interaction data
- **Unbiased estimators**: inverse propensity scoring (IPS) to estimate online metrics from offline data
