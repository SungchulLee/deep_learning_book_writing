# Human Evaluation for Generative Models

## Overview

Despite advances in automatic metrics, human evaluation remains the gold standard for assessing generative models. Humans can detect subtle artifacts, assess semantic coherence, and evaluate qualities that no metric can fully capture. This section covers methodologies, best practices, and practical implementations for human evaluation.

!!! info "Learning Objectives"
    By the end of this section, you will be able to:
    
    - Design effective human evaluation studies
    - Implement common evaluation paradigms (A/B testing, rating scales, Turing tests)
    - Analyze and report human evaluation results with statistical rigor
    - Understand when and how to combine human evaluation with automatic metrics
    - Avoid common pitfalls in human evaluation study design

## Why Human Evaluation?

### Limitations of Automatic Metrics

| Automatic Metric | What It Misses |
|-----------------|----------------|
| **FID** | Subtle artifacts, semantic errors, memorization |
| **IS** | Within-class diversity, unrealistic combinations |
| **Perplexity** | Coherence, factual accuracy, creativity |
| **BLEU/ROUGE** | Semantic equivalence, fluency, style |

### What Humans Excel At

- **Perceptual Quality**: Detecting visual artifacts invisible to metrics
- **Semantic Coherence**: Judging if generated content makes sense
- **Naturalness**: Assessing if output "feels" authentic
- **Task Relevance**: Evaluating if generation meets intended purpose
- **Preference**: Comparing subjective quality between options

## Evaluation Paradigms

### 1. Absolute Rating

Evaluators rate individual samples on a scale.

**Common Scales:**

- **Likert Scale (1-5)**: Strongly Disagree to Strongly Agree
- **MOS (Mean Opinion Score)**: 1-5 quality rating
- **Binary**: Real/Fake, Acceptable/Not Acceptable

```python
from dataclasses import dataclass
from typing import List, Optional, Dict
import numpy as np
from scipy import stats


@dataclass
class AbsoluteRating:
    """
    Absolute rating evaluation results.
    """
    sample_id: str
    evaluator_id: str
    rating: float  # e.g., 1-5
    criteria: str  # e.g., "quality", "coherence", "naturalness"
    timestamp: str
    comments: Optional[str] = None


class AbsoluteRatingAnalyzer:
    """
    Analyzer for absolute rating evaluations.
    """
    
    def __init__(self, ratings: List[AbsoluteRating]):
        self.ratings = ratings
    
    def compute_mos(self, criteria: Optional[str] = None) -> Dict:
        """
        Compute Mean Opinion Score with confidence interval.
        
        Args:
            criteria: Optional filter by criteria
        
        Returns:
            Dictionary with MOS statistics
        """
        if criteria:
            filtered = [r for r in self.ratings if r.criteria == criteria]
        else:
            filtered = self.ratings
        
        scores = [r.rating for r in filtered]
        
        n = len(scores)
        mean = np.mean(scores)
        std = np.std(scores, ddof=1)
        se = std / np.sqrt(n)
        
        # 95% CI
        ci_low = mean - 1.96 * se
        ci_high = mean + 1.96 * se
        
        return {
            'mos': mean,
            'std': std,
            'n': n,
            'ci_95': (ci_low, ci_high),
            'median': np.median(scores)
        }
    
    def compute_inter_annotator_agreement(self) -> Dict:
        """
        Compute inter-annotator agreement using Krippendorff's alpha.
        """
        # Group ratings by sample
        from collections import defaultdict
        
        by_sample = defaultdict(list)
        for r in self.ratings:
            by_sample[r.sample_id].append((r.evaluator_id, r.rating))
        
        # Build reliability matrix
        evaluators = list(set(r.evaluator_id for r in self.ratings))
        samples = list(by_sample.keys())
        
        matrix = np.full((len(evaluators), len(samples)), np.nan)
        
        for i, sample in enumerate(samples):
            for eval_id, rating in by_sample[sample]:
                j = evaluators.index(eval_id)
                matrix[j, i] = rating
        
        # Compute Krippendorff's alpha (simplified)
        # In practice, use krippendorff library
        observed_var = np.nanvar(matrix.flatten())
        if observed_var == 0:
            return {'alpha': 1.0, 'interpretation': 'Perfect agreement'}
        
        # Expected variance under random assignment
        all_values = matrix[~np.isnan(matrix)]
        expected_var = np.var(all_values)
        
        alpha = 1 - observed_var / expected_var if expected_var > 0 else 1.0
        
        interpretation = (
            'Excellent' if alpha > 0.8 else
            'Good' if alpha > 0.6 else
            'Moderate' if alpha > 0.4 else
            'Fair' if alpha > 0.2 else
            'Poor'
        )
        
        return {
            'alpha': alpha,
            'interpretation': interpretation,
            'n_evaluators': len(evaluators),
            'n_samples': len(samples)
        }


# Example evaluation form
def create_rating_task(sample, criteria=['quality', 'coherence', 'naturalness']):
    """
    Create a rating task specification.
    """
    return {
        'sample': sample,
        'instructions': """
Please rate the following generated image on a scale of 1-5:
1 = Very Poor
2 = Poor  
3 = Fair
4 = Good
5 = Excellent

Consider the following aspects:
- Quality: Are there visible artifacts, blur, or distortions?
- Coherence: Does the image make sense as a whole?
- Naturalness: Does it look like a real photograph?
        """,
        'criteria': {c: {'scale': (1, 5), 'required': True} for c in criteria}
    }
```

### 2. Pairwise Comparison (A/B Testing)

Evaluators choose between two options.

**Advantages:**
- Simpler than absolute rating
- More reliable for subtle differences
- Natural for comparing models

**Variants:**
- **2AFC (Two-Alternative Forced Choice)**: Must choose one
- **Best-of-N**: Choose best from N options
- **With Ties**: Allow "no preference"

```python
@dataclass
class PairwiseComparison:
    """
    Pairwise comparison result.
    """
    sample_a_id: str
    sample_b_id: str
    evaluator_id: str
    choice: str  # 'A', 'B', or 'tie'
    confidence: Optional[float] = None  # 1-5 confidence scale
    response_time_ms: Optional[int] = None


class PairwiseComparisonAnalyzer:
    """
    Analyzer for pairwise comparison evaluations.
    """
    
    def __init__(self, comparisons: List[PairwiseComparison]):
        self.comparisons = comparisons
    
    def compute_win_rate(self, 
                        model_a_samples: set,
                        model_b_samples: set,
                        include_ties: bool = False) -> Dict:
        """
        Compute win rate between two models.
        
        Args:
            model_a_samples: Set of sample IDs from model A
            model_b_samples: Set of sample IDs from model B
            include_ties: Whether to count ties as 0.5 win each
        
        Returns:
            Win rate statistics
        """
        a_wins = 0
        b_wins = 0
        ties = 0
        
        for c in self.comparisons:
            is_a_vs_b = (c.sample_a_id in model_a_samples and 
                        c.sample_b_id in model_b_samples)
            is_b_vs_a = (c.sample_a_id in model_b_samples and 
                        c.sample_b_id in model_a_samples)
            
            if is_a_vs_b:
                if c.choice == 'A':
                    a_wins += 1
                elif c.choice == 'B':
                    b_wins += 1
                else:
                    ties += 1
            elif is_b_vs_a:
                if c.choice == 'A':
                    b_wins += 1
                elif c.choice == 'B':
                    a_wins += 1
                else:
                    ties += 1
        
        total = a_wins + b_wins + (ties if include_ties else 0)
        
        if total == 0:
            return {'error': 'No comparisons found'}
        
        if include_ties:
            a_rate = (a_wins + 0.5 * ties) / total
        else:
            a_rate = a_wins / (a_wins + b_wins) if (a_wins + b_wins) > 0 else 0.5
        
        # Statistical test (binomial test)
        n = a_wins + b_wins
        p_value = stats.binom_test(a_wins, n, p=0.5) if n > 0 else 1.0
        
        return {
            'model_a_win_rate': a_rate,
            'model_b_win_rate': 1 - a_rate,
            'a_wins': a_wins,
            'b_wins': b_wins,
            'ties': ties,
            'total': total,
            'p_value': p_value,
            'significant': p_value < 0.05
        }
    
    def compute_bradley_terry(self, 
                             models: List[str],
                             sample_to_model: Dict[str, str]) -> Dict:
        """
        Compute Bradley-Terry model strengths from pairwise comparisons.
        
        The Bradley-Terry model estimates the "strength" of each model
        such that P(A beats B) = strength_A / (strength_A + strength_B).
        
        Args:
            models: List of model names
            sample_to_model: Mapping from sample ID to model name
        
        Returns:
            Model strengths and rankings
        """
        # Build win matrix
        n_models = len(models)
        wins = np.zeros((n_models, n_models))
        
        model_idx = {m: i for i, m in enumerate(models)}
        
        for c in self.comparisons:
            model_a = sample_to_model.get(c.sample_a_id)
            model_b = sample_to_model.get(c.sample_b_id)
            
            if model_a and model_b and c.choice != 'tie':
                i = model_idx[model_a]
                j = model_idx[model_b]
                
                if c.choice == 'A':
                    wins[i, j] += 1
                else:
                    wins[j, i] += 1
        
        # Iterative Bradley-Terry estimation
        strengths = np.ones(n_models)
        
        for _ in range(100):  # Iterate until convergence
            new_strengths = np.zeros(n_models)
            
            for i in range(n_models):
                numerator = wins[i, :].sum()
                denominator = 0
                
                for j in range(n_models):
                    if i != j:
                        n_ij = wins[i, j] + wins[j, i]
                        if n_ij > 0:
                            denominator += n_ij / (strengths[i] + strengths[j])
                
                if denominator > 0:
                    new_strengths[i] = numerator / denominator
                else:
                    new_strengths[i] = strengths[i]
            
            # Normalize
            new_strengths = new_strengths / new_strengths.sum() * n_models
            
            if np.allclose(strengths, new_strengths, atol=1e-6):
                break
            
            strengths = new_strengths
        
        # Convert to rankings and probabilities
        rankings = np.argsort(-strengths)
        
        return {
            'strengths': {models[i]: strengths[i] for i in range(n_models)},
            'rankings': [models[i] for i in rankings],
            'pairwise_probs': {
                f'{models[i]}_vs_{models[j]}': strengths[i] / (strengths[i] + strengths[j])
                for i in range(n_models) for j in range(n_models) if i != j
            }
        }
```

### 3. Turing Test (Real vs. Fake)

Evaluators classify samples as real or generated.

```python
@dataclass
class TuringTestResult:
    """
    Turing test classification result.
    """
    sample_id: str
    evaluator_id: str
    is_real_ground_truth: bool
    predicted_real: bool
    confidence: float  # 0-1


class TuringTestAnalyzer:
    """
    Analyzer for Turing test (real vs. fake) evaluations.
    """
    
    def __init__(self, results: List[TuringTestResult]):
        self.results = results
    
    def compute_metrics(self) -> Dict:
        """
        Compute Turing test metrics.
        
        Key metrics:
        - Fooling Rate: % of fake samples classified as real
        - Accuracy: Overall classification accuracy
        - Confidence Calibration: Is confidence aligned with accuracy?
        """
        tp = sum(1 for r in self.results if r.is_real_ground_truth and r.predicted_real)
        tn = sum(1 for r in self.results if not r.is_real_ground_truth and not r.predicted_real)
        fp = sum(1 for r in self.results if not r.is_real_ground_truth and r.predicted_real)
        fn = sum(1 for r in self.results if r.is_real_ground_truth and not r.predicted_real)
        
        n = len(self.results)
        n_real = sum(1 for r in self.results if r.is_real_ground_truth)
        n_fake = n - n_real
        
        accuracy = (tp + tn) / n if n > 0 else 0
        
        # Fooling rate: fake samples classified as real
        fooling_rate = fp / n_fake if n_fake > 0 else 0
        
        # Detection rate: fake samples correctly identified
        detection_rate = tn / n_fake if n_fake > 0 else 0
        
        # False positive rate (real classified as fake)
        false_rejection = fn / n_real if n_real > 0 else 0
        
        # 50% fooling rate = indistinguishable from real
        return {
            'accuracy': accuracy,
            'fooling_rate': fooling_rate,
            'detection_rate': detection_rate,
            'false_rejection_rate': false_rejection,
            'n_total': n,
            'n_real': n_real,
            'n_fake': n_fake,
            'tp': tp, 'tn': tn, 'fp': fp, 'fn': fn,
            'interpretation': self._interpret_fooling_rate(fooling_rate)
        }
    
    def _interpret_fooling_rate(self, rate: float) -> str:
        """Interpret fooling rate."""
        if rate > 0.45:
            return "Excellent - Nearly indistinguishable from real"
        elif rate > 0.30:
            return "Good - Often mistaken for real"
        elif rate > 0.15:
            return "Moderate - Sometimes convincing"
        else:
            return "Poor - Usually detected as fake"
    
    def compute_confidence_calibration(self) -> Dict:
        """
        Analyze if confidence is calibrated with accuracy.
        
        Well-calibrated: High confidence → Higher accuracy
        """
        # Bin by confidence
        bins = [(0, 0.2), (0.2, 0.4), (0.4, 0.6), (0.6, 0.8), (0.8, 1.0)]
        calibration = []
        
        for low, high in bins:
            in_bin = [r for r in self.results if low <= r.confidence < high]
            
            if len(in_bin) > 0:
                correct = sum(1 for r in in_bin 
                             if r.predicted_real == r.is_real_ground_truth)
                acc = correct / len(in_bin)
                mean_conf = np.mean([r.confidence for r in in_bin])
                
                calibration.append({
                    'bin': f'{low:.1f}-{high:.1f}',
                    'n': len(in_bin),
                    'accuracy': acc,
                    'mean_confidence': mean_conf,
                    'calibration_error': abs(acc - mean_conf)
                })
        
        ece = np.mean([c['calibration_error'] * c['n'] for c in calibration]) / len(self.results)
        
        return {
            'bins': calibration,
            'expected_calibration_error': ece
        }
```

## Study Design Best Practices

### 1. Sample Selection

```python
def design_evaluation_sample(
    generated_samples: List,
    real_samples: List,
    n_eval: int = 100,
    include_real_ratio: float = 0.3,
    stratify_by: Optional[str] = None
) -> Dict:
    """
    Design a balanced evaluation sample.
    
    Args:
        generated_samples: All generated samples
        real_samples: All real samples
        n_eval: Total samples to evaluate
        include_real_ratio: Fraction of real samples to include
        stratify_by: Optional stratification key
    
    Returns:
        Evaluation set specification
    """
    import random
    
    n_real = int(n_eval * include_real_ratio)
    n_gen = n_eval - n_real
    
    # Random selection (could add stratification)
    selected_real = random.sample(real_samples, min(n_real, len(real_samples)))
    selected_gen = random.sample(generated_samples, min(n_gen, len(generated_samples)))
    
    # Shuffle and assign IDs
    all_samples = []
    for i, s in enumerate(selected_real):
        all_samples.append({
            'id': f'sample_{i:04d}',
            'content': s,
            'is_real': True,
            'source': 'real'
        })
    
    for i, s in enumerate(selected_gen):
        all_samples.append({
            'id': f'sample_{n_real + i:04d}',
            'content': s,
            'is_real': False,
            'source': 'generated'
        })
    
    random.shuffle(all_samples)
    
    return {
        'samples': all_samples,
        'n_real': n_real,
        'n_generated': n_gen,
        'total': len(all_samples)
    }
```

### 2. Evaluator Selection and Training

**Key Considerations:**

- **Domain Expertise**: For specialized content (medical, legal), use experts
- **Diversity**: Include evaluators from different backgrounds
- **Training**: Provide clear instructions and practice examples
- **Calibration**: Use gold standard samples to calibrate evaluators

```python
def create_evaluator_guidelines(task_type: str) -> str:
    """
    Create evaluation guidelines based on task type.
    """
    guidelines = {
        'image_quality': """
## Image Quality Evaluation Guidelines

### Your Task
Rate each image on a scale of 1-5 based on visual quality.

### Rating Scale
- 5 (Excellent): Sharp, clear, no visible artifacts
- 4 (Good): Minor imperfections, but realistic overall
- 3 (Fair): Noticeable issues but recognizable
- 2 (Poor): Significant artifacts or distortions
- 1 (Very Poor): Severely degraded or unrecognizable

### What to Look For
- Blurriness or out-of-focus regions
- Unnatural colors or lighting
- Distorted shapes or proportions
- Repeated patterns or artifacts
- Missing or extra body parts (for people)

### Practice Examples
[Include 5 examples with ratings and explanations]
        """,
        
        'text_coherence': """
## Text Coherence Evaluation Guidelines

### Your Task
Rate each text passage on coherence and fluency.

### Rating Scale
- 5 (Excellent): Perfectly coherent, natural flow
- 4 (Good): Minor awkwardness but coherent
- 3 (Fair): Some incoherence but understandable
- 2 (Poor): Frequent incoherence or errors
- 1 (Very Poor): Incomprehensible

### What to Look For
- Logical flow between sentences
- Consistent topic and tone
- Grammatical correctness
- Natural word choices
- Factual consistency
        """,
        
        'real_vs_fake': """
## Real vs. Fake Classification Guidelines

### Your Task
Determine if each sample is REAL or GENERATED.

### Instructions
- Take your time with each sample
- Report your confidence (1-5)
- Don't overthink - trust your first impression

### Hints (What to Look For)
Images:
- Unusual textures or patterns
- Asymmetries in faces
- Background inconsistencies
- Strange lighting or shadows

Text:
- Repetitive phrases
- Factual errors
- Unusual word combinations
- Lack of coherent narrative
        """
    }
    
    return guidelines.get(task_type, "No guidelines available for this task type.")
```

### 3. Statistical Considerations

```python
def compute_sample_size(
    expected_effect: float,
    power: float = 0.8,
    alpha: float = 0.05,
    test_type: str = 'two_proportion'
) -> int:
    """
    Compute required sample size for human evaluation.
    
    Args:
        expected_effect: Expected difference (e.g., 0.1 for 10% difference)
        power: Statistical power (default 0.8)
        alpha: Significance level (default 0.05)
        test_type: Type of statistical test
    
    Returns:
        Required sample size per condition
    """
    from scipy.stats import norm
    
    # For two-proportion z-test
    if test_type == 'two_proportion':
        # Cohen's h for proportions
        p1 = 0.5  # baseline (e.g., 50% win rate)
        p2 = p1 + expected_effect
        
        # Pooled proportion
        p = (p1 + p2) / 2
        
        # Effect size (Cohen's h)
        h = 2 * (np.arcsin(np.sqrt(p1)) - np.arcsin(np.sqrt(p2)))
        
        # Sample size formula
        z_alpha = norm.ppf(1 - alpha / 2)
        z_beta = norm.ppf(power)
        
        n = 2 * ((z_alpha + z_beta) / h) ** 2
        
        return int(np.ceil(n))
    
    # For continuous outcomes (MOS comparison)
    elif test_type == 'continuous':
        # Assume effect size in standard deviation units
        d = expected_effect  # Cohen's d
        
        z_alpha = norm.ppf(1 - alpha / 2)
        z_beta = norm.ppf(power)
        
        n = 2 * ((z_alpha + z_beta) / d) ** 2
        
        return int(np.ceil(n))
    
    return 100  # Default


def analyze_results_significance(
    results_a: List[float],
    results_b: List[float],
    test_type: str = 'paired'
) -> Dict:
    """
    Statistical analysis of evaluation results.
    
    Args:
        results_a: Results for condition A
        results_b: Results for condition B
        test_type: 'paired' or 'independent'
    
    Returns:
        Statistical test results
    """
    if test_type == 'paired':
        statistic, p_value = stats.wilcoxon(results_a, results_b)
        test_name = 'Wilcoxon signed-rank'
    else:
        statistic, p_value = stats.mannwhitneyu(results_a, results_b)
        test_name = 'Mann-Whitney U'
    
    # Effect size (rank-biserial correlation for Mann-Whitney)
    n1, n2 = len(results_a), len(results_b)
    effect_size = 1 - (2 * statistic) / (n1 * n2) if test_type != 'paired' else None
    
    return {
        'test': test_name,
        'statistic': statistic,
        'p_value': p_value,
        'significant_005': p_value < 0.05,
        'significant_001': p_value < 0.01,
        'effect_size': effect_size,
        'mean_a': np.mean(results_a),
        'mean_b': np.mean(results_b),
        'median_a': np.median(results_a),
        'median_b': np.median(results_b)
    }
```

## Reporting Results

### Standard Reporting Format

```python
def generate_evaluation_report(
    mos_results: Dict,
    pairwise_results: Dict,
    turing_results: Dict,
    auto_metrics: Dict
) -> str:
    """
    Generate a comprehensive evaluation report.
    """
    report = """
# Human Evaluation Report

## Summary Statistics

### Mean Opinion Score (MOS)
| Criterion | MOS | 95% CI | N |
|-----------|-----|--------|---|
"""
    
    for criterion, stats in mos_results.items():
        report += f"| {criterion} | {stats['mos']:.2f} | [{stats['ci_95'][0]:.2f}, {stats['ci_95'][1]:.2f}] | {stats['n']} |\n"
    
    report += """
### Pairwise Comparisons
| Comparison | Win Rate | p-value | Significant |
|------------|----------|---------|-------------|
"""
    
    for comp, stats in pairwise_results.items():
        sig = '✓' if stats['significant'] else '✗'
        report += f"| {comp} | {stats['model_a_win_rate']:.1%} | {stats['p_value']:.4f} | {sig} |\n"
    
    report += f"""
### Turing Test
- Fooling Rate: {turing_results['fooling_rate']:.1%}
- Detection Accuracy: {turing_results['accuracy']:.1%}
- Interpretation: {turing_results['interpretation']}

## Automatic Metrics (for Reference)
- FID: {auto_metrics.get('fid', 'N/A')}
- IS: {auto_metrics.get('is', 'N/A')}
- Precision: {auto_metrics.get('precision', 'N/A')}
- Recall: {auto_metrics.get('recall', 'N/A')}

## Inter-Annotator Agreement
- Krippendorff's α: [VALUE]
- Interpretation: [INTERPRETATION]

## Limitations
- Sample size: [N] samples evaluated
- Evaluator pool: [N] evaluators
- Potential biases: [LIST]
"""
    
    return report
```

## Combining Human and Automatic Evaluation

### Correlation Analysis

```python
def analyze_metric_correlation(
    human_scores: np.ndarray,
    auto_scores: Dict[str, np.ndarray]
) -> Dict:
    """
    Analyze correlation between human and automatic metrics.
    
    Args:
        human_scores: Human evaluation scores [N]
        auto_scores: Dictionary of automatic metric scores
    
    Returns:
        Correlation analysis results
    """
    correlations = {}
    
    for metric_name, scores in auto_scores.items():
        # Pearson correlation
        r, p = stats.pearsonr(human_scores, scores)
        
        # Spearman correlation (rank-based)
        rho, p_spearman = stats.spearmanr(human_scores, scores)
        
        correlations[metric_name] = {
            'pearson_r': r,
            'pearson_p': p,
            'spearman_rho': rho,
            'spearman_p': p_spearman,
            'interpretation': _interpret_correlation(rho)
        }
    
    return correlations


def _interpret_correlation(r: float) -> str:
    """Interpret correlation coefficient."""
    r = abs(r)
    if r > 0.7:
        return "Strong"
    elif r > 0.5:
        return "Moderate"
    elif r > 0.3:
        return "Weak"
    else:
        return "Very weak or none"
```

## Common Pitfalls to Avoid

### 1. Evaluation Set Bias

```python
def check_evaluation_bias(samples: List[Dict]) -> Dict:
    """
    Check for potential biases in evaluation set.
    """
    warnings = []
    
    # Check for diversity
    if 'category' in samples[0]:
        categories = [s['category'] for s in samples]
        unique = len(set(categories))
        if unique < 5:
            warnings.append(f"Low category diversity: only {unique} categories")
    
    # Check for difficulty balance
    if 'difficulty' in samples[0]:
        difficulties = [s['difficulty'] for s in samples]
        if np.std(difficulties) < 0.5:
            warnings.append("Samples may be too similar in difficulty")
    
    return {
        'warnings': warnings,
        'passed': len(warnings) == 0
    }
```

### 2. Order Effects

- Randomize sample order for each evaluator
- Balance which model appears first in pairwise comparisons
- Include attention checks

### 3. Evaluator Fatigue

- Limit session length (30-45 minutes max)
- Include breaks
- Monitor response times for quality

## Summary

!!! success "Key Takeaways"
    
    1. **Human evaluation is essential**: Automatic metrics miss important qualities
    
    2. **Choose the right paradigm**: Absolute rating, pairwise comparison, or Turing test
    
    3. **Design carefully**: 
       - Balanced samples
       - Clear guidelines
       - Trained evaluators
       - Sufficient sample size
    
    4. **Report properly**:
       - Include confidence intervals
       - Report inter-annotator agreement
       - Use appropriate statistical tests
    
    5. **Combine with automatic metrics**: Human evaluation validates that metrics are meaningful

## References

1. van der Lee, C., et al. (2019). "Best Practices for the Human Evaluation of Automatically Generated Text." *INLG*.

2. Karpinska, M., et al. (2021). "Perils of Using AMT for Evaluation of Generative Models." *Findings of EMNLP*.

3. Zhou, Y., et al. (2019). "How Many Judges Does It Take to Evaluate Text Generation?" *NeurIPS*.

4. Celikyilmaz, A., et al. (2020). "Evaluation of Text Generation: A Survey." *arXiv*.

5. Borji, A. (2019). "Pros and Cons of GAN Evaluation Measures." *Computer Vision and Image Understanding*.
