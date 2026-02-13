"""
Tutorial 10: Language Model Evaluation
=======================================

Comprehensive evaluation metrics and methods for language models.

Evaluation Aspects:
1. Intrinsic metrics (perplexity, bits-per-character)
2. Extrinsic metrics (downstream task performance)
3. Human evaluation
4. Diversity and quality metrics

Key Metrics:
-----------

1. Perplexity (PPL):
   PPL = exp(-1/N ∑ log P(w_i | context))
   - Lower is better
   - Measures how "surprised" model is by test data

2. Bits-per-character (BPC):
   BPC = -1/(N*log(2)) ∑ log P(w_i | context)
   - Lower is better
   - Normalized by character count

3. Cross-Entropy:
   H = -1/N ∑ log P(w_i | context)
   - Direct loss measure
   - Perplexity = exp(H)
"""

import torch
import torch.nn as nn
import numpy as np
from collections import Counter
from typing import List, Dict
import math


class LanguageModelEvaluator:
    """Comprehensive evaluation suite for language models."""
    
    def __init__(self, model, vocab, device='cpu'):
        self.model = model
        self.vocab = vocab
        self.device = device
        self.model.to(device)
    
    def compute_perplexity(self, test_corpus: List[str]) -> float:
        """
        Compute perplexity on test corpus.
        
        Perplexity = exp(average negative log-likelihood)
        
        Args:
            test_corpus: List of test sentences
            
        Returns:
            Perplexity score (lower is better)
        """
        self.model.eval()
        total_loss = 0
        total_words = 0
        
        criterion = nn.CrossEntropyLoss(reduction='sum', ignore_index=0)
        
        with torch.no_grad():
            for sentence in test_corpus:
                words = sentence.lower().split()
                words = [self.vocab.START_TOKEN] + words + [self.vocab.END_TOKEN]
                indices = [self.vocab.word_to_idx(w) for w in words]
                
                if len(indices) < 2:
                    continue
                
                # Create input and target
                input_seq = torch.tensor([indices[:-1]], dtype=torch.long).to(self.device)
                target_seq = torch.tensor([indices[1:]], dtype=torch.long).to(self.device)
                
                # Forward pass
                if hasattr(self.model, 'lstm') or hasattr(self.model, 'rnn'):
                    logits, _ = self.model(input_seq)
                else:
                    logits = self.model(input_seq)
                
                # Compute loss
                loss = criterion(logits.view(-1, logits.size(-1)), target_seq.view(-1))
                
                total_loss += loss.item()
                total_words += len(indices) - 1
        
        # Perplexity = exp(average loss)
        avg_loss = total_loss / total_words
        perplexity = math.exp(avg_loss)
        
        return perplexity
    
    def compute_bits_per_character(self, test_corpus: List[str]) -> float:
        """
        Compute bits-per-character metric.
        
        Args:
            test_corpus: List of test sentences
            
        Returns:
            BPC score (lower is better)
        """
        total_log_prob = 0
        total_chars = 0
        
        self.model.eval()
        
        with torch.no_grad():
            for sentence in test_corpus:
                words = sentence.lower().split()
                words = [self.vocab.START_TOKEN] + words + [self.vocab.END_TOKEN]
                indices = [self.vocab.word_to_idx(w) for w in words]
                
                if len(indices) < 2:
                    continue
                
                input_seq = torch.tensor([indices[:-1]], dtype=torch.long).to(self.device)
                target_seq = torch.tensor(indices[1:], dtype=torch.long).to(self.device)
                
                if hasattr(self.model, 'lstm') or hasattr(self.model, 'rnn'):
                    logits, _ = self.model(input_seq)
                else:
                    logits = self.model(input_seq)
                
                # Get log probabilities
                log_probs = torch.log_softmax(logits[0], dim=-1)
                
                # Sum log probabilities of target words
                for i, target_word_idx in enumerate(target_seq):
                    total_log_prob += log_probs[i, target_word_idx].item()
                
                # Count characters in original sentence
                total_chars += len(sentence)
        
        # BPC = -log_prob / (chars * log(2))
        bpc = -total_log_prob / (total_chars * math.log(2))
        
        return bpc
    
    def evaluate_generation_diversity(self, num_samples: int = 100,
                                     max_length: int = 20) -> Dict:
        """
        Evaluate diversity of generated samples.
        
        Metrics:
        - Unique n-grams
        - Self-BLEU (lower = more diverse)
        - Entropy
        
        Args:
            num_samples: Number of samples to generate
            max_length: Maximum length per sample
            
        Returns:
            Dictionary of diversity metrics
        """
        from tutorial_09_conditional_generation import GenerationStrategies
        
        # Generate samples
        samples = []
        for _ in range(num_samples):
            start_token = torch.tensor([[self.vocab.word_to_idx(self.vocab.START_TOKEN)]])
            generated = GenerationStrategies.nucleus_sampling(
                self.model, start_token, max_length=max_length,
                vocab=self.vocab
            )
            
            # Convert to text
            tokens = generated[0].tolist()
            words = [self.vocab.idx_to_word(idx) for idx in tokens
                    if idx != self.vocab.word_to_idx(self.vocab.PAD_TOKEN)]
            samples.append(' '.join(words))
        
        # Compute metrics
        metrics = {}
        
        # 1. Unique samples
        unique_samples = len(set(samples))
        metrics['unique_samples'] = unique_samples
        metrics['repetition_rate'] = 1 - (unique_samples / num_samples)
        
        # 2. Unique n-grams
        for n in [2, 3, 4]:
            all_ngrams = []
            for sample in samples:
                words = sample.split()
                ngrams = [tuple(words[i:i+n]) for i in range(len(words)-n+1)]
                all_ngrams.extend(ngrams)
            
            if all_ngrams:
                unique_ngrams = len(set(all_ngrams))
                total_ngrams = len(all_ngrams)
                metrics[f'unique_{n}grams'] = unique_ngrams / total_ngrams
        
        # 3. Average length
        metrics['avg_length'] = np.mean([len(s.split()) for s in samples])
        
        # 4. Vocabulary coverage
        all_words = set()
        for sample in samples:
            all_words.update(sample.split())
        metrics['vocab_coverage'] = len(all_words)
        
        return metrics


class BenchmarkSuite:
    """Standard benchmarks for language models."""
    
    @staticmethod
    def penn_treebank_benchmark(model, vocab):
        """Evaluate on Penn Treebank benchmark."""
        # This would load actual PTB data
        # Placeholder implementation
        print("Penn Treebank Benchmark")
        print("-" * 50)
        print("Vocabulary: ~10k words")
        print("Training: ~1M words")
        print("Validation: ~70k words")
        print("Test: ~80k words")
        print()
        print("State-of-the-art perplexities:")
        print("  LSTM (3-layer): ~60-80")
        print("  Transformer (6-layer): ~50-70")
        print("  AWD-LSTM: ~57")
        print("  Transformer-XL: ~54")
    
    @staticmethod
    def wikitext_benchmark(model, vocab):
        """Evaluate on WikiText-2 benchmark."""
        print("WikiText-2 Benchmark")
        print("-" * 50)
        print("Vocabulary: ~33k words")
        print("Training: ~2M words")
        print("More challenging than PTB (longer context)")
        print()
        print("State-of-the-art perplexities:")
        print("  LSTM (2-layer): ~100-120")
        print("  Transformer (6-layer): ~80-100")
        print("  GPT-2 (small): ~30-40")


def demonstrate_evaluation():
    """Demonstrate evaluation metrics."""
    
    print("=" * 70)
    print("Language Model Evaluation Metrics")
    print("=" * 70)
    
    print("""
1. PERPLEXITY
-------------
- Most common intrinsic metric
- Measures how well model predicts test data
- Lower = better
- Interpretation: "On average, model is choosing from PPL words"

Typical Values:
  N-gram models: 200-400 (PTB)
  LSTM: 80-120 (PTB)
  Transformer: 60-80 (PTB)
  Large pretrained: 20-40 (PTB)

Limitations:
- Not directly correlated with generation quality
- Can't compare across different vocabularies
- Doesn't measure semantic coherence


2. BITS-PER-CHARACTER
---------------------
- Normalized metric
- Comparable across different tokenizations
- Lower = better
- Used for character-level models


3. DIVERSITY METRICS
--------------------
- Distinct n-grams: Ratio of unique to total n-grams
- Self-BLEU: BLEU score between generated samples (lower = more diverse)
- Entropy: Shannon entropy of word distribution
- Repetition rate: Frequency of repeated sequences


4. HUMAN EVALUATION
-------------------
Aspects to evaluate:
- Fluency: Grammatical correctness
- Coherence: Logical flow
- Consistency: No contradictions
- Relevance: On-topic
- Factuality: Truthfulness

Rating scales:
- Likert scale (1-5)
- Pairwise comparison
- Best-worst scaling


5. DOWNSTREAM TASKS
-------------------
Evaluate on specific applications:
- Text completion
- Question answering
- Summarization
- Translation
- Dialogue


EVALUATION BEST PRACTICES:
---------------------------
1. Use multiple metrics (intrinsic + extrinsic)
2. Report confidence intervals
3. Test on multiple domains
4. Include human evaluation
5. Check for biases
6. Measure efficiency (speed, memory)
7. Evaluate failure modes
8. Compare against strong baselines
    """)


def compare_models():
    """Template for comparing different models."""
    
    print("\n" + "=" * 70)
    print("Model Comparison Framework")
    print("=" * 70)
    
    results = {
        'N-gram': {
            'perplexity': 350,
            'speed': 'Very Fast',
            'memory': 'Low',
            'diversity': 'Low'
        },
        'LSTM': {
            'perplexity': 100,
            'speed': 'Medium',
            'memory': 'Medium',
            'diversity': 'Medium'
        },
        'Transformer': {
            'perplexity': 70,
            'speed': 'Slow (train), Fast (inference)',
            'memory': 'High',
            'diversity': 'High'
        },
        'GPT-2': {
            'perplexity': 35,
            'speed': 'Medium',
            'memory': 'Very High',
            'diversity': 'Very High'
        }
    }
    
    print("\nModel Performance Comparison:")
    print("-" * 70)
    print(f"{'Model':<15} {'Perplexity':<12} {'Speed':<20} {'Memory':<10}")
    print("-" * 70)
    
    for model, metrics in results.items():
        print(f"{model:<15} {metrics['perplexity']:<12} "
              f"{metrics['speed']:<20} {metrics['memory']:<10}")


if __name__ == "__main__":
    demonstrate_evaluation()
    compare_models()
    
    print("""

EXERCISES:
1. Implement BLEU score for language models
2. Calculate self-BLEU for diversity measurement
3. Implement A/B testing framework
4. Create visualization dashboard for metrics
5. Implement cross-entropy decomposition analysis
6. Build human evaluation interface
7. Measure correlation between metrics and human judgments
8. Evaluate model calibration (confidence vs accuracy)
9. Test on adversarial examples
10. Measure fairness and bias in generated text
    """)
