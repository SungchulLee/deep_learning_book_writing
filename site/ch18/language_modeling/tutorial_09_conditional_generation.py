"""
Tutorial 09: Controlled Text Generation Strategies
===================================================

Advanced techniques for controlling and improving text generation quality.

Topics:
1. Sampling strategies (greedy, beam search, nucleus, top-k)
2. Length normalization
3. Repetition penalty
4. Constrained decoding
5. Controllable generation (sentiment, style, topic)

Generation Strategies:
----------------------

1. Greedy Decoding:
   w_t = argmax P(w | context)
   - Fast, deterministic
   - Can be suboptimal, repetitive

2. Beam Search:
   - Keep top-k hypotheses
   - Score = log P / length_penalty
   - Better quality than greedy
   - Computationally expensive

3. Sampling:
   - Random: Sample from P(w | context)
   - Top-k: Sample from k most likely
   - Nucleus (top-p): Sample from smallest set with cumulative prob ≥ p

4. Temperature Scaling:
   P'(w) ∝ exp(logit / T)
   - T < 1: More deterministic
   - T > 1: More random
"""

import torch
import torch.nn.functional as F
import numpy as np
from typing import List, Optional


class GenerationStrategies:
    """Collection of text generation strategies."""
    
    @staticmethod
    def greedy_search(model, input_ids, max_length=50, vocab=None):
        """Greedy decoding - always pick most probable word."""
        model.eval()
        generated = input_ids.clone()
        
        for _ in range(max_length):
            with torch.no_grad():
                if hasattr(model, 'lstm') or hasattr(model, 'rnn'):
                    logits, _ = model(generated)
                else:
                    logits = model(generated)
                
                # Get logits for last position
                next_token_logits = logits[:, -1, :]
                
                # Greedy: pick most probable
                next_token = torch.argmax(next_token_logits, dim=-1)
                
                generated = torch.cat([generated, next_token.unsqueeze(-1)], dim=-1)
                
                # Check for end token
                if vocab and next_token.item() == vocab.word_to_idx(vocab.END_TOKEN):
                    break
        
        return generated
    
    @staticmethod
    def top_k_sampling(model, input_ids, max_length=50, k=50, 
                      temperature=1.0, vocab=None):
        """Sample from top k most probable tokens."""
        model.eval()
        generated = input_ids.clone()
        
        for _ in range(max_length):
            with torch.no_grad():
                if hasattr(model, 'lstm') or hasattr(model, 'rnn'):
                    logits, _ = model(generated)
                else:
                    logits = model(generated)
                
                next_token_logits = logits[:, -1, :]
                
                # Apply temperature
                next_token_logits = next_token_logits / temperature
                
                # Top-k filtering
                top_k_logits, top_k_indices = torch.topk(next_token_logits, k)
                probs = F.softmax(top_k_logits, dim=-1)
                
                # Sample from top k
                next_token_idx = torch.multinomial(probs, 1)
                next_token = top_k_indices.gather(-1, next_token_idx)
                
                generated = torch.cat([generated, next_token], dim=-1)
                
                if vocab and next_token.item() == vocab.word_to_idx(vocab.END_TOKEN):
                    break
        
        return generated
    
    @staticmethod
    def nucleus_sampling(model, input_ids, max_length=50, p=0.95,
                        temperature=1.0, vocab=None):
        """
        Nucleus (top-p) sampling.
        Sample from smallest set of tokens with cumulative prob >= p.
        """
        model.eval()
        generated = input_ids.clone()
        
        for _ in range(max_length):
            with torch.no_grad():
                if hasattr(model, 'lstm') or hasattr(model, 'rnn'):
                    logits, _ = model(generated)
                else:
                    logits = model(generated)
                
                next_token_logits = logits[:, -1, :] / temperature
                probs = F.softmax(next_token_logits, dim=-1)
                
                # Sort probabilities
                sorted_probs, sorted_indices = torch.sort(probs, descending=True)
                
                # Compute cumulative probabilities
                cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
                
                # Remove tokens with cumulative probability above threshold
                sorted_indices_to_remove = cumulative_probs > p
                # Keep at least one token
                sorted_indices_to_remove[..., 0] = False
                
                # Create mask
                indices_to_remove = sorted_indices_to_remove.scatter(
                    -1, sorted_indices, sorted_indices_to_remove
                )
                
                # Filter probabilities
                filtered_probs = probs.clone()
                filtered_probs[indices_to_remove] = 0
                filtered_probs = filtered_probs / filtered_probs.sum()
                
                # Sample
                next_token = torch.multinomial(filtered_probs, 1)
                generated = torch.cat([generated, next_token], dim=-1)
                
                if vocab and next_token.item() == vocab.word_to_idx(vocab.END_TOKEN):
                    break
        
        return generated
    
    @staticmethod
    def beam_search(model, input_ids, max_length=50, beam_width=5,
                   length_penalty=1.0, vocab=None):
        """
        Beam search decoding.
        Maintains beam_width hypotheses at each step.
        """
        model.eval()
        batch_size = input_ids.size(0)
        vocab_size = model.fc.out_features
        
        # Initialize beams: (batch_size * beam_width, seq_len)
        beams = input_ids.unsqueeze(1).repeat(1, beam_width, 1)
        beams = beams.view(batch_size * beam_width, -1)
        
        # Scores for each beam
        beam_scores = torch.zeros(batch_size, beam_width)
        beam_scores[:, 1:] = -float('inf')  # Only first beam active initially
        beam_scores = beam_scores.view(-1)
        
        for step in range(max_length):
            with torch.no_grad():
                if hasattr(model, 'lstm') or hasattr(model, 'rnn'):
                    logits, _ = model(beams)
                else:
                    logits = model(beams)
                
                next_token_logits = logits[:, -1, :]
                next_token_scores = F.log_softmax(next_token_logits, dim=-1)
                
                # Add to beam scores
                next_scores = beam_scores.unsqueeze(-1) + next_token_scores
                next_scores = next_scores.view(batch_size, -1)
                
                # Get top beam_width candidates
                top_scores, top_indices = torch.topk(next_scores, beam_width, dim=-1)
                
                # Compute which beam and which token
                beam_indices = top_indices // vocab_size
                token_indices = top_indices % vocab_size
                
                # Update beams
                new_beams = []
                new_scores = []
                
                for i in range(batch_size):
                    for j in range(beam_width):
                        beam_idx = i * beam_width + beam_indices[i, j]
                        new_beam = torch.cat([
                            beams[beam_idx],
                            token_indices[i, j].unsqueeze(0)
                        ])
                        new_beams.append(new_beam)
                        new_scores.append(top_scores[i, j])
                
                beams = torch.stack(new_beams)
                beam_scores = torch.tensor(new_scores)
        
        # Return best beam
        best_beam_idx = beam_scores[:beam_width].argmax()
        return beams[best_beam_idx].unsqueeze(0)


class RepetitionPenalty:
    """Apply repetition penalty to logits."""
    
    @staticmethod
    def apply(logits, generated_tokens, penalty=1.2):
        """
        Penalize repeated tokens by dividing their logits by penalty.
        
        Args:
            logits: (vocab_size,) logits
            generated_tokens: List of previously generated token ids
            penalty: Penalty factor (> 1.0)
        """
        for token in set(generated_tokens):
            logits[token] /= penalty
        return logits


def demonstrate_generation_strategies():
    """Compare different generation strategies."""
    
    print("Text Generation Strategies Comparison")
    print("=" * 70)
    
    print("""
Strategy Characteristics:
------------------------

1. Greedy Search:
   - Deterministic
   - Fast
   - Can get stuck in repetitive patterns
   - Use for: Simple completion, factual text

2. Beam Search:
   - More thorough search
   - Better quality than greedy
   - Still can be repetitive
   - Use for: Translation, summarization

3. Top-k Sampling:
   - Stochastic, diverse
   - Filters low-probability words
   - k=50 often works well
   - Use for: Creative writing, chat

4. Nucleus (Top-p) Sampling:
   - Dynamic vocabulary size
   - Adapts to probability distribution
   - p=0.9 to 0.95 recommended
   - Use for: General purpose, creative tasks

5. Temperature Sampling:
   - Controls randomness
   - T=0.7: More focused
   - T=1.0: Standard
   - T=1.5: More creative

Combining Strategies:
--------------------
Best practice: Nucleus + Temperature
- Top-p=0.95 for quality
- Temperature=0.8 for creativity
- Repetition penalty=1.2 to avoid loops
    """)


if __name__ == "__main__":
    demonstrate_generation_strategies()
    
    print("""
EXERCISES:
1. Implement repetition penalty in generation
2. Compare beam search with different beam widths
3. Implement length normalization for beam search
4. Try combining top-k and nucleus sampling
5. Implement constrained decoding (force certain words)
6. Create controllable generation with prefix tuning
7. Implement diverse beam search (multiple diverse outputs)
8. Add coverage mechanism to avoid repetition
    """)
