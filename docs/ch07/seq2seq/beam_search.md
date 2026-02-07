# Beam Search

Beam search is a heuristic search algorithm that expands the most promising nodes in a limited set, providing a balance between greedy decoding (which keeps only the single best candidate) and exhaustive search (which is computationally intractable). In sequence-to-sequence models, beam search significantly improves generation quality by exploring multiple hypotheses simultaneously.

## The Fundamental Decoding Problem

Given an encoder representation, the decoder must find the most likely output sequence:

$$\hat{\mathbf{y}} = \arg\max_{\mathbf{y}} P(\mathbf{y}|\mathbf{x}) = \arg\max_{\mathbf{y}} \prod_{t=1}^{T} P(y_t|y_{<t}, \mathbf{x})$$

This optimization problem is intractable for exact solution. For a vocabulary of size $V$ and maximum sequence length $T$, the number of possible output sequences is $V^T$. Even for modest values ($V=10{,}000$, $T=50$), exhaustive search requires evaluating approximately $10^{200}$ sequences—far exceeding computational feasibility.

## Limitations of Greedy Decoding

Greedy decoding selects the highest probability token at each timestep:

$$\hat{y}_t = \arg\max_{y} P(y | \hat{y}_{<t}, \mathbf{c})$$

While computationally efficient ($O(TV)$ complexity), greedy decoding has fundamental limitations:

**Local optima**: The locally optimal choice at each step may not lead to the globally optimal sequence. Consider translation where "Il" (French) could map to "He" or "It"—choosing "He" might seem locally optimal but could lead to poor subsequent probabilities if the subject is actually inanimate.

**Irreversibility**: Once a token is selected, the decision cannot be reconsidered. Early mistakes propagate through the entire sequence.

**Mode collapse**: Greedy decoding tends to produce generic, high-frequency outputs rather than diverse, contextually appropriate responses.

**Probability concentration illusion**: High confidence at one step does not guarantee sequence quality. A 90% confident first token followed by poor continuations may yield a worse sequence than a 60% confident start with excellent continuations.

## Beam Search Algorithm

### Core Concept

Beam search maintains the $K$ most promising partial hypotheses (beams) at each timestep, ranked by their cumulative log probability:

$$\text{score}(y_1, \ldots, y_t) = \sum_{i=1}^{t} \log P(y_i | y_{<i}, \mathbf{c})$$

### Mathematical Formulation

Let $\mathcal{B}_t = \{(\mathbf{y}^{(1)}, s^{(1)}), \ldots, (\mathbf{y}^{(K)}, s^{(K)})\}$ be the beam at timestep $t$, where $\mathbf{y}^{(k)}$ is the partial sequence and $s^{(k)}$ is its score.

For each beam $k$ and vocabulary item $v$:

$$s_{new} = s^{(k)} + \log P(v | \mathbf{y}^{(k)}, \mathbf{c})$$

The new beam $\mathcal{B}_{t+1}$ contains the $K$ candidates with highest $s_{new}$.

### Algorithm Visualization

```
Beam width K=3

Step 0: [<SOS>]

Step 1: Expand <SOS> → top 3:
  ["The" (−1.2), "A" (−1.5), "In" (−1.8)]

Step 2: Expand each → 3V candidates → keep top 3:
  ["The cat" (−2.4), "A dog" (−2.6), "The dog" (−2.7)]

Step 3: Continue until <EOS> or max length
```

### Computational Complexity

- **Time**: $O(T \cdot K \cdot V)$ where $T$ is sequence length, $K$ is beam width, $V$ is vocabulary size
- **Space**: $O(K \cdot T)$ for storing $K$ hypotheses of length up to $T$
- **Practical optimization**: Top-$K$ selection can be done in $O(V + K \log K)$ using partial sorting

## PyTorch Implementation

### Hypothesis Data Structure

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
from typing import List, Tuple, Optional


@dataclass
class BeamHypothesis:
    """
    Represents a single beam hypothesis during search.
    
    Attributes:
        tokens: Generated token sequence including <SOS>
        score: Cumulative log probability score
        hidden: Decoder hidden state for continuation
        cell: LSTM cell state (if applicable)
        attention_weights: History of attention distributions for coverage
    """
    tokens: List[int]
    score: float
    hidden: torch.Tensor
    cell: Optional[torch.Tensor] = None
    attention_weights: Optional[List[torch.Tensor]] = None
    
    def __len__(self) -> int:
        return len(self.tokens)
    
    def __repr__(self) -> str:
        return f"Hypothesis(len={len(self)}, score={self.score:.4f})"
```

### Core Beam Search Decoder

```python
class BeamSearchDecoder:
    """
    Beam search decoder for sequence-to-sequence models.
    
    Implements standard beam search with length normalization using
    Google's length penalty formula from Wu et al. (2016).
    
    Args:
        model: Seq2seq model with encoder and decoder attributes
        beam_width: Number of beams to maintain (K)
        max_length: Maximum output sequence length
        sos_idx: Start-of-sequence token index
        eos_idx: End-of-sequence token index
        length_penalty: Length normalization exponent (alpha)
        device: Computation device
    """
    
    def __init__(
        self,
        model: nn.Module,
        beam_width: int = 5,
        max_length: int = 50,
        sos_idx: int = 1,
        eos_idx: int = 2,
        length_penalty: float = 0.6,
        device: torch.device = None
    ):
        self.model = model
        self.beam_width = beam_width
        self.max_length = max_length
        self.sos_idx = sos_idx
        self.eos_idx = eos_idx
        self.length_penalty = length_penalty
        self.device = device or torch.device('cpu')
        
    def length_normalize(self, score: float, length: int) -> float:
        """
        Apply length normalization to prevent bias toward shorter sequences.
        
        Uses the formula from Wu et al. (2016):
            lp(Y) = ((5 + |Y|) / 6)^alpha
        
        The constant 5 provides smoothing for very short sequences.
        """
        lp = ((5.0 + length) / 6.0) ** self.length_penalty
        return score / lp
    
    @torch.no_grad()
    def decode(
        self,
        src: torch.Tensor,
        src_lengths: Optional[torch.Tensor] = None,
        return_attention: bool = False
    ) -> Tuple[List[int], float]:
        """
        Perform beam search decoding for a single source sequence.
        
        Args:
            src: Source sequence tensor (1, src_len)
            src_lengths: Actual source length for masking
            return_attention: Whether to return attention weight history
            
        Returns:
            Tuple of (best_sequence, best_score)
        """
        self.model.eval()
        
        # Encode source sequence
        encoder_outputs, hidden, cell = self.model.encoder(src, src_lengths)
        
        mask = None
        if hasattr(self.model, 'create_mask'):
            mask = self.model.create_mask(src)
        
        # Initialize beam with <SOS> token
        initial_hypothesis = BeamHypothesis(
            tokens=[self.sos_idx],
            score=0.0,
            hidden=hidden,
            cell=cell,
            attention_weights=[] if return_attention else None
        )
        
        beams = [initial_hypothesis]
        completed = []
        
        for step in range(self.max_length):
            if not beams:
                break
                
            all_candidates = []
            
            for beam in beams:
                if beam.tokens[-1] == self.eos_idx:
                    completed.append(beam)
                    continue
                
                decoder_input = torch.tensor(
                    [[beam.tokens[-1]]], device=self.device
                )
                
                # Decoder forward pass
                if hasattr(self.model.decoder, 'attention'):
                    output, new_hidden, new_cell, attention = self.model.decoder(
                        decoder_input, beam.hidden, encoder_outputs, beam.cell, mask
                    )
                else:
                    output, new_hidden, new_cell = self.model.decoder(
                        decoder_input, beam.hidden, beam.cell
                    )
                    attention = None
                
                log_probs = F.log_softmax(output, dim=-1).squeeze(0)
                top_log_probs, top_indices = log_probs.topk(self.beam_width)
                
                for log_prob, token_idx in zip(top_log_probs, top_indices):
                    new_tokens = beam.tokens + [token_idx.item()]
                    new_score = beam.score + log_prob.item()
                    
                    new_attention = None
                    if return_attention and attention is not None:
                        new_attention = (beam.attention_weights or []) + [attention]
                    
                    candidate = BeamHypothesis(
                        tokens=new_tokens,
                        score=new_score,
                        hidden=new_hidden,
                        cell=new_cell,
                        attention_weights=new_attention
                    )
                    all_candidates.append(candidate)
            
            # Prune to top K by length-normalized score
            all_candidates.sort(
                key=lambda h: self.length_normalize(h.score, len(h)),
                reverse=True
            )
            beams = all_candidates[:self.beam_width]
        
        completed.extend(beams)
        
        if not completed:
            return [self.sos_idx], 0.0
        
        best = max(
            completed,
            key=lambda h: self.length_normalize(h.score, len(h))
        )
        
        if return_attention:
            return best.tokens, best.score, best.attention_weights
        return best.tokens, best.score
```

### Batched Beam Search

For production efficiency, process multiple beams in parallel on the GPU:

```python
class BatchedBeamSearch:
    """
    Batched beam search leveraging GPU parallelism.
    
    The key insight is representing all K beams for B batch items
    as a single tensor of shape (B*K, ...), allowing parallel
    computation across all hypotheses.
    """
    
    def __init__(
        self,
        model: nn.Module,
        beam_width: int = 5,
        max_length: int = 50,
        sos_idx: int = 1,
        eos_idx: int = 2,
        length_penalty: float = 0.6,
        device: torch.device = None
    ):
        self.model = model
        self.beam_width = beam_width
        self.max_length = max_length
        self.sos_idx = sos_idx
        self.eos_idx = eos_idx
        self.length_penalty = length_penalty
        self.device = device or torch.device('cpu')
        
    @torch.no_grad()
    def decode(self, src, src_lengths=None):
        """
        Batched beam search decoding.
        
        Args:
            src: Source sequences (batch_size, src_len)
            src_lengths: Actual source lengths per item
            
        Returns:
            sequences: Best output sequences (batch_size, max_len)
            scores: Corresponding scores (batch_size,)
        """
        self.model.eval()
        batch_size = src.size(0)
        
        encoder_outputs, hidden, cell = self.model.encoder(src, src_lengths)
        
        # Expand for beam search: (B, ...) -> (B*K, ...)
        encoder_outputs = self._expand_for_beam(encoder_outputs, batch_size)
        hidden = self._expand_for_beam(hidden, batch_size)
        if cell is not None:
            cell = self._expand_for_beam(cell, batch_size)
        
        # Initialize beam scores (only first beam per batch active)
        beam_scores = torch.zeros(batch_size * self.beam_width, device=self.device)
        beam_scores[1::self.beam_width] = -1e9
        
        generated = torch.full(
            (batch_size * self.beam_width, self.max_length),
            self.sos_idx, dtype=torch.long, device=self.device
        )
        
        for step in range(1, self.max_length):
            prev_tokens = generated[:, step - 1].unsqueeze(1)
            
            if hasattr(self.model.decoder, 'attention'):
                output, hidden, cell, _ = self.model.decoder(
                    prev_tokens, hidden, encoder_outputs, cell
                )
            else:
                output, hidden, cell = self.model.decoder(prev_tokens, hidden, cell)
            
            log_probs = F.log_softmax(output, dim=-1)
            vocab_size = log_probs.size(-1)
            
            next_scores = beam_scores.unsqueeze(-1) + log_probs
            next_scores = next_scores.view(batch_size, -1)
            
            top_scores, top_indices = next_scores.topk(2 * self.beam_width, dim=-1)
            beam_indices = top_indices // vocab_size
            token_indices = top_indices % vocab_size
            
            new_scores = []
            new_tokens = []
            for b in range(batch_size):
                batch_offset = b * self.beam_width
                for rank in range(self.beam_width):
                    beam_idx = batch_offset + beam_indices[b, rank].item()
                    token_idx = token_indices[b, rank].item()
                    new_scores.append(top_scores[b, rank].item())
                    new_tokens.append((beam_idx, token_idx))
            
            for i, (beam_idx, token_idx) in enumerate(new_tokens):
                generated[i, :step] = generated[beam_idx, :step]
                generated[i, step] = token_idx
                
            beam_scores = torch.tensor(new_scores, device=self.device)
            
            if (generated[:, step] == self.eos_idx).all():
                break
        
        beam_scores = beam_scores.view(batch_size, self.beam_width)
        best_beams = beam_scores.argmax(dim=-1)
        
        best_sequences = []
        best_scores_list = []
        for b in range(batch_size):
            beam_idx = b * self.beam_width + best_beams[b].item()
            best_sequences.append(generated[beam_idx])
            best_scores_list.append(beam_scores[b, best_beams[b]].item())
        
        return torch.stack(best_sequences), torch.tensor(best_scores_list)
    
    def _expand_for_beam(self, tensor, batch_size):
        """Expand tensor from (batch, ...) to (batch * beam, ...)."""
        if tensor.dim() == 2:
            return tensor.unsqueeze(1).expand(-1, self.beam_width, -1).reshape(
                batch_size * self.beam_width, -1)
        elif tensor.dim() == 3:
            return tensor.unsqueeze(2).expand(-1, -1, self.beam_width, -1).reshape(
                tensor.size(0), batch_size * self.beam_width, -1)
        else:
            raise ValueError(f"Unexpected tensor dimension: {tensor.dim()}")
```

## Coverage Penalty

In attention-based models, beam search may repeatedly attend to the same source positions while ignoring others. The coverage penalty tracks cumulative attention and penalizes under-attended positions:

```python
def coverage_penalty(
    attention_weights: List[torch.Tensor],
    beta: float = 0.2
) -> float:
    """
    Compute coverage penalty from attention history.
    
    For each source position j:
        penalty_j = log(min(coverage_j, 1))
    
    This is 0 when coverage >= 1 (adequately attended)
    and negative when coverage < 1 (under-attended).
    """
    if not attention_weights:
        return 0.0
    
    coverage = torch.stack(attention_weights).sum(dim=0)
    penalty = torch.sum(torch.log(torch.clamp(coverage, max=1.0)))
    return beta * penalty.item()
```

## Diverse Beam Search

Standard beam search often produces nearly identical hypotheses. Diverse beam search (Vijayakumar et al., 2016) divides beams into $G$ groups, each with $K/G$ beams. When processing group $g$, tokens selected by groups $0 \ldots g-1$ receive a diversity penalty $\lambda$, encouraging novel word choices:

```python
class DiverseBeamSearch:
    """
    Diverse beam search with inter-group dissimilarity penalty.
    """
    
    def __init__(self, model, beam_width=6, num_groups=3, 
                 diversity_penalty=0.5, max_length=50, 
                 sos_idx=1, eos_idx=2, device=None):
        self.model = model
        self.beam_width = beam_width
        self.num_groups = num_groups
        self.group_size = beam_width // num_groups
        self.diversity_penalty = diversity_penalty
        self.max_length = max_length
        self.sos_idx = sos_idx
        self.eos_idx = eos_idx
        self.device = device or torch.device('cpu')
    
    @torch.no_grad()
    def decode(self, src, src_lengths=None):
        self.model.eval()
        encoder_outputs, hidden, cell = self.model.encoder(src, src_lengths)
        
        # Initialize groups
        groups = []
        for g in range(self.num_groups):
            groups.append([BeamHypothesis(
                tokens=[self.sos_idx], score=0.0,
                hidden=hidden.clone(),
                cell=cell.clone() if cell is not None else None
            )])
        
        for step in range(self.max_length):
            selected_tokens = set()
            
            for g, beams in enumerate(groups):
                all_candidates = []
                for beam in beams:
                    if beam.tokens[-1] == self.eos_idx:
                        all_candidates.append(beam)
                        continue
                    
                    decoder_input = torch.tensor(
                        [[beam.tokens[-1]]], device=self.device)
                    output, new_h, new_c = self.model.decoder(
                        decoder_input, beam.hidden, beam.cell)
                    
                    log_probs = F.log_softmax(output, dim=-1).squeeze(0)
                    penalized = log_probs.clone()
                    for token in selected_tokens:
                        penalized[token] -= self.diversity_penalty
                    
                    top_probs, top_idx = penalized.topk(self.group_size * 2)
                    for lp, ti in zip(top_probs, top_idx):
                        all_candidates.append(BeamHypothesis(
                            tokens=beam.tokens + [ti.item()],
                            score=beam.score + log_probs[ti].item(),
                            hidden=new_h, cell=new_c))
                
                all_candidates.sort(key=lambda h: h.score, reverse=True)
                groups[g] = all_candidates[:self.group_size]
                
                for beam in groups[g]:
                    if len(beam.tokens) > step:
                        selected_tokens.add(beam.tokens[-1])
        
        results = []
        for beams in groups:
            for beam in beams:
                results.append((beam.tokens, beam.score))
        results.sort(key=lambda x: x[1], reverse=True)
        return results
```

## Practical Optimizations

### Early Stopping

Terminate search when the best completed hypothesis beats all active beams:

```python
def should_stop_early(beams, completed, length_normalize_fn):
    """
    Since log probabilities are non-positive, active beam scores
    can only decrease. Stop if best completed beats best active.
    """
    if not beams or not completed:
        return not beams
    
    best_completed = max(
        length_normalize_fn(h.score, len(h)) for h in completed)
    best_active = max(
        length_normalize_fn(h.score, len(h)) for h in beams)
    
    return best_completed > best_active
```

### Repetition Prevention

```python
def apply_repetition_penalty(log_probs, generated_tokens, 
                              penalty=1.2, no_repeat_ngram=3):
    """
    Token-level penalty + n-gram blocking to prevent degenerate repetition.
    """
    log_probs = log_probs.clone()
    
    for token in set(generated_tokens):
        log_probs[token] /= penalty
    
    if len(generated_tokens) >= no_repeat_ngram - 1:
        prefix = tuple(generated_tokens[-(no_repeat_ngram-1):])
        for i in range(len(generated_tokens) - no_repeat_ngram + 1):
            if tuple(generated_tokens[i:i+no_repeat_ngram-1]) == prefix:
                log_probs[generated_tokens[i + no_repeat_ngram - 1]] = float('-inf')
    
    return log_probs
```

## Comparison of Decoding Strategies

| Method | Quality | Speed | Diversity | Best For |
|--------|---------|-------|-----------|----------|
| Greedy | Low | Fastest | None | Real-time, simple tasks |
| Beam search | High | Medium | Low | Translation, summarization |
| Sampling | Variable | Fast | High | Creative generation |
| Top-$k$ sampling | Good | Fast | Medium | Open-ended dialogue |
| Nucleus (top-$p$) | Good | Fast | Medium | Story generation |
| Diverse beam | Good | Slow | High | Multiple suggestions |

## Hyperparameter Guidelines

| Parameter | Typical Range | Notes |
|-----------|---------------|-------|
| Beam width ($K$) | 4–10 | 4–5 for production, 10+ for research |
| Length penalty ($\alpha$) | 0.6–1.0 | Higher = longer outputs |
| Max length | 1.5–2× source | Prevent runaway generation |
| Coverage penalty ($\beta$) | 0.0–0.4 | For attention-based models |
| Diversity penalty ($\lambda$) | 0.3–0.8 | Higher = more diversity |
| No-repeat n-gram | 2–4 | Block repeated phrases |

### Beam Width Selection

| Beam Width | Quality | Latency | Use Case |
|------------|---------|---------|----------|
| 1 (greedy) | Lowest | Fastest | Real-time applications |
| 4–5 | Good | Fast | Production systems |
| 10–20 | Better | Moderate | Quality-focused offline |
| 50+ | Diminishing returns | Slow | Research/analysis |

## Summary

Beam search provides a principled approach to sequence generation that balances exploration and computational efficiency. The core mechanism maintains $K$ best partial hypotheses at each step, pruning the search space from $V^T$ to $K \cdot T \cdot V$ operations. Length normalization is essential to prevent bias toward shorter sequences, and coverage penalty ensures complete source coverage in attention-based models. Diverse beam search encourages variety among top hypotheses through inter-group dissimilarity penalties.

Beam search remains the dominant decoding strategy for tasks requiring faithful reproduction of information (translation, summarization), while stochastic methods like nucleus sampling are preferred for open-ended creative generation where diversity matters more than finding the single highest-probability sequence.
