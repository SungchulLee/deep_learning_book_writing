# Beam Search

Beam search is a heuristic search algorithm that expands the most promising nodes in a limited set, providing a balance between greedy decoding (which keeps only the single best candidate) and exhaustive search (which is computationally intractable). In sequence-to-sequence models, beam search significantly improves generation quality by exploring multiple hypotheses simultaneously.

## The Fundamental Decoding Problem

Given an encoder representation, the decoder must find the most likely output sequence:

$$\hat{y} = \arg\max_{y} P(y|x) = \arg\max_{y} \prod_{t=1}^{T} P(y_t|y_{<t}, x)$$

This optimization problem is intractable for exact solution. For a vocabulary of size $V$ and maximum sequence length $T$, the number of possible output sequences is $V^T$. Even for modest values ($V=10000$, $T=50$), exhaustive search requires evaluating approximately $10^{200}$ sequences—far exceeding computational feasibility.

## Limitations of Greedy Decoding

Greedy decoding selects the highest probability token at each timestep:

$$\hat{y}_t = \arg\max_{y} P(y | \hat{y}_{<t}, \mathbf{c})$$

While computationally efficient ($O(TV)$ complexity), greedy decoding has fundamental limitations:

**Local Optima**: The locally optimal choice at each step may not lead to the globally optimal sequence. Consider translation where "Il" (French) could map to "He" or "It"—choosing "He" might seem locally optimal but could lead to poor subsequent probabilities if the subject is actually inanimate.

**Irreversibility**: Once a token is selected, the decision cannot be reconsidered. Early mistakes propagate through the entire sequence, compounding errors.

**Mode Collapse**: Greedy decoding tends to produce generic, high-frequency outputs rather than diverse, contextually appropriate responses. The decoder gravitates toward safe, common phrases.

**Probability Concentration Illusion**: High confidence at one step doesn't guarantee sequence quality. A 90% confident first token followed by poor options may yield worse sequences than a 60% confident start with excellent continuations.

## Beam Search Algorithm

### Core Concept

Beam search provides a tractable approximation by maintaining only the $K$ most promising partial hypotheses (beams) at each timestep, ranked by their cumulative log probability:

$$\text{score}(y_1, \ldots, y_t) = \sum_{i=1}^{t} \log P(y_i | y_{<i}, \mathbf{c})$$

### Mathematical Formulation

Let $\mathcal{B}_t = \{(y^{(1)}, s^{(1)}), \ldots, (y^{(K)}, s^{(K)})\}$ be the beam at timestep $t$, where $y^{(k)}$ is the partial sequence and $s^{(k)}$ is its score.

For each beam $k$ and vocabulary item $v$:

$$s_{new} = s^{(k)} + \log P(v | y^{(k)}, \mathbf{c})$$

The new beam $\mathcal{B}_{t+1}$ contains the $K$ candidates with highest $s_{new}$.

### Algorithm Visualization

```
Beam width k=3

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
- **Practical optimization**: Top-K selection can be done in $O(V + K \log K)$ using partial sorting

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
        length_penalty: Length normalization exponent (α)
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
            lp(Y) = ((5 + |Y|) / 6)^α
        
        The constant 5 provides smoothing for very short sequences.
        
        Args:
            score: Raw cumulative log probability
            length: Sequence length
            
        Returns:
            Length-normalized score
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
            Tuple of (best_sequence, best_score) or
            (best_sequence, best_score, attention_weights) if return_attention
        """
        self.model.eval()
        
        # Encode source sequence
        encoder_outputs, hidden, cell = self.model.encoder(src, src_lengths)
        
        # Create attention mask if model supports it
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
                # Move completed sequences to completed list
                if beam.tokens[-1] == self.eos_idx:
                    completed.append(beam)
                    continue
                
                # Prepare decoder input: last generated token
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
                
                # Compute log probabilities over vocabulary
                log_probs = F.log_softmax(output, dim=-1).squeeze(0)
                
                # Get top K candidates for expansion
                top_log_probs, top_indices = log_probs.topk(self.beam_width)
                
                # Create new hypotheses
                for log_prob, token_idx in zip(top_log_probs, top_indices):
                    new_tokens = beam.tokens + [token_idx.item()]
                    new_score = beam.score + log_prob.item()
                    
                    # Track attention history if requested
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
        
        # Add remaining active beams to completed
        completed.extend(beams)
        
        if not completed:
            return [self.sos_idx], 0.0
        
        # Select best hypothesis by normalized score
        best = max(
            completed,
            key=lambda h: self.length_normalize(h.score, len(h))
        )
        
        if return_attention:
            return best.tokens, best.score, best.attention_weights
        return best.tokens, best.score
```

### Batched Beam Search

For production efficiency, process multiple beams in parallel:

```python
class BatchedBeamSearch:
    """
    Batched beam search leveraging GPU parallelism.
    
    Processes all beams for a batch simultaneously, achieving
    significant speedup over sequential processing.
    
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
    def decode(
        self,
        src: torch.Tensor,
        src_lengths: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
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
        
        # Encode all source sequences
        encoder_outputs, hidden, cell = self.model.encoder(src, src_lengths)
        
        # Expand encoder outputs for beam search: (B, ...) → (B*K, ...)
        encoder_outputs = self._expand_for_beam(encoder_outputs, batch_size)
        hidden = self._expand_for_beam(hidden, batch_size)
        if cell is not None:
            cell = self._expand_for_beam(cell, batch_size)
        
        # Create and expand attention mask
        mask = None
        if hasattr(self.model, 'create_mask'):
            mask = self.model.create_mask(src)
            mask = self._expand_for_beam(mask, batch_size)
        
        # Initialize beam scores
        # Shape: (batch_size * beam_width,)
        # Only first beam per batch item is active initially (others set to -inf)
        beam_scores = torch.zeros(batch_size * self.beam_width, device=self.device)
        beam_scores[1::self.beam_width] = -1e9
        
        # Track generated tokens: (batch * beam, max_len)
        generated = torch.full(
            (batch_size * self.beam_width, self.max_length),
            self.sos_idx,
            dtype=torch.long,
            device=self.device
        )
        
        # Track completion status
        done = torch.zeros(
            batch_size * self.beam_width, 
            dtype=torch.bool, 
            device=self.device
        )
        
        for step in range(1, self.max_length):
            # Get previous tokens for all beams
            prev_tokens = generated[:, step - 1].unsqueeze(1)
            
            # Parallel decoder step for all beams
            if hasattr(self.model.decoder, 'attention'):
                output, hidden, cell, _ = self.model.decoder(
                    prev_tokens, hidden, encoder_outputs, cell, mask
                )
            else:
                output, hidden, cell = self.model.decoder(
                    prev_tokens, hidden, cell
                )
            
            # Log probabilities: (batch * beam, vocab)
            log_probs = F.log_softmax(output, dim=-1)
            vocab_size = log_probs.size(-1)
            
            # Combine with beam scores: (batch * beam, vocab)
            next_scores = beam_scores.unsqueeze(-1) + log_probs
            
            # Reshape for top-k selection: (batch, beam * vocab)
            next_scores = next_scores.view(batch_size, -1)
            
            # Get top 2K candidates per batch item (extra for EOS handling)
            top_scores, top_indices = next_scores.topk(
                2 * self.beam_width, dim=-1
            )
            
            # Decode beam and token indices
            beam_indices = top_indices // vocab_size
            token_indices = top_indices % vocab_size
            
            # Select new beams for each batch item
            new_scores = []
            new_tokens = []
            
            for b in range(batch_size):
                batch_offset = b * self.beam_width
                selected_count = 0
                
                for rank in range(2 * self.beam_width):
                    if selected_count >= self.beam_width:
                        break
                    
                    beam_idx = batch_offset + beam_indices[b, rank].item()
                    token_idx = token_indices[b, rank].item()
                    score = top_scores[b, rank].item()
                    
                    new_scores.append(score)
                    new_tokens.append((beam_idx, token_idx))
                    selected_count += 1
            
            # Update beam states
            for i, (beam_idx, token_idx) in enumerate(new_tokens):
                generated[i, :step] = generated[beam_idx, :step]
                generated[i, step] = token_idx
                
            beam_scores = torch.tensor(new_scores, device=self.device)
            
            # Early stopping if all beams have generated EOS
            done = generated[:, step] == self.eos_idx
            if done.all():
                break
        
        # Extract best sequence per batch item
        beam_scores = beam_scores.view(batch_size, self.beam_width)
        best_beams = beam_scores.argmax(dim=-1)
        
        best_sequences = []
        best_scores = []
        
        for b in range(batch_size):
            beam_idx = b * self.beam_width + best_beams[b].item()
            best_sequences.append(generated[beam_idx])
            best_scores.append(beam_scores[b, best_beams[b]].item())
        
        return torch.stack(best_sequences), torch.tensor(best_scores)
    
    def _expand_for_beam(self, tensor: torch.Tensor, batch_size: int) -> torch.Tensor:
        """
        Expand tensor from (batch, ...) to (batch * beam, ...).
        
        Each batch item is replicated beam_width times.
        """
        if tensor.dim() == 2:
            # (batch, hidden) → (batch * beam, hidden)
            return tensor.unsqueeze(1).expand(-1, self.beam_width, -1).reshape(
                batch_size * self.beam_width, -1
            )
        elif tensor.dim() == 3:
            # (layers, batch, hidden) → (layers, batch * beam, hidden)
            return tensor.unsqueeze(2).expand(-1, -1, self.beam_width, -1).reshape(
                tensor.size(0), batch_size * self.beam_width, -1
            )
        else:
            raise ValueError(f"Unexpected tensor dimension: {tensor.dim()}")
```

## Length Normalization

### The Short Sequence Bias Problem

Raw beam search inherently favors shorter sequences because log probabilities are negative and accumulate:

$$\text{score}(y_1, \ldots, y_T) = \sum_{t=1}^{T} \log P(y_t | y_{<t}) < 0$$

A sequence of length 5 with average token probability 0.3 scores $5 \times \log(0.3) \approx -6.0$, while a length-10 sequence with the same average scores $-12.0$. This bias causes the decoder to prefer truncated outputs.

### Normalization Strategies

**Simple Length Normalization**:

$$\text{score}_{norm} = \frac{1}{T} \sum_{t=1}^{T} \log P(y_t | y_{<t})$$

This is the average log probability per token—equivalent to comparing geometric mean probabilities.

**Google's Length Penalty** (Wu et al., 2016):

$$\text{score}_{norm} = \frac{\text{score}}{lp(Y)}, \quad lp(Y) = \frac{(5 + |Y|)^\alpha}{(5 + 1)^\alpha}$$

The constant 5 provides smoothing for very short sequences, preventing extreme normalization. Typical values: $\alpha \in [0.6, 0.7]$.

```python
def google_length_penalty(length: int, alpha: float = 0.6) -> float:
    """
    Google's length penalty from Wu et al. (2016).
    
    The penalty grows sublinearly with length when α < 1,
    providing a balance between raw scores and per-token averages.
    """
    return ((5.0 + length) ** alpha) / ((5.0 + 1.0) ** alpha)


def normalized_score(log_prob_sum: float, length: int, alpha: float = 0.6) -> float:
    """Compute length-normalized beam score."""
    return log_prob_sum / google_length_penalty(length, alpha)
```

### Effect of α Values

| α Value | Behavior | Use Case |
|---------|----------|----------|
| 0.0 | No normalization | When brevity is desired |
| 0.5 | Moderate smoothing | General translation |
| 0.6-0.7 | Standard choice | Most seq2seq tasks |
| 1.0 | Full normalization | When length shouldn't matter |

## Coverage Penalty

### Attention Coverage Problem

In attention-based models, beam search may repeatedly attend to the same source positions while ignoring others, leading to:
- Repeated phrases in output (over-attending)
- Missing information (under-attending)

### Coverage Mechanism

Track cumulative attention and penalize under-attended positions:

```python
def coverage_penalty(
    attention_weights: List[torch.Tensor],
    beta: float = 0.2
) -> float:
    """
    Compute coverage penalty from attention history.
    
    The penalty encourages the decoder to attend to all source
    positions at least once before repeating attention.
    
    For each source position j:
        penalty_j = log(min(coverage_j, 1))
    
    This is 0 when coverage ≥ 1 (position adequately attended)
    and negative when coverage < 1 (under-attended).
    
    Args:
        attention_weights: List of attention tensors from each step
        beta: Coverage penalty weight (0 = disabled)
        
    Returns:
        Penalty term (negative, added to log probability)
    """
    if not attention_weights:
        return 0.0
    
    # Sum attention over all decoding steps: coverage_j = Σ_t α_t,j
    coverage = torch.stack(attention_weights).sum(dim=0)  # (src_len,)
    
    # Penalty for under-attended positions
    # log(min(coverage, 1)) for each position, then sum
    penalty = torch.sum(torch.log(torch.clamp(coverage, max=1.0)))
    
    return beta * penalty.item()


class BeamSearchWithCoverage:
    """
    Beam search with coverage penalty for attention models.
    
    Combines length normalization and coverage penalty for
    comprehensive scoring.
    """
    
    def __init__(
        self,
        model: nn.Module,
        beam_width: int = 5,
        max_length: int = 50,
        sos_idx: int = 1,
        eos_idx: int = 2,
        length_penalty: float = 0.6,
        coverage_penalty: float = 0.2,
        device: torch.device = None
    ):
        self.model = model
        self.beam_width = beam_width
        self.max_length = max_length
        self.sos_idx = sos_idx
        self.eos_idx = eos_idx
        self.length_penalty = length_penalty
        self.coverage_penalty_weight = coverage_penalty
        self.device = device or torch.device('cpu')
    
    def score_hypothesis(self, hypothesis: BeamHypothesis) -> float:
        """
        Compute final hypothesis score with all penalties.
        
        final_score = (log_prob / length_penalty) + coverage_penalty
        """
        base_score = hypothesis.score
        length = len(hypothesis.tokens)
        
        # Length normalization
        lp = ((5.0 + length) / 6.0) ** self.length_penalty
        normalized = base_score / lp
        
        # Coverage penalty
        if hypothesis.attention_weights:
            cp = coverage_penalty(
                hypothesis.attention_weights, 
                self.coverage_penalty_weight
            )
            normalized += cp
        
        return normalized
```

## Diverse Beam Search

### The Homogeneity Problem

Standard beam search often produces nearly identical hypotheses that differ only in minor word choices. This is problematic when:
- Multiple valid outputs exist (paraphrasing, dialogue)
- Downstream reranking benefits from variety
- The user wants alternative suggestions

### Grouped Diverse Beam Search

Vijayakumar et al. (2016) proposed dividing beams into groups and penalizing tokens already selected by previous groups:

```python
class DiverseBeamSearch:
    """
    Diverse beam search with inter-group dissimilarity penalty.
    
    Beams are divided into G groups, each with K/G beams.
    When processing group g, tokens selected by groups 0..g-1
    receive a diversity penalty, encouraging novel word choices.
    
    Args:
        model: Seq2seq model
        beam_width: Total number of beams (must be divisible by num_groups)
        num_groups: Number of diverse groups (G)
        diversity_penalty: Penalty λ for tokens selected by earlier groups
        max_length: Maximum sequence length
    """
    
    def __init__(
        self,
        model: nn.Module,
        beam_width: int = 6,
        num_groups: int = 3,
        diversity_penalty: float = 0.5,
        max_length: int = 50,
        sos_idx: int = 1,
        eos_idx: int = 2,
        device: torch.device = None
    ):
        self.model = model
        self.beam_width = beam_width
        self.num_groups = num_groups
        self.group_size = beam_width // num_groups
        self.diversity_penalty = diversity_penalty
        self.max_length = max_length
        self.sos_idx = sos_idx
        self.eos_idx = eos_idx
        self.device = device or torch.device('cpu')
        
        assert beam_width % num_groups == 0, \
            f"beam_width ({beam_width}) must be divisible by num_groups ({num_groups})"
        
    @torch.no_grad()
    def decode(
        self,
        src: torch.Tensor,
        src_lengths: Optional[torch.Tensor] = None
    ) -> List[Tuple[List[int], float]]:
        """
        Perform diverse beam search.
        
        Returns:
            List of (sequence, score) tuples from all groups,
            sorted by score descending.
        """
        self.model.eval()
        
        # Encode source
        encoder_outputs, hidden, cell = self.model.encoder(src, src_lengths)
        
        mask = None
        if hasattr(self.model, 'create_mask'):
            mask = self.model.create_mask(src)
        
        # Initialize groups with identical starting points
        groups = []
        for g in range(self.num_groups):
            group_beams = [BeamHypothesis(
                tokens=[self.sos_idx],
                score=0.0,
                hidden=hidden.clone(),
                cell=cell.clone() if cell is not None else None
            )]
            groups.append(group_beams)
        
        for step in range(self.max_length):
            # Track tokens selected at this step by earlier groups
            selected_tokens = set()
            
            for g, beams in enumerate(groups):
                if not beams:
                    continue
                
                all_candidates = []
                
                for beam in beams:
                    # Preserve completed sequences
                    if beam.tokens[-1] == self.eos_idx:
                        all_candidates.append(beam)
                        continue
                    
                    decoder_input = torch.tensor(
                        [[beam.tokens[-1]]], device=self.device
                    )
                    
                    # Decoder step
                    if hasattr(self.model.decoder, 'attention'):
                        output, new_hidden, new_cell, _ = self.model.decoder(
                            decoder_input, beam.hidden, encoder_outputs, beam.cell, mask
                        )
                    else:
                        output, new_hidden, new_cell = self.model.decoder(
                            decoder_input, beam.hidden, beam.cell
                        )
                    
                    log_probs = F.log_softmax(output, dim=-1).squeeze(0)
                    
                    # Apply diversity penalty for tokens selected by previous groups
                    penalized_log_probs = log_probs.clone()
                    for token in selected_tokens:
                        penalized_log_probs[token] -= self.diversity_penalty
                    
                    # Get top candidates (extra for filtering)
                    top_log_probs, top_indices = penalized_log_probs.topk(
                        self.group_size * 2
                    )
                    
                    for log_prob, token_idx in zip(top_log_probs, top_indices):
                        # Use original (unpenalized) score for final ranking
                        original_log_prob = log_probs[token_idx]
                        candidate = BeamHypothesis(
                            tokens=beam.tokens + [token_idx.item()],
                            score=beam.score + original_log_prob.item(),
                            hidden=new_hidden,
                            cell=new_cell
                        )
                        all_candidates.append(candidate)
                
                # Select top beams for this group
                all_candidates.sort(key=lambda h: h.score, reverse=True)
                groups[g] = all_candidates[:self.group_size]
                
                # Record selected tokens for diversity penalty
                for beam in groups[g]:
                    if len(beam.tokens) > step:
                        selected_tokens.add(beam.tokens[-1])
        
        # Collect and sort results from all groups
        results = []
        for beams in groups:
            for beam in beams:
                results.append((beam.tokens, beam.score))
        
        results.sort(key=lambda x: x[1], reverse=True)
        return results
```

## N-Best List Generation

Generate multiple hypotheses for downstream reranking or ensemble methods:

```python
def generate_nbest(
    decoder: BeamSearchDecoder,
    src: torch.Tensor,
    src_lengths: Optional[torch.Tensor] = None,
    n: int = 10,
    beam_width: int = 20
) -> List[Tuple[List[int], float]]:
    """
    Generate n-best list of diverse hypotheses.
    
    Uses a larger beam width than desired n to ensure
    sufficient diversity in the final list.
    
    Args:
        decoder: Configured beam search decoder
        src: Source sequence tensor
        src_lengths: Source lengths for masking
        n: Number of hypotheses to return
        beam_width: Beam width for search (should be ≥ 2n)
        
    Returns:
        List of (sequence, normalized_score) tuples
    """
    # Temporarily increase beam width
    original_beam_width = decoder.beam_width
    decoder.beam_width = beam_width
    
    # Collect all completed hypotheses
    # (Requires modified decode that returns all completed)
    all_hypotheses = decoder.decode_all_hypotheses(src, src_lengths)
    
    # Restore original beam width
    decoder.beam_width = original_beam_width
    
    # Deduplicate by sequence content
    seen = set()
    nbest = []
    
    for tokens, score in all_hypotheses:
        tokens_tuple = tuple(tokens)
        if tokens_tuple not in seen:
            seen.add(tokens_tuple)
            normalized = decoder.length_normalize(score, len(tokens))
            nbest.append((tokens, normalized))
            if len(nbest) >= n:
                break
    
    return nbest
```

## Practical Optimizations

### Early Stopping

Terminate search when best completed hypothesis beats all active beams:

```python
def should_stop_early(
    beams: List[BeamHypothesis],
    completed: List[BeamHypothesis],
    length_normalize_fn,
    min_length: int = 5
) -> bool:
    """
    Determine if beam search should terminate early.
    
    Stopping conditions:
    1. All active beams have ended (trivial case)
    2. Best completed sequence definitively beats best possible
       continuation of active beams
    
    Args:
        beams: Currently active (non-EOS) hypotheses
        completed: Hypotheses that reached EOS
        length_normalize_fn: Length normalization function
        min_length: Minimum sequence length before early stopping
        
    Returns:
        True if search should terminate
    """
    if not beams:
        return True
    
    if not completed:
        return False
    
    # Best normalized score among completed
    best_completed = max(
        length_normalize_fn(h.score, len(h)) for h in completed
    )
    
    # Optimistic estimate: best active beam can only get worse
    # (log probs are ≤ 0, so scores decrease)
    best_active_optimistic = max(
        length_normalize_fn(h.score, len(h)) for h in beams
    )
    
    # Stop if completed is definitively better
    return best_completed > best_active_optimistic
```

### Repetition Prevention

Prevent degenerate repetitive outputs common in neural text generation:

```python
def apply_repetition_penalty(
    log_probs: torch.Tensor,
    generated_tokens: List[int],
    penalty: float = 1.2,
    no_repeat_ngram: int = 3
) -> torch.Tensor:
    """
    Apply penalties to prevent repetitive generation.
    
    Combines two strategies:
    1. Token-level penalty: Reduce probability of previously generated tokens
    2. N-gram blocking: Set probability to -inf for tokens that would
       complete a repeated n-gram
    
    Args:
        log_probs: Log probability distribution (vocab_size,)
        generated_tokens: Previously generated token indices
        penalty: Multiplicative factor for repeated tokens (>1 = discourage)
        no_repeat_ngram: Block n-grams of this size from repeating
        
    Returns:
        Modified log probability distribution
    """
    log_probs = log_probs.clone()
    
    # Token-level repetition penalty
    for token in set(generated_tokens):
        # Divide by penalty for previously seen tokens
        # (for log probs this is equivalent to raising prob to 1/penalty)
        log_probs[token] /= penalty
    
    # N-gram blocking
    if len(generated_tokens) >= no_repeat_ngram - 1:
        # Current n-gram prefix (last n-1 tokens)
        ngram_prefix = tuple(generated_tokens[-(no_repeat_ngram-1):])
        
        # Find all positions where this prefix occurred
        for i in range(len(generated_tokens) - no_repeat_ngram + 1):
            if tuple(generated_tokens[i:i+no_repeat_ngram-1]) == ngram_prefix:
                # Block the token that would complete the repeated n-gram
                blocked_token = generated_tokens[i + no_repeat_ngram - 1]
                log_probs[blocked_token] = float('-inf')
    
    return log_probs
```

## Comparison of Decoding Strategies

| Method | Quality | Speed | Diversity | Best For |
|--------|---------|-------|-----------|----------|
| Greedy | Low | Fastest | None | Real-time, simple tasks |
| Beam Search | High | Medium | Low | Translation, summarization |
| Sampling | Variable | Fast | High | Creative generation |
| Top-k Sampling | Good | Fast | Medium | Open-ended dialogue |
| Nucleus (Top-p) | Good | Fast | Medium | Story generation |
| Diverse Beam | Good | Slow | High | Multiple suggestions |

## Hyperparameter Guidelines

| Parameter | Typical Range | Notes |
|-----------|---------------|-------|
| Beam width (K) | 4-10 | 4-5 for production, 10+ for research |
| Length penalty (α) | 0.6-1.0 | Higher = longer outputs |
| Max length | 1.5-2× source | Prevent runaway generation |
| Coverage penalty (β) | 0.0-0.4 | For attention-based models |
| Diversity penalty (λ) | 0.3-0.8 | Higher = more diversity |
| No-repeat n-gram | 2-4 | Block repeated phrases |

### Beam Width Selection Guide

| Beam Width | Quality | Latency | Use Case |
|------------|---------|---------|----------|
| 1 (greedy) | Lowest | Fastest | Real-time applications |
| 4-5 | Good | Fast | Production systems |
| 10-20 | Better | Moderate | Quality-focused offline |
| 50+ | Diminishing returns | Slow | Research/analysis |

## Summary

Beam search provides a principled approach to sequence generation that balances exploration and computational efficiency:

**Core Mechanism**: Maintain K best partial hypotheses at each step, pruning the search space from $V^T$ to $K \cdot T \cdot V$ operations.

**Length Normalization**: Essential to prevent bias toward shorter sequences. Google's length penalty with α ≈ 0.6-0.7 is the standard choice.

**Coverage Penalty**: For attention-based models, ensures complete source coverage and reduces repetition artifacts.

**Diverse Beam Search**: Encourages variety among top hypotheses through inter-group dissimilarity penalties.

**Practical Considerations**:
- Beam width trades quality for speed (4-10 is typical)
- Batched implementation is crucial for GPU efficiency
- Early stopping and repetition prevention improve practical performance
- N-best lists enable downstream reranking with external models

Beam search remains the dominant decoding strategy for tasks requiring faithful reproduction of information (translation, summarization), while stochastic methods like nucleus sampling are preferred for open-ended creative generation where diversity matters more than finding the single highest-probability sequence.
