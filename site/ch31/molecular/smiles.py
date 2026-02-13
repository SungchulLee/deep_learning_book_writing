"""
31.5.2 SMILES-Based Molecular Generation — Implementation

Character-level SMILES language models using LSTM and Transformer
architectures, with temperature sampling, fine-tuning support,
and SELFIES integration.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Optional, Tuple
from collections import Counter


# ============================================================
# SMILES Vocabulary and Tokenizer
# ============================================================

SMILES_CHARS = [
    "<pad>", "<sos>", "<eos>",
    "C", "c", "N", "n", "O", "o", "S", "s", "F", "P", "B",
    "Cl", "Br", "I",
    "(", ")", "[", "]",
    "=", "#", "-", ":", "/", "\\",
    "+",
    "1", "2", "3", "4", "5", "6", "7", "8", "9",
    "@", ".", "%",
]


class SMILESVocab:
    """
    Tokenizer for SMILES strings.

    Handles multi-character tokens (Cl, Br) and special tokens
    (<pad>, <sos>, <eos>).
    """

    def __init__(self, chars: Optional[List[str]] = None):
        self.chars = chars or SMILES_CHARS
        self.char_to_idx = {c: i for i, c in enumerate(self.chars)}
        self.idx_to_char = {i: c for i, c in enumerate(self.chars)}
        self.pad_idx = self.char_to_idx["<pad>"]
        self.sos_idx = self.char_to_idx["<sos>"]
        self.eos_idx = self.char_to_idx["<eos>"]

    def __len__(self) -> int:
        return len(self.chars)

    def tokenize(self, smiles: str) -> List[str]:
        """Split a SMILES string into tokens, handling multi-char tokens."""
        tokens = []
        i = 0
        while i < len(smiles):
            # Check for two-character tokens first
            if i + 1 < len(smiles) and smiles[i:i+2] in self.char_to_idx:
                tokens.append(smiles[i:i+2])
                i += 2
            elif smiles[i] in self.char_to_idx:
                tokens.append(smiles[i])
                i += 1
            else:
                # Unknown character — skip
                i += 1
        return tokens

    def encode(self, smiles: str, add_special: bool = True) -> List[int]:
        """Encode SMILES string to list of token indices."""
        tokens = self.tokenize(smiles)
        indices = [self.char_to_idx.get(t, self.pad_idx) for t in tokens]
        if add_special:
            indices = [self.sos_idx] + indices + [self.eos_idx]
        return indices

    def decode(self, indices: List[int], remove_special: bool = True) -> str:
        """Decode list of token indices back to SMILES string."""
        chars = []
        for idx in indices:
            token = self.idx_to_char.get(idx, "")
            if remove_special and token in ("<pad>", "<sos>", "<eos>"):
                if token == "<eos>":
                    break
                continue
            chars.append(token)
        return "".join(chars)

    def batch_encode(
        self, smiles_list: List[str], max_len: int = 128,
    ) -> torch.Tensor:
        """Encode and pad a batch of SMILES strings."""
        encoded = [self.encode(s) for s in smiles_list]
        # Truncate
        encoded = [seq[:max_len] for seq in encoded]
        # Pad
        padded = [
            seq + [self.pad_idx] * (max_len - len(seq))
            for seq in encoded
        ]
        return torch.tensor(padded, dtype=torch.long)


# ============================================================
# LSTM-Based SMILES Generator
# ============================================================

class SMILESLSTMGenerator(nn.Module):
    """
    Character-level SMILES generator using a multi-layer LSTM.

    Architecture:
        Embedding → LSTM → Linear → Softmax

    Args:
        vocab_size: Number of tokens in the SMILES vocabulary.
        embed_dim: Dimension of token embeddings.
        hidden_dim: LSTM hidden state dimension.
        num_layers: Number of stacked LSTM layers.
        dropout: Dropout rate between LSTM layers.
    """

    def __init__(
        self,
        vocab_size: int,
        embed_dim: int = 128,
        hidden_dim: int = 512,
        num_layers: int = 2,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.lstm = nn.LSTM(
            embed_dim, hidden_dim, num_layers,
            batch_first=True, dropout=dropout if num_layers > 1 else 0.0,
        )
        self.dropout = nn.Dropout(dropout)
        self.output_proj = nn.Linear(hidden_dim, vocab_size)

    def forward(
        self,
        x: torch.Tensor,
        hidden: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Forward pass.

        Args:
            x: Input token indices [batch, seq_len].
            hidden: Optional (h_0, c_0) tuple.

        Returns:
            logits [batch, seq_len, vocab_size] and updated hidden state.
        """
        emb = self.embedding(x)  # [B, T, E]
        output, hidden = self.lstm(emb, hidden)  # [B, T, H]
        output = self.dropout(output)
        logits = self.output_proj(output)  # [B, T, V]
        return logits, hidden

    def init_hidden(self, batch_size: int, device: torch.device):
        """Initialize LSTM hidden state to zeros."""
        h0 = torch.zeros(self.num_layers, batch_size, self.hidden_dim, device=device)
        c0 = torch.zeros(self.num_layers, batch_size, self.hidden_dim, device=device)
        return (h0, c0)

    @torch.no_grad()
    def generate(
        self,
        vocab: SMILESVocab,
        num_samples: int = 64,
        max_len: int = 128,
        temperature: float = 1.0,
        device: torch.device = torch.device("cpu"),
    ) -> List[str]:
        """
        Generate SMILES strings by autoregressive sampling.

        Args:
            vocab: SMILESVocab instance.
            num_samples: Number of molecules to generate.
            max_len: Maximum sequence length.
            temperature: Sampling temperature (lower = more conservative).
            device: Compute device.

        Returns:
            List of generated SMILES strings.
        """
        self.eval()
        hidden = self.init_hidden(num_samples, device)

        # Start with <sos> token
        current = torch.full(
            (num_samples, 1), vocab.sos_idx, dtype=torch.long, device=device,
        )

        sequences = [[] for _ in range(num_samples)]
        finished = [False] * num_samples

        for step in range(max_len):
            logits, hidden = self.forward(current, hidden)
            logits = logits[:, -1, :] / max(temperature, 1e-8)  # [B, V]

            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, 1)  # [B, 1]

            for i in range(num_samples):
                if not finished[i]:
                    token_idx = next_token[i, 0].item()
                    if token_idx == vocab.eos_idx:
                        finished[i] = True
                    else:
                        sequences[i].append(token_idx)

            if all(finished):
                break

            current = next_token

        return [vocab.decode(seq, remove_special=False) for seq in sequences]


# ============================================================
# Transformer-Based SMILES Generator
# ============================================================

class SMILESTransformerGenerator(nn.Module):
    """
    GPT-style Transformer for SMILES generation.

    Uses causal (left-to-right) self-attention with learned
    positional embeddings.

    Args:
        vocab_size: Number of tokens.
        d_model: Model dimension.
        nhead: Number of attention heads.
        num_layers: Number of Transformer decoder layers.
        max_len: Maximum sequence length.
        dropout: Dropout rate.
    """

    def __init__(
        self,
        vocab_size: int,
        d_model: int = 256,
        nhead: int = 8,
        num_layers: int = 4,
        max_len: int = 256,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.d_model = d_model
        self.max_len = max_len

        self.token_emb = nn.Embedding(vocab_size, d_model, padding_idx=0)
        self.pos_emb = nn.Embedding(max_len, d_model)
        self.dropout = nn.Dropout(dropout)

        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=4 * d_model,
            dropout=dropout, batch_first=True, activation="gelu",
        )
        self.transformer = nn.TransformerDecoder(decoder_layer, num_layers)
        self.output_proj = nn.Linear(d_model, vocab_size)

        # Causal mask (registered as buffer for device handling)
        self.register_buffer(
            "causal_mask",
            torch.triu(torch.ones(max_len, max_len), diagonal=1).bool(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with causal masking.

        Args:
            x: Token indices [batch, seq_len].

        Returns:
            Logits [batch, seq_len, vocab_size].
        """
        B, T = x.shape
        positions = torch.arange(T, device=x.device).unsqueeze(0)

        h = self.token_emb(x) + self.pos_emb(positions)
        h = self.dropout(h)

        # Causal mask for this sequence length
        mask = self.causal_mask[:T, :T]

        # Use as self-attention decoder (memory = dummy zeros)
        memory = torch.zeros(B, 1, self.d_model, device=x.device)
        h = self.transformer(h, memory, tgt_mask=mask)

        return self.output_proj(h)

    @torch.no_grad()
    def generate(
        self,
        vocab: SMILESVocab,
        num_samples: int = 64,
        max_len: int = 128,
        temperature: float = 1.0,
        top_k: Optional[int] = None,
        device: torch.device = torch.device("cpu"),
    ) -> List[str]:
        """
        Generate SMILES via autoregressive sampling with optional top-k.

        Args:
            vocab: Vocabulary object.
            num_samples: Batch size for parallel generation.
            max_len: Maximum generated length.
            temperature: Sampling temperature.
            top_k: If set, restrict sampling to top-k tokens.
            device: Compute device.

        Returns:
            List of generated SMILES strings.
        """
        self.eval()
        generated = torch.full(
            (num_samples, 1), vocab.sos_idx, dtype=torch.long, device=device,
        )

        finished = torch.zeros(num_samples, dtype=torch.bool, device=device)

        for _ in range(max_len - 1):
            logits = self.forward(generated)[:, -1, :]  # [B, V]
            logits = logits / max(temperature, 1e-8)

            if top_k is not None:
                # Zero out logits below top-k threshold
                topk_vals, _ = logits.topk(top_k, dim=-1)
                threshold = topk_vals[:, -1].unsqueeze(-1)
                logits[logits < threshold] = float("-inf")

            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, 1)  # [B, 1]

            # Mask finished sequences to <pad>
            next_token[finished] = vocab.pad_idx
            finished |= (next_token.squeeze(-1) == vocab.eos_idx)

            generated = torch.cat([generated, next_token], dim=1)

            if finished.all():
                break

        # Decode
        results = []
        for i in range(num_samples):
            indices = generated[i].tolist()
            results.append(vocab.decode(indices))

        return results


# ============================================================
# Training Utilities
# ============================================================

def train_smiles_model(
    model: nn.Module,
    train_smiles: List[str],
    vocab: SMILESVocab,
    epochs: int = 10,
    batch_size: int = 64,
    lr: float = 1e-3,
    max_len: int = 128,
    device: torch.device = torch.device("cpu"),
) -> List[float]:
    """
    Train a SMILES language model with teacher forcing.

    Args:
        model: SMILES generator model (LSTM or Transformer).
        train_smiles: List of training SMILES strings.
        vocab: SMILESVocab instance.
        epochs: Number of training epochs.
        batch_size: Batch size.
        lr: Learning rate.
        max_len: Maximum sequence length.
        device: Compute device.

    Returns:
        List of per-epoch average losses.
    """
    model = model.to(device)
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss(ignore_index=vocab.pad_idx)

    # Encode all training data
    all_encoded = vocab.batch_encode(train_smiles, max_len=max_len)

    losses = []
    n_batches = (len(train_smiles) + batch_size - 1) // batch_size

    for epoch in range(epochs):
        # Shuffle
        perm = torch.randperm(all_encoded.size(0))
        all_encoded = all_encoded[perm]

        epoch_loss = 0.0
        for i in range(0, len(all_encoded), batch_size):
            batch = all_encoded[i:i+batch_size].to(device)
            inputs = batch[:, :-1]   # everything except last token
            targets = batch[:, 1:]   # everything except first token

            if isinstance(model, SMILESLSTMGenerator):
                logits, _ = model(inputs)
            else:
                logits = model(inputs)

            loss = criterion(
                logits.reshape(-1, logits.size(-1)),
                targets.reshape(-1),
            )

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            epoch_loss += loss.item()

        avg_loss = epoch_loss / n_batches
        losses.append(avg_loss)
        print(f"Epoch {epoch+1}/{epochs}  Loss: {avg_loss:.4f}")

    return losses


# ============================================================
# REINFORCE Fine-Tuning for Property Optimization
# ============================================================

class REINFORCEFinetuner:
    """
    Fine-tune a SMILES generator using REINFORCE with a
    property-based reward function.

    The reward function scores generated molecules on desired
    properties (e.g., QED, logP, binding affinity).

    Args:
        model: Pre-trained SMILES generator.
        vocab: Vocabulary object.
        reward_fn: Function mapping SMILES → float reward.
        lr: Learning rate for policy gradient updates.
        baseline_momentum: Exponential moving average momentum for baseline.
    """

    def __init__(
        self,
        model: nn.Module,
        vocab: SMILESVocab,
        reward_fn,
        lr: float = 1e-4,
        baseline_momentum: float = 0.9,
    ):
        self.model = model
        self.vocab = vocab
        self.reward_fn = reward_fn
        self.optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        self.baseline = 0.0
        self.momentum = baseline_momentum

    def step(
        self,
        batch_size: int = 64,
        max_len: int = 128,
        temperature: float = 1.0,
        device: torch.device = torch.device("cpu"),
    ) -> Dict[str, float]:
        """
        One REINFORCE update step.

        Returns:
            Dict with mean_reward, loss, valid_fraction.
        """
        self.model.train()

        # ---- Generate sequences with log-probabilities ----
        generated = torch.full(
            (batch_size, 1), self.vocab.sos_idx, dtype=torch.long, device=device,
        )
        log_probs_all = []
        finished = torch.zeros(batch_size, dtype=torch.bool, device=device)

        if isinstance(self.model, SMILESLSTMGenerator):
            hidden = self.model.init_hidden(batch_size, device)
            for _ in range(max_len - 1):
                logits, hidden = self.model(generated[:, -1:], hidden)
                logits = logits[:, -1, :] / temperature
                probs = F.softmax(logits, dim=-1)
                dist = torch.distributions.Categorical(probs)
                action = dist.sample()
                log_prob = dist.log_prob(action)

                log_prob[finished] = 0.0
                finished |= (action == self.vocab.eos_idx)
                action[finished & (action != self.vocab.eos_idx)] = self.vocab.pad_idx

                log_probs_all.append(log_prob)
                generated = torch.cat(
                    [generated, action.unsqueeze(1)], dim=1,
                )
                if finished.all():
                    break
        else:
            # Transformer: must pass full sequence each step
            for step_idx in range(max_len - 1):
                logits = self.model(generated)[:, -1, :]
                logits = logits / temperature
                probs = F.softmax(logits, dim=-1)
                dist = torch.distributions.Categorical(probs)
                action = dist.sample()
                log_prob = dist.log_prob(action)

                log_prob[finished] = 0.0
                finished |= (action == self.vocab.eos_idx)
                action[finished & (action != self.vocab.eos_idx)] = self.vocab.pad_idx

                log_probs_all.append(log_prob)
                generated = torch.cat(
                    [generated, action.unsqueeze(1)], dim=1,
                )
                if finished.all():
                    break

        # Stack log probs: [batch_size, seq_len]
        log_probs = torch.stack(log_probs_all, dim=1)

        # ---- Decode and compute rewards ----
        smiles_list = []
        for i in range(batch_size):
            indices = generated[i].tolist()
            smiles_list.append(self.vocab.decode(indices))

        rewards = torch.tensor(
            [self.reward_fn(s) for s in smiles_list],
            dtype=torch.float32, device=device,
        )

        # ---- Update baseline ----
        mean_reward = rewards.mean().item()
        self.baseline = self.momentum * self.baseline + (1 - self.momentum) * mean_reward

        # ---- Policy gradient loss ----
        advantage = rewards - self.baseline  # [B]
        # Sum log probs per sequence, weight by advantage
        seq_log_probs = log_probs.sum(dim=1)  # [B]
        loss = -(advantage * seq_log_probs).mean()

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
        self.optimizer.step()

        valid_count = sum(1 for s in smiles_list if s and len(s) > 0)
        return {
            "mean_reward": mean_reward,
            "loss": loss.item(),
            "valid_fraction": valid_count / batch_size,
        }


# ============================================================
# Demo
# ============================================================

if __name__ == "__main__":
    # ---- Setup ----
    vocab = SMILESVocab()
    print(f"Vocabulary size: {len(vocab)}")

    # Test tokenization
    test_smiles = "CC(=O)Oc1ccccc1C(=O)O"
    tokens = vocab.tokenize(test_smiles)
    encoded = vocab.encode(test_smiles)
    decoded = vocab.decode(encoded)
    print(f"SMILES:   {test_smiles}")
    print(f"Tokens:   {tokens}")
    print(f"Encoded:  {encoded}")
    print(f"Decoded:  {decoded}")

    # ---- LSTM Generator ----
    print("\n=== LSTM Generator ===")
    lstm_gen = SMILESLSTMGenerator(vocab_size=len(vocab))
    print(f"Parameters: {sum(p.numel() for p in lstm_gen.parameters()):,}")

    # Generate before training (random)
    samples = lstm_gen.generate(vocab, num_samples=5, temperature=1.0)
    print(f"Random samples: {samples[:3]}")

    # ---- Transformer Generator ----
    print("\n=== Transformer Generator ===")
    tf_gen = SMILESTransformerGenerator(vocab_size=len(vocab))
    print(f"Parameters: {sum(p.numel() for p in tf_gen.parameters()):,}")

    samples = tf_gen.generate(vocab, num_samples=5, temperature=1.0)
    print(f"Random samples: {samples[:3]}")

    # ---- Quick training demo ----
    print("\n=== Training Demo ===")
    toy_smiles = [
        "CCO", "CCCO", "c1ccccc1", "CC(=O)O", "CCN",
        "c1ccc(O)cc1", "CC=O", "C(=O)O", "CCCl", "CC(C)C",
    ] * 10  # repeat for more data

    losses = train_smiles_model(
        lstm_gen, toy_smiles, vocab, epochs=3, batch_size=16,
    )
    print(f"Final loss: {losses[-1]:.4f}")

    # Generate after minimal training
    samples = lstm_gen.generate(vocab, num_samples=10, temperature=0.8)
    print(f"Trained samples: {samples[:5]}")

    # ---- REINFORCE demo ----
    print("\n=== REINFORCE Fine-Tuning Demo ===")

    def simple_reward(smiles: str) -> float:
        """Reward: longer valid-looking SMILES get higher reward."""
        if not smiles or len(smiles) < 2:
            return 0.0
        # Bonus for containing rings or branches
        score = min(len(smiles) / 20.0, 1.0)
        if "1" in smiles:  # ring closure
            score += 0.2
        if "(" in smiles:  # branch
            score += 0.1
        return min(score, 1.0)

    finetuner = REINFORCEFinetuner(
        model=lstm_gen, vocab=vocab, reward_fn=simple_reward, lr=1e-4,
    )

    for step in range(3):
        info = finetuner.step(batch_size=32, temperature=0.9)
        print(f"Step {step+1}: reward={info['mean_reward']:.3f}, "
              f"loss={info['loss']:.4f}")
