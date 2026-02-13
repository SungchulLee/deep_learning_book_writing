"""
Tutorial 04: Feedforward Neural Language Model
===============================================

This tutorial introduces neural language models using feedforward networks.
We move from discrete n-gram counting to continuous word representations.

Learning Objectives:
--------------------
1. Understand distributed representations for words
2. Build feedforward neural language models
3. Train models using backpropagation
4. Compare neural LMs with n-gram models
5. Understand the fixed context window limitation

Mathematical Background:
------------------------

Neural Language Model Architecture (Bengio et al. 2003):
--------------------------------------------------------

Input: Context words w_{t-n+1}, ..., w_{t-1}
Output: Probability distribution over next word w_t

Architecture:
1. **Embedding Layer**: 
   - Convert each word index to embedding vector
   - C(w) ∈ ℝ^d where d is embedding dimension
   - Context becomes: [C(w_{t-n+1}), ..., C(w_{t-1})] ∈ ℝ^{(n-1)×d}

2. **Hidden Layer**:
   - Concatenate embeddings: x = concat(C(w_{t-n+1}), ..., C(w_{t-1}))
   - Hidden state: h = tanh(Wx + b)
   - W ∈ ℝ^{h×(n-1)d}, h is hidden size

3. **Output Layer**:
   - Scores: s = Uh + c
   - Probabilities: P(w_t | context) = softmax(s)
   - U ∈ ℝ^{V×h}, V is vocabulary size

Loss Function:
--------------
Negative log-likelihood (cross-entropy):
L = -1/N ∑ log P(w_t | w_{t-n+1}, ..., w_{t-1})

Optimization:
-------------
- Stochastic Gradient Descent (SGD) or Adam
- Backpropagation through time (BPTT)
- Gradient clipping to prevent explosion

Advantages over N-grams:
-------------------------
1. Learns distributed representations (semantically similar words have similar embeddings)
2. Generalizes to unseen contexts through learned representations
3. No explicit smoothing needed
4. Can capture semantic similarities

Limitations:
------------
1. Fixed context window (cannot handle arbitrary-length contexts)
2. Computational cost of softmax over large vocabulary
3. No parameter sharing across time (unlike RNNs)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from typing import List, Dict, Tuple
import numpy as np
import random


class Vocabulary:
    """
    Vocabulary management for neural language models.
    
    Converts between words and indices, handles special tokens.
    
    Attributes:
        word2idx (dict): Mapping from word to index
        idx2word (dict): Mapping from index to word
        special_tokens (list): List of special tokens
    """
    
    def __init__(self):
        """Initialize vocabulary with special tokens."""
        self.word2idx = {}
        self.idx2word = {}
        
        # Special tokens
        self.PAD_TOKEN = "<pad>"
        self.UNK_TOKEN = "<unk>"
        self.START_TOKEN = "<s>"
        self.END_TOKEN = "</s>"
        
        self.special_tokens = [
            self.PAD_TOKEN,
            self.UNK_TOKEN,
            self.START_TOKEN,
            self.END_TOKEN
        ]
        
        # Add special tokens first
        for token in self.special_tokens:
            self._add_word(token)
    
    def _add_word(self, word: str) -> int:
        """Add word to vocabulary if not present."""
        if word not in self.word2idx:
            idx = len(self.word2idx)
            self.word2idx[word] = idx
            self.idx2word[idx] = word
        return self.word2idx[word]
    
    def build_from_corpus(self, corpus: List[str], min_freq: int = 1) -> None:
        """
        Build vocabulary from corpus.
        
        Args:
            corpus: List of sentences
            min_freq: Minimum frequency for word inclusion
        """
        # Count word frequencies
        word_counts = {}
        for sentence in corpus:
            words = sentence.lower().split()
            for word in words:
                word_counts[word] = word_counts.get(word, 0) + 1
        
        # Add words that meet minimum frequency
        for word, count in word_counts.items():
            if count >= min_freq:
                self._add_word(word)
        
        print(f"Vocabulary built: {len(self.word2idx)} words (min_freq={min_freq})")
    
    def word_to_idx(self, word: str) -> int:
        """Convert word to index (returns UNK index if not in vocab)."""
        return self.word2idx.get(word, self.word2idx[self.UNK_TOKEN])
    
    def idx_to_word(self, idx: int) -> str:
        """Convert index to word."""
        return self.idx2word[idx]
    
    def __len__(self) -> int:
        """Return vocabulary size."""
        return len(self.word2idx)


class LanguageModelDataset(Dataset):
    """
    Dataset for neural language modeling.
    
    Creates context-target pairs for training.
    For context_size=3: [..., w_{t-3}, w_{t-2}, w_{t-1}] → w_t
    
    Attributes:
        contexts (list): List of context word indices
        targets (list): List of target word indices
        vocab (Vocabulary): Vocabulary object
    """
    
    def __init__(self, corpus: List[str], vocab: Vocabulary, context_size: int):
        """
        Initialize dataset.
        
        Args:
            corpus: List of sentences
            vocab: Vocabulary object
            context_size: Number of context words to use
        """
        self.vocab = vocab
        self.context_size = context_size
        self.contexts = []
        self.targets = []
        
        # Create context-target pairs
        for sentence in corpus:
            words = sentence.lower().split()
            
            # Add boundary tokens
            words = [vocab.START_TOKEN] * context_size + words + [vocab.END_TOKEN]
            
            # Convert to indices
            indices = [vocab.word_to_idx(word) for word in words]
            
            # Extract context-target pairs
            # For each position t, context is [w_{t-n}, ..., w_{t-1}], target is w_t
            for i in range(context_size, len(indices)):
                context = indices[i - context_size:i]
                target = indices[i]
                
                self.contexts.append(context)
                self.targets.append(target)
    
    def __len__(self) -> int:
        """Return number of examples."""
        return len(self.contexts)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get example at index.
        
        Returns:
            Tuple of (context tensor, target index)
        """
        context = torch.tensor(self.contexts[idx], dtype=torch.long)
        target = torch.tensor(self.targets[idx], dtype=torch.long)
        return context, target


class FeedforwardLanguageModel(nn.Module):
    """
    Feedforward neural language model (Bengio et al. 2003).
    
    Architecture:
    1. Embedding layer: words → dense vectors
    2. Hidden layer: concatenated embeddings → hidden representation
    3. Output layer: hidden → probability over vocabulary
    
    Args:
        vocab_size (int): Size of vocabulary
        embedding_dim (int): Dimension of word embeddings
        context_size (int): Number of context words
        hidden_dim (int): Dimension of hidden layer
    """
    
    def __init__(self, vocab_size: int, embedding_dim: int,
                 context_size: int, hidden_dim: int):
        """Initialize feedforward language model."""
        super(FeedforwardLanguageModel, self).__init__()
        
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.context_size = context_size
        self.hidden_dim = hidden_dim
        
        # Embedding layer: vocab_size → embedding_dim
        self.embeddings = nn.Embedding(vocab_size, embedding_dim)
        
        # Hidden layer: (context_size * embedding_dim) → hidden_dim
        self.fc1 = nn.Linear(context_size * embedding_dim, hidden_dim)
        
        # Output layer: hidden_dim → vocab_size
        self.fc2 = nn.Linear(hidden_dim, vocab_size)
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights using Xavier initialization."""
        nn.init.xavier_uniform_(self.embeddings.weight)
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.xavier_uniform_(self.fc2.weight)
        nn.init.zeros_(self.fc1.bias)
        nn.init.zeros_(self.fc2.bias)
    
    def forward(self, context: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            context: (batch_size, context_size) tensor of word indices
            
        Returns:
            (batch_size, vocab_size) tensor of logits
        """
        # Get embeddings: (batch_size, context_size, embedding_dim)
        embeds = self.embeddings(context)
        
        # Flatten embeddings: (batch_size, context_size * embedding_dim)
        embeds_flat = embeds.view(embeds.size(0), -1)
        
        # Hidden layer with tanh activation
        hidden = torch.tanh(self.fc1(embeds_flat))
        
        # Output layer (logits)
        logits = self.fc2(hidden)
        
        return logits
    
    def get_next_word_probs(self, context: List[int]) -> torch.Tensor:
        """
        Get probability distribution over next words given context.
        
        Args:
            context: List of context word indices
            
        Returns:
            Probability distribution (vocab_size,)
        """
        # Convert to tensor and add batch dimension
        context_tensor = torch.tensor([context], dtype=torch.long)
        
        # Forward pass
        with torch.no_grad():
            logits = self.forward(context_tensor)
            probs = F.softmax(logits, dim=1)
        
        return probs.squeeze(0)


def train_feedforward_lm(train_corpus: List[str],
                         val_corpus: List[str] = None,
                         embedding_dim: int = 64,
                         hidden_dim: int = 128,
                         context_size: int = 3,
                         min_freq: int = 1,
                         batch_size: int = 32,
                         epochs: int = 10,
                         learning_rate: float = 0.001) -> Tuple:
    """
    Train feedforward language model.
    
    Args:
        train_corpus: List of training sentences
        val_corpus: List of validation sentences (optional)
        embedding_dim: Dimension of word embeddings
        hidden_dim: Dimension of hidden layer
        context_size: Number of context words
        min_freq: Minimum word frequency for vocabulary
        batch_size: Batch size for training
        epochs: Number of training epochs
        learning_rate: Learning rate for optimizer
        
    Returns:
        Tuple of (model, vocab, train_losses, val_perplexities)
    """
    print("=" * 70)
    print("Training Feedforward Neural Language Model")
    print("=" * 70)
    print(f"Embedding dim: {embedding_dim}")
    print(f"Hidden dim: {hidden_dim}")
    print(f"Context size: {context_size}")
    print(f"Batch size: {batch_size}")
    print(f"Learning rate: {learning_rate}")
    print()
    
    # Build vocabulary
    vocab = Vocabulary()
    vocab.build_from_corpus(train_corpus, min_freq=min_freq)
    
    # Create datasets
    train_dataset = LanguageModelDataset(train_corpus, vocab, context_size)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    
    if val_corpus:
        val_dataset = LanguageModelDataset(val_corpus, vocab, context_size)
        val_loader = DataLoader(val_dataset, batch_size=batch_size)
    
    # Initialize model
    model = FeedforwardLanguageModel(
        vocab_size=len(vocab),
        embedding_dim=embedding_dim,
        context_size=context_size,
        hidden_dim=hidden_dim
    )
    
    # Loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    # Training loop
    train_losses = []
    val_perplexities = []
    
    for epoch in range(epochs):
        # Training phase
        model.train()
        epoch_loss = 0.0
        num_batches = 0
        
        for contexts, targets in train_loader:
            # Forward pass
            logits = model(contexts)
            loss = criterion(logits, targets)
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            
            # Gradient clipping (prevent explosion)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            
            optimizer.step()
            
            epoch_loss += loss.item()
            num_batches += 1
        
        avg_loss = epoch_loss / num_batches
        train_losses.append(avg_loss)
        
        # Validation phase
        if val_corpus:
            model.eval()
            val_loss = 0.0
            val_batches = 0
            
            with torch.no_grad():
                for contexts, targets in val_loader:
                    logits = model(contexts)
                    loss = criterion(logits, targets)
                    val_loss += loss.item()
                    val_batches += 1
            
            avg_val_loss = val_loss / val_batches
            # Perplexity = exp(cross_entropy)
            val_ppl = np.exp(avg_val_loss)
            val_perplexities.append(val_ppl)
            
            print(f"Epoch {epoch+1}/{epochs} - "
                  f"Train Loss: {avg_loss:.4f} - "
                  f"Val Perplexity: {val_ppl:.2f}")
        else:
            print(f"Epoch {epoch+1}/{epochs} - Train Loss: {avg_loss:.4f}")
    
    print("\nTraining complete!")
    return model, vocab, train_losses, val_perplexities


def generate_text(model: FeedforwardLanguageModel,
                 vocab: Vocabulary,
                 context_size: int,
                 max_length: int = 20,
                 temperature: float = 1.0) -> str:
    """
    Generate text using trained feedforward language model.
    
    Args:
        model: Trained model
        vocab: Vocabulary
        context_size: Context window size
        max_length: Maximum words to generate
        temperature: Sampling temperature
        
    Returns:
        Generated text
    """
    model.eval()
    
    # Start with context of START tokens
    context = [vocab.word_to_idx(vocab.START_TOKEN)] * context_size
    generated = []
    
    for _ in range(max_length):
        # Get next word probabilities
        probs = model.get_next_word_probs(context)
        
        # Apply temperature
        if temperature != 1.0:
            probs = torch.pow(probs, 1.0 / temperature)
            probs = probs / probs.sum()
        
        # Sample next word
        next_word_idx = torch.multinomial(probs, 1).item()
        next_word = vocab.idx_to_word(next_word_idx)
        
        # Stop if end token
        if next_word == vocab.END_TOKEN:
            break
        
        if next_word not in vocab.special_tokens:
            generated.append(next_word)
        
        # Update context (shift window)
        context = context[1:] + [next_word_idx]
    
    return ' '.join(generated)


def demonstrate_feedforward_lm():
    """
    Demonstrate feedforward neural language model.
    """
    print("=" * 70)
    print("Feedforward Neural Language Model Demo")
    print("=" * 70)
    
    # Sample corpus
    train_corpus = [
        "the cat sat on the mat",
        "the dog sat on the log",
        "the cat and the dog are friends",
        "cats and dogs play together",
        "the quick brown fox jumps over the lazy dog",
        "a cat catches a mouse",
        "dogs like to run and play",
        "the cat sleeps all day",
        "small dogs and big cats live together",
        "the brown dog runs fast"
    ] * 10  # Repeat for more training data
    
    val_corpus = [
        "the cat runs",
        "dogs and cats play",
        "a quick fox jumps"
    ]
    
    # Train model
    model, vocab, train_losses, val_ppls = train_feedforward_lm(
        train_corpus=train_corpus,
        val_corpus=val_corpus,
        embedding_dim=32,
        hidden_dim=64,
        context_size=3,
        batch_size=16,
        epochs=20,
        learning_rate=0.001
    )
    
    # Generate text samples
    print("\n" + "=" * 70)
    print("Generated Text Samples")
    print("=" * 70)
    
    for i in range(5):
        text = generate_text(model, vocab, context_size=3,
                            max_length=15, temperature=1.0)
        print(f"{i+1}. {text}")
    
    # Test different temperatures
    print("\n" + "=" * 70)
    print("Effect of Temperature")
    print("=" * 70)
    
    for temp in [0.5, 1.0, 1.5]:
        print(f"\nTemperature = {temp}:")
        for i in range(3):
            text = generate_text(model, vocab, context_size=3,
                                max_length=12, temperature=temp)
            print(f"  {text}")
    
    print("\n" + "=" * 70)
    print("KEY INSIGHTS")
    print("=" * 70)
    print("""
1. Word Embeddings:
   - Neural models learn distributed representations
   - Similar words have similar embeddings
   - Captures semantic relationships

2. Generalization:
   - Can handle unseen contexts better than n-grams
   - Learns patterns in the embedding space
   - No explicit smoothing needed

3. Fixed Context:
   - Limited to fixed-size context window
   - Cannot handle arbitrary-length dependencies
   - This limitation addressed by RNNs (next tutorial)

4. Training Dynamics:
   - Requires more data than n-gram models
   - Longer training time
   - Needs hyperparameter tuning

5. Comparison with N-grams:
   - Neural: Better generalization, continuous representations
   - N-gram: Faster, simpler, interpretable
   - Trade-off: flexibility vs. simplicity
    """)


if __name__ == "__main__":
    # Set random seeds for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    random.seed(42)
    
    demonstrate_feedforward_lm()
    
    print("\n" + "=" * 70)
    print("EXERCISES")
    print("=" * 70)
    print("""
1. Experiment with different embedding dimensions (16, 32, 64, 128)
2. Try different context sizes (2, 3, 5, 7)
3. Implement dropout for regularization
4. Add a second hidden layer
5. Visualize word embeddings using PCA or t-SNE
6. Compare perplexity with n-gram baseline
7. Implement scheduled learning rate decay
8. Try different optimizers (SGD, RMSprop, Adam)
    """)
