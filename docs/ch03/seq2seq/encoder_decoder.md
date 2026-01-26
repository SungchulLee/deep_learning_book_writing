# Encoder-Decoder Architecture

The encoder-decoder architecture represents a fundamental paradigm in sequence-to-sequence learning, enabling models to transform variable-length input sequences into variable-length output sequences. Introduced by Sutskever et al. (2014), this architecture revolutionized machine translation and enabled diverse applications from text summarization to conversational AI.

## Conceptual Foundation

The encoder-decoder framework addresses a fundamental challenge in sequence modeling: how to map an input sequence of arbitrary length to an output sequence of potentially different length. Traditional fixed-size neural networks cannot naturally handle this variability, making the encoder-decoder design essential for sequence transduction tasks.

```
Input: "How are you?"
         ↓
    [Encoder]
         ↓
    Context Vector
         ↓
    [Decoder]
         ↓
Output: "Comment allez-vous?"
```

The architecture consists of two distinct components that work in concert:

**Encoder**: Processes the entire input sequence and compresses it into a fixed-dimensional representation called the context vector or thought vector. This representation captures the semantic and syntactic information of the input.

**Decoder**: Takes the context vector and generates the output sequence one token at a time, conditioning each prediction on both the context and previously generated tokens.

## Mathematical Formulation

### Encoder Dynamics

Given an input sequence $\mathbf{x} = (x_1, x_2, \ldots, x_T)$ where $T$ is the source sequence length, the encoder processes this sequence through an embedding layer followed by a recurrent network:

$$\mathbf{e}_t = \text{Embed}(x_t) \in \mathbb{R}^{d_e}$$

where $d_e$ is the embedding dimension. The recurrent computation proceeds as:

$$\mathbf{h}_t^{enc} = f_{enc}(\mathbf{e}_t, \mathbf{h}_{t-1}^{enc})$$

For an LSTM encoder, this expands to the complete gate equations:

$$\mathbf{i}_t = \sigma(\mathbf{W}_{xi}\mathbf{e}_t + \mathbf{W}_{hi}\mathbf{h}_{t-1} + \mathbf{b}_i)$$

$$\mathbf{f}_t = \sigma(\mathbf{W}_{xf}\mathbf{e}_t + \mathbf{W}_{hf}\mathbf{h}_{t-1} + \mathbf{b}_f)$$

$$\mathbf{o}_t = \sigma(\mathbf{W}_{xo}\mathbf{e}_t + \mathbf{W}_{ho}\mathbf{h}_{t-1} + \mathbf{b}_o)$$

$$\tilde{\mathbf{c}}_t = \tanh(\mathbf{W}_{xc}\mathbf{e}_t + \mathbf{W}_{hc}\mathbf{h}_{t-1} + \mathbf{b}_c)$$

$$\mathbf{c}_t = \mathbf{f}_t \odot \mathbf{c}_{t-1} + \mathbf{i}_t \odot \tilde{\mathbf{c}}_t$$

$$\mathbf{h}_t = \mathbf{o}_t \odot \tanh(\mathbf{c}_t)$$

The final hidden state $\mathbf{h}_T^{enc}$ serves as the context vector $\mathbf{c}$ that summarizes the entire input sequence:

$$c = h_T^{\text{enc}} \quad \text{or} \quad c = f(h_1^{\text{enc}}, \ldots, h_T^{\text{enc}})$$

### Bidirectional Encoding

For enhanced context capture, bidirectional encoders process the sequence in both directions:

$$\overrightarrow{\mathbf{h}}_t = f_{enc}(\mathbf{e}_t, \overrightarrow{\mathbf{h}}_{t-1})$$

$$\overleftarrow{\mathbf{h}}_t = f_{enc}(\mathbf{e}_t, \overleftarrow{\mathbf{h}}_{t+1})$$

The bidirectional hidden states are combined through concatenation:

$$\mathbf{h}_t = [\overrightarrow{\mathbf{h}}_t; \overleftarrow{\mathbf{h}}_t] \in \mathbb{R}^{2d_h}$$

This provides each position with context from both past and future tokens, which is particularly valuable for tasks where the meaning of a word depends on surrounding context in both directions.

### Decoder Dynamics

The decoder generates the target sequence $\mathbf{y} = (y_1, y_2, \ldots, y_{T'})$ autoregressively. At each timestep $t$, it predicts the next token based on the context vector and all previously generated tokens:

$$P(\mathbf{y}|\mathbf{x}) = \prod_{t=1}^{T'} P(y_t | y_{<t}, \mathbf{c})$$

The decoder hidden state update follows:

$$\mathbf{s}_t = f_{dec}([\mathbf{e}_{y_{t-1}}; \mathbf{c}], \mathbf{s}_{t-1})$$

where $\mathbf{s}_0 = \mathbf{h}_T^{enc}$ (initialized with the encoder's final state) and $y_0 = \langle\text{sos}\rangle$ (start-of-sequence token).

The output probability distribution is computed through a projection layer:

$$P(y_t | y_{<t}, \mathbf{c}) = \text{softmax}(\mathbf{W}_o \mathbf{s}_t + \mathbf{b}_o)$$

## PyTorch Implementation

### Basic Encoder

```python
import torch
import torch.nn as nn
import torch.nn.functional as F


class BasicEncoder(nn.Module):
    """
    RNN-based encoder for sequence-to-sequence models.
    
    Processes input sequences through an embedding layer and recurrent network,
    producing hidden state representations for each position.
    
    Args:
        input_size: Size of input vocabulary
        embedding_dim: Dimension of word embeddings
        hidden_size: Size of hidden state
        num_layers: Number of recurrent layers (default: 1)
        dropout: Dropout probability (default: 0.1)
        bidirectional: Whether to use bidirectional RNN (default: False)
        rnn_type: Type of RNN cell - 'LSTM' or 'GRU' (default: 'LSTM')
    """
    
    def __init__(
        self,
        input_size: int,
        embedding_dim: int,
        hidden_size: int,
        num_layers: int = 1,
        dropout: float = 0.1,
        bidirectional: bool = False,
        rnn_type: str = 'LSTM'
    ):
        super().__init__()
        
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.rnn_type = rnn_type
        
        # Embedding layer transforms token indices to dense vectors
        self.embedding = nn.Embedding(input_size, embedding_dim)
        self.dropout = nn.Dropout(dropout)
        
        # Select RNN architecture
        rnn_cls = nn.LSTM if rnn_type == 'LSTM' else nn.GRU
        self.rnn = rnn_cls(
            input_size=embedding_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=bidirectional,
            batch_first=True
        )
        
    def forward(
        self,
        input_seq: torch.Tensor,
        input_lengths: torch.Tensor = None
    ) -> tuple:
        """
        Encode input sequence.
        
        Args:
            input_seq: Input token indices (batch_size, seq_len)
            input_lengths: Actual lengths of sequences for packing (optional)
            
        Returns:
            outputs: All hidden states (batch_size, seq_len, hidden_size * num_directions)
            hidden: Final hidden state (num_layers * num_directions, batch_size, hidden_size)
            cell: Final cell state for LSTM (num_layers * num_directions, batch_size, hidden_size)
        """
        # Embed input tokens
        embedded = self.dropout(self.embedding(input_seq))
        
        # Pack sequences for efficient computation with variable lengths
        if input_lengths is not None:
            embedded = nn.utils.rnn.pack_padded_sequence(
                embedded, 
                input_lengths.cpu(), 
                batch_first=True, 
                enforce_sorted=False
            )
        
        # Process through RNN
        if self.rnn_type == 'LSTM':
            outputs, (hidden, cell) = self.rnn(embedded)
        else:
            outputs, hidden = self.rnn(embedded)
            cell = None
        
        # Unpack if packed
        if input_lengths is not None:
            outputs, _ = nn.utils.rnn.pad_packed_sequence(
                outputs, batch_first=True
            )
        
        # Combine bidirectional states if necessary
        if self.bidirectional:
            hidden = self._combine_bidirectional(hidden)
            if cell is not None:
                cell = self._combine_bidirectional(cell)
        
        return outputs, hidden, cell
    
    def _combine_bidirectional(self, hidden: torch.Tensor) -> torch.Tensor:
        """
        Combine forward and backward hidden states for bidirectional RNN.
        
        Transforms shape from (num_layers * 2, batch, hidden) 
        to (num_layers, batch, hidden * 2)
        """
        batch_size = hidden.size(1)
        hidden = hidden.view(self.num_layers, 2, batch_size, self.hidden_size)
        hidden = torch.cat([hidden[:, 0], hidden[:, 1]], dim=2)
        return hidden
```

### Basic Decoder

```python
class BasicDecoder(nn.Module):
    """
    RNN-based decoder for sequence-to-sequence models.
    
    Generates output sequence one token at a time, conditioning on the
    encoder's context and previously generated tokens.
    
    Args:
        output_size: Size of output vocabulary
        embedding_dim: Dimension of word embeddings
        hidden_size: Size of hidden state
        num_layers: Number of recurrent layers (default: 1)
        dropout: Dropout probability (default: 0.1)
        rnn_type: Type of RNN cell - 'LSTM' or 'GRU' (default: 'LSTM')
    """
    
    def __init__(
        self,
        output_size: int,
        embedding_dim: int,
        hidden_size: int,
        num_layers: int = 1,
        dropout: float = 0.1,
        rnn_type: str = 'LSTM'
    ):
        super().__init__()
        
        self.output_size = output_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.rnn_type = rnn_type
        
        self.embedding = nn.Embedding(output_size, embedding_dim)
        self.dropout = nn.Dropout(dropout)
        
        rnn_cls = nn.LSTM if rnn_type == 'LSTM' else nn.GRU
        self.rnn = rnn_cls(
            input_size=embedding_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            batch_first=True
        )
        
        # Output projection to vocabulary
        self.fc_out = nn.Linear(hidden_size, output_size)
        
    def forward(
        self,
        input_token: torch.Tensor,
        hidden: torch.Tensor,
        cell: torch.Tensor = None
    ) -> tuple:
        """
        Decode one timestep.
        
        Args:
            input_token: Previous token (batch_size, 1)
            hidden: Previous hidden state
            cell: Previous cell state (LSTM only)
            
        Returns:
            output: Token logits (batch_size, output_size)
            hidden: Updated hidden state
            cell: Updated cell state (LSTM only)
        """
        # Embed input token
        embedded = self.dropout(self.embedding(input_token))
        
        # Single RNN step
        if self.rnn_type == 'LSTM':
            rnn_output, (hidden, cell) = self.rnn(embedded, (hidden, cell))
        else:
            rnn_output, hidden = self.rnn(embedded, hidden)
            cell = None
        
        # Project to vocabulary space
        output = self.fc_out(rnn_output.squeeze(1))
        
        return output, hidden, cell
```

### Complete Seq2Seq Model

```python
class Seq2Seq(nn.Module):
    """
    Complete sequence-to-sequence model combining encoder and decoder.
    
    Implements the full encoding-decoding pipeline with support for
    teacher forcing during training.
    
    Args:
        encoder: Encoder module
        decoder: Decoder module  
        device: Computation device
        sos_idx: Start-of-sequence token index (default: 1)
        eos_idx: End-of-sequence token index (default: 2)
    """
    
    def __init__(
        self,
        encoder: nn.Module,
        decoder: nn.Module,
        device: torch.device,
        sos_idx: int = 1,
        eos_idx: int = 2
    ):
        super().__init__()
        
        self.encoder = encoder
        self.decoder = decoder
        self.device = device
        self.sos_idx = sos_idx
        self.eos_idx = eos_idx
        
    def forward(
        self,
        src: torch.Tensor,
        trg: torch.Tensor,
        teacher_forcing_ratio: float = 0.5,
        src_lengths: torch.Tensor = None
    ) -> torch.Tensor:
        """
        Forward pass through encoder-decoder.
        
        Args:
            src: Source sequence (batch_size, src_len)
            trg: Target sequence (batch_size, trg_len)
            teacher_forcing_ratio: Probability of using ground truth as input
            src_lengths: Actual source lengths for packing
            
        Returns:
            outputs: Predicted logits (batch_size, trg_len, vocab_size)
        """
        batch_size = src.size(0)
        trg_len = trg.size(1)
        vocab_size = self.decoder.output_size
        
        # Storage for decoder outputs
        outputs = torch.zeros(batch_size, trg_len, vocab_size).to(self.device)
        
        # Encode source sequence
        encoder_outputs, hidden, cell = self.encoder(src, src_lengths)
        
        # First decoder input is <sos> token
        decoder_input = trg[:, 0].unsqueeze(1)
        
        # Decode sequence one token at a time
        for t in range(1, trg_len):
            output, hidden, cell = self.decoder(decoder_input, hidden, cell)
            outputs[:, t] = output
            
            # Teacher forcing: use ground truth or model prediction
            use_teacher_forcing = torch.rand(1).item() < teacher_forcing_ratio
            
            if use_teacher_forcing:
                decoder_input = trg[:, t].unsqueeze(1)
            else:
                decoder_input = output.argmax(dim=1).unsqueeze(1)
        
        return outputs
    
    def generate(
        self,
        src: torch.Tensor,
        max_len: int = 50,
        src_lengths: torch.Tensor = None
    ) -> torch.Tensor:
        """
        Generate output sequence using greedy decoding.
        
        Args:
            src: Source sequence (batch_size, src_len)
            max_len: Maximum generation length
            src_lengths: Actual source lengths
            
        Returns:
            generated: Generated token indices (batch_size, generated_len)
        """
        self.eval()
        batch_size = src.size(0)
        
        with torch.no_grad():
            # Encode
            _, hidden, cell = self.encoder(src, src_lengths)
            
            # Initialize with <sos>
            decoder_input = torch.full(
                (batch_size, 1), self.sos_idx, dtype=torch.long, device=self.device
            )
            
            generated = [decoder_input]
            finished = torch.zeros(batch_size, dtype=torch.bool, device=self.device)
            
            for _ in range(max_len):
                output, hidden, cell = self.decoder(decoder_input, hidden, cell)
                predicted = output.argmax(dim=1).unsqueeze(1)
                
                generated.append(predicted)
                finished |= (predicted.squeeze(1) == self.eos_idx)
                
                if finished.all():
                    break
                    
                decoder_input = predicted
            
            return torch.cat(generated, dim=1)
```

## Teacher Forcing and Scheduled Sampling

During training, the decoder can either use:
1. **Ground truth**: Feed actual target tokens (teacher forcing)
2. **Predictions**: Feed model's own predictions

**Teacher forcing** accelerates training by providing perfect context at each step, but creates **exposure bias**—the model never learns to recover from its own mistakes during training, leading to error accumulation at inference time.

### Scheduled Sampling

Gradually decrease teacher forcing over training to mitigate exposure bias:

```python
def train_with_scheduled_sampling(model, src, tgt, epoch, max_epochs):
    """
    Gradually decrease teacher forcing over training.
    
    Linear decay from 1.0 to 0.0 helps the model learn to recover
    from its own errors while still benefiting from stable early training.
    """
    # Linear decay from 1.0 to 0.0
    teacher_forcing_ratio = 1.0 - (epoch / max_epochs)
    
    outputs = model(src, tgt, teacher_forcing_ratio)
    return outputs
```

## Inference Strategies

### Greedy Decoding

The simplest inference approach selects the most probable token at each step:

```python
def greedy_decode(model, src, max_length, sos_idx, eos_idx):
    """
    Generate sequence using greedy decoding.
    
    Fast but may miss globally optimal sequences due to local decisions.
    """
    model.eval()
    
    with torch.no_grad():
        # Encode source
        _, hidden, cell = model.encoder(src)
        
        # Start with <SOS>
        decoder_input = torch.tensor([[sos_idx]], device=src.device)
        
        outputs = []
        for _ in range(max_length):
            output, hidden, cell = model.decoder(decoder_input, hidden, cell)
            
            # Greedy selection
            predicted = output.argmax(dim=-1)
            outputs.append(predicted.item())
            
            # Stop at <EOS>
            if predicted.item() == eos_idx:
                break
            
            decoder_input = predicted.unsqueeze(0)
    
    return outputs
```

### Beam Search

Beam search maintains multiple hypotheses for better quality:

```python
def beam_search(model, src, max_length, sos_idx, eos_idx, beam_width=5):
    """
    Generate sequence using beam search.
    
    Maintains beam_width hypotheses at each step, exploring multiple
    paths through the search space for higher quality outputs.
    
    Args:
        model: Seq2Seq model
        src: Source sequence (1, src_len)
        max_length: Maximum generation length
        sos_idx: Start-of-sequence token index
        eos_idx: End-of-sequence token index
        beam_width: Number of hypotheses to maintain
        
    Returns:
        Best sequence as list of token indices
    """
    model.eval()
    
    with torch.no_grad():
        # Encode source
        _, hidden, cell = model.encoder(src)
        
        # Initialize beam
        # Each hypothesis: (sequence, score, hidden_state, cell_state)
        beams = [([sos_idx], 0.0, hidden, cell)]
        completed = []
        
        for _ in range(max_length):
            candidates = []
            
            for seq, score, h, c in beams:
                if seq[-1] == eos_idx:
                    completed.append((seq, score))
                    continue
                
                # Decode one step
                decoder_input = torch.tensor([[seq[-1]]], device=src.device)
                output, new_h, new_c = model.decoder(decoder_input, h, c)
                
                # Get log probabilities
                log_probs = F.log_softmax(output, dim=-1)
                
                # Get top-k candidates
                topk_probs, topk_ids = log_probs.topk(beam_width)
                
                for prob, idx in zip(topk_probs[0], topk_ids[0]):
                    new_seq = seq + [idx.item()]
                    new_score = score + prob.item()
                    candidates.append((new_seq, new_score, new_h, new_c))
            
            # Keep top beam_width candidates
            candidates.sort(key=lambda x: x[1], reverse=True)
            beams = candidates[:beam_width]
            
            # Early stopping if all beams completed
            if len(beams) == 0:
                break
        
        # Return best sequence with length normalization
        all_seqs = completed + [(seq, score) for seq, score, _, _ in beams]
        all_seqs.sort(key=lambda x: x[1] / len(x[0]), reverse=True)
        
        return all_seqs[0][0]
```

## Information Bottleneck Problem

A fundamental limitation of the basic encoder-decoder architecture is the **information bottleneck**: the entire input sequence must be compressed into a single fixed-dimensional context vector. This creates several challenges:

**Capacity Limitation**: For long sequences, the fixed-size context vector may be insufficient to capture all relevant information. Information from early parts of the sequence tends to be overwritten by later processing.

**Gradient Flow**: During backpropagation, gradients must flow through the entire sequence, making it difficult to learn long-range dependencies effectively.

**Uniform Representation**: All input tokens contribute to a single representation, even though different output tokens may need to focus on different parts of the input.

### Information-Theoretic Analysis

The severity of this problem can be analyzed mathematically. Consider the mutual information between the input $\mathbf{X}$ and the context vector $\mathbf{c}$:

$$I(\mathbf{X}; \mathbf{c}) \leq H(\mathbf{c}) \leq d_c \cdot \log(|\mathcal{V}|)$$

where $d_c$ is the context dimension. This upper bound on information transfer is independent of input length, creating a fundamental bottleneck for long sequences.

This limitation motivated the development of attention mechanisms, which allow the decoder to selectively access different parts of the encoder's hidden states rather than relying on a single compressed representation.

## Multi-Layer Architectures

Stacking multiple RNN layers increases model capacity and allows learning hierarchical representations:

```python
class DeepEncoder(nn.Module):
    """
    Multi-layer encoder with residual connections.
    
    Deeper architectures can capture more abstract representations,
    with residual connections enabling gradient flow.
    """
    
    def __init__(
        self,
        input_size: int,
        embedding_dim: int,
        hidden_size: int,
        num_layers: int = 4,
        dropout: float = 0.2
    ):
        super().__init__()
        
        self.embedding = nn.Embedding(input_size, embedding_dim)
        self.input_projection = nn.Linear(embedding_dim, hidden_size)
        
        # Stack of LSTM layers with residual connections
        self.layers = nn.ModuleList([
            nn.LSTM(
                hidden_size, hidden_size,
                batch_first=True, bidirectional=True
            )
            for _ in range(num_layers)
        ])
        
        # Project bidirectional output back to hidden_size
        self.layer_projections = nn.ModuleList([
            nn.Linear(hidden_size * 2, hidden_size)
            for _ in range(num_layers)
        ])
        
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(hidden_size)
        
    def forward(self, x: torch.Tensor) -> tuple:
        # Embed and project
        embedded = self.input_projection(self.embedding(x))
        
        outputs = embedded
        hidden_states = []
        cell_states = []
        
        for lstm, projection in zip(self.layers, self.layer_projections):
            # Store residual
            residual = outputs
            
            # Process through bidirectional LSTM
            lstm_out, (h, c) = lstm(outputs)
            
            # Project and apply residual connection
            outputs = projection(lstm_out)
            outputs = self.dropout(outputs) + residual
            outputs = self.layer_norm(outputs)
            
            hidden_states.append(h)
            cell_states.append(c)
        
        # Combine layer states
        final_hidden = torch.cat(hidden_states, dim=0)
        final_cell = torch.cat(cell_states, dim=0)
        
        return outputs, final_hidden, final_cell
```

## Bidirectional Encoder with Projection

Using a bidirectional encoder improves context representation, but requires projecting combined states to match decoder dimensions:

```python
class BidirectionalEncoder(nn.Module):
    """
    Bidirectional LSTM encoder with state projection.
    
    Captures context from both directions and projects the combined
    states to match the decoder's expected dimensions.
    """
    
    def __init__(self, vocab_size, embed_dim, hidden_size, num_layers=1, dropout=0.3):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.lstm = nn.LSTM(
            embed_dim, hidden_size, num_layers,
            batch_first=True, bidirectional=True,
            dropout=dropout if num_layers > 1 else 0
        )
        
        # Project bidirectional hidden to decoder size
        self.fc_h = nn.Linear(hidden_size * 2, hidden_size)
        self.fc_c = nn.Linear(hidden_size * 2, hidden_size)
    
    def forward(self, x):
        embedded = self.embedding(x)
        outputs, (h_n, c_n) = self.lstm(embedded)
        
        # Combine forward and backward final states
        # h_n: (num_layers * 2, batch, hidden)
        
        # Concatenate directions and project
        h_combined = torch.cat([h_n[-2], h_n[-1]], dim=-1)  # (batch, hidden*2)
        c_combined = torch.cat([c_n[-2], c_n[-1]], dim=-1)
        
        h_projected = torch.tanh(self.fc_h(h_combined))  # (batch, hidden)
        c_projected = torch.tanh(self.fc_c(c_combined))
        
        return outputs, (h_projected.unsqueeze(0), c_projected.unsqueeze(0))
```

## Practical Considerations

### Weight Initialization

Proper initialization is crucial for training stability:

```python
def initialize_seq2seq(model: nn.Module) -> None:
    """
    Initialize seq2seq model weights.
    
    Uses orthogonal initialization for recurrent weights and
    Xavier initialization for other parameters.
    """
    for name, param in model.named_parameters():
        if 'weight_ih' in name:
            # Input-hidden weights: Xavier
            nn.init.xavier_uniform_(param)
        elif 'weight_hh' in name:
            # Hidden-hidden weights: Orthogonal for stable RNN training
            nn.init.orthogonal_(param)
        elif 'bias' in name:
            # Biases: Zero, except forget gate (set to 1)
            nn.init.zeros_(param)
            if 'bias_hh' in name:
                # For LSTM: set forget gate bias to 1
                n = param.size(0)
                param.data[n//4:n//2].fill_(1.0)
        elif 'embedding' in name:
            nn.init.normal_(param, mean=0, std=0.01)
```

### Training Loop with Gradient Clipping

Essential for preventing exploding gradients in RNN training:

```python
def train_step(
    model: nn.Module,
    batch: tuple,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    clip_value: float = 1.0
) -> float:
    """
    Single training step with gradient clipping.
    """
    model.train()
    optimizer.zero_grad()
    
    src, trg, src_lengths = batch
    outputs = model(src, trg, teacher_forcing_ratio=0.5, src_lengths=src_lengths)
    
    # Reshape for loss computation
    # outputs: (batch, tgt_len, vocab_size)
    # trg: (batch, tgt_len)
    output_dim = outputs.size(-1)
    outputs = outputs[:, 1:].contiguous().view(-1, output_dim)
    trg = trg[:, 1:].contiguous().view(-1)
    
    loss = criterion(outputs, trg)
    loss.backward()
    
    # Clip gradients to prevent explosion
    torch.nn.utils.clip_grad_norm_(model.parameters(), clip_value)
    
    optimizer.step()
    return loss.item()


def train_seq2seq(model, train_loader, optimizer, criterion, device, 
                  clip=1.0, teacher_forcing_ratio=0.5):
    """
    Train Seq2Seq for one epoch.
    """
    model.train()
    total_loss = 0
    
    for batch_idx, (src, tgt) in enumerate(train_loader):
        src, tgt = src.to(device), tgt.to(device)
        
        optimizer.zero_grad()
        
        # Forward pass
        outputs = model(src, tgt, teacher_forcing_ratio)
        
        # Reshape for loss computation
        output_dim = outputs.shape[-1]
        outputs = outputs[:, 1:].reshape(-1, output_dim)
        tgt = tgt[:, 1:].reshape(-1)
        
        loss = criterion(outputs, tgt)
        loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
        
        optimizer.step()
        total_loss += loss.item()
    
    return total_loss / len(train_loader)
```

### Handling Variable-Length Sequences

Efficient batching with padding and packing:

```python
def collate_sequences(
    batch: list,
    pad_idx: int = 0
) -> tuple:
    """
    Collate variable-length sequences into a batch.
    
    Args:
        batch: List of (source, target) tuples
        pad_idx: Padding token index
        
    Returns:
        Padded source, target tensors and length tensors
    """
    sources, targets = zip(*batch)
    
    # Get lengths before padding
    src_lengths = torch.tensor([len(s) for s in sources])
    trg_lengths = torch.tensor([len(t) for t in targets])
    
    # Pad sequences
    src_padded = nn.utils.rnn.pad_sequence(
        [torch.tensor(s) for s in sources],
        batch_first=True,
        padding_value=pad_idx
    )
    trg_padded = nn.utils.rnn.pad_sequence(
        [torch.tensor(t) for t in targets],
        batch_first=True,
        padding_value=pad_idx
    )
    
    return src_padded, trg_padded, src_lengths, trg_lengths
```

## Common Issues and Solutions

| Problem | Cause | Solution |
|---------|-------|----------|
| Information Bottleneck | Fixed context vector cannot capture long sequences | Attention mechanisms |
| Exposure Bias | Model trained with teacher forcing struggles at inference | Scheduled sampling, beam search, RL fine-tuning |
| Unknown Words | Fixed vocabulary cannot handle all words | Subword tokenization (BPE, WordPiece), copy mechanisms |
| Vanishing Gradients | Long sequences attenuate gradients | LSTM/GRU gating, gradient clipping |
| Slow Training | Sequential processing | Layer-wise parallelization, truncated BPTT |

## Historical Context and Evolution

The encoder-decoder architecture emerged from several parallel developments in neural sequence modeling:

**Sutskever et al. (2014)** demonstrated that LSTMs could effectively learn to map sequences to sequences by training on reversed input sequences, achieving breakthrough results in machine translation.

**Cho et al. (2014)** introduced the GRU architecture and proposed learning phrase representations using RNN encoder-decoders for statistical machine translation.

The basic encoder-decoder architecture has since evolved through several innovations:

1. **Attention mechanisms** (Bahdanau et al., 2015) addressed the information bottleneck
2. **Transformer architecture** (Vaswani et al., 2017) replaced recurrence with self-attention
3. **Pre-trained models** (BERT, GPT) leveraged large-scale unsupervised pre-training

## Summary

The encoder-decoder architecture provides an elegant solution to the sequence-to-sequence learning problem through a two-stage approach: encoding the input into a fixed representation and decoding this representation into an output sequence.

**Key components**:
- Encoder compresses source sequence into context vector
- Decoder generates target sequence autoregressively from context
- Bidirectional encoders capture full source context
- Teacher forcing accelerates training using ground truth

**Key techniques**:
- Scheduled sampling mitigates exposure bias
- Beam search improves generation quality
- Packed sequences handle variable lengths efficiently
- Gradient clipping prevents exploding gradients
- Proper weight initialization ensures training stability

The fundamental limitation—compressing all information into a fixed-size vector—motivates attention mechanisms covered in the next section. Despite this limitation, understanding the encoder-decoder paradigm remains essential as it provides the conceptual foundation for modern sequence-to-sequence learning, including transformers and large language models.
