# Code: French→English Translation

## Overview

This implementation builds a complete sequence-to-sequence translation system that translates French sentences into English using an encoder-decoder architecture with Bahdanau (additive) attention. The pipeline covers data acquisition, text preprocessing, vocabulary construction, model definition, training with teacher forcing, greedy decoding, and attention visualization.

**Source.** Adapted from the [PyTorch seq2seq translation tutorial](https://github.com/pytorch/tutorials/blob/main/intermediate_source/seq2seq_translation_tutorial.py), restructured as a single self-contained script with detailed annotations.

## Architecture

The model consists of three components working in concert:

```
Source: "je suis étudiant"
         ↓
   ┌──────────────────────┐
   │   Encoder (GRU)      │
   │                      │
   │  x₁ → h₁            │
   │  x₂ → h₂            │  → all encoder hidden states H = [h₁, h₂, ..., h_T]
   │  ...                 │  → final hidden state h_T
   │  x_T → h_T           │
   └──────────────────────┘
              ↓
   ┌──────────────────────┐
   │   Bahdanau Attention  │
   │                      │
   │  score(sₜ, hⱼ) =     │
   │    Vᵀ tanh(Wa sₜ +   │  → context vector cₜ = Σⱼ αₜⱼ hⱼ
   │         Ua hⱼ)       │  → attention weights αₜ
   └──────────────────────┘
              ↓
   ┌──────────────────────┐
   │   Decoder (GRU)      │
   │                      │
   │  [embed(yₜ₋₁); cₜ]  │
   │       → GRU          │  → ŷₜ (log-softmax over target vocab)
   │       → Linear       │
   └──────────────────────┘
              ↓
Target: "i am a student"
```

### Mathematical Formulation

**Encoder.** For source tokens $(x_1, \ldots, x_T)$:

$$\mathbf{e}_t = \text{Embed}(x_t) \in \mathbb{R}^H$$
$$\mathbf{h}_t^{enc} = \text{GRU}(\mathbf{e}_t, \mathbf{h}_{t-1}^{enc}) \in \mathbb{R}^H$$

**Attention.** At decoder step $t$, the Bahdanau additive attention computes alignment scores between the decoder state $\mathbf{s}_t$ and each encoder state $\mathbf{h}_j^{enc}$:

$$\text{score}(\mathbf{s}_t, \mathbf{h}_j) = \mathbf{v}_a^T \tanh(\mathbf{W}_a \mathbf{s}_t + \mathbf{U}_a \mathbf{h}_j)$$
$$\alpha_{t,j} = \frac{\exp(\text{score}(\mathbf{s}_t, \mathbf{h}_j))}{\sum_{k=1}^{T} \exp(\text{score}(\mathbf{s}_t, \mathbf{h}_k))}$$
$$\mathbf{c}_t = \sum_{j=1}^{T} \alpha_{t,j} \, \mathbf{h}_j^{enc}$$

**Decoder.** At each step, the decoder concatenates the embedded previous token with the attention context and feeds the result through a GRU:

$$\mathbf{s}_t = \text{GRU}([\text{Embed}(y_{t-1}); \mathbf{c}_t], \mathbf{s}_{t-1})$$
$$P(y_t | y_{<t}, \mathbf{x}) = \text{softmax}(\mathbf{W}_o \mathbf{s}_t)$$

Teacher forcing with probability $p = 0.5$ randomly chooses between feeding the ground-truth previous token $y_{t-1}$ and the model's own prediction $\hat{y}_{t-1}$ during training.

## Data Pipeline

### Download and Preprocessing

The dataset is the Tatoeba Project's English-French sentence pairs, distributed as a tab-separated text file. Preprocessing normalizes Unicode characters to ASCII, lowercases all text, and inserts spaces before punctuation:

```python
SOS_TOKEN = 0
EOS_TOKEN = 1
MAX_LENGTH = 10

def unicode_to_ascii(s: str) -> str:
    return "".join(
        c for c in unicodedata.normalize("NFD", s)
        if unicodedata.category(c) != "Mn"
    )

def normalize_string(s: str) -> str:
    s = unicode_to_ascii(s.lower().strip())
    s = re.sub(r"([.!?])", r" \1", s)
    s = re.sub(r"[^a-zA-Z.!?]+", r" ", s)
    return s
```

Pairs are filtered to sentences under 10 tokens where the English sentence starts with common prefixes (`"i am"`, `"he is"`, `"they are"`, etc.), yielding a focused dataset of simple declarative sentences suitable for training a small model.

### Vocabulary Construction

The `Lang` class maintains bidirectional word↔index mappings and word frequency counts:

```python
class Lang:
    def __init__(self, name: str):
        self.name = name
        self.word2index: dict[str, int] = {}
        self.word2count: dict[str, int] = {}
        self.index2word: dict[int, str] = {0: "SOS", 1: "EOS"}
        self.n_words: int = 2  # SOS + EOS

    def add_sentence(self, sentence: str) -> None:
        for word in sentence.split(" "):
            self.add_word(word)

    def add_word(self, word: str) -> None:
        if word not in self.word2index:
            self.word2index[word] = self.n_words
            self.word2count[word] = 1
            self.index2word[self.n_words] = word
            self.n_words += 1
        else:
            self.word2count[word] += 1
```

After constructing vocabularies for both languages, sentence pairs are converted to padded integer tensors and wrapped in a `DataLoader` with batch size 32.

## Model Implementation

### Encoder

The encoder embeds each source token and processes the sequence through a single-layer GRU, collecting hidden states at every position:

```python
class EncoderRNN(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, dropout_p: float = 0.1):
        super().__init__()
        self.hidden_size = hidden_size
        self.embedding = nn.Embedding(input_size, hidden_size)
        self.gru = nn.GRU(hidden_size, hidden_size, batch_first=True)
        self.dropout = nn.Dropout(dropout_p)

    def forward(self, x: torch.Tensor):
        """
        Parameters
        ----------
        x : LongTensor, shape (B, T)

        Returns
        -------
        all_hidden : Tensor, shape (B, T, H)
        last_hidden : Tensor, shape (1, B, H)
        """
        hidden = None
        all_hidden_list = []
        for t in range(x.size(1)):
            embedded = self.dropout(self.embedding(x[:, t].unsqueeze(1)))
            output, hidden = self.gru(embedded, hidden)
            all_hidden_list.append(hidden.squeeze(0).unsqueeze(1))
        all_hidden = torch.cat(all_hidden_list, dim=1)
        return all_hidden, hidden
```

The step-by-step loop (rather than processing the full sequence at once) is used here for pedagogical clarity—it makes explicit how the hidden state evolves at each timestep. In production code, passing the entire sequence to `self.gru` in a single call is more efficient.

### Bahdanau Attention

The attention module computes additive alignment scores between the decoder query and all encoder keys:

```python
class BahdanauAttention(nn.Module):
    def __init__(self, hidden_size: int):
        super().__init__()
        self.Wa = nn.Linear(hidden_size, hidden_size)
        self.Ua = nn.Linear(hidden_size, hidden_size)
        self.Va = nn.Linear(hidden_size, 1)

    def forward(self, query: torch.Tensor, keys: torch.Tensor):
        """
        query : (B, 1, H) — decoder hidden state
        keys  : (B, T, H) — all encoder hidden states

        Returns context (B, 1, H) and weights (B, 1, T)
        """
        scores = self.Va(torch.tanh(self.Wa(query) + self.Ua(keys)))
        scores = scores.squeeze(2).unsqueeze(1)  # (B, 1, T)
        weights = F.softmax(scores, dim=-1)
        context = torch.bmm(weights, keys)        # (B, 1, H)
        return context, weights
```

The broadcasting in `self.Wa(query) + self.Ua(keys)` works because `query` has shape $(B, 1, H)$ and `self.Ua(keys)` has shape $(B, T, H)$—PyTorch broadcasts the query across all $T$ encoder positions.

### Attention Decoder

The decoder combines embedding, attention, and GRU into an autoregressive generation loop:

```python
class AttnDecoderRNN(nn.Module):
    def __init__(self, hidden_size: int, output_size: int, dropout_p: float = 0.1):
        super().__init__()
        self.embedding = nn.Embedding(output_size, hidden_size)
        self.attention = BahdanauAttention(hidden_size)
        self.gru = nn.GRU(2 * hidden_size, hidden_size, batch_first=True)
        self.out = nn.Linear(hidden_size, output_size)
        self.dropout = nn.Dropout(dropout_p)

    def forward(self, encoder_hidden, all_encoder_hidden, target_tensor=None):
        batch_size = all_encoder_hidden.size(0)
        decoder_input = torch.empty(
            batch_size, 1, dtype=torch.long, device=DEVICE
        ).fill_(SOS_TOKEN)
        decoder_hidden = encoder_hidden

        decoder_outputs, attentions = [], []
        for t in range(MAX_LENGTH):
            dec_out, decoder_hidden, attn_w = self._step(
                decoder_input, decoder_hidden, all_encoder_hidden
            )
            decoder_outputs.append(dec_out)
            attentions.append(attn_w)

            if target_tensor is not None and random.random() < TEACHER_FORCING_RATIO:
                decoder_input = target_tensor[:, t].unsqueeze(1)
            else:
                _, topi = dec_out.topk(1)
                decoder_input = topi.squeeze(-1).detach()

        decoder_outputs = torch.cat(decoder_outputs, dim=1)
        decoder_outputs = F.log_softmax(decoder_outputs, dim=-1)
        attentions = torch.cat(attentions, dim=1)
        return decoder_outputs, decoder_hidden, attentions
```

Key design choices:

- **GRU input dimension is $2H$**: the concatenation of the embedded token ($H$) and the attention context ($H$).
- **Teacher forcing ratio 0.5**: balances exposure bias (always seeing ground truth) against training instability (always seeing own predictions). See the [Scheduled Sampling](scheduled_sampling.md) section for curriculum-based alternatives.
- **`detach()` on argmax predictions**: prevents gradients from flowing through the discrete sampling operation during free-running mode.

## Training

Training uses Adam optimization with learning rate $10^{-3}$ and negative log-likelihood loss on the decoder's log-softmax output:

```python
def train_epoch(dataloader, encoder, decoder, enc_opt, dec_opt, criterion):
    total_loss = 0.0
    for input_t, target_t in dataloader:
        enc_opt.zero_grad()
        dec_opt.zero_grad()

        all_hidden, last_hidden = encoder(input_t)
        dec_out, _, _ = decoder(last_hidden, all_hidden, target_t)

        loss = criterion(
            dec_out.view(-1, dec_out.size(-1)),
            target_t.view(-1),
        )
        loss.backward()
        enc_opt.step()
        dec_opt.step()
        total_loss += loss.item()

    return total_loss / len(dataloader)
```

The loss is computed over all decoder positions simultaneously by reshaping the output to $(B \cdot T, V)$ and the target to $(B \cdot T)$. Padding tokens (index 0) contribute to the loss—a production system would pass `ignore_index=0` to `nn.NLLLoss` to mask them out.

Training runs for 80 epochs on the filtered dataset (~10,000 pairs), which takes approximately 5–10 minutes on a modern GPU or 20–30 minutes on CPU.

## Evaluation and Inference

### Greedy Decoding

At inference time, the decoder runs without teacher forcing, selecting the highest-probability token at each step:

```python
def evaluate_sentence(encoder, decoder, sentence, input_lang, output_lang):
    with torch.no_grad():
        inp_t = tensor_from_sentence(input_lang, sentence).unsqueeze(0)
        padded = torch.zeros(1, MAX_LENGTH, dtype=torch.long, device=DEVICE)
        padded[0, :inp_t.size(1)] = inp_t

        all_hidden, last_hidden = encoder(padded)
        dec_out, _, attentions = decoder(last_hidden, all_hidden)

        decoded_words = []
        for step in range(MAX_LENGTH):
            _, topi = dec_out[:, step, :].topk(1)
            idx = topi[0, 0].item()
            if idx == EOS_TOKEN:
                decoded_words.append("<EOS>")
                break
            decoded_words.append(output_lang.index2word.get(idx, "<UNK>"))
    return decoded_words, attentions
```

This implements greedy search—taking the argmax at each step. For higher-quality translations, beam search (see [Beam Search](beam_search.md)) maintains multiple hypotheses and generally produces better results at the cost of increased computation.

### Attention Visualization

The attention weight matrix $\boldsymbol{\alpha} \in \mathbb{R}^{T' \times T}$ reveals which source tokens the decoder attends to at each generation step. Visualizing this matrix as a heatmap provides interpretable evidence of the model's alignment behavior:

```python
def show_attention(input_sentence, output_words, attentions):
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111)
    n_out = len(output_words)
    n_in = len(input_sentence.split(" ")) + 1  # +1 for <EOS>
    cax = ax.matshow(attentions[0, :n_out, :n_in].cpu().numpy(), cmap="bone")
    fig.colorbar(cax)
    ax.set_xticklabels([""] + input_sentence.split(" ") + ["<EOS>"], rotation=90)
    ax.set_yticklabels([""] + output_words)
    plt.tight_layout()
    plt.show()
```

For well-trained models, the attention matrix approximates a monotonic diagonal for simple translations (word order is preserved) and shows characteristic crossing patterns for language pairs with different word orders (e.g., adjective-noun order reversal between French and English).

## Running the Code

```bash
python seq2seq_attention.py
```

The script automatically downloads the Tatoeba dataset, trains for 80 epochs, saves encoder and decoder weights to `model/`, and runs evaluation on random pairs plus four held-out test sentences with attention visualization.

### Expected Output

```
Data ready.
  Pairs: 11445
  fra vocab: 4601
  eng vocab: 2991
Example pair: ['je suis pret .', 'i m ready .']

0m 45s (- 11m 15s) (epoch 5 6%) loss=3.8234
1m 28s (- 10m 12s) (epoch 10 12%) loss=2.9156
...
11m 02s (- 0m 0s) (epoch 80 100%) loss=0.5483

> je suis desole si c est une question idiote
= i m sorry if it s a stupid question
< i m sorry if it s a stupid question . <EOS>
```

## Hyperparameters

| Parameter | Value | Notes |
|-----------|-------|-------|
| `HIDDEN_SIZE` | 128 | GRU hidden dimension for both encoder and decoder |
| `BATCH_SIZE` | 32 | Training mini-batch size |
| `TEACHER_FORCING_RATIO` | 0.5 | Probability of using ground truth at each decoder step |
| `MAX_LENGTH` | 10 | Maximum sentence length (tokens); longer pairs are filtered |
| `lr` | 0.001 | Adam learning rate |
| `n_epochs` | 80 | Training epochs |
| `dropout_p` | 0.1 | Dropout on embeddings |

## Connections to Other Sections

This implementation integrates concepts from across Chapter 7:

- **Word Embeddings (§7.1)**: Both encoder and decoder use learned `nn.Embedding` layers that map token indices to dense vectors. These embeddings are trained jointly with the rest of the model rather than using pretrained vectors.
- **RNN / Hidden States (§7.2)**: The GRU cells in encoder and decoder maintain hidden states that accumulate sequential information, directly implementing the recurrence equations from the RNN sections.
- **LSTM/GRU Gates (§7.3)**: The GRU's update and reset gates enable selective memory retention across the sequence, mitigating vanishing gradients that would plague a vanilla RNN on these translation sequences.
- **Bahdanau Attention (§7.4)**: The additive attention mechanism eliminates the information bottleneck of the fixed context vector, allowing the decoder to query relevant encoder states at each generation step.
- **Encoder-Decoder Framework (§7.5)**: The overall architecture instantiates the encoder-decoder paradigm with attention-augmented context vectors.
- **Teacher Forcing (§7.5)**: Training uses stochastic teacher forcing to balance convergence speed against exposure bias.

## Quantitative Finance Extensions

The same encoder-decoder-attention architecture applies to several financial sequence transduction tasks:

- **Signal generation from text**: Encode analyst report sentences, decode structured trading signals (direction, magnitude, confidence) conditioned on the encoded financial narrative.
- **Order execution**: Encode a parent order specification (size, urgency, constraints), decode a sequence of child order actions (limit prices, quantities, timing) that optimize execution quality.
- **Cross-asset translation**: Encode a sequence of macro indicators or yield curve movements, decode predicted equity factor returns—effectively "translating" between asset class languages.
- **Report summarization**: Encode lengthy earnings call transcripts, decode concise summaries highlighting material information for portfolio managers.

In each case, the attention mechanism provides the critical benefit of interpretability—attention weights identify which input elements most influenced each output decision, supporting the explainability requirements common in financial applications.
