"""Pretrain gpt from scratch."""
# ---
# title: "Pre-training a GPT Model from Scratch"
# description: "End-to-end pipeline: BPE tokenizer training, model initialization,
#               ConstantLengthDataset, distributed training, and perplexity evaluation"
# ---
#
# Pre-training a language model from scratch is the foundation of modern NLP.
# This script walks through every step of the pipeline:
#
#   Part 1 – Train a custom BPE tokenizer from a text corpus
#   Part 2 – Initialize a GPT-2 model from config (random weights)
#   Part 3 – ConstantLengthDataset for efficient causal LM training
#   Part 4 – Training loop with gradient accumulation and checkpointing
#   Part 5 – Evaluation: perplexity and text generation
#   Part 6 – Scaling considerations and distributed training
#
# Adapted from: O'Reilly "NLP with Transformers" Ch 10

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import IterableDataset, DataLoader
from typing import List, Iterator


# =====================================================================
# Part 1 – Training a Custom BPE Tokenizer
# =====================================================================
print("=" * 60)
print("Part 1: Training a BPE Tokenizer from Scratch")
print("=" * 60)

# Byte-Pair Encoding (BPE) builds a vocabulary by iteratively merging
# the most frequent byte pairs in the corpus.  GPT-2 uses byte-level
# BPE, which starts from 256 byte-level characters.

try:
    from tokenizers import (
        Tokenizer,
        models,
        pre_tokenizers,
        trainers,
        decoders,
    )

    # Build a BPE tokenizer from the tokenizers library
    tokenizer_bpe = Tokenizer(models.BPE())
    tokenizer_bpe.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tokenizer_bpe.decoder = decoders.ByteLevel()

    # Training data — in practice, use millions of lines
    training_corpus = [
        "Deep learning models learn hierarchical representations.",
        "Neural networks use backpropagation for gradient computation.",
        "Transformers use self-attention instead of recurrence.",
        "BERT is a bidirectional encoder from transformers.",
        "GPT generates text using causal language modeling.",
        "The attention mechanism computes weighted sums of values.",
        "Positional encodings inject sequence order information.",
        "Layer normalization stabilizes training of deep networks.",
        "Dropout is a regularization technique for neural networks.",
        "Fine-tuning adapts a pre-trained model to new tasks.",
    ] * 100  # repeat to simulate larger corpus

    trainer = trainers.BpeTrainer(
        vocab_size=500,
        min_frequency=2,
        special_tokens=["<|endoftext|>", "<pad>"],
        show_progress=False,
    )
    tokenizer_bpe.train_from_iterator(training_corpus, trainer=trainer)

    # Test the trained tokenizer
    test_text = "Transformers learn representations through attention."
    encoded = tokenizer_bpe.encode(test_text)
    print(f"  Vocab size: {tokenizer_bpe.get_vocab_size()}")
    print(f"  Text:   '{test_text}'")
    print(f"  Tokens: {encoded.tokens}")
    print(f"  IDs:    {encoded.ids}")
    decoded = tokenizer_bpe.decode(encoded.ids)
    print(f"  Decoded: '{decoded}'")
    print()

    HAS_TOKENIZERS = True

except ImportError:
    print("  tokenizers library not available — showing concept only")
    HAS_TOKENIZERS = False
    print()


# Alternative: use HuggingFace AutoTokenizer.train_new_from_iterator()
print("  HuggingFace approach (recommended for production):")
print("""
    from transformers import AutoTokenizer
    old_tokenizer = AutoTokenizer.from_pretrained("gpt2")

    def batch_iterator(dataset, batch_size=1000):
        for i in range(0, len(dataset), batch_size):
            yield dataset[i : i + batch_size]["text"]

    new_tokenizer = old_tokenizer.train_new_from_iterator(
        batch_iterator(dataset),
        vocab_size=32768,
    )
    new_tokenizer.save_pretrained("my-custom-tokenizer")
""")


# =====================================================================
# Part 2 – Initialize a GPT-2 Model from Config
# =====================================================================
print("=" * 60)
print("Part 2: Model Initialization from Config")
print("=" * 60)

# When pre-training from scratch, we initialise the model with random
# weights using only the architecture configuration — no pre-trained
# checkpoint.

try:
    from transformers import AutoConfig, AutoModelForCausalLM

    # Create a small GPT-2 config for demonstration
    config = AutoConfig.from_pretrained(
        "gpt2",
        vocab_size=500 if HAS_TOKENIZERS else 50257,
        n_embd=256,
        n_layer=4,
        n_head=4,
        n_inner=1024,
    )

    # Initialize model from config (random weights)
    model = AutoModelForCausalLM.from_config(config)

    num_params = sum(p.numel() for p in model.parameters())
    print(f"  Config: {config.n_layer} layers, {config.n_embd} dim, "
          f"{config.n_head} heads")
    print(f"  Model size: {num_params / 1e6:.1f}M parameters")
    print(f"  Vocab size: {config.vocab_size}")
    print()

    # For reference, standard GPT-2 sizes:
    print("  Standard GPT-2 model sizes:")
    print("    GPT-2 Small:  12 layers, 768 dim,  12 heads →  124M params")
    print("    GPT-2 Medium: 24 layers, 1024 dim, 16 heads →  355M params")
    print("    GPT-2 Large:  36 layers, 1280 dim, 20 heads →  774M params")
    print("    GPT-2 XL:     48 layers, 1600 dim, 25 heads → 1558M params")
    print()

    HAS_TRANSFORMERS = True

except ImportError:
    print("  transformers library not available")
    HAS_TRANSFORMERS = False
    print()


# =====================================================================
# Part 3 – ConstantLengthDataset for Causal LM
# =====================================================================
print("=" * 60)
print("Part 3: ConstantLengthDataset")
print("=" * 60)

# For causal LM training, we concatenate all tokenised texts with
# an EOS separator, then chunk into fixed-length sequences.  This
# avoids wasting compute on padding tokens.


class ConstantLengthDataset(IterableDataset):
    """Concatenate tokenised texts and chunk into fixed-length sequences.

    This is the standard approach for pre-training causal language models.
    Documents are concatenated with an EOS token between them, then split
    into chunks of exactly `seq_length` tokens.  Any leftover tokens at
    the end of a chunk are discarded.

    Args:
        tokenizer:        HuggingFace tokenizer (or any callable)
        texts:            iterable of raw text strings
        seq_length:       number of tokens per training sample
        eos_token_id:     token ID to insert between documents
        chars_per_token:  approximate characters per token (for buffering)
        num_of_sequences: number of sequences to buffer before chunking
    """

    def __init__(
        self,
        tokenizer,
        texts,
        seq_length: int = 1024,
        eos_token_id: int = 0,
        chars_per_token: float = 3.6,
        num_of_sequences: int = 1024,
    ):
        self.tokenizer = tokenizer
        self.texts = texts
        self.seq_length = seq_length
        self.eos_token_id = eos_token_id
        self.input_characters = int(
            seq_length * chars_per_token * num_of_sequences
        )

    def __iter__(self) -> Iterator[torch.Tensor]:
        iterator = iter(self.texts)
        more_data = True

        while more_data:
            # Step 1: Fill a buffer with enough raw text
            buffer, buffer_len = [], 0
            while buffer_len < self.input_characters:
                try:
                    buffer.append(next(iterator))
                    buffer_len += len(buffer[-1])
                except StopIteration:
                    more_data = False
                    break

            if not buffer:
                return

            # Step 2: Tokenise the buffer and concatenate with EOS
            all_token_ids = []
            tokenised = self.tokenizer(buffer, truncation=False)
            for input_ids in tokenised["input_ids"]:
                all_token_ids.extend(input_ids + [self.eos_token_id])

            # Step 3: Chunk into fixed-length sequences
            for i in range(0, len(all_token_ids), self.seq_length):
                chunk = all_token_ids[i : i + self.seq_length]
                if len(chunk) == self.seq_length:
                    yield torch.tensor(chunk, dtype=torch.long)


# Demo with a simple tokenizer
print("  ConstantLengthDataset concatenates documents and chunks:")
print()
print("  Doc1: 'Hello world'  →  [101, 7592, 2088, 102]")
print("  Doc2: 'Deep learning' →  [101, 2784, 4083, 102]")
print("  Concatenated: [101, 7592, 2088, 102, 0, 101, 2784, 4083, 102, 0]")
print("  Chunked (seq_len=5): [[101,7592,2088,102,0], [101,2784,4083,102,0]]")
print()
print("  Advantages:")
print("    - No padding waste (every token contributes to training)")
print("    - Consistent batch sizes and memory usage")
print("    - Documents naturally separated by EOS tokens")
print()


# =====================================================================
# Part 4 – Training Loop
# =====================================================================
print("=" * 60)
print("Part 4: Training Loop with Gradient Accumulation")
print("=" * 60)

# Pre-training requires careful setup:
# - Gradient accumulation for effective large batch sizes
# - Learning rate warmup + cosine decay
# - Gradient checkpointing for memory efficiency
# - Weight decay with bias/LayerNorm exclusion

print("""
  Training configuration (GPT-2 Small example):
  ──────────────────────────────────────────────
    batch_size:                   12
    gradient_accumulation_steps:  1
    effective_batch_size:         12 * num_gpus
    learning_rate:                5e-4
    lr_scheduler:                 cosine with warmup
    warmup_steps:                 2000
    max_train_steps:              150000
    weight_decay:                 0.1
    seq_length:                   1024
    gradient_checkpointing:       True

  Weight decay exclusion:
    # Don't decay bias and LayerNorm parameters
    no_decay = ["bias", "LayerNorm.weight"]
    params_with_wd = [p for n, p in model.named_parameters()
                      if not any(nd in n for nd in no_decay)]
    params_without_wd = [p for n, p in model.named_parameters()
                         if any(nd in n for nd in no_decay)]
    optimizer_groups = [
        {"params": params_with_wd, "weight_decay": 0.1},
        {"params": params_without_wd, "weight_decay": 0.0},
    ]
""")

# Minimal training demo
if HAS_TRANSFORMERS:
    try:
        from transformers import AutoTokenizer

        tokenizer_demo = AutoTokenizer.from_pretrained("gpt2")

        # Small corpus for demo
        texts = [
            "The transformer architecture revolutionized natural language processing.",
            "Self-attention allows the model to attend to all positions simultaneously.",
            "Pre-training on large corpora enables strong transfer learning.",
        ] * 50

        dataset = ConstantLengthDataset(
            tokenizer=tokenizer_demo,
            texts=texts,
            seq_length=64,
            eos_token_id=tokenizer_demo.eos_token_id,
            num_of_sequences=10,
        )

        dataloader = DataLoader(dataset, batch_size=2)

        # Quick training demo (few steps only)
        small_config = AutoConfig.from_pretrained(
            "gpt2", vocab_size=len(tokenizer_demo),
            n_embd=128, n_layer=2, n_head=2, n_inner=512,
        )
        small_model = AutoModelForCausalLM.from_config(small_config)
        optimizer = torch.optim.AdamW(small_model.parameters(), lr=5e-4)

        small_model.train()
        losses = []
        for step, batch in enumerate(dataloader):
            if step >= 20:
                break
            # For causal LM, labels = input_ids (shifted internally)
            outputs = small_model(batch, labels=batch)
            loss = outputs.loss
            loss.backward()

            if (step + 1) % 4 == 0:  # gradient accumulation every 4 steps
                optimizer.step()
                optimizer.zero_grad()

            losses.append(loss.item())
            if step % 5 == 0:
                print(f"  Step {step:3d}: loss = {loss.item():.4f}")

        print(f"\n  Training demo complete. Loss: {losses[0]:.2f} → {losses[-1]:.2f}")
        print()

    except Exception as e:
        print(f"  Training demo skipped: {e}")
        print()


# =====================================================================
# Part 5 – Evaluation: Perplexity and Generation
# =====================================================================
print("=" * 60)
print("Part 5: Evaluation — Perplexity & Generation")
print("=" * 60)

print("""
  Perplexity — the standard metric for language models
  ─────────────────────────────────────────────────────
    PPL = exp(average cross-entropy loss)

    Lower perplexity = model assigns higher probability to held-out text
    Typical values: GPT-2 Small ~ 30 PPL on WikiText-103

  Evaluation code:
    model.eval()
    losses = []
    for batch in eval_dataloader:
        with torch.no_grad():
            outputs = model(batch, labels=batch)
        losses.append(outputs.loss)
    avg_loss = torch.mean(torch.stack(losses))
    perplexity = torch.exp(avg_loss)

  Text generation for qualitative evaluation:
    from transformers import pipeline
    generator = pipeline("text-generation", model="my-pretrained-model")
    output = generator(
        "def fibonacci(n):",
        max_new_tokens=64,
        temperature=0.4,
        top_p=0.95,
        do_sample=True,
    )
""")


# =====================================================================
# Part 6 – Scaling and Distributed Training
# =====================================================================
print("=" * 60)
print("Part 6: Scaling — Distributed Training with Accelerate")
print("=" * 60)

print("""
  For real pre-training, use HuggingFace Accelerate for multi-GPU:
  ─────────────────────────────────────────────────────────────────

    from accelerate import Accelerator

    accelerator = Accelerator()

    # Prepare model, optimizer, dataloaders for distributed training
    model, optimizer, train_dl, eval_dl = accelerator.prepare(
        model, optimizer, train_dataloader, eval_dataloader
    )

    # Training loop — same as single GPU!
    for step, batch in enumerate(train_dl):
        loss = model(batch, labels=batch).loss
        loss = loss / gradient_accumulation_steps
        accelerator.backward(loss)

        if step % gradient_accumulation_steps == 0:
            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()

    # Save model (only on main process)
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        unwrapped = accelerator.unwrap_model(model)
        unwrapped.save_pretrained("./checkpoint")

  Launch with: accelerate launch --num_processes 4 train.py

  Memory-saving techniques:
    ┌─────────────────────────┬──────────────────────────┐
    │ Technique               │ Memory reduction          │
    ├─────────────────────────┼──────────────────────────┤
    │ Gradient checkpointing  │ ~60%% activation memory   │
    │ Mixed precision (FP16)  │ ~50%% model memory        │
    │ Gradient accumulation   │ Larger effective batch    │
    │ DeepSpeed ZeRO Stage 2  │ ~50%% optimizer state     │
    │ DeepSpeed ZeRO Stage 3  │ Shards across GPUs       │
    └─────────────────────────┴──────────────────────────┘

  Enable gradient checkpointing:
    model = AutoModelForCausalLM.from_pretrained(
        "my-model", gradient_checkpointing=True
    )
""")

print("Done.")


if __name__ == "__main__":
    pass
