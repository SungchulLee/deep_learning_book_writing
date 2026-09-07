# GPT 모델을 맨바닥부터 미리 익히기

말 모델을 맨바닥부터 미리 익히는 일은 요즘 자연어 다루기의 바탕이다. 이 끝에서 끝까지의 물길은 모든 단계를 다룬다. 곧 맞춤 BPE 토막내개 익히기, 자리매김에서 GPT-2 모델 첫자리매김하기, 효율적인 인과 말 나타내기를 위한 ConstantLengthDataset 세우기, 기울기 쌓기를 곁들인 익히기 되풀이 돌리기, 헷갈림도로 값매김하기, HuggingFace Accelerate로 흩뿌린 익히기까지 키우기이다.

## 코드

```python
"""GPT를 맨바닥부터 미리 익히기."""
# ---
# title: "GPT 모델을 맨바닥부터 미리 익히기"
# description: "끝에서 끝까지의 물길: BPE 토막내개 익히기, 모델 첫자리매김,
#               ConstantLengthDataset, 흩뿌린 익히기, 헷갈림도 값매김"
# ---
#
# 말 모델을 맨바닥부터 미리 익히는 일은 요즘 자연어 다루기의 바탕이다.
# 이 각본은 물길의 모든 단계를 짚어 나간다:
#
#   1부 – 글 말뭉치로 맞춤 BPE 토막내개 익히기
#   2부 – 자리매김에서 GPT-2 모델 첫자리매김(마구잡이 무게)
#   3부 – 효율적인 인과 말 모델 익히기를 위한 ConstantLengthDataset
#   4부 – 기울기 쌓기와 중간 저장을 곁들인 익히기 되풀이
#   5부 – 값매김: 헷갈림도와 글 만들어 내기
#   6부 – 규모 키우기에서 헤아릴 점과 흩뿌린 익히기
#
# 바탕: O'Reilly "NLP with Transformers" 10장

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import IterableDataset, DataLoader
from typing import List, Iterator


# =====================================================================
# 1부 – 맞춤 BPE 토막내개 익히기
# =====================================================================
print("=" * 60)
print("Part 1: Training a BPE Tokenizer from Scratch")
print("=" * 60)

try:
    from tokenizers import (
        Tokenizer,
        models,
        pre_tokenizers,
        trainers,
        decoders,
    )

    tokenizer_bpe = Tokenizer(models.BPE())
    tokenizer_bpe.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tokenizer_bpe.decoder = decoders.ByteLevel()

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
    ] * 100

    trainer = trainers.BpeTrainer(
        vocab_size=500,
        min_frequency=2,
        special_tokens=["<|endoftext|>", "<pad>"],
        show_progress=False,
    )
    tokenizer_bpe.train_from_iterator(training_corpus, trainer=trainer)

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
# 2부 – 자리매김에서 GPT-2 모델 첫자리매김
# =====================================================================
print("=" * 60)
print("Part 2: Model Initialization from Config")
print("=" * 60)

try:
    from transformers import AutoConfig, AutoModelForCausalLM

    config = AutoConfig.from_pretrained(
        "gpt2",
        vocab_size=500 if HAS_TOKENIZERS else 50257,
        n_embd=256,
        n_layer=4,
        n_head=4,
        n_inner=1024,
    )

    model = AutoModelForCausalLM.from_config(config)

    num_params = sum(p.numel() for p in model.parameters())
    print(f"  Config: {config.n_layer} layers, {config.n_embd} dim, "
          f"{config.n_head} heads")
    print(f"  Model size: {num_params / 1e6:.1f}M parameters")
    print(f"  Vocab size: {config.vocab_size}")
    print()

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
# 3부 – 인과 말 모델을 위한 ConstantLengthDataset
# =====================================================================
print("=" * 60)
print("Part 3: ConstantLengthDataset")
print("=" * 60)


class ConstantLengthDataset(IterableDataset):
    """토막낸 글을 이어 붙이고 붙박이 길이의 차례로 자르기."""

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

            all_token_ids = []
            tokenised = self.tokenizer(buffer, truncation=False)
            for input_ids in tokenised["input_ids"]:
                all_token_ids.extend(input_ids + [self.eos_token_id])

            for i in range(0, len(all_token_ids), self.seq_length):
                chunk = all_token_ids[i : i + self.seq_length]
                if len(chunk) == self.seq_length:
                    yield torch.tensor(chunk, dtype=torch.long)


print("  ConstantLengthDataset concatenates documents and chunks:")
print()
print("  Doc1: 'Hello world'  →  [101, 7592, 2088, 102]")
print("  Doc2: 'Deep learning' →  [101, 2784, 4083, 102]")
print("  Concatenated: [101, 7592, 2088, 102, 0, 101, 2784, 4083, 102, 0]")
print("  Chunked (seq_len=5): [[101,7592,2088,102,0], [101,2784,4083,102,0]]")
print()


# =====================================================================
# 4부 – 익히기 되풀이
# =====================================================================
print("=" * 60)
print("Part 4: Training Loop with Gradient Accumulation")
print("=" * 60)

print("""
  익히기 자리매김(GPT-2 Small 보기):
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
""")


# =====================================================================
# 5부 – 값매김: 헷갈림도와 만들어 내기
# =====================================================================
print("=" * 60)
print("Part 5: Evaluation — Perplexity & Generation")
print("=" * 60)

print("""
  헷갈림도 — 말 모델의 표준 잣대
    PPL = exp(평균 엇갈린 엔트로피 손실)
    헷갈림도가 낮다 = 모델이 남겨 둔 글에 더 높은 확률을 준다
""")


# =====================================================================
# 6부 – 규모 키우기와 흩뿌린 익히기
# =====================================================================
print("=" * 60)
print("Part 6: Scaling — Distributed Training with Accelerate")
print("=" * 60)

print("""
  실제로 미리 익힐 때는 여러 GPU를 쓰려 HuggingFace Accelerate를 쓴다.
  이렇게 띄운다: accelerate launch --num_processes 4 train.py
""")

print("Done.")


if __name__ == "__main__":
    pass
```

## 논의

GPT 모델의 미리 익히기 물길은 서로 얽힌 조각 여럿으로 이루어지며, 저마다 익히기의 성공에 결정적이다. 첫 단계는 분야에 맞춘 BPE 토막내개를 익히는 것인데, 이는 낱말 곳간 크기와 차례 길이의 균형을 잡는 아래낱말 단위로 글을 나누는 법을 배운다. GPT-2가 쓰는 바이트 수준 BPE는 바이트 수준 글자 256개에서 시작해 가장 잦은 짝을 거듭 어울려, 모르는 토막 없이 어떤 글이든 온전히 덮는다. 낱말 곳간 크기는 핵심 웃매개변수이다. 곧 곳간이 클수록 차례는 짧아지지만 묻힘 행렬이 커진다.

ConstantLengthDataset은 인과 말 모델 익히기에 꼭 필요한 다듬기이다. 글월마다 붙박이 길이까지 덧대는(덧대는 토막에 셈을 낭비하는) 대신, 토막낸 글을 모두 EOS 나눔표로 이어 붙인 뒤 붙박이 길이의 차례로 잘라 낸다. 묶음마다 모든 토막이 익힘 신호에 보태져 GPU 씀씀이를 가장 크게 한다. 이 방식은 큰 규모의 미리 익히기에서 표준이며 막무가내 덧대기에 견주어 익히는 시간을 크게 줄일 수 있다.

익히기 되풀이에는 큰 규모 익히기를 든든하게 하는 재주가 여럿 들어 있다. 곧 기울기 쌓기는 GPU 기억 공간이 허락하는 것보다 큰 실효 묶음 크기를 가능하게 하고, 배움 비율 몸풀기는 익히기 초반의 흔들림을 막으며, 코사인 줄이기는 매끄러운 담금질을 주고, 치우침과 LayerNorm 매개변수에서 무게 줄이기를 빼는 것은 BERT와 GPT-2 익히기에서 자리 잡은 좋은 버릇을 따른다. GPU 하나를 넘어 키울 때는 HuggingFace Accelerate가 코드를 거의 고치지 않고 여러 GPU와 여러 마디 익히기를 감싸는 깔끔한 추상을 준다. 기울기 중간 저장, 섞인 정밀도 익히기, DeepSpeed ZeRO 단계 같은 기억 공간 아끼기 재주는 그러지 않으면 GPU 기억 공간을 넘어설 모델도 익힐 수 있게 한다.

## 연습문제

**연습문제 1.**
`ConstantLengthDataset` 클래스를 쓸 때, 글자 100만 개의 말뭉치에서 차례 길이 1024와 토막마다 평균 글자 3.6개를 쓰면 익힘 차례가 대략 몇 개 나오는지 셈하여라.

??? success "연습문제 1 풀이"
    온 토막 수는 $\approx 1{,}000{,}000 / 3.6 \approx 277{,}778$개다.
    
    온전한 이음의 수는 $= \lfloor 277{,}778 / 1024 \rfloor = 271$개다.
    
    이음마다 토막이 꼭 1024개이므로 자료 묶음은 남은 토막 $277{,}778 - 271 \times 1024 = 277{,}778 - 277{,}504 = 274$개를 버린다.

---

**연습문제 2.**
GPT-2를 익힐 때 치우침 매개변수와 LayerNorm 무게에서 무게 줄이기를 빼야 하는 까닭을 밝혀라. 모든 매개변수에 무게 줄이기를 한결같이 쓰면 어떻게 되겠는가?

??? success "연습문제 2 풀이"
    짐 줄이기(L2 다독임)는 잃음에 $\lambda \|w\|^2$을 더해 큰 매개변수 값에 벌을 준다. 치우침 마디와 켜 잣대 잡기 매개변수에는 이 다독임이 도리어 해롭다.
    
    - **치우침 항**은 깨어남을 옮기는 몫이므로 0 쪽으로 묶어서는 안 된다. 그러면 층마다 알맞은 어긋남을 배우는 힘이 줄어든다.
    - **LayerNorm 무게**(잣수 매개변수)는 1.0으로 첫자리매김되며 고르게 맞춘 깨어남의 잣수를 다스린다. 이를 0 쪽으로 줄이면 사실상 층의 내놓음을 눌러 익히기를 흔들리게 한다.
    
    무게 줄이기를 한결같이 쓰면 치우침이 0 쪽으로 밀리고 LayerNorm이 깨어남의 잣수를 제대로 다시 맞추지 못해 모델이 덜 배우게 된다. 표준 버릇은 매개변수를 가장 좋게 하개 묶음 둘로 가른다. 곧 무게 줄이기를 쓰는 쪽(빽빽한 무게 행렬)과 쓰지 않는 쪽(치우침과 고르게 맞추기 매개변수)이다.

---

**연습문제 3.**
처음 `warmup_steps` 걸음 동안 한 줄로 몸풀기를 하고 그 뒤 `total_steps`에 걸쳐 코사인으로 0까지 줄이는 단순한 배움 비율 일정 함수를 짜라. 걸음 `t`에서의 배움 비율 곱셈수를 돌려주어라.

??? success "연습문제 3 풀이"
    ```python
    import math
    
    def lr_schedule(t, warmup_steps, total_steps):
        """걸음 t의 배움 비율 곱셈수 셈하기.
        
        인수:
            t: 지금 걸음
            warmup_steps: 몸풀기 걸음의 개수
            total_steps: 전체 학습 단계 수
            
        반환값:
            [0, 1] 안의 배움 비율 곱셈수
        """
        if t < warmup_steps:
            # 한 줄 몸풀기: 0에서 1로 잣수 맞추기
            return t / warmup_steps
        else:
            # 1에서 0으로 코사인 줄이기
            progress = (t - warmup_steps) / (total_steps - warmup_steps)
            return 0.5 * (1.0 + math.cos(math.pi * progress))
    
    # 쓰는 보기:
    base_lr = 5e-4
    for step in [0, 1000, 2000, 50000, 150000]:
        multiplier = lr_schedule(step, warmup_steps=2000, total_steps=150000)
        print(f"Step {step:>6d}: lr = {base_lr * multiplier:.6f}")
    ```
