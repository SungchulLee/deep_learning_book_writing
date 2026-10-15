# GPT 모델을 처음부터 사전 학습하기

처음부터 사전 학습한다는 것은 사전 학습 가중치를 가져오지 않고 새 매개변수에서 언어 모델 학습을 시작한다는 뜻이다. 이 페이지는 BPE 토크나이저 학습, 설정 기반 모델 생성, 고정 길이 데이터 구성의 일부를 보여 준다. 뒤의 기울기 누적·평가·분산 학습 부분은 설명을 출력할 뿐 실행 가능한 전체 학습 파이프라인을 제공하지 않는다.

## 1. 코드와 기존 출력

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
    lr_scheduler:                 데우기를 곁들인 코사인
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

**기존 출력:**

```
============================================================
Part 1: Training a BPE Tokenizer from Scratch
============================================================
  tokenizers library not available — showing concept only

  HuggingFace approach (recommended for production):

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

============================================================
Part 2: Model Initialization from Config
============================================================
  transformers library not available

============================================================
Part 3: ConstantLengthDataset
============================================================
  ConstantLengthDataset concatenates documents and chunks:

  Doc1: 'Hello world'  →  [101, 7592, 2088, 102]
  Doc2: 'Deep learning' →  [101, 2784, 4083, 102]
  Concatenated: [101, 7592, 2088, 102, 0, 101, 2784, 4083, 102, 0]
  Chunked (seq_len=5): [[101,7592,2088,102,0], [101,2784,4083,102,0]]

============================================================
Part 4: Training Loop with Gradient Accumulation
============================================================

  익히기 자리매김(GPT-2 Small 보기):
    batch_size:                   12
    gradient_accumulation_steps:  1
    effective_batch_size:         12 * num_gpus
    learning_rate:                5e-4
    lr_scheduler:                 데우기를 곁들인 코사인
    warmup_steps:                 2000
    max_train_steps:              150000
    weight_decay:                 0.1
    seq_length:                   1024
    gradient_checkpointing:       True

============================================================
Part 5: Evaluation — Perplexity & Generation
============================================================

  헷갈림도 — 말 모델의 표준 잣대
    PPL = exp(평균 엇갈린 엔트로피 손실)
    헷갈림도가 낮다 = 모델이 남겨 둔 글에 더 높은 확률을 준다

============================================================
Part 6: Scaling — Distributed Training with Accelerate
============================================================

  실제로 미리 익힐 때는 여러 GPU를 쓰려 HuggingFace Accelerate를 쓴다.
  이렇게 띄운다: accelerate launch --num_processes 4 train.py

Done.
```

## 2. 토크나이저와 모델의 대응

첫 부분은 작은 예시 문장을 반복해 BPE 병합을 학습한다. 바이트 수준 전처리와 디코더를 사용하지만 모든 바이트 문자를 학습 초기 alphabet에 강제로 포함하지는 않는다. 따라서 이 설정을 임의 문자에 대해 미등록 토큰 없이 복원 가능한 완전한 바이트 어휘라고 단정하지 않는다. 반복된 소수 문장은 큰 실제 말뭉치의 언어 분포를 대신하지 못한다.

토크나이저의 실제 어휘 크기·특수 토큰 ID를 모델 설정과 맞춰야 한다. 코드의 config는 토크나이저가 있으면 어휘 크기를 고정 500으로 지정하며 실제 크기를 조회하지 않는다. 기존 GPT-2 설정의 EOS·BOS ID도 새 토크나이저의 ID와 자동으로 일치하지 않는다. 토크나이저를 새로 학습한 경우 이전 모델의 임베딩·특수 ID를 그대로 사용할 수 없다.

AutoConfig.from_pretrained는 GPT-2 설정을 가져오지만 AutoModelForCausalLM.from_config는 새 가중치를 초기화한다. 설정을 불러오는 것과 사전 학습 가중치를 불러오는 것은 다르다. 기본 설정 취득에 외부 접근이 필요할 수도 있으며 이 편집에서는 실행하지 않는다.

## 3. 고정 길이 청크와 문서 경계

ConstantLengthDataset은 문서들을 토큰화하고 각 문서 뒤에 지정한 EOS ID를 붙여 seq_length 길이로 나눈다. 호출 가능한 토크나이저가 input_ids의 목록을 반환한다고 가정한다. 앞서 만든 저수준 Tokenizer 객체를 이 인터페이스에 그대로 연결하는 예는 없으며 데이터셋 자체도 본문에서 실제 순회하지 않는다.

각 문자 버퍼의 마지막 짧은 청크는 버리고 다음 버퍼로 이월하지 않는다. 따라서 말뭉치 전체에서 마지막 나머지만 버리는 구현과 다르다. chars_per_token은 버퍼 크기의 추정에 쓰일 뿐 실제 토큰 수를 보장하지 않는다. seq_length와 버퍼 크기는 양수여야 하며, 다중 worker·분산 프로세스가 각각 같은 texts를 순회하면 데이터가 중복되지 않도록 분할해야 한다.

EOS를 사이에 넣어도 문서 간 어텐션이 자동으로 차단되지는 않는다. 청크 내부 다음 토큰 예측에는 이전 문서의 문맥이 사용될 수 있고, 청크 경계에서는 문맥이 끊긴다. 모델 내부에서 labels를 이동하면 청크의 첫 토큰은 해당 청크의 목표로 예측되지 않는다. 모든 저장 토큰이 동일하게 손실에 기여한다고 설명하지 않는다.

## 4. 실제 학습에 필요한 구성

학습에는 청크의 input_ids·labels, 유효 목표 집계, 옵티마이저, 역전파·갱신, 검증과 재개 가능한 체크포인트가 더 필요하다. 현재 코드의 Part 4~6은 설정·PPL·Accelerate 실행 문구를 출력하며 이 계산을 구현하지 않는다. 기존 출력은 사전 학습 완료나 검증 PPL을 입증하지 않는다.

기울기 누적의 실효 배치는 보통 장치당 microbatch×누적 횟수×장치 수이며 손실 스케일과 마지막 불완전 누적을 일관되게 처리해야 한다. warmup과 코사인 스케줄은 옵티마이저 갱신 횟수를 기준으로 맞춘다. 편향·정규화 가중치의 weight decay 제외는 흔한 선택이지만 모든 설정에서 필수인 법칙은 아니다. AdamW의 분리된 weight decay와 적응형 옵티마이저 손실에 L2 항을 더하는 방식도 일반적으로 같지 않다.

분산 학습·혼합 정밀도·활성화 체크포인팅은 별도 실행 구성이 필요하다. 도구 이름을 출력하는 것만으로 구현되거나 메모리·속도 향상이 검증되지는 않는다. 실제 자료 분리·토큰화·문맥 규칙을 고정한 검증·시험 평가로 학습 상태를 판단한다.

## 연습문제

<div class="drillbox" markdown>

**연습문제 1.** <span class="diff med" title="중간"></span>
문자 100만 개, 평균 문자/토큰 비율 3.6, 청크 길이 1024일 때 단일 토큰 스트림의 청크 수를 추정하고 실제 데이터셋과의 차이를 설명하라.

</div>

??? success "연습문제 1 풀이"

    EOS 추가를 무시하고 총 토큰을 약 277,778개로 가정하면 완전한 청크는 floor(277,778/1024)=271개, 나머지는 274개다. 이는 추정치이며 실제 토큰화·문서별 EOS 수에 따라 달라진다. 현재 구현은 버퍼마다 나머지를 버리므로 여러 버퍼에서는 전체를 한 번 연결한 계산과 청크 수가 다를 수 있다.



---

<div class="drillbox" markdown>

**연습문제 2.** <span class="diff med" title="중간"></span>
편향·LayerNorm을 weight decay에서 제외하는 이유와 이를 반드시 적용해야 한다고 단정할 수 없는 이유를 설명하라.

</div>

??? success "연습문제 2 풀이"

    편향은 위치 이동, LayerNorm의 가중치는 정규화 뒤 스케일을 제어하므로 큰 행렬 가중치와 다른 정규화 정책을 쓸 수 있다. 이들을 제외하는 것은 널리 쓰이는 설정 선택이며 모두 포함하면 언제나 학습이 실패한다는 뜻은 아니다. AdamW의 분리된 감소와 손실의 L2 페널티도 구분하고 동일 자료·검증 기준에서 정책을 비교한다. 현재 본문에는 실제 옵티마이저 그룹 구성이 없다.



---

<div class="drillbox" markdown>

**연습문제 3.** <span class="diff hard" title="어려움"></span>
아래 학습률 배율의 시간 구간과 total_steps의 의미, 범위를 벗어난 입력의 문제를 설명하라.

</div>

??? success "연습문제 3 풀이"

    0≤t<warmup_steps에서 선형 증가하고 warmup_steps에서 1, total_steps에서 0이 된다. total_steps는 warmup 이후의 추가 길이가 아니라 warmup을 포함한 전체 종료 시점이다. 0<warmup_steps<total_steps를 가정하며 t가 종료를 넘으면 코사인 때문에 다시 증가할 수 있으므로 실제 스케줄은 유효 구간이나 끝점 고정 정책을 명시해야 한다. 아래는 배율 계산 예이며 본문 학습 루프에 연결되어 있지 않다.

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


## 정리하며

새 가중치의 모델 생성과 사전 학습 완료는 다르다. 토크나이저·특수 ID·청크 경계와 손실 목표를 맞추고 실제 갱신·검증·분산 자료 분할까지 구성해야 완전한 학습 파이프라인이 된다.
