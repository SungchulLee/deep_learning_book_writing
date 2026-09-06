# 텍스트 전처리

02_text_preprocessing.py RNN을 위한 텍스트 전처리

순차열 모형은 시간적이고 순서가 있는 데이터를 다루는 데 바탕이 된다. 이 구현은 순환 신경망의 핵심 착상을 다루며, 순환 계산과 학습된 표현이 시각 사이의 의존을 어떻게 붙잡는지 보인다.

## 코드

```python
"""
02_text_preprocessing.py
=========================
RNN을 위한 텍스트 전처리

날 텍스트를 RNN이 다룰 수 있는 수의 형태로 바꾸는 법을 익힌다.
어떤 자연어 처리 과제에서든 꼭 필요한 기술이다!

주제: 토큰화, 어휘, 낱말 임베딩, 덧대기
난이도: 쉬움
시간: 30~45분
"""

import torch
import torch.nn as nn
import numpy as np
from collections import Counter
import rnn_utils

print("=" * 70)
print("Text Preprocessing for RNNs")
print("=" * 70)

# 예제 텍스트
texts = [
    "I love machine learning",
    "Deep learning is amazing",
    "PyTorch makes deep learning easy",
    "I love PyTorch",
    "Machine learning is the future"
]

print("\n" + "=" * 70)
print("SECTION 1: Tokenization")
print("=" * 70)

print("\nOriginal texts:")
for i, text in enumerate(texts, 1):
    print(f"  {i}. '{text}'")

# 토큰으로 나누기 (낱말로 쪼개기)
tokenized = [text.lower().split() for text in texts]

print("\nTokenized (split into words):")
for i, tokens in enumerate(tokenized, 1):
    print(f"  {i}. {tokens}")

# =============================================================================
# 2절: 어휘 만들기
# =============================================================================

print("\n" + "=" * 70)
print("SECTION 2: Building Vocabulary")
print("=" * 70)

# 어휘 만들기
vocab = rnn_utils.Vocabulary()
for text in texts:
    vocab.add_sentence(text.lower())

print(f"\nVocabulary size: {len(vocab)} words")
print(f"\nSpecial tokens:")
print(f"  <PAD>: {vocab.word2idx['<PAD>']} (for padding)")
print(f"  <UNK>: {vocab.word2idx['<UNK>']} (unknown words)")
print(f"  <SOS>: {vocab.word2idx['<SOS>']} (start of sequence)")
print(f"  <EOS>: {vocab.word2idx['<EOS>']} (end of sequence)")

print(f"\nWord to Index mapping:")
for word, idx in sorted(vocab.word2idx.items(), key=lambda x: x[1])[:15]:
    print(f"  '{word}' → {idx}")

# =============================================================================
# 3절: 텍스트를 순차열로
# =============================================================================

print("\n" + "=" * 70)
print("SECTION 3: Converting Text to Number Sequences")
print("=" * 70)

print("\nConverting texts to sequences of indices:")
sequences = []
for text in texts:
    seq = rnn_utils.text_to_sequence(text, vocab)
    sequences.append(seq)
    print(f"\n'{text}'")
    print(f"  → {seq}")
    print(f"  Words: {[vocab.idx2word[idx] for idx in seq]}")

# =============================================================================
# 4절: 덧대기
# =============================================================================

print("\n" + "=" * 70)
print("SECTION 4: Padding Sequences")
print("=" * 70)

max_len = 6
padded_sequences = []

print(f"\nPadding all sequences to length {max_len}:")
for text in texts:
    seq = rnn_utils.text_to_sequence(text, vocab, max_length=max_len)
    padded_sequences.append(seq)
    print(f"\n'{text}'")
    print(f"  Padded: {seq}")

# 텐서로 바꾸기
seq_tensor = torch.tensor(padded_sequences)
print(f"\nFinal tensor shape: {seq_tensor.shape}")
print(f"  (batch_size={len(texts)}, sequence_length={max_len})")

# =============================================================================
# 5절: 임베딩
# =============================================================================

print("\n" + "=" * 70)
print("SECTION 5: Word Embeddings")
print("=" * 70)

print("""
낱말 임베딩은 낱말의 색인을 조밀한 벡터로 바꾼다.
  • 낱말마다 → 수의 벡터 (100차원 따위)
  • 비슷한 낱말은 벡터도 비슷하다
  • 학습 중에 배운다!

Example:
  'king' → [0.2, -0.5, 0.8, …]
  'queen' → [0.3, -0.4, 0.7, …]  (비슷하다!)
  'car' → [-0.8, 0.3, -0.2, …]  (다르다)
""")

embedding_dim = 50
embedding = nn.Embedding(len(vocab), embedding_dim)

print(f"\nEmbedding layer:")
print(f"  Vocabulary size: {len(vocab)}")
print(f"  Embedding dimension: {embedding_dim}")
print(f"  Total parameters: {len(vocab) * embedding_dim:,}")

# 순차열 임베딩하기
embedded = embedding(seq_tensor)
print(f"\nAfter embedding:")
print(f"  Input shape: {seq_tensor.shape}")
print(f"  Output shape: {embedded.shape}")
print(f"  (batch, seq_len, embedding_dim)")

# =============================================================================
# 요약
# =============================================================================

print("\n" + "=" * 70)
print("SUMMARY")
print("=" * 70)

print("""
✅ 텍스트 전처리 흐름:
   1. 토큰화: 텍스트 → 낱말
   2. 어휘: 낱말 → 색인
   3. 순차열: 텍스트 → 색인의 목록
   4. 덧대기: 다양한 길이 → 고정된 길이
   5. 임베딩: 색인 → 조밀한 벡터

✅ 핵심 부품:
   • 어휘: 낱말과 색인을 잇는다
   • 특수 토큰: <PAD>, <UNK>, <SOS>, <EOS>
   • 임베딩 층: 학습 가능한 낱말 벡터
   • 덧대기: 다양한 길이를 다룬다

✅ 다음: 03_time_series_basics.py
   그다음: 04_simple_rnn.py (첫 RNN 만들기!)

흔한 어휘 크기:
  • 작은 데이터셋: 낱말 5,000~10,000개
  • 중간: 낱말 20,000~50,000개
  • 큰 것: 낱말 100,000개 이상 (GPT는 50,257개를 쓴다)
""")

print("=" * 70)
print("Tutorial Complete! ✓")
print("=" * 70)


if __name__ == "__main__":
    pass
```

## 논의

이 구현은 깔끔하고 읽기 좋은 PyTorch 코드로 순환 신경망의 핵심 개념을 보인다. 모듈식 짜임 덕분에 부품 하나하나를 살펴보고 다른 과제나 데이터셋에 맞추어 고치기 쉽다.

여기서 보인 방식은 더 복잡한 상황으로 자연스럽게 넓혀진다. 초매개변수와 구조의 변형, 다른 데이터셋으로 실험해 보면 이해가 깊어지고 순차열 모형 과제에 대한 실용적인 직관이 쌓인다.

## 연습문제

**연습문제 1.**
코드를 훑으며 핵심 설계 결정을 찾아라. 구체적인 구현 선택 세 가지를 열거하고 각각이 순환 신경망에 알맞은 까닭을 설명하라.

??? success "연습문제 1 풀이"
    설계 결정은 구현마다 다르지만 흔히 다음이 포함된다. (1) 활성화 함수의 선택 — ReLU 계열은 포화되지 않는 경사를 주어 학습을 빠르게 한다. (2) 정규화 전략 — 배치 정규화는 내부 공변량 이동을 줄여 학습을 안정시킨다. (3) 잔차 연결 — 있을 경우 건너뛰는 경로를 제공하여 깊은 신경망에서도 경사가 흐르게 한다. 각 선택은 표현력, 계산 비용, 학습 안정성 사이의 절충을 반영한다.

---

**연습문제 2.**
입력이 기대하는 모양과 자료형을 갖는지 확인하도록 주 함수나 클래스에 입력 검증을 추가하라. 잘못된 입력에는 유익한 오류 메시지를 내라.

??? success "연습문제 2 풀이"
    `forward` 메서드(또는 해당 함수)의 첫머리에 다음과 같은 검사를 추가한다. `assert x.dim() == expected_dims, f'Expected {expected_dims}D input, got {x.dim()}D'`와 `assert x.dtype == torch.float32, f'Expected float32, got {x.dtype}'`. 모양을 검증할 때는 중요한 차원을 확인한다. `B, C, H, W = x.shape; assert C == self.expected_channels`. 유익한 오류 메시지는 디버깅 속도를 크게 높이고 코드를 재사용하기에도 더 견고하게 만든다.

---

**연습문제 3.**
이 구현에서 생길 수 있는 실패 양상 두 가지를 서술하고, 각각을 어떻게 진단하고 고칠지 설명하라.

??? success "연습문제 3 풀이"
    흔한 실패 양상은 다음과 같다. (1) **경사 소실/폭발** — 경사의 노름을 지켜보아 진단한다(`torch.nn.utils.clip_grad_norm_`을 쓰거나 층마다 `param.grad.norm()`을 기록한다). 경사 자르기, 더 나은 초기화(Xavier/Kaiming), 또는 구조 변경(잔차 연결, 정규화)으로 고친다. (2) **과적합** — 학습 손실은 줄어드는데 검증 손실이 늘어나면 진단된다. 정칙화(드롭아웃, 가중치 감쇠, 데이터 증강)나 모델 용량 축소로 고친다. 이런 문제를 일찍 잡아내려면 언제나 학습 지표와 검증 지표를 함께 살펴라.

---

**연습문제 4.**
텍스트 전처리 구현을 검증하는 종합 시험 함수를 작성하라. 빈 입력, 원소가 하나뿐인 입력, 아주 큰 입력, 극단적인 값(0, 아주 큰 수)을 가진 입력 같은 경계 상황을 시험하라.

??? success "연습문제 4 풀이"
    경계 조건을 두루 시험하는 함수를 만든다.
    ```python
    def test_text preprocessing():
        model = Text Preprocessing(...)
        # 보통의 입력
        assert model(normal_input).shape == expected_shape
        # 원소가 하나인 배치
        assert model(single_input).shape == (1, ...)
        # 큰 값 (넘침을 확인한다)
        out = model(torch.ones(...) * 1000)
        assert torch.isfinite(out).all()
        # 경사의 흐름
        out = model(normal_input)
        out.sum().backward()
        for p in model.parameters():
            assert p.grad is not None
    ```
    경사의 흐름을 시험하는 것은 그 구조가 처음부터 끝까지 이어지는 학습을 지원하는지 확인하는 데 특히 중요하다.
