# 텍스트 전처리

텍스트를 소문자·공백 기준으로 토큰화하고, 단어를 정수 ID로 바꾼 뒤 패딩과 임베딩을 적용하는 예제다. 전처리 규칙과 특수 토큰의 역할을 이해하는 데 초점을 맞춘다. 여기서는 임베딩을 학습하거나 RNN을 실행하지 않는다.

## 1. 코드

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

# --- rnn_utils 에서 이 쪽이 쓰는 것만 ---
class Vocabulary:
    """텍스트 데이터의 어휘를 만들고 관리하기"""
    def __init__(self):
        self.word2idx = {'<PAD>': 0, '<UNK>': 1, '<SOS>': 2, '<EOS>': 3}
        self.idx2word = {0: '<PAD>', 1: '<UNK>', 2: '<SOS>', 3: '<EOS>'}
        self.word_count = {}
        self.n_words = 4
    
    def add_sentence(self, sentence):
        for word in sentence.split():
            self.add_word(word)
    
    def add_word(self, word):
        if word not in self.word2idx:
            self.word2idx[word] = self.n_words
            self.idx2word[self.n_words] = word
            self.word_count[word] = 1
            self.n_words += 1
        else:
            self.word_count[word] += 1
    
    def __len__(self):
        return self.n_words


def tokenize(text):
    """간단한 토큰 나누개"""
    return text.lower().split()


def text_to_sequence(text, vocab, max_length=None):
    """텍스트를 색인의 순차열로 바꾸기"""
    tokens = tokenize(text)
    sequence = [vocab.word2idx.get(word, vocab.word2idx['<UNK>']) for word in tokens]
    
    if max_length:
        if len(sequence) < max_length:
            sequence += [vocab.word2idx['<PAD>']] * (max_length - len(sequence))
        else:
            sequence = sequence[:max_length]
    
    return sequence

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
vocab = Vocabulary()
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
    seq = text_to_sequence(text, vocab)
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
    seq = text_to_sequence(text, vocab, max_length=max_len)
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
   • 임베딩 레이어: 학습 가능한 낱말 벡터
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

??? note "전체 출력 (154줄)"

    ```
    ======================================================================
    Text Preprocessing for RNNs
    ======================================================================

    ======================================================================
    SECTION 1: Tokenization
    ======================================================================

    Original texts:
      1. 'I love machine learning'
      2. 'Deep learning is amazing'
      3. 'PyTorch makes deep learning easy'
      4. 'I love PyTorch'
      5. 'Machine learning is the future'

    Tokenized (split into words):
      1. ['i', 'love', 'machine', 'learning']
      2. ['deep', 'learning', 'is', 'amazing']
      3. ['pytorch', 'makes', 'deep', 'learning', 'easy']
      4. ['i', 'love', 'pytorch']
      5. ['machine', 'learning', 'is', 'the', 'future']

    ======================================================================
    SECTION 2: Building Vocabulary
    ======================================================================

    Vocabulary size: 16 words

    Special tokens:
      <PAD>: 0 (for padding)
      <UNK>: 1 (unknown words)
      <SOS>: 2 (start of sequence)
      <EOS>: 3 (end of sequence)

    Word to Index mapping:
      '<PAD>' → 0
      '<UNK>' → 1
      '<SOS>' → 2
      '<EOS>' → 3
      'i' → 4
      'love' → 5
      'machine' → 6
      'learning' → 7
      'deep' → 8
      'is' → 9
      'amazing' → 10
      'pytorch' → 11
      'makes' → 12
      'easy' → 13
      'the' → 14

    ======================================================================
    SECTION 3: Converting Text to Number Sequences
    ======================================================================

    Converting texts to sequences of indices:

    'I love machine learning'
      → [4, 5, 6, 7]
      Words: ['i', 'love', 'machine', 'learning']

    'Deep learning is amazing'
      → [8, 7, 9, 10]
      Words: ['deep', 'learning', 'is', 'amazing']

    'PyTorch makes deep learning easy'
      → [11, 12, 8, 7, 13]
      Words: ['pytorch', 'makes', 'deep', 'learning', 'easy']

    'I love PyTorch'
      → [4, 5, 11]
      Words: ['i', 'love', 'pytorch']

    'Machine learning is the future'
      → [6, 7, 9, 14, 15]
      Words: ['machine', 'learning', 'is', 'the', 'future']

    ======================================================================
    SECTION 4: Padding Sequences
    ======================================================================

    Padding all sequences to length 6:

    'I love machine learning'
      Padded: [4, 5, 6, 7, 0, 0]

    'Deep learning is amazing'
      Padded: [8, 7, 9, 10, 0, 0]

    'PyTorch makes deep learning easy'
      Padded: [11, 12, 8, 7, 13, 0]

    'I love PyTorch'
      Padded: [4, 5, 11, 0, 0, 0]

    'Machine learning is the future'
      Padded: [6, 7, 9, 14, 15, 0]

    Final tensor shape: torch.Size([5, 6])
      (batch_size=5, sequence_length=6)

    ======================================================================
    SECTION 5: Word Embeddings
    ======================================================================

    낱말 임베딩은 낱말의 색인을 조밀한 벡터로 바꾼다.
      • 낱말마다 → 수의 벡터 (100차원 따위)
      • 비슷한 낱말은 벡터도 비슷하다
      • 학습 중에 배운다!

    Example:
      'king' → [0.2, -0.5, 0.8, …]
      'queen' → [0.3, -0.4, 0.7, …]  (비슷하다!)
      'car' → [-0.8, 0.3, -0.2, …]  (다르다)


    Embedding layer:
      Vocabulary size: 16
      Embedding dimension: 50
      Total parameters: 800

    After embedding:
      Input shape: torch.Size([5, 6])
      Output shape: torch.Size([5, 6, 50])
      (batch, seq_len, embedding_dim)

    ======================================================================
    SUMMARY
    ======================================================================

    ✅ 텍스트 전처리 흐름:
       1. 토큰화: 텍스트 → 낱말
       2. 어휘: 낱말 → 색인
       3. 순차열: 텍스트 → 색인의 목록
       4. 덧대기: 다양한 길이 → 고정된 길이
       5. 임베딩: 색인 → 조밀한 벡터

    ✅ 핵심 부품:
       • 어휘: 낱말과 색인을 잇는다
       • 특수 토큰: <PAD>, <UNK>, <SOS>, <EOS>
       • 임베딩 레이어: 학습 가능한 낱말 벡터
       • 덧대기: 다양한 길이를 다룬다

    ✅ 다음: 03_time_series_basics.py
       그다음: 04_simple_rnn.py (첫 RNN 만들기!)

    흔한 어휘 크기:
      • 작은 데이터셋: 낱말 5,000~10,000개
      • 중간: 낱말 20,000~50,000개
      • 큰 것: 낱말 100,000개 이상 (GPT는 50,257개를 쓴다)

    ======================================================================
    Tutorial Complete! ✓
    ======================================================================
    ```


## 2. 논의

`Vocabulary`는 PAD, UNK, SOS, EOS의 네 특수 토큰에 ID 0~3을 부여하고, 처음 등장한 순서대로 단어를 추가한다. 예제의 실제 단어는 12종류이므로 총 어휘 크기는 16이다. 소문자 변환은 대소문자 차이를 없애고, 공백 분리는 구두점과 형태소를 따로 처리하지 않는다. 한국어 등 다른 언어에도 같은 방식이 충분하다고 가정할 수 없다.

`text_to_sequence`는 알 수 없는 단어를 UNK로 바꾸고, 양수 max_length가 주어지면 짧은 문장에는 PAD를 붙이며 긴 문장은 뒤를 자른다. SOS와 EOS는 사전에 있을 뿐 이 함수가 자동으로 추가하지는 않는다. 0은 조건문에서 거짓으로 취급되므로 `max_length=0`은 빈 시퀀스로 자르는 설정이 아니다. 음수 길이는 의도하지 않은 슬라이싱이 될 수 있어 별도 검사가 필요하다.

본문의 다섯 문장을 길이 6으로 맞추면 ID 텐서는 $(5,6)$이고, 50차원 임베딩 출력은 $(5,6,50)$이다. 임베딩 파라메터는 $16\times50=800$개다. 로스와 옵티마이저 갱신이 없으므로 출력 벡터는 무작위 초기값이며 의미 유사도를 학습하지 않았다.

`nn.Embedding`에 `padding_idx`를 지정하지 않았으므로 PAD도 일반 행처럼 조회된다. PAD가 0이라는 ID 규칙만으로 출력 벡터가 0이 되거나 이후 RNN에서 무시되는 것은 아니다. 실제 모델에서는 길이와 마스크, 로스의 패딩 제외를 함께 설계한다. 단어 사전과 전처리 규칙은 학습 데이터로 정하고 검증·테스트에도 같은 것을 적용한다.

코드의 특정 GPT 어휘 크기는 모든 GPT 모델에 공통인 값이 아니다. 현대 언어 모델의 토큰은 단어와 일대일로 대응하지 않을 수 있으므로 단어 수와 토큰 어휘 크기도 구분해야 한다.

## 연습문제

<div class="drillbox" markdown>

**연습문제 1.** <span class="diff med" title="중간"></span>
이 예제의 어휘 크기와 임베딩 출력 형태·파라메터 수를 구하라.

</div>

??? success "풀이"
    일반 단어 12개와 특수 토큰 4개로 $V=16$이다. 길이 6의 문장 5개를 50차원으로 임베딩하므로 출력은 $(5,6,50)$이고 파라메터는 800개다. 이 벡터들은 학습된 의미 표현이 아니다.

---

<div class="drillbox" markdown>

**연습문제 2.** <span class="diff med" title="중간"></span>
text_to_sequence가 처리하는 OOV, 길이 제한과 문장 시작·끝 토큰을 설명하라.

</div>

??? success "풀이"
    어휘 밖 단어는 UNK로 바뀐다. 양수 길이 제한에서 짧으면 PAD를 붙이고 길면 자른다. SOS와 EOS는 자동으로 넣지 않는다. 제한 0과 음수는 의도한 처리인지 별도 검증해야 하며, 잘린 문장에서 중요한 정보나 EOS가 사라질 수 있다는 점도 고려한다.

---

<div class="drillbox" markdown>

**연습문제 3.** <span class="diff med" title="중간"></span>
패딩된 문장을 RNN으로 분류할 때 마지막 배열 위치의 출력만 쓰면 왜 문제가 될 수 있는가?

</div>

??? success "풀이"
    짧은 문장에서는 그 위치가 PAD에 해당하며, RNN은 PAD에서도 상태를 갱신할 수 있다. 실제 마지막 토큰의 상태를 선택하거나 길이를 반영한 packed sequence 등을 사용한다. 입력·출력 마스크와 로스의 패딩 제외도 과제에 맞춰 처리한다.

---

<div class="drillbox" markdown>

**연습문제 4.** <span class="diff hard" title="어려움"></span>
아래의 테스트 코드 조각을 검토하라. 정의되지 않은 클래스와 잘못된 이름, 입력 모양 가정을 찾아 이 페이지의 실제 함수에 맞는 점검 항목을 제안하라.

</div>

??? success "풀이"
    아래 조각은 실제 실행 가능한 테스트가 아니다. 함수 이름과 모델 이름부터 이 예제에 맞게 작성해야 한다.
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
    이 페이지는 완성된 사용자 정의 모델 클래스를 제공하지 않는다. 실제 함수의 입력·출력 모양, 허용 자료형, 유한한 값과 경사의 유무를 기준으로 검사한다. 빈 입력을 처리하는 방식은 연산마다 다르며, 무조건 정상적인 로스나 경사가 나와야 한다고 가정하지 않는다.

## 정리하며

토큰화, 단어 사전, 정수 ID, 패딩과 임베딩은 서로 다른 단계다. 특수 토큰을 등록한 것과 실제 삽입·무시 처리를 한 것을 구분한다. 학습과 추론에서 같은 사전과 전처리를 유지해야 한다.
