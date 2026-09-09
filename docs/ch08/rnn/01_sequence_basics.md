# 순차열의 기초

01_sequence_basics.py 순차열과 순차 데이터 이해하기

순차열 모형은 시간적이고 순서가 있는 데이터를 다루는 데 바탕이 된다. 이 구현은 순환 신경망의 핵심 착상을 다루며, 순환 계산과 학습된 표현이 시각 사이의 의존을 어떻게 붙잡는지 보인다.

## 1. 코드

```python
"""
01_sequence_basics.py
=====================
순차열과 순차 데이터 이해하기

이 실습은 순차열이라는 바탕 개념과, 그것이 보통의 데이터와 달리
왜 특별한 대접을 받아야 하는지 소개한다.

배울 내용:
- 데이터를 "순차적"으로 만드는 것
- 순차열을 텐서로 나타내는 법
- 시간적 의존과 순서
- 순차 데이터를 배치로 묶기
- 덧대기와 가림막

난이도: 쉬움
예상 시간: 30~45분
"""

import torch
import numpy as np
import matplotlib.pyplot as plt

print("=" * 70)
print("Understanding Sequences")
print("=" * 70)

# =============================================================================
# 1절: 순차열이란 무엇인가
# =============================================================================

print("\n" + "=" * 70)
print("SECTION 1: What is a Sequence?")
print("=" * 70)

print("""
순차열이란 다음을 만족하는, 순서가 있는 데이터의 모음이다.
1. 순서가 중요하다 (원소를 바꾸면 뜻이 달라진다)
2. 원소가 시간이나 자리로 이어져 있다
3. 지난 원소가 앞으로의 예측에 영향을 준다

예제:
---------
✓ 텍스트: "I love PyTorch"와 "PyTorch love I"
✓ 시계열: 주가 [100, 102, 105, 103]
✓ 음향: 소리의 파형
✓ 영상: 시간에 따른 프레임
✗ 이미지: (대개 순차적이지 않으며 CNN이 공간적으로 다룬다)
✗ 표 형태 데이터: (대개 순서가 없다)

보통의 데이터와의 핵심 차이:
---------------------------------
보통: 표본마다 서로 독립이다
순차: 원소마다 앞선 원소에 기댄다!
""")

# =============================================================================
# 2절: 간단한 순차열 예제
# =============================================================================

print("\n" + "=" * 70)
print("SECTION 2: Simple Sequence Examples")
print("=" * 70)

# 예 1: 수의 순차열
number_seq = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
print("\nExample 1: Number Sequence")
print(f"  Sequence: {number_seq}")
print(f"  Pattern: Each number = previous + 1")
print(f"  Next prediction: {number_seq[-1] + 1}")

# 예 2: 텍스트 순차열 (낱말)
text_seq = ["The", "cat", "sat", "on", "the", "mat"]
print("\nExample 2: Text Sequence")
print(f"  Sequence: {' '.join(text_seq)}")
print(f"  Order matters: Rearrange → different meaning!")
wrong_order = ["mat", "the", "cat", "The", "on", "sat"]
print(f"  Wrong order: {' '.join(wrong_order)} (nonsense!)")

# 예 3: 시계열 (기온)
temps = [72, 73, 75, 78, 82, 85, 87, 85, 80, 75]
hours = list(range(len(temps)))

plt.figure(figsize=(10, 4))
plt.plot(hours, temps, marker='o')
plt.xlabel('Hour')
plt.ylabel('Temperature (°F)')
plt.title('Temperature Over Time (Sequential Data)')
plt.grid(True)
plt.savefig('/home/claude/pytorch_rnn_tutorial/sequence_example.png', dpi=150, bbox_inches='tight')
print("\nExample 3: Time Series (Temperature)")
print(f"  Data: {temps}")
print(f"  Pattern: Temperature rises then falls (daily cycle)")
print("  Plot saved as 'sequence_example.png'")
plt.close()

# =============================================================================
# 3절: PyTorch 텐서로 나타낸 순차열
# =============================================================================

print("\n" + "=" * 70)
print("SECTION 3: Sequences as PyTorch Tensors")
print("=" * 70)

print("\nSequence Tensor Shapes:")
print("-" * 50)

# 순차열 하나
single_seq = torch.tensor([1, 2, 3, 4, 5])
print(f"\n1. Single Sequence:")
print(f"   Data: {single_seq}")
print(f"   Shape: {single_seq.shape}")
print(f"   Interpretation: (sequence_length,)")

# 특징이 있는 순차열 하나
single_seq_features = torch.randn(5, 3)  # 시각 5개, 시각마다 특징 3개
print(f"\n2. Single Sequence with Features:")
print(f"   Shape: {single_seq_features.shape}")
print(f"   Interpretation: (sequence_length, num_features)")
print(f"   Example: 5 timesteps, each with [x, y, z] coordinates")

# 순차열의 배치 (RNN에서 가장 흔하다)
batch_sequences = torch.randn(32, 10, 5)
print(f"\n3. Batch of Sequences (Standard RNN Input):")
print(f"   Shape: {batch_sequences.shape}")
print(f"   Interpretation: (batch_size, sequence_length, num_features)")
print(f"   Example: 32 sequences, each 10 timesteps, 5 features per step")

print("\nShape Convention for RNNs:")
print("  • batch_first=True:  (batch, seq_len, features)")
print("  • batch_first=False: (seq_len, batch, features)")
print("  • We'll use batch_first=True (more intuitive)")

# =============================================================================
# 4절: 길이가 다른 순차열
# =============================================================================

print("\n" + "=" * 70)
print("SECTION 4: Handling Variable Length Sequences")
print("=" * 70)

print("\nProblem: Real sequences have different lengths!")

# 길이가 서로 다른 예제 문장
sentences = [
    "I love AI",           # 낱말 3개
    "Deep learning is amazing",  # 낱말 4개
    "PyTorch makes it easy to build neural networks"  # 낱말 9개
]

print("\nExample Sentences:")
for i, sent in enumerate(sentences):
    words = sent.split()
    print(f"  {i+1}. '{sent}' → {len(words)} words")

print("\nSolution: PADDING")
print("-" * 50)

# <PAD> 토큰(색인 0)으로 덧대기 흉내 내기
max_length = 9
padded_sequences = []

for sent in sentences:
    words = sent.split()
    # 색인으로 나타내기 (간단히)
    indices = list(range(1, len(words) + 1))
    # max_length까지 덧대기
    padded = indices + [0] * (max_length - len(indices))
    padded_sequences.append(padded)

print("\nPadded Sequences:")
for i, (sent, padded) in enumerate(zip(sentences, padded_sequences)):
    print(f"  {i+1}. {padded}")
    print(f"      Real: {sent.split()}")
    print(f"      Padding: {[0] * (max_length - len(sent.split()))}")

# =============================================================================
# 5절: 시간적 의존
# =============================================================================

print("\n" + "=" * 70)
print("SECTION 5: Understanding Time Dependencies")
print("=" * 70)

print("""
왜 순차열에 RNN을 쓰는가?
----------------------
문제: 원소마다 앞선 원소에 기댄다

예: 다음 낱말 예측하기
  "The cat sat on the ___"
  
"mat"을 예측하려면 모델이 다음을 알아야 한다.
  • 앞 낱말: "the"
  • 문맥: "cat sat on"
  • 문법: 명사가 와야 한다
  
보통의 신경망:
  ✗ 낱말을 저마다 따로 처리한다
  ✗ 앞선 낱말의 기억이 없다
  ✗ 시간적인 무늬를 붙잡지 못한다

RNN (순환 신경망):
  ✓ 숨은 상태(기억)를 지닌다
  ✓ t-1에서 t로 정보를 넘긴다
  ✓ 시간적인 무늬를 배울 수 있다
  
수학으로 보면:
-----------------
보통 신경망:  y = f(x)
RNN:         y_t = f(x_t, h_{t-1})
             
여기서 h_{t-1}은 앞 시각의 숨은 상태이다!
""")

# 시간 의존을 간단히 그려 보기
sequence = [10, 12, 15, 19, 24, 30]
differences = [sequence[i] - sequence[i-1] for i in range(1, len(sequence))]

print(f"\nExample: Number Pattern")
print(f"Sequence: {sequence}")
print(f"Differences: {differences}")
print(f"Pattern: Each number increases by (prev_diff + 1)")
print(f"Next prediction: {sequence[-1] + differences[-1] + 1} = {sequence[-1] + differences[-1] + 1}")

# =============================================================================
# 6절: 학습 데이터 만들기
# =============================================================================

print("\n" + "=" * 70)
print("SECTION 6: Creating Training Data from Sequences")
print("=" * 70)

print("\nHow to create (input, target) pairs from sequences:")

# 예: 다음 수 예측하기
full_sequence = list(range(1, 11))
window_size = 3

print(f"\nFull sequence: {full_sequence}")
print(f"Window size: {window_size}")
print("\nTraining pairs (input → target):")
print("-" * 50)

for i in range(len(full_sequence) - window_size):
    input_seq = full_sequence[i:i+window_size]
    target = full_sequence[i+window_size]
    print(f"  {input_seq} → {target}")

print("\nThis is called 'sliding window' approach")
print("RNNs learn to predict next element given previous elements!")

# =============================================================================
# 요약
# =============================================================================

print("\n" + "=" * 70)
print("SUMMARY - Key Takeaways")
print("=" * 70)

print("""
✅ 배운 것:
   1. 순차열은 순서가 중요한, 순서 있는 데이터이다
   2. 텍스트, 시계열, 음향, 영상에서 흔하다
   3. 원소마다 앞선 원소에 기댄다
   4. 3차원 텐서 (배치, seq_len, 특징)로 나타낸다

✅ 핵심 개념:
   • 시간적 의존: x_t는 x_{t-1}, x_{t-2}, …에 기댄다
   • 길이가 다양하다: 배치로 묶으려면 덧대기가 필요하다
   • 미끄럼창: 학습 쌍을 만든다
   • 기억: RNN은 숨은 상태를 지닌다

✅ 다음 단계:
   → 02_text_preprocessing.py: 텍스트 토큰화 익히기
   → 03_time_series_basics.py: 시계열 준비하기
   → 04_simple_rnn.py: 첫 RNN 만들기!

중요한 구별:
----------------------
CNN:  공간적인 관계 (이미지)
RNN:  시간적인 관계 (순차열)
CNN:  입력 크기가 고정
RNN:  순차열의 길이가 다양 (덧대기와 함께)
CNN:  기억이 없다
RNN:  숨은 상태라는 기억

요령:
--------
RNN에 넣기 전에 순차열을 꼭 그려 보라!
데이터의 짜임을 이해하면 절반은 이긴 셈이다.
""")

print("=" * 70)
print("Tutorial Complete! ✓")
print("=" * 70)


if __name__ == "__main__":
    pass
```

## 2. 논의

시각화는 모델의 거동을 이해하고 학습 문제를 진단하는 데 중요한 역할을 한다. 그림을 그리는 코드는 학습된 표현, 수렴의 움직임, 평가 지표에 대한 통찰을 주어 추상적인 계산을 손에 잡히게 만든다.

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
순차열의 기초 구현을 검증하는 종합 시험 함수를 작성하라. 빈 입력, 원소가 하나뿐인 입력, 아주 큰 입력, 극단적인 값(0, 아주 큰 수)을 가진 입력 같은 경계 상황을 시험하라.

??? success "연습문제 4 풀이"
    경계 조건을 두루 시험하는 함수를 만든다.
    ```python
    def test_sequence basics():
        model = Sequence Basics(...)
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

## 정리하며

**다룬 것** — 순차열의 기초

시각화는 모델의 거동을 이해하고 학습 문제를 진단하는 데 중요한 역할을 한다.

앞의 연습문제 4개로 직접 확인할 수 있다.
