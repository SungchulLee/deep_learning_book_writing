# 시퀀스 기초

시퀀스는 순서가 있는 데이터다. 이 예제는 숫자열, 문장과 기온 데이터를 통해 순서의 역할을 살펴보고, 시퀀스를 텐서와 학습 쌍으로 표현한다. RNN을 정의하거나 학습하는 코드는 아니며 데이터 표현을 준비하는 단계다.

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
# 무작위로 뽑는 값이 아래에 나온다. 씨앗을 고정해야 이 쪽에 실린
# 수가 다시 나온다 — 고정하지 않으면 돌릴 때마다 다른 수가 찍힌다
torch.manual_seed(0)

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
plt.savefig('sequence_example.png', dpi=150, bbox_inches='tight')
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

# 피처가 있는 순차열 하나
single_seq_features = torch.randn(5, 3)  # 시각 5개, 시각마다 피처 3개
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
  
보통의 뉴럴 넷트웍:
  ✗ 낱말을 저마다 따로 처리한다
  ✗ 앞선 낱말의 기억이 없다
  ✗ 시간적인 무늬를 붙잡지 못한다

RNN (순환 뉴럴 넷트웍):
  ✓ 숨은 상태(기억)를 지닌다
  ✓ t-1에서 t로 정보를 넘긴다
  ✓ 시간적인 무늬를 배울 수 있다
  
수학으로 보면:
-----------------
보통 뉴럴 넷트웍:  y = f(x)
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
   4. 3차원 텐서 (배치, seq_len, 피처)로 나타낸다

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

??? note "전체 출력 (199줄)"

    ```
    ======================================================================
    Understanding Sequences
    ======================================================================

    ======================================================================
    SECTION 1: What is a Sequence?
    ======================================================================

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


    ======================================================================
    SECTION 2: Simple Sequence Examples
    ======================================================================

    Example 1: Number Sequence
      Sequence: [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
      Pattern: Each number = previous + 1
      Next prediction: 11

    Example 2: Text Sequence
      Sequence: The cat sat on the mat
      Order matters: Rearrange → different meaning!
      Wrong order: mat the cat The on sat (nonsense!)

    Example 3: Time Series (Temperature)
      Data: [72, 73, 75, 78, 82, 85, 87, 85, 80, 75]
      Pattern: Temperature rises then falls (daily cycle)
      Plot saved as 'sequence_example.png'

    ======================================================================
    SECTION 3: Sequences as PyTorch Tensors
    ======================================================================

    Sequence Tensor Shapes:
    --------------------------------------------------

    1. Single Sequence:
       Data: tensor([1, 2, 3, 4, 5])
       Shape: torch.Size([5])
       Interpretation: (sequence_length,)

    2. Single Sequence with Features:
       Shape: torch.Size([5, 3])
       Interpretation: (sequence_length, num_features)
       Example: 5 timesteps, each with [x, y, z] coordinates

    3. Batch of Sequences (Standard RNN Input):
       Shape: torch.Size([32, 10, 5])
       Interpretation: (batch_size, sequence_length, num_features)
       Example: 32 sequences, each 10 timesteps, 5 features per step

    Shape Convention for RNNs:
      • batch_first=True:  (batch, seq_len, features)
      • batch_first=False: (seq_len, batch, features)
      • We'll use batch_first=True (more intuitive)

    ======================================================================
    SECTION 4: Handling Variable Length Sequences
    ======================================================================

    Problem: Real sequences have different lengths!

    Example Sentences:
      1. 'I love AI' → 3 words
      2. 'Deep learning is amazing' → 4 words
      3. 'PyTorch makes it easy to build neural networks' → 8 words

    Solution: PADDING
    --------------------------------------------------

    Padded Sequences:
      1. [1, 2, 3, 0, 0, 0, 0, 0, 0]
          Real: ['I', 'love', 'AI']
          Padding: [0, 0, 0, 0, 0, 0]
      2. [1, 2, 3, 4, 0, 0, 0, 0, 0]
          Real: ['Deep', 'learning', 'is', 'amazing']
          Padding: [0, 0, 0, 0, 0]
      3. [1, 2, 3, 4, 5, 6, 7, 8, 0]
          Real: ['PyTorch', 'makes', 'it', 'easy', 'to', 'build', 'neural', 'networks']
          Padding: [0]

    ======================================================================
    SECTION 5: Understanding Time Dependencies
    ======================================================================

    왜 순차열에 RNN을 쓰는가?
    ----------------------
    문제: 원소마다 앞선 원소에 기댄다

    예: 다음 낱말 예측하기
      "The cat sat on the ___"
      
    "mat"을 예측하려면 모델이 다음을 알아야 한다.
      • 앞 낱말: "the"
      • 문맥: "cat sat on"
      • 문법: 명사가 와야 한다
      
    보통의 뉴럴 넷트웍:
      ✗ 낱말을 저마다 따로 처리한다
      ✗ 앞선 낱말의 기억이 없다
      ✗ 시간적인 무늬를 붙잡지 못한다

    RNN (순환 뉴럴 넷트웍):
      ✓ 숨은 상태(기억)를 지닌다
      ✓ t-1에서 t로 정보를 넘긴다
      ✓ 시간적인 무늬를 배울 수 있다
      
    수학으로 보면:
    -----------------
    보통 뉴럴 넷트웍:  y = f(x)
    RNN:         y_t = f(x_t, h_{t-1})
                 
    여기서 h_{t-1}은 앞 시각의 숨은 상태이다!


    Example: Number Pattern
    Sequence: [10, 12, 15, 19, 24, 30]
    Differences: [2, 3, 4, 5, 6]
    Pattern: Each number increases by (prev_diff + 1)
    Next prediction: 37 = 37

    ======================================================================
    SECTION 6: Creating Training Data from Sequences
    ======================================================================

    How to create (input, target) pairs from sequences:

    Full sequence: [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    Window size: 3

    Training pairs (input → target):
    --------------------------------------------------
      [1, 2, 3] → 4
      [2, 3, 4] → 5
      [3, 4, 5] → 6
      [4, 5, 6] → 7
      [5, 6, 7] → 8
      [6, 7, 8] → 9
      [7, 8, 9] → 10

    This is called 'sliding window' approach
    RNNs learn to predict next element given previous elements!

    ======================================================================
    SUMMARY - Key Takeaways
    ======================================================================

    ✅ 배운 것:
       1. 순차열은 순서가 중요한, 순서 있는 데이터이다
       2. 텍스트, 시계열, 음향, 영상에서 흔하다
       3. 원소마다 앞선 원소에 기댄다
       4. 3차원 텐서 (배치, seq_len, 피처)로 나타낸다

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

    ======================================================================
    Tutorial Complete! ✓
    ======================================================================
    ```


## 2. 논의

시퀀스에서 순서가 중요하다는 것과 모든 원소가 앞 원소에 통계적으로 의존한다는 것은 같은 주장이 아니다. 독립적인 관측도 시간 순서로 기록할 수 있고, 표 형식 데이터에도 시간·개체별 의존성이 있을 수 있다. 이미지 역시 패치나 픽셀의 순서를 정해 시퀀스로 처리할 수 있다. 데이터의 저장 형식만으로 모델을 결정하지 않는다.

피처 벡터가 있는 시퀀스 배치는 `batch_first=True`일 때 $(B,L,F)$로 나타낸다. 단어 ID 배치는 임베딩 전에는 $(B,L)$의 정수 텐서이고, 임베딩 후에 피처 축이 추가된다. 이 예제의 문장 패딩은 단어 사전을 만든 것이 아니라 각 문장 안의 위치를 1부터 매긴 모형이다. 따라서 같은 ID가 여러 문장에서 같은 단어를 뜻하지 않는다. 마지막 문장은 주석과 달리 8단어이며, 길이 9로 맞추면 패딩이 하나 붙는다.

패딩은 배치의 길이를 맞추지만, 패딩 위치를 모델이 자동으로 무시하게 하지는 않는다. 실제 길이와 마스크, 또는 packed sequence 등의 처리가 필요하다. 패딩 벡터를 0으로 만들더라도 순환 상태는 바이어스와 이전 상태에 의해 갱신될 수 있다.

길이 10의 숫자열에서 입력 창 길이가 3이고 바로 다음 원소 하나를 정답으로 삼으면 학습 쌍은 7개다. 겹치는 창을 만든 뒤 무작위로 학습·검증에 나누면 같은 시점의 정보가 양쪽에 포함될 수 있다. 미래 예측 평가에서는 시간순 분할과 각 예측 시점에 이용 가능한 정보를 기준으로 구성한다.

코드의 CNN·RNN 비교는 입문용 설명을 단순화한 것이다. CNN도 시간 축의 컨볼루션으로 시퀀스를 처리할 수 있고, 풀링 등으로 가변 입력 크기를 다룰 수 있다. 고정 문맥의 피드포워드 모델 역시 여러 시점의 입력을 함께 처리할 수 있다. RNN의 피처는 은닉 상태를 순환적으로 갱신하며 파라메터를 시점 간 공유한다는 점이다.

## 연습문제

<div class="drillbox" markdown>

**연습문제 1.** <span class="diff med" title="중간"></span>
이 코드의 세 가지 데이터 표현을 구분하라: 임의의 피처 텐서, 문장의 위치 ID, 슬라이딩 윈도 입력·정답 쌍.

</div>

??? success "연습문제 1 풀이"
    임의의 피처 텐서는 시퀀스 길이와 피처 축을 보여 주는 예시다. 문장의 위치 ID는 패딩을 시연할 뿐 공통 단어 사전이 아니다. 슬라이딩 윈도는 과거 길이 3의 값과 다음 값 하나를 대응시킨다. 세 표현 모두 RNN 학습 결과는 아니다.

---

<div class="drillbox" markdown>

**연습문제 2.** <span class="diff med" title="중간"></span>
단어 ID 배치와 실수 피처 배치를 RNN에 넣기 전 어떤 형태 검사를 해야 하는가?

</div>

??? success "연습문제 2 풀이"
    단어 ID는 $(B,L)$의 정수이며 임베딩 인덱스 범위를 만족해야 한다. 임베딩이나 연속 피처는 $(B,L,F)$의 실수 텐서이며 $F$가 RNN의 input_size와 같아야 한다. 길이 정보는 실제 시퀀스에 맞아야 하고 패딩 위치는 로스·상태 처리에서 구분한다.

---

<div class="drillbox" markdown>

**연습문제 3.** <span class="diff med" title="중간"></span>
슬라이딩 윈도 예제를 무작위로 나눌 때 평가가 낙관적으로 나올 수 있는 이유를 설명하라.

</div>

??? success "연습문제 3 풀이"
    인접 창들이 같은 원소를 공유하므로 학습과 검증 사이에 강한 중복이 생길 수 있다. 미래 예측 목적이라면 시간순으로 나누고, 전처리 통계도 학습 구간에서만 추정한다. 필요한 경우 창 중복을 막는 간격을 두되 실제 예측 시점의 이용 가능 정보에 맞춰 설계한다.

---

<div class="drillbox" markdown>

**연습문제 4.** <span class="diff hard" title="어려움"></span>
아래의 테스트 코드 조각을 검토하라. 정의되지 않은 클래스와 잘못된 이름, 입력 모양 가정을 찾아 이 페이지의 실제 함수에 맞는 점검 항목을 제안하라.

</div>

??? success "연습문제 4 풀이"
    아래 조각은 실제 실행 가능한 테스트가 아니다. 함수 이름과 모델 이름부터 이 예제에 맞게 작성해야 한다.
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
    이 페이지는 완성된 사용자 정의 모델 클래스를 제공하지 않는다. 실제 함수의 입력·출력 모양, 허용 자료형, 유한한 값과 경사의 유무를 기준으로 검사한다. 빈 입력을 처리하는 방식은 연산마다 다르며, 무조건 정상적인 로스나 경사가 나와야 한다고 가정하지 않는다.

## 정리하며

시퀀스를 다루려면 순서, 피처 축, 실제 길이와 예측 대상을 먼저 정한다. 패딩은 길이를 맞추는 작업이며 무시 처리는 별도로 필요하다. 이 페이지는 데이터 표현 예제이고 RNN의 학습 성능을 보여 주지는 않는다.
