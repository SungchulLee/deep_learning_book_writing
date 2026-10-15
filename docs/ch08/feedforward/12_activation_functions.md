# 활성화 함수

ReLU, 시그모이드, tanh, Leaky ReLU, ELU, Softplus를 같은 입력 구간에서 그려 비교한다. 이 코드는 함수 모양을 비교하며 모델 학습 성능을 직접 비교하지는 않는다.

## 1. 코드

```python
"""
06_activation_functions.py - 활성화 함수 이해하기

여러 활성화 함수와 그것이 학습에 미치는 영향을 비교한다.
각각의 거동을 시각화하고 언제 쓰는지 배운다.

소요 시간: 20~25분 | 난이도: ⭐⭐☆☆☆
"""

import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np

# ========================================================================
# 메인
# ========================================================================

print("="*70)
print("Activation Functions Comparison")
print("="*70)

# 시각화를 위한 입력 구간 만들기.
# -5에서 5면 시그모이드와 tanh가 양 끝에서 평평해지는 모습이 보인다.
# 이 구간을 좁히면 그 포화가 그림에서 사라져 기울기 소실을 볼 수 없다
x = torch.linspace(-5, 5, 200)

# 활성화 함수 사전.
# 여섯 가지가 두 갈래로 나뉜다. 시그모이드와 tanh는 위아래로 막혀 있어
# 양 끝에서 기울기가 0으로 죽고, 나머지 넷은 양수 쪽이 뚫려 있어
# 기울기가 살아 남는다. 깊은 신경망에서 뒤쪽 넷을 쓰는 까닭이다
activations = {
    'ReLU': nn.ReLU(),
    'Sigmoid': nn.Sigmoid(),
    'Tanh': nn.Tanh(),
    # 기본값 0.01이 아니라 0.1을 준 것은 그림에서 음수 쪽 기울기가
    # 눈에 띄게 하려는 뜻이다. 0.01이면 거의 ReLU처럼 보인다
    'LeakyReLU': nn.LeakyReLU(0.1),
    'ELU': nn.ELU(),
    'Softplus': nn.Softplus()
}

# 활성화 그리기
fig, axes = plt.subplots(2, 3, figsize=(15, 8))
axes = axes.flatten()

for idx, (name, activation) in enumerate(activations.items()):
    # 활성화 함수는 원소마다 따로 적용된다. 그래서 200개짜리 벡터를
    # 통째로 넣어도 되고, 값 하나하나에 함수를 건 것과 결과가 같다.
    # 학습되는 매개변수가 없으므로 no_grad도 필요 없다
    y = activation(x)
    axes[idx].plot(x.numpy(), y.numpy(), linewidth=2)
    axes[idx].set_title(name, fontsize=14, fontweight='bold')
    axes[idx].grid(True, alpha=0.3)
    axes[idx].axhline(y=0, color='k', linewidth=0.5)
    axes[idx].axvline(x=0, color='k', linewidth=0.5)
    axes[idx].set_xlabel('Input')
    axes[idx].set_ylabel('Output')

plt.tight_layout()
plt.savefig('06_activations.png', dpi=150)
print("Activation functions plotted and saved!")

print("\n" + "="*70)
print("ACTIVATION FUNCTION GUIDE")
print("="*70)
print("""
ReLU(정류 선형 유닛)
  Formula: f(x) = max(0, x)
  장점: 단순하고 빠르며 잘 작동한다
  단점: 죽은 뉴런(모든 입력에 0을 내놓는 뉴런)
  쓰임: 은닉층의 기본 선택

Sigmoid
  Formula: f(x) = 1 / (1 + e^(-x))
  장점: 매끄럽고 [0,1]로 유계이다
  단점: 기울기 소실, 느린 수렴
  쓰임: 이진 분류 출력, LSTM의 게이트

Tanh
  Formula: f(x) = (e^x - e^(-x)) / (e^x + e^(-x))
  장점: 0을 중심으로 하며 [-1,1]로 유계이다
  단점: 기울기 소실
  쓰임: 때때로 RNN에서, 은닉층에서는 시그모이드보다 낫다

LeakyReLU
  Formula: f(x) = max(0.1x, x)
  장점: 죽은 ReLU 문제를 고치며 x < 0에서도 작은 기울기가 있다
  단점: 이득이 한결같지 않다
  쓰임: 죽은 ReLU 문제를 만났을 때

ELU(지수 선형 유닛)
  Formula: f(x) = x if x>0 else α(e^x - 1)
  장점: 매끄럽고 평균 활성값이 0에 더 가깝다
  단점: 계산 비용이 더 크다
  쓰임: ReLU가 잘 안 될 때

Softplus
  Formula: f(x) = log(1 + e^x)
  장점: ReLU를 매끄럽게 근사한다
  단점: 비용이 더 크다
  쓰임: 실무에서는 드물고 이론적 관심 대상이다

RECOMMENDATIONS:
  - 은닉층: ReLU(기본), 필요하면 LeakyReLU
  - 이진 출력: 시그모이드
  - 다중 클래스 출력: 소프트맥스(CrossEntropyLoss로)
  - 회귀 출력: 없음(선형)
""")

print("\nEXERCISES:")
print("1. Train MNIST with each activation - compare results")
print("2. Visualize gradient flow for each activation")
print("3. Implement custom activation function")
plt.show()


if __name__ == "__main__":
    pass
```

**출력:**

```
======================================================================
Activation Functions Comparison
======================================================================
Activation functions plotted and saved!

======================================================================
ACTIVATION FUNCTION GUIDE
======================================================================

ReLU(정류 선형 유닛)
  Formula: f(x) = max(0, x)
  장점: 단순하고 빠르며 잘 작동한다
  단점: 죽은 뉴런(모든 입력에 0을 내놓는 뉴런)
  쓰임: 은닉층의 기본 선택

Sigmoid
  Formula: f(x) = 1 / (1 + e^(-x))
  장점: 매끄럽고 [0,1]로 유계이다
  단점: 기울기 소실, 느린 수렴
  쓰임: 이진 분류 출력, LSTM의 게이트

Tanh
  Formula: f(x) = (e^x - e^(-x)) / (e^x + e^(-x))
  장점: 0을 중심으로 하며 [-1,1]로 유계이다
  단점: 기울기 소실
  쓰임: 때때로 RNN에서, 은닉층에서는 시그모이드보다 낫다

LeakyReLU
  Formula: f(x) = max(0.1x, x)
  장점: 죽은 ReLU 문제를 고치며 x < 0에서도 작은 기울기가 있다
  단점: 이득이 한결같지 않다
  쓰임: 죽은 ReLU 문제를 만났을 때

ELU(지수 선형 유닛)
  Formula: f(x) = x if x>0 else α(e^x - 1)
  장점: 매끄럽고 평균 활성값이 0에 더 가깝다
  단점: 계산 비용이 더 크다
  쓰임: ReLU가 잘 안 될 때

Softplus
  Formula: f(x) = log(1 + e^x)
  장점: ReLU를 매끄럽게 근사한다
  단점: 비용이 더 크다
  쓰임: 실무에서는 드물고 이론적 관심 대상이다

RECOMMENDATIONS:
  - 은닉층: ReLU(기본), 필요하면 LeakyReLU
  - 이진 출력: 시그모이드
  - 다중 클래스 출력: 소프트맥스(CrossEntropyLoss로)
  - 회귀 출력: 없음(선형)


EXERCISES:
1. Train MNIST with each activation - compare results
2. Visualize gradient flow for each activation
3. Implement custom activation function
```

## 2. 논의

각 곡선에서 출력 범위, 음수 입력 처리, 포화와 매끄러움을 비교한다. 함수값의 모양만으로 깊은 모델의 경사나 최종 정확도를 판단할 수는 없다.

ReLU는 양수 영역에서 도함수가 1이고 음수 영역에서 0이다. 시그모이드와 tanh는 입력 절댓값이 커지면 도함수가 작아진다. Leaky ReLU는 음수 기울기를 두고, ELU와 Softplus는 다른 방식으로 음수 또는 작은 입력을 처리한다.

활성화를 학습 실험에서 비교하려면 같은 구조와 데이터 분할을 사용하되, 활성화에 맞는 초기화와 학습률도 검토해야 한다.

## 연습문제

<div class="drillbox" markdown>

**연습문제 1.** <span class="diff med" title="중간"></span>
이 코드가 비교하는 여섯 함수에서 출력이 음수가 될 수 있는 함수를 골라라.

</div>

??? success "연습문제 1 풀이"
    tanh, 음수 기울기가 양수인 Leaky ReLU, 양의 alpha를 쓰는 ELU는 음수 출력을 낼 수 있다. ReLU와 Softplus는 음수가 아니고 시그모이드는 0과 1 사이이다. 부동소수점 구현에서는 포화와 반올림도 고려한다.

---

<div class="drillbox" markdown>

**연습문제 2.** <span class="diff med" title="중간"></span>
시그모이드 곡선의 평평한 구간이 역전파에 미치는 영향을 설명하라.

</div>

??? success "연습문제 2 풀이"
    평평한 구간에서는 국소 도함수가 작으므로 그 지점을 통과하는 경사가 줄어든다. 전체 경사에는 가중치와 다른 층도 곱해지므로 활성화 하나의 모양만으로 경사 소실의 정도를 결정하지 않는다.

---

<div class="drillbox" markdown>

**연습문제 3.** <span class="diff med" title="중간"></span>
Softplus와 ReLU의 매끄러움 차이를 설명하라.

</div>

??? success "연습문제 3 풀이"
    ReLU는 0에서 미분 가능하지 않다. Softplus는 매끄러운 근사이며 그 도함수는 스케일에 맞는 시그모이드이다. 매끄럽다는 이유만으로 모든 학습 과제에서 더 좋거나 더 빠르다고 할 수 없다.

---

<div class="drillbox" markdown>

**연습문제 4.** <span class="diff hard" title="어려움"></span>
아래의 테스트 코드 조각을 검토하라. 정의되지 않은 클래스와 잘못된 이름, 입력 모양 가정을 찾아 이 페이지의 실제 함수에 맞는 점검 항목을 제안하라.

</div>

??? success "연습문제 4 풀이"
    아래 조각은 실제 실행 가능한 테스트가 아니다. 함수 이름과 모델 이름부터 이 예제에 맞게 작성해야 한다.
    ```python
    def test_activation functions():
        model = Activation Functions(...)
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
    이 페이지는 완성된 사용자 정의 모델 클래스를 제공하지 않는다. 실제 함수의 입력·출력 모양, 허용 자료형, 유한한 값과 경사의 유무를 기준으로 검사한다. 빈 입력을 처리하는 방식은 연산마다 다르며, 무조건 정상적인 손실이나 경사가 나와야 한다고 가정하지 않는다.

## 정리하며

이 예제는 활성화의 출력 모양을 비교한다. 출력 범위와 도함수를 이해하고, 실제 성능 비교에서는 초기화와 최적화 조건까지 함께 살펴야 한다.
