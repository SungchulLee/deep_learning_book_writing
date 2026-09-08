# 활성화 함수

06_activation_functions.py - 활성화 함수 이해하기. 여러 활성화 함수와 그것이 학습에 미치는 영향을 비교한다.

순방향 신경망을 이해하는 것은 깊은 신경망을 효과적으로 만들고 학습시키는 데 필수적이다. 이 구현은 그 핵심 개념을 PyTorch로 보여주며, 현대적인 구조의 구성 요소를 직접 다뤄 볼 기회를 준다.

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

# 시각화를 위한 입력 구간 만들기
x = torch.linspace(-5, 5, 200)

# 활성화 함수 사전
activations = {
    'ReLU': nn.ReLU(),
    'Sigmoid': nn.Sigmoid(),
    'Tanh': nn.Tanh(),
    'LeakyReLU': nn.LeakyReLU(0.1),
    'ELU': nn.ELU(),
    'Softplus': nn.Softplus()
}

# 활성화 그리기
fig, axes = plt.subplots(2, 3, figsize=(15, 8))
axes = axes.flatten()

for idx, (name, activation) in enumerate(activations.items()):
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
    pass```

## 2. 논의

손실 계산은 모델의 출력을 최적화 목표와 이어 준다. 알맞은 손실 함수를 고르는 일은 결정적으로 중요하다. 손실 함수가 모델이 무엇을 최적화하도록 배울지를 정하며, 학습된 표현과 결정 경계를 직접 빚어내기 때문이다.

시각화는 모델의 거동을 이해하고 학습 문제를 진단하는 데 중요한 역할을 한다. 그림을 그리는 코드는 학습된 표현, 수렴의 움직임, 평가 지표에 대한 통찰을 주어 추상적인 계산을 손에 잡히게 만든다.

여기서 보인 방식은 더 복잡한 상황으로 자연스럽게 확장된다. 초매개변수, 구조의 변형, 여러 데이터셋을 두루 실험해 보면 이해가 깊어지고 딥러닝의 기초 과제에 대한 실용적인 직관이 쌓인다.

## 연습문제

**연습문제 1.**
코드를 끝까지 읽고 핵심적인 설계 결정을 찾아내라. 구체적인 구현 선택 세 가지를 나열하고, 각각이 순방향 신경망에 왜 적절한지 설명하라.

??? success "연습문제 1 풀이"
    설계 결정은 구현마다 다르지만 흔히 다음이 포함된다. (1) 활성화 함수의 선택 — ReLU 계열은 포화되지 않는 경사를 주어 학습을 빠르게 한다. (2) 정규화 전략 — 배치 정규화는 내부 공변량 이동을 줄여 학습을 안정시킨다. (3) 잔차 연결 — 있을 경우 건너뛰는 경로를 제공하여 깊은 신경망에서도 경사가 흐르게 한다. 각 선택은 표현력, 계산 비용, 학습 안정성 사이의 절충을 반영한다.

---

**연습문제 2.**
입력이 기대하는 모양과 자료형을 갖는지 확인하도록 주 함수나 클래스에 입력 검증을 추가하라. 잘못된 입력에는 유익한 오류 메시지를 내라.

??? success "연습문제 2 풀이"
    `forward` 메서드(또는 해당 함수)의 첫머리에 다음과 같은 검사를 추가한다. `assert x.dim() == expected_dims, f'Expected {expected_dims}D input, got {x.dim()}D'`와 `assert x.dtype == torch.float32, f'Expected float32, got {x.dtype}'`. 모양을 검증할 때는 중요한 차원을 확인한다. `B, C, H, W = x.shape; assert C == self.expected_channels`. 유익한 오류 메시지는 디버깅 속도를 크게 높이고 코드를 재사용하기에도 더 견고하게 만든다.

---

**연습문제 3.**
은닉 크기가 $h$이고 입력 크기가 $x$로 같을 때 LSTM 셀과 GRU 셀의 매개변수 개수를 비교하라. 어느 쪽이 더 적으며 그 이유는 무엇인가?

??? success "연습문제 3 풀이"
    LSTM에는 4개의 게이트(입력, 망각, 셀, 출력)가 있고 각 게이트가 입력과 은닉 상태 양쪽에 대한 가중치 행렬을 가지므로 $4 \times (x \cdot h + h \cdot h + h) = 4(xh + h^2 + h)$개의 매개변수를 갖는다. GRU에는 3개의 게이트(재설정, 갱신, 새 상태)가 있어 $3 \times (x \cdot h + h \cdot h + h) = 3(xh + h^2 + h)$개이다. GRU는 게이트를 4개 대신 3개 쓰고 셀 상태와 은닉 상태를 합치므로 LSTM의 75%에 해당하는 매개변수를 갖는다. 실무에서 GRU는 매개변수가 적은데도 LSTM에 견줄 만한 성능을 내는 경우가 많다.

---

**연습문제 4.**
활성화 함수 구현을 검증하는 종합적인 시험 함수를 작성하라. 빈 입력, 원소가 하나뿐인 입력, 아주 큰 입력, 그리고 극단적인 값(0, 아주 큰 수)을 가진 입력 등 경계 사례를 시험하라.

??? success "연습문제 4 풀이"
    경계 조건을 두루 시험하는 함수를 만든다.
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
    경사의 흐름을 시험하는 것은 그 구조가 처음부터 끝까지 이어지는 학습을 지원하는지 확인하는 데 특히 중요하다.

## 정리하며

**다룬 것** — 활성화 함수

손실 계산은 모델의 출력을 최적화 목표와 이어 준다.

앞의 연습문제 4개로 직접 확인할 수 있다.
