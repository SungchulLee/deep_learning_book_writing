# GRU 이론

GRU의 이론과 밑바닥부터의 구현. 문 달린 순환 단위(GRU)는 LSTM을 간소화한 판본이다.

순차열 모형은 시간적이고 순서가 있는 데이터를 다루는 데 바탕이 된다. 이 구현은 순환 신경망의 핵심 착상을 다루며, 순환 계산과 학습된 표현이 시각 사이의 의존을 어떻게 붙잡는지 보인다.

## 코드

```python
"""
GRU의 이론과 밑바닥부터의 구현
==========================================

문 달린 순환 단위(GRU)는 망각 문과 입력 문을 갱신 문 하나로 합치고
세포 상태와 숨은 상태를 하나로 묶은, LSTM을 간소화한 판본이다.


수식:
------------------------
시각 t에 대해:

Reset Gate:    r_t = σ(W_r · [h_{t-1}, x_t] + b_r)
Update Gate:   z_t = σ(W_z · [h_{t-1}, x_t] + b_z)
Candidate:     h̃_t = tanh(W_h · [r_t ⊙ h_{t-1}, x_t] + b_h)
Hidden State:  h_t = (1 - z_t) ⊙ h_{t-1} + z_t ⊙ h̃_t

여기서 각 기호는 다음과 같다.
- σ는 시그모이드 함수이다
- ⊙는 성분별 곱이다
- W는 가중치 행렬이다
- b는 편향 벡터이다

LSTM과의 핵심 차이:
- 문이 3개가 아니라 2개이다 (출력 문이 없다)
- 세포 상태를 따로 두지 않는다
- 매개변수가 적다 (학습이 빠르다)
- 많은 과제에서 성능이 비슷하다
"""

import numpy as np
import matplotlib.pyplot as plt

# ========================================================================
# 메인
# ========================================================================


class GRUCell:
    """밑바닥부터 만든 GRU 세포 하나의 구현."""
    
    def __init__(self, input_size, hidden_size):
        """
        GRU 세포의 매개변수를 초기화한다.
        
        인수:
            input_size: 입력 특징의 차원
            hidden_size: 숨은 상태의 차원
        """
        self.input_size = input_size
        self.hidden_size = hidden_size
        
        # 자비에르 초기화로 가중치 초기화
        scale = 1.0 / np.sqrt(input_size + hidden_size)
        
        # 재설정 문의 가중치
        self.W_r = np.random.randn(hidden_size, input_size + hidden_size) * scale
        self.b_r = np.zeros((hidden_size, 1))
        
        # 갱신 문의 가중치
        self.W_z = np.random.randn(hidden_size, input_size + hidden_size) * scale
        self.b_z = np.zeros((hidden_size, 1))
        
        # 후보 숨은 상태의 가중치
        self.W_h = np.random.randn(hidden_size, input_size + hidden_size) * scale
        self.b_h = np.zeros((hidden_size, 1))
        
    def sigmoid(self, x):
        """시그모이드 활성화 함수."""
        return 1 / (1 + np.exp(-np.clip(x, -500, 500)))
    
    def tanh(self, x):
        """쌍곡탄젠트 활성화 함수."""
        return np.tanh(x)
    
    def forward(self, x, h_prev):
        """
        GRU 세포를 지나는 순전파.
        
        인수:
            x: 지금 시각의 입력 (input_size, 1)
            h_prev: 이전 숨은 상태 (hidden_size, 1)
            
        반환값:
            h: 새 숨은 상태
            cache: 역전파에 필요한 값
        """
        # 이전 숨은 상태와 지금 입력 이어 붙이기
        combined = np.vstack((h_prev, x))
        
        # 재설정 문 — 과거 정보를 얼마나 잊을지 정한다
        r = self.sigmoid(self.W_r @ combined + self.b_r)
        
        # 갱신 문 — 얼마나 갱신할지 정한다
        z = self.sigmoid(self.W_z @ combined + self.b_z)
        
        # 재설정 문을 쓴 후보 숨은 상태
        # 재설정 문이 이전 숨은 상태를 조절한다
        combined_reset = np.vstack((r * h_prev, x))
        h_tilde = self.tanh(self.W_h @ combined_reset + self.b_h)
        
        # 새 숨은 상태: 이전 상태와 후보 사이의 보간
        # z가 "망각 문"과 "입력 문"을 합친 구실을 한다
        h = (1 - z) * h_prev + z * h_tilde
        
        # 역전파를 위한 저장
        cache = (x, h_prev, combined, r, z, h_tilde)
        
        return h, cache


class GRU:
    """여러 걸음을 다루는 GRU 구현."""
    
    def __init__(self, input_size, hidden_size, output_size):
        """
        GRU 신경망을 초기화한다.
        
        인수:
            input_size: 입력 특징의 차원
            hidden_size: 숨은 상태의 차원
            output_size: 출력의 차원
        """
        self.cell = GRUCell(input_size, hidden_size)
        
        # 출력층의 가중치
        self.W_y = np.random.randn(output_size, hidden_size) * 0.01
        self.b_y = np.zeros((output_size, 1))
        
        self.hidden_size = hidden_size
        
    def forward(self, inputs):
        """
        순차열 전체를 지나는 순전파.
        
        인수:
            inputs: 모양이 (input_size, 1)인 입력 벡터의 목록
            
        반환값:
            outputs: 출력 예측의 목록
            hidden_states: 그려 보려고 모은 숨은 상태의 목록
        """
        h = np.zeros((self.hidden_size, 1))
        
        outputs = []
        hidden_states = []
        
        for x in inputs:
            h, _ = self.cell.forward(x, h)
            y = self.W_y @ h + self.b_y
            
            outputs.append(y)
            hidden_states.append(h.copy())
            
        return outputs, hidden_states


def demonstrate_gru_gates():
    """그림으로 GRU의 문이 어떻게 움직이는지 보인다."""
    print("=" * 60)
    print("GRU Gate Mechanics Demonstration")
    print("=" * 60)
    
    # 간단한 GRU 세포 만들기
    input_size = 3
    hidden_size = 4
    gru_cell = GRUCell(input_size, hidden_size)
    
    # 예제 입력 순차열 만들기
    sequence_length = 20
    inputs = [np.random.randn(input_size, 1) for _ in range(sequence_length)]
    
    # 문의 활성값 좇기
    reset_gates = []
    update_gates = []
    hidden_states = []
    candidates = []
    
    h = np.zeros((hidden_size, 1))
    
    for x in inputs:
        combined = np.vstack((h, x))
        
        r = gru_cell.sigmoid(gru_cell.W_r @ combined + gru_cell.b_r)
        z = gru_cell.sigmoid(gru_cell.W_z @ combined + gru_cell.b_z)
        
        combined_reset = np.vstack((r * h, x))
        h_tilde = gru_cell.tanh(gru_cell.W_h @ combined_reset + gru_cell.b_h)
        
        h = (1 - z) * h + z * h_tilde
        
        reset_gates.append(r.mean())
        update_gates.append(z.mean())
        candidates.append(h_tilde.mean())
        hidden_states.append(h.mean())
    
    # 문의 활성값 그려 보기
    plt.figure(figsize=(12, 8))
    
    plt.subplot(2, 2, 1)
    plt.plot(reset_gates, 'r-', linewidth=2)
    plt.title('Reset Gate Activation')
    plt.ylabel('Activation (0-1)')
    plt.xlabel('Time Step')
    plt.grid(True, alpha=0.3)
    plt.ylim([0, 1])
    
    plt.subplot(2, 2, 2)
    plt.plot(update_gates, 'g-', linewidth=2)
    plt.title('Update Gate Activation')
    plt.ylabel('Activation (0-1)')
    plt.xlabel('Time Step')
    plt.grid(True, alpha=0.3)
    plt.ylim([0, 1])
    
    plt.subplot(2, 2, 3)
    plt.plot(candidates, 'b-', linewidth=2)
    plt.title('Candidate Hidden State')
    plt.ylabel('Mean Value')
    plt.xlabel('Time Step')
    plt.grid(True, alpha=0.3)
    
    plt.subplot(2, 2, 4)
    plt.plot(hidden_states, 'm-', linewidth=2)
    plt.title('Hidden State Evolution')
    plt.ylabel('Mean Value')
    plt.xlabel('Time Step')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('/home/claude/lstm_gru_module/gru_gates.png', dpi=150, bbox_inches='tight')
    print("\n✓ Gate activation plot saved as 'gru_gates.png'")
    plt.close()


def compare_gate_mechanics():
    """갱신 문이 이전 상태와 새 상태를 어떻게 보간하는지 견준다."""
    print("\n" + "=" * 60)
    print("Understanding the Update Gate")
    print("=" * 60)
    
    print("\nThe update gate z_t controls:")
    print("h_t = (1 - z_t) ⊙ h_{t-1} + z_t ⊙ h̃_t")
    print("\nWhen z_t ≈ 0: Keep old hidden state (h_t ≈ h_{t-1})")
    print("When z_t ≈ 1: Use new candidate (h_t ≈ h̃_t)")
    print("When z_t ≈ 0.5: Mix equally")
    
    # 예제로 시연하기
    h_prev = np.array([[1.0], [2.0], [3.0]])
    h_tilde = np.array([[5.0], [6.0], [7.0]])
    
    scenarios = [
        ("z ≈ 0 (Keep old)", np.array([[0.1], [0.1], [0.1]])),
        ("z ≈ 1 (Use new)", np.array([[0.9], [0.9], [0.9]])),
        ("z ≈ 0.5 (Mix)", np.array([[0.5], [0.5], [0.5]])),
    ]
    
    print("\nExample with h_{t-1} = [1, 2, 3] and h̃_t = [5, 6, 7]:")
    for name, z in scenarios:
        h = (1 - z) * h_prev + z * h_tilde
        print(f"\n{name}:")
        print(f"  Result: {h.flatten()}")


def parameter_comparison():
    """LSTM과 GRU의 매개변수 수를 견준다."""
    print("\n" + "=" * 60)
    print("LSTM vs GRU Parameter Comparison")
    print("=" * 60)
    
    input_size = 100
    hidden_size = 200
    
    # LSTM의 매개변수
    lstm_params = 4 * (hidden_size * (input_size + hidden_size) + hidden_size)
    
    # GRU의 매개변수
    gru_params = 3 * (hidden_size * (input_size + hidden_size) + hidden_size)
    
    reduction = (1 - gru_params / lstm_params) * 100
    
    print(f"\nFor input_size={input_size}, hidden_size={hidden_size}:")
    print(f"LSTM parameters: {lstm_params:,}")
    print(f"GRU parameters:  {gru_params:,}")
    print(f"Reduction:       {reduction:.1f}%")
    
    print("\nGRU has ~25% fewer parameters than LSTM")
    print("This means:")
    print("  ✓ Faster training")
    print("  ✓ Less memory usage")
    print("  ✓ Less prone to overfitting on small datasets")


def main():
    """모든 GRU 시연을 실행한다."""
    print("\n" + "=" * 60)
    print("GRU: Gated Recurrent Unit Networks")
    print("=" * 60)
    
    print("\nKey Features of GRU:")
    print("1. Simpler than LSTM (2 gates vs 3)")
    print("2. No separate cell state")
    print("3. Fewer parameters → faster training")
    print("4. Often performs similarly to LSTM")
    
    print("\nGRU Gates:")
    print("- Reset gate (r): Controls how much past info to forget")
    print("- Update gate (z): Controls how much to update")
    print("  * Acts as both forget and input gate")
    
    # 시연 실행
    demonstrate_gru_gates()
    compare_gate_mechanics()
    parameter_comparison()
    
    print("\n" + "=" * 60)
    print("GRU demonstrations complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()```

## 논의

이 구현은 클래스 두 개(`GRUCell`, `GRU`)를 정의하며, 이들이 어우러져 완전한 순환 신경망 구조를 이룬다. 클래스마다 별개의 부품을 감싸므로 코드가 모듈식이고 넓히기 쉽다. `forward` 메서드가 PyTorch의 자동 미분이 쓰는 계산 그래프를 정의한다.

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
은닉 크기가 $h$이고 입력 크기가 $x$로 같을 때 LSTM 셀과 GRU 셀의 매개변수 개수를 비교하라. 어느 쪽이 더 적으며 그 이유는 무엇인가?

??? success "연습문제 3 풀이"
    LSTM에는 4개의 게이트(입력, 망각, 셀, 출력)가 있고 각 게이트가 입력과 은닉 상태 양쪽에 대한 가중치 행렬을 가지므로 $4 \times (x \cdot h + h \cdot h + h) = 4(xh + h^2 + h)$개의 매개변수를 갖는다. GRU에는 3개의 게이트(재설정, 갱신, 새 상태)가 있어 $3 \times (x \cdot h + h \cdot h + h) = 3(xh + h^2 + h)$개이다. GRU는 게이트를 4개 대신 3개 쓰고 셀 상태와 은닉 상태를 합치므로 LSTM의 75%에 해당하는 매개변수를 갖는다. 실무에서 GRU는 매개변수가 적은데도 LSTM에 견줄 만한 성능을 내는 경우가 많다.

---

**연습문제 4.**
GRU 이론 구현을 검증하는 종합 시험 함수를 작성하라. 빈 입력, 원소가 하나뿐인 입력, 아주 큰 입력, 극단적인 값(0, 아주 큰 수)을 가진 입력 같은 경계 상황을 시험하라.

??? success "연습문제 4 풀이"
    경계 조건을 두루 시험하는 함수를 만든다.
    ```python
    def test_grucell():
        model = GRUCell(...)
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
