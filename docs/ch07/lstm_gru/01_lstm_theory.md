# LSTM 이론

LSTM의 이론과 밑바닥부터의 구현. 장단기 기억(LSTM) 신경망은 기울기 소실 문제를 푼다.

순차열 모형은 시간적이고 순서가 있는 데이터를 다루는 데 바탕이 된다. 이 구현은 순환 신경망의 핵심 착상을 다루며, 순환 계산과 학습된 표현이 시각 사이의 의존을 어떻게 붙잡는지 보인다.

## 코드

```python
"""
LSTM의 이론과 밑바닥부터의 구현
============================================

장단기 기억(LSTM) 신경망은 문 장치와 세포 상태를 들여와 전통적인 RNN의
기울기 소실 문제를 푼다.

수식:
------------------------
시각 t에 대해:

Forget Gate:    f_t = σ(W_f · [h_{t-1}, x_t] + b_f)
Input Gate:     i_t = σ(W_i · [h_{t-1}, x_t] + b_i)
Cell Candidate: C̃_t = tanh(W_C · [h_{t-1}, x_t] + b_C)
Cell State:     C_t = f_t ⊙ C_{t-1} + i_t ⊙ C̃_t
Output Gate:    o_t = σ(W_o · [h_{t-1}, x_t] + b_o)
Hidden State:   h_t = o_t ⊙ tanh(C_t)

여기서 각 기호는 다음과 같다.
- σ는 시그모이드 함수이다
- ⊙는 성분별 곱이다
- W는 가중치 행렬이다
- b는 편향 벡터이다
"""

import numpy as np
import matplotlib.pyplot as plt

# ========================================================================
# 메인
# ========================================================================


class LSTMCell:
    """밑바닥부터 만든 LSTM 세포 하나의 구현."""
    
    def __init__(self, input_size, hidden_size):
        """
        LSTM 세포의 매개변수를 초기화한다.
        
        인수:
            input_size: 입력 특징의 차원
            hidden_size: 숨은 상태의 차원
        """
        self.input_size = input_size
        self.hidden_size = hidden_size
        
        # 자비에르 초기화로 가중치 초기화
        scale = 1.0 / np.sqrt(input_size + hidden_size)
        
        # 망각 문의 가중치
        self.W_f = np.random.randn(hidden_size, input_size + hidden_size) * scale
        self.b_f = np.zeros((hidden_size, 1))
        
        # 입력 문의 가중치
        self.W_i = np.random.randn(hidden_size, input_size + hidden_size) * scale
        self.b_i = np.zeros((hidden_size, 1))
        
        # 세포 후보의 가중치
        self.W_c = np.random.randn(hidden_size, input_size + hidden_size) * scale
        self.b_c = np.zeros((hidden_size, 1))
        
        # 출력 문의 가중치
        self.W_o = np.random.randn(hidden_size, input_size + hidden_size) * scale
        self.b_o = np.zeros((hidden_size, 1))
        
    def sigmoid(self, x):
        """시그모이드 활성화 함수."""
        return 1 / (1 + np.exp(-np.clip(x, -500, 500)))
    
    def tanh(self, x):
        """쌍곡탄젠트 활성화 함수."""
        return np.tanh(x)
    
    def forward(self, x, h_prev, c_prev):
        """
        LSTM 세포를 지나는 순전파.
        
        인수:
            x: 지금 시각의 입력 (input_size, 1)
            h_prev: 이전 숨은 상태 (hidden_size, 1)
            c_prev: 이전 세포 상태 (hidden_size, 1)
            
        반환값:
            h: 새 숨은 상태
            c: 새 세포 상태
            cache: 역전파에 필요한 값
        """
        # 이전 숨은 상태와 지금 입력 이어 붙이기
        combined = np.vstack((h_prev, x))
        
        # 망각 문
        f = self.sigmoid(self.W_f @ combined + self.b_f)
        
        # 입력 문
        i = self.sigmoid(self.W_i @ combined + self.b_i)
        
        # 세포 후보
        c_tilde = self.tanh(self.W_c @ combined + self.b_c)
        
        # 새 세포 상태
        c = f * c_prev + i * c_tilde
        
        # 출력 문
        o = self.sigmoid(self.W_o @ combined + self.b_o)
        
        # 새 숨은 상태
        h = o * self.tanh(c)
        
        # 역전파를 위한 저장
        cache = (x, h_prev, c_prev, combined, f, i, c_tilde, c, o)
        
        return h, c, cache


class LSTM:
    """여러 걸음을 다루는 LSTM 구현."""
    
    def __init__(self, input_size, hidden_size, output_size):
        """
        LSTM 신경망을 초기화한다.
        
        인수:
            input_size: 입력 특징의 차원
            hidden_size: 숨은 상태의 차원
            output_size: 출력의 차원
        """
        self.cell = LSTMCell(input_size, hidden_size)
        
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
        c = np.zeros((self.hidden_size, 1))
        
        outputs = []
        hidden_states = []
        
        for x in inputs:
            h, c, _ = self.cell.forward(x, h, c)
            y = self.W_y @ h + self.b_y
            
            outputs.append(y)
            hidden_states.append(h.copy())
            
        return outputs, hidden_states


def demonstrate_lstm_gates():
    """그림으로 LSTM의 문이 어떻게 움직이는지 보인다."""
    print("=" * 60)
    print("LSTM Gate Mechanics Demonstration")
    print("=" * 60)
    
    # 간단한 LSTM 세포 만들기
    input_size = 3
    hidden_size = 4
    lstm_cell = LSTMCell(input_size, hidden_size)
    
    # 예제 입력 순차열 만들기
    sequence_length = 20
    inputs = [np.random.randn(input_size, 1) for _ in range(sequence_length)]
    
    # 문의 활성값 좇기
    forget_gates = []
    input_gates = []
    output_gates = []
    cell_states = []
    
    h = np.zeros((hidden_size, 1))
    c = np.zeros((hidden_size, 1))
    
    for x in inputs:
        combined = np.vstack((h, x))
        
        f = lstm_cell.sigmoid(lstm_cell.W_f @ combined + lstm_cell.b_f)
        i = lstm_cell.sigmoid(lstm_cell.W_i @ combined + lstm_cell.b_i)
        c_tilde = lstm_cell.tanh(lstm_cell.W_c @ combined + lstm_cell.b_c)
        c = f * c + i * c_tilde
        o = lstm_cell.sigmoid(lstm_cell.W_o @ combined + lstm_cell.b_o)
        h = o * lstm_cell.tanh(c)
        
        forget_gates.append(f.mean())
        input_gates.append(i.mean())
        output_gates.append(o.mean())
        cell_states.append(c.mean())
    
    # 문의 활성값 그려 보기
    plt.figure(figsize=(12, 8))
    
    plt.subplot(2, 2, 1)
    plt.plot(forget_gates, 'r-', linewidth=2)
    plt.title('Forget Gate Activation')
    plt.ylabel('Activation (0-1)')
    plt.xlabel('Time Step')
    plt.grid(True, alpha=0.3)
    plt.ylim([0, 1])
    
    plt.subplot(2, 2, 2)
    plt.plot(input_gates, 'g-', linewidth=2)
    plt.title('Input Gate Activation')
    plt.ylabel('Activation (0-1)')
    plt.xlabel('Time Step')
    plt.grid(True, alpha=0.3)
    plt.ylim([0, 1])
    
    plt.subplot(2, 2, 3)
    plt.plot(output_gates, 'b-', linewidth=2)
    plt.title('Output Gate Activation')
    plt.ylabel('Activation (0-1)')
    plt.xlabel('Time Step')
    plt.grid(True, alpha=0.3)
    plt.ylim([0, 1])
    
    plt.subplot(2, 2, 4)
    plt.plot(cell_states, 'm-', linewidth=2)
    plt.title('Cell State Evolution')
    plt.ylabel('Mean Cell State Value')
    plt.xlabel('Time Step')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('/home/claude/lstm_gru_module/lstm_gates.png', dpi=150, bbox_inches='tight')
    print("\n✓ Gate activation plot saved as 'lstm_gates.png'")
    plt.close()


def sequence_prediction_example():
    """예: 사인파의 다음 값 예측하기."""
    print("\n" + "=" * 60)
    print("LSTM Sequence Prediction Example: Sine Wave")
    print("=" * 60)
    
    # 사인파 데이터 만들기
    t = np.linspace(0, 20, 200)
    data = np.sin(t)
    
    # 순차열 준비 (지난 10개 점으로 다음 점 예측)
    seq_length = 10
    sequences = []
    targets = []
    
    for i in range(len(data) - seq_length):
        sequences.append(data[i:i+seq_length])
        targets.append(data[i+seq_length])
    
    # LSTM 만들기
    lstm = LSTM(input_size=1, hidden_size=20, output_size=1)
    
    # 첫 순차열로 시연
    test_seq = sequences[0]
    inputs = [np.array([[x]]) for x in test_seq]
    
    outputs, hidden_states = lstm.forward(inputs)
    
    print(f"\nInput sequence length: {len(inputs)}")
    print(f"Hidden state dimension: {hidden_states[0].shape}")
    print(f"Number of LSTM parameters:")
    print(f"  - Forget gate: {lstm.cell.W_f.size + lstm.cell.b_f.size}")
    print(f"  - Input gate: {lstm.cell.W_i.size + lstm.cell.b_i.size}")
    print(f"  - Cell candidate: {lstm.cell.W_c.size + lstm.cell.b_c.size}")
    print(f"  - Output gate: {lstm.cell.W_o.size + lstm.cell.b_o.size}")
    total_params = (lstm.cell.W_f.size + lstm.cell.b_f.size +
                    lstm.cell.W_i.size + lstm.cell.b_i.size +
                    lstm.cell.W_c.size + lstm.cell.b_c.size +
                    lstm.cell.W_o.size + lstm.cell.b_o.size +
                    lstm.W_y.size + lstm.b_y.size)
    print(f"  - Total: {total_params}")


def main():
    """모든 LSTM 시연을 실행한다."""
    print("\n" + "=" * 60)
    print("LSTM: Long Short-Term Memory Networks")
    print("=" * 60)
    
    print("\nKey Advantages of LSTM:")
    print("1. Solves vanishing gradient problem")
    print("2. Can learn long-term dependencies")
    print("3. Gates control information flow")
    print("4. Cell state acts as 'memory highway'")
    
    print("\nLSTM vs Traditional RNN:")
    print("- RNN: Simple recurrent connection")
    print("- LSTM: Gated architecture with cell state")
    print("- LSTM has 4x more parameters than RNN")
    
    # 시연 실행
    demonstrate_lstm_gates()
    sequence_prediction_example()
    
    print("\n" + "=" * 60)
    print("LSTM demonstrations complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()```

## 논의

이 구현은 클래스 두 개(`LSTMCell`, `LSTM`)를 정의하며, 이들이 어우러져 완전한 순환 신경망 구조를 이룬다. 클래스마다 별개의 부품을 감싸므로 코드가 모듈식이고 넓히기 쉽다. `forward` 메서드가 PyTorch의 자동 미분이 쓰는 계산 그래프를 정의한다.

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
LSTM 이론 구현을 검증하는 종합 시험 함수를 작성하라. 빈 입력, 원소가 하나뿐인 입력, 아주 큰 입력, 극단적인 값(0, 아주 큰 수)을 가진 입력 같은 경계 상황을 시험하라.

??? success "연습문제 4 풀이"
    경계 조건을 두루 시험하는 함수를 만든다.
    ```python
    def test_lstmcell():
        model = LSTMCell(...)
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
