# LSTM 이론

NumPy로 LSTM 셀의 순전파를 구현하고 게이트와 셀 상태의 변화를 살펴본다. 이 코드는 PyTorch 모듈이나 학습 루프가 아니며, 로스·역전파·파라메터 갱신을 구현하지 않는다. LSTM의 상태 전달 구조와 실제 학습 결과를 구분한다.

## 1. 코드

```python
"""
LSTM의 이론과 밑바닥부터의 구현
============================================

장단기 기억(LSTM) 뉴럴 넷트웍은 문 장치와 세포 상태를 들여와 전통적인 RNN의
그래디언트 소실 문제를 푼다.

수식:
------------------------
시각 t에 대해:

Forget Gate:    f_t = σ(W_f · [h_{t-1}, x_t] + b_f)
Input Gate:     i_t = σ(W_i · [h_{t-1}, x_t] + b_i)
칸 후보: C̃_t = tanh(W_C · [h_{t-1}, x_t] + b_C)
Cell State:     C_t = f_t ⊙ C_{t-1} + i_t ⊙ C̃_t
Output Gate:    o_t = σ(W_o · [h_{t-1}, x_t] + b_o)
숨은 상태:   h_t = o_t ⊙ tanh(C_t)

여기서 각 기호는 다음과 같다.
- σ는 시그모이드 함수이다
- ⊙는 성분별 곱이다
- W는 웨이트 행렬이다
- b는 바이어스 벡터이다
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
        LSTM 세포의 파라메터를 초기화한다.
        
        인수:
            input_size: 입력 피처의 차원
            hidden_size: 숨은 상태의 차원
        """
        self.input_size = input_size
        self.hidden_size = hidden_size
        
        # 자비에르 초기화로 웨이트 초기화
        scale = 1.0 / np.sqrt(input_size + hidden_size)
        
        # 망각 문의 웨이트
        self.W_f = np.random.randn(hidden_size, input_size + hidden_size) * scale
        self.b_f = np.zeros((hidden_size, 1))
        
        # 입력 문의 웨이트
        self.W_i = np.random.randn(hidden_size, input_size + hidden_size) * scale
        self.b_i = np.zeros((hidden_size, 1))
        
        # 세포 후보의 웨이트
        self.W_c = np.random.randn(hidden_size, input_size + hidden_size) * scale
        self.b_c = np.zeros((hidden_size, 1))
        
        # 출력 문의 웨이트
        self.W_o = np.random.randn(hidden_size, input_size + hidden_size) * scale
        self.b_o = np.zeros((hidden_size, 1))
        
    def sigmoid(self, x):
        """시그모이드 엑티베이션 함수."""
        return 1 / (1 + np.exp(-np.clip(x, -500, 500)))
    
    def tanh(self, x):
        """쌍곡탄젠트 엑티베이션 함수."""
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
        LSTM 뉴럴 넷트웍을 초기화한다.
        
        인수:
            input_size: 입력 피처의 차원
            hidden_size: 숨은 상태의 차원
            output_size: 출력의 차원
        """
        self.cell = LSTMCell(input_size, hidden_size)
        
        # 출력 레이어의 웨이트
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
    
    # 문의 엑티베이션 좇기
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
    
    # 문의 엑티베이션 그려 보기
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
    plt.savefig('lstm_gates.png', dpi=150, bbox_inches='tight')
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
    main()
```

**출력:**

```
============================================================
LSTM: Long Short-Term Memory Networks
============================================================

Key Advantages of LSTM:
1. Solves vanishing gradient problem
2. Can learn long-term dependencies
3. Gates control information flow
4. Cell state acts as 'memory highway'

LSTM vs Traditional RNN:
- RNN: Simple recurrent connection
- LSTM: Gated architecture with cell state
- LSTM has 4x more parameters than RNN
============================================================
LSTM Gate Mechanics Demonstration
============================================================

✓ Gate activation plot saved as 'lstm_gates.png'

============================================================
LSTM Sequence Prediction Example: Sine Wave
============================================================

Input sequence length: 10
Hidden state dimension: (20, 1)
Number of LSTM parameters:
  - Forget gate: 440
  - Input gate: 440
  - Cell candidate: 440
  - Output gate: 440
  - Total: 1781

============================================================
LSTM demonstrations complete!
============================================================
```

## 2. 논의

LSTMCell은 이전 은닉 상태와 입력을 이어 붙여 망각·입력·출력 게이트와 셀 후보를 계산한다. 각 게이트와 후보에 하나의 바이어스를 두며, 셀 상태는 이전 상태의 보존 항과 새 후보의 입력 항을 더해 갱신한다. 은닉 상태는 출력 게이트와 셀 상태의 tanh로 만든다.

LSTM 클래스는 매 호출마다 은닉·셀 상태를 0으로 초기화하고 입력 벡터 목록을 순서대로 처리한다. 출력 레이어는 정규화되지 않은 선형 출력을 반환한다. 캐시를 만든다는 사실만으로 역전파가 구현된 것은 아니며, NumPy 배열에는 PyTorch 자동 미분이 적용되지 않는다.

게이트 시연은 무작위 초기 웨이트와 무작위 입력에서 각 게이트 원소의 평균을 그린다. 평균만으로 개별 셀의 정보를 얼마나 보존했는지 알 수 없고, 학습된 기억 전략으로 해석할 수도 없다. 난수 시드가 고정되지 않아 같은 수치나 곡선도 보장되지 않는다.

사인파 예제는 입력 창과 타깃을 만들지만 첫 창의 순전파와 파라메터 개수만 확인한다. 타깃을 이용한 학습이나 예측 오차 평가는 하지 않는다. 출력의 “그래디언트 소실 해결”과 “4배 파라메터”는 조건을 생략한 설명이다. LSTM은 그래디언트 전달을 돕지만 항상 소실을 없애지는 않으며, 4배 비교는 같은 은닉 크기와 바이어스 규약의 순환 셀 부분에 해당한다.

## 연습문제

<div class="drillbox" markdown>

**연습문제 1.** <span class="diff med" title="중간"></span>
이 NumPy LSTM 셀의 세 게이트와 셀 후보, 두 상태의 역할을 설명하라.

</div>

??? success "연습문제 1 풀이"
    망각 게이트는 이전 셀 상태의 직접 보존 항을, 입력 게이트는 새 후보의 입력 항을 조절한다. 출력 게이트는 셀 상태에서 은닉 출력으로 전달할 부분을 조절한다. 후보는 새로 쓸 값이며 별도의 게이트가 아니다. 셀 상태와 은닉 상태를 구분한다.

---

<div class="drillbox" markdown>

**연습문제 2.** <span class="diff med" title="중간"></span>
셀과 전체 모델이 기대하는 입력 형태를 설명하라.

</div>

??? success "연습문제 2 풀이"
    셀 입력은 $(x,1)$, 이전 은닉·셀 상태는 각각 $(h,1)$인 NumPy 열벡터다. 전체 모델은 이런 입력 벡터의 목록을 받는다. 배치 텐서나 PyTorch long ID를 그대로 받는 모델은 아니다.

---

<div class="drillbox" markdown>

**연습문제 3.** <span class="diff med" title="중간"></span>
입력 크기 1, 은닉 크기 20, 출력 크기 1인 예제의 파라메터 수를 계산하라.

</div>

??? success "연습문제 3 풀이"
    각 게이트·후보 묶음은 $20(1+20)+20=440$개이고 네 묶음은 1760개다. 출력 레이어의 웨이트 20개와 바이어스 1개를 더하면 1781개다. 이 구현은 각 묶음에 바이어스 하나를 둔다.

---

<div class="drillbox" markdown>

**연습문제 4.** <span class="diff hard" title="어려움"></span>
아래의 테스트 코드 조각을 검토하라. 정의되지 않은 클래스와 잘못된 이름, 입력 모양 가정을 찾아 이 페이지의 실제 함수에 맞는 점검 항목을 제안하라.

</div>

??? success "연습문제 4 풀이"
    아래 조각은 실제 실행 가능한 테스트가 아니다. 함수 이름과 모델 이름부터 이 예제에 맞게 작성해야 한다.
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
    이 페이지는 완성된 사용자 정의 모델 클래스를 제공하지 않는다. 실제 함수의 입력·출력 모양, 허용 자료형, 유한한 값과 경사의 유무를 기준으로 검사한다. 빈 입력을 처리하는 방식은 연산마다 다르며, 무조건 정상적인 로스나 경사가 나와야 한다고 가정하지 않는다.

## 정리하며

이 예제는 LSTM의 상태 갱신을 직접 계산하는 순전파 시연이다. 무작위 게이트 곡선이나 사인파 입력을 학습 결과로 해석하지 않는다. 그래디언트 전달의 이점은 전체 미분과 학습 조건을 고려해 설명해야 한다.
