# GRU 이론

NumPy로 GRU의 순전파를 구현하고 리셋·업데이트 게이트의 역할을 살펴본다. 별도의 셀 상태 없이 이전 은닉 상태와 후보를 결합한다. 코드는 게이트 동작과 파라메터 수를 시연하며 학습이나 정확도 비교를 수행하지 않는다.

## 1. 코드

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
- W는 웨이트 행렬이다
- b는 바이어스 벡터이다

LSTM과의 핵심 차이:
- 문이 3개가 아니라 2개이다 (출력 문이 없다)
- 세포 상태를 따로 두지 않는다
- 파라메터가 적다 (학습이 빠르다)
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
        GRU 세포의 파라메터를 초기화한다.
        
        인수:
            input_size: 입력 피처의 차원
            hidden_size: 숨은 상태의 차원
        """
        self.input_size = input_size
        self.hidden_size = hidden_size
        
        # 자비에르 초기화로 웨이트 초기화
        scale = 1.0 / np.sqrt(input_size + hidden_size)
        
        # 재설정 문의 웨이트
        self.W_r = np.random.randn(hidden_size, input_size + hidden_size) * scale
        self.b_r = np.zeros((hidden_size, 1))
        
        # 갱신 문의 웨이트
        self.W_z = np.random.randn(hidden_size, input_size + hidden_size) * scale
        self.b_z = np.zeros((hidden_size, 1))
        
        # 후보 숨은 상태의 웨이트
        self.W_h = np.random.randn(hidden_size, input_size + hidden_size) * scale
        self.b_h = np.zeros((hidden_size, 1))
        
    def sigmoid(self, x):
        """시그모이드 엑티베이션 함수."""
        return 1 / (1 + np.exp(-np.clip(x, -500, 500)))
    
    def tanh(self, x):
        """쌍곡탄젠트 엑티베이션 함수."""
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
    """여러 스텝을 다루는 GRU 구현."""
    
    def __init__(self, input_size, hidden_size, output_size):
        """
        GRU 뉴럴 넷트웍을 초기화한다.
        
        인수:
            input_size: 입력 피처의 차원
            hidden_size: 숨은 상태의 차원
            output_size: 출력의 차원
        """
        self.cell = GRUCell(input_size, hidden_size)
        
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
    
    # 문의 엑티베이션 좇기
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
    
    # 문의 엑티베이션 그려 보기
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
    plt.savefig('gru_gates.png', dpi=150, bbox_inches='tight')
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
    """LSTM과 GRU의 파라메터 수를 견준다."""
    print("\n" + "=" * 60)
    print("LSTM vs GRU Parameter Comparison")
    print("=" * 60)
    
    input_size = 100
    hidden_size = 200
    
    # LSTM의 파라메터
    lstm_params = 4 * (hidden_size * (input_size + hidden_size) + hidden_size)
    
    # GRU의 파라메터
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
    main()
```

**출력:**

```

============================================================
GRU: Gated Recurrent Unit Networks
============================================================

Key Features of GRU:
1. Simpler than LSTM (2 gates vs 3)
2. No separate cell state
3. Fewer parameters → faster training
4. Often performs similarly to LSTM

GRU Gates:
- Reset gate (r): Controls how much past info to forget
- Update gate (z): Controls how much to update
  * Acts as both forget and input gate
============================================================
GRU Gate Mechanics Demonstration
============================================================

✓ Gate activation plot saved as 'gru_gates.png'

============================================================
Understanding the Update Gate
============================================================

The update gate z_t controls:
h_t = (1 - z_t) ⊙ h_{t-1} + z_t ⊙ h̃_t

When z_t ≈ 0: Keep old hidden state (h_t ≈ h_{t-1})
When z_t ≈ 1: Use new candidate (h_t ≈ h̃_t)
When z_t ≈ 0.5: Mix equally

Example with h_{t-1} = [1, 2, 3] and h̃_t = [5, 6, 7]:

z ≈ 0 (Keep old):
  Result: [1.4 2.4 3.4]

z ≈ 1 (Use new):
  Result: [4.6 5.6 6.6]

z ≈ 0.5 (Mix):
  Result: [3. 4. 5.]

============================================================
LSTM vs GRU Parameter Comparison
============================================================

For input_size=100, hidden_size=200:
LSTM parameters: 240,800
GRU parameters:  180,600
Reduction:       25.0%

GRU has ~25% fewer parameters than LSTM
This means:
  ✓ Faster training
  ✓ Less memory usage
  ✓ Less prone to overfitting on small datasets

============================================================
GRU demonstrations complete!
============================================================
```

## 2. 논의

이 구현은 $h_t=(1-z_t)\odot h_{t-1}+z_t\odot\tilde h_t$를 사용한다. 따라서 z가 작으면 이전 상태를 유지하고 크면 후보를 더 반영한다. 문헌과 라이브러리에 따라 z의 보존·갱신 계수를 반대로 정의하므로 이름만으로 의미를 판단하면 안 된다.

리셋 게이트는 후보 계산에 들어가는 이전 상태에 먼저 적용된다. PyTorch GRU는 후보의 순환 선형 변환 뒤에 리셋을 적용하는 방식이며, 일반적으로 두 연산은 같지 않다. 웨이트 이름과 크기만 맞춘다고 이 NumPy 구현이 내장 GRU와 일치하지는 않는다.

GRU 클래스는 호출마다 상태를 0으로 시작하고 선형 출력과 은닉 상태 목록을 반환한다. 캐시는 저장하지만 역전파와 최적화는 없다. 게이트 평균 곡선은 무작위 순전파의 관찰이며 학습된 장기 기억을 보여 주지 않는다.

파라메터 비교는 각 계산 묶음에 바이어스 하나를 두는 규약이다. x=100,h=200이면 LSTM은 240800개, GRU는 180600개로 GRU가 25% 적다. 출력 레이어는 제외한다. 파라메터가 적다는 사실만으로 실제 학습 시간이 줄거나 과적합이 덜하다고 보장할 수 없다.

## 연습문제

<div class="drillbox" markdown>

**연습문제 1.** <span class="diff med" title="중간"></span>
이 페이지의 업데이트 게이트가 0과 1에 가까울 때 상태가 어떻게 되는가?

</div>

??? success "풀이"
    식 $h=(1-z)h_{prev}+z\tilde h$에서 z가 0에 가까우면 이전 상태, 1에 가까우면 후보를 주로 사용한다. PyTorch의 z는 이전 상태의 계수이므로 같은 이름을 쓴다는 이유로 해석을 그대로 옮기면 안 된다.

---

<div class="drillbox" markdown>

**연습문제 2.** <span class="diff med" title="중간"></span>
입력 형태와 시간에 따른 상태 초기화 규칙을 설명하라.

</div>

??? success "풀이"
    셀은 $(x,1)$ 입력과 $(h,1)$ 이전 상태의 NumPy 배열을 받는다. 전체 모델은 입력 목록을 순회하고 호출마다 상태를 0으로 초기화한다. 서로 다른 호출을 연속 스트림으로 이어가려면 상태를 외부에서 전달하는 설계가 추가로 필요하다.

---

<div class="drillbox" markdown>

**연습문제 3.** <span class="diff med" title="중간"></span>
이 NumPy 셀과 PyTorch 기본 GRU의 파라메터 수가 다른 이유를 설명하라.

</div>

??? success "풀이"
    NumPy 구현은 $3h(x+h+1)$개로 계산 묶음마다 바이어스 하나다. PyTorch 기본 셀은 입력·순환 경로의 바이어스를 각각 두어 $3h(x+h+2)$개다. 후보의 리셋 적용 위치도 다르므로 바이어스 개수만 같게 해서는 계산이 일치하지 않는다.

---

<div class="drillbox" markdown>

**연습문제 4.** <span class="diff hard" title="어려움"></span>
아래의 테스트 코드 조각을 검토하라. 정의되지 않은 클래스와 잘못된 이름, 입력 모양 가정을 찾아 이 페이지의 실제 함수에 맞는 점검 항목을 제안하라.

</div>

??? success "풀이"
    아래 조각은 실제 실행 가능한 테스트가 아니다. 함수 이름과 모델 이름부터 이 예제에 맞게 작성해야 한다.
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
    이 페이지는 완성된 사용자 정의 모델 클래스를 제공하지 않는다. 실제 함수의 입력·출력 모양, 허용 자료형, 유한한 값과 경사의 유무를 기준으로 검사한다. 빈 입력을 처리하는 방식은 연산마다 다르며, 무조건 정상적인 로스나 경사가 나와야 한다고 가정하지 않는다.

## 정리하며

GRU 식은 업데이트 계수의 정의와 리셋 적용 위치까지 확인해야 한다. 이 예제의 NumPy 순전파와 PyTorch 내장 GRU는 같은 계산이라고 가정할 수 없다. 파라메터 수와 실제 성능도 별도로 비교한다.
