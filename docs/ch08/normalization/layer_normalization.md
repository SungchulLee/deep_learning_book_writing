# 레이어 정규화

레이어 정규화는 표본마다 지정한 마지막 축들의 통계량을 사용한다. NumPy 순전파와 MLP, RNN, 트랜스포머 예제를 통해 정규화 축과 구조 안의 위치를 구분한다.

## 1. 코드

```python
"""
층 정규화의 구현과 예제
================================================

층 정규화는 특징 차원에 걸쳐 입력을 정규화한다.
배치 정규화와 달리 배치 크기에 기대지 않아 RNN이나 작은 배치에 좋다.

논문: "Layer Normalization" (Ba, Kiros & Hinton, 2016)
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# ========================================================================
# 메인
# ========================================================================


class LayerNormNumPy:
    """
    NumPy로 바닥부터 구현한 층 정규화.
    배치 차원이 아니라 특징 차원에 걸쳐 정규화한다.
    """
    
    def __init__(self, normalized_shape, eps=1e-5):
        """
        인수:
            normalized_shape: 정규화할 특징의 모양
            eps: 수치 안정성을 위한 작은 상수
        """
        if isinstance(normalized_shape, int):
            normalized_shape = (normalized_shape,)
        
        self.normalized_shape = normalized_shape
        self.eps = eps
        
        # 학습 가능한 매개변수
        self.gamma = np.ones(normalized_shape)  # 배율 매개변수
        self.beta = np.zeros(normalized_shape)  # 이동 매개변수
        
    def forward(self, x):
        """
        층 정규화의 순전파.
        
        인수:
            x: 모양이 (batch_size, *normalized_shape)인 입력
            
        반환값:
            입력과 같은 모양의 정규화된 출력
        """
        # 마지막 len(normalized_shape)개 차원에 대해 평균과 분산 계산.
        # 뒤에서부터 세므로 배치 축은 결코 들어가지 않는다. 이것이
        # 배치 정규화와 갈리는 핵심이다. 배치 정규화는 배치 축을 따라
        # 통계를 내어 표본들이 서로 영향을 주지만, 층 정규화는 표본
        # 하나 안에서만 통계를 내므로 옆에 어떤 표본이 있든 결과가 같다.
        # 그래서 배치 크기가 1이어도 되고, 학습과 추론의 동작이 같아
        # 이동 통계를 따로 둘 필요도 없다
        axes = tuple(range(-len(self.normalized_shape), 0))
        
        mean = np.mean(x, axis=axes, keepdims=True)
        var = np.var(x, axis=axes, keepdims=True)
        
        # 정규화
        x_normalized = (x - mean) / np.sqrt(var + self.eps)
        
        # 배율 조정과 이동.
        # 정규화가 평균 0, 분산 1을 강요하는데 그것이 늘 좋은 것은
        # 아니므로, 신경망이 그 제약을 되돌릴 수 있게 손잡이를 준다.
        # gamma=1, beta=0에서 시작해 필요한 만큼만 배우며,
        # 극단적으로는 원래 분포를 그대로 되살릴 수도 있다.
        # RMSNorm에 beta가 없는 것은 애초에 평균을 빼지 않아
        # 되돌릴 이동이 없기 때문이다
        out = self.gamma * x_normalized + self.beta
        
        return out


class RNNWithLayerNorm(nn.Module):
    """
    층 정규화를 갖춘 RNN 셀.
    층 정규화는 RNN에 특히 쓸모 있다.
    """
    
    def __init__(self, input_size, hidden_size):
        super(RNNWithLayerNorm, self).__init__()
        
        self.hidden_size = hidden_size
        
        # 입력 변환과 은닉 변환
        self.W_ih = nn.Linear(input_size, hidden_size, bias=False)
        self.W_hh = nn.Linear(hidden_size, hidden_size, bias=False)
        
        # 층 정규화.
        # 순환 신경망이 층 정규화가 태어난 자리다. 배치 정규화를 여기
        # 쓰려면 시각마다 따로 통계를 두어야 하는데, 길이가 다른 계열이
        # 섞이면 감당이 되지 않는다. 층 정규화는 시각마다 그 표본
        # 하나만 보므로 그런 문제가 없다.
        # 정규화 층 하나를 모든 시각에 돌려 쓴다는 점도 눈여겨보라.
        # 학습되는 것은 gamma와 beta뿐이고 이동 통계가 없어서 가능하다
        self.ln = nn.LayerNorm(hidden_size)
        
    def forward(self, x, hidden=None):
        """
        인수:
            x: 모양이 (batch_size, seq_len, input_size)인 입력
            hidden: 처음 은닉 상태 (batch_size, hidden_size)
            
        반환값:
            outputs: 모든 은닉 상태 (batch_size, seq_len, hidden_size)
            hidden: 마지막 은닉 상태 (batch_size, hidden_size)
        """
        batch_size, seq_len, _ = x.size()
        
        if hidden is None:
            hidden = torch.zeros(batch_size, self.hidden_size, device=x.device)
        
        outputs = []
        
        for t in range(seq_len):
            x_t = x[:, t, :]
            
            # 새 은닉 상태 계산.
            # 두 Linear 모두 bias=False인데, 바로 뒤 층 정규화가 평균을
            # 빼 버려 편향이 할 일이 없기 때문이다. beta가 그 몫을 한다
            hidden = self.W_ih(x_t) + self.W_hh(hidden)
            
            # 층 정규화 적용.
            # 활성화 앞이라는 자리가 중요하다. 순환 신경망은 같은
            # 가중치를 시각마다 되풀이해 곱하므로 은닉 상태의 크기가
            # 계열이 길어질수록 부풀거나 사그라들기 쉬운데, 매 시각
            # 크기를 다시 맞추어 주면 그 폭주를 막을 수 있다
            hidden = self.ln(hidden)
            
            # 활성화 적용
            hidden = torch.tanh(hidden)
            
            outputs.append(hidden.unsqueeze(1))
        
        outputs = torch.cat(outputs, dim=1)
        
        return outputs, hidden


class TransformerBlockWithLayerNorm(nn.Module):
    """
    층 정규화를 쓰는 간략한 트랜스포머 블록.
    이것이 트랜스포머의 표준 정규화이다.
    """
    
    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1):
        super(TransformerBlockWithLayerNorm, self).__init__()
        
        # 다중 머리 어텐션
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        
        # 순방향 신경망
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        
        # 층 정규화 (사례 2개)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        
    def forward(self, x, attn_mask=None, key_padding_mask=None):
        """
        인수:
            x: 모양이 (seq_len, batch_size, d_model)인 입력
            
        반환값:
            입력과 같은 모양의 출력
        """
        # 잔차 연결과 층 정규화를 갖춘 자기 어텐션
        x2 = self.self_attn(x, x, x, attn_mask=attn_mask,
                           key_padding_mask=key_padding_mask)[0]
        x = x + self.dropout1(x2)
        x = self.norm1(x)
        
        # 잔차 연결과 층 정규화를 갖춘 순방향 신경망
        x2 = self.linear2(self.dropout(F.relu(self.linear1(x))))
        x = x + self.dropout2(x2)
        x = self.norm2(x)
        
        return x


class SimpleNetworkWithLayerNorm(nn.Module):
    """
    층 정규화를 갖춘 간단한 순방향 신경망.
    """
    
    def __init__(self, input_size=784, hidden_size=256, num_classes=10):
        super(SimpleNetworkWithLayerNorm, self).__init__()
        
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.ln1 = nn.LayerNorm(hidden_size)
        
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.ln2 = nn.LayerNorm(hidden_size)
        
        self.fc3 = nn.Linear(hidden_size, num_classes)
        
    def forward(self, x):
        if len(x.shape) > 2:
            x = x.view(x.size(0), -1)
        
        x = self.fc1(x)
        x = self.ln1(x)
        x = F.relu(x)
        
        x = self.fc2(x)
        x = self.ln2(x)
        x = F.relu(x)
        
        x = self.fc3(x)
        
        return x


def demonstrate_layer_norm():
    """
    층 정규화가 어떻게 작동하는지 보인다.
    """
    print("=" * 60)
    print("Layer Normalization Demonstration")
    print("=" * 60)
    
    np.random.seed(42)
    
    # 예시 데이터 만들기
    batch_size = 4
    num_features = 5
    
    # 표본마다 척도가 다른 특징을 갖는다
    x = np.array([
        [1.0, 2.0, 3.0, 4.0, 5.0],      # 표본 1
        [10.0, 20.0, 30.0, 40.0, 50.0],  # 표본 2 (척도가 크다)
        [0.1, 0.2, 0.3, 0.4, 0.5],      # 표본 3 (척도가 작다)
        [5.0, 5.0, 5.0, 5.0, 5.0],      # 표본 4 (상수)
    ])
    
    print("\nOriginal data:")
    print(x)
    print(f"\nMean per sample: {np.mean(x, axis=1)}")
    print(f"Std per sample:  {np.std(x, axis=1)}")
    
    # 층 정규화 적용
    ln = LayerNormNumPy(num_features)
    x_normalized = ln.forward(x)
    
    print("\nAfter Layer Normalization:")
    print(x_normalized)
    print(f"\nMean per sample: {np.mean(x_normalized, axis=1)}")
    print(f"Std per sample:  {np.std(x_normalized, axis=1)}")
    
    print("\nKey observations:")
    print("- Each SAMPLE is normalized independently")
    print("- Mean ≈ 0 and Std ≈ 1 for EACH sample")
    print("- Works well for variable batch sizes")
    print("- No dependence on other samples in the batch")


def compare_batchnorm_layernorm():
    """
    배치 정규화와 층 정규화를 비교한다.
    """
    print("\n" + "=" * 60)
    print("Batch Norm vs Layer Norm Comparison")
    print("=" * 60)
    
    np.random.seed(42)
    torch.manual_seed(42)
    
    # 예시 데이터 만들기
    x = torch.randn(8, 10)  # 표본 8개, 특징 10개
    
    print("\nOriginal data shape:", x.shape)
    print("Original data:\n", x[:2])  # 처음 표본 2개 보이기
    
    # 배치 정규화
    bn = nn.BatchNorm1d(10)
    bn.eval()  # 이동 통계를 쓰지 않으려고 평가 모드를 쓴다
    x_bn = bn(x)
    
    # 층 정규화
    ln = nn.LayerNorm(10)
    x_ln = ln(x)
    
    print("\nAfter Batch Normalization:")
    print("Mean per feature (across batch):", x_bn.mean(dim=0).detach().numpy()[:5])
    print("Mean per sample (across features):", x_bn.mean(dim=1).detach().numpy()[:3])
    
    print("\nAfter Layer Normalization:")
    print("Mean per feature (across batch):", x_ln.mean(dim=0).detach().numpy()[:5])
    print("Mean per sample (across features):", x_ln.mean(dim=1).detach().numpy()[:3])
    
    print("\n" + "-" * 60)
    print("Key Differences:")
    print("-" * 60)
    print("Batch Normalization:")
    print("  - Normalizes across the BATCH dimension")
    print("  - Each feature is normalized using batch statistics")
    print("  - Depends on batch size (problematic for small batches)")
    print("  - Different behavior in train vs eval mode")
    print("  - Best for: CNNs, large batches, feedforward networks")
    
    print("\nLayer Normalization:")
    print("  - Normalizes across the FEATURE dimension")
    print("  - Each sample is normalized independently")
    print("  - Independent of batch size")
    print("  - Same behavior in train and eval mode")
    print("  - Best for: RNNs, Transformers, small batches")


def demonstrate_small_batch_problem():
    """
    왜 작은 배치에서 층 정규화가 나은지 보인다.
    """
    print("\n" + "=" * 60)
    print("Small Batch Problem")
    print("=" * 60)
    
    torch.manual_seed(42)
    
    # 작은 배치
    x_small = torch.randn(2, 10)
    
    # 큰 배치
    x_large = torch.randn(64, 10)
    
    # 배치 정규화
    bn = nn.BatchNorm1d(10)
    bn.train()
    
    # 층 정규화
    ln = nn.LayerNorm(10)
    
    print("\nWith Batch Normalization:")
    with torch.no_grad():
        out_small_bn = bn(x_small)
        out_large_bn = bn(x_large)
    
    print(f"Small batch (n=2) std: {out_small_bn.std():.4f}")
    print(f"Large batch (n=64) std: {out_large_bn.std():.4f}")
    
    print("\nWith Layer Normalization:")
    with torch.no_grad():
        out_small_ln = ln(x_small)
        out_large_ln = ln(x_large)
    
    print(f"Small batch (n=2) std: {out_small_ln.std():.4f}")
    print(f"Large batch (n=64) std: {out_large_ln.std():.4f}")
    
    print("\nObservation:")
    print("- BatchNorm is sensitive to batch size")
    print("- LayerNorm is consistent across batch sizes")
    print("- Use LayerNorm when batch size is small or variable")


if __name__ == "__main__":
    demonstrate_layer_norm()
    compare_batchnorm_layernorm()
    demonstrate_small_batch_problem()
    
    print("\n" + "=" * 60)
    print("When to use Layer Normalization:")
    print("=" * 60)
    print("✓ RNNs and LSTMs")
    print("✓ Transformers (standard choice)")
    print("✓ Small batch sizes")
    print("✓ Online learning (batch size = 1)")
    print("✓ When batch statistics are unreliable")
```

**출력:**

```
============================================================
Layer Normalization Demonstration
============================================================

Original data:
[[ 1.   2.   3.   4.   5. ]
 [10.  20.  30.  40.  50. ]
 [ 0.1  0.2  0.3  0.4  0.5]
 [ 5.   5.   5.   5.   5. ]]

Mean per sample: [ 3.  30.   0.3  5. ]
Std per sample:  [ 1.41421356 14.14213562  0.14142136  0.        ]

After Layer Normalization:
[[-1.41421003 -0.70710501  0.          0.70710501  1.41421003]
 [-1.41421353 -0.70710676  0.          0.70710676  1.41421353]
 [-1.41386014 -0.70693007  0.          0.70693007  1.41386014]
 [ 0.          0.          0.          0.          0.        ]]

Mean per sample: [-4.44089210e-17  4.44089210e-17  1.33226763e-16  0.00000000e+00]
Std per sample:  [0.9999975  0.99999998 0.99975009 0.        ]

Key observations:
- Each SAMPLE is normalized independently
- Mean ≈ 0 and Std ≈ 1 for EACH sample
- Works well for variable batch sizes
- No dependence on other samples in the batch

============================================================
Batch Norm vs Layer Norm Comparison
============================================================

Original data shape: torch.Size([8, 10])
Original data:
 tensor([[ 1.9269,  1.4873,  0.9007, -2.1055,  0.6784, -1.2345, -0.0431, -1.6047,
         -0.7521,  1.6487],
        [-0.3925, -1.4036, -0.7279, -0.5594, -0.7688,  0.7624,  1.6423, -0.1596,
         -0.4974,  0.4396]])

After Batch Normalization:
Mean per feature (across batch): [-0.26340058  0.8332318  -0.39194098  0.08024922  0.04736656]
Mean per sample (across features): [ 0.09021159 -0.16648701  0.71324587]

After Layer Normalization:
Mean per feature (across batch): [-0.45110974  0.748088   -0.5378934   0.16673607 -0.05450348]
Mean per sample (across features): [-3.57627883e-08 -1.45286325e-08  7.15255766e-08]

------------------------------------------------------------
Key Differences:
------------------------------------------------------------
Batch Normalization:
  - Normalizes across the BATCH dimension
  - Each feature is normalized using batch statistics
  - Depends on batch size (problematic for small batches)
  - Different behavior in train vs eval mode
  - Best for: CNNs, large batches, feedforward networks

Layer Normalization:
  - Normalizes across the FEATURE dimension
  - Each sample is normalized independently
  - Independent of batch size
  - Same behavior in train and eval mode
  - Best for: RNNs, Transformers, small batches

============================================================
Small Batch Problem
============================================================

With Batch Normalization:
Small batch (n=2) std: 1.0254
Large batch (n=64) std: 1.0008

With Layer Normalization:
Small batch (n=2) std: 1.0260
Large batch (n=64) std: 1.0008

Observation:
- BatchNorm is sensitive to batch size
- LayerNorm is consistent across batch sizes
- Use LayerNorm when batch size is small or variable

============================================================
When to use Layer Normalization:
============================================================
✓ RNNs and LSTMs
✓ Transformers (standard choice)
✓ Small batch sizes
✓ Online learning (batch size = 1)
✓ When batch statistics are unreliable
```

## 2. 논의

LayerNormNumPy는 normalized_shape에 해당하는 마지막 축에서 평균·분산을 계산한다. gamma와 beta도 그 모양을 가지지만 NumPy 배열이므로 PyTorch의 자동 미분 매개변수는 아니다.

RNN 예제는 각 시점의 은닉 특징을 정규화한다. 트랜스포머 예제는 기본 MultiheadAttention의 `(시퀀스, 배치, 특징)` 순서를 사용하고, 잔차 합 뒤에 레이어 정규화를 적용하는 post-norm 구조이다.

레이어 정규화는 현재 표본의 통계량을 학습과 평가에서 모두 사용한다. 기본적으로 이동 평균·분산을 누적하지 않으며, 배치가 작아도 같은 표본의 정규화 통계가 다른 표본에 의존하지 않는다. 모델의 드롭아웃 같은 다른 연산은 여전히 모드에 따라 달라진다.

## 연습문제

<div class="drillbox" markdown>

**연습문제 1.** <span class="diff easy" title="쉬움"></span>
normalized_shape=D일 때 배율과 이동 계수의 수와 상태의 종류를 설명하라.

</div>

??? success "연습문제 1 풀이"
    gamma와 beta에 각각 $D$개, 합계 $2D$개가 있다. PyTorch에서는 기본 설정에서 학습 매개변수로 등록되지만 NumPy 시연에서는 일반 배열이다. 배치 정규화의 running_mean·running_var와 같은 이동 통계량은 없다.

---

<div class="drillbox" markdown>

**연습문제 2.** <span class="diff med" title="중간"></span>
입력이 `(T, B, D)`이고 LayerNorm(D)일 때 정규화 축은 무엇인가?

</div>

??? success "연습문제 2 풀이"
    각 시점과 표본마다 마지막 D개 특징에서 평균과 분산을 구한다. 시간 T나 배치 B 전체를 섞지 않는다. normalized_shape를 바꾸면 정규화 축과 아핀 계수의 모양도 달라진다.

---

<div class="drillbox" markdown>

**연습문제 3.** <span class="diff med" title="중간"></span>
본문 트랜스포머 블록의 post-norm과 pre-norm을 구분하라.

</div>

??? success "연습문제 3 풀이"
    본문은 하위층 출력과 잔차를 더한 뒤 정규화한다. pre-norm은 하위층의 입력을 먼저 정규화하고 그 출력을 잔차에 더한다. 정규화 위치를 바꾸는 것은 단순한 문법 차이가 아니며 경사 경로도 달라진다.

---

<div class="drillbox" markdown>

**연습문제 4.** <span class="diff hard" title="어려움"></span>
아래 풀이의 `ModuleList` 코드가 원래 모델에 바로 적용될 수 있는지 검토하라. 입력·출력 차원, 활성화, 생성자와 `forward`에서 필요한 수정을 설명하라.

</div>

??? success "연습문제 4 풀이"
    아래 코드는 깊이를 설정하는 아이디어를 위한 조각이다. 원래 모델의 생성자와 텐서 차원을 반영한 완성 구현은 아니다.
    ```python
    self.layers = nn.ModuleList()
    for i in range(num_layers):
        self.layers.append(YourBlock(dim, ...))
    ```
    `ModuleList`는 하위 모듈을 등록하지만 실행 순서나 활성화를 자동으로 정하지 않는다. 입력층과 출력층은 과제의 차원을 유지하고, 은닉층 사이에 필요한 비선형성을 넣어야 한다. 선형층만 반복하면 아핀 함수 하나로 합쳐진다. 원래 클래스에 없는 `num_layers` 인수를 그대로 넘길 수 없으므로 생성자와 `forward`를 함께 수정해야 한다.

## 정리하며

레이어 정규화는 배치가 아닌 지정 특징 축을 사용한다. 정규화 축, 아핀 계수의 모양, 잔차 합 전후의 위치를 구분하면 MLP와 시퀀스 모델의 구현을 읽기 쉽다.
