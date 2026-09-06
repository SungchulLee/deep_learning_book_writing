# 층 정규화

층 정규화의 구현과 예제. 층 정규화는 특징 차원에 걸쳐 입력을 정규화한다.

정규화 기법을 이해하는 것은 깊은 신경망을 효과적으로 만들고 학습시키는 데 필수적이다. 이 구현은 그 핵심 개념을 PyTorch로 보여주며, 현대적인 구조의 구성 요소를 직접 다뤄 볼 기회를 준다.

## 코드

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
        # 마지막 len(normalized_shape)개 차원에 대해 평균과 분산 계산
        axes = tuple(range(-len(self.normalized_shape), 0))
        
        mean = np.mean(x, axis=axes, keepdims=True)
        var = np.var(x, axis=axes, keepdims=True)
        
        # 정규화
        x_normalized = (x - mean) / np.sqrt(var + self.eps)
        
        # 배율 조정과 이동
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
        
        # 층 정규화
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
            
            # 새 은닉 상태 계산
            hidden = self.W_ih(x_t) + self.W_hh(hidden)
            
            # 층 정규화 적용
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
    print("✓ When batch statistics are unreliable")```

## 논의

이 구현은 4개의 클래스(`LayerNormNumPy`, `RNNWithLayerNorm`, `TransformerBlockWithLayerNorm`, `SimpleNetworkWithLayerNorm`)를 정의하며, 이들이 함께 작동하여 완전한 정규화 기법 구조를 이룬다. 각 클래스가 서로 다른 구성 요소를 감싸므로 코드가 모듈식이 되고 확장하기 쉬워진다. `forward` 메서드들이 PyTorch가 자동 미분에 사용하는 계산 그래프를 정의한다.

여기서 보인 방식은 더 복잡한 상황으로 자연스럽게 확장된다. 초매개변수, 구조의 변형, 여러 데이터셋을 두루 실험해 보면 이해가 깊어지고 학습 최적화 과제에 대한 실용적인 직관이 쌓인다.

## 연습문제

**연습문제 1.**
기본 초기화 상태의 `LayerNormNumPy`에서 학습 가능한 매개변수의 총 개수를 계산하라. 가중치와 편향을 모두 포함하여 층별로 나누어 세어라.

??? success "연습문제 1 풀이"
    `nn.Linear(in_features, out_features)` 각각에는 `in_features * out_features`개의 가중치 매개변수와 (`bias=False`가 아닌 한) `out_features`개의 편향 매개변수가 있다. `nn.Conv2d(in_c, out_c, k)`에는 `in_c * out_c * k * k`개의 가중치와 `out_c`개의 편향이 있다. `nn.Embedding(num, dim)`에는 `num * dim`개의 매개변수가 있다. 모든 층에 대해 더하면 된다. `sum(p.numel() for p in model.parameters())`로 확인할 수 있다.

---

**연습문제 2.**
어텐션 가중치 뒤에(값과 곱하기 전에) 드롭아웃 층을 추가하라. 학습 중에는 드롭아웃 비율 0.1을 쓴다. 어텐션 드롭아웃이 정칙화에 도움이 되는 이유를 설명하라.

??? success "연습문제 2 풀이"
    `__init__`에 `self.attn_dropout = nn.Dropout(0.1)`을 추가하고 소프트맥스 뒤에 적용한다. `attn_weights = self.attn_dropout(F.softmax(scores, dim=-1))`이다. 어텐션 드롭아웃은 학습 중에 일부 어텐션 가중치를 무작위로 0으로 만들어, 모델이 특정 토큰 사이의 관계에 지나치게 기대지 않게 한다. 이는 모델이 어텐션을 더 고르게 분산시키고 더 견고한 표현을 배우도록 북돋우며, 표준 드롭아웃이 뉴런의 공적응을 막는 것과 비슷하다.

---

**연습문제 3.**
자기 어텐션의 계산 복잡도를 열의 길이 $n$과 모델 차원 $d$의 함수로 설명하라. 이것이 왜 긴 열에 대해 Longformer나 Linformer 같은 구조의 동기가 되는가?

??? success "연습문제 3 풀이"
    표준 자기 어텐션은 $n \times n$ 어텐션 행렬을 계산하므로 시간 복잡도가 $O(n^2 d)$이고 어텐션 가중치에 $O(n^2)$의 메모리가 든다. 열이 길면(예: $n = 4096$) 감당하기 어려워진다. Longformer는 국소적인 미끄럼창 어텐션($w$이 창 크기일 때 $O(n \cdot w \cdot d)$)과 선택된 토큰에 대한 희소한 전역 어텐션을 결합한다. Linformer는 키와 값을 더 낮은 차원 $k \ll n$으로 사영하여 복잡도를 $O(n \cdot k \cdot d)$으로 줄인다. 둘 다 표현력을 조금 내주고 긴 입력에서의 실용적인 효율을 얻는다.

---

**연습문제 4.**
층이나 블록의 개수를 설정할 수 있도록 `LayerNormNumPy`를 확장하라. `__init__`에 `num_layers` 매개변수를 추가하고 `nn.ModuleList`로 깊이가 변하는 구조를 만들어라. 층 2개, 4개, 8개로 시험하라.

??? success "연습문제 4 풀이"
    고정되어 있던 층들을 다음으로 바꾼다.
    ```python
    self.layers = nn.ModuleList()
    for i in range(num_layers):
        self.layers.append(YourBlock(dim, ...))
    ```
    `forward` 메서드에서 `for layer in self.layers: x = layer(x)`처럼 순회한다. (평범한 파이썬 리스트가 아니라) `nn.ModuleList`를 써야 PyTorch가 모든 매개변수를 최적화 대상으로 등록한다. 시험은 다음과 같이 한다. `for n in [2, 4, 8]: model = LayerNormNumPy(num_layers=n); print(f'Layers={n}, params={sum(p.numel() for p in model.parameters()):,}')`.
