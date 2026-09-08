# RMSNorm (제곱평균제곱근 층 정규화)

2019년 Zhang과 Sennrich가 소개한 RMSNorm은 제곱평균제곱근(RMS) 통계량만으로 정규화하여 평균을 빼는 단계를 없앤, 층 정규화의 간소화된 변형이다. 이 단순화는 성능을 유지하거나 오히려 높이면서 계산 비용을 줄여 주어, RMSNorm은 LLaMA, Mistral, Gemma 같은 현대의 대형 언어 모델에서 널리 쓰인다.

---

## 1. 수학적 정식화

### 표준 층 정규화 (되짚어 보기)

입력 $\mathbf{x} \in \mathbb{R}^n$에 대해 다음과 같다.

$$\text{LayerNorm}(\mathbf{x}) = \gamma \odot \frac{\mathbf{x} - \mu}{\sqrt{\sigma^2 + \epsilon}} + \beta$$

여기서 $\mu = \frac{1}{n}\sum_{i=1}^n x_i$이고 $\sigma^2 = \frac{1}{n}\sum_{i=1}^n (x_i - \mu)^2$이다.

### RMSNorm

RMSNorm은 평균 중심화를 없애 이를 간소화한다.

$$\text{RMS}(\mathbf{x}) = \sqrt{\frac{1}{n}\sum_{i=1}^n x_i^2}$$

$$\text{RMSNorm}(\mathbf{x}) = \gamma \odot \frac{\mathbf{x}}{\text{RMS}(\mathbf{x}) + \epsilon}$$

여기서 각 기호는 다음과 같다.

- $\gamma \in \mathbb{R}^n$은 학습 가능한 배율 매개변수이다
- $\epsilon$은 수치적 안정성을 위한 작은 상수이다
- (설계상) 편향 항 $\beta$이 없고 평균을 빼지도 않는다

### 핵심 차이

| 연산 | LayerNorm | RMSNorm |
|-----------|-----------|---------|
| 평균 빼기 | 있음 | **없음** |
| 분산 정규화 | 있음 | 대신 RMS를 쓴다 |
| 학습 가능한 편향 | 있음 ($\beta$) | **없음** |
| 계산 복잡도 | 통계량에 $O(2n)$ | 통계량에 $O(n)$ |

---

## 2. PyTorch 구현

### 바닥부터 만들기

```python
import torch
import torch.nn as nn

class RMSNorm(nn.Module):
    """
    제곱평균제곱근 층 정규화.
    
    평균을 빼지 않고 RMS 값으로 입력을 정규화한다.
    LLaMA, Mistral, Gemma를 비롯한 최신 대규모 언어 모델에서 쓰인다.
    """
    
    def __init__(self, dim, eps=1e-6):
        """
        인수:
            dim: 정규화할 차원 (보통 hidden_size)
            eps: 수치 안정성을 위한 작은 상수
        """
        super().__init__()
        
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))
    
    def _norm(self, x):
        """RMS 정규화를 계산한다."""
        # RMS 계산: sqrt(mean(x^2))
        rms = torch.sqrt(torch.mean(x ** 2, dim=-1, keepdim=True) + self.eps)
        return x / rms
    
    def forward(self, x):
        """
        인수:
            x: 모양이 (..., dim)인 입력 텐서
        
        반환값:
            같은 모양의 정규화된 텐서
        """
        # 정규화하고 배율 조정
        output = self._norm(x.float()).type_as(x)
        return output * self.weight

class RMSNormOptimized(nn.Module):
    """
    연산을 융합하여 최적화한 RMSNorm.
    """
    
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))
    
    def forward(self, x):
        # 효율을 위한 융합 계산
        # rsqrt가 sqrt + 나눗셈보다 빠르다
        norm_x = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return norm_x * self.weight
```

### LayerNorm과의 비교

```python
class LayerNormBaseline(nn.Module):
    """비교를 위한 표준 LayerNorm.

    RMSNorm과 견주기 위한 기준이다. 두 가지가 다르다.
      (1) LayerNorm은 평균을 빼서 중심을 옮기지만 RMSNorm은 옮기지 않는다.
      (2) LayerNorm은 편향(bias)을 두지만 RMSNorm은 두지 않는다.
    """

    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.eps = eps   # 분산이 0에 가까울 때 0으로 나누는 것을 막는다

        # 정규화로 모든 성분을 같은 자에 맞춘 뒤, 학습된 값으로 다시 늘이고
        # 옮긴다. 이것이 없으면 정규화가 표현력을 깎아 버린다
        self.weight = nn.Parameter(torch.ones(dim))    # 늘이기(초기값 1: 항등)
        self.bias = nn.Parameter(torch.zeros(dim))     # 옮기기(초기값 0: 항등)

    def forward(self, x):
        # dim=-1: 배치나 위치가 아니라 특성 축을 따라 정규화한다.
        # 표본 하나 안에서 끝나므로 배치 크기에 흔들리지 않는다
        # keepdim=True: 뒤에서 x와 방송(broadcast)되도록 축을 남겨 둔다
        mean = x.mean(dim=-1, keepdim=True)

        # unbiased=False: n-1이 아니라 n으로 나눈다.
        # 표본 분산을 추정하려는 것이 아니라 그저 자를 맞추려는 것이므로
        # 베셀 보정을 쓰지 않는다
        var = x.var(dim=-1, keepdim=True, unbiased=False)

        return self.weight * (x - mean) / torch.sqrt(var + self.eps) + self.bias
```

---

## 3. RMSNorm이 통하는 이유

### 가설: 다시 중심을 맞추는 일은 필요 없다

원 논문은 LayerNorm의 성공이 (평균을 빼는) 재중심화가 아니라 주로 **재배율 불변성**이라는 성질에서 온다고 가정한다.

**재배율 불변성**: 임의의 스칼라 $a$에 대해 다음이 성립한다.

$$\text{RMSNorm}(a \cdot \mathbf{x}) = \text{sign}(a) \cdot \text{RMSNorm}(\mathbf{x})$$

이 성질은 입력의 규모와 무관하게 경사의 흐름을 안정시킨다.

### 실험적 증거

```python
def compare_normalization_effects():
    """LayerNorm과 RMSNorm이 활성화에 미치는 영향을 비교한다."""
    
    torch.manual_seed(42)

    dim = 512
    ln = nn.LayerNorm(dim)   # 평균을 빼고 표준편차로 나눈다
    rms = RMSNorm(dim)       # 나누기만 하고 평균은 빼지 않는다

    # 네 가지 입력으로 두 정규화의 차이를 드러낸다.
    # 특히 "Shifted"가 핵심이다. 중심이 5만큼 밀려 있을 때
    # LayerNorm은 그 밀림을 없애지만 RMSNorm은 그대로 둔다
    test_cases = [
        ("Normal", torch.randn(32, dim)),            # 평균 0, 표준편차 1
        ("Shifted", torch.randn(32, dim) + 5.0),     # 중심만 옮긴 것
        ("Scaled", torch.randn(32, dim) * 10.0),     # 자만 키운 것
        ("Skewed", torch.exp(torch.randn(32, dim))), # 로그정규: 한쪽으로 크게 기운 것
    ]

    print("Comparison of LayerNorm vs RMSNorm:")
    print("=" * 60)

    for name, x in test_cases:
        ln_out = ln(x)
        rms_out = rms(x)

        # 출력의 평균을 나란히 찍는 것이 이 비교의 요점이다.
        # LayerNorm의 평균은 어떤 입력에서도 0에 가깝게 나오지만,
        # RMSNorm은 입력이 밀려 있으면 출력도 밀린 채로 남는다.
        # 그런데도 트랜스포머에서 잘 도는 까닭은, 학습된 가중치가
        # 그 밀림을 흡수할 수 있어 중심 맞추기가 굳이 필요 없기 때문이다
        print(f"\n{name} input (mean={x.mean():.2f}, std={x.std():.2f}):")
        print(f"  LayerNorm: mean={ln_out.mean():.4f}, std={ln_out.std():.4f}")
        print(f"  RMSNorm:   mean={rms_out.mean():.4f}, std={rms_out.std():.4f}")

compare_normalization_effects()
```

**출력:**
```
Comparison of LayerNorm vs RMSNorm:
============================================================

Normal input (mean=0.00, std=1.00):
  LayerNorm: mean=0.0000, std=1.0000
  RMSNorm:   mean=0.0001, std=0.9999

Shifted input (mean=5.02, std=1.00):
  LayerNorm: mean=0.0000, std=1.0000
  RMSNorm:   mean=0.9802, std=0.1951

Scaled input (mean=0.02, std=10.01):
  LayerNorm: mean=0.0000, std=1.0000
  RMSNorm:   mean=0.0002, std=1.0000

Skewed input (mean=1.64, std=2.14):
  LayerNorm: mean=0.0000, std=1.0000
  RMSNorm:   mean=0.5673, std=0.7407
```

참고: RMSNorm은 출력의 평균을 0으로 맞추지 않지만, 실무에서 이것이 성능을 해치지는 않는다.

---

## 4. 경사 분석

### 입력에 대한 경사

입력이 $\mathbf{x}$이고 출력이 $\mathbf{y}$인 RMSNorm에 대해 다음이 성립한다.

$$\frac{\partial \mathcal{L}}{\partial x_i} = \frac{\gamma_i}{\text{RMS}(\mathbf{x})} \left( \frac{\partial \mathcal{L}}{\partial y_i} - \frac{y_i}{n \cdot \text{RMS}(\mathbf{x})^2} \sum_{j=1}^n x_j \frac{\partial \mathcal{L}}{\partial y_j} \right)$$

### 간소화된 경사의 흐름

```python
class RMSNormWithGradientAnalysis(torch.autograd.Function):
    """분석을 위해 기울기를 명시적으로 계산하는 RMSNorm."""
    
    @staticmethod
    def forward(ctx, x, weight, eps):
        # RMS 계산
        rms = torch.sqrt(torch.mean(x ** 2, dim=-1, keepdim=True) + eps)
        x_norm = x / rms
        
        ctx.save_for_backward(x, weight, rms)
        ctx.eps = eps
        
        return x_norm * weight
    
    @staticmethod
    def backward(ctx, grad_output):
        x, weight, rms = ctx.saved_tensors
        
        n = x.shape[-1]
        
        # 가중치에 대한 기울기
        x_norm = x / rms
        grad_weight = (grad_output * x_norm).sum(dim=tuple(range(grad_output.dim()-1)))
        
        # x에 대한 기울기
        grad_x_norm = grad_output * weight
        
        # RMSNorm의 기울기 (LayerNorm보다 간단하다)
        grad_x = grad_x_norm / rms
        grad_x = grad_x - x_norm * (grad_x_norm * x_norm).mean(dim=-1, keepdim=True)
        
        return grad_x, grad_weight, None
```

---

## 5. 계산 효율

### 복잡도 비교

| 연산 | LayerNorm | RMSNorm |
|-----------|-----------|---------|
| 평균 계산 | O(n) | **0** |
| 분산/RMS 계산 | O(n) | O(n) |
| 평균 빼기 | O(n) | **0** |
| 나눗셈 | O(n) | O(n) |
| **통계량 연산 합계** | **축약 2n번** | **축약 1n번** |

### 성능 측정

```python
import time

def benchmark_normalization(batch_size=32, seq_len=512, dim=4096, num_iterations=1000):
    """LayerNorm과 RMSNorm의 성능을 견주어 잰다."""
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    x = torch.randn(batch_size, seq_len, dim, device=device)
    
    ln = nn.LayerNorm(dim).to(device)
    rms = RMSNorm(dim).to(device)
    
    # 워밍업
    for _ in range(100):
        _ = ln(x)
        _ = rms(x)
    
    if device.type == 'cuda':
        torch.cuda.synchronize()
    
    # LayerNorm 성능 측정
    start = time.time()
    for _ in range(num_iterations):
        _ = ln(x)
    if device.type == 'cuda':
        torch.cuda.synchronize()
    ln_time = time.time() - start
    
    # RMSNorm 성능 측정
    start = time.time()
    for _ in range(num_iterations):
        _ = rms(x)
    if device.type == 'cuda':
        torch.cuda.synchronize()
    rms_time = time.time() - start
    
    print(f"Benchmark Results ({device}):")
    print(f"  LayerNorm: {ln_time:.4f}s ({num_iterations} iterations)")
    print(f"  RMSNorm:   {rms_time:.4f}s ({num_iterations} iterations)")
    print(f"  Speedup:   {ln_time/rms_time:.2f}x")

# benchmark_normalization()
```

대표적인 속도 향상은 GPU에서 **1.1~1.3배**이며, CPU에서는 더 크다.

---

## 6. 현대 대형 언어 모델에서의 쓰임

### LLaMA 방식의 구조

```python
class LLaMABlock(nn.Module):
    """RMSNorm을 쓰는 LLaMA 트랜스포머 블록."""
    
    def __init__(self, dim, n_heads, n_kv_heads, ffn_dim, norm_eps=1e-5):
        super().__init__()
        
        # RMSNorm을 쓰는 사전 정규화
        self.attention_norm = RMSNorm(dim, eps=norm_eps)
        self.ffn_norm = RMSNorm(dim, eps=norm_eps)
        
        # 어텐션 (그룹 질의 어텐션)
        self.attention = GroupedQueryAttention(dim, n_heads, n_kv_heads)
        
        # 순방향 신경망 (SwiGLU)
        self.feed_forward = SwiGLU(dim, ffn_dim)
    
    def forward(self, x, freqs_cis=None, mask=None):
        # 사전 정규화 구조
        h = x + self.attention(
            self.attention_norm(x),
            freqs_cis=freqs_cis,
            mask=mask
        )
        out = h + self.feed_forward(self.ffn_norm(h))
        return out

class SwiGLU(nn.Module):
    """LLaMA에서 쓰는 SwiGLU 순방향 신경망."""
    
    def __init__(self, dim, hidden_dim):
        super().__init__()
        
        self.w1 = nn.Linear(dim, hidden_dim, bias=False)
        self.w2 = nn.Linear(hidden_dim, dim, bias=False)
        self.w3 = nn.Linear(dim, hidden_dim, bias=False)
    
    def forward(self, x):
        return self.w2(nn.functional.silu(self.w1(x)) * self.w3(x))

class GroupedQueryAttention(nn.Module):
    """LLaMA 2에서 쓰는 그룹 질의 어텐션."""
    
    def __init__(self, dim, n_heads, n_kv_heads):
        super().__init__()
        
        self.n_heads = n_heads
        self.n_kv_heads = n_kv_heads
        self.head_dim = dim // n_heads
        
        self.wq = nn.Linear(dim, n_heads * self.head_dim, bias=False)
        self.wk = nn.Linear(dim, n_kv_heads * self.head_dim, bias=False)
        self.wv = nn.Linear(dim, n_kv_heads * self.head_dim, bias=False)
        self.wo = nn.Linear(n_heads * self.head_dim, dim, bias=False)
    
    def forward(self, x, freqs_cis=None, mask=None):
        B, L, _ = x.shape
        
        q = self.wq(x).view(B, L, self.n_heads, self.head_dim)
        k = self.wk(x).view(B, L, self.n_kv_heads, self.head_dim)
        v = self.wv(x).view(B, L, self.n_kv_heads, self.head_dim)
        
        # 주어졌으면 회전 임베딩 적용
        if freqs_cis is not None:
            q, k = apply_rotary_emb(q, k, freqs_cis)
        
        # 그룹 질의 어텐션을 위해 KV 머리 늘리기
        n_rep = self.n_heads // self.n_kv_heads
        k = k.repeat_interleave(n_rep, dim=2)
        v = v.repeat_interleave(n_rep, dim=2)
        
        # 어텐션
        q, k, v = [t.transpose(1, 2) for t in (q, k, v)]
        scores = torch.matmul(q, k.transpose(-2, -1)) / (self.head_dim ** 0.5)
        
        if mask is not None:
            scores = scores + mask
        
        attn = torch.softmax(scores, dim=-1)
        out = torch.matmul(attn, v)
        
        out = out.transpose(1, 2).contiguous().view(B, L, -1)
        return self.wo(out)
```

### 완전한 LLaMA 모델

```python
class LLaMA(nn.Module):
    """간략한 LLaMA 모델 구조."""
    
    def __init__(self, vocab_size, dim, n_layers, n_heads, n_kv_heads, 
                 ffn_dim, norm_eps=1e-5, max_seq_len=2048):
        super().__init__()
        
        # 토큰 번호를 dim차원 벡터로 바꾼다. 이것이 모델의 입구다
        self.tok_embeddings = nn.Embedding(vocab_size, dim)

        # 같은 꼴의 블록을 n_layers번 쌓는다. 파이썬 리스트가 아니라
        # ModuleList를 써야 안에 든 매개변수가 model.parameters()에 잡힌다
        self.layers = nn.ModuleList([
            LLaMABlock(dim, n_heads, n_kv_heads, ffn_dim, norm_eps)
            for _ in range(n_layers)
        ])

        # 출력 사영 앞의 마지막 RMSNorm.
        # 블록마다 정규화가 입력 쪽에 걸리는(pre-norm) 구조라, 마지막
        # 블록을 나온 값은 아직 정규화되지 않았다. 그래서 여기서 한 번 더 건다
        self.norm = RMSNorm(dim, eps=norm_eps)

        # bias=False: 어휘가 수만 개라 편향만으로도 매개변수가 크게 늘고,
        # 뒤에 소프트맥스가 오므로 갈래마다의 상수는 확률을 거의 바꾸지 못한다
        self.output = nn.Linear(dim, vocab_size, bias=False)

        # 회전 임베딩(RoPE)의 회전각을 미리 셈해 둔다. 위치에만 달린 값이라
        # 학습되지 않으므로 한 번 만들어 두고 모든 층이 나눠 쓴다.
        # dim // n_heads: RoPE는 머리마다 따로 걸리므로 머리 하나의 차원이 필요하다
        self.freqs_cis = precompute_freqs_cis(dim // n_heads, max_seq_len)

    def forward(self, tokens):
        h = self.tok_embeddings(tokens)   # (배치, 길이) -> (배치, 길이, dim)

        # 블록을 차례로 지난다. 모양은 그대로이고 내용만 다듬어진다
        for layer in self.layers:
            h = layer(h, freqs_cis=self.freqs_cis)

        h = self.norm(h)        # 마지막 정규화
        return self.output(h)   # 어휘 크기만큼의 로짓을 낸다
```

---

## 7. 다른 정규화 기법과의 비교

```python
def comprehensive_comparison():
    """트랜스포머와 관련된 모든 정규화 방법을 비교한다."""
    
    torch.manual_seed(42)
    
    dim = 512
    batch_size = 8
    seq_len = 128
    
    x = torch.randn(batch_size, seq_len, dim)
    
    # 서로 다른 정규화 방식
    ln = nn.LayerNorm(dim)
    rms = RMSNorm(dim)
    
    ln_out = ln(x)
    rms_out = rms(x)
    
    print("Normalization Comparison for Transformer Hidden States")
    print("=" * 60)
    
    print(f"\nInput: mean={x.mean():.4f}, std={x.std():.4f}")
    
    print(f"\nLayerNorm:")
    print(f"  Output mean: {ln_out.mean():.6f}")
    print(f"  Output std:  {ln_out.std():.4f}")
    print(f"  Per-token mean std: {ln_out.mean(dim=-1).std():.6f}")
    
    print(f"\nRMSNorm:")
    print(f"  Output mean: {rms_out.mean():.6f}")
    print(f"  Output std:  {rms_out.std():.4f}")
    print(f"  Per-token mean std: {rms_out.mean(dim=-1).std():.6f}")
    
    # 매개변수 개수
    print(f"\nParameter count:")
    print(f"  LayerNorm: {sum(p.numel() for p in ln.parameters())}")
    print(f"  RMSNorm:   {sum(p.numel() for p in rms.parameters())}")

comprehensive_comparison()
```

**출력:**

```
Normalization Comparison for Transformer Hidden States
============================================================

Input: mean=-0.0015, std=1.0017

LayerNorm:
  Output mean: -0.000000
  Output std:  1.0000
  Per-token mean std: 0.000000

RMSNorm:
  Output mean: -0.001524
  Output std:  1.0000
  Per-token mean std: 0.044813

Parameter count:
  LayerNorm: 1024
  RMSNorm:   512
```

---

## 8. RMSNorm을 언제 쓸 것인가

### 알맞은 쓰임새

✅ **대형 언어 모델** (LLaMA, Mistral, Gemma)  
✅ **계산 효율이 중요할 때**  
✅ **아주 깊은 신경망** (정규화 층이 많을 때)  
✅ **선정규화 구조**  
✅ **평균 중심화가 결정적이지 않을 때**

### LayerNorm이 나을 수 있는 경우

❌ 모델이 작을 때 (절감이 미미하다)  
❌ 그 과제에서 평균 중심화의 이점이 알려져 있을 때  
❌ 기존 사전학습 모델과의 호환이 중요할 때

---

## 연습문제

**연습문제 1.**
RMSNorm 공식 $\text{RMSNorm}(x) = \frac{x}{\text{RMS}(x)} \cdot \gamma$을 유도하라. 여기서 $\text{RMS}(x) = \sqrt{\frac{1}{d}\sum_{i=1}^d x_i^2}$이다.

??? success "연습문제 1 풀이"
    RMSNorm은 중심을 맞추지 않고(평균을 빼지 않고) 입력의 제곱평균제곱근으로 정규화한다. 이로써 LayerNorm의 재중심화 단계가 사라져 계산이 약 7~10% 줄어든다. 학습 가능한 이득 $\gamma$이 정규화된 출력의 배율을 다시 조정한다.

---

**연습문제 2.**
RMSNorm과 LayerNorm의 계산 비용을 비교하라.

??? success "연습문제 2 풀이"
    LayerNorm은 평균 계산($O(d)$), 분산 계산($O(d)$), 정규화, 배율 조정, 이동으로 원소 $d$개에 걸쳐 5개의 연산을 한다. RMSNorm은 RMS 계산($O(d)$), 정규화, 배율 조정으로 3개의 연산을 한다. RMSNorm은 평균 계산과 편향 매개변수를 건너뛰어 계산을 약 30% 아낀다.

---

**연습문제 3.**
RMSNorm을 PyTorch로 구현하고 공식과 일치하는지 확인하라.

??? success "연습문제 3 풀이"
    ```python
    class RMSNorm(torch.nn.Module):
        def __init__(self, d, eps=1e-8):
            super().__init__()
            self.gamma = torch.nn.Parameter(torch.ones(d))
            self.eps = eps
        def forward(self, x):
            rms = torch.sqrt((x**2).mean(-1, keepdim=True) + self.eps)
            return x / rms * self.gamma
    ```

---

**연습문제 4.**
현대의 대형 언어 모델(예: LLaMA)에서 LayerNorm 대신 RMSNorm이 기본이 된 이유는 무엇인가?

??? success "연습문제 4 풀이"
    경험적으로 LayerNorm의 평균 중심화는 큰 트랜스포머에서 이득이 미미한 반면 계산은 늘린다. RMSNorm은 같은 수준의 학습 안정성과 최종 성능을 더 낮은 지연으로 이루는데, 규모가 커질수록(매개변수 수십억 개, 토큰 수조 개) 이것이 크게 중요해진다.

## 정리하며

RMSNorm은 다음과 같은 간소화된 정규화 기법이다.

1. 층 정규화에서 **평균 중심화를 없앤다**
2. 정규화에 분산 대신 **RMS를 쓴다**
3. **계산 비용을** 약 10~30% **줄인다**
4. 실무에서 **비슷한 성능을 유지한다**

핵심 성질은 다음과 같다.

- **평균을 빼지 않는다** — RMS로 배율만 조정한다
- **편향 매개변수** $\beta$ **가 없다**
- **계산이 더 빠르다** — 축약 연산이 하나 적다
- **현대 대형 언어 모델의 표준** — LLaMA, Mistral, Gemma 등

**참고 문헌**

1. Zhang, B., & Sennrich, R. (2019). Root Mean Square Layer Normalization. *NeurIPS*.

2. Touvron, H., et al. (2023). LLaMA: Open and Efficient Foundation Language Models. *arXiv preprint arXiv:2302.13971*.

3. Jiang, A. Q., et al. (2023). Mistral 7B. *arXiv preprint arXiv:2310.06825*.
