# 자기 되돌이 인수 나누기
## 들어가며

자기 되돌이 모델은 만들어 내는 모델의 가장 바탕이 되는 방식 가운데 하나이다. 핵심 생각은 아름답도록 단순하다. 곧 결합 확률 분포를 조건부 분포의 곱으로 나누어 나타내며, 변수마다 앞선 모든 변수를 바탕으로 미리 헤아린다. 이 장은 옛부터의 시계열 모델에서 요즘의 큰 말 모델까지 모든 자기 되돌이 만들어 내는 모델을 받치는 수학의 바탕을 세운다.

## 확률의 사슬 규칙

### 수학 바탕

아무 변수 차례 $\mathbf{x} = (x_1, x_2, \ldots, x_n)$의 결합 분포에 대해 확률의 사슬 규칙은 정확한 인수 나누기를 준다.

$$P(\mathbf{x}) = P(x_1, x_2, \ldots, x_n) = \prod_{i=1}^{n} P(x_i | x_1, x_2, \ldots, x_{i-1})$$

이는 어림이 아니라 어떤 확률 분포에서도 참인 항등식이다. 첫 항 $P(x_1)$은 조건이 없고 뒤따르는 항은 저마다 앞선 모든 변수를 조건으로 삼는다.

**보기: 변수 셋**

차례 $(x_1, x_2, x_3)$에 대해:

$$P(x_1, x_2, x_3) = P(x_1) \cdot P(x_2|x_1) \cdot P(x_3|x_1, x_2)$$

이 나누기는 바탕 자료 분포와 상관없이 정확하다.

### 차례와 그 뜻

사슬 규칙은 변수의 차례를 골라야 한다. 차례가 달라도 수학으로는 같은 인수 나누기가 나오지만 셈의 성질은 다를 수 있다.

$$P(x_1, x_2, x_3) = P(x_1) P(x_2|x_1) P(x_3|x_1, x_2)$$

$$P(x_1, x_2, x_3) = P(x_3) P(x_2|x_3) P(x_1|x_2, x_3)$$

둘 다 옳지만 차례를 어떻게 고르느냐가 다음에 영향을 준다.

1. **자연스러운 짜임 맞춤**: 차례 자료(글, 소리, 시계열)에서는 왼쪽에서 오른쪽 차례가 때의 인과와 맞는다
2. **셈의 효율**: 어떤 차례는 나란한 셈을 더 효율 좋게 해 준다
3. **미리 든 치우침**: 차례는 매임에 대한 가정을 은근히 담는다

그림에서는 PixelCNN이 쓰는 가로 훑기(왼쪽에서 오른쪽, 위에서 아래)나 거친 데서 고운 데로 다루는 여러 잣수 방식이 흔한 차례이다.

## 자기 되돌이 성질

### 정의

모델이 조건부 분포 $P(x_i | x_{<i})$마다 배울 수 있는 매개변수로 나타내면 그 모델은 **자기 되돌이**이다. 여기서 $x_{<i} = (x_1, \ldots, x_{i-1})$은 자리 $i$ 앞의 모든 변수를 뜻한다.

핵심 통찰은 다음을 나타낸다는 것이다.

$$P_\theta(\mathbf{x}) = \prod_{i=1}^{n} P_\theta(x_i | x_{<i})$$

여기서 $\theta$은 배운 매개변수를 나타낸다. 조건부 분포는 저마다 $x_{<i}$을 들임으로 받는 신경망으로 나타낼 수 있다.

### 왜 "자기 되돌이"인가?

이 말은 시계열 살피기에서 왔다. 옛부터의 AR(p) 모델에서:

$$x_t = \sum_{j=1}^{p} \phi_j x_{t-j} + \epsilon_t$$

변수 $x_t$이 제 지난 값에 "되돌아 기댄다". 그래서 "auto"(스스로) + "regressive"이다. 신경 자기 되돌이 모델은 이를 비선형 매임과 아무 자료 갈래로 넓힌다.

### 다룰 수 있는 가능도

자기 되돌이 모델의 결정적인 장점은 **다룰 수 있는 가능도 셈하기**이다. 자료 점 $\mathbf{x}$이 주어지면 그 정확한 로그 가능도를 셈할 수 있다.

$$\log P_\theta(\mathbf{x}) = \sum_{i=1}^{n} \log P_\theta(x_i | x_{<i})$$

이는 항 $n$개의 합이며 각 항은 신경망이 내놓은 것이다. 그래서 다음이 된다.

1. **정확한 밀도 따지기**: 변분 자기 부호기나 맞겨루기 만들개와 달리 어떤 자료 점의 확률도 정확히 셈할 수 있다
2. **최대 가능도 익히기**: $\log P_\theta(\mathbf{x})$을 곧바로 가장 좋게 한다
3. **모델 견주기**: 헷갈림도와 차원마다 비트가 뜻있는 잣대이다

## 조건부 분포를 매개변수로 나타내기

### 띄엄띄엄한 변수

낱말 수가 $V$인 띄엄띄엄한 변수(예컨대 글자, 토큰, 화소 밝기)에서는 조건부 분포가 흔히 범주형이다.

$$P_\theta(x_i | x_{<i}) = \text{Categorical}(x_i; \pi_\theta(x_{<i}))$$

여기서 $\pi_\theta(x_{<i}) \in \mathbb{R}^V$은 신경망이 내놓은 것에 소프트맥스를 써서 얻은 확률 벡터이다.

$$\pi_\theta(x_{<i}) = \text{softmax}(f_\theta(x_{<i}))$$

**PyTorch 짜기:**

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class DiscreteARModel(nn.Module):
    """
    띄엄띄엄한 차례용 자기 되돌이 모델.
    
    자리마다 낱말에 대한 범주형 분포를 헤아린다.
    """
    
    def __init__(self, vocab_size: int, embedding_dim: int, hidden_dim: int):
        super().__init__()
        self.vocab_size = vocab_size
        
        # 박아 넣기 층: 띄엄띄엄한 토큰을 이어진 벡터로 바꾼다
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        
        # 맥락 부호기: 앞 토큰을 다룬다
        self.encoder = nn.GRU(
            embedding_dim, 
            hidden_dim, 
            batch_first=True
        )
        
        # 내놓기 쏘기: 숨은 상태 -> 낱말 로짓
        self.output_proj = nn.Linear(hidden_dim, vocab_size)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        자리마다 다음 토큰 헤아리기의 로짓을 셈한다.
        
        인수:
            x: 들임 차례 [묶음 크기, 차례 길이]
            
        반환값:
            로짓 [묶음 크기, 차례 길이, 낱말 수]
        """
        # 들임 토큰을 박아 넣는다
        embedded = self.embedding(x)  # [묶음, 차례 길이, 박아 넣기 차원]
        
        # 맥락 부호화
        hidden_states, _ = self.encoder(embedded)  # [묶음, 차례 길이, 숨은 차원]
        
        # 어휘로 사영한다
        logits = self.output_proj(hidden_states)  # [묶음, 차례 길이, 낱말 수]
        
        return logits
    
    def compute_loss(self, x: torch.Tensor) -> torch.Tensor:
        """
        음의 로그 가능도 손실 셈하기.
        
        스승 밀어 넣기를 쓴다. 곧 x[0:t]으로 x[t]을 헤아린다.
        """
        # 다음 토큰 맞히기를 위해 민다
        inputs = x[:, :-1]   # 마지막 토큰만 뺀 모두
        targets = x[:, 1:]   # 첫 토큰만 뺀 모두
        
        # 예측을 얻는다
        logits = self.forward(inputs)
        
        # 어긋 엔트로피 손실 = 음의 로그 가능도
        loss = F.cross_entropy(
            logits.reshape(-1, self.vocab_size),
            targets.reshape(-1)
        )
        
        return loss
```

### 이어진 변수

이어진 자료(예컨대 소리 물결꼴, 실수 값 특징)에서 흔한 고르기는 다음과 같다.

**정규 분포 내놓기:**

$$P_\theta(x_i | x_{<i}) = \mathcal{N}(x_i; \mu_\theta(x_{<i}), \sigma^2_\theta(x_{<i}))$$

신경망이 평균과 흩어짐 매개변수를 내놓는다.

**로지스틱 섞기:**

$$P_\theta(x_i | x_{<i}) = \sum_{k=1}^{K} \pi_k \cdot \text{Logistic}(x_i; \mu_k, s_k)$$

가둔 이어진 값을 나타내려 WaveNet과 PixelCNN++에서 쓴다.

**띄엄띄엄하게 만들기:**

쓸모 있는 방식은 이어진 값을 띄엄띄엄하게 만들고(예컨대 8비트 소리 → 256단계) 범주형 분포를 쓰는 것이다.

```python
class ContinuousARModel(nn.Module):
    """
    정규 분포 내놓기를 갖춘 이어진 차례용 자기 되돌이 모델.
    """
    
    def __init__(self, input_dim: int, hidden_dim: int):
        super().__init__()
        
        self.encoder = nn.GRU(input_dim, hidden_dim, batch_first=True)
        
        # 평균과 로그 흩어짐을 내놓는다
        self.mean_proj = nn.Linear(hidden_dim, input_dim)
        self.logvar_proj = nn.Linear(hidden_dim, input_dim)
    
    def forward(self, x: torch.Tensor):
        """
        자리마다 정규 분포 매개변수를 셈한다.
        
        인수:
            x: 들임 차례 [묶음 크기, 차례 길이, 들임 차원]
            
        반환값:
            mean: [묶음 크기, 차례 길이, 들임 차원]
            logvar: [묶음 크기, 차례 길이, 들임 차원]
        """
        hidden_states, _ = self.encoder(x)
        
        mean = self.mean_proj(hidden_states)
        logvar = self.logvar_proj(hidden_states)
        
        return mean, logvar
    
    def compute_loss(self, x: torch.Tensor) -> torch.Tensor:
        """
        정규 분포의 음의 로그 가능도를 셈한다.
        """
        inputs = x[:, :-1, :]
        targets = x[:, 1:, :]
        
        mean, logvar = self.forward(inputs)
        
        # 정규 분포 음의 로그 가능도: 0.5 * (log(2π) + logvar + (x-μ)²/var)
        nll = 0.5 * (
            logvar + 
            (targets - mean).pow(2) / logvar.exp() +
            torch.log(torch.tensor(2 * torch.pi))
        )
        
        return nll.mean()
```

## 자기 되돌이 모델 익히기

### 최대 가능도 어림

익히기는 음의 로그 가능도를 가장 작게 한다.

$$\mathcal{L}(\theta) = -\mathbb{E}_{\mathbf{x} \sim p_{\text{data}}} \left[ \log P_\theta(\mathbf{x}) \right] = -\mathbb{E}_{\mathbf{x}} \left[ \sum_{i=1}^{n} \log P_\theta(x_i | x_{<i}) \right]$$

이는 서로 매이지 않은 항으로 나뉘어 효율 좋은 작은 묶음 익히기를 할 수 있다.

### 스승 강제

익히는 동안 **스승 밀어 넣기**를 쓴다. 곧 모델이 $x_i$을 헤아릴 때 제가 헤아린 값이 아니라 참 앞 토큰 $x_{<i}$을 들임으로 받는다. 그래서 다음이 가능하다.

1. **나란히 하기**: 모든 자리를 한꺼번에 셈할 수 있다
2. **안정된 기울기**: 띄엄띄엄한 뽑기를 지나는 기울기 흐름이 없다
3. **효율 좋은 익히기**: 온 차례에 앞먹임 한 번

```python
def train_step(model, optimizer, batch):
    """
    스승 밀어 넣기로 하는 익히기 걸음 하나.
    
    모델은 자리마다 참 앞 토큰을 본다.
    """
    optimizer.zero_grad()
    
    # 스승 밀어 넣기: 참 들임을 쓴다
    loss = model.compute_loss(batch)
    
    loss.backward()
    optimizer.step()
    
    return loss.item()
```

### 드러남 치우침

스승 밀어 넣기는 **드러남 치우침**이라 부르는 익히기와 시험 사이의 어긋남을 만든다. 곧 익힐 때는 모델이 참 자료를 보지만 만들어 낼 때는 제가 헤아린 어긋날 수 있는 값을 본다. 그래서 만들어 내는 동안 어긋남이 쌓일 수 있다.

덜어 내는 전략은 다음과 같다:

1. **차례 잡은 뽑기**: 익히는 동안 참 토큰을 모델이 헤아린 것으로 차츰 바꾼다
2. **빔 찾기**: 만들어 내는 동안 여러 가설을 살핀다
3. **차례 수준 익히기**: 만들어 낸 차례의 잣대를 가장 좋게 한다

## 자기 되돌이 모델에서 뽑기

### 조상 뽑기

여느 만들어 내기 절차는 조건부 분포에서 차례대로 뽑는다.

$$x_1 \sim P_\theta(x_1)$$

$$x_2 \sim P_\theta(x_2 | x_1)$$

$$\vdots$$

$$x_n \sim P_\theta(x_n | x_{<n})$$

이는 배운 분포에서 정확한 표본을 낸다.

```python
@torch.no_grad()
def sample(model, start_token, max_length, temperature=1.0):
    """
    조상 뽑기로 차례를 만든다.
    
    인수:
        model: 익힌 자기 되돌이 모델
        start_token: 첫 토큰
        max_length: 최대 차례 길이
        temperature: 표본 추출의 온도 (높을수록 더 무작위)
    
    반환값:
        만든 차례
    """
    model.eval()
    
    # 시작 토큰으로 첫자리매김한다
    generated = [start_token]
    hidden = None
    
    for _ in range(max_length - 1):
        # 지금 들임을 얻는다
        x = torch.tensor([[generated[-1]]])
        
        # 다음 토큰의 로짓을 셈한다
        embedded = model.embedding(x)
        output, hidden = model.encoder(embedded, hidden)
        logits = model.output_proj(output[:, -1, :])
        
        # 온도를 적용한다
        logits = logits / temperature
        
        # 분포에서 뽑는다
        probs = F.softmax(logits, dim=-1)
        next_token = torch.multinomial(probs, num_samples=1).item()
        
        generated.append(next_token)
    
    return generated
```

### 온도 눈금 조절

온도 $\tau$은 뽑기 분포의 뾰족함을 다스린다.

$$P_\tau(x_i | x_{<i}) \propto P(x_i | x_{<i})^{1/\tau}$$

- $\tau < 1$: 더 뾰족한 분포이며 더 정해져 있다(확률 높은 토큰을 좋아한다)
- $\tau = 1$: 본디 분포
- $\tau > 1$: 더 평평한 분포이며 더 마구잡이다(확률 낮은 토큰도 살핀다)

### 웃 k 뽑기와 웃 p 뽑기

다양함을 지키면서 표본의 품질을 높이려면:

**웃 k 뽑기:** 확률이 가장 높은 토큰 $k$개만 살핀다.

```python
def top_k_sampling(logits, k):
    """가장 그럴듯한 웃 k개 토큰에서 뽑는다."""
    values, indices = torch.topk(logits, k)
    probs = F.softmax(values, dim=-1)
    sampled_idx = torch.multinomial(probs, 1)
    return indices.gather(-1, sampled_idx)
```

**웃 p(핵) 뽑기:** 쌓인 확률이 $p$을 넘을 때까지 토큰을 넣는다.

```python
def top_p_sampling(logits, p):
    """쌓인 확률이 p 이상이 되는 가장 작은 토큰 모임에서 뽑는다."""
    sorted_logits, sorted_indices = torch.sort(logits, descending=True)
    cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
    
    # 자를 자리를 찾는다
    sorted_indices_to_remove = cumulative_probs > p
    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
    sorted_indices_to_remove[..., 0] = False
    
    # 거르고 뽑는다
    sorted_logits[sorted_indices_to_remove] = float('-inf')
    probs = F.softmax(sorted_logits, dim=-1)
    sampled_idx = torch.multinomial(probs, 1)
    
    return sorted_indices.gather(-1, sampled_idx)
```

## 계산에 대한 고려

### 차례대로 만들어 내기의 병목

자기 되돌이 모델의 으뜸 한계는 **차례대로 만들어 내기**이다. 길이 $n$인 차례를 만들려면 신경망을 $n$번 앞먹임해야 한다. $x_i$마다 앞선 모든 내놓기에 매이기 때문이다.

| 갈래 | 익히기 | 만들어 내기 |
|--------|----------|------------|
| **나란히 하기** | 온전함(모든 자리를 한꺼번에) | 없음(엄격히 차례대로) |
| **복잡도** | 앞먹임 $O(1)$번 | 앞먹임 $O(n)$번 |
| **실제 빠르기** | 빠르다 | 긴 차례에서는 느리다 |

### 효율 좋은 얼개

얼개의 여러 새로움이 만들어 내기 병목을 다룬다.

1. **저장턱**: 중간 셈을 담아 둔다(예컨대 변환기의 열쇠-값 저장턱)
2. **나란한 미리 헤아리기**: 토큰 여럿을 한꺼번에 헤아린다(넘겨짚는 풀기)
3. **자기 되돌이가 아닌 모델**: 가능도의 다룰 수 있음을 나란한 만들어 내기와 맞바꾼다

## 다른 만들어 내는 모델과 견주기

| 성질 | 자기 되돌이 | 변분 자기 부호기 | 맞겨루기 만들개 | 흐름 | 퍼짐 |
|----------|---------------|-----|-----|------|-----------|
| **정확한 가능도** | ✓ | ✗ (증거 하한) | ✗ | ✓ | ✗ |
| **빠른 익히기** | ✓ | ✓ | ✗ | ✓ | ✗ |
| **빠른 만들어 내기** | ✗ | ✓ | ✓ | ✓ | ✗ |
| **표본 품질** | 높다 | 보통 | 높다 | 보통 | 아주 높다 |
| **봉우리 덮기** | 높다 | 높다 | 낮다 | 높다 | 높다 |

자기 되돌이 모델은 다음일 때 뛰어나다.

- 정확한 가능도가 중요할 때(밀도 어림, 누르기)
- 자료에 자연스러운 차례 짜임이 있을 때(글, 소리, 시계열)
- 만들어 내는 빠르기보다 익히기 효율이 앞설 때

## 계량 금융에서의 쓰임

### 시계열 예측

돈살림 시계열은 자기 되돌이 틀에 자연스럽게 들어맞는다.

$$P(r_t | r_{<t}) = P(r_t | r_{t-1}, r_{t-2}, \ldots)$$

여기서 $r_t$은 수익률, 값, 또는 다른 돈살림 양을 나타낸다.

```python
class FinancialARModel(nn.Module):
    """
    돈살림 시계열용 자기 되돌이 모델.
    
    돈살림 수익률에 흔한 두꺼운 꼬리를 담으려
    스튜던트 t 분포의 매개변수를 내놓는다.
    """
    
    def __init__(self, input_dim: int, hidden_dim: int):
        super().__init__()
        
        self.lstm = nn.LSTM(input_dim, hidden_dim, batch_first=True)
        
        # 스튜던트 t 매개변수: 자리, 잣수, 자유도
        self.loc_proj = nn.Linear(hidden_dim, 1)
        self.scale_proj = nn.Linear(hidden_dim, 1)
        self.df_proj = nn.Linear(hidden_dim, 1)
    
    def forward(self, x):
        hidden, _ = self.lstm(x)
        
        loc = self.loc_proj(hidden)
        scale = F.softplus(self.scale_proj(hidden)) + 1e-6
        df = F.softplus(self.df_proj(hidden)) + 2  # 흩어짐이 유한하려면 df > 2
        
        return loc, scale, df
```

### 시나리오 만들어 내기

자기 되돌이 모델은 여러 자산이나 인자의 배운 결합 분포에서 뽑아 그럴듯한 저자 시나리오를 만들 수 있다.

### 주문 흐름 나타내기

잦은 거래 쓰임새는 주문의 도착과 성질을 자기 되돌이로 나타내어 저자 미시 짜임의 차례 성질을 담는다.

## 요약

자기 되돌이 모델은 확률의 사슬 규칙에 바탕한 만들어 내는 모델의 원칙 있는 틀을 준다. 핵심 성질은 다음과 같다.

1. **정확한 가능도**: 곧바로 최대 가능도 익히기와 뜻있는 밀도 따지기를 할 수 있다
2. **너그러움**: 어떤 조건부 분포도 신경망으로 매개변수화할 수 있다
3. **차례 자료에 잘 맞음**: 글, 소리, 시계열에는 본디 차례가 있다
4. **차례대로 만들어 내기**: 셈의 으뜸 한계

다음 마디에서는 그림을 위한 PixelCNN, 소리를 위한 WaveNet, 일반 차례를 위한 변환기 같은 구체적인 자기 되돌이 얼개를 살핀다.

## 참고 문헌

1. Bengio, Y., & Bengio, S. (2000). Modeling High-Dimensional Discrete Data with Multi-Layer Neural Networks. *NeurIPS*.
2. Larochelle, H., & Murray, I. (2011). The Neural Autoregressive Distribution Estimator. *AISTATS*.
3. Uria, B., Côté, M. A., Gregor, K., Murray, I., & Larochelle, H. (2016). Neural Autoregressive Distribution Estimation. *JMLR*.
4. Papamakarios, G., Nalisnick, E., Rezende, D. J., Mohamed, S., & Lakshminarayanan, B. (2021). Normalizing Flows for Probabilistic Modeling and Inference. *JMLR*.

---

## 연습문제

1. **사슬 규칙 확인하기**: 단순한 변수 셋 분포에서 차례를 달리해도 같은 결합 확률이 나옴을 확인하라.

2. **온도 살펴보기**: 온도 뽑기를 짜고 만들어 낸 차례가 온도에 따라 어떻게 바뀌는지 그려 보라.

3. **드러남 치우침**: 드러남 치우침이 만들어 내기 품질에 미치는 영향을 재는 실험을 짜라.

4. **돈살림 쓰임새**: 날마다의 주식 수익률에 대한 자기 되돌이 모델을 세우고 그 밀도 어림 솜씨를 따져 보라.
