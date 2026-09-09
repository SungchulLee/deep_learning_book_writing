# 글 만들어 내기의 표집 전략

자기되돌리기 말 모델은 앞선 토막이 주어질 때 다음 토막에 대한 확률 분포를 정한다. 미룸 때에는 이 분포에서 **풀어내어** 글을 만들어야 한다. 어떤 풀기 전략을 고르느냐가 내놓음의 좋음, 여러 갈래임, 조리, 셈 값에 깊이 영향을 준다.

$p_\theta(x_t \mid x_{<t})$을 앞뒤 흐름 $x_{<t} = (x_1, \ldots, x_{t-1})$이 주어졌을 때 토막 $x_t$에 대한 모델의 조건부 분포라 하자. 모델은 **로짓** $z \in \mathbb{R}^{|V|}$(낱말 사전 토막마다 하나)을 내놓고, 이를 소프트맥스로 확률로 바꾼다.

$$
p_\theta(x_t = v \mid x_{<t}) = \frac{\exp(z_v)}{\sum_{v' \in V} \exp(z_{v'})}
$$

풀기 전략은 크게 두 갈래로 나뉜다:

1. **정해진 대로의 방법**: 가장 좋게 하기로 토막을 고른다(욕심쟁이, 빔 찾기)
2. **확률에 맡기는 방법**: (고친) 분포에서 뽑는다(온도, 상위 k, 상위 p)

---

## 1. 욕심쟁이 풀기

가장 단순한 전략. 곧 늘 가장 그럴듯한 토막을 고른다.

$$
x_t^* = \arg\max_{v \in V} \, p_\theta(v \mid x_{<t})
$$

```python
import torch
import torch.nn.functional as F
from typing import Optional

def greedy_decode(
    model,
    input_ids: torch.Tensor,
    max_length: int = 50,
    eos_token_id: Optional[int] = None
) -> torch.Tensor:
    """욕심쟁이 풀기: 늘 확률이 가장 높은 토막을 고른다."""
    generated = input_ids.clone()
    
    for _ in range(max_length):
        with torch.no_grad():
            outputs = model(generated)
            logits = outputs.logits[:, -1, :]  # [batch, vocab]
        
        next_token = logits.argmax(dim=-1, keepdim=True)  # [batch, 1]
        generated = torch.cat([generated, next_token], dim=-1)
        
        if eos_token_id is not None and (next_token == eos_token_id).all():
            break
    
    return generated
```

**성질**:

- 빠르다: 이음마다 $O(T \cdot |V|)$
- 정해진 대로: 같은 들임은 늘 같은 내놓음을 낸다
- 봉우리 좇기: 전체 최적이 아니라 가까운 최댓값을 찾는다
- 되풀이와 밋밋한 내놓음에 빠지기 쉽다

**쓸 때**: 사실 물음 답하기, 갈래 매기기, 같은 결과가 나와야 할 때.

---

## 2. 온도 표집

온도 맞추기는 뽑기 앞에서 확률 분포의 "뾰족함"을 바꾼다. 로짓 $z$과 온도 $T > 0$이 주어질 때:

$$
p_T(x_t = v \mid x_{<t}) = \frac{\exp(z_v / T)}{\sum_{v'} \exp(z_{v'} / T)}
$$

### 수학으로 살피기

온도는 분포의 **엔트로피**에 영향을 준다. $H(p) = -\sum_v p_v \log p_v$을 엔트로피라 하자.

**정리(온도와 엔트로피).** 로짓이 $z$인 아무 분포 $p$에 대해:

1. $\lim_{T \to 0^+} p_T$은 $\arg\max_v z_v$ 한 점에 모여든다
2. $\lim_{T \to \infty} p_T$은 고른 분포로 모여든다
3. $H(p_T)$은 $T$에 대해 단조로 늘어난다

*밝히기 얼개*: $T \to 0$이면 $z_v$이 가장 큰지에 따라 $z_v/T \to \pm\infty$이 되어 무게가 한곳에 몰린다. $T \to \infty$이면 모든 $v$에 대해 $z_v/T \to 0$이 되어 고른 분포가 된다. 한 방향으로만 바뀜은 로그-합-지수의 결에서 따라 나온다. $\square$

### 엔트로피 관점

달리 보면, 온도 맞추기는 확률을 거듭제곱하는 것과 같다:

$$
p_T(v) \propto p_1(v)^{1/T}
$$

$\exp(z_v/T) = \exp(z_v)^{1/T}$이므로 이는 따라 나온다.

```python
def temperature_sample(
    logits: torch.Tensor,
    temperature: float = 1.0,
    num_samples: int = 1
) -> torch.Tensor:
    """
    온도를 맞춘 분포에서 뽑기.
    
    인수:
        logits: 고르게 맞추지 않은 로그 확률 [batch, vocab]
        temperature: 잣수 인자(0 < T). 낮을수록 더 정해진 대로.
        num_samples: 뽑을 표본의 개수
        
    반환값:
        뽑은 토막 번호 [batch, num_samples]
    """
    if temperature <= 0:
        raise ValueError("Temperature must be positive")
    
    if temperature == 1.0:
        scaled_logits = logits
    else:
        scaled_logits = logits / temperature
    
    probs = F.softmax(scaled_logits, dim=-1)
    return torch.multinomial(probs, num_samples=num_samples)
```

### 실전 지침

| 온도 | 효과 | 쓰임새 |
|-------------|--------|----------|
| 0.0~0.3 | 거의 정해진 대로 | 코드, 사실을 말하는 답 |
| 0.5~0.7 | 균형 | 일반 도우미 일 |
| 0.7~1.0 | 창의적 | 이야기 쓰기, 생각 모으기 |
| 1.0~1.5 | 흩어짐이 큼 | 여러 갈래 만들어 내기, 살펴보기 |
| > 1.5 | 흔히 앞뒤가 안 맞음 | 거의 쓸모없음 |

---

## 3. 상위 k 표집

**상위 k 표집**(Fan 외, 2018)은 가장 그럴듯한 토막 $k$개로 뽑기를 제한하고 그들 사이에 확률 무게를 다시 나눈다.

$$
p_{\text{top-}k}(v \mid x_{<t}) = 
\begin{cases}
\frac{p_\theta(v \mid x_{<t})}{\sum_{v' \in V_k} p_\theta(v' \mid x_{<t})} & \text{if } v \in V_k \\
0 & \text{otherwise}
\end{cases}
$$

여기서 $V_k$은 확률이 가장 높은 토막 $k$개의 모음이다.

```python
def top_k_sample(
    logits: torch.Tensor,
    k: int = 50,
    temperature: float = 1.0,
    filter_value: float = float('-inf')
) -> torch.Tensor:
    """
    상위 k 표집: 가장 그럴듯한 토막 k개에서 뽑는다.
    
    인수:
        logits: 고르게 맞추지 않은 로그 확률 [batch, vocab]
        k: 남길 상위 토막의 개수
        temperature: 온도 맞추기(거르기 앞에 쓴다)
        filter_value: 걸러진 토막에 매길 값
        
    반환값:
        뽑은 토막 번호 [batch, 1]
    """
    logits = logits / temperature
    
    # k번째로 큰 값(문턱값) 찾기
    top_k_values, _ = torch.topk(logits, k, dim=-1)
    threshold = top_k_values[:, -1, None]  # [batch, 1]
    
    # 문턱값 아래 토막을 0으로 만들기
    logits = torch.where(logits >= threshold, logits, filter_value)
    
    probs = F.softmax(logits, dim=-1)
    return torch.multinomial(probs, num_samples=1)
```

### 한계: 붙박이 k

상위 k의 근본 약점은 분포의 꼴과 상관없이 $k$이 붙박이라는 것이다:

- **뾰족한 분포**(모델이 자신 있음): 상위 k가 확률 낮은 잡음 토막을 넣을 수 있다
- **평평한 분포**(모델이 흔들림): 상위 k가 그럴듯한 대안을 빼 버릴 수 있다

**보기**: $p = (0.9, 0.05, 0.02, 0.01, 0.01, 0.01, ...)$이고 $k=10$이면 있으나 마나 한 확률의 토막까지 넣게 된다. 토막 20개에 고르게 $p = (0.1, 0.1, 0.1, ..., 0.1)$이고 $k=10$이면 그럴듯한 선택지의 반을 까닭 없이 빼게 된다.

---

## 4. 알갱이(상위 p) 표집

**알갱이 표집**(Holtzman 외, 2020)은 쌓인 확률 무게에 따라 후보 모음을 그때그때 맞추어 상위 k의 한계를 다룬다.

### 정의

문턱 $p \in (0, 1]$이 주어질 때 **알맹이** $V_p$은 쌓은 확률이 $p$을 넘는 가장 작은 토막 묶음이다.

$$
V_p = \arg\min_{V' \subseteq V} |V'| \quad \text{s.t.} \quad \sum_{v \in V'} p_\theta(v \mid x_{<t}) \geq p
$$

여기서 토막은 확률이 줄어드는 차례로 더한다.

뽑는 분포는 다음과 같다:

$$
p_{\text{nucleus}}(v) = 
\begin{cases}
\frac{p_\theta(v)}{\sum_{v' \in V_p} p_\theta(v')} & \text{if } v \in V_p \\
0 & \text{otherwise}
\end{cases}
$$

### 왜 "알갱이"인가?

이 이름은 확률이 높은 토막이 분포의 "고갱이"(알갱이)를 이루는 반면, 긴 꼬리에는 모델이 낱말 곳간 전체에 무게를 나눠야 하기 때문에 어쩔 수 없이 확률을 준 믿기 어려운 토막이 들어 있다는 살핌에서 왔다.

```python
def nucleus_sample(
    logits: torch.Tensor,
    p: float = 0.9,
    temperature: float = 1.0,
    min_tokens_to_keep: int = 1
) -> torch.Tensor:
    """
    알갱이(상위 p) 표집: 쌓인 확률이 p 이상인 가장 작은 모음에서 뽑는다.
    
    인수:
        logits: 고르게 맞추지 않은 로그 확률 [batch, vocab]
        p: 쌓인 확률 문턱값
        temperature: 온도 맞추기(거르기 앞에 쓴다)
        min_tokens_to_keep: 늘 이만큼의 토막은 남긴다
        
    반환값:
        뽑은 토막 번호 [batch, 1]
    """
    logits = logits / temperature
    
    # 확률로 정렬(내림차순)
    sorted_logits, sorted_indices = torch.sort(logits, descending=True, dim=-1)
    sorted_probs = F.softmax(sorted_logits, dim=-1)
    cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
    
    # 자름점 찾기: 쌓인 확률이 p을 넘는 첫 자리
    # 문턱값을 넘는 토막을 남기려 오른쪽으로 밀기
    sorted_indices_to_remove = cumulative_probs > p
    sorted_indices_to_remove[..., min_tokens_to_keep:] = sorted_indices_to_remove[..., :-min_tokens_to_keep].clone()
    sorted_indices_to_remove[..., :min_tokens_to_keep] = False
    
    # 본디 차례로 되돌려 흩뿌리기
    indices_to_remove = sorted_indices_to_remove.scatter(
        dim=-1, index=sorted_indices, src=sorted_indices_to_remove
    )
    logits = logits.masked_fill(indices_to_remove, float('-inf'))
    
    probs = F.softmax(logits, dim=-1)
    return torch.multinomial(probs, num_samples=1)
```

### 알맞게 맞추는 몸짓

핵심 이점은 알갱이의 크기가 분포의 엔트로피에 맞춰진다는 것이다.

| 분포 갈래 | 엔트로피 | 알갱이 크기 |
|------------------|---------|--------------|
| 모델이 자신 있음 | 낮음 | 작음(토막 몇 개) |
| 모델이 흔들림 | 높음 | 큼(토막 여럿) |

이는 사람의 직관과 맞아떨어진다. 곧 이어질 만한 것이 여럿이면 다 넣고, 하나가 뚜렷이 가장 좋으면 거기에 초점을 둔다.

---

## 5. 아우른 표집 전략

실전에서는 더 세밀히 다스리려 여러 전략을 아우르는 일이 많다.

### 상위 k + 상위 p

두 거르개를 모두 쓴다(후보 모음의 교집합):

```python
def combined_sample(
    logits: torch.Tensor,
    k: int = 50,
    p: float = 0.9,
    temperature: float = 1.0
) -> torch.Tensor:
    """
    상위 k와 상위 p를 아우른 표집.
    
    연산의 차례:
    1. 온도 맞추기
    2. 상위 k 거르기(후보에 딱 잘린 한계)
    3. 상위 p 거르기(확률 바탕 다듬기)
    4. 남은 분포에서 뽑는다
    """
    logits = logits / temperature
    
    # 1단계: 상위 k 거르기
    if k > 0 and k < logits.size(-1):
        top_k_values, _ = torch.topk(logits, k, dim=-1)
        threshold_k = top_k_values[:, -1, None]
        logits = torch.where(logits >= threshold_k, logits, float('-inf'))
    
    # 2단계: 상위 p 거르기
    if p < 1.0:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True, dim=-1)
        sorted_probs = F.softmax(sorted_logits, dim=-1)
        cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
        
        sorted_indices_to_remove = cumulative_probs > p
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = False
        
        indices_to_remove = sorted_indices_to_remove.scatter(
            dim=-1, index=sorted_indices, src=sorted_indices_to_remove
        )
        logits = logits.masked_fill(indices_to_remove, float('-inf'))
    
    probs = F.softmax(logits, dim=-1)
    return torch.multinomial(probs, num_samples=1)
```

### 연산의 차례

표준 차례는 **온도 → 상위 k → 상위 p → 뽑기**이다

왜 이 차례인가?

1. 온도가 먼저: 상위 k와 상위 p가 걸러 낼 상대 확률을 바꾼다
2. 상위 k가 다음: 딱 잘린 한계를 준다(병리적인 분포에 대한 안전망)
3. 상위 p가 셋째: 확률 무게로 상위 k 안에서 다듬는다
4. 뽑기가 마지막: 마지막으로 걸러진 분포에서 뽑는다

---

## 6. 최소 p 표집

**최소 p 표집**(top-p의 요즘 대안)은 확률이 적어도 $p_{\min} \times p_{\max}$인 토막을 모두 남긴다. 여기서 $p_{\max}$은 가장 높은 토막 확률이다.

### 정의

$$
V_{\text{min-}p} = \{v \in V : p_\theta(v) \geq p_{\min} \cdot \max_{v'} p_\theta(v')\}
$$

### 직관

최소 p는 가장 그럴듯한 토막을 기준으로 문턱값의 잣수를 맞춘다. 곧:

- 모델이 자신할 때($p_{\max}$이 높을 때) 문턱이 높아 토막이 적다
- 아리송할 때($p_{\max}$이 낮을 때) 문턱이 낮아 토막이 많다

```python
def min_p_sample(
    logits: torch.Tensor,
    p_min: float = 0.1,
    temperature: float = 1.0
) -> torch.Tensor:
    """
    최소 p 표집: 확률이 p_min * max_prob 이상인 토막을 남긴다.
    
    인수:
        logits: 고르게 맞추지 않은 로그 확률 [batch, vocab]
        p_min: 최댓값 대비 최소 확률(0 < p_min <= 1)
        temperature: 온도 맞추기
        
    반환값:
        뽑은 토막 번호 [batch, 1]
    """
    logits = logits / temperature
    probs = F.softmax(logits, dim=-1)
    
    # 최대 확률에 바탕한 그때그때의 문턱값
    max_prob = probs.max(dim=-1, keepdim=True).values
    threshold = p_min * max_prob
    
    # 문턱값 아래 토막 가리기
    mask = probs < threshold
    logits = logits.masked_fill(mask, float('-inf'))
    
    probs = F.softmax(logits, dim=-1)
    return torch.multinomial(probs, num_samples=1)
```

### 견줌: 상위 p와 최소 p

| 갈래 | 상위 p(알갱이) | 최소 p |
|--------|-----------------|-------|
| 문턱값 갈래 | 절대적인 쌓임 | 최댓값 기준 상대 |
| 아주 뾰족한 분포 | 토막 1개만 넣을 수 있음 | 늘 1개 이상 넣음 |
| 평평한 분포 | 여럿 넣음 | 여럿 넣음 |
| 매개변수의 뜻 | "이만큼의 확률을 덮어라" | "가장 좋은 것의 이 배수 안에 들어라" |

---

## 7. 전형 표집

**전형 표집**(Meister 외, 2023)은 앎 이론으로 보아 토막이 얼마나 "전형적"인지, 곧 얼마나 "기대할 만한지"에 따라 고른다.

### 앎 이론의 바탕

토막 $v$의 **앎의 양**(놀라움)은 다음과 같다:

$$
I(v) = -\log p_\theta(v \mid x_{<t})
$$

분포의 **엔트로피**는 앎의 기댓값이다:

$$
H = \mathbb{E}_{v \sim p}[I(v)] = -\sum_v p_\theta(v) \log p_\theta(v)
$$

토막의 앎의 양이 엔트로피에 가까우면 그 토막은 **전형적**이다. 직관으로 보면 전형적인 토막은 "너무 놀랍지도 너무 뻔하지도 않다".

### 정의

**흔한 묶음** $A_\epsilon$은 소식 양이 엔트로피에서 $\epsilon$ 안쪽인 토막을 담는다.

$$
A_\epsilon = \{v \in V : |I(v) - H| < \epsilon\}
$$

실전에서는 $|I(v) - H|$으로 정렬하고 쌓인 확률이 문턱값을 넘을 때까지 토막을 가져온다.

```python
def typical_sample(
    logits: torch.Tensor,
    p: float = 0.9,
    temperature: float = 1.0
) -> torch.Tensor:
    """
    전형 표집: 앎의 양이 엔트로피에 가까운 토막을 고른다.
    
    인수:
        logits: 고르게 맞추지 않은 로그 확률 [batch, vocab]
        p: 전형 모음의 쌓인 확률 문턱값
        temperature: 온도 맞추기
        
    반환값:
        뽑은 토막 번호 [batch, 1]
    """
    logits = logits / temperature
    probs = F.softmax(logits, dim=-1)
    log_probs = F.log_softmax(logits, dim=-1)
    
    # 엔트로피와 앎의 양 셈하기
    entropy = -(probs * log_probs).sum(dim=-1, keepdim=True)  # [batch, 1]
    information = -log_probs  # [batch, vocab]
    
    # 엔트로피와의 거리(토막마다 얼마나 "전형에서 벗어났는지")
    deviation = torch.abs(information - entropy)
    
    # 전형다움으로 정렬(벗어남이 작은 것 먼저)
    sorted_deviation, sorted_indices = torch.sort(deviation, dim=-1)
    sorted_probs = probs.gather(dim=-1, index=sorted_indices)
    cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
    
    # 쌓인 확률이 p을 넘는 자름점 찾기
    sorted_indices_to_remove = cumulative_probs > p
    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
    sorted_indices_to_remove[..., 0] = False
    
    # 되돌려 흩뿌리고 가리기
    indices_to_remove = sorted_indices_to_remove.scatter(
        dim=-1, index=sorted_indices, src=sorted_indices_to_remove
    )
    logits = logits.masked_fill(indices_to_remove, float('-inf'))
    
    probs = F.softmax(logits, dim=-1)
    return torch.multinomial(probs, num_samples=1)
```

### 왜 전형 표집인가?

보통의 표집 방법은 확률 높은 토막 쪽으로 치우친다. 그러나 앎 이론의 점근 등분할 성질(AEP)은 긴 차례에서는 확률 무게가 거의 모두 **전형 차례**, 곧 토막마다의 앎이 엔트로피에 가까운 차례에 몰린다고 일러 준다.

전형 표집은 이 눈썰미를 토막 수준에서 써먹는다.

---

## 8. 에타(η) 표집

**에타 표집**은 전형 표집의 알맞게 맞춤에, 분포의 엔트로피가 아주 클 때 상위 p로 물러나는 길을 곁들인다.

### 정의

1. 분포의 엔트로피 $H$을 셈한다
2. 문턱을 잡는다: $\eta = \min(\epsilon, \sqrt{\epsilon} \cdot e^{-H})$
3. $p(v) > \eta$인 토막을 남긴다
4. 남는 토막이 없으면 상위 p로 물러난다

고갱이 깨침은 엔트로피가 커질수록 $\eta$이 작아진다는 것이다(더 아리송할수록 문턱이 낮아진다).

```python
def eta_sample(
    logits: torch.Tensor,
    epsilon: float = 0.0003,
    fallback_p: float = 0.9,
    temperature: float = 1.0
) -> torch.Tensor:
    """
    에타 표집: 엔트로피에 바탕한 알맞게 맞추는 문턱값.
    
    인수:
        logits: 고르게 맞추지 않은 로그 확률 [batch, vocab]
        epsilon: 바탕 문턱값 매개변수
        fallback_p: 에타 거르기가 토막을 모두 없앴을 때 쓸 상위 p 문턱값
        temperature: 온도 맞추기
        
    반환값:
        뽑은 토막 번호 [batch, 1]
    """
    logits = logits / temperature
    probs = F.softmax(logits, dim=-1)
    log_probs = F.log_softmax(logits, dim=-1)
    
    # 엔트로피 셈하기
    entropy = -(probs * log_probs).sum(dim=-1, keepdim=True)
    
    # 알맞게 맞추는 문턱값 에타 셈하기
    eta = torch.minimum(
        torch.tensor(epsilon),
        torch.sqrt(torch.tensor(epsilon)) * torch.exp(-entropy)
    )
    
    # 문턱값 아래 토막 거르기
    mask = probs < eta
    filtered_logits = logits.masked_fill(mask, float('-inf'))
    
    # 남은 토막이 있는지 살피기
    valid_count = (~mask).sum(dim=-1)
    
    # 모두 걸러지면 상위 p로 물러나기
    if (valid_count == 0).any():
        # 물러날 곳으로 알갱이 표집 쓰기
        return nucleus_sample(logits * temperature, p=fallback_p, temperature=temperature)
    
    probs = F.softmax(filtered_logits, dim=-1)
    return torch.multinomial(probs, num_samples=1)
```

---

## 9. 미로스탯 표집

**미로스탯**(Basu 외, 2021)은 목표 **헷갈림도**(같은 말로 목표 엇갈린 엔트로피 또는 놀라움 비율)를 지키도록 온도를 그때그때 맞춘다.

### 동기

붙박이 표집 매개변수는 맥락에 따라 내놓음의 좋음이 들쭉날쭉해지기 쉽다. 미로스탯은 풀기를 **다스림 문제**로 본다. 곧 바라는 "놀라움 수준"을 좇도록 매개변수를 돌아가는 도중에 맞춘다.

### 알고리즘(미로스탯-2)

1. 과녁 놀람도 $\tau$을 잡는다(보기: 5.0 ≈ 헷갈림도 148)
2. 놀라움의 흐르는 어림값을 좇는다
3. 목표 쪽으로 이끌도록 상위 k를 그때그때 맞춘다

```python
def mirostat_v2_sample(
    logits: torch.Tensor,
    tau: float = 5.0,       # 목표 놀라움
    eta: float = 0.1,       # 배움 비율
    mu: float = None        # 지금의 놀라움 어림값(상태)
) -> tuple[torch.Tensor, float]:
    """
    미로스탯-2 표집: 헷갈림도를 알맞게 좇기.
    
    인수:
        logits: 고르게 맞추지 않은 로그 확률 [batch, vocab]
        tau: 목표 놀라움(목표 헷갈림도의 밑 2 로그)
        eta: 놀라움 좇기의 배움 비율
        mu: 지금의 놀라움 어림값(2*tau로 첫자리매김)
        
    반환값:
        (뽑은 토막 번호, 고친 mu) 튜플
    """
    if mu is None:
        mu = 2 * tau  # 첫자리매김
    
    probs = F.softmax(logits, dim=-1)
    log_probs = torch.log2(probs + 1e-10)  # 비트를 얻으려 밑이 2인 로그 쓰기
    
    # 확률 내림차순으로 정렬
    sorted_probs, sorted_indices = torch.sort(probs, descending=True, dim=-1)
    sorted_log_probs = log_probs.gather(dim=-1, index=sorted_indices)
    sorted_surprisals = -sorted_log_probs
    
    # 쌓인 값이 목표에 이르는 k 찾기
    # mu를 그때그때의 문턱값으로 쓰기
    cumsum_probs = torch.cumsum(sorted_probs, dim=-1)
    
    # 미로스탯-2: mu에서 이끌어 낸 확률 문턱값에서 잘라 내기
    # k = 남길 토막의 개수
    prob_threshold = torch.exp2(torch.tensor(-mu))
    
    # 확률이 문턱값 이상인 토막 남기기
    mask = sorted_probs >= prob_threshold
    if mask.sum() == 0:
        mask[..., 0] = True  # 적어도 상위 토막 하나는 남기기
    
    # 걸러진 것에서 뽑기
    filtered_probs = sorted_probs * mask.float()
    filtered_probs = filtered_probs / filtered_probs.sum(dim=-1, keepdim=True)
    
    sample_idx = torch.multinomial(filtered_probs, num_samples=1)
    token_idx = sorted_indices.gather(dim=-1, index=sample_idx)
    
    # 본 놀라움에 따라 mu 고치기
    observed_surprisal = sorted_surprisals.gather(dim=-1, index=sample_idx).item()
    mu = mu - eta * (observed_surprisal - tau)
    
    return token_idx, mu
```

### 쓰는 무늬

```python
def generate_with_mirostat(model, input_ids, max_length, tau=5.0, eta=0.1):
    """한결같은 헷갈림도를 위해 미로스탯-2로 만들어 내기."""
    mu = 2 * tau  # 처음의 놀라움 어림값
    generated = input_ids.clone()
    
    for _ in range(max_length):
        with torch.no_grad():
            logits = model(generated).logits[:, -1, :]
        
        token, mu = mirostat_v2_sample(logits, tau=tau, eta=eta, mu=mu)
        generated = torch.cat([generated, token], dim=-1)
    
    return generated
```

---

## 10. 되풀이 벌주기

### 단순 되풀이 벌주기

앞서 만든 토막의 로짓을 낮춰 되풀이를 말린다:

$$
z'_v = \begin{cases}
z_v / \theta & \text{if } z_v > 0 \text{ and } v \in \text{generated} \\
z_v \cdot \theta & \text{if } z_v < 0 \text{ and } v \in \text{generated} \\
z_v & \text{otherwise}
\end{cases}
$$

여기서 $\theta > 1$은 벌 값이다.

```python
def apply_repetition_penalty(
    logits: torch.Tensor,
    generated_ids: torch.Tensor,
    penalty: float = 1.2
) -> torch.Tensor:
    """
    앞서 만든 토막에 되풀이 벌주기 쓰기.
    
    양수 로짓은 벌주기로 나눈다(확률이 줄어든다).
    음수 로짓은 벌주기를 곱한다(확률도 줄어든다).
    
    인수:
        logits: 지금 걸음의 로짓 [batch, vocab]
        generated_ids: 앞서 만든 토막 번호 [batch, seq_len]
        penalty: 벌주기 인자(> 1이면 말리고, < 1이면 북돋운다)
        
    반환값:
        고친 로짓
    """
    # 만든 토막의 로짓 모으기
    generated_logits = torch.gather(logits, dim=-1, index=generated_ids)
    
    # 부호에 따라 벌주기
    penalized = torch.where(
        generated_logits > 0,
        generated_logits / penalty,
        generated_logits * penalty
    )
    
    # 되돌려 흩뿌리기
    logits = logits.scatter(dim=-1, index=generated_ids, src=penalized)
    return logits
```

### 잦기 벌주기와 있음 벌주기

(곱하기가 아니라) 더하기로 주는 OpenAI 방식의 벌주기:

$$
z'_v = z_v - \alpha_{\text{freq}} \cdot \text{count}(v) - \alpha_{\text{pres}} \cdot \mathbf{1}[v \in \text{generated}]
$$

- **잦기 벌**($\alpha_{\text{freq}}$): 토막이 나온 잦기에 비례해 벌을 준다
- **나옴 벌**($\alpha_{\text{pres}}$): 토막이 한 번이라도 나왔으면 똑같이 벌을 준다

```python
def apply_frequency_presence_penalty(
    logits: torch.Tensor,
    generated_ids: torch.Tensor,
    frequency_penalty: float = 0.5,
    presence_penalty: float = 0.5
) -> torch.Tensor:
    """
    잦기 벌주기와 있음 벌주기 쓰기(더하기, OpenAI 방식).
    
    인수:
        logits: 지금 걸음의 로짓 [batch, vocab]
        generated_ids: 앞서 만든 토막 번호 [batch, seq_len]
        frequency_penalty: 나올 때마다의 벌
        presence_penalty: 하나라도 있으면 주는 일정한 벌
        
    반환값:
        고친 로짓
    """
    batch_size, vocab_size = logits.shape
    
    # 토막마다 나온 횟수 세기
    token_counts = torch.zeros(batch_size, vocab_size, device=logits.device)
    token_counts.scatter_add_(
        dim=-1, 
        index=generated_ids,
        src=torch.ones_like(generated_ids, dtype=logits.dtype)
    )
    
    # 잦기 벌주기: 셈에 비례
    logits = logits - frequency_penalty * token_counts
    
    # 있음 벌주기: 셈이 0보다 크면 일정하게
    presence_mask = (token_counts > 0).float()
    logits = logits - presence_penalty * presence_mask
    
    return logits
```

### n-그램 되풀이 막기

n-그램 차례가 똑같이 되풀이되는 것을 막는다:

```python
def apply_no_repeat_ngram(
    logits: torch.Tensor,
    generated_ids: list[int],
    n: int = 3
) -> torch.Tensor:
    """
    이미 나온 n-그램이 다시 나오지 못하게 막기.
    
    인수:
        logits: 지금 걸음의 로짓 [vocab]
        generated_ids: 앞서 만든 토막 번호(목록)
        n: 막을 n-그램의 크기
        
    반환값:
        막은 이음을 -inf로 둔 고친 로짓
    """
    if len(generated_ids) < n - 1:
        return logits
    
    # 마지막 (n-1)개 토막 얻기
    prefix = tuple(generated_ids[-(n-1):])
    
    # 발자취의 모든 n-그램을 찾아 막을 이음 모으기
    banned_tokens = set()
    for i in range(len(generated_ids) - n + 1):
        if tuple(generated_ids[i:i+n-1]) == prefix:
            banned_tokens.add(generated_ids[i + n - 1])
    
    # 막은 토막의 로짓을 -inf로 두기
    for token_id in banned_tokens:
        logits[token_id] = float('-inf')
    
    return logits
```

---

## 11. 빔 찾기

**빔 찾기**는 모델 아래에서 거의 가장 좋은 차례를 찾으려 가설(빔)을 여럿 지닌다.

### 형식적 정의

걸음마다 빔마다 상위 $k$개의 이음으로 뻗고, 전체에서 상위 $B$개 후보를 남긴다:

$$
\text{score}(x_{1:t}) = \sum_{i=1}^{t} \log p_\theta(x_i \mid x_{<i})
$$

### 길이 고르게 맞추기

날 로그 확률은 짧은 차례를 더 좋게 본다. **길이 벌주기**가 이를 바로잡는다:

$$
\text{score}_{\text{LP}}(x_{1:t}) = \frac{\sum_{i=1}^{t} \log p_\theta(x_i \mid x_{<i})}{((5 + t) / 6)^\alpha}
$$

여기서 $\alpha > 0$이면 더 긴 이음을 북돋우고 $\alpha < 0$이면 더 짧은 쪽을 좋아한다.

```python
def beam_search(
    model,
    input_ids: torch.Tensor,
    num_beams: int = 5,
    max_length: int = 50,
    length_penalty: float = 1.0,
    eos_token_id: Optional[int] = None,
    early_stopping: bool = True
) -> list[tuple[float, list[int]]]:
    """
    빔 찾기 풀기.
    
    인수:
        model: .forward()가 로짓을 돌려주는 말 모델
        input_ids: 처음 토막 번호 [1, seq_len]
        num_beams: 지닐 빔의 개수
        max_length: 만들어 낼 최대 토막 수
        length_penalty: 길이 고르게 맞추기의 지수(> 0)
        eos_token_id: 차례 끝 토막의 번호
        early_stopping: 가설이 num_beams개 완성되면 멈춘다
        
    반환값:
        점수 내림차순으로 정렬한 (score, token_ids) 튜플의 목록
    """
    device = input_ids.device
    initial_seq = input_ids[0].tolist()
    
    # 빔마다: (쌓인 로그 확률, 토막 차례)
    beams = [(0.0, initial_seq)]
    complete_hypotheses = []
    
    for step in range(max_length):
        all_candidates = []
        
        for log_prob, seq in beams:
            # 이미 끝났는지 살피기
            if eos_token_id is not None and seq[-1] == eos_token_id:
                complete_hypotheses.append((log_prob, seq))
                continue
            
            # 다음 토막의 분포 얻기
            seq_tensor = torch.tensor([seq], device=device)
            with torch.no_grad():
                outputs = model(seq_tensor)
                logits = outputs.logits[0, -1, :]
                log_probs = F.log_softmax(logits, dim=-1)
            
            # 상위 토막으로 뻗기
            top_log_probs, top_indices = torch.topk(log_probs, num_beams * 2)
            
            for next_log_prob, token_id in zip(top_log_probs.tolist(), top_indices.tolist()):
                new_seq = seq + [token_id]
                new_log_prob = log_prob + next_log_prob
                
                # 매김을 위한 길이 고르게 맞춘 점수
                length_factor = ((5 + len(new_seq)) / 6) ** length_penalty
                normalized_score = new_log_prob / length_factor
                
                all_candidates.append((normalized_score, new_log_prob, new_seq))
        
        if not all_candidates:
            break
        
        # 상위 빔 남기기
        all_candidates.sort(key=lambda x: x[0], reverse=True)
        beams = [(c[1], c[2]) for c in all_candidates[:num_beams]]
        
        # 일찍 멈추기 살피기
        if early_stopping and len(complete_hypotheses) >= num_beams:
            break
    
    # 남은 빔을 가설에 더하기
    complete_hypotheses.extend(beams)
    
    # 길이 고르게 맞춘 점수로 정렬
    complete_hypotheses.sort(
        key=lambda x: x[0] / ((5 + len(x[1])) / 6) ** length_penalty,
        reverse=True
    )
    
    return complete_hypotheses
```

### 여러 갈래 빔 찾기

보통의 빔 찾기는 비슷비슷한 가설을 내놓기 쉽다. **여러 갈래 빔 찾기**(Vijayakumar 외, 2018)는 빔을 묶음으로 나누고 여러 갈래임에 대한 벌주기를 둔다:

$$
\text{score}_g(v) = \text{score}(v) - \lambda \sum_{g' < g} \mathbf{1}[v \in \text{beams}_{g'}]
$$

```python
def diverse_beam_search(
    model,
    input_ids: torch.Tensor,
    num_beams: int = 4,
    num_groups: int = 4,
    diversity_penalty: float = 0.5,
    max_length: int = 50
) -> list[list[tuple[float, list[int]]]]:
    """
    여러 갈래 빔 찾기: 묶음 찾기로 여러 갈래의 가설 만들기.
    
    인수:
        model: 말 모델
        input_ids: 처음 토막 번호
        num_beams: 묶음마다의 빔 수
        num_groups: 빔 묶음의 개수
        diversity_penalty: 앞 묶음이 고른 토막에 주는 벌
        max_length: 만들어 낼 최대 길이
        
    반환값:
        (score, sequence) 튜플을 담은 묶음의 목록
    """
    device = input_ids.device
    all_groups = []
    
    for group_idx in range(num_groups):
        # 자리마다 앞 묶음이 고른 토막 좇기
        previous_group_tokens = []
        for prev_group in all_groups:
            for _, seq in prev_group:
                previous_group_tokens.extend(seq[len(input_ids[0]):])
        
        # 여러 갈래임 벌주기를 곁들인 빔 찾기 돌리기
        group_beams = beam_search_with_penalty(
            model, input_ids, num_beams, max_length,
            penalty_tokens=set(previous_group_tokens),
            penalty_weight=diversity_penalty
        )
        
        all_groups.append(group_beams)
    
    return all_groups
```

---

## 12. 맞대어 찾기

**맞대어 찾기**(Su 외, 2022)는 가능도와, 이미 있는 맥락과 얼마나 다른지 사이의 균형을 잡는다.

### 목표

$$
x_t = \arg\max_{v \in V_k} \left[ (1 - \alpha) \cdot p_\theta(v \mid x_{<t}) - \alpha \cdot \max_{j < t} \text{sim}(h_v, h_{x_j}) \right]
$$

여기서 각 기호는 다음과 같다.

- $V_k$은 확률로 뽑은 상위 k개 후보이다
- $h_v$은 토막 $v$의 숨은 나타냄이다
- $\text{sim}$은 코사인 닮음이다
- $\alpha \in [0, 1]$은 그럴듯함과 남다름 사이를 저울질한다

### 직관

**퇴화 벌** 마디 $\max_j \text{sim}(h_v, h_{x_j})$은 나타냄이 앞선 앞뒤 흐름과 너무 닮은 토막을 눌러 되풀이되는 무늬를 줄인다.

```python
def contrastive_search(
    model,
    input_ids: torch.Tensor,
    k: int = 4,
    alpha: float = 0.6,
    max_length: int = 50
) -> torch.Tensor:
    """
    맞대어 찾기: 확률과 나타냄의 다름 사이 균형 잡기.
    
    인수:
        model: 로짓과 숨은 상태를 돌려주는 모델
        input_ids: 처음 토막 번호 [1, seq_len]
        k: 헤아릴 후보 토막의 개수
        alpha: 균형 인자(0 = 가능도만, 1 = 다름만)
        max_length: 만들어 낼 최대 토막 수
        
    반환값:
        만든 차례
    """
    generated = input_ids.clone()
    
    for _ in range(max_length):
        with torch.no_grad():
            outputs = model(generated, output_hidden_states=True)
        
        logits = outputs.logits[0, -1, :]
        probs = F.softmax(logits, dim=-1)
        
        # 맥락 숨은 상태(마지막을 뺀 모든 자리)
        context_hidden = outputs.hidden_states[-1][0, :-1, :]  # [ctx_len, hidden]
        
        # 상위 k개 후보를 얻는다
        top_probs, top_indices = torch.topk(probs, k)
        
        best_score = float('-inf')
        best_token = top_indices[0].item()  # 붙박이는 상위 1
        
        for prob, token_id in zip(top_probs.tolist(), top_indices.tolist()):
            # 앞먹임으로 후보의 숨은 상태 얻기
            candidate_input = torch.cat(
                [generated, torch.tensor([[token_id]], device=generated.device)],
                dim=-1
            )
            with torch.no_grad():
                candidate_output = model(candidate_input, output_hidden_states=True)
            candidate_hidden = candidate_output.hidden_states[-1][0, -1, :]  # [hidden]
            
            # 아무 맥락 자리와의 최대 닮음
            similarities = F.cosine_similarity(
                candidate_hidden.unsqueeze(0),
                context_hidden,
                dim=-1
            )
            max_sim = similarities.max().item()
            
            # 대조 점수
            score = (1 - alpha) * prob - alpha * max_sim
            
            if score > best_score:
                best_score = score
                best_token = token_id
        
        generated = torch.cat(
            [generated, torch.tensor([[best_token]], device=generated.device)],
            dim=-1
        )
    
    return generated
```

---

## 13. 미리 짚어 풀기

**미리 짚어 풀기**(Leviathan 외, 2023; Chen 외, 2023)는 작은 "밑그림" 모델이 토막을 내놓고 큰 목표 모델이 이를 나란히 확인하게 하여 미룸을 빠르게 한다.

### 핵심 통찰

자기되돌리기로 만들어 내기는 **기억 공간에 묶여** 있다. 곧 토막마다 온전한 앞먹임이 필요하지만 셈은 대부분 기억 공간을 기다린다. 한 번의 훑기로 토막 여럿을 확인할 수 있다면 실제 시간을 아낀다.

### 알고리즘

1. 초안 모델이 자기되돌리기로 토막 $\gamma$개를 만든다
2. 과녁 모델이 앞으로 걸음 한 번에 토막 $\gamma$개를 모두 점수 매긴다
3. 물리치기 표집으로 토막마다 받아들이거나 물리친다
4. 물리쳤으면 바로잡은 분포에서 다시 뽑는다

### 옳음 보장

미리 짚어 풀기는 (같은 마구잡이 씨앗을 쓸 때) 목표 모델에서 보통대로 뽑는 것과 **정확히 같은 분포**를 낸다.

```python
def speculative_decode(
    draft_model,
    target_model,
    input_ids: torch.Tensor,
    gamma: int = 4,
    temperature: float = 1.0
) -> tuple[torch.Tensor, int]:
    """
    미리 짚어 풀기: 작은 모델로 밑그림을 그리고 큰 모델로 확인한다.
    
    인수:
        draft_model: 밑그림을 그릴 작고 빠른 모델
        target_model: 확인할 크고 정확한 모델
        input_ids: 처음 토막 번호 [1, seq_len]
        gamma: 바퀴마다 밑그림 그릴 토막의 개수
        temperature: 표집 온도
        
    반환값:
        (받아들인 토막 텐서, 받아들인 개수) 튜플
    """
    device = input_ids.device
    
    # 1단계: 작은 모델로 감마 개의 토막 밑그림 그리기
    draft_ids = []
    draft_probs = []
    current_ids = input_ids.clone()
    
    for _ in range(gamma):
        with torch.no_grad():
            draft_output = draft_model(current_ids)
            draft_logits = draft_output.logits[:, -1, :] / temperature
            probs = F.softmax(draft_logits, dim=-1)
        
        # 밑그림에서 뽑기
        token = torch.multinomial(probs, num_samples=1)
        draft_ids.append(token.item())
        draft_probs.append(probs[0, token.item()].item())
        
        current_ids = torch.cat([current_ids, token], dim=-1)
    
    # 2단계: 목표 모델로 모든 토막을 한 번에 확인하기
    with torch.no_grad():
        target_output = target_model(current_ids)
        target_logits = target_output.logits / temperature
    
    # 3단계: 물리치기 표집으로 받아들이거나 물리치기
    accepted = []
    input_len = input_ids.size(1)
    
    for i, (token, q_prob) in enumerate(zip(draft_ids, draft_probs)):
        # 자리 input_len + i의 목표 확률
        target_probs = F.softmax(target_logits[0, input_len + i - 1, :], dim=-1)
        p_prob = target_probs[token].item()
        
        # 받아들임 확률: min(1, p/q)
        accept_prob = min(1.0, p_prob / (q_prob + 1e-10))
        
        if torch.rand(1).item() < accept_prob:
            accepted.append(token)
        else:
            # 물리침: 고르게 맞춘 (p - q)+에서 뽑기
            # 이것이 "남은" 분포이다
            diff = target_probs - F.softmax(target_logits[0, input_len + i - 1, :], dim=-1)
            diff = torch.clamp(diff, min=0)
            if diff.sum() > 0:
                diff = diff / diff.sum()
                corrected_token = torch.multinomial(diff, num_samples=1)
                accepted.append(corrected_token.item())
            break
    
    accepted_tensor = torch.tensor([accepted], device=device)
    return accepted_tensor, len(accepted)
```

### 빨라짐 살피기

$\alpha$을 평균 받아들임 비율이라 하자. 되돌이마다 바라는 토막 수는 다음과 같다.

$$
\mathbb{E}[\text{tokens}] = \sum_{i=1}^{\gamma} \alpha^{i-1}(1-\alpha) \cdot i + \alpha^\gamma \cdot \gamma = \frac{1 - \alpha^{\gamma+1}}{1 - \alpha}
$$

빨라짐은 다음에 달렸다:

- 초안 모델의 좋음(더 높은 $\alpha$)
- 밑그림 모델과 목표 모델의 값 비율
- $\gamma$(초안이 많을수록 얻을 것도 크지만 물리면 버리는 일감도 는다)

---

## 14. 이끌어 만들어 내기

### 갈래 매개 없는 이끌기(CFG)

본디 퍼짐 모델에서 온 CFG는 글 만들어 내기도 낫게 할 수 있다:

$$
\tilde{z} = z_{\text{uncond}} + w \cdot (z_{\text{cond}} - z_{\text{uncond}})
$$

여기서 $w > 1$이 조건 걸기의 효과를 키운다.

```python
def classifier_free_guidance(
    model,
    input_ids: torch.Tensor,
    guidance_scale: float = 1.5,
    uncond_input_ids: Optional[torch.Tensor] = None
) -> torch.Tensor:
    """
    로짓에 갈래 매개 없는 이끌기 쓰기.
    
    인수:
        model: 말 모델
        input_ids: 조건을 건 들임 번호 [1, seq_len]
        guidance_scale: 키움 인자(w > 1이면 조건 걸기가 세진다)
        uncond_input_ids: 조건 없는 들임(보기로 빈 시킴말)
        
    반환값:
        이끈 로짓
    """
    with torch.no_grad():
        # 조건을 건 로짓
        cond_logits = model(input_ids).logits[:, -1, :]
        
        # 조건 없는 로짓
        if uncond_input_ids is None:
            # 차례의 첫머리를 "조건 없음"으로 쓰기
            uncond_input_ids = input_ids[:, :1]
        uncond_logits = model(uncond_input_ids).logits[:, -1, :]
    
    # 이끈 로짓
    guided_logits = uncond_logits + guidance_scale * (cond_logits - uncond_logits)
    
    return guided_logits
```

---

## 15. 온전한 만들어 내기 물길

모두 모아 보면:

```python
class TextGenerator:
    """여러 표집 전략을 갖춘 유연한 글 만들어 내기."""
    
    def __init__(
        self,
        model,
        tokenizer,
        device: str = 'cuda'
    ):
        self.model = model.to(device).eval()
        self.tokenizer = tokenizer
        self.device = device
    
    def generate(
        self,
        prompt: str,
        max_new_tokens: int = 100,
        temperature: float = 1.0,
        top_k: int = 0,
        top_p: float = 1.0,
        min_p: float = 0.0,
        typical_p: float = 1.0,
        repetition_penalty: float = 1.0,
        frequency_penalty: float = 0.0,
        presence_penalty: float = 0.0,
        no_repeat_ngram_size: int = 0,
        eos_token_id: Optional[int] = None,
        **kwargs
    ) -> str:
        """
        표집 전략을 정할 수 있는 글 만들어 내기.
        
        인수:
            prompt: 입력 글
            max_new_tokens: 만들 토큰의 최대 개수
            temperature: 표집 온도
            top_k: 상위 k 거르기(0이면 끔)
            top_p: 알갱이 표집 문턱값(1.0이면 끔)
            min_p: 최소 p 표집 문턱값(0.0이면 끔)
            typical_p: 전형 표집 문턱값(1.0이면 끔)
            repetition_penalty: 되풀이된 토막에 곱하는 벌
            frequency_penalty: 토막이 나올 때마다 더하는 벌
            presence_penalty: 토막이 있으면 더하는 벌
            no_repeat_ngram_size: n-그램 되풀이 막기(0이면 끔)
            eos_token_id: 이 토막에서 만들어 내기를 멈춘다
            
        반환값:
            만든 글자열
        """
        # 프롬프트를 인코딩한다
        input_ids = self.tokenizer.encode(prompt, return_tensors='pt').to(self.device)
        generated_ids = input_ids[0].tolist()
        
        if eos_token_id is None:
            eos_token_id = self.tokenizer.eos_token_id
        
        for _ in range(max_new_tokens):
            # 순전파
            input_tensor = torch.tensor([generated_ids], device=self.device)
            with torch.no_grad():
                outputs = self.model(input_tensor)
                logits = outputs.logits[0, -1, :].clone()
            
            # 벌주기 쓰기
            if repetition_penalty != 1.0:
                for token_id in set(generated_ids):
                    if logits[token_id] > 0:
                        logits[token_id] /= repetition_penalty
                    else:
                        logits[token_id] *= repetition_penalty
            
            if frequency_penalty > 0 or presence_penalty > 0:
                token_counts = {}
                for tid in generated_ids:
                    token_counts[tid] = token_counts.get(tid, 0) + 1
                for tid, count in token_counts.items():
                    logits[tid] -= frequency_penalty * count
                    logits[tid] -= presence_penalty
            
            if no_repeat_ngram_size > 0:
                logits = apply_no_repeat_ngram(logits, generated_ids, no_repeat_ngram_size)
            
            # 온도 맞추기
            if temperature != 1.0:
                logits = logits / temperature
            
            # 거르기 전략(차례대로 쓴다)
            if top_k > 0:
                top_k_vals, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < top_k_vals[-1]] = float('-inf')
            
            if min_p > 0:
                probs = F.softmax(logits, dim=-1)
                threshold = min_p * probs.max()
                logits[probs < threshold] = float('-inf')
            
            if top_p < 1.0:
                sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                cumsum = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                mask = cumsum > top_p
                mask[1:] = mask[:-1].clone()
                mask[0] = False
                sorted_logits[mask] = float('-inf')
                logits = torch.zeros_like(logits).scatter_(-1, sorted_indices, sorted_logits)
            
            if typical_p < 1.0:
                probs = F.softmax(logits, dim=-1)
                log_probs = F.log_softmax(logits, dim=-1)
                entropy = -(probs * log_probs).sum()
                deviation = torch.abs(-log_probs - entropy)
                sorted_dev, sorted_idx = torch.sort(deviation)
                sorted_probs = probs[sorted_idx]
                cumsum = torch.cumsum(sorted_probs, dim=-1)
                mask = cumsum > typical_p
                mask[1:] = mask[:-1].clone()
                mask[0] = False
                remove_idx = sorted_idx[mask]
                logits[remove_idx] = float('-inf')
            
            # 뽑기
            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1).item()
            
            generated_ids.append(next_token)
            
            if next_token == eos_token_id:
                break
        
        return self.tokenizer.decode(generated_ids, skip_special_tokens=True)
```

---

## 16. 견줌과 권하는 바

### 전략 견줌

| 전략 | 여러 갈래임 | 조리 | 빠르기 | 알맞게 맞춤 |
|----------|-----------|-----------|-------|------------|
| 욕심쟁이 | 낮음 | 높음 | ★★★★★ | 없음 |
| 온도 | 다듬을 수 있음 | 들쭉날쭉 | ★★★★★ | 없음 |
| 상위 k | 가운데 | 좋음 | ★★★★★ | 없음 |
| 상위 p(알갱이) | 가운데 | 좋음 | ★★★★☆ | 분포의 꼴 |
| 최소 p | 가운데 | 좋음 | ★★★★☆ | 최대 확률 |
| 전형 | 가운데~높음 | 좋음 | ★★★★☆ | 엔트로피 |
| 에타 | 가운데 | 좋음 | ★★★★☆ | 엔트로피 |
| 미로스탯 | 가운데 | 좋음 | ★★★★☆ | 흐르는 헷갈림도 |
| 빔 찾기 | 낮음 | 높음 | ★★☆☆☆ | 없음 |
| 맞대어 찾기 | 높음 | 높음 | ★★☆☆☆ | 맥락의 닮음 |
| 미리 짚기 | 들쭉날쭉 | 들쭉날쭉 | ★★★★★* | 밑그림 모델의 좋음 |

*미리 짚어 풀기의 빠르기는 받아들임 비율과 모델 값 비율에 달렸다.

### 일마다 권하는 자리매김

| 일 | 온도 | 상위 p | 상위 k | 그 밖 |
|------|-------------|-------|-------|-------|
| 코드 만들기 | 0.2~0.4 | 0.95 | — | 낮은 되풀이 벌주기 |
| 창작 글쓰기 | 0.8~1.0 | 0.9 | 50 | 알맞은 되풀이 벌주기 |
| 대화/채팅 | 0.7 | 0.9 | 40 | 있음 벌주기 0.1~0.3 |
| 간추리기 | 0.3~0.5 | 0.9 | — | 3-그램 되풀이 막기 |
| 옮김 | 0.0(욕심쟁이) | — | — | 또는 빔 찾기 |
| 사실 물음 답하기 | 0.0~0.3 | 0.95 | — | — |

### 판단 흐름도

```
Start
  │
  ├─ Need exact reproducibility? ──Yes──► Greedy (T=0)
  │
  ├─ Factual/code task? ──Yes──► Low temperature (0.2-0.4) + Top-p (0.95)
  │
  ├─ Creative task? ──Yes──► Higher temperature (0.7-1.0) + Top-p (0.9)
  │
  ├─ Experiencing repetition? ──Yes──► Add repetition/frequency penalty
  │
  ├─ Need diverse outputs? ──Yes──► Contrastive or diverse beam search
  │
  └─ Need speed? ──Yes──► Speculative decoding (if draft model available)
```

---

## 연습문제

**연습문제 1.**
$n$-그램 말 모델에서 모델의 복잡함과 자료의 성김 사이 맞바꿈을 밝혀라. $n$을 늘려도 왜 늘 헷갈림도가 나아지지는 않는가?

??? success "연습문제 1 풀이"
    $n$이 클수록 더 먼 매임을 담지만 확률을 믿을 만하게 어림하려면 자료가 지수만큼 더 든다. 크기가 $V$인 낱말 사전에는 $n$낱말이 $V^n$가지 있을 수 있는데, 대부분은 익힘 뭉치에 한 번도 나오지 않아 확률이 0으로 어림된다. 매끄럽게 하기(라플라스, 크네서-네이)로 이를 눅일 수는 있으나 온전히 메우지는 못한다. 참으로는 어지간한 크기의 뭉치에서 3낱말이나 5낱말이 더 높은 차수의 모델보다 나은 일이 잦은데, 자료가 적을 때는 치우침-흩어짐 맞바꿈이 더 단순한 모델의 손을 들어 주기 때문이다.

---

**연습문제 2.**
헷갈림도를 정의하고 엇갈린 엔트로피와의 관계를 보여라. 어떤 모델이 시험 뭉치에서 헷갈림도 50을 얻었다. 직관으로 이는 무슨 뜻인가?

??? success "연습문제 2 풀이"
    헷갈림도는 $\text{PP} = 2^{H(p, q)}$으로 매기며, $H(p, q) = -\frac{1}{N}\sum_{i=1}^N \log_2 q(w_i | w_{<i})$은 참 분포가 $p$인 시험 자료에서 모델 $q$의 엇결 엔트로피다. 헷갈림도가 50이면 모델이 평균으로 보아 자리마다 낱말 50개 가운데 고르게 하나를 고르는 만큼 아리송하다는 뜻이다. 헷갈림도가 낮을수록 미루어 봄이 낫다.

---

**연습문제 3.**
글 만들어 내기에서 상위 $k$ 표집, 알갱이(상위 $p$) 표집, 온도 맞추기를 견주어라. 저마다 언제 쓰겠는가?

??? success "연습문제 3 풀이"
    **위 $k$**: 확률이 가장 높은 토막 $k$개에서 뽑는다. 단순하지만 붙박인 $k$은 (뾰족한 분포에서는) 너무 좁고 (평평한 분포에서는) 너무 헐거울 수 있다. **알맹이/위 $p$**: 쌓은 확률이 $p$을 넘는 가장 작은 토막 묶음에서 뽑아 분포의 꼴에 맞춘다. **온도 $\tau$**: 소프트맥스 앞에서 로짓에 $1/\tau$을 곱한다. $\tau < 1$이면 뾰족해지고 $\tau > 1$이면 평평해진다. 열린 글 만들기에는 흔히 위 $p$을 쓰고(맞추어 잘라 낸다), 온도는 새로움을 다스리는 데 쓸모 있으며, 위 $k$은 옭아맨 글 만들기에 단순하고 잘 듣는다.

---

**연습문제 4.**
신경 말 모델은 왜 낱말 수준이나 글자 수준 토막내기 대신 아래낱말 토막내기(보기로 BPE)를 쓰는가?

??? success "연습문제 4 풀이"
    **낱말 수준** 토막내기는 낱말 곳간이 크고(10만 이상) 드문 낱말에 매개변수를 낭비하며 곳간 밖 낱말을 다루지 못한다. **글자 수준** 토막내기는 곳간은 아주 작지만 차례가 무척 길어져 먼 거리 얽힘을 담아내기 어렵고 익히기가 느리다. **아래낱말** 방법(BPE, WordPiece, Unigram)은 둘의 균형을 잡는다. 곧 알맞은 크기의 곳간(3만 2천~6만 4천)을 지니고, 아래낱말을 엮어 어떤 낱말이든 나타낼 수 있으며(곳간 밖 낱말 없음), 꼴이 비슷한 낱말끼리 통계 앎을 나눠 갖고, 알맞게 짧은 차례를 낸다.

## 정리하며

1. **p=0.9인 상위 p(알갱이) 표집**은 대부분의 일에 든든한 붙박이 값이다
2. **온도**는 마구잡이 정도를 직관적으로 다스리게 해 준다
3. **전략을 아울러라**: 온도 + 상위 p + 되풀이 벌주기면 대부분의 필요를 덮는다
4. **알맞게 맞추는 방법**(전형, 에타, 미로스탯)은 한결같음을 낫게 할 수 있다
5. **일에 맞춘 다듬기**가 내놓음의 좋음에 크게 영향을 준다
6. **미리 짚어 풀기**는 좋음을 잃지 않고 크게 빨라지게 한다

---

**참고 문헌**

1. Fan, A., Lewis, M., & Dauphin, Y. (2018). Hierarchical Neural Story Generation. *ACL*.

2. Holtzman, A., Buys, J., Du, L., Forbes, M., & Choi, Y. (2020). The Curious Case of Neural Text Degeneration. *ICLR*.

3. Meister, C., Pimentel, T., Wiher, G., & Cotterell, R. (2023). Locally Typical Sampling. *TACL*.

4. Su, Y., Lan, T., Wang, Y., Yogatama, D., Kong, L., & Collier, N. (2022). A Contrastive Framework for Neural Text Generation. *NeurIPS*.

5. Basu, S., Ramachandran, G. S., Keskar, N. S., & Varshney, L. R. (2021). Mirostat: A Neural Text Decoding Algorithm that Directly Controls Perplexity. *ICLR*.

6. Leviathan, Y., Kalman, M., & Matias, Y. (2023). Fast Inference from Transformers via Speculative Decoding. *ICML*.

7. Chen, C., Borgeaud, S., Irving, G., Lespiau, J.-B., Sifre, L., & Jumper, J. (2023). Accelerating Large Language Model Decoding with Speculative Sampling. *arXiv*.

8. Vijayakumar, A. K., et al. (2018). Diverse Beam Search: Decoding Diverse Solutions from Neural Sequence Models. *AAAI*.
