# 텍스트 생성의 디코딩·샘플링 전략

자기회귀 언어 모델은 각 prefix에서 다음 토큰의 로짓을 만든다. 디코딩은 그 분포를 사용하거나 변형해 실제 토큰을 선택하는 규칙이다. 생성의 다양성·반복·모델 로그확률·실제 유용성은 다른 평가 기준이며 낮은 온도나 큰 beam만으로 사실성이 보장되지는 않는다. 아래 코드는 각 방법의 예와 골격으로, 서로 다른 입력·배치·장치 계약을 확인해야 한다.

## 1. 탐욕적 선택과 종료 상태

탐욕적 디코딩은 매 단계 argmax를 택한다. 국소 최댓값의 선택이지 전체 문장의 최대 확률 탐색이 아니다. 동점 규칙·평가 모드·실행 환경까지 같아야 재현성을 기대할 수 있고, 특정 답을 반복한다는 사실만으로 정답인 것은 아니다.

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

이 함수의 max_length는 **추가 생성 반복 수**다. 모델 반환에 .logits가 있어야 하며 no_grad만 적용하고 eval은 호출하지 않는다. 배치 종료는 모든 행이 **같은 반복에서** EOS를 출력할 때만 멈춘다. 이미 EOS를 낸 행을 따로 고정하지 않아 그 뒤에도 다른 토큰이 붙을 수 있다. 입력 padding·attention_mask·KV cache도 사용하지 않고 전체 prefix를 다시 계산한다.

## 2. 온도와 엔트로피

T>0에서 p_T(v)=softmax(z/T)_v다. 유한 어휘·유한 로짓이면 T→∞에서 균일 분포, T→0+에서 **최대 로짓을 공유하는 토큰들의 균일 분포**가 된다. 최댓값이 유일할 때만 한 점으로 모인다. −∞로 제외한 후보가 있으면 남은 지원 집합을 기준으로 해석한다.

자연로그 엔트로피 H에서 다음 식은 온도에 대한 비감소성을 보여 준다.

$$\frac{dH(p_T)}{dT}=\frac{\operatorname{Var}_{p_T}(z)}{T^3}\geq0.$$

이 성질은 한 단계의 같은 로짓에 대한 것이며 전체 생성의 정답률·다양성·논리성을 온도만으로 정하지 않는다. T=0은 위 식의 입력이 아니고 별도의 greedy 분기가 필요하다.

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

이 함수는 양의 온도를 확인하지만 다른 샘플러 대부분은 같은 검사를 하지 않는다. logits의 shape·유효값과 num_samples·복원 여부도 확인해야 한다. 기본 multinomial은 replacement=False이므로 num_samples>1은 같은 독립 복원 표본을 여러 번 뽑는 것과 다르다.

## 3. Top-k와 nucleus(top-p)

Top-k는 확률 순위의 상위 후보만 남겨 재정규화한다. 고정된 k는 매우 뾰족한 분포에서는 작은 꼬리까지 포함하고, 평평한 분포에서는 그럴듯한 후보를 많이 제외할 수 있다. 예를 들어20개가 균일하면 각 확률은0.05이며 k10은 그중 절반을 남긴다.

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
    
    # k번째로 큰 값(임계값) 찾기
    top_k_values, _ = torch.topk(logits, k, dim=-1)
    threshold = top_k_values[:, -1, None]  # [batch, 1]
    
    # 임계값 아래 토막을 0으로 만들기
    logits = torch.where(logits >= threshold, logits, filter_value)
    
    probs = F.softmax(logits, dim=-1)
    return torch.multinomial(probs, num_samples=1)
```

위 코드는 k번째 **값 이상**을 남겨 동점이 있으면 k개보다 많이 유지할 수 있다. B×V의2차원 입력과1≤k≤V, 양의 온도를 가정한다. 순위는 양의 온도 변화로 바뀌지 않지만 남은 후보의 상대 확률은 바뀐다.

Nucleus는 확률 내림차순의 **최소 prefix**로 누적 질량이 p에 도달하도록 선택한다. p∈(0,1]이며 경계를 넘게 만드는 토큰도 포함한다. 임의의 부분집합 cardinality만 최소화하는 문제와 같은 알고리즘으로 쓰지 않고, 정렬·동점 규칙을 명시한다. 후보 수는 분포에 적응하지만 엔트로피 하나만으로 결정되지 않는다.

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
        p: 쌓인 확률 임계값
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
    # 임계값을 넘는 토막을 남기려 오른쪽으로 밀기
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

기본 min_tokens_to_keep1에서는 삭제 마스크를 한 칸 밀어 경계 토큰을 보존한다. 인수를 더 크게 주면 현재 구현은 삭제 마스크 전체를 그 수만큼 이동해 단순히 ‘최소m개’만 보장하는 규칙과 다르게 추가 후보를 유지할 수 있다. p·최소 후보 수의 유효 범위와 배치 shape를 확인한다.

## 4. 결합 순서와 min-p

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

combined_sample은 온도→top-k→그 결과에서 다시 정규화한 top-p→샘플링 순서다. 원래 분포에서 독립 계산한 top-k 집합과 nucleus 집합의 단순 교집합과는 다를 수 있다. 순서와 재정규화 기준을 바꾸면 다른 분포가 된다. 모든 도구가 이 순서를 강제하는 보편 표준이라고 설명하지 않는다.

min-p는 p(v)≥m·p_max인 후보를 유지한다. m은 최대 확률에 대한 비율이며 유효한 범위0≤m≤1에서 최소 최댓값 후보는 남는다. p_max가 커진다는 사실만으로 후보 수의 단조 변화가 정해지지는 않는다. 다른 후보의 상대 분포도 중요하다.

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
    
    # 최대 확률에 바탕한 그때그때의 임계값
    max_prob = probs.max(dim=-1, keepdim=True).values
    threshold = p_min * max_prob
    
    # 임계값 아래 토막 가리기
    mask = probs < threshold
    logits = logits.masked_fill(mask, float('-inf'))
    
    probs = F.softmax(logits, dim=-1)
    return torch.multinomial(probs, num_samples=1)
```

m0은 양수 확률 후보를 모두, m1은 최대 확률 동점들을 남기는 경계다. 이 함수는 m·온도를 검증하지 않으며 부동소수점 underflow나 미리 마스킹한 후보의 처리도 고려해야 한다. 모든 방법을 조합하면 원하는 후보가 남는지 다시 확인한다.

## 5. Locally typical sampling과 η cutoff

Locally typical sampling은 확률 순위 대신 정보량−log p(v)와 엔트로피 H의 차이가 작은 후보부터 질량을 채운다. 토큰 수준의 국소 typicality 규칙이며, 장기 시퀀스의 AEP가 이 디코더의 품질·정확성을 보장하는 정리는 아니다. 높은 확률 토큰이 먼저 선택되는 것과도 다르다. [연구 논문](https://arxiv.org/abs/2202.00666)의 목적과 평가 조건을 구분한다.

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
        p: 전형 모음의 쌓인 확률 임계값
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

코드는0·−∞가 섞인 필터 결과에서 p·log p를 곱하면0×(−∞)=NaN이 될 수 있다. 따라서 다른 필터 뒤에 그대로 연결해도 항상 유효한 엔트로피가 계산된다고 가정하면 안 된다. 유효 지원 집합과0 log0의 수치 처리가 필요하다.

[η-sampling](https://arxiv.org/abs/2210.15191)은 H로 정한 절대 확률 cutoff를 사용한다.

$$\eta=\min(\epsilon,\sqrt\epsilon\exp(-H)).$$

이는 typicality 순위와 같지 않으며 ‘엔트로피가 크면 기본적으로 top-p로 전환하는 방법’도 아니다. 큰 H에서는 cutoff가 더 낮아질 수 있다.

```python
def eta_sample(
    logits: torch.Tensor,
    epsilon: float = 0.0003,
    fallback_p: float = 0.9,
    temperature: float = 1.0
) -> torch.Tensor:
    """
    에타 표집: 엔트로피에 바탕한 알맞게 맞추는 임계값.
    
    인수:
        logits: 고르게 맞추지 않은 로그 확률 [batch, vocab]
        epsilon: 바탕 임계값 파라메터
        fallback_p: 에타 거르기가 토막을 모두 없앴을 때 쓸 상위 p 임계값
        temperature: 온도 맞추기
        
    반환값:
        뽑은 토막 번호 [batch, 1]
    """
    logits = logits / temperature
    probs = F.softmax(logits, dim=-1)
    log_probs = F.log_softmax(logits, dim=-1)
    
    # 엔트로피 셈하기
    entropy = -(probs * log_probs).sum(dim=-1, keepdim=True)
    
    # 알맞게 맞추는 임계값 에타 셈하기
    eta = torch.minimum(
        torch.tensor(epsilon),
        torch.sqrt(torch.tensor(epsilon)) * torch.exp(-entropy)
    )
    
    # 임계값 아래 토막 거르기
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

fallback_p는 이 예제의 방어 분기다. 어떤 배치 행이라도 후보가0이면 현재 코드는 **배치 전체**를 nucleus로 다시 뽑아 다른 행의 방법까지 바꾼다. 일반적인 유효 확률·ε범위와 수치 처리에서는 최대 후보가 남는지 확인하고, 오류·빈 지원 집합의 처리는 행별로 명시한다. entropy 계산의 NaN·장치·dtype와 ε범위도 검증해야 한다.

## 6. Mirostat의 피드백과 로그 단위

[Mirostat](https://arxiv.org/abs/2007.14966)은 선택된 토큰의 surprisal을 목표 수준으로 조절하는 피드백 방식이다. 아래 v2형 코드에서는 μ로 p(v)≥2^(−μ)의 후보를 정하고 관측 정보량과 목표τ의 차이로 μ를 갱신한다. 온도를 계속 바꾸는 구현은 아니며 원래 버전의 adaptive top-k와도 구분한다.

```python
def mirostat_v2_sample(
    logits: torch.Tensor,
    tau: float = 5.0,       # 목표 놀라움
    eta: float = 0.1,       # 러닝 비율
    mu: float = None        # 지금의 놀라움 어림값(상태)
) -> tuple[torch.Tensor, float]:
    """
    미로스탯-2 표집: 헷갈림도를 알맞게 좇기.
    
    인수:
        logits: 고르게 맞추지 않은 로그 확률 [batch, vocab]
        tau: 목표 놀라움(목표 헷갈림도의 밑 2 로그)
        eta: 놀라움 좇기의 러닝 비율
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
    # mu를 그때그때의 임계값으로 쓰기
    cumsum_probs = torch.cumsum(sorted_probs, dim=-1)
    
    # 미로스탯-2: mu에서 이끌어 낸 확률 임계값에서 잘라 내기
    # k = 남길 토막의 개수
    prob_threshold = torch.exp2(torch.tensor(-mu))
    
    # 확률이 임계값 이상인 토막 남기기
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

log2를 쓰므로 τ는 **bit 단위**다. τ5는 PPL2^5=32에 대응하며 exp5≈148은 nat일 때의 값이다. 이는 생성된 토큰의 원 모델 surprisal을 제어하려는 목표이지 정답 시험 집합의 PPL을 같은 값으로 보장한다는 뜻은 아니다. log2(probs+1e−10)는 매우 작은 확률의 실제 정보량을 절단하는 근사이기도 하다.

.item()을 사용한 단일 observed_surprisal과 μ 때문에 실질적으로 배치1을 가정한다. 빈 후보의 검사는 전체 mask 합으로 해 행별 빈 집합을 처리하지 않으며 batch가 커지면 일부 행의 정규화가0일 수 있다. 다음 호출에도 μ를 전달해야 피드백 상태가 이어진다.

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

이 생성 루프에는 EOS 종료·평가 모드 전환·cache가 없다. 매번 전체 prefix를 계산하므로 목표 surprisal을 정했다고 전체 서비스 속도나 문장 완결성이 보장되지는 않는다.

## 7. 반복 벌점과 금지 n-그램

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
        logits: 지금 스텝의 로짓 [batch, vocab]
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

곱셈형 repetition penalty는 이미 나온 토큰의 양수 로짓을 나누고 음수는 곱해 상대 점수를 줄인다. 벌점>1을 가정하며 등장 횟수에 비례하는 규칙은 아니다. 로짓에 공통 상수를 더하면 원 softmax는 같지만 이 부호 기반 변형 결과는 달라질 수 있다. 고유명사·코드·문법상 필요한 반복까지 억제할 수 있어 목적에 맞춰 평가한다.

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
        logits: 지금 스텝의 로짓 [batch, vocab]
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

frequency penalty는 횟수, presence penalty는 한 번이라도 등장했는지에 따라 가산 조절한다. 이 구현의 token_counts는 기본 float32인데 scatter_add의 src는 logits.dtype을 써 FP16 등에서 타입이 어긋날 수 있다. 입력 ID 범위·배치·장치·dtype를 맞춰야 한다. 음수 계수는 오히려 반복을 장려할 수 있으므로 의미를 명시한다.

```python
def apply_no_repeat_ngram(
    logits: torch.Tensor,
    generated_ids: list[int],
    n: int = 3
) -> torch.Tensor:
    """
    이미 나온 n-그램이 다시 나오지 못하게 막기.
    
    인수:
        logits: 지금 스텝의 로짓 [vocab]
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

금지 n-그램 함수는1차원 로짓을 제자리 수정한다. 현재 n1에서는 `[-0:]`가 전체 목록을 뜻해 빈 prefix 규칙과 달라, unigram 반복 금지를 일반적으로 구현하지 못한다. n≥2와 입력 길이를 확인해야 한다. 모든 후보가 금지되면 softmax·multinomial이 유효하지 않으므로 남은 지원 집합·종료·fallback을 정의한다.

## 8. Beam search와 다양한 가설

Beam search는 여러 prefix를 유지하는 근사 탐색이며 제한된 beam에서 전체 최적열을 보장하지 않는다. 길이 정규화는 log확률을((5+길이)/6)^α로 나누는 등 목적을 바꾸는 선택이다. α>0이면 음수 로그합에 대한 길이 불이익을 줄일 수 있지만 임의의 길이가 항상 더 좋은 것은 아니다.

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

현재 코드는 input_ids[0]만 사용하고 각 beam마다 model을 별도 호출한다. num_beams×2≤어휘 크기·비어 있지 않은 prompt를 가정하며 cache·eval 전환이 없다. 정규화 길이에는 prompt도 포함된다. 완료 가설 개수만으로 early stop하는 것은 남은 가설의 최종 점수 상한을 검사한 최적성 보장이 아니다. 최대 길이에서 EOS 없는 미완성 beam도 반환할 수 있다.

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
    여러 클래스 빔 찾기: 배치 찾기로 여러 클래스의 가설 만들기.
    
    인수:
        model: 말 모델
        input_ids: 처음 토막 번호
        num_beams: 배치마다의 빔 수
        num_groups: 빔 배치의 개수
        diversity_penalty: 앞 배치가 고른 토막에 주는 벌
        max_length: 만들어 낼 최대 길이
        
    반환값:
        (score, sequence) 튜플을 담은 배치의 목록
    """
    device = input_ids.device
    all_groups = []
    
    for group_idx in range(num_groups):
        # 자리마다 앞 배치가 고른 토막 좇기
        previous_group_tokens = []
        for prev_group in all_groups:
            for _, seq in prev_group:
                previous_group_tokens.extend(seq[len(input_ids[0]):])
        
        # 여러 클래스임 벌주기를 곁들인 빔 찾기 돌리기
        group_beams = beam_search_with_penalty(
            model, input_ids, num_beams, max_length,
            penalty_tokens=set(previous_group_tokens),
            penalty_weight=diversity_penalty
        )
        
        all_groups.append(group_beams)
    
    return all_groups
```

diverse_beam_search는 정의되지 않은 beam_search_with_penalty에 의존한다. num_beams를 그룹마다 전달하고 이전 그룹의 **전체 생성 토큰**을 모아 벌점을 주려는 골격으로, 같은 단계의 그룹 간 다양성 벌점과 자동으로 동일한 알고리즘은 아니다. 선언된 함수만으로 완성된 diverse beam 실험이 제공되는 것은 아니다.

## 9. Contrastive search의 표현 비교

Contrastive search는 상위 후보의 모델 확률과, 후보를 이어 붙여 얻은 **문맥화 hidden 표현**이 이전 표현과 갖는 유사도를 함께 고려한다. 단순 토큰 embedding의 사전적 의미 차이를 최대화하는 규칙은 아니다. 웨이트와 top-k를 정하고 어떤 과거 위치를 비교하는지 명시한다.

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
            # 피드포워드로 후보의 숨은 상태 얻기
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

이 함수는 배치1을 가정하고 context_hidden에서 마지막 기존 토큰을 제외한다. prompt가 한 토큰이면 비교 집합이 비어 max가 실패할 수 있고, 일반적인 모든 기존 문맥과의 비교와도 다르다. 후보마다 전체 prefix를 다시 계산해 비용이 크며 EOS·cache·eval 전환은 없다. 선택된 표현 유사도가 낮다는 사실만으로 사실성·의미적 새로움이 보장되지는 않는다.

## 10. Speculative decoding의 정확성 조건

[Speculative decoding](https://proceedings.mlr.press/v202/leviathan23a.html)은 초안 q의 후보를 목표 p로 검증한다. 같은 prefix의 초안 토큰x를 min(1,p(x)/q(x))로 수락하고, 거절 시 **전체 분포의 양의 차이 (p−q)_+를 정규화한 잔여 분포**에서 대체 토큰을 뽑는다. 모두 수락하면 목표 모델의 다음 분포에서 추가 토큰을 뽑는 구성도 사용한다. 정확한 알고리즘의 목표 분포 보존은 같은 seed에서 같은 문자열을 낸다는 뜻이 아니다.

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

위 코드는 거절 분기에서 target_probs에서 **같은 target softmax를 다시 빼** 차이가 항상0이 된다. q 전체 분포를 저장하지도 않아 필요한 잔여 분포를 구하지 못한다. 거절 시 대체 토큰 없이 끝나 아무 토큰도 반환하지 않을 수 있고, 모두 수락한 뒤의 추가 토큰도 없다. 따라서 이 구현에 분포 보존·품질 무로스 보장을 적용할 수 없다. q 분모의 ε도 정확 수락비와는 다른 수치 근사다.

모든 단계의 조건부 수락률을 상수 α로 근사하고 거절 시 대체1개·전부 수락 시 추가1개를 내는 알고리즘이라면1회 평균 출력 수는 Σ_{i=0}^γ α^i=(1−α^(γ+1))/(1−α)다. α=1에서는 γ+1의 극한을 쓴다. 단순 평균 수락률만으로 항상 각 prefix 확률이 α^i가 되는 것은 아니며, 이 식은 대체·추가가 없는 위 코드의 반환 수식이 아니다. 실제 속도는 두 모델·검증·cache·장치·배치·거절률에 달렸다.

## 11. Classifier-free guidance의 조건 비교

```python
def classifier_free_guidance(
    model,
    input_ids: torch.Tensor,
    guidance_scale: float = 1.5,
    uncond_input_ids: Optional[torch.Tensor] = None
) -> torch.Tensor:
    """
    로짓에 클래스 매개 없는 이끌기 쓰기.
    
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

코드는 z_uncond+w(z_cond−z_uncond)를 계산한다. w=1이면 조건 로짓, w=0이면 비조건 로짓이다. 의미 있는 guidance에는 의도한 조건만 다르고 생성 prefix는 대응하는 두 분포가 필요하다. 기본 uncond를 첫 토큰 하나로 두면 전체 생성 문맥까지 없애는 것이라 조건 제거와 같지 않을 수 있다. 단순 로짓 차이가 모든 모델·조건에서 품질 향상을 보장하지 않는다.

## 12. 통합 생성기의 실제 순서와 한계

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
            top_p: 알갱이 표집 임계값(1.0이면 끔)
            min_p: 최소 p 표집 임계값(0.0이면 끔)
            typical_p: 전형 표집 임계값(1.0이면 끔)
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

이 TextGenerator는 배치1의 prompt를 포함한 전체 문자열을 반환한다. 반복 벌점도 prompt 토큰을 포함한다. 실제 순서는 반복·frequency/presence·n-그램 제약→온도→top-k→min-p→top-p→typical→샘플링이다. 각 단계는 앞 필터의 결과에서 확률을 다시 계산하므로 개별 샘플러와 독립적인 집합 교집합으로 읽지 않는다.

typical 전에 −∞ 필터가 있으면0×(−∞) 엔트로피가 NaN이 될 수 있다. T=0을 greedy로 분기하지 않고 나누기 때문에 ‘온도0 선택’은 지원되는 계약이 아니다. 유효 파라미터·모든 후보 금지·빈 prompt·최대 문맥·장치 지원의 검증도 부족하다. **kwargs는 사용하지 않아 임의 옵션을 넘겨도 효과가 없고, KV cache 없이 전체 prefix를 반복 계산한다.

다음 기존 흐름도는 방법을 고르는 개념도일 뿐 구현의 검증·보편적 추천표가 아니다. ‘재현성이 필요하면 T=0’도 이 함수에서는 동작하지 않으므로 별도 greedy 경로가 필요하다.

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

## 13. 비교할 때 보고할 내용

모델·prompt·평가 자료·토큰화·생성 길이·seed·실행 모드·필터 순서·반복 벌점 대상과 EOS를 기록한다. 정답성·반복률·다양성·사용자 평가·지연시간을 목적에 맞게 측정한다. 특정 온도·top-p 값이나 별점 속도표를 모든 모델·과제의 정답으로 제시하지 않는다. 낮은 온도도 오답을 확신해서 선택할 수 있고, beam이나 speculation의 품질·비용도 구현과 목적에 달렸다.

## 연습문제

<div class="drillbox" markdown>

**연습문제 1.** <span class="diff med" title="중간"></span>
같은 최대 로짓을 가진 토큰이 두 개일 때 T→0+ 극한과 greedy 선택을 비교하라.

</div>

??? success "풀이"

    softmax의 극한은 두 최댓값 토큰에 균일하게 질량을 준다. greedy는 동점 해소 규칙에 따라 하나를 고를 수 있다. 양의 온도 공식에0을 직접 대입하는 것과 별도 greedy 분기를 구분한다.



---

<div class="drillbox" markdown>

**연습문제 2.** <span class="diff hard" title="어려움"></span>
Mirostat 코드의 τ5는 PPL32인가,148인가?

</div>

??? success "풀이"

    log2 정보량을 쓰므로2^5=32에 대응한다. nat 단위5일 때 exp5≈148이다. 생성된 토큰의 surprisal 목표와 정답 시험 집합의 PPL도 같은 평가가 아니다.



---

<div class="drillbox" markdown>

**연습문제 3.** <span class="diff med" title="중간"></span>
top-k 후 top-p가 원래 분포에서 구한 두 집합의 단순 교집합과 다른 이유를 설명하라.

</div>

??? success "풀이"

    top-k로 잘린 질량을 재정규화한 뒤 top-p의 누적 기준을 계산하기 때문이다. 온도·동점·최소 후보 수와 필터 순서까지 같아야 같은 샘플링 분포를 얻는다.



---

<div class="drillbox" markdown>

**연습문제 4.** <span class="diff med" title="중간"></span>
speculative_decode가 목표 분포를 보존하지 못하는 코드상의 이유를 설명하라.

</div>

??? success "풀이"

    거절 분기에서 p−q가 아니라 p−p를 계산해 잔여 분포가0이다. 초안의 전체 분포도 저장하지 않고 대체·추가 토큰 처리도 빠져 있다. 정확한 speculative 알고리즘의 보장을 이 코드에 적용할 수 없다.



## 정리하며

샘플링은 로짓·지원 집합·상태·종료를 다루는 확률 규칙이다. 온도·필터 순서·로그 단위·반복 제약·배치 계약을 확인하고, 현재 코드의 미완성 beam 의존 함수·entropy NaN·speculative 잔여 분포 오류를 알고 읽어야 한다. 정답성과 속도는 실제 목적·환경에서 별도로 평가한다.

**참고 문헌**

1. Fan, A., Lewis, M., & Dauphin, Y. (2018). Hierarchical Neural Story Generation. *ACL*.

2. Holtzman, A., Buys, J., Du, L., Forbes, M., & Choi, Y. (2020). The Curious Case of Neural Text Degeneration. *ICLR*.

3. Meister, C., Pimentel, T., Wiher, G., & Cotterell, R. (2023). Locally Typical Sampling. *TACL*.

4. Su, Y., Lan, T., Wang, Y., Yogatama, D., Kong, L., & Collier, N. (2022). A Contrastive Framework for Neural Text Generation. *NeurIPS*.

5. Basu, S., Ramachandran, G. S., Keskar, N. S., & Varshney, L. R. (2021). Mirostat: A Neural Text Decoding Algorithm that Directly Controls Perplexity. *ICLR*.

6. Leviathan, Y., Kalman, M., & Matias, Y. (2023). Fast Inference from Transformers via Speculative Decoding. *ICML*.

7. Chen, C., Borgeaud, S., Irving, G., Lespiau, J.-B., Sifre, L., & Jumper, J. (2023). Accelerating Large Language Model Decoding with Speculative Sampling. *arXiv*.

8. Vijayakumar, A. K., et al. (2018). Diverse Beam Search: Decoding Diverse Solutions from Neural Sequence Models. *AAAI*.
