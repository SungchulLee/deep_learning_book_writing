# 길이 정규화
길이 정규화는 생성 후보의 누적 로그 확률에 길이별 보정을 적용하는 방법이다. 원래 모델 확률과 다른 기준으로 순위를 정하므로 확률 모델 자체를 바꾸는 것과 구분해야 한다. EOS를 너무 일찍 내는 경향을 완화할 수 있지만 출력의 완전함이나 번역 품질을 보장하지는 않는다.

---

## 1. 짧은 시퀀스 쪽으로 치우치는 문제

### 빔 탐색이 짧은 시퀀스를 편드는 까닭

빔 탐색은 누적 로그 확률로 가설의 순위를 매긴다.

$$\text{score}(y_1, \ldots, y_T) = \sum_{t=1}^{T} \log P(y_t | y_{<t}, \mathbf{c})$$

각 조건부 확률이 0보다 크고 1 이하이면 로그 확률은 0 이하이다. 따라서 같은 접두사에 토큰을 덧붙인 누적 점수는 증가하지 않는다. 이는 서로 다른 두 완성 문장에서는 긴 쪽이 항상 불리하다는 정리가 아니다. 모든 토큰 확률이 0.3이면 길이 5와 10의 로그 점수는 각각 약 -6.02와 -12.04다. 일반적인 확률의 산술 평균만으로 로그 점수 합을 계산할 수는 없다.

모델의 EOS 확률과 길이 보정, 빔의 가지치기가 결합해 짧은 후보가 선호될 수 있다. 디코딩 점수만 바꾸는 것은 모델이 새로 학습하는 과정이 아니다. 완성 문장 비교에는 EOS의 로그 확률과 특수 토큰을 길이에 포함하는 규칙도 명시해야 한다.

### 보기 예

번역 후보 둘을 생각해 보자.

| 가설 | 길이 | 평균 토큰 확률 | 원점수 | 품질 |
|-----------|--------|----------------|-----------|---------|
| "The cat" | 2 | 0.5 | -1.39 | 나쁨 (불완전) |
| "The cat sat on the mat" | 6 | 0.4 | -5.50 | 좋음 (완전) |

위 표의 숫자와 길이 규약만 사용하면 짧은 후보의 점수가 높다. 하지만 두 후보의 점수 차이를 토큰 수만으로 설명할 수는 없고 각 확률과 종료 확률이 함께 작용한다. 이 표는 정해진 점수 예시이지 모든 번역에 대한 결론이 아니다.

---

## 2. 정규화 전략

### 단순 길이 정규화

가장 기본적인 방법은 누적 점수를 시퀀스의 길이로 나누는 것이다.

$$\text{score}_{norm} = \frac{1}{T} \sum_{t=1}^{T} \log P(y_t | y_{<t})$$

이는 토큰 확률의 **기하 평균**을 비교하는 것과 같으며, 길이와 무관한 토큰당 평균이다. 직관적이기는 하지만 단순 정규화는 지나치게 바로잡을 수 있다. 토큰당 확률이 같으면 토큰 2개짜리 시퀀스와 50개짜리 시퀀스를 똑같이 좋다고 보는데, 긴 쪽이 더 많은 정보를 담고 쓸모 있을 때도 그렇다.

```python
def simple_length_normalize(score: float, length: int) -> float:
    """
    점수를 순차열의 길이로 나누어 정규화한다.
    
    확률의 기하 평균을 견주는 것과 같다.
    간단하지만 길이를 지나치게 보정할 수 있다.
    """
    if length == 0:
        return float('-inf')
    return score / length
```

### 구글의 길이 패널티 (Wu 등, 2016)

Wu 등의 GNMT에서는 다음 길이 패널티를 사용했다. 이는 보정 함수의 한 선택이며 최적 지수는 검증 데이터에서 정해야 한다.

$$lp(Y) = \frac{(5 + |Y|)^\alpha}{(5 + 1)^\alpha}$$

$$\text{score}_{norm} = \frac{\text{score}}{lp(Y)}$$

상수 5는 아주 짧은 시퀀스를 매끄럽게 하여 $|Y|$이 작을 때 극단적인 정규화를 막는다. 지수 $\alpha$이 정규화의 세기를 다스린다.

```python
def google_length_penalty(length: int, alpha: float = 0.6) -> float:
    """
    Wu 등(2016)의 구글 길이 벌점.
    
    alpha < 1이면 벌점이 길이에 대해 선형보다 느리게 커져,
    날 점수와 토큰당 평균 사이의 균형을 잡아 준다.
    
    인수:
        length: 순차열의 길이
        alpha: 정규화 지수 (0이면 정규화 없음, 1이면 완전 정규화)
        
    반환값:
        길이 벌점의 제수
    """
    return ((5.0 + length) ** alpha) / ((5.0 + 1.0) ** alpha)

def normalized_score(log_prob_sum: float, length: int, alpha: float = 0.6) -> float:
    """길이로 정규화한 빔 점수를 계산한다."""
    return log_prob_sum / google_length_penalty(length, alpha)
```

### 알파 값에 따른 효과

이동 상수가 없는 $T^\alpha$ 방식은 $\alpha=1$에서 토큰당 평균이 된다. 그러나 위 GNMT 식은 $\alpha=1$일 때 분모가 $(5+T)/6$이므로 정확한 토큰당 평균과 같지 않다. 두 함수를 구분해야 한다.

$\alpha=0$이면 보정이 없고 양수 값을 키우면 긴 후보의 음수 점수가 상대적으로 덜 불리해질 수 있다. 적절한 범위는 길이 정의, 모델과 검색 조건에 따라 달라진다. 특정 수치가 대부분의 Seq2Seq 과제에 최적이라고 가정하지 않는다.

```python
import numpy as np
import matplotlib.pyplot as plt

def compare_alpha_values():
    """alpha 값에 따라 점수가 어떻게 달라지는지 그려 본다."""
    lengths = np.arange(1, 51)
    
    # 길이와 상관없이 토큰당 로그 확률이 같다
    per_token_logprob = -1.5
    raw_scores = per_token_logprob * lengths
    
    plt.figure(figsize=(10, 6))
    
    for alpha in [0.0, 0.3, 0.6, 0.7, 1.0]:
        penalties = np.array([
            google_length_penalty(l, alpha) for l in lengths
        ])
        normalized = raw_scores / penalties
        plt.plot(lengths, normalized, label=f'alpha={alpha}')
    
    plt.xlabel('Sequence Length')
    plt.ylabel('Normalized Score')
    plt.title('Length Normalization: Effect of Alpha')
    plt.legend()
    plt.grid(True)
    plt.show()
```

### 그 밖의 정규화 함수

아래 대안 함수들은 양수 길이와 적절한 파라메터를 가정한 수식 예시다. 음수 로그 점수를 큰 양수로 나누면 점수가 0에 가까워져 오히려 유리해진다. 따라서 adaptive 함수의 큰 분모를 그대로 사용하면 목표 길이에서 멀어진 후보를 벌주는 대신 유리하게 만들 수 있다. 의도한 페널티의 부호와 결합 방식을 먼저 확인한다.

```python
def exponential_length_penalty(length: int, beta: float = 0.1) -> float:
    """
    길이에 따라 더 가파르게 커지는 지수 벌점.
    
    아주 긴 출력이 나올 법하지 않은 과제에 더 센 보정을
    준다.
    """
    return np.exp(beta * length)

def logarithmic_length_penalty(length: int, gamma: float = 1.0) -> float:
    """
    부드럽게 정규화하는 로그 벌점.
    
    천천히 커져 보통 길이에는 거의 손대지 않지만 아주 긴 순차열의
    극단적인 치우침은 막는다.
    """
    return gamma * np.log(1 + length)

def adaptive_length_penalty(
    length: int, 
    target_length: int, 
    sigma: float = 5.0
) -> float:
    """
    기대되는 표적 길이를 중심으로 하는 가우스 모양의 벌점.
    
    기대 길이에서 양쪽으로 벗어난 순차열에 벌점을 주며, 표적의 길이를
    가늠할 수 있을 때 쓸모가 있다(길이 제약이 있는 요약 따위).
    
    """
    deviation = (length - target_length) ** 2
    return np.exp(deviation / (2 * sigma ** 2))
```

---

## 3. 빔 탐색과 결합하기

### 결합된 점수 함수

실제로 빔 탐색에서는 길이 정규화를 다른 점수 조정과 함께 쓴다.

```python
class BeamScorer:
    """
    빔 탐색 가설을 위한 결합 점수 함수.
    
    길이 정규화와 덮개 벌점, 그리고 선택적인 추가 항을
    아우른다.
    """
    
    def __init__(
        self,
        length_penalty_alpha: float = 0.6,
        coverage_penalty_beta: float = 0.0
    ):
        self.alpha = length_penalty_alpha
        self.beta = coverage_penalty_beta
    
    def score(
        self,
        log_prob_sum: float,
        length: int,
        attention_weights: list = None
    ) -> float:
        """
        최종 가설 점수를 계산한다.
        
        final_score = log_prob / lp(length) + beta * cp(attention)
        
        인수:
            log_prob_sum: 누적 로그 확률
            length: 순차열의 길이
            attention_weights: 어텐션 분포의 목록 (선택)
            
        반환값:
            결합하여 정규화한 점수
        """
        # 길이 정규화
        lp = ((5.0 + length) ** self.alpha) / ((5.0 + 1.0) ** self.alpha)
        normalized = log_prob_sum / lp
        
        # 덮개 벌점 (어텐션 기반 모델용)
        if self.beta > 0 and attention_weights:
            import torch
            coverage = torch.stack(attention_weights).sum(dim=0)
            cp = torch.sum(torch.log(torch.clamp(coverage, max=1.0)))
            normalized += self.beta * cp.item()
        
        return normalized
```

### 커버리지 패널티와의 상호작용

커버리지 패널티는 입력 위치별 누적 어텐션이 충분한지를 점수에 반영하는 휴리스틱이다. 길이 보정과 함께 쓸 수 있지만 입력 정보를 빠짐없이 번역했다는 증거는 아니다. PAD 위치는 제외해야 하며 아래 구현은 0 커버리지의 로그가 -inf가 될 수 있다.

$$\text{score}_{final} = \frac{\sum_{t} \log P(y_t | y_{<t}, \mathbf{x})}{lp(|\mathbf{y}|)} + \beta \cdot cp(\mathbf{y})$$

여기서 커버리지 패널티는 다음과 같다.

$$cp(\mathbf{y}) = \sum_{j=1}^{T_x} \log\left(\min\left(\sum_{t=1}^{T_y} \alpha_{t,j},\; 1\right)\right)$$

이 패널티는 모든 원본 자리에 적어도 한 번 주목했을 때(덮개 $\geq 1$) 0이고, 덜 주목한 자리가 있으면(덮개 $< 1$) 음수이다.

### 탐색 중에 점수 매기기와 탐색 뒤에 점수 매기기

검색 도중의 후보 순위와 최종 순위를 같은 함수로 정할지 명시한다. 같은 길이의 진행 중 후보끼리는 길이 보정만으로 순서가 바뀌지 않는다. 길이가 다른 완료 후보와 비교할 때 효과가 생긴다. 부분 후보 점수가 이후 증가할 수 있으므로 종료 조건도 점수에 맞춰 설계해야 하며, 검색 중 적용이 항상 더 좋은 것은 아니다.

```python
def beam_step_with_normalization(
    beams: list,
    beam_width: int,
    scorer: 'BeamScorer',
    normalize_during_search: bool = True
) -> list:
    """
    탐색 중 정규화를 선택적으로 적용하며 빔을 쳐 낸다.
    
    normalize_during_search=True이면 길이가 다른 가설을 견줄 때 길이
    정규화를 적용한다. 그러면 짧은 가설이 빔을 독차지하는 것을
    막는다.
    """
    if normalize_during_search:
        beams.sort(
            key=lambda h: scorer.score(h.score, len(h.tokens), h.attention_weights),
            reverse=True
        )
    else:
        # 날 점수만 (정규화는 탐색 뒤에 적용)
        beams.sort(key=lambda h: h.score, reverse=True)
    
    return beams[:beam_width]
```

---

## 4. 실험적 분석

### 길이 분포 비교

아래 함수는 외부의 beam_decoder와 test_loader를 요구하는 분석 조각이다. model 인수는 사용되지 않으며 실제 길이 분포 측정값은 제시하지 않는다. 길이뿐 아니라 공통 정답 기준의 품질도 평가하고 하이퍼파라미터 선택에는 검증셋을 사용해야 한다.

```python
def analyze_length_effects(
    model,
    test_loader,
    beam_decoder,
    alpha_values: list = [0.0, 0.3, 0.6, 0.7, 1.0]
):
    """
    alpha 값에 따른 출력 길이의 분포를 견준다.
    
    기준 길이에 견주어 출력 길이가 어떻게 달라지는지 살펴 알맞은
    정규화의 세기를 찾는 데 도움이 된다.
    """
    results = {}
    
    for alpha in alpha_values:
        beam_decoder.length_penalty = alpha
        lengths = []
        
        for src, trg in test_loader:
            tokens, score = beam_decoder.decode(src)
            lengths.append(len(tokens))
        
        results[alpha] = {
            'mean_length': np.mean(lengths),
            'std_length': np.std(lengths),
            'lengths': lengths
        }
    
    # 비교 그리기
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # 길이의 분포
    for alpha in alpha_values:
        axes[0].hist(results[alpha]['lengths'], alpha=0.5, 
                     label=f'alpha={alpha}', bins=20)
    axes[0].set_xlabel('Output Length')
    axes[0].set_ylabel('Count')
    axes[0].set_title('Output Length Distribution by Alpha')
    axes[0].legend()
    
    # 평균 길이
    means = [results[a]['mean_length'] for a in alpha_values]
    axes[1].bar(range(len(alpha_values)), means, 
                tick_label=[f'{a}' for a in alpha_values])
    axes[1].set_xlabel('Alpha Value')
    axes[1].set_ylabel('Mean Output Length')
    axes[1].set_title('Mean Output Length by Alpha')
    
    plt.tight_layout()
    plt.show()
    
    return results
```

---

## 5. 실무 지침

길이 정규화 지수는 검증 데이터에서 출력 길이·반복·품질을 함께 살펴 선택한다. 짧거나 긴 것이 그 자체로 좋다는 기준을 두지 않는다.

SOS·EOS·PAD를 길이에 포함할지, 미완료 후보를 최종 선택에 넣을지, 검색 중과 최종 점수에 어떤 보정을 쓸지를 명시한다. 빔 너비와 최소·최대 길이를 바꾸면 후보 집합도 달라진다. 코드의 지수·로그·가우시안 형태 함수를 결합할 때는 음수 원점수의 부호를 특히 확인한다.

## 연습문제

<div class="drillbox" markdown>

**연습문제 1.** <span class="diff med" title="중간"></span>
누적 로그 확률의 단조성과 서로 다른 길이의 완성 문장 비교를 구분하라.

</div>

??? success "풀이"
    같은 접두사에 토큰 하나를 더하면 0 이하의 로그 확률이 추가되므로 점수가 증가하지 않는다. 하지만 서로 다른 문장은 각 토큰과 EOS 확률도 달라 긴 쪽이 항상 낮은 것은 아니다. 길이 보정은 모델 확률과 다른 순위 기준을 만드는 선택이다.

---

<div class="drillbox" markdown>

**연습문제 2.** <span class="diff hard" title="어려움"></span>
T의 알파 제곱으로 원점수를 나누는 방식에서 알파의 역할을 설명하라.

</div>

??? success "풀이"
    $\alpha=0$은 원점수, $\alpha=1$은 토큰 로그 확률의 평균이다. 양수 지수는 길이에 따른 누적 페널티를 완화하려는 휴리스틱이며 확률 법칙에서 유일하게 유도되는 값은 아니다. GNMT의 이동 상수 5가 있는 식과 구분한다.

---

<div class="drillbox" markdown>

**연습문제 3.** <span class="diff hard" title="어려움"></span>
빔 탐색 디코더에 길이 정규화를 구현하라.

</div>

??? success "풀이"
    ```python
    def score_beam(log_probs, length, alpha=0.7):
        lp = ((5 + length) / 6) ** alpha  # 구글의 식
        return sum(log_probs) / lp
    ```

---

<div class="drillbox" markdown>

**연습문제 4.** <span class="diff med" title="중간"></span>
T, T의 알파 제곱, GNMT 길이 패널티를 비교하라.

</div>

??? success "풀이"
    T로 나누면 정확한 토큰당 평균이다. $T^\alpha$는 보정 세기를 조절한다. GNMT는 $(5+T)^\alpha/6^\alpha$를 사용해 길이 1에서 분모가 1이다. 알파가 1이어도 T와는 다르다. 어느 방식도 짧은 출력 바이어스나 품질 문제의 완전한 제거를 보장하지 않는다.

## 정리하며

길이 정규화는 검색 점수의 보정이다. 토큰당 평균과 GNMT 식을 구분하고 음수 점수에 나눗셈을 적용하는 방향을 확인한다. 종료·마스킹 규칙과 검증 품질을 함께 살펴야 한다.
