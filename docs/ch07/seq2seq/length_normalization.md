# 길이 정규화
길이 정규화는 빔 탐색의 체계적인 치우침을 바로잡는다. 날 로그 확률 점수는 본디 짧은 순차열을 편든다. 이를 바로잡지 않으면 복호기가 잘린 출력을 즐겨 만들어 생성의 품질을 떨어뜨린다. 이 절은 그 문제를 자세히 살피고, 표준적인 정규화 전략을 보이며, 어텐션 기반 모델에서 덮개 벌점과 어떻게 어우러지는지 알아본다.

## 짧은 순차열 쪽으로 치우치는 문제

### 빔 탐색이 짧은 순차열을 편드는 까닭

빔 탐색은 누적 로그 확률로 가설의 순위를 매긴다.

$$\text{score}(y_1, \ldots, y_T) = \sum_{t=1}^{T} \log P(y_t | y_{<t}, \mathbf{c})$$

조건부 확률이 모두 $0 < P(y_t | y_{<t}, \mathbf{c}) \leq 1$을 만족하므로 로그 확률은 모두 음이 아닌 값이 아니다. 곧 $\log P(\cdot) \leq 0$이다. 따라서 토큰을 하나 더할 때마다 전체 점수는 낮아지기만 한다. 토큰 확률의 평균이 0.3인 길이 5의 순차열은 $5 \times \log(0.3) \approx -6.0$점이고, 평균이 같은 길이 10의 순차열은 $-12.0$점이다.

그래서 짧은 출력이 체계적으로 유리해진다. 복호기는 쓸모 있는 내용을 계속 만드는 것보다 순차열 끝 토큰을 일찍 내는 편이 순위가 높다는 것을 배우고, 그 결과 잘리고 불완전한 출력을 낸다.

### 보기 예

번역 후보 둘을 생각해 보자.

| 가설 | 길이 | 평균 토큰 확률 | 날 점수 | 품질 |
|-----------|--------|----------------|-----------|---------|
| "The cat" | 2 | 0.5 | -1.39 | 나쁨 (불완전) |
| "The cat sat on the mat" | 6 | 0.4 | -5.50 | 좋음 (완전) |

정규화가 없으면 빔 탐색은 누가 봐도 못한 번역인 "The cat"을 고른다. 짧은 순차열이 이기는 까닭은 오로지 음수 항을 덜 모으기 때문이다.

## 정규화 전략

### 단순 길이 정규화

가장 기본적인 방법은 누적 점수를 순차열의 길이로 나누는 것이다.

$$\text{score}_{norm} = \frac{1}{T} \sum_{t=1}^{T} \log P(y_t | y_{<t})$$

이는 토큰 확률의 **기하 평균**을 견주는 것과 같으며, 길이와 무관한 토큰당 평균이다. 직관적이기는 하지만 단순 정규화는 지나치게 바로잡을 수 있다. 토큰당 확률이 같으면 토큰 2개짜리 순차열과 50개짜리 순차열을 똑같이 좋다고 보는데, 긴 쪽이 더 많은 정보를 담고 쓸모 있을 때도 그렇다.

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

### 구글의 길이 벌점 (Wu 등, 2016)

실전 시스템의 표준적인 방법은 매개변수가 있는 벌점으로 선형보다 약한 정규화를 하는 것이다.

$$lp(Y) = \frac{(5 + |Y|)^\alpha}{(5 + 1)^\alpha}$$

$$\text{score}_{norm} = \frac{\text{score}}{lp(Y)}$$

상수 5는 아주 짧은 순차열을 매끄럽게 하여 $|Y|$이 작을 때 극단적인 정규화를 막는다. 지수 $\alpha$이 정규화의 세기를 다스린다.

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

### $\alpha$ 값에 따른 효과

매개변수 $\alpha$은 정규화 없음($\alpha = 0$, 날 점수)과 완전한 토큰당 정규화($\alpha = 1$, 기하 평균과 같다) 사이를 잇는다.

| $\alpha$ 값 | 거동 | 쓰임새 |
|---------|----------|----------|
| 0.0 | 정규화 없음 (날 점수) | 짧게 하고 싶을 때 |
| 0.5 | 알맞게 매끄럽게 | 일반적인 번역 |
| 0.6~0.7 | 표준적인 선택 | 대부분의 seq2seq 과제 |
| 1.0 | 완전한 정규화 (토큰당 평균) | 길이가 중요하지 않을 때 |

최적의 $\alpha$은 과제에 달려 있다. 기계 번역에서는 $\alpha \in [0.6, 0.7]$이 자리 잡은 표준이며, 잘림을 막을 만큼 바로잡으면서도 간결함을 살짝 선호하게 해 준다.

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

구글의 벌점 말고도 여러 대안이 연구되었다.

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

## 빔 탐색과 결합하기

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

### 덮개 벌점과의 상호작용

덮개 벌점과 길이 정규화는 서로 보완하는 문제를 다룬다. 길이 정규화는 복호기가 너무 일찍 멈추는 것을 막고, 덮개 벌점은 입력의 일부를 무시하는 것을 막는다. 둘이 함께 완전하고 잘 짜인 출력을 이끌어 낸다.

$$\text{score}_{final} = \frac{\sum_{t} \log P(y_t | y_{<t}, \mathbf{x})}{lp(|\mathbf{y}|)} + \beta \cdot cp(\mathbf{y})$$

여기서 덮개 벌점은 다음과 같다.

$$cp(\mathbf{y}) = \sum_{j=1}^{T_x} \log\left(\min\left(\sum_{t=1}^{T_y} \alpha_{t,j},\; 1\right)\right)$$

이 벌점은 모든 원본 자리에 적어도 한 번 주목했을 때(덮개 $\geq 1$) 0이고, 덜 주목한 자리가 있으면(덮개 $< 1$) 음수이다.

### 탐색 중에 점수 매기기와 탐색 뒤에 점수 매기기

중요한 구현 세부가 있다. 길이 정규화는 빔을 쳐 낼 때 **함께** 적용하여 어떤 가설이 살아남을지에 영향을 줄 수도 있고, 탐색이 **끝난 뒤** 최종 가설을 고를 때만 쓸 수도 있다. 대체로 탐색 중에 적용하는 편이 낫다. 가능성 있는 긴 가설이 일찍 걸러지는 것을 막기 때문이다.

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

## 실험적 분석

### 길이 분포 견주기

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

## 실무 지침

### $\alpha$ 고르기

최적의 $\alpha$은 과제에 달려 있으며 검증 집합으로 맞추어야 한다. 일반적인 권장값은 다음과 같다.

| 과제 | 권장 $\alpha$ | 근거 |
|------|---------------------|-----------|
| 기계 번역 | 0.6~0.7 | 완전함과 간결함 사이의 균형 |
| 요약 | 0.8~1.0 | 긴 요약이 정보를 더 담는 편이다 |
| 대화 응답 | 0.5~0.6 | 간결함을 조금 선호 |
| 코드 생성 | 0.6 | 코드 블록이 일찍 끝나는 것을 피한다 |

### 흔히 빠지는 함정

**$\alpha$이 너무 낮으면**($< 0.4$) 출력이 잘려 중요한 정보를 놓친다. 모델이 순차열 끝 토큰을 일찍 낸다.

**$\alpha$이 너무 높으면**($> 1.0$) 출력이 지나치게 길어져 되풀이나 군더더기가 들어간다. 정규화가 과하게 보정하여 긴 순차열이 지나치게 유리해진다.

**일관되지 않은 정규화**: 빔을 쳐 낼 때는 쓰지 않고 최종 선택에만 길이 정규화를 적용하면 짧은 가설이 빔을 독차지하여, 가능성 있는 긴 후보가 완성되기도 전에 걸러진다.

### 다른 매개변수와의 상호작용

길이 정규화는 다른 여러 초매개변수와 얽힌다.

| 매개변수의 상호작용 | 효과 |
|----------------------|--------|
| 넓은 빔 + 높은 $\alpha$ | 길이를 더 다양하게 살핀다 |
| 덮개 벌점 + 길이 벌점 | 서로 보완한다. 길이는 잘림을, 덮개는 건너뜀을 막는다 |
| 되풀이 벌점 + $\alpha$ | 되풀이를 다스리지 않고 $\alpha$만 높이면 길고 되풀이되는 출력이 나올 수 있다 |
| 최소 길이 + $\alpha$ | 최소 길이는 단단한 바닥을, $\alpha$은 부드러운 유도를 준다 |

## 요약

길이 정규화는 실전 빔 탐색에 꼭 필요하며, 음의 로그 확률이 쌓이면서 생기는 짧은 순차열 쪽 치우침을 바로잡는다. $\alpha \approx 0.6\text{~}0.7$인 구글의 길이 벌점이 자리 잡은 표준으로, 완전함과 간결함의 균형을 잡는 선형 이하의 정규화를 준다. 가능성 있는 긴 가설이 일찍 걸러지지 않도록 최종 선택 때만이 아니라 빔을 쳐 낼 때도 정규화를 적용해야 한다. 어텐션 기반 모델에서 덮개 벌점과 함께 쓰면 길이 정규화는 (원본 정보를 빠짐없이 다룬다는 뜻에서) 완전하고 (잘리지도 군더더기로 채워지지도 않았다는 뜻에서) 알맞은 길이의 출력을 얻는 데 도움이 된다.

## 연습문제

**연습문제 1.**
길이 정규화가 없는 빔 탐색이 짧은 순차열 쪽으로 치우치는 까닭을 설명하라.

??? success "연습문제 1 풀이"
    로그 확률이 음수이므로 긴 순차열일수록 음수 항이 더 많이 쌓인다. $\log P(y_1, \ldots, y_T) = \sum_t \log P(y_t|y_{<t})$이다. 짧은 순차열은 음수 항이 적어 점수가 (덜 음수여서) 높다. 이 치우침 때문에 모델이 EOS 토큰을 일찍 내는 쪽을 좋아하게 된다.

---

**연습문제 2.**
길이로 정규화한 점수 함수 $\frac{1}{T^\alpha}\sum_t \log P(y_t)$을 유도하고 $\alpha$의 구실을 설명하라.

??? success "연습문제 2 풀이"
    $T^\alpha$으로 나누면 토큰당 점수에 순차열의 길이만큼 벌점이 매겨진다. $\alpha = 0$이면 정규화가 없어 짧은 쪽으로 치우친다. $\alpha = 1$이면 토큰당 평균이 되어 길고 장황한 순차열이 유리할 수 있다. $\alpha \in [0.6, 0.8]$이 짧은 것과 긴 것의 균형을 잡는 흔한 값이다(Wu 등, 2016).

---

**연습문제 3.**
빔 탐색 복호기에 길이 정규화를 구현하라.

??? success "연습문제 3 풀이"
    ```python
    def score_beam(log_probs, length, alpha=0.7):
        lp = ((5 + length) / 6) ** alpha  # 구글의 식
        return sum(log_probs) / lp
    ```

---

**연습문제 4.**
여러 길이 정규화 전략을 견주어라. $T$으로 나누기, $T^\alpha$으로 나누기, 구글의 식.

??? success "연습문제 4 풀이"
    $T$으로 나누는 것은 짧은 순차열에 너무 가혹하다. $\alpha < 1$인 $T^\alpha$은 유연하고 표준적인 선택이다. 구글의 식 $\frac{(5+T)^\alpha}{6^\alpha}$은 짧은 순차열에 벌점이 과하지 않도록 5를 더하며 실험적으로 잘 통한다. 모두 짧은 출력 쪽으로의 체계적인 치우침을 없앤다는 같은 목표를 이룬다.
