# 샘플링 방법

다음 토큰의 로짓에서 하나의 토큰을 고르는 탐욕 디코딩, top-k와 top-p(뉴클리어스) 샘플링을 비교한다. 후보 제한과 온도는 출력 분포를 바꾸지만 사실성·문법·다양성의 개선을 보장하지 않는다.

## 1. 코드

```python
import torch
import torch.nn.functional as F


def greedy_sampling(logits):
    return torch.argmax(logits, dim=-1)


def top_k_sampling(logits, k=50, temperature=1.0):
    values, indices = torch.topk(logits / temperature, k)
    probs = F.softmax(values, dim=-1)
    next_token = torch.multinomial(probs, 1)
    return indices.gather(-1, next_token)


def nucleus_sampling(logits, p=0.9, temperature=1.0):
    sorted_logits, sorted_indices = torch.sort(
        logits / temperature, descending=True
    )
    cumulative_probs = torch.cumsum(
        F.softmax(sorted_logits, dim=-1), dim=-1
    )

    sorted_indices_to_remove = cumulative_probs > p
    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
    sorted_indices_to_remove[..., 0] = 0

    indices_to_remove = sorted_indices_to_remove.scatter(
        1, sorted_indices, sorted_indices_to_remove
    )
    logits[indices_to_remove] = float('-inf')

    probs = F.softmax(logits, dim=-1)
    return torch.multinomial(probs, 1)


if __name__ == "__main__":
    pass
```

## 2. 논의

함수들은 주로 $(B,V)$ 로짓을 가정한다. greedy_sampling은 각 행의 argmax를 선택해 (B,)를 반환한다. 다른 두 함수는 (B,1)을 반환하므로 생성 루프에 연결할 때 축을 맞춘다. 고정 로짓에 대한 argmax는 결정적이지만 모델 드롭아웃·입력 변화나 연산의 비결정성까지 제거하는 것은 아니다.

top-k는 양의 온도로 나눈 로짓에서 k개를 골라 소프트맥스로 재정규화한다. 1≤k≤V와 temperature>0이 필요하다. k=1은 최댓값 한 개만 남긴다. top-p는 확률 내림차순 누적합을 이용해 후보 수를 바꾸며 경계 토큰을 포함한다. 이 코드의 엄격한 > 비교는 누적합이 정확히 p인 경계에서 최소 집합보다 토큰 하나를 더 남길 수 있다.

**nucleus_sampling의 제한:** 후보 집합은 logits/temperature로 계산하지만 마지막 표본은 온도를 적용하지 않은 원래 logits에서 뽑는다. 따라서 온도가 후보 선택에는 반영되어도 남은 후보의 확률 비율에는 반영되지 않는다. 또한 전달받은 logits의 제외 위치를 제자리로 -inf로 바꾸므로 원본을 재사용하려면 복사 여부를 확인해야 한다. scatter(1,…)을 사용해 임의 차원 입력을 지원하는 함수도 아니다.

연습문제의 혼합 함수는 top-k 이후의 분포를 다시 정규화해 top-p를 적용한다. 이때 p는 원래 전체 어휘가 아니라 top-k 안의 분포에 대한 기준이다. 필터를 결합했다고 품질이 자동으로 높아지거나 모델의 오류가 사라지지는 않는다.

## 연습문제

<div class="drillbox" markdown>

**연습문제 1.** <span class="diff med" title="중간"></span>
아래 로짓 구성에서 탐욕·확률 샘플링 결과를 실행 없이 단정할 수 있는 범위를 설명하라.

</div>

??? success "풀이"
    42번 로짓을 10으로 덮어쓰지만 나머지는 상한이 없는 정규분포 표본이므로 42가 반드시 최댓값이라는 보장은 없다. 42가 유일한 최댓값이면 탐욕은 이를 고른다. top-p에서 이 토큰의 확률이 p보다 크면 후보 하나만 남아 확정적으로 선택하지만, 그 조건은 실제 로짓 전체로 확인해야 한다.

    ```python
    logits = torch.randn(1, 100)
    logits[0, 42] = 10.0  # 뚜렷한 꼭대기

    greedy = greedy_sampling(logits.clone())
    print(f"Greedy: {greedy.item()}")  # 언제나 42

    top_k_results = [top_k_sampling(logits.clone(), k=10).item() for _ in range(20)]
    print(f"Top-k: {top_k_results}")  # 대체로 42

    nucleus_results = [
        nucleus_sampling(logits.clone(), p=0.5).item() for _ in range(20)
    ]
    print(f"Nucleus: {nucleus_results}")  # 대체로 42
    ```

---

<div class="drillbox" markdown>

**연습문제 2.** <span class="diff med" title="중간"></span>
top-p 마스크를 오른쪽으로 한 칸 미는 이유와 이 구현의 경계 조건을 설명하라.

</div>

??? success "풀이"
    누적합이 p를 처음 넘는 토큰까지 남겨 충분한 확률 질량을 확보하려는 것이다. 가장 높은 토큰도 최소 하나 남긴다. 엄격한 > 비교 때문에 누적합이 정확히 p인 경우 다음 토큰까지 포함할 수 있다. 0<p≤1 범위를 확인하고 모든 로짓이 -inf인 입력 등 유효하지 않은 분포도 별도 처리해야 한다.

---

<div class="drillbox" markdown>

**연습문제 3.** <span class="diff hard" title="어려움"></span>
아래 top-k·top-p 결합 함수에서 p의 기준 분포와 온도 처리를 설명하라.

</div>

??? success "풀이"
    ```python
    def top_k_nucleus_sampling(logits, k=50, p=0.9, temperature=1.0):
        scaled = logits / temperature

        # 1단계: 상위 k 거르기
        top_k_values, top_k_indices = torch.topk(scaled, k)
        filtered = torch.full_like(scaled, float('-inf'))
        filtered.scatter_(1, top_k_indices, top_k_values)

        # 2단계: 상위 k 집합에 커널 거르기
        sorted_logits, sorted_indices = torch.sort(filtered, descending=True)
        cumulative = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
        remove = cumulative > p
        remove[..., 1:] = remove[..., :-1].clone()
        remove[..., 0] = 0
        mask = remove.scatter(1, sorted_indices, remove)
        filtered[mask] = float('-inf')

        probs = F.softmax(filtered, dim=-1)
        return torch.multinomial(probs, 1)
    ```
    먼저 온도를 적용한 top-k 로짓 밖을 -inf로 두고 그 분포를 재정규화한 뒤 top-p를 적용한다. 최종 확률에도 스케일된 로짓을 사용하므로 앞의 nucleus_sampling과 달리 온도가 표본 비율에 반영된다. 후보 수는 최대 k개지만 실제 품질의 개선이나 잘못된 토큰 제거가 보장되지는 않는다.

## 정리하며

탐욕 선택과 확률 샘플링의 반환 축, 후보 집합과 최종 재정규화를 구분한다. 특히 기본 nucleus_sampling은 온도를 후보 선택에만 사용하고 입력을 변경한다. 온도·k·p의 유효 범위와 모델 출력의 품질은 별도로 확인한다.
