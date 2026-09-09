# 표집 방법

언어 모형으로 글을 지으려면 예측한 확률 분포에서 다음 토큰을 어떻게 고를지 정해야 한다. 표집 방법에 따라 글의 다양함과 앞뒤 맞음과 창의가 달라진다. 이 모듈은 근본 되는 방법 셋, 곧 탐욕 표집, 상위 $k$ 표집, 핵(상위 $p$) 표집을 구현한다.

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

탐욕 표집은 언제나 확률이 가장 높은 토큰을 고른다. 단계마다 가장 그럴듯한 토큰 하나를 내지만, 모형이 확률 높은 고리에 갇혀 되풀이되고 밋밋한 글이 되기 일쑤이다. 탐욕 디코딩은 결정론적이어서 같은 입력이 언제나 같은 출력을 낸다.

상위 $k$ 표집은 후보를 가장 그럴듯한 $k$개의 토큰으로 제한하고 그 줄어든 분포에서 뽑는다. 온도 매개변수는 고르기 전 분포의 뾰족함을 다스린다. 1.0보다 작으면 확률이 위쪽 토큰으로 뾰족해지고 1.0보다 크면 분포가 평평해진다. $k$을 고정하면 문제가 될 수 있는데, 어떤 분포는 다른 것보다 본디 더 뾰족하기 때문이다. $k = 50$으로 두면 자신 있는 예측에서는 쓰레기 토큰이 끼고 애매한 맥락에서는 너무 옥죈다.

핵(상위 $p$) 표집은 누적 확률에 따라 후보 집합을 그때그때 맞추어 이 한계를 푼다. 개수를 고정하는 대신 누적 확률이 문턱값 $p$을 넘는 가장 작은 토큰 집합을 담는다. 그러면 모형이 자신 있을 때는 후보가 적고 헷갈릴 때는 많아진다. 구현은 토큰을 확률로 정렬하고 누적합을 셈한 뒤 문턱을 넘는 토큰을 가리고 표집한다. 핵 표집은 대체로 상위 $k$ 표집보다 자연스럽고 앞뒤 맞는 글을 낸다.

## 연습문제

**연습문제 1.**
색인 42에 뚜렷한 꼭대기가 있는 꼴 `(1, 100)`의 로짓 텐서를 만들고 세 표집 방법을 모두 적용하라. 탐욕은 언제나 42를 돌려주고, 상위 $k$은 대체로 42를 돌려주며, $p = 0.5$일 때 핵 표집이 높은 확률로 42를 돌려주는지 확인하라.

??? success "연습문제 1 풀이"
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

**연습문제 2.**
핵 표집이 `sorted_indices_to_remove` 가림을 오른쪽으로 한 자리 미는 까닭(`sorted_indices_to_remove[..., 1:] = ...` 줄)을 설명하라. 이 밀기가 없으면 어떻게 되는가?

??? success "연습문제 2 풀이"
    자리 $i$의 누적 확률에는 토큰 $i$ 자신의 확률이 들어 있다. 오른쪽으로 밀지 않으면 누적 확률이 $p$을 넘게 만든 그 토큰이 후보에서 빠진다. 밀어 주면 이 경계 토큰이 남아 남긴 토큰들의 실제 누적 확률이 적어도 $p$이 된다. 밀지 않으면 후보의 누적 확률이 $p$보다 한참 작을 수 있어 문턱값 매개변수가 뜻한 대로 움직이지 않는다.

---

**연습문제 3.**
먼저 상위 $k$개 토큰으로 좁힌 뒤 그 안에서 핵 표집을 하는 "상위 $k$ + 핵"이라는 섞은 표집 방법을 구현하라. 각 방법을 따로 쓸 때와 견주어라.

??? success "연습문제 3 풀이"
    ```python
    def top_k_nucleus_sampling(logits, k=50, p=0.9, temperature=1.0):
        scaled = logits / temperature

        # 1단계: 상위 k 거르기
        top_k_values, top_k_indices = torch.topk(scaled, k)
        filtered = torch.full_like(scaled, float('-inf'))
        filtered.scatter_(1, top_k_indices, top_k_values)

        # 2단계: 상위 k 집합에 핵 거르기
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
    이 섞은 방식은 ($k$으로) 후보의 최대 개수를 묶으면서 ($p$으로) 예측의 자신감에도 맞춘다. 두 방법의 가장 나쁜 경우를 피한다. 상위 $k$만 쓰면 아주 그럴듯하지 않은 토큰이 낄 수 있고, 핵만 쓰면 평평한 분포에서 후보가 엄청나게 커질 수 있다.

## 정리하며

**다룬 것** — 표집 방법

탐욕 표집은 언제나 확률이 가장 높은 토큰을 고른다.

앞의 연습문제 3개로 직접 확인할 수 있다.
