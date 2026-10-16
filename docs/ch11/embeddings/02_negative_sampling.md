# 네거티브 샘플링

네거티브 샘플링은 관측된 중심·문맥 쌍과 잡음 분포에서 뽑은 쌍을 구분하도록 임베딩을 학습하는 방법이다. 어휘 전체에 대한 점수 대신 소수의 선택된 점수를 계산한다. 본문 코드는 안내 메시지만 출력하며, 로스 함수의 예시는 연습문제 3에 있다.

## 1. 코드

```python
"""음성 표본 추출."""

# ========================================================================
# 메인
# ========================================================================
# 효율적인 학습을 위한 음성 표본 추출
# 음성 예제를 뽑아 학습을 빠르게 한다
print("Negative Sampling - Advanced optimization technique for large vocabularies")


if __name__ == "__main__":
    pass
```

**출력:**

```
Negative Sampling - Advanced optimization technique for large vocabularies
```

## 2. 논의

전체 소프트맥스는 중심 벡터와 어휘 전체의 출력 벡터 사이의 내적을 계산한다. 임베딩 차원이 $d$이고 어휘 크기가 $V$이면 쌍 하나의 점수 계산 비용은 대략 $O(Vd)$다. 큰 어휘에서는 이 연산이 학습 비용의 상당 부분을 차지할 수 있다.

네거티브 샘플링은 관측 쌍 하나에 대해 잡음 단어 $w_i\sim P_n$를 $k$개 뽑고, $\log\sigma(u_{w_O}^{\top}v_{w_I})+\sum_{i=1}^{k}\log\sigma(-u_{w_i}^{\top}v_{w_I})$를 최대화한다. 양성 항 1개와 음성 항 $k$개이며 점수 계산은 $O((k+1)d)$다. 이는 전체 소프트맥스의 정규화된 조건부 확률을 그대로 계산하는 방식이 아니다.

자주 사용하는 잡음 분포는 $P_n(w)=f(w)^{3/4}/\sum_j f(j)^{3/4}$다. 유니그램 빈도에 비해 흔한 단어와 드문 단어의 확률 비율을 줄이되, 균등 분포보다는 빈도를 반영한다. 지수와 음성 샘플 수는 학습 설정이며 모든 말뭉치에서 같은 값이 최적인 것은 아니다. 자세한 목적 함수는 [Word2Vec 논문](https://arxiv.org/abs/1310.4546)을 참고할 수 있다.

샘플 수를 줄이면 계산량을 줄일 수 있지만 임베딩 품질을 그대로 보장하지는 않는다. 실제 학습 시간에는 배치 연산 효율, 샘플 생성과 메모리 접근도 영향을 준다. 음성 샘플은 잡음 분포에서 뽑은 예제라는 의미이며, 언어적으로 반드시 잘못된 문맥을 뜻하지 않는다.

## 연습문제

<div class="drillbox" markdown>

**연습문제 1.** <span class="diff med" title="중간"></span>
어휘 크기 50,000, 음성 샘플 수 10일 때 쌍 하나당 계산하는 점수 수를 비교하라. 이 비율을 실제 속도 향상으로 해석할 수 있는가?

</div>

??? success "연습문제 1 풀이"
    전체 소프트맥스는 50,000개, 네거티브 샘플링은 양성 1개와 음성 10개로 11개 점수를 계산한다. 점수 개수의 비는 약 $4545$다. 이는 실행 시간 비율이 아니다. 행렬 연산 효율, 샘플링, 메모리와 학습 목표가 다르므로 같은 품질에 도달하는 시간은 별도로 비교해야 한다.

---

<div class="drillbox" markdown>

**연습문제 2.** <span class="diff med" title="중간"></span>
유니그램 분포에 3/4 거듭제곱을 적용하면 두 단어의 샘플링 확률 비율은 어떻게 바뀌는가? 빈도가 0.05와 0.00001인 경우를 계산하라.

</div>

??? success "연습문제 2 풀이"
    정규화 상수는 비율에서 상쇄되므로 원래 비율 $5000$이 $5000^{3/4}\approx594.6$으로 줄어든다. 균등 분포에서는 비율이 1이다. 드문 단어가 흔한 단어에 비해 상대적으로 더 자주 뽑히지만, 개별 단어의 절대 확률 증가 배수는 전체 어휘의 정규화 상수도 알아야 계산할 수 있다.

---

<div class="drillbox" markdown>

**연습문제 3.** <span class="diff hard" title="어려움"></span>
네거티브 샘플링 로스 함수를 PyTorch로 구현하라. 가운데 임베딩과 양성 문맥 임베딩, 음성 문맥 임베딩이 주어졌을 때 이진 교차 엔트로피 로스를 계산하고, 그것이 임베딩을 갱신하는 그래디언트를 내놓는지 확인하라.

</div>

??? success "연습문제 3 풀이"
    ```python
    import torch
    import torch.nn as nn

    def negative_sampling_loss(center, positive, negatives):
        """
        인수:
            center: (배치 크기, emb_dim)
            positive: (배치 크기, emb_dim)
            negatives: (배치 크기, num_neg, emb_dim)
        반환값:
            loss: 스칼라
        """
        pos_score = torch.sum(center * positive, dim=1)      # (배치,)
        neg_scores = torch.bmm(negatives, center.unsqueeze(2)).squeeze(2)  # (배치, num_neg)
        
        pos_loss = -torch.log(torch.sigmoid(pos_score) + 1e-10)
        neg_loss = -torch.log(torch.sigmoid(-neg_scores) + 1e-10).sum(dim=1)
        
        return (pos_loss + neg_loss).mean()

    # 시험
    emb = nn.Embedding(100, 32)
    center_idx = torch.tensor([0, 1, 2])
    pos_idx = torch.tensor([5, 6, 7])
    neg_idx = torch.randint(0, 100, (3, 5))

    c = emb(center_idx)
    p = emb(pos_idx)
    n = emb(neg_idx)

    loss = negative_sampling_loss(c, p, n)
    loss.backward()
    print(f"Loss: {loss.item():.4f}")
    print(f"Gradient exists: {emb.weight.grad is not None}")  # True
    ```

    풀이에서는 하나의 임베딩 레이어를 중심과 문맥에 공유한다. 중심·문맥 행렬을 따로 사용하는 표준 Skip-gram 설정과 다르다. `randint`는 균등 샘플링이며 빈도의 3/4 거듭제곱 분포를 구현하지 않는다. 양성 단어와 같은 음성 ID가 나올 수도 있다. 또한 `sigmoid` 뒤에 로그와 작은 상수를 적용하는 계산은 극단적인 점수에서 로스와 그래디언트를 왜곡할 수 있으므로 안정적인 `logsigmoid`나 로짓 기반 이진 로스로 표현하는 편이 낫다. 그래디언트 텐서가 존재한다는 검사만으로 구현의 정확성이나 학습 품질이 검증되는 것은 아니다.

## 정리하며

네거티브 샘플링은 관측 쌍과 잡음 쌍을 구분하는 목적 함수다. 계산할 점수 수를 줄이지만 실제 속도와 임베딩 품질은 별도로 확인해야 한다. 잡음 분포, 중심·문맥 웨이트의 공유 여부와 로스의 수치 안정성을 함께 살펴본다.
