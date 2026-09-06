# 음성 표본 추출

음성 표본 추출은 어휘가 클 때도 낱말 임베딩 모델을 학습시킬 수 있게 해 주는 최적화 기법이다. 학습 단계마다 어휘 전체에 소프트맥스를 계산하면 낱말 수십만 개의 점수를 매겨야 하는데, 음성 표본 추출은 그 대신 문제를 여러 이진 분류 과제로 바꾼다. 모델은 참된 문맥 낱말을 무작위로 뽑은 몇 개의 "음성" 낱말과 가려내는 법을 배우며, 임베딩의 품질은 지키면서 계산 비용을 크게 줄인다.

## 코드

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

## 논의

낱말 예측의 표준 소프트맥스 손실은 $p(w_O | w_I) = \frac{\exp(v'_{w_O} \cdot v_{w_I})}{\sum_{w=1}^{V} \exp(v'_w \cdot v_{w_I})}$을 계산해야 하며 여기서 $V$은 어휘 크기이다. 분모가 모든 낱말에 걸쳐 합해지므로 기울기 갱신마다 $O(V)$이 든다. 낱말이 10만 개를 넘는 어휘에서는 감당할 수 없이 비싸다.

음성 표본 추출은 이를 이진 로지스틱 회귀 $k$개로 바꾼다. 양성 (가운데, 문맥) 쌍 $(w_I, w_O)$에 대해 목표는 $\log \sigma(v'_{w_O} \cdot v_{w_I}) + \sum_{i=1}^{k} \mathbb{E}_{w_i \sim P_n} [\log \sigma(-v'_{w_i} \cdot v_{w_I})]$을 최대로 하는 것이며, $\sigma$은 시그모이드 함수이고 $P_n$은 잡음 분포(보통 유니그램 분포의 3/4 거듭제곱)이다. 이로써 갱신마다의 비용이 $O(V)$에서 $O(k)$으로 줄고, $k$은 대체로 5~20이다.

잡음 분포 $P_n(w) \propto f(w)^{3/4}$($f(w)$은 낱말의 빈도)이 매우 중요하다. 3/4 지수가 분포를 매끄럽게 하여 드문 낱말이 날 빈도가 시사하는 것보다 높은 확률로 뽑히게 한다. 그러면 모델이 가장 흔한 낱말에만 매달리지 않는다. Mikolov 등은 이 지수가 유니그램 분포 자체나 균등 분포보다 훨씬 잘 통함을 실험으로 알아냈다.

## 연습문제

**연습문제 1.**
어휘 크기가 $V = 50{,}000$일 때 전체 소프트맥스와 음성 표본 $k = 10$개를 쓰는 음성 표본 추출의 학습 단계당 연산 수를 견주어라. 몇 배 빨라지는지 나타내어라.

??? success "연습문제 1 풀이"
    전체 소프트맥스는 낱말 $V = 50{,}000$개 모두에 대해 내적과 지수를 계산해야 하므로 단계마다 $O(50{,}000)$의 연산이 든다. 음성 표본 추출은 양성 1개와 음성 10개, 곧 낱말 11개의 점수만 계산하므로 단계마다 $O(11)$이다. 빨라지는 배수는 약 $50{,}000 / 11 \approx 4{,}545\times$이다. 실제로는 음성을 뽑는 부담 때문에 조금 덜하지만, 큰 어휘에서는 여전히 몇 자릿수 빠르다.

---

**연습문제 2.**
잡음 분포가 날 빈도가 아니라 유니그램 빈도의 3/4 거듭제곱을 쓰는 까닭을 설명하라. 균등 분포, 유니그램, 3/4 거듭제곱일 때 각각 어떻게 되는지 살펴보라.

??? success "연습문제 2 풀이"
    날 유니그램 분포를 쓰면 가장 잦은 낱말("the", "a", "is" 따위)이 음성 표본을 독차지하고 드문 낱말은 음성으로 거의 나오지 않는다. 그러면 모델이 드문 낱말끼리 가려내는 법을 결코 배우지 못한다. 균등 분포를 쓰면 아주 드문 낱말이 실제로 나타나는 정도에 견주어 음성으로 너무 자주 나와, 있을 법하지 않은 음성에 용량을 낭비한다. 3/4 거듭제곱은 그 사이의 타협이다. "the"의 빈도가 0.05이고 "quantum"의 빈도가 0.00001이면 그 비가 $5{,}000$이 아니라 $0.05^{0.75} / 0.00001^{0.75} \approx 1{,}778$이 된다. 곧 "quantum"이 날 빈도가 예측하는 것보다 약 3배 자주 음성으로 나오므로, 가장 흔한 낱말이 학습을 뒤덮지 않으면서도 모델이 그 임베딩을 배울 만큼의 신호를 얻는다.

---

**연습문제 3.**
음성 표본 추출 손실 함수를 PyTorch로 구현하라. 가운데 임베딩과 양성 문맥 임베딩, 음성 문맥 임베딩이 주어졌을 때 이진 교차 엔트로피 손실을 계산하고, 그것이 임베딩을 갱신하는 기울기를 내놓는지 확인하라.

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
