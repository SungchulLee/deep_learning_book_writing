# 균형 잡힌 데이터로더

클래스 불균형은 한 클래스가 다른 클래스보다 훨씬 많은, 기계학습에서 흔한 어려움이다. 균형 잡힌 DataLoader는 `WeightedRandomSampler`를 써서 학습 배치마다 클래스가 대체로 고르게 섞이게 하여 모델이 다수 클래스로 치우치는 것을 막는다. 이 기법은 사기 탐지, 의료 진단을 비롯해 레이블 분포가 치우친 모든 분야에서 꼭 필요하다.

## 코드

```python
"""
균형 잡힌 DataLoader — 고양이와 개
===================================
심하게 불균형한 데이터셋(고양이 100마리와 개 1,000마리)의 균형을 맞추어
학습 배치마다 두 클래스가 대체로 고르게 섞이도록
``WeightedRandomSampler``를 쓰는 법을 보인다.

사용법
-----
    python balanced_dataloader.py
"""

import torch
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler

# ========================================================================
# 메인
# ========================================================================


def create_balanced_data_loader():
    """
    클래스 불균형을 다루려고 WeightedRandomSampler를 쓰는 DataLoader를 만든다.
    고양이 100마리와 개 1,000마리로 이루어진 데이터셋의 특징을
    합성으로 만든다.

    반환값
    -------
    DataLoader
        가중 무작위 표집으로 설정된 데이터 로더.
    """
    # 합성 특징
    cat_features = torch.randn(100, 10)       # 고양이 100마리
    dog_features = torch.randn(1000, 10)      # 개 1,000마리

    features = torch.cat((cat_features, dog_features), dim=0)   # (1100, 10)
    labels   = torch.cat((torch.zeros(100), torch.ones(1000)))  # (1100,)

    dataset = TensorDataset(features, labels)

    # 빈도의 역수로 가중
    class_counts   = torch.tensor(
        [(labels == 0).sum(), (labels == 1).sum()], dtype=torch.float32
    )
    class_weights  = 1.0 / class_counts                         # (1/100, 1/1000)
    sample_weights = torch.tensor(
        [class_weights[int(label)] for label in labels]          # (1100,)
    )

    sampler = WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(sample_weights),
        replacement=True,
    )

    return DataLoader(dataset, batch_size=100, sampler=sampler)


def main():
    loader = create_balanced_data_loader()

    for batch_idx, (batch_features, batch_labels) in enumerate(loader):
        n_cats = (batch_labels == 0).sum().item()
        n_dogs = (batch_labels == 1).sum().item()
        print(f"Batch {batch_idx + 1}: Cats={n_cats}, Dogs={n_dogs}")


if __name__ == "__main__":
    main()```

## 논의

균형 표집의 핵심 착상은 빈도의 역수로 가중치를 주는 것이다. 각 표본은 그 클래스의 빈도에 반비례하는 가중치를 받으므로 표집기가 소수 클래스의 표본을 더 자주 뽑는다. 이 예에서 고양이(표본 100개)는 표본마다 가중치 $1/100 = 0.01$을, 개(표본 1000개)는 $1/1000 = 0.001$을 받는다. 그러면 `WeightedRandomSampler`가 이 가중치에 따라 복원추출로 표본을 뽑는다.

이 방식은 클래스 불균형이 보통 정도일 때(대략 1:10에서 1:100 사이) 특히 효과적이다. 불균형이 극심하면 가중 표집에 다른 기법을 함께 쓰는 일이 많다. 소수 클래스를 과표집하거나(SMOTE), 다수 클래스를 과소표집하거나, 드문 클래스의 오분류에 벌점을 더 크게 주는 손실 함수를 쓴다.

가중 표집은 실효 학습 분포를 바꿀 뿐 데이터 자체를 바꾸지는 않는다는 점에 유의하라. 모델이 보는 특징은 그대로이고 다만 소수 클래스의 예를 더 자주 만날 뿐이다. 모델의 성능을 현실적으로 가늠하려면 평가는 언제나 원래의 불균형한 분포에서 해야 한다.

## 연습문제

**연습문제 1.**
표본이 각각 50개, 200개, 1000개인 세 클래스로 된 데이터셋을 만들도록 코드를 고쳐라. 클래스 가중치를 계산하고, WeightedRandomSampler가 배치마다 클래스를 대체로 고르게 뽑는지 확인하라.

??? success "연습문제 1 풀이"
    `class_counts = torch.tensor([50, 200, 1000], dtype=torch.float32)`, `class_weights = 1.0 / class_counts`으로 둔다. 레이블은 `torch.cat((torch.zeros(50), torch.ones(200), 2*torch.ones(1000)))`으로 만든다. 레이블마다 `class_weights[int(label)]`으로 표본 가중치를 준다. 배치 크기가 60이면 배치마다 각 클래스에서 대략 20개씩 뽑힌다.

---


**연습문제 2.**
WeightedRandomSampler에서 `replacement=False`으로 두면 어떻게 되는가? 동작의 차이와 각 설정이 알맞은 상황을 설명하라.

??? success "연습문제 2 풀이"
    `replacement=False`이면 각 표본을 에포크마다 많아야 한 번 뽑으므로 전체 표본 수가 데이터셋의 크기와 같아지지만, 뽑힐 확률은 여전히 소수 클래스에 유리하다. (여기서 쓴) `replacement=True`이면 같은 표본을 여러 번 뽑을 수 있으므로 지정한 분포로 정확히 `num_samples`번 뽑을 수 있다. 클래스 균형을 엄격히 맞추려면 `replacement=True`을, 에포크마다 모든 표본을 대략 한 번씩 보려면 `replacement=False`을 쓴다.

---


**연습문제 3.**
`WeightedRandomSampler` 대신 `torch.utils.data.Subset`과 과표집을 쓰는 균형 잡힌 DataLoader를 구현하라. 소수 클래스를 복제하여 다수 클래스의 크기에 맞춘 새 데이터셋을 만들어라.

??? success "연습문제 3 풀이"
    `minority_idx = (labels == 0).nonzero().squeeze()`으로 소수 클래스의 인덱스를 구한다. `oversampled_idx = minority_idx.repeat(len(labels[labels==1]) // len(minority_idx) + 1)[:len(labels[labels==1])]`으로 복제한다. `all_idx = torch.cat([minority_idx, oversampled_idx, (labels==1).nonzero().squeeze()])`으로 합친다. `Subset(dataset, all_idx)`을 만들고 `shuffle=True`인 DataLoader로 감싼다.

