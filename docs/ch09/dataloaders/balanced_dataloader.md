# 균형 잡힌 데이터로더

WeightedRandomSampler로 클래스 빈도의 역수에 비례하는 표본 가중치를 사용한다. 복원 샘플링에서 기대 클래스 비율을 바꾸는 방법이며 매 배치의 균형을 정확히 보장하지는 않는다.

## 1. 코드

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
    main()
```

**출력:**

```
Batch 1: Cats=54, Dogs=46
Batch 2: Cats=51, Dogs=49
Batch 3: Cats=42, Dogs=58
Batch 4: Cats=51, Dogs=49
Batch 5: Cats=44, Dogs=56
Batch 6: Cats=52, Dogs=48
Batch 7: Cats=49, Dogs=51
Batch 8: Cats=50, Dogs=50
Batch 9: Cats=56, Dogs=44
Batch 10: Cats=49, Dogs=51
Batch 11: Cats=43, Dogs=57
```

## 2. 논의

고양이 100개에는 각각 1/100, 개 1000개에는 각각 1/1000의 가중치를 주면 두 클래스의 전체 가중치 합은 같다. 복원추출 한 번에서 각 클래스를 뽑을 확률은 1/2이고 유한한 배치의 수는 변동한다.

샘플링은 학습 분포를 바꾸며 소수 표본을 반복해서 볼 수 있다. 클래스 가중 손실과 결합하면 보정이 중복될 수 있어 효과를 따로 검증해야 한다. 원래 분포의 확률을 원한다면 예측의 보정도 고려한다.

평가는 목표 운영 분포와 지표에 맞춰 수행한다. 모든 불균형 과제에 균형 샘플링이 필수이거나 특정 불균형 비율에서 항상 효과적이라는 규칙은 없다.

## 연습문제

<div class="drillbox" markdown>

**연습문제 1.** <span class="diff med" title="중간"></span>
표본 수가 50, 200, 1000인 세 클래스에 역빈도 가중치를 주면 배치 크기 60의 클래스 수는 어떻게 되는가?

</div>

??? success "연습문제 1 풀이"
    각 클래스의 전체 가중치가 같아 복원추출의 기대 개수는 20이다. 실제 배치가 정확히 20개씩인 것은 아니다. 정확한 균형이 필요하면 클래스별 인덱스를 조합하는 BatchSampler 같은 설계가 필요하다.

---


<div class="drillbox" markdown>

**연습문제 2.** <span class="diff med" title="중간"></span>
replacement=False에서 num_samples가 전체 데이터 수이면 한 에포크의 클래스 수는 어떻게 되는가?

</div>

??? success "연습문제 2 풀이"
    각 표본을 한 번씩 모두 뽑으면 원래 클래스 개수가 그대로 남는다. 가중치는 순서에 영향을 주지만 소수 표본을 복제하지 않는다. num_samples는 별도 설정값이며 자동으로 데이터 크기가 되는 것은 아니다.

---


<div class="drillbox" markdown>

**연습문제 3.** <span class="diff hard" title="어려움"></span>
인덱스를 복제해 소수 클래스를 다수 클래스 수에 맞추려면 원래 소수 인덱스를 또 합쳐야 하는가?

</div>

??? success "연습문제 3 풀이"
    복제 결과 자체를 목표 개수까지 잘랐다면 거기에 원래 소수 인덱스를 다시 더하지 않는다. 예를 들어 소수 인덱스 100개를 1000개로 반복해 만든 뒤 다수 인덱스 1000개와만 합친다. 중복 인덱스를 허용하는 Subset을 사용하고 순서를 섞는다.

## 정리하며

역빈도 가중치는 기대 샘플링 비율을 바꾸며 배치별 균형을 보장하지 않는다. 복원 여부, 샘플 수와 실제 운영 분포를 구분해 평가한다.
