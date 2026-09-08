# 데이터 로더

소수 예시 학습을 위한 에피소드 데이터 로더. 소수 예시 학습의 학습과 평가에 쓸 에피소드(과제)를 만든다.

모자란 데이터나 서로 이어진 데이터에서 효율적으로 배우는 것은 오늘날 깊은 학습의 한가운데 놓인 어려움이다. 이 모듈은 모델이 앞선 앎을 살려 새 과제에 재빨리 맞추어 가게 하는 소수 예시 학습 기법을 보여 준다.

## 1. 코드

```python
"""
소수 예시 학습을 위한 에피소드 데이터 로더

소수 예시 학습의 학습과 평가에 쓸 에피소드(과제)를 만든다.
에피소드마다 부류 N개에서 뽑은 받침 집합과 물음 집합으로 이루어진다.
"""

import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
from collections import defaultdict

# ========================================================================
# 메인
# ========================================================================


class EpisodicDataset(Dataset):
    """
    소수 예시 학습용 에피소드를 만들어 내는 데이터셋.
    
    에피소드마다 받침 집합과 물음 집합을 갖춘 N-갈래 K-예시 과제이다.
    """
    def __init__(self, data, labels, n_way, k_shot, n_query, n_episodes):
        """
        인수:
            data: (N, *input_shape) - 쓸 수 있는 모든 데이터
            labels: (N,) - 모든 데이터의 이름표
            n_way: 에피소드마다의 부류 개수
            k_shot: 부류마다의 받침 보기 개수
            n_query: 부류마다의 물음 보기 개수
            n_episodes: 만들어 낼 에피소드 개수
        """
        self.data = data
        self.labels = labels
        self.n_way = n_way
        self.k_shot = k_shot
        self.n_query = n_query
        self.n_episodes = n_episodes
        
        # 부류별로 데이터를 정리한다
        self.classes = torch.unique(labels).tolist()
        self.class_to_indices = defaultdict(list)
        for idx, label in enumerate(labels):
            self.class_to_indices[label.item()].append(idx)
    
    def __len__(self):
        return self.n_episodes
    
    def __getitem__(self, idx):
        """
        에피소드 하나를 만든다.
        
        반환값:
            support_set: (n_way * k_shot, *input_shape)
            support_labels: (n_way * k_shot,)
            query_set: (n_way * n_query, *input_shape)
            query_labels: (n_way * n_query,)
        """
        # n_way개의 부류를 아무렇게나 고른다
        episode_classes = np.random.choice(self.classes, self.n_way, replace=False)
        
        support_data = []
        support_labels = []
        query_data = []
        query_labels = []
        
        for class_idx, class_label in enumerate(episode_classes):
            # 이 부류의 모든 첨자를 얻는다
            class_indices = self.class_to_indices[class_label]
            
            # k_shot + n_query개의 보기를 뽑는다
            selected_indices = np.random.choice(
                class_indices,
                self.k_shot + self.n_query,
                replace=False
            )
            
            # 받침과 물음으로 쪼갠다
            support_indices = selected_indices[:self.k_shot]
            query_indices = selected_indices[self.k_shot:]
            
            # 받침 집합에 더한다
            support_data.append(self.data[support_indices])
            support_labels.extend([class_idx] * self.k_shot)
            
            # 물음 집합에 더한다
            query_data.append(self.data[query_indices])
            query_labels.extend([class_idx] * self.n_query)
        
        # 모든 부류를 이어 붙인다
        support_set = torch.cat(support_data, dim=0)
        support_labels = torch.tensor(support_labels)
        query_set = torch.cat(query_data, dim=0)
        query_labels = torch.tensor(query_labels)
        
        return support_set, support_labels, query_set, query_labels


class MiniImageNetLoader:
    """
    mini-ImageNet이나 비슷한 데이터셋을 위한 데이터 로더.
    에피소드 방식 소수 예시 학습을 위해 데이터를 정리한다.
    """
    def __init__(self, data_path=None):
        self.data_path = data_path
        # 실제로는 여기서 진짜 데이터셋을 불러온다
        # 지금은 흉내 데이터를 만든다
    
    def get_dataloader(self, split='train', n_way=5, k_shot=5, n_query=15, n_episodes=100, batch_size=4):
        """
        에피소드 데이터 로더를 만든다.
        
        인수:
            split: 'train', 'val' 또는 'test'
            n_way: 에피소드마다의 부류 개수
            k_shot: 부류마다의 받침 보기 개수
            n_query: 부류마다의 물음 보기 개수
            n_episodes: 에피소드 개수
            batch_size: 배치 크기(배치마다의 에피소드 개수)
        """
        # 데이터를 불러온다(보기용 흉내 데이터)
        if split == 'train':
            n_samples = 1000
            n_classes = 64
        elif split == 'val':
            n_samples = 300
            n_classes = 16
        else:  # 시험
            n_samples = 300
            n_classes = 20
        
        # 임시 데이터 만들기
        data = torch.randn(n_samples, 3, 84, 84)  # 표준 mini-ImageNet 크기
        labels = torch.randint(0, n_classes, (n_samples,))
        
        # 에피소드 데이터셋을 만든다
        dataset = EpisodicDataset(data, labels, n_way, k_shot, n_query, n_episodes)
        
        # 데이터로더를 만든다
        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=0  # 간단하게 하려고 0으로 둔다
        )
        
        return dataloader


class OmniglotLoader:
    """
    Omniglot 데이터셋(손으로 쓴 글자)을 위한 데이터 로더.
    소수 예시 학습의 표준 잣대.
    """
    def __init__(self, data_path=None):
        self.data_path = data_path
    
    def get_dataloader(self, split='train', n_way=5, k_shot=1, n_query=15, n_episodes=100):
        """
        Omniglot용 에피소드 데이터 로더를 만든다.
        """
        # 데이터를 불러온다(보기용 흉내)
        if split == 'train':
            n_samples = 1000
            n_classes = 1200  # 바탕 집합
        else:
            n_samples = 500
            n_classes = 423  # 평가 집합
        
        # 흉내 데이터를 만든다(28x28 흑백 그림)
        data = torch.randn(n_samples, 1, 28, 28)
        labels = torch.randint(0, n_classes, (n_samples,))
        
        # 에피소드 데이터셋을 만든다
        dataset = EpisodicDataset(data, labels, n_way, k_shot, n_query, n_episodes)
        
        return DataLoader(dataset, batch_size=1, shuffle=True)


def create_episode(data, labels, n_way, k_shot, n_query):
    """
    데이터에서 에피소드 하나를 만드는 도움 함수.
    
    인수:
        data: (N, *input_shape) - 모든 데이터
        labels: (N,) - 모든 이름표
        n_way: 부류 개수
        k_shot: 부류마다의 받침 보기
        n_query: 부류마다의 물음 보기
    
    반환값:
        support_set, support_labels, query_set, query_labels
    """
    # 서로 다른 부류를 얻는다
    unique_classes = torch.unique(labels)
    
    # n_way개의 부류를 뽑는다
    episode_classes = unique_classes[torch.randperm(len(unique_classes))[:n_way]]
    
    support_data = []
    support_labels = []
    query_data = []
    query_labels = []
    
    for class_idx, class_label in enumerate(episode_classes):
        # 이 부류의 첨자를 얻는다
        class_mask = (labels == class_label)
        class_data = data[class_mask]
        
        # 뒤섞고 쪼갠다
        perm = torch.randperm(len(class_data))
        support_indices = perm[:k_shot]
        query_indices = perm[k_shot:k_shot + n_query]
        
        # 집합에 더한다
        support_data.append(class_data[support_indices])
        support_labels.extend([class_idx] * k_shot)
        query_data.append(class_data[query_indices])
        query_labels.extend([class_idx] * n_query)
    
    support_set = torch.cat(support_data, dim=0)
    support_labels = torch.tensor(support_labels)
    query_set = torch.cat(query_data, dim=0)
    query_labels = torch.tensor(query_labels)
    
    return support_set, support_labels, query_set, query_labels


# 사용 예
if __name__ == "__main__":
    # 보기 1: 에피소드 데이터셋 만들기
    n_samples = 500
    n_classes = 20
    
    # 흉내 데이터(28x28 흑백 그림)
    data = torch.randn(n_samples, 1, 28, 28)
    labels = torch.randint(0, n_classes, (n_samples,))
    
    # 에피소드 데이터셋을 만든다(5-갈래 1-예시)
    dataset = EpisodicDataset(
        data=data,
        labels=labels,
        n_way=5,
        k_shot=1,
        n_query=15,
        n_episodes=100
    )
    
    # 에피소드 하나를 얻는다
    support_set, support_labels, query_set, query_labels = dataset[0]
    print(f"Support set shape: {support_set.shape}")  # (5, 1, 28, 28)
    print(f"Support labels: {support_labels}")
    print(f"Query set shape: {query_set.shape}")  # (75, 1, 28, 28)
    print(f"Query labels: {query_labels}")
    
    # 보기 2: 데이터 로더 쓰기
    dataloader = DataLoader(dataset, batch_size=4, shuffle=True)
    
    for batch_idx, batch in enumerate(dataloader):
        support_sets, support_label_sets, query_sets, query_label_sets = batch
        print(f"\nBatch {batch_idx}:")
        print(f"Support sets shape: {support_sets.shape}")  # (4, 5, 1, 28, 28)
        print(f"Query sets shape: {query_sets.shape}")  # (4, 75, 1, 28, 28)
        
        if batch_idx == 0:
            break
    
    # 보기 3: Mini-ImageNet 로더
    mini_loader = MiniImageNetLoader()
    train_loader = mini_loader.get_dataloader(
        split='train',
        n_way=5,
        k_shot=5,
        n_query=15,
        n_episodes=100,
        batch_size=4
    )
    
    for batch in train_loader:
        support, support_labels, query, query_labels = batch
        print(f"\nMini-ImageNet batch:")
        print(f"Support shape: {support.shape}")
        print(f"Query shape: {query.shape}")
        break
```

## 2. 논의

이 구현은 깔끔하고 읽기 쉬운 파이토치 코드로 소수 예시 학습의 핵심 개념을 보여 준다. 모듈 방식의 짜임 덕분에 낱낱의 부품을 살펴보고 다른 과제나 데이터셋에 맞추어 고치기 쉽다.

여기서 보인 본새는 더 복잡한 상황으로도 자연스럽게 넓어진다. 초매개변수, 구조의 변형, 여러 데이터셋을 두고 실험해 보면 이해가 깊어지고 메타 학습 과제에 대한 실전 감각이 쌓인다.

## 연습문제

**연습문제 1.**
코드를 끝까지 읽고 핵심 설계 결정을 짚어라. 구체적인 구현 선택 세 가지를 들고 각각이 소수 예시 학습에 왜 알맞은지 설명하라.

??? success "연습문제 1 풀이"
    설계 결정은 구현마다 다르지만 흔히 다음이 포함된다. (1) 활성화 함수의 선택 — ReLU 계열은 포화되지 않는 경사를 주어 학습을 빠르게 한다. (2) 정규화 전략 — 배치 정규화는 내부 공변량 이동을 줄여 학습을 안정시킨다. (3) 잔차 연결 — 있을 경우 건너뛰는 경로를 제공하여 깊은 신경망에서도 경사가 흐르게 한다. 각 선택은 표현력, 계산 비용, 학습 안정성 사이의 절충을 반영한다.

---

**연습문제 2.**
입력이 기대하는 모양과 자료형을 갖는지 확인하도록 주 함수나 클래스에 입력 검증을 추가하라. 잘못된 입력에는 유익한 오류 메시지를 내라.

??? success "연습문제 2 풀이"
    `forward` 메서드(또는 해당 함수)의 첫머리에 다음과 같은 검사를 추가한다. `assert x.dim() == expected_dims, f'Expected {expected_dims}D input, got {x.dim()}D'`와 `assert x.dtype == torch.float32, f'Expected float32, got {x.dtype}'`. 모양을 검증할 때는 중요한 차원을 확인한다. `B, C, H, W = x.shape; assert C == self.expected_channels`. 유익한 오류 메시지는 디버깅 속도를 크게 높이고 코드를 재사용하기에도 더 견고하게 만든다.

---

**연습문제 3.**
이 구현에서 생길 수 있는 실패 양상 두 가지를 서술하고, 각각을 어떻게 진단하고 고칠지 설명하라.

??? success "연습문제 3 풀이"
    흔한 실패 양상은 다음과 같다. (1) **경사 소실/폭발** — 경사의 노름을 지켜보아 진단한다(`torch.nn.utils.clip_grad_norm_`을 쓰거나 층마다 `param.grad.norm()`을 기록한다). 경사 자르기, 더 나은 초기화(Xavier/Kaiming), 또는 구조 변경(잔차 연결, 정규화)으로 고친다. (2) **과적합** — 학습 손실은 줄어드는데 검증 손실이 늘어나면 진단된다. 정칙화(드롭아웃, 가중치 감쇠, 데이터 증강)나 모델 용량 축소로 고친다. 이런 문제를 일찍 잡아내려면 언제나 학습 지표와 검증 지표를 함께 살펴라.

---

**연습문제 4.**
데이터 로더 구현을 검증하는 두루 갖춘 시험 함수를 작성하라. 빈 입력, 원소가 하나인 입력, 아주 큰 입력, 극단적인 값(0, 아주 큰 수)을 가진 입력 같은 모서리 경우를 시험하라.

??? success "연습문제 4 풀이"
    경계 조건을 두루 시험하는 함수를 만든다.
    ```python
    def test_episodicdataset():
        model = EpisodicDataset(...)
        # 보통의 입력
        assert model(normal_input).shape == expected_shape
        # 원소가 하나인 배치
        assert model(single_input).shape == (1, ...)
        # 큰 값 (넘침을 확인한다)
        out = model(torch.ones(...) * 1000)
        assert torch.isfinite(out).all()
        # 경사의 흐름
        out = model(normal_input)
        out.sum().backward()
        for p in model.parameters():
            assert p.grad is not None
    ```
    경사의 흐름을 시험하는 것은 그 구조가 처음부터 끝까지 이어지는 학습을 지원하는지 확인하는 데 특히 중요하다.

## 정리하며

**다룬 것** — 데이터 로더

이 구현은 깔끔하고 읽기 쉬운 파이토치 코드로 소수 예시 학습의 핵심 개념을 보여 준다.

핵심 클래스는 `EpisodicDataset`, `MiniImageNetLoader`, `OmniglotLoader`이며 앞의 연습문제 4개로 직접 확인할 수 있다.
