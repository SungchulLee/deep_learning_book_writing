# 데이터 로더

클래스별 소수의 서포트 표본과 별도의 쿼리 표본을 묶어 에피소드를 만드는 데이터 로더다. 여기서 MiniImageNetLoader·OmniglotLoader는 실제 파일을 읽지 않고 무작위 텐서를 생성하는 모양 시연용 클래스다.

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
    
    에피소드마다 받침 집합과 물음 집합을 갖춘 N-클래스 K-예시 과제이다.
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
            n_classes = 64
        elif split == 'val':
            n_classes = 16
        else:  # 시험
            n_classes = 20

        # 클래스마다 k_shot + n_query 보다 넉넉히 두어야 에피소드를 뽑을 수 있다.
        # 표본 수를 먼저 못박고 이름표를 randint로 뿌리면, 어떤 클래스는 모자라서
        # "Cannot take a larger sample than population" 이 난다.
        per_class = k_shot + n_query + 5
        n_samples = n_classes * per_class

        # 임시 데이터 만들기
        data = torch.randn(n_samples, 3, 84, 84)  # 표준 mini-ImageNet 크기
        labels = torch.arange(n_classes).repeat_interleave(per_class)
        
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
        # 참 Omniglot은 클래스가 1200개이지만, 흉내 자료로 그만큼 만들면
        # 쓸데없이 커진다. 클래스 수를 줄이고 클래스마다 넉넉히 둔다.
        if split == 'train':
            n_classes = 60   # 참 바탕 집합은 1200개이다
        else:
            n_classes = 20   # 참 평가 집합은 423개이다

        per_class = k_shot + n_query + 5
        n_samples = n_classes * per_class

        # 흉내 데이터를 만든다(28x28 흑백 그림)
        data = torch.randn(n_samples, 1, 28, 28)
        labels = torch.arange(n_classes).repeat_interleave(per_class)
        
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
    # randint로 이름표를 뿌리면 클래스마다 개수가 들쭉날쭉해져서, 어떤 클래스는
    # k_shot + n_query 보다 적어진다. 클래스마다 똑같이 나누어 준다.
    labels = torch.arange(n_classes).repeat_interleave(n_samples // n_classes)
    
    # 에피소드 데이터셋을 만든다(5-클래스 1-예시)
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

**출력:**

```
Support set shape: torch.Size([5, 1, 28, 28])
Support labels: tensor([0, 1, 2, 3, 4])
Query set shape: torch.Size([75, 1, 28, 28])
Query labels: tensor([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1,
        1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 3, 3, 3,
        3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4,
        4, 4, 4])

Batch 0:
Support sets shape: torch.Size([4, 5, 1, 28, 28])
Query sets shape: torch.Size([4, 75, 1, 28, 28])

Mini-ImageNet batch:
Support shape: torch.Size([4, 25, 3, 84, 84])
Query shape: torch.Size([4, 75, 3, 84, 84])
```

## 2. 논의

EpisodicDataset은 N개 클래스를 중복 없이 선택하고 클래스별 K+Q개 표본을 중복 없이 뽑아 앞 K개를 서포트, 뒤 Q개를 쿼리로 나눈다. 같은 원본 인덱스가 한 에피소드의 두 집합에 들어가지 않는다. 레이블은 에피소드 안의 0,…,N−1로 새로 매핑하며 원래 클래스 ID와 다르다.

이미지 입력에서 서포트는 $(NK,C,H,W)$, 쿼리는 $(NQ,C,H,W)$다. DataLoader의 batch_size=B는 이미지가 아닌 에피소드 수를 뜻해 각각 $(B,NK,C,H,W)$와 $(B,NQ,C,H,W)$가 된다. 모델이 한 에피소드만 받는지 메타 배치를 받는지 확인해야 한다.

n_way≤클래스 수, 각 클래스 표본≥K+Q와 양의 N·K·Q를 먼저 검증한다. __getitem__은 idx를 무시하고 새 난수를 뽑으므로 같은 인덱스도 다른 과제가 된다. shuffle=True만으로 고정 에피소드 재현이 되지 않으며 NumPy·PyTorch 난수와 워커 폴리시를 함께 관리한다.

MiniImageNetLoader의 data_path는 사용되지 않고 split 이름도 train·val 외에는 모두 테스트 크기로 처리한다. OmniglotLoader도 60·20개의 난수 클래스를 만들 뿐 원래 데이터셋·클래스 분할을 구현하지 않는다. create_episode는 부족한 표본·클래스에서 슬라이스가 조용히 짧아져 데이터와 선언한 레이블 수가 달라질 수 있다. 데이터 생성과 실제 벤치마크 로딩을 구분한다.

## 연습문제

<div class="drillbox" markdown>

**연습문제 1.** <span class="diff med" title="중간"></span>
5-way 1-shot, 클래스별 쿼리 15개인 RGB 84×84 과제의 한 에피소드와 네 에피소드 배치 모양을 적어라.

</div>

??? success "풀이"
    한 에피소드는 서포트 (5,3,84,84), 레이블 (5,), 쿼리 (75,3,84,84), 레이블 (75,)다. 배치 4이면 각각 앞에 4축이 붙는다. 레이블 0~4는 각 과제에서 독립적인 클래스 대응이다.

---

<div class="drillbox" markdown>

**연습문제 2.** <span class="diff med" title="중간"></span>
서포트·쿼리 중복과 부족한 자료를 어떻게 검증해야 하는가?

</div>

??? success "풀이"
    각 원래 인덱스의 교집합이 없고 클래스당 K+Q개가 충분한지 확인한다. 중복 이미지·어그멘테이션 사본이 다른 인덱스여도 의미상 같은 자료일 수 있어 분할 단계에서도 관리한다. create_episode의 슬라이스는 부족해도 요청 수를 자동 보장하지 않는다.

---

<div class="drillbox" markdown>

**연습문제 3.** <span class="diff med" title="중간"></span>
이 페이지의 Loader 이름만으로 실제 Mini-ImageNet·Omniglot 성능 평가가 되는가?

</div>

??? success "풀이"
    아니다. 경로에서 파일·분할을 읽지 않고 무작위 피처·레이블을 만든다. 실제 데이터와 클래스 분할·전처리·에피소드 시드가 필요하다. 출력 모양을 확인해도 표현의 적응 성능이나 벤치마크 정확도를 검증한 것은 아니다.

---

<div class="drillbox" markdown>

**연습문제 4.** <span class="diff hard" title="어려움"></span>
아래의 테스트 코드 조각을 검토하라. 정의되지 않은 클래스와 잘못된 이름, 입력 모양 가정을 찾아 이 페이지의 실제 함수에 맞는 점검 항목을 제안하라.

</div>

??? success "풀이"
    아래 조각은 실제 실행 가능한 테스트가 아니다. 함수 이름과 모델 이름부터 이 예제에 맞게 작성해야 한다.
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
    이 페이지는 완성된 사용자 정의 모델 클래스를 제공하지 않는다. 실제 함수의 입력·출력 모양, 허용 자료형, 유한한 값과 경사의 유무를 기준으로 검사한다. 빈 입력을 처리하는 방식은 연산마다 다르며, 무조건 정상적인 로스나 경사가 나와야 한다고 가정하지 않는다.

## 정리하며

에피소드별 레이블 매핑과 서포트·쿼리 분리, 메타 배치 축을 명확히 한다. 난수 자료 로더를 실제 벤치마크와 구분하고 표본 수·자료형·장치와 재현할 에피소드 구성을 검증한다.
