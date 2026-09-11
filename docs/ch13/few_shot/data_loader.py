"""data_loader — data_loader 페이지의 코드를 모듈로 쓸 수 있게 옮겨 놓은 것이다.

이 파일은 ch13/few_shot/data_loader.md 에서 자동으로 뽑아낸 것이므로, 고칠 일이 있으면
그 페이지를 고치고 다시 뽑아내는 편이 낫다.
"""

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
            n_classes = 64
        elif split == 'val':
            n_classes = 16
        else:  # 시험
            n_classes = 20

        # 갈래마다 k_shot + n_query 보다 넉넉히 두어야 에피소드를 뽑을 수 있다.
        # 표본 수를 먼저 못박고 이름표를 randint로 뿌리면, 어떤 갈래는 모자라서
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
        # 참 Omniglot은 갈래가 1200개이지만, 흉내 자료로 그만큼 만들면
        # 쓸데없이 커진다. 갈래 수를 줄이고 갈래마다 넉넉히 둔다.
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
    # randint로 이름표를 뿌리면 갈래마다 개수가 들쭉날쭉해져서, 어떤 갈래는
    # k_shot + n_query 보다 적어진다. 갈래마다 똑같이 나누어 준다.
    labels = torch.arange(n_classes).repeat_interleave(n_samples // n_classes)
    
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
