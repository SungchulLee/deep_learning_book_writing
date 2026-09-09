# 소수 예시 학습의 기초

소수 예시 학습은 여느 깊은 학습의 가장 큰 한계 가운데 하나인 엄청난 양의 이름표 붙은 데이터가 있어야 한다는 점을 다룬다. 사람은 보기 한둘만 보고도 새 물체를 알아보지만, 보통의 신경망은 부류마다 이름표 붙은 표본이 수천 개 있어야 한다. 소수 예시 학습은 아주 적은 데이터로도 배울 수 있는 알고리즘을 길러, 적은 경험에서 일반화하는 사람의 힘을 흉내 낸다.

---

## 1. 문제의 얼개

### 형식적 정의

소수 예시 학습에서는 부류의 모임 $\mathcal{C}$ 위에서 정의된 과제 $\mathcal{T}$을 생각한다. 보통의 지도 학습과 갈리는 핵심은 데이터를 얼마나 쓸 수 있느냐에 있다.

**학습 단계**: 부류 $\mathcal{C}_{\text{train}}$을 아우르는 크고 이름표 붙은 데이터셋 $\mathcal{D}_{\text{train}}$을 쓸 수 있다.

**시험 단계**: $\mathcal{C}_{\text{train}} \cap \mathcal{C}_{\text{test}} = \emptyset$인 새 부류 $\mathcal{C}_{\text{test}}$의 보기를, 부류마다 이름표 붙은 보기 몇 개만 가지고 가려야 한다.

### N-갈래 K-예시 분류

표준 평가 규약은 **N-갈래 K-예시** 과제를 쓴다.

- **N-갈래**: 과제마다의 부류 개수
- **K-예시**: 부류마다 이름표 붙은 보기의 개수

흔한 설정은 다음과 같다.

- **5-갈래 1-예시**: 부류 5개에 보기 각 1개(가장 어렵다)
- **5-갈래 5-예시**: 부류 5개에 보기 각 5개
- **20-갈래 1-예시**: 부류가 더 많아 규모 확장성을 시험한다

### 받침 집합과 물음 집합

소수 예시 과제마다 두 집합으로 짜인다.

**받침 집합** $\mathcal{S}$: 배우는 데 쓰는 이름표 붙은 작은 보기 모임

$$\mathcal{S} = \{(x_1, y_1), (x_2, y_2), \ldots, (x_{N \times K}, y_{N \times K})\}$$

**물음 집합** $\mathcal{Q}$: 가려내야 하는 보기들

$$\mathcal{Q} = \{(x_1^q, y_1^q), (x_2^q, y_2^q), \ldots, (x_{N \times Q}, y_{N \times Q})\}$$

여기서 $Q$은 부류마다의 물음 보기 개수이다.

---

## 2. 수학적 틀

### 에피소드 학습

소수 예시 모델은 **에피소드 학습**으로 익히는데, 학습 되풀이마다 소수 예시 시험 상황을 흉내 낸다.

1. $\mathcal{C}_{\text{train}}$에서 부류 $N$개를 뽑는다.
2. 부류마다 받침 보기 $K$개와 물음 보기 $Q$개를 뽑는다.
3. 받침 보기만 써서 물음 보기를 가려내도록 모델을 익힌다.
4. 물음 집합의 성능을 바탕으로 모델 매개변수를 고친다.

이 학습 틀은 모델이 특정 부류의 특징을 외우기보다 적은 데이터에서 **배우는 법**을 익히도록 만든다.

### 메타 학습 목표

메타 학습 목표는 다음과 같이 정식화할 수 있다.

$$\theta^* = \argmin_\theta \mathbb{E}_{\mathcal{T} \sim p(\mathcal{T})} \left[ \mathcal{L}(\mathcal{T}; \theta) \right]$$

여기서 각 기호는 다음과 같다.

- $\theta$은 모델 매개변수를 나타낸다
- $\mathcal{T}$은 과제 분포 $p(\mathcal{T})$에서 뽑은 과제이다
- $\mathcal{L}(\mathcal{T}; \theta)$은 과제 $\mathcal{T}$에서의 손실이다

### 일반화 한계

소수 예시 학습의 일반화 능력은 학습 이론의 렌즈로 뜯어볼 수 있다. 가설 부류가 $\mathcal{H}$인 모델에서 새 과제의 기대 오차는 다음으로 눌린다.

$$\mathbb{E}_{\mathcal{T}}[\text{err}(\mathcal{T})] \leq \hat{\text{err}}_{\text{train}} + \sqrt{\frac{d_{\mathcal{H}} \log(m) + \log(1/\delta)}{2m}}$$

여기서 $d_{\mathcal{H}}$은 가설 부류의 VC 차원이고 $m$은 학습 과제의 개수이다.

---

## 3. 접근법의 갈래

소수 예시 학습 방법은 크게 세 갈래로 나눌 수 있다.

### 1. 거리 기반 방법

거리를 견주어 가려낼 수 있는 묻힘 공간을 배운다.

$$p(y = c | x, \mathcal{S}) = \frac{\exp(-d(f_\theta(x), \mu_c))}{\sum_{c'} \exp(-d(f_\theta(x), \mu_{c'}))}$$

여기서 $f_\theta$은 묻힘 함수, $d$은 거리 재기, $\mu_c$은 묻힘 공간에서 부류 $c$을 나타낸다.

**주요 방법**: 샴 망, 맞춤 망, 원형 망, 관계 망

### 2. 최적화 기반 방법(메타 학습)

기울기 기반 최적화로 새 과제에 재빨리 맞추어 가는 법을 배운다.

$$\theta' = \theta - \alpha \nabla_\theta \mathcal{L}_{\mathcal{S}}(\theta)$$

모델은 기울기 걸음 몇 번으로 재빨리 맞추어 갈 수 있게 하는 초기화 $\theta$을 배운다.

**주요 방법**: MAML, Reptile, Meta-SGD, FOMAML

### 3. 모델 기반 방법

바깥 기억이나 특수한 부품을 갖춘 모델 구조를 쓴다.

$$h_t = f(x_t, h_{t-1}, \mathcal{M})$$

여기서 $\mathcal{M}$은 바깥 기억 모듈이다.

**주요 방법**: 기억 덧댄 신경망, 소수 예시용 신경 튜링 기계

---

## 4. 핵심 개념

### 귀납 치우침

쓸모 있는 소수 예시 학습은 알맞은 귀납 치우침에 크게 기댄다.

1. **조합성**: 물체는 다시 쓸 수 있는 부품으로 이루어진다
2. **닮음 짜임**: 닮은 부류는 닮은 표현을 가져야 한다
3. **과제 짜임**: 과제들은 밑바탕의 공통 짜임을 나누어 가진다

### 전이 학습과 소수 예시 학습

| 갈래 | 전이 학습 | 소수 예시 학습 |
|--------|------------------|-------------------|
| 대상 데이터 | 이름표 붙은 데이터가 어느 정도 있다 | 이름표 붙은 데이터가 아주 적다(보기 1~5개) |
| 부류 겹침 | 부류를 나누어 가질 수 있다 | 서로 겹치지 않는 부류 |
| 맞추기 | 대상에서 미세 조정 | 재빠른 맞춤 장치 |
| 학습 | 보통의 지도 학습 | 에피소드 학습 |

### 미리 학습의 몫

오늘날의 소수 예시 학습은 미리 학습된 표현을 자주 끌어 쓴다.

$$f_\theta = g_\psi \circ h_\phi$$

여기서 $h_\phi$은 미리 학습된 부호기(얼리거나 미세 조정한다)이고 $g_\psi$은 과제에 맞춘 적응 층이다.

---

## 5. 잣대 데이터셋

### Omniglot

알파벳 50가지에서 뽑은 글자 1,623개로 이루어진, MNIST의 "전치"에 해당한다.

- 글자마다 보기 20개
- 28×28 흑백 그림
- 표준 쪼갬: 학습 글자 964개, 시험 글자 659개

### 미니 이미지넷

소수 예시 평가를 위한 ImageNet의 부분집합이다.

- 부류 100개, 부류마다 그림 600장
- 84×84 색 그림
- 쪼갬: 학습 부류 64개, 검증 부류 16개, 시험 부류 20개

### 계층 이미지넷

더 큰 규모에 층층 짜임을 갖춘다.

- 부류 608개를 갈래 34개로 묶었다
- 뜻으로 갈라지도록 갈래를 나누었다
- 더 어려운 전이 상황

### Meta-Dataset

여러 영역에 걸친 잣대이다.

- ImageNet, Omniglot, Aircraft, Birds, Textures, Quick Draw, Fungi, VGG Flower, Traffic Signs, MSCOCO
- 영역을 넘나드는 일반화를 시험한다

---

## 6. 평가 규약

### 표준 절차

1. 시험 에피소드를 600~1000개 뽑는다.
2. 에피소드마다 다음을 한다.
   - 시험 집합에서 부류 N개를 뽑는다
   - 부류마다 받침 보기 K개와 물음 보기 Q개를 뽑는다
   - 받침 집합으로 물음의 이름표를 맞힌다
   - 정확도를 셈한다
3. 95% 믿음 구간과 함께 평균 정확도를 알린다.

### 믿음 구간 셈하기

에피소드 정확도 $\{a_1, a_2, \ldots, a_n\}$이 주어지면 다음과 같다.

$$\bar{a} = \frac{1}{n}\sum_{i=1}^n a_i, \quad s = \sqrt{\frac{1}{n-1}\sum_{i=1}^n (a_i - \bar{a})^2}$$

95% 믿음 구간은 다음과 같다.

$$\text{CI} = \bar{a} \pm t_{0.975, n-1} \cdot \frac{s}{\sqrt{n}}$$

---

## 7. PyTorch 구현

### 에피소드 뽑개

```python
import torch
import numpy as np
from collections import defaultdict

class EpisodeSampler:
    """
    소수 예시 학습의 학습과 평가에 쓸 에피소드를 뽑는다.
    
    에피소드마다 아무렇게나 고른 부류 N개에서 뽑은
    받침 집합과 물음 집합을 담는다.
    """
    
    def __init__(self, labels, n_way, k_shot, n_query, n_episodes):
        """
        인수:
            labels: 모든 표본의 부류 이름표 텐서
            n_way: 에피소드마다의 부류 개수
            k_shot: 부류마다의 받침 보기 개수
            n_query: 부류마다의 물음 보기 개수
            n_episodes: 만들어 낼 에피소드의 총 개수
        """
        self.labels = labels
        self.n_way = n_way
        self.k_shot = k_shot
        self.n_query = n_query
        self.n_episodes = n_episodes
        
        # 첨자 대응을 만든다: 부류 -> 표본 첨자 목록
        self.class_to_indices = defaultdict(list)
        for idx, label in enumerate(labels):
            self.class_to_indices[label.item()].append(idx)
        
        self.classes = list(self.class_to_indices.keys())
        
        # 부류마다 표본이 넉넉한지 확인한다
        min_samples = k_shot + n_query
        for c, indices in self.class_to_indices.items():
            if len(indices) < min_samples:
                raise ValueError(
                    f"Class {c} has {len(indices)} samples, "
                    f"need at least {min_samples}"
                )
    
    def sample_episode(self):
        """
        에피소드 하나를 뽑는다.
        
        반환값:
            support_indices: 받침 집합의 첨자
            query_indices: 물음 집합의 첨자
            class_mapping: 본디 이름표를 0...N-1로 옮기는 사전
        """
        # 이 에피소드에 쓸 부류 N개를 뽑는다
        episode_classes = np.random.choice(
            self.classes, 
            self.n_way, 
            replace=False
        )
        
        support_indices = []
        query_indices = []
        class_mapping = {}
        
        for new_label, original_class in enumerate(episode_classes):
            class_mapping[original_class] = new_label
            
            # 이 부류의 모든 첨자를 얻는다
            available_indices = self.class_to_indices[original_class]
            
            # 되돌리지 않고 뽑는다
            selected = np.random.choice(
                available_indices,
                self.k_shot + self.n_query,
                replace=False
            )
            
            support_indices.extend(selected[:self.k_shot])
            query_indices.extend(selected[self.k_shot:])
        
        return support_indices, query_indices, class_mapping
    
    def __iter__(self):
        for _ in range(self.n_episodes):
            yield self.sample_episode()
    
    def __len__(self):
        return self.n_episodes

class FewShotDataset(torch.utils.data.Dataset):
    """
    보통의 데이터셋에서 소수 예시 에피소드를 만드는 감싸개.
    """
    
    def __init__(self, data, labels, n_way, k_shot, n_query, n_episodes):
        """
        인수:
            data: 꼴이 (N, *input_shape)인 텐서
            labels: 꼴이 (N,)인 텐서
            n_way: 에피소드마다의 부류 개수
            k_shot: 부류마다의 받침 보기
            n_query: 부류마다의 물음 보기
            n_episodes: 에피소드 개수
        """
        self.data = data
        self.labels = labels
        self.sampler = EpisodeSampler(
            labels, n_way, k_shot, n_query, n_episodes
        )
        self.n_way = n_way
        self.k_shot = k_shot
        self.n_query = n_query
    
    def __getitem__(self, idx):
        # 에피소드를 하나 뽑는다
        support_idx, query_idx, class_map = self.sampler.sample_episode()
        
        # 데이터를 뽑아낸다
        support_data = self.data[support_idx]
        query_data = self.data[query_idx]
        
        # 이름표를 0...N-1로 다시 매긴다
        support_labels = torch.tensor([
            class_map[self.labels[i].item()] 
            for i in support_idx
        ])
        query_labels = torch.tensor([
            class_map[self.labels[i].item()] 
            for i in query_idx
        ])
        
        return support_data, support_labels, query_data, query_labels
    
    def __len__(self):
        return self.sampler.n_episodes
```

### 바탕 부호기 구조

```python
import torch.nn as nn
import torch.nn.functional as F

class ConvBlock(nn.Module):
    """
    소수 예시 학습에서 쓰는 표준 합성곱 블록.
    Conv -> BatchNorm -> ReLU -> MaxPool
    """
    
    def __init__(self, in_channels, out_channels, pool=True):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.bn = nn.BatchNorm2d(out_channels)
        self.pool = pool
    
    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = F.relu(x)
        if self.pool:
            x = F.max_pool2d(x, 2)
        return x

class Conv4Encoder(nn.Module):
    """
    소수 예시 학습에서 흔히 쓰는 4층 합성곱 부호기.
    
    이 구조는 원형 망, 맞춤 망, MAML 같은
    논문에서 표준 등뼈로 쓰인다.
    
    출력: 그림마다 64차원 묻힘(28x28 입력일 때)
    """
    
    def __init__(self, in_channels=1, hidden_dim=64):
        super().__init__()
        
        self.encoder = nn.Sequential(
            ConvBlock(in_channels, hidden_dim),      # 28 -> 14
            ConvBlock(hidden_dim, hidden_dim),       # 14 -> 7
            ConvBlock(hidden_dim, hidden_dim),       # 7 -> 3
            ConvBlock(hidden_dim, hidden_dim),       # 3 -> 1
        )
        
        # 입력 크기를 바탕으로 출력 차원을 셈한다
        # 28x28일 때: 64 * 1 * 1 = 64
        # 84x84일 때: 64 * 5 * 5 = 1600
        self.output_dim = hidden_dim
    
    def forward(self, x):
        """
        인수:
            x: 입력 그림 (batch, channels, height, width)
        
        반환값:
            embeddings: (batch, output_dim)
        """
        features = self.encoder(x)
        # 입력 크기가 바뀌어도 되도록 전역 평균 풀링
        features = F.adaptive_avg_pool2d(features, 1)
        return features.view(features.size(0), -1)

class ResNetEncoder(nn.Module):
    """
    소수 예시 학습을 위한 ResNet 기반 부호기.
    
    미리 학습했거나 아무렇게나 초기화한 ResNet 등뼈를 쓰되
    마지막 가려내기 층은 없앤다.
    """
    
    def __init__(self, pretrained=True):
        super().__init__()
        
        import torchvision.models as models
        
        resnet = models.resnet18(pretrained=pretrained)
        
        # 마지막 온연결 층을 없앤다
        self.encoder = nn.Sequential(*list(resnet.children())[:-1])
        self.output_dim = 512
    
    def forward(self, x):
        features = self.encoder(x)
        return features.view(features.size(0), -1)
```

### 평가 도구

```python
import numpy as np
from scipy import stats

def compute_accuracy(predictions, labels):
    """
    가려내기 정확도를 셈한다.
    
    인수:
        predictions: (n,) 맞힌 부류 첨자
        labels: (n,) 참 부류 첨자
    
    반환값:
        accuracy: [0, 1] 안의 실수
    """
    return (predictions == labels).float().mean().item()

def compute_confidence_interval(accuracies, confidence=0.95):
    """
    정확도 측정값의 믿음 구간을 셈한다.
    
    작은 표본에서 제대로 추론하도록 스튜던트 t분포를 쓴다.
    
    인수:
        accuracies: 여러 에피소드에서 얻은 정확도 값의 목록
        confidence: 믿음 수준(기본값 0.95는 95% 믿음 구간)
    
    반환값:
        mean: 평균 정확도
        ci: 믿음 구간의 반너비
    """
    accuracies = np.array(accuracies)
    n = len(accuracies)
    
    mean = np.mean(accuracies)
    std_error = stats.sem(accuracies)  # 평균의 표준 오차
    
    # 주어진 믿음 수준의 t값
    t_value = stats.t.ppf((1 + confidence) / 2, n - 1)
    ci = t_value * std_error
    
    return mean, ci

def evaluate_few_shot(model, test_dataset, n_episodes=600, device='cuda'):
    """
    표준 소수 예시 평가 규약.
    
    인수:
        model: forward(support, support_labels, query)를 갖춘 소수 예시 모델
        test_dataset: 평가용 FewShotDataset
        n_episodes: 평가할 에피소드 개수
        device: 평가를 돌릴 장치
    
    반환값:
        mean_accuracy: 에피소드에 걸친 평균 정확도
        ci: 95% 믿음 구간
    """
    model.eval()
    accuracies = []
    
    dataloader = torch.utils.data.DataLoader(
        test_dataset, 
        batch_size=1, 
        shuffle=True
    )
    
    with torch.no_grad():
        for i, (support, support_labels, query, query_labels) in enumerate(dataloader):
            if i >= n_episodes:
                break
            
            # 배치 차원을 없앤다
            support = support.squeeze(0).to(device)
            support_labels = support_labels.squeeze(0).to(device)
            query = query.squeeze(0).to(device)
            query_labels = query_labels.squeeze(0).to(device)
            
            # 예측을 얻는다
            logits = model(support, support_labels, query)
            predictions = logits.argmax(dim=1)
            
            # 정확도를 적는다
            acc = compute_accuracy(predictions, query_labels)
            accuracies.append(acc)
    
    mean_acc, ci = compute_confidence_interval(accuracies)
    
    return mean_acc, ci
```

---

## 8. 실용적인 고려

### 데이터 늘리기

보기가 적으므로 소수 예시 학습에서 데이터 늘리기는 매우 중요하다.

```python
import torchvision.transforms as T

few_shot_transforms = T.Compose([
    T.RandomResizedCrop(84, scale=(0.8, 1.0)),
    T.RandomHorizontalFlip(),
    T.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], 
                std=[0.229, 0.224, 0.225])
])
```

### 초매개변수 민감도

주요 초매개변수와 흔한 범위는 다음과 같다.

| 초매개변수 | 흔한 범위 | 비고 |
|---------------|---------------|-------|
| 학습률 | 1e-4 ~ 1e-3 | 예열이 필요할 때가 많다 |
| 묻힘 차원 | 64 ~ 1600 | 과제의 복잡도에 달렸다 |
| 학습 에피소드 | 2만 ~ 10만 | 복잡한 과제일수록 더 많이 |
| 갱신마다의 에피소드 | 1 ~ 8 | 에피소드 배치 |

### 흔히 빠지는 함정

1. **이름표 새기**: 시험 부류가 학습 중에 결코 드러나지 않게 하라
2. **치우친 에피소드**: 늘 부류마다 같은 수의 보기를 뽑아라
3. **학습 부류에 지나친 맞춤**: 검증 과제를 바탕으로 일찍 멈추기를 쓰라
4. **미리 학습 무시하기**: 오늘날의 방법은 미리 학습된 특징에서 큰 이득을 본다

---

## 연습문제

**연습문제 1.**
$N$-갈래 $K$-예시 학습을 정의하고 에피소드 학습 틀을 설명하라.

??? success "연습문제 1 풀이"
    $N$-갈래 $K$-예시에서는 부류마다 이름표 붙은 보기 $K$개만 가지고 부류 $N$개 사이에서 가린다. 에피소드 학습은 시험 상황을 흉내 낸 '에피소드'를 뽑는다. 곧 받침 집합(보기 $N \times K$개)과 물음 집합이다. 모델은 특정 부류를 외우는 것이 아니라 적은 보기에서 배우는 법을 배운다.

---

**연습문제 2.**
소수 예시 학습의 거리 기반, 최적화 기반, 지어내기 기반 접근법을 견주어라.

??? success "연습문제 2 풀이"
    거리 기반(샴, 원형): 거리가 부류의 닮음을 비추는 묻힘 공간을 배운다. 최적화 기반(MAML): 재빨리 맞추어 가는 초기화를 배운다. 지어내기 기반: 학습 보기를 더 만들어 낸다. 거리 방법이 가장 단순하고, MAML이 가장 두루 쓰이며, 지어내기는 데이터 늘리기가 뜻있을 때 도움이 된다.

---

**연습문제 3.**
파이토치로 간단한 $N$-갈래 $K$-예시 에피소드 뽑개를 구현하라.

??? success "연습문제 3 풀이"
    ```python
    def sample_episode(dataset, N=5, K=5, Q=15):
        classes = random.sample(dataset.classes, N)
        support, query = [], []
        for cls in classes:
            indices = [i for i, y in enumerate(dataset.targets) if y == cls]
            chosen = random.sample(indices, K + Q)
            support.extend(chosen[:K])
            query.extend(chosen[K:])
        return support, query
    ```

---

**연습문제 4.**
소수 예시 학습, 영 예시 학습, 메타 학습의 차이는 무엇인가?

??? success "연습문제 4 풀이"
    소수 예시: 이름표 붙은 보기 몇 개로 배운다. 영 예시: 이름표 붙은 보기 없이 곁다리 정보(속성, 글 설명)로 가린다. 메타 학습: 과제 분포에서 '배우는 법을 배우는' 더 넓은 틀이다. 소수 예시 학습은 대개 메타 학습의 응용이고, 영 예시 학습은 딸린 앎의 전이에 기댄다.

## 정리하며

소수 예시 학습은 적은 데이터로 배우는 근본 어려움을 다음과 같이 다룬다.

1. 작은 과제를 많이 두고 모델을 익힌다(에피소드 학습).
2. 재빨리 맞추어 갈 수 있게 하는 표현을 배운다.
3. 닮음과 조합성에 대한 귀납 치우침을 끌어 쓴다.

이 분야는 단순한 거리 기반 방법에서 정교한 메타 학습 알고리즘으로 발전해 왔고, 최근에는 큰 규모의 미리 학습과 여러 갈래 학습까지 아우른다.

**참고 문헌**

1. Vinyals, O., et al. "Matching Networks for One Shot Learning." NeurIPS 2016.
2. Snell, J., et al. "Prototypical Networks for Few-shot Learning." NeurIPS 2017.
3. Finn, C., et al. "Model-Agnostic Meta-Learning for Fast Adaptation of Deep Networks." ICML 2017.
4. Koch, G., et al. "Siamese Neural Networks for One-shot Image Recognition." ICML Deep Learning Workshop 2015.
5. Triantafillou, E., et al. "Meta-Dataset: A Dataset of Datasets for Learning to Learn from Few Examples." ICLR 2020.
