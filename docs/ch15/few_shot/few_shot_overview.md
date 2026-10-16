# 퓨샷 학습의 기초

퓨샷 학습은 목표 과제에서 사용할 수 있는 레이블 예시가 적은 상황을 다룬다. 적은 목표 예시가 전체 학습 데이터·계산도 적다는 뜻은 아니며, 사전 학습이나 여러 과제 학습을 활용할 수 있다. 새 클래스 분류의 에피소드 규약과 표현·적응·평가를 구분한다.

## 1. 문제의 구조

### 형식적 정의

퓨샷 학습에서는 클래스 집합 $\mathcal{C}$ 위에서 정의된 과제 $\mathcal{T}$을 생각한다. 일반적인 지도 학습보다 목표 과제에서 사용할 라벨 수가 제한되어 있다.

**학습 단계**: 클래스 $\mathcal{C}_{\text{train}}$을 아우르는 크고 레이블 붙은 데이터셋 $\mathcal{D}_{\text{train}}$을 쓸 수 있다.

**시험 단계**: $\mathcal{C}_{\text{train}} \cap \mathcal{C}_{\text{test}} = \emptyset$인 새 클래스 $\mathcal{C}_{\text{test}}$의 보기를, 클래스마다 레이블 붙은 보기 몇 개만 가지고 가려야 한다.

### N-way K-shot 분류

새 클래스 분류의 흔한 규약은 N-way K-shot이다. way는 후보 클래스 수, shot은 클래스별 레이블 있는 서포트 수를 뜻한다. 회귀·분할·가변 클래스 과제 등 모든 퓨샷 문제가 같은 형식인 것은 아니다.

- **N-way**: 과제마다의 클래스 개수
- **K-shot**: 클래스마다 레이블 붙은 보기의 개수

흔한 설정은 다음과 같다.

- **5-way 1-shot**: 클래스 5개에 서포트 각 1개
- **5-way 5-shot**: 클래스 5개에 보기 각 5개
- **20-way 1-shot**: 클래스가 더 많아 규모 확장성을 시험한다

### 서포트 집합과 쿼리 집합

퓨샷 과제마다 두 집합으로 짜인다.

**서포트 집합** $\mathcal{S}$: 배우는 데 쓰는 레이블 붙은 소규모 표본 집합

$$\mathcal{S} = \{(x_1, y_1), (x_2, y_2), \ldots, (x_{N \times K}, y_{N \times K})\}$$

**쿼리 집합** $\mathcal{Q}$: 예측할 표본들

$$\mathcal{Q} = \{(x_1^q, y_1^q), (x_2^q, y_2^q), \ldots, (x_{N \times Q}, y_{N \times Q})\}$$

여기서 Q는 클래스별 쿼리 수다. 쿼리 정답은 메타 학습·평가 지표 계산에 쓰며 테스트 적응에는 허용하지 않는다. 비지도·트랜스덕티브 설정에서 쿼리 입력을 함께 사용하는 경우도 정답 사용과 구분해 명시한다.

---

## 2. 수학적 틀

### 에피소드 학습

에피소드 학습은 학습 클래스에서 작은 서포트·쿼리 과제를 반복 구성하는 방식이다. 사전 학습한 일반 표현 위에 최근접 프로토타입·선형 분류·미세 조정을 적용하는 비에피소드 기준도 가능하다.

1. $\mathcal{C}_{\text{train}}$에서 클래스 $N$개를 뽑는다.
2. 클래스마다 서포트 표본 $K$개와 쿼리 표본 $Q$개를 뽑는다.
3. 서포트만으로 과제에 적응하거나 클래스 표현을 구성한다.
4. 쿼리 집합의 성능을 기반으로 모델 파라메터를 고친다.

서포트·쿼리 로스의 역할을 나누는 설계지만 특정 클래스를 외우지 않거나 새 과제 일반화가 자동 보장되는 것은 아니다.

### 메타 학습 목표

메타 학습 목표는 다음과 같이 정식화할 수 있다.

$$\theta^* = \argmin_\theta \mathbb{E}_{\mathcal{T} \sim p(\mathcal{T})} \left[ \mathcal{L}(\mathcal{T}; \theta) \right]$$

여기서 각 기호는 다음과 같다.

- $\theta$은 모델 파라메터를 나타낸다
- $\mathcal{T}$은 과제 분포 $p(\mathcal{T})$에서 뽑은 과제이다
- $\mathcal{L}(\mathcal{T}; \theta)$은 과제 $\mathcal{T}$에서의 로스다

### 일반화의 조건

새 과제 성능은 학습·평가 과제 분포, 공유 표현, 적응 규칙과 서포트 수에 의존한다. 학습 과제 수만 VC 표본 수로 대입한 식으로 임의의 메타 학습 모델을 보장할 수는 없다. 과제 내부 표본과 과제 간 의존, 로스·가설 복잡도의 가정을 명시해야 한다.

## 3. 접근법의 종류

퓨샷 학습 방법은 크게 세 클래스로 나눌 수 있다.

### 1. 거리 기반 방법

거리를 비교하여 가려낼 수 있는 임베딩 공간을 배운다.

$$p(y = c | x, \mathcal{S}) = \frac{\exp(-d(f_\theta(x), \mu_c))}{\sum_{c'} \exp(-d(f_\theta(x), \mu_{c'}))}$$

여기서 $f_\theta$은 임베딩 함수, $d$은 거리 함수, $\mu_c$은 임베딩 공간에서 클래스 $c$을 나타낸다.

**주요 방법**: 샴 네트워크, 매칭 네트워크, 프로토타입 네트워크, 관계 네트워크

### 2. 최적화 기반 방법(메타 학습)

그래디언트 기반 최적화로 새 과제에 재빨리 맞추어 가는 법을 배운다.

$$\theta' = \theta - \alpha \nabla_\theta \mathcal{L}_{\mathcal{S}}(\theta)$$

모델은 그래디언트 스텝 몇 번으로 재빨리 맞추어 갈 수 있게 하는 초기화 $\theta$을 배운다.

**주요 방법**: MAML, Reptile, Meta-SGD, FOMAML

### 3. 모델 기반 방법

외부 메모리나 특수한 부품을 갖춘 모델 구조를 쓴다.

$$h_t = f(x_t, h_{t-1}, \mathcal{M})$$

여기서 $\mathcal{M}$은 외부 메모리 모듈이다.

**주요 방법**: 메모리 증강 뉴럴 넷트웍, 소수 예시용 뉴럴 튜링 기계

---

## 4. 핵심 개념

### 귀납 바이어스

쓸모 있는 퓨샷 학습은 알맞은 귀납 바이어스에 크게 기댄다.

1. **조합성**: 물체는 다시 쓸 수 있는 부품으로 이루어진다
2. **유사도 구조**: 과제에 유용한 유사도 관계를 표현한다
3. **과제 구조**: 과제들은 기초의 공통 구조를 나누어 가진다

### 전이 학습과 퓨샷 학습

| 관점 | 설명 |
|---|---|
| 전이 학습 | 사전 학습한 지식·웨이트·표현을 새 과제에 재사용 |
| 퓨샷 설정 | 목표 레이블 예시 수가 제한된 문제 |
| 에피소드 메타 학습 | 여러 과제를 통해 적응 규칙·표현을 학습 |

셋은 서로 배타적이지 않다. 퓨샷에 미세 조정을 쓸 수 있고 모든 퓨샷이 클래스 비중복·에피소드 학습을 필수로 하는 것도 아니다. 이 장의 새 클래스 벤치마크에서는 학습·검증·테스트 클래스를 분리한다.

### 사전 학습의 역할

오늘날의 퓨샷 학습은 사전 학습된 표현을 자주 끌어 쓴다.

$$f_\theta = g_\psi \circ h_\phi$$

여기서 $h_\phi$은 사전 학습된 인코더(동결하거나 미세 조정한다)이고 $g_\psi$은 과제에 맞춘 적응 레이어다.

---

## 5. 벤치마크 데이터셋

### Omniglot

Omniglot은 50개 문자 체계의 글자 클래스 1,623개로 이루어진다. 각 글자는 서로 다른 사람이 쓴 표본 20개를 포함한다.

- 글자마다 보기 20개
- 원본 흑백 이미지와 모델 입력 전처리를 구분한다. 이 장의 28×28은 축소한 입력 규약이며 원본 해상도가 아니다.

### miniImageNet

퓨샷 평가를 위한 ImageNet의 부분집합이다.

- 클래스 100개, 클래스마다 이미지 600장
- 84×84 RGB 이미지
- 분할: 학습 클래스 64개, 검증 클래스 16개, 시험 클래스 20개

### tieredImageNet

tieredImageNet은 계층적 클래스 분할로 더 큰 의미 범주의 일반화를 평가한다.

- 클래스 608개를 상위 범주 34개로 묶는다.
- 상위 범주가 겹치지 않도록 학습·검증·테스트를 나눈다.
- 클래스 수와 분할 버전을 결과와 함께 기록한다.

### Meta-Dataset

여러 영역에 걸친 벤치마크이다.

- ImageNet, Omniglot, Aircraft, Birds, Textures, Quick Draw, Fungi, VGG Flower, Traffic Signs, MSCOCO
- 영역을 넘나드는 일반화를 시험한다

---

## 6. 평가 규약

### 표준 절차

1. 평가 클래스·N·K·Q와 에피소드 샘플링 규칙·횟수를 미리 정한다.
2. 각 과제에서 서포트만으로 허용된 적응을 하고 쿼리 정답은 평가에만 쓴다.
3. 실제 평가한 에피소드 수와 평균·불확실성을 보고한다.
4. 트랜스덕티브 입력 활용, BatchNorm 통계와 사전 학습 데이터 겹침을 명시한다.

### 신뢰 구간 계산

에피소드 정확도 $\{a_1, a_2, \ldots, a_n\}$이 주어지면 다음과 같다.

$$\bar{a} = \frac{1}{n}\sum_{i=1}^n a_i, \quad s = \sqrt{\frac{1}{n-1}\sum_{i=1}^n (a_i - \bar{a})^2}$$

95% 신뢰 구간은 다음과 같다.

$$\text{CI} = \bar{a} \pm t_{0.975, n-1} \cdot \frac{s}{\sqrt{n}}$$

---

## 7. PyTorch 구현

### 에피소드 샘플러

EpisodeSampler는 클래스별 최소 표본을 검사하지만 전체 클래스 수·양의 N·K·Q를 별도로 검증해야 한다. labels 원소에 item()을 호출하므로 일반 정수 목록을 바로 지원하지 않는다. FewShotDataset은 idx를 무시해 매번 새 에피소드를 만들며 입력 data도 목록 인덱싱 가능한 텐서 등이어야 한다.

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

### 기초 인코더 구조

Conv4Encoder는 네 번 풀링 후 전역 평균해 hidden_dim 피처를 반환한다. 이미지 크기는 네 풀링을 통과할 수 있어야 한다. 배치 정규화의 학습·평가와 서포트·쿼리 통계 혼합은 평가 규약에 영향을 준다. ResNetEncoder의 사전 학습 데이터·입력 채널·전처리와 클래스 중복도 기록한다.

```python
import torch.nn as nn
import torch.nn.functional as F

class ConvBlock(nn.Module):
    """
    소수 예시 학습에서 쓰는 표준 컨볼루션 블록.
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
    소수 예시 학습에서 흔히 쓰는 4레이어 컨볼루션 인코더.
    
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
    소수 예시 학습을 위한 ResNet 기반 인코더.
    
    미리 학습했거나 아무렇게나 초기화한 ResNet 등뼈를 쓰되
    마지막 가려내기 레이어는 없앤다.
    """
    
    def __init__(self, pretrained=True):
        super().__init__()
        
        import torchvision.models as models
        
        resnet = models.resnet18(pretrained=pretrained)
        
        # 마지막 온연결 레이어를 없앤다
        self.encoder = nn.Sequential(*list(resnet.children())[:-1])
        self.output_dim = 512
    
    def forward(self, x):
        features = self.encoder(x)
        return features.view(features.size(0), -1)
```

### 평가 도구

evaluate_few_shot은 model(support,support_labels,query)가 로짓을 반환하는 API만 가정한다. no_grad로 전체 평가를 감싸 MAML처럼 경사 적응이 필요한 모델에는 그대로 맞지 않는다. 기본 CUDA 장치와 모델 장치를 맞추고 로더 길이가 요청 에피소드보다 짧을 때 실제 횟수도 확인한다.

Student-t 구간은 독립 에피소드 표본과 유효한 분산 추정에 기반한 근사다. 같은 이미지·고정 클래스 집합을 공유한 평가의 모든 불확실성을 설명하지는 않는다. n<2나 빈 쿼리에서 sem·정확도가 유효하지 않으며 confidence 범위도 검증한다.

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

### 데이터 증강

보기가 적으므로 퓨샷 학습에서 데이터 증강은 매우 중요하다.

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

### 하이퍼파라미터 민감도

설정은 독립적인 메타 검증 과제에서 선택한다.

에피소드 구조·적응 러닝레이트·스텝·인코더·사전 학습과 평가 모드를 메타 검증 클래스에서 선택한다. 정해진 임베딩 폭·2만~10만 과제를 모든 문제의 충분조건으로 제시하지 않는다.

### 흔히 빠지는 함정

1. **정보 누수**: 평가 정답·동일 원본·사전 학습 중복을 검사한다.
2. **과제 구성**: 균형 에피소드는 하나의 규약이며 불균형·가변 way/shot도 명시적으로 평가할 수 있다.
3. **모델 선택**: 메타 테스트가 아니라 메타 검증 과제로 선택한다.
4. **기준 모델**: 같은 인코더·사전 학습·튜닝 예산의 단순 기준을 포함한다.

---

## 연습문제

<div class="drillbox" markdown>

**연습문제 1.** <span class="diff med" title="중간"></span>
N-way K-shot과 에피소드 학습의 서포트·쿼리 역할을 정의하라.

</div>

??? success "연습문제 1 풀이"
    N개 클래스 각각 K개의 레이블 있는 서포트를 사용해 별도 쿼리를 예측한다. 메타 학습은 서포트 기반 적응·표현 구성 뒤 쿼리 로스로 공유 계수를 갱신할 수 있다. 테스트 쿼리 정답을 적응에 사용하지 않으며 에피소드 학습만으로 새 과제 일반화를 보장하지 않는다.

---

<div class="drillbox" markdown>

**연습문제 2.** <span class="diff med" title="중간"></span>
거리·최적화·모델 기반 접근과 데이터 생성의 관계를 설명하라.

</div>

??? success "연습문제 2 풀이"
    거리 기반은 피처 간 비교, 최적화 기반은 서포트 로스에 따른 파라메터 적응, 모델 기반은 메모리·조건부 계산 등을 사용한다. 데이터 생성·증강은 이들과 함께 적용할 수 있는 별도 요소다. 특정 방식이 가장 단순·보편적·우수하다는 고정 순위는 없다.

---

<div class="drillbox" markdown>

**연습문제 3.** <span class="diff hard" title="어려움"></span>
아래 간단한 샘플러의 클래스 ID·레이블 대응과 유효 입력 조건을 검토하라.

</div>

??? success "연습문제 3 풀이"
    random import가 필요하다. ImageFolder류의 dataset.classes는 클래스 이름이고 targets는 정수여서 cls와 y를 그대로 비교하면 후보가 비어 있을 수 있다. 실제 클래스 ID를 사용하고 클래스 수·K+Q 표본, 중복과 로컬 레이블 매핑을 확인한다. 반환값은 데이터·새 레이블이 아니라 인덱스 두 목록뿐이다.

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

<div class="drillbox" markdown>

**연습문제 4.** <span class="diff easy" title="쉬움"></span>
퓨샷·제로샷·메타 학습이 구분하는 기준을 설명하라.

</div>

??? success "연습문제 4 풀이"
    퓨샷·제로샷은 목표 과제·클래스의 레이블 예시를 몇 개 쓰는지에 관한 설정이다. 메타 학습은 여러 과제에서 적응 규칙·초기화·표현을 배우는 방법이다. 제로샷은 레이블 이름·설명·사전 학습 지식을 쓸 수 있어 아무 정보도 없는 학습은 아니다. 퓨샷이 모두 메타 학습인 것도 아니다.

## 정리하며

목표 예시 수, 사전 학습·메타 학습의 자원과 평가 과제를 구분한다. 서포트·쿼리·로컬 레이블·BatchNorm과 경사 적응 규약을 맞추고, 정보 누수 없는 분할과 실제 에피소드 수·불확실성을 보고한다.

**참고 문헌**

1. Vinyals, O., et al. "Matching Networks for One Shot Learning." NeurIPS 2016.
2. Snell, J., et al. "Prototypical Networks for Few-shot Learning." NeurIPS 2017.
3. Finn, C., et al. "Model-Agnostic Meta-Learning for Fast Adaptation of Deep Networks." ICML 2017.
4. Koch, G., et al. "Siamese Neural Networks for One-shot Image Recognition." ICML Deep Learning Workshop 2015.
5. Triantafillou, E., et al. "Meta-Dataset: A Dataset of Datasets for Learning to Learn from Few Examples." ICLR 2020.
