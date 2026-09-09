# 대조 학습 SimCLR

SimCLR(시각 표현의 대조 학습을 위한 간단한 틀)는 같은 그림을 다르게 불린 시야 사이의 일치를 가장 크게 하여 표현을 배우는 자기 지도 학습 방법이다. MoCo와 달리 SimCLR는 같은 작은 배치의 다른 예에서 음성을 얻는 배치 안 대조 방식을 써서 구조는 더 간단하지만 큰 배치가 필요하다. 이 틀은 등뼈 부호기, 사영 머리, NT-Xent(정규화 온도 조정 교차 엔트로피) 손실 함수로 이루어진다.

## 1. 코드

```python
"""
SimCLR: 시각 표현의 대조 학습을 위한 간단한 틀
자기 지도 학습을 위한 SimCLR 알고리즘 구현.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

# ========================================================================
# 메인
# ========================================================================


class ProjectionHead(nn.Module):
    """SimCLR의 사영 머리"""
    def __init__(self, input_dim=2048, hidden_dim=2048, output_dim=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, x):
        return self.net(x)


class SimCLR(nn.Module):
    """
    대조 학습을 위한 SimCLR 모형

    인수:
        base_model: 등뼈 구조 (resnet18, resnet50 등)
        projection_dim: 사영 머리 출력의 차원
    """
    def __init__(self, base_model='resnet50', projection_dim=128):
        super().__init__()

        # 등뼈를 싣는다
        if base_model == 'resnet18':
            self.encoder = models.resnet18(pretrained=False)
            feature_dim = 512
        elif base_model == 'resnet50':
            self.encoder = models.resnet50(pretrained=False)
            feature_dim = 2048
        else:
            raise ValueError(f"Unknown model: {base_model}")

        # 마지막 분류 층을 없앤다
        self.encoder.fc = nn.Identity()

        # 사영 머리를 더한다
        self.projection_head = ProjectionHead(
            input_dim=feature_dim,
            hidden_dim=feature_dim,
            output_dim=projection_dim
        )

    def forward(self, x):
        # 특징을 뽑는다
        features = self.encoder(x)
        # 대조 공간으로 사영한다
        projections = self.projection_head(features)
        return features, projections


class NTXentLoss(nn.Module):
    """
    정규화 온도 조정 교차 엔트로피 손실 (NT-Xent)
    SimCLR가 쓰는 대조 손실 함수
    """
    def __init__(self, temperature=0.5):
        super().__init__()
        self.temperature = temperature

    def forward(self, z_i, z_j):
        """
        인수:
            z_i: 불린 시야 1의 사영, 꼴은 (batch_size, projection_dim)
            z_j: 불린 시야 2의 사영, 꼴은 (batch_size, projection_dim)
        """
        batch_size = z_i.shape[0]

        # 임베딩을 정규화한다
        z_i = F.normalize(z_i, dim=1)
        z_j = F.normalize(z_j, dim=1)

        # 두 시야를 이어 붙인다
        representations = torch.cat([z_i, z_j], dim=0)  # (2*batch_size, projection_dim)

        # 비슷함 행렬을 셈한다
        similarity_matrix = F.cosine_similarity(
            representations.unsqueeze(1),
            representations.unsqueeze(0),
            dim=2
        )  # (2*batch_size, 2*batch_size)

        # 이름표를 만든다: 양성 쌍이 대각 블록이다
        labels = torch.arange(batch_size, device=z_i.device)
        labels = torch.cat([labels + batch_size, labels], dim=0)

        # 제 자신과의 비슷함을 가린다
        mask = torch.eye(2 * batch_size, dtype=torch.bool, device=z_i.device)
        similarity_matrix = similarity_matrix.masked_fill(mask, -float('inf'))

        # 온도 조정을 적용한다
        similarity_matrix = similarity_matrix / self.temperature

        # 교차 엔트로피 손실을 셈한다
        loss = F.cross_entropy(similarity_matrix, labels)

        return loss


def train_step(model, optimizer, criterion, batch_views, device):
    """
    SimCLR의 학습 단계 하나

    인수:
        model: SimCLR 모형
        optimizer: 최적화기
        criterion: NTXentLoss
        batch_views: (view1, view2) 짝. 배치의 불린 시야 둘
        device: 토치 장치
    """
    model.train()
    optimizer.zero_grad()

    view1, view2 = batch_views
    view1, view2 = view1.to(device), view2.to(device)

    # 두 시야를 모두 앞먹임한다
    _, projections1 = model(view1)
    _, projections2 = model(view2)

    # 대조 손실을 셈한다
    loss = criterion(projections1, projections2)

    # 역전파
    loss.backward()
    optimizer.step()

    return loss.item()


def extract_features(model, dataloader, device):
    """
    학습된 SimCLR 부호기로 특징을 뽑는다
    아래쪽 과제에 쓸모 있다
    """
    model.eval()
    features_list = []
    labels_list = []

    with torch.no_grad():
        for images, labels in dataloader:
            images = images.to(device)
            features, _ = model(images)
            features_list.append(features.cpu())
            labels_list.append(labels)

    features = torch.cat(features_list, dim=0)
    labels = torch.cat(labels_list, dim=0)

    return features, labels


if __name__ == "__main__":
    # 사용 예
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 모형을 시작한다
    model = SimCLR(base_model='resnet50', projection_dim=128).to(device)

    # 손실과 최적화기를 시작한다
    criterion = NTXentLoss(temperature=0.5)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    # 학습 고리 보기 (의사 코드)
    print("SimCLR Model initialized successfully!")
    print(f"Encoder: {model.encoder.__class__.__name__}")
    print(f"Projection head output dim: 128")

    # 시험용 임시 배치를 만든다
    dummy_view1 = torch.randn(32, 3, 224, 224).to(device)
    dummy_view2 = torch.randn(32, 3, 224, 224).to(device)

    _, proj1 = model(dummy_view1)
    _, proj2 = model(dummy_view2)

    loss = criterion(proj1, proj2)
    print(f"\nTest forward pass successful!")
    print(f"Loss value: {loss.item():.4f}")
```

## 2. 논의

SimCLR의 구조에는 핵심 단계가 둘 있다. (대개 ResNet인) **부호기**가 불린 시야마다 표현 $\mathbf{h} = f(\mathbf{x})$을 뽑는다. 그다음 **사영 머리**가 이를 대조 손실을 적용하는 더 낮은 차원의 공간 $\mathbf{z} = g(\mathbf{h})$으로 잇댄다. 본디 논문의 매우 중요한 발견은 사영 머리가 좋은 성능에 꼭 필요하다는 것이다. 사영 머리 앞의 표현 $\mathbf{h}$이 사영된 표현 $\mathbf{z}$보다 아래쪽 과제에 훨씬 낫다. 사영 머리가 대조 학습에는 상관없지만 아래쪽 과제에는 쓸모 있는 불리기에 관한 정보를 버리기 때문이다.

NT-Xent 손실은 (저마다 불린 시야를 둘 내는) 그림 $N$장의 배치에서 $2N \times 2N$ 비슷함 행렬을 만든다. 닻마다 짝지어진 불린 시야가 양성이고 남은 $2(N-1)$개 항목이 음성 노릇을 한다. 손실은 온도로 크기를 조정한 교차 엔트로피로, 단위 초구면에서 양성 쌍은 모으고 음성 쌍은 밀어낸다. 온도 $\tau$이 분포의 매끄러움을 다스린다. 값이 낮으면 양성 쌍 둘레에 뾰족한 꼭대기가 생기고, 높으면 더 고른 분포가 나온다. 본디 논문은 $\tau = 0.5$이 잘 통함을 밝혔지만 배치가 충분히 크면 $0.1$까지 낮은 값이 성능을 높일 수 있다.

SimCLR의 핵심 한계는 배치 안 음성을 넉넉히 얻으려고 큰 배치(논문은 4096이나 8192를 쓴다)에 기댄다는 점이다. 배치 크기가 $N$이면 예마다 음성이 $2(N-1)$개이므로 배치가 작으면 대조 신호가 약해진다. 이 요구 때문에 SimCLR는 음성의 수를 배치 크기에서 떼어 놓는 MoCo 같은 큐 기반 방법에 견주어 계산이 비싸다. 그러나 SimCLR의 간결함과 좋은 성능 덕분에 가장 영향력 있는 대조 학습 틀 가운데 하나가 되었다.

## 연습문제

**연습문제 1.**
배치 크기가 $N = 256$일 때 NT-Xent 손실에서 닻마다 보는 양성 쌍과 음성 쌍의 수를 셈하라. 음성과 양성의 비는 얼마인가? $N = 32$이면 이 비가 어떻게 달라지는가?

??? success "연습문제 1 풀이"
    그림이 $N = 256$장이면 불린 시야가 $2N = 512$개이다. 닻마다 다음과 같다.

    - **양성 쌍**: 1개 (짝지어진 불린 시야)
    - **음성 쌍**: $2N - 2 = 510$개 (제 자신과 그 양성만 뺀 나머지 전부)
    - **비**: $510 : 1 = 510$

    $N = 32$이면 시야가 $2N = 64$개이고 닻마다 양성 1개와 음성 $62$개가 있어 비가 $62 : 1$이다.

    $N = 256$에서 $N = 32$으로 가면 비가 510에서 62로 떨어진다. SimCLR가 큰 배치를 요구하는 까닭이 이것이다. 배치가 작으면 음성이 적어 모형이 가려내는 특징을 배우기 어려워진다. 음성 대 양성 비가 $8.2$배 줄어든 것이다.

---

**연습문제 2.**
사전 학습 뒤에 사영 머리를 버리고 부호기의 표현만 아래쪽 과제에 쓰는 까닭을 설명하라. 사영 머리는 어떤 정보를 버리도록 배울 법한가?

??? success "연습문제 2 풀이"
    대조 손실은 사영된 표현이 적용한 불리기(자르기, 색 뒤틀기, 뒤집기)에 대해 변하지 않도록 북돋운다. 사영 머리는 서로 다른 그림을 가르는 데는 쓸모없지만 아래쪽 과제에는 값질 수 있는, 불리기에 딸린 정보를 없애도록 배운다. 이를테면 (색 흔들기가 불리기이므로) 색에 관한 정보나 (무작위 자르기를 쓰므로) 자른 조각 안의 공간 위치를 버릴 수 있다. 그런데 세밀한 분류 같은 아래쪽 과제에는 색 정보가 필요할 수 있고 물체 탐지에는 공간 정보가 반드시 필요하다.

    (사영 머리 앞의) 부호기 표현 $\mathbf{h}$을 아래쪽 과제에 쓰면 이 더 넉넉한 정보를 지키면서도, 부호기에 뜻있는 특징을 뽑도록 가르친 대조 사전 학습의 덕을 그대로 본다. 경험으로 Chen 외(2020)는 $\mathbf{z}$ 대신 $\mathbf{h}$을 쓸 때 ImageNet의 선형 평가 정확도가 10% 넘게 나아짐을 보였다.

---

**연습문제 3.**
학습된 SimCLR 부호기가 주어졌을 때 상위 $k$ 검색 정확도를 셈하는 함수를 구현하라. 이 함수는 질의 그림 묶음과 그림 데이터베이스를 부호화한 뒤, 질의마다 (부호기 특징 공간의 코사인 비슷함으로) 가장 가까운 이웃 $k$개를 찾아 그 가운데 같은 부류 이름표를 가진 것이 있는지 살펴야 한다.

??? success "연습문제 3 풀이"
    ```python
    @torch.no_grad()
    def topk_retrieval_accuracy(model, query_loader, db_loader, device, k=5):
        """SimCLR 부호기의 특징으로 상위 k 검색 정확도를 셈한다."""
        model.eval()

        # 질의 특징과 이름표를 뽑는다
        q_features, q_labels = extract_features(model, query_loader, device)
        q_features = F.normalize(q_features, dim=1)

        # 데이터베이스 특징과 이름표를 뽑는다
        db_features, db_labels = extract_features(model, db_loader, device)
        db_features = F.normalize(db_features, dim=1)

        # 코사인 비슷함 행렬을 셈한다
        similarity = torch.mm(q_features, db_features.t())

        # 질의마다 상위 k개 색인을 얻는다
        _, topk_indices = similarity.topk(k, dim=1)

        # 상위 k개 이웃 가운데 질의와 같은 이름표가 있는지 살핀다
        correct = 0
        for i in range(len(q_labels)):
            neighbor_labels = db_labels[topk_indices[i]]
            if q_labels[i] in neighbor_labels:
                correct += 1

        accuracy = correct / len(q_labels)
        return accuracy
    ```

    이 함수는 질의 그림과 데이터베이스 그림을 모두 (사영은 버리고) 부호기에 넣고, 코사인 비슷함을 위해 특징을 정규화한 뒤, 가장 가까운 이웃 $k$개 가운데 올바른 이름표를 가진 것이 있는지 살핀다. 학습이 나아가면 이 정확도가 올라가 부호기가 뜻있는 표현을 배움을 확인해 준다.

## 정리하며

**다룬 것** — 대조 학습 SimCLR

SimCLR의 구조에는 핵심 단계가 둘 있다.

핵심 클래스는 `ProjectionHead`, `SimCLR`, `NTXentLoss`이며 앞의 연습문제 3개로 직접 확인할 수 있다.
