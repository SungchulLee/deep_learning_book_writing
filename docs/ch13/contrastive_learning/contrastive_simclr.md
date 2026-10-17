# 대조 학습 SimCLR

SimCLR는 같은 이미지에서 만든 두 어그멘테이션 뷰를 양성 쌍으로 두고, 배치의 다른 이미지 뷰와 구분하도록 표현을 학습한다. 인코더·투영 헤드와 NT-Xent 로스의 역할을 나누어 살펴본다. 큰 배치는 음성 수를 늘리는 선택이며 사용 가능성을 정하는 절대 조건은 아니다.

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

        # 마지막 분류 레이어를 없앤다
        self.encoder.fc = nn.Identity()

        # 사영 머리를 더한다
        self.projection_head = ProjectionHead(
            input_dim=feature_dim,
            hidden_dim=feature_dim,
            output_dim=projection_dim
        )

    def forward(self, x):
        # 피처를 뽑는다
        features = self.encoder(x)
        # 대조 공간으로 사영한다
        projections = self.projection_head(features)
        return features, projections


class NTXentLoss(nn.Module):
    """
    정규화 온도 조정 교차 엔트로피 로스 (NT-Xent)
    SimCLR가 쓰는 대조 로스 함수
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

        # 교차 엔트로피 로스를 셈한다
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

    # 두 시야를 모두 피드포워드한다
    _, projections1 = model(view1)
    _, projections2 = model(view2)

    # 대조 로스를 셈한다
    loss = criterion(projections1, projections2)

    # 역전파
    loss.backward()
    optimizer.step()

    return loss.item()


def extract_features(model, dataloader, device):
    """
    학습된 SimCLR 인코더로 피처를 뽑는다
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

    # 로스와 최적화기를 시작한다
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

**출력:**

```
SimCLR Model initialized successfully!
Encoder: ResNet
Projection head output dim: 128

Test forward pass successful!
Loss value: 4.1416
```

## 2. 논의

SimCLR는 ResNet의 최종 fc를 Identity로 바꾸고 인코더 피처 h와 투영 헤드 출력 z를 함께 반환한다. 기본 ResNet-50의 피처는 2048, 투영은 128차원이다. ResNet-18은 피처 512를 사용한다. 체크포인트를 읽지 않아 무작위 초기화 모델이다. 코드의 pretrained=False 표기는 사용한 torchvision 버전의 인터페이스와 함께 확인한다.

NTXentLoss는 두 뷰의 배치 순서가 같다고 가정한다. 각 뷰를 정규화한 뒤 2N×2N 유사도를 만들고 대각선만 제외한다. 각 앵커의 후보는 양성 1개와 음성 2N−2개다. 서로 다른 이미지라도 같은 의미 클래스를 가질 수 있으므로 음성이 반드시 의미적으로 다른 대상은 아니다.

온도는 양성만이 아니라 모든 후보 유사도 로짓을 나눈다. 낮으면 가장 큰 유사도 후보에 집중하며 그것이 잘못된 음성일 수도 있다. temperature>0과 두 뷰의 같은 배치 크기·피처 차원을 확인한다. N=1이면 음성이 없어 로스가 0인 퇴화된 문제가 된다. 행렬과 브로드캐스트 유사도 계산은 큰 배치에서 메모리를 사용한다.

투영 헤드는 대조 목표와 다운스트림 피처를 분리하는 데 도움이 될 수 있다. 원 연구에서 인코더 h를 선형 평가에 사용했지만 모든 과제에서 h가 z보다 우수하거나 특정 정보를 반드시 버린다고 단정하지 않는다. main의 두 뷰는 같은 이미지의 어그멘테이션이 아니라 독립 난수 텐서이며 학습 갱신도 호출하지 않아 의미 있는 대조 학습 결과가 아니다.

## 연습문제

<div class="drillbox" markdown>

**연습문제 1.** <span class="diff med" title="중간"></span>
N=256과 N=32인 두 뷰 배치에서 앵커별 양성·음성 수를 계산하고 해석하라.

</div>

??? success "연습문제 1 풀이"
    N=256은 양성 1개·음성 510개, N=32는 양성 1개·음성 62개다. 후보 수 차이는 크지만 음성의 의미·중복과 어그멘테이션·학습 예산도 품질에 영향을 준다. 배치가 작으면 학습이 불가능하거나 성능이 반드시 나빠진다고 단정하지 않는다.

---

<div class="drillbox" markdown>

**연습문제 2.** <span class="diff med" title="중간"></span>
사전 학습 후 투영 헤드 앞 피처를 사용하는 이유와 검증 조건을 설명하라.

</div>

??? success "연습문제 2 풀이"
    투영 공간은 어그멘테이션 뷰를 일치시키는 목표에 맞게 학습되므로 그 앞 피처가 다운스트림 과제에 더 유용할 수 있다. 이는 실험적 설계이며 색·위치 정보가 정확히 어느 레이어에서 얼마나 사라지는지는 확인이 필요하다. 같은 평가 분할·분류기 예산에서 h와 z를 비교한다.

---

<div class="drillbox" markdown>

**연습문제 3.** <span class="diff hard" title="어려움"></span>
아래 검색 함수가 계산하는 지표와 자기 검색·메모리·유효 입력 조건을 설명하라.

</div>

??? success "연습문제 3 풀이"
    ```python
    @torch.no_grad()
    def topk_retrieval_accuracy(model, query_loader, db_loader, device, k=5):
        """SimCLR 인코더의 피처로 상위 k 검색 정확도를 셈한다."""
        model.eval()

        # 질의 피처와 이름표를 뽑는다
        q_features, q_labels = extract_features(model, query_loader, device)
        q_features = F.normalize(q_features, dim=1)

        # 데이터베이스 피처와 이름표를 뽑는다
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

    쿼리마다 상위 k개 중 같은 레이블이 하나라도 있으면 성공인 hit@k다. k-NN 다수결 분류 정확도와 다르며 k가 커질수록 성공 가능성이 늘어난다. 쿼리가 DB에도 있으면 자기 항목을 제외하고 중복도 관리한다. 1≤k≤DB 크기·비어 있지 않은 로더와 클래스 구성을 확인한다. 전체 쿼리×DB 유사도를 저장하므로 큰 자료에는 청크·검색 인덱스가 필요하다.

## 정리하며

같은 이미지의 두 어그멘테이션 뷰, 배치 후보와 투영 공간의 로스를 연결한다. 온도·배치 크기의 효과를 고정 성능 법칙으로 해석하지 않고, 무작위 순전파와 실제 표현 학습·독립 검색 평가를 구분한다.
