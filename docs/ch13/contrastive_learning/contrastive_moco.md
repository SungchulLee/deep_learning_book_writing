# 대조 학습 MoCo

MoCo(Momentum Contrast)는 EMA로 갱신하는 키 인코더와 과거 특징 큐를 사용해 대조 학습의 후보 집합을 구성한다. 큐 크기와 현재 배치 크기를 분리할 수 있지만 모델·큐·점수와 학습 활성의 비용은 남는다. 아래 구현과 원래 MoCo 버전들의 차이도 함께 살펴본다.

## 1. 코드

```python
"""
MoCo: 지도 없는 시각 표현 학습을 위한 관성 대조
자기 지도 학습을 위한 MoCo v2 알고리즘 구현.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
import copy

# ========================================================================
# 메인
# ========================================================================


class MoCo(nn.Module):
    """
    관성 대조(MoCo) 모형

    효율적인 대조 학습을 위해 관성 인코더와 큐 구조를 쓴다.

    인수:
        base_encoder: 등뼈 구조
        dim: 특징 차원
        K: 큐 크기
        m: 관성 계수
        T: 온도
    """
    def __init__(self, base_encoder='resnet50', dim=128, K=65536, m=0.999, T=0.07):
        super().__init__()

        self.K = K
        self.m = m
        self.T = T

        # 질의 인코더를 만든다
        if base_encoder == 'resnet50':
            self.encoder_q = models.resnet50(pretrained=False)
            feature_dim = 2048
        elif base_encoder == 'resnet18':
            self.encoder_q = models.resnet18(pretrained=False)
            feature_dim = 512
        else:
            raise ValueError(f"Unknown encoder: {base_encoder}")

        # 완전 연결 층을 사영 머리로 바꾼다
        self.encoder_q.fc = nn.Sequential(
            nn.Linear(feature_dim, feature_dim),
            nn.ReLU(),
            nn.Linear(feature_dim, dim)
        )

        # 열쇠 인코더(관성 인코더)를 만든다
        self.encoder_k = copy.deepcopy(self.encoder_q)

        # 열쇠 인코더를 얼린다. 관성으로 갱신된다
        for param in self.encoder_k.parameters():
            param.requires_grad = False

        # 큐를 만든다
        self.register_buffer("queue", torch.randn(dim, K))
        self.queue = F.normalize(self.queue, dim=0)

        # 큐 포인터
        self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))

    @torch.no_grad()
    def _momentum_update_key_encoder(self):
        """
        열쇠 인코더의 관성 갱신
        """
        for param_q, param_k in zip(self.encoder_q.parameters(), self.encoder_k.parameters()):
            param_k.data = param_k.data * self.m + param_q.data * (1. - self.m)

    @torch.no_grad()
    def _dequeue_and_enqueue(self, keys):
        """
        새 열쇠로 큐를 고친다
        """
        batch_size = keys.shape[0]

        ptr = int(self.queue_ptr)

        # ptr 자리의 열쇠를 바꾼다
        if ptr + batch_size <= self.K:
            self.queue[:, ptr:ptr + batch_size] = keys.T
        else:
            # 필요하면 처음으로 돌아간다
            remaining = self.K - ptr
            self.queue[:, ptr:] = keys[:remaining].T
            self.queue[:, :batch_size - remaining] = keys[remaining:].T

        # 포인터를 옮긴다
        ptr = (ptr + batch_size) % self.K
        self.queue_ptr[0] = ptr

    @torch.no_grad()
    def _batch_shuffle_ddp(self, x):
        """
        분산 학습을 위한 배치 뒤섞기 (간추린 판)
        온전한 구현에서는 여러 GPU에 걸쳐 뒤섞는다
        """
        idx_shuffle = torch.randperm(x.shape[0], device=x.device)
        idx_unshuffle = torch.argsort(idx_shuffle)
        return x[idx_shuffle], idx_unshuffle

    @torch.no_grad()
    def _batch_unshuffle_ddp(self, x, idx_unshuffle):
        """
        배치 뒤섞기를 되돌린다
        """
        return x[idx_unshuffle]

    def forward(self, im_q, im_k):
        """
        앞먹임

        인수:
            im_q: 질의 그림 (batch_size, 3, H, W)
            im_k: 열쇠 그림 (batch_size, 3, H, W)

        반환값:
            logits: (batch_size, 1 + K)
            labels: (batch_size,)
        """
        # 질의 특징을 셈한다
        q = self.encoder_q(im_q)
        q = F.normalize(q, dim=1)

        # 열쇠 특징을 셈한다
        with torch.no_grad():
            # 관성 인코더를 갱신한다
            self._momentum_update_key_encoder()

            # 배치 정규화를 제대로 쓰려고 뒤섞는다
            im_k, idx_unshuffle = self._batch_shuffle_ddp(im_k)

            k = self.encoder_k(im_k)
            k = F.normalize(k, dim=1)

            # 뒤섞은 것을 되돌린다
            k = self._batch_unshuffle_ddp(k, idx_unshuffle)

        # 로짓을 계산한다
        # 양성 로짓: (batch_size, 1)
        l_pos = torch.einsum('nc,nc->n', [q, k]).unsqueeze(-1)

        # 음성 로짓: (batch_size, K)
        l_neg = torch.einsum('nc,ck->nk', [q, self.queue.clone().detach()])

        # 로짓을 이어 붙인다
        logits = torch.cat([l_pos, l_neg], dim=1)

        # 온도를 적용한다
        logits /= self.T

        # 이름표: 양성이 색인 0에 있다
        labels = torch.zeros(logits.shape[0], dtype=torch.long, device=logits.device)

        # 큐에서 빼고 큐에 넣는다
        self._dequeue_and_enqueue(k)

        return logits, labels


def train_step_moco(model, optimizer, batch_views, device):
    """
    MoCo의 학습 단계 하나

    인수:
        model: MoCo 모형
        optimizer: 최적화기
        batch_views: (view1, view2) 짝. 불린 시야 둘
        device: 토치 장치
    """
    model.train()
    optimizer.zero_grad()

    im_q, im_k = batch_views
    im_q, im_k = im_q.to(device), im_k.to(device)

    # 순전파
    logits, labels = model(im_q, im_k)

    # 손실(교차 엔트로피)을 셈한다
    loss = F.cross_entropy(logits, labels)

    # 역전파
    loss.backward()
    optimizer.step()

    return loss.item()


class MoCoV3(nn.Module):
    """
    MoCo v3: 비전 트랜스포머를 쓰는 개선판
    핵심 개념에 집중한 간추린 구현
    """
    def __init__(self, base_encoder='resnet50', dim=256, mlp_dim=4096, T=0.2):
        super().__init__()

        self.T = T

        # 바탕 인코더
        if base_encoder == 'resnet50':
            backbone = models.resnet50(pretrained=False)
            feature_dim = 2048
            backbone.fc = nn.Identity()
        else:
            raise ValueError(f"Unknown encoder: {base_encoder}")

        self.encoder = backbone

        # 사영 머리 (3층 다층 퍼셉트론)
        self.projector = nn.Sequential(
            nn.Linear(feature_dim, mlp_dim),
            nn.BatchNorm1d(mlp_dim),
            nn.ReLU(inplace=True),
            nn.Linear(mlp_dim, mlp_dim),
            nn.BatchNorm1d(mlp_dim),
            nn.ReLU(inplace=True),
            nn.Linear(mlp_dim, dim),
            nn.BatchNorm1d(dim)
        )

        # 예측 머리
        self.predictor = nn.Sequential(
            nn.Linear(dim, mlp_dim),
            nn.BatchNorm1d(mlp_dim),
            nn.ReLU(inplace=True),
            nn.Linear(mlp_dim, dim)
        )

    def forward(self, x1, x2):
        """
        MoCo v3의 앞먹임
        관성 인코더 없이 대칭 손실을 쓴다
        """
        # 두 시야를 모두 부호화한다
        f1 = self.encoder(x1)
        f2 = self.encoder(x2)

        # 사영한다
        z1 = self.projector(f1)
        z2 = self.projector(f2)

        # 예측한다
        p1 = self.predictor(z1)
        p2 = self.predictor(z2)

        # 대칭 손실을 셈한다
        loss = self.contrastive_loss(p1, z2) / 2 + self.contrastive_loss(p2, z1) / 2

        return loss

    def contrastive_loss(self, q, k):
        """
        MoCo v3의 대조 손실
        """
        # 정규화
        q = F.normalize(q, dim=1)
        k = F.normalize(k, dim=1)

        # (분산 학습에서) 모든 목표를 모은다
        # 여기서는 배치 자체를 쓴다
        logits = torch.mm(q, k.t()) / self.T
        labels = torch.arange(logits.shape[0], device=logits.device)

        loss = F.cross_entropy(logits, labels)
        return loss


if __name__ == "__main__":
    # 사용 예
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # MoCo v2를 시작한다
    print("Testing MoCo v2...")
    model_v2 = MoCo(base_encoder='resnet50', dim=128, K=4096).to(device)
    optimizer = torch.optim.SGD(model_v2.parameters(), lr=0.03, momentum=0.9, weight_decay=1e-4)

    # 순전파 시험
    im_q = torch.randn(32, 3, 224, 224).to(device)
    im_k = torch.randn(32, 3, 224, 224).to(device)

    logits, labels = model_v2(im_q, im_k)
    loss = F.cross_entropy(logits, labels)

    print(f"MoCo v2 initialized successfully!")
    print(f"Queue size: {model_v2.K}")
    print(f"Momentum coefficient: {model_v2.m}")
    print(f"Logits shape: {logits.shape}")
    print(f"Loss: {loss.item():.4f}")

    # MoCo v3을 시작한다
    print("\nTesting MoCo v3...")
    model_v3 = MoCoV3(base_encoder='resnet50', dim=256).to(device)

    loss_v3 = model_v3(im_q, im_k)
    print(f"MoCo v3 initialized successfully!")
    print(f"Loss: {loss_v3.item():.4f}")
```

**출력:**

```
Testing MoCo v2...
MoCo v2 initialized successfully!
Queue size: 4096
Momentum coefficient: 0.999
Logits shape: torch.Size([32, 4097])
Loss: 0.0082

Testing MoCo v3...
MoCo v3 initialized successfully!
Loss: 3.5362
```

## 2. 논의

MoCo 클래스는 역전파로 학습하는 쿼리 인코더와 초기 복사한 키 인코더를 갖는다. 키 계수는 $\theta_k\leftarrow m\theta_k+(1-m)\theta_q$로 갱신하고 키는 no_grad에서 계산한다. 큐는 (d,K)의 정규화한 특징 버퍼와 포인터를 저장한다. 로짓은 양성 하나와 큐 K개를 붙인 (B,K+1), 정답은 모두 0이다.

EMA는 매개변수만 갱신하고 BatchNorm 이동 통계 버퍼는 별도로 다뤄진다. no_grad·requires_grad=False만으로 평가 모드가 되는 것은 아니다. _batch_shuffle_ddp라는 이름과 달리 이 코드는 로컬 배치만 섞고 분산 all-gather를 하지 않는다. 같은 장치 안에서 순서만 바꾸어도 BatchNorm의 배치 통계는 같아 원래 분산 ShuffleBN의 역할을 재현하지 못한다.

forward가 평가 모드에서도 EMA와 큐를 변경한다. 순전파를 한 번 확인하는 호출도 상태에 영향을 준다. 큐는 초기 난수로 시작하고 K보다 큰 배치를 넣는 일반적인 순환 삽입은 처리하지 않는다. K>0, 0≤m≤1, T>0과 배치·장치 규약을 확인한다. 큐의 다른 항목도 같은 의미의 이미지일 수 있어 거짓 음성이 존재한다.

**MoCo v3와의 차이:** 원래 MoCo v3는 큐를 쓰지 않지만 모멘텀 인코더를 유지하고 배치 안 대조 목표를 사용한다. 이 페이지 MoCoV3는 모멘텀 목표 인코더도 stop-gradient도 없고 양쪽 가지에 경사를 전달한다. 따라서 예측 헤드가 있다는 사실만으로 원래 v3나 BYOL·SimSiam과 동등하지 않다. [공식 MoCo v3 구현](https://github.com/facebookresearch/moco-v3/blob/main/moco/builder.py)을 참고한다.

main은 독립 난수 이미지의 로짓·손실을 계산할 뿐, 같은 이미지의 증강·최적화·독립 평가를 포함하지 않는다. BatchNorm1d가 있는 v3는 학습 모드의 배치 1에서도 별도 제약을 갖는다.

## 연습문제

<div class="drillbox" markdown>

**연습문제 1.** <span class="diff med" title="중간"></span>
K=65536, d=128의 FP32 큐와 같은 개수의 FP32 RGB 224×224 이미지 저장량을 비교하라.

</div>

??? success "연습문제 1 풀이"
    큐는 33,554,432바이트=32MiB(약 33.55MB)다. 이미지는 39,460,012,032바이트=36.75GiB(약 39.46GB)이며 비율은 1176배다. MB와 MiB를 섞어 비율을 구하지 않는다. 이는 특징·이미지 원시 저장 비교이며 인코더·경사·큐 로짓 등의 메모리는 별도다.

---

<div class="drillbox" markdown>

**연습문제 2.** <span class="diff med" title="중간"></span>
EMA 계수 m과 큐 특징의 시간적 일관성 사이의 절충을 설명하라.

</div>

??? success "연습문제 2 풀이"
    m이 1에 가까우면 키 계수가 천천히 변해 과거 특징과의 변화가 완만해질 수 있지만 쿼리의 변화 반영도 늦어진다. m=0은 현재 쿼리 계수를 복사하고 m=1은 초기 키 계수를 유지한다. 0.999가 모든 큐·학습률에서 최적이거나 m=0.5가 항상 실패하는 것은 아니다. BatchNorm 버퍼와 큐 나이도 함께 고려한다.

---

<div class="drillbox" markdown>

**연습문제 3.** <span class="diff hard" title="어려움"></span>
아래 compute_alignment_uniformity가 계산하는 통계와 표준 균일성 지표의 차이를 설명하라.

</div>

??? success "연습문제 3 풀이"
    ```python
    @torch.no_grad()
    def compute_alignment_uniformity(self, im_q, im_k):
        """정렬(양성 비슷함)과 고름(음성 비슷함)을 셈한다."""
        q = F.normalize(self.encoder_q(im_q), dim=1)

        im_k_shuffled, idx_unshuffle = self._batch_shuffle_ddp(im_k)
        k = F.normalize(self.encoder_k(im_k_shuffled), dim=1)
        k = self._batch_unshuffle_ddp(k, idx_unshuffle)

        # 정렬: 양성 쌍의 평균 코사인 비슷함
        alignment = (q * k).sum(dim=1).mean().item()

        # 고름: 큐의 음성과의 평균 코사인 비슷함
        neg_sim = torch.mm(q, self.queue.clone().detach())
        uniformity = neg_sim.mean().item()

        return alignment, uniformity
    ```

    반환값은 양성 코사인 유사도 평균과 쿼리·큐 코사인 유사도 평균이다. 후자는 표현이 초구면에 고르게 퍼졌다는 사실을 충분히 검증하는 표준 uniformity 지표가 아니다. 평균이 0이어도 두 반대 방향에만 몰린 분포가 가능하다. 양성을 1로 만드는 것만으로 좋은 표현을 보장하지도 않는다. 평가 모드를 별도로 설정하지 않으면 no_grad에서도 BatchNorm 통계가 바뀔 수 있다.

## 정리하며

MoCo의 EMA·큐·온도와 배치 규약을 연결한다. 로컬 셔플·상태 변경·버퍼와 미완성 v3를 원래 알고리즘과 구분하며, 평균 유사도만으로 붕괴 방지·균일성·다운스트림 품질을 판정하지 않는다.
