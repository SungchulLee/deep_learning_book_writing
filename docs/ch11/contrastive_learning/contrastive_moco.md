# 대조 학습 MoCo

관성 대조(MoCo)는 유동적인 큐에 담아 둔 큰 음성 예 무리와 양성 쌍을 견주어 시각 표현을 배우는 자기 지도 학습 틀이다. MoCo는 대조 학습의 핵심 어려움, 곧 크고 한결같은 음성 예 집합을 지키는 일을 관성으로 갱신되는 부호기와 선입선출 큐로 푼다. 이 구조는 사전의 크기를 작은 배치 크기에서 떼어 놓아 GPU 기억이 넉넉하지 않아도 대조 학습이 잘 되게 한다.

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

    효율적인 대조 학습을 위해 관성 부호기와 큐 얼개를 쓴다.

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

        # 질의 부호기를 만든다
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

        # 열쇠 부호기(관성 부호기)를 만든다
        self.encoder_k = copy.deepcopy(self.encoder_q)

        # 열쇠 부호기를 얼린다. 관성으로 갱신된다
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
        열쇠 부호기의 관성 갱신
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
            # 관성 부호기를 갱신한다
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

        # 바탕 부호기
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
        관성 부호기 없이 대칭 손실을 쓴다
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

## 2. 논의

MoCo v2는 부호기 둘을 지킨다. 표준 역전파로 갱신되는 **질의 부호기**와, 관성 계수 $m$으로 질의 부호기의 지수 이동 평균으로 갱신되는 **열쇠 부호기**이다: $\theta_k \leftarrow m \cdot \theta_k + (1 - m) \cdot \theta_q$. 이 관성 갱신 덕분에 큐 안의 열쇠들이 조금씩 다른 부호기 상태에서 나왔는데도 얼추 한결같이 남는다. 큐는 어떤 작은 배치보다도 훨씬 클 수 있는 음성 예의 사전 노릇을 하며 대개 열쇠 65,536개를 담는다.

대조 목표는 같은 그림에서 나온 질의-열쇠 쌍을 양성으로, 큐의 모든 항목을 음성으로 다룬다. 교차 엔트로피 손실을 셈하기 전에 로짓에 온도 매개변수 $T$을 나누는데, 온도가 낮을수록 분포가 뾰족해져 모형이 더 잘 가려내는 표현을 내도록 북돋운다. 열쇠 부호기 앞에서 배치를 뒤섞으면 모형이 배치 정규화 통계를 지름길로 삼아 배운 표현이 무너지는 일을 막는다.

MoCo v3은 큐와 관성 부호기를 아주 없애고, 대신 사영 위에 예측 머리를 얹은 대칭 손실을 써서 구조를 간단하게 한다. BYOL과 SimSiam에서 영감을 얻은 이 설계가 통하는 것은 예측기와 목표 가지의 기울기 멈춤 사이의 비대칭이 표현이 무너지는 것을 막기 때문이다. v2에서 v3으로의 옮김은 드러난 음성 예보다 구조의 비대칭에 기대는 더 간단한 틀로 향하는, 자기 지도 학습의 더 넓은 흐름을 보여 준다.

## 연습문제

**연습문제 1.**
큐 크기가 $K = 65536$, 배치 크기가 $B = 256$, 특징 차원이 $d = 128$인 MoCo v2 모형에서 (32비트 부동소수점이라 하고) 큐가 차지하는 기억을 메가바이트로 셈하라. $224 \times 224 \times 3$ 해상도의 그림 $K$장을 통째로 담는 것과 견주면 어떠한가?

??? success "연습문제 1 풀이"
    큐는 float32 값 $d \times K = 128 \times 65536 = 8{,}388{,}608$개를 담는다. float32 하나가 4바이트이므로 큐는 $8{,}388{,}608 \times 4 = 33{,}554{,}432$바이트, 곧 약 32MB를 쓴다.

    $224 \times 224 \times 3$ 해상도의 그림 $K = 65536$장을 float32으로 담으면 $65536 \times 224 \times 224 \times 3 \times 4 \approx 39.3$GB가 든다. 큐는 날그림 대신 128차원 특징 벡터만 담아 $39{,}300 / 32 \approx 1{,}228$배로 눌러 담는다. 이 엄청난 줄임이 큰 사전을 가능케 한다.

---

**연습문제 2.**
관성 계수 $m$을 왜 대개 1에 아주 가깝게(이를테면 0.999로) 두는지 설명하라. $m = 0$이나 $m = 0.5$이면 어떻게 되는가? $m$은 큐에 있는 음성 예의 한결같음과 어떤 관계인가?

??? success "연습문제 2 풀이"
    $m = 0.999$이면 열쇠 부호기가 아주 천천히 갱신되어 단계마다 가중치의 0.1%만 질의 부호기 쪽으로 옮긴다. 그러면 서로 다른 학습 단계에서 나온 열쇠들이 거의 같은 부호기에서 만들어져 큐 전체에 걸쳐 한결같음이 지켜진다. 큐가 지난 여러 배치의 열쇠를 담으므로 한결같음이 매우 중요하다. 부호화 함수가 단계마다 빠르게 바뀌면 음성 예들이 서로 맞지 않는 특징 공간에서 와서 대조 신호의 질이 떨어진다.

    $m = 0$이면 열쇠 부호기가 단계마다 질의 부호기를 그대로 베껴 두 부호기가 같아진다. 그러면 관성의 이점이 사라지고 큐의 열쇠들이 아주 다른 부호기 상태에서 와서 음성 예가 들쭉날쭉해진다. $m = 0.5$이면 열쇠 부호기가 너무 빨리 바뀌어 여전히 큐 전체에 걸쳐 크게 들쭉날쭉해진다. 경험으로 보아 0.999쯤의 값이 부호기의 한결같음과 열쇠 부호기가 끝내 질의 부호기의 나아짐을 좇게 하는 것 사이의 균형을 잡는다.

---

**연습문제 3.**
학습 중에 양성 쌍 사이와 질의-음성 쌍 사이의 평균 코사인 비슷함을 좇아 기록하도록 `MoCo` 클래스를 고쳐라. 이를 두 지표를 함께 돌려주는 `compute_alignment_uniformity` 메서드로 구현하라. 학습이 나아감에 따라 어떤 값이 나올 법한지 설명하라.

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

    학습이 나아가면 **정렬**(양성 쌍의 비슷함)이 1.0 쪽으로 올라야 하는데, 같은 그림의 불린 시야가 가까운 점으로 잇대어진다는 뜻이다. **고름**(질의-음성 비슷함)은 0이나 살짝 음수 쪽으로 내려가야 하는데, 서로 다른 그림의 표현이 초구면에 고르게 퍼진다는 뜻이다. 잘 학습된 모형은 높은 정렬과 낮은 고름을 함께 보인다.

## 정리하며

**다룬 것** — 대조 학습 MoCo

MoCo v2는 부호기 둘을 지킨다.

핵심 클래스는 `MoCo`, `MoCoV3`이며 앞의 연습문제 3개로 직접 확인할 수 있다.
