# LightGCN 기반 추천 시스템

사용자–상품 상호작용을 이분 그래프로 나타내면, 연결된 상품과 사용자를 따라 정보를 전달해 협업 필터링 임베딩을 만들 수 있다. [LightGCN](https://arxiv.org/abs/2002.02126)은 피처 변환 행렬·비선형 활성화 없이 정규화 이웃 집계와 여러 레이어의 표현 결합을 사용한다. 여기서는 그 기본 계산과 쌍별 순위 로스를 살펴본다.

## 1. 코드와 출력

```python
"""29.7.4: 그래프 뉴럴 넷트웍 바탕 추천 시스템(LightGCN 방식)."""
import torch, torch.nn as nn, torch.nn.functional as F
import numpy as np

# ========================================================================
# 메인
# ========================================================================

class LightGCN(nn.Module):
    def __init__(self, n_users, n_items, embed_dim=32, num_layers=3):
        super().__init__()
        self.user_emb = nn.Embedding(n_users, embed_dim)
        self.item_emb = nn.Embedding(n_items, embed_dim)
        self.num_layers = num_layers
        nn.init.normal_(self.user_emb.weight, std=0.1)
        nn.init.normal_(self.item_emb.weight, std=0.1)

    def forward(self, edge_index, n_users, n_items):
        n = n_users + n_items
        x = torch.cat([self.user_emb.weight, self.item_emb.weight], dim=0)
        all_embs = [x]
        src, dst = edge_index[0], edge_index[1]
        deg = torch.zeros(n, device=x.device)
        deg.scatter_add_(0, dst, torch.ones(dst.shape[0], device=x.device))
        norm = (deg[src]*deg[dst]).pow(-0.5); norm[norm==float('inf')]=0
        for _ in range(self.num_layers):
            msg = x[src]*norm.unsqueeze(1)
            out = torch.zeros(n, x.shape[1], device=x.device)
            out.scatter_add_(0, dst.unsqueeze(1).expand_as(msg), msg)
            x = out; all_embs.append(x)
        x = torch.stack(all_embs, dim=0).mean(dim=0)
        return x[:n_users], x[n_users:]

    def bpr_loss(self, user_emb, item_emb, users, pos_items, neg_items):
        u = user_emb[users]; pi = item_emb[pos_items]; ni = item_emb[neg_items]
        pos_s = (u*pi).sum(dim=-1); neg_s = (u*ni).sum(dim=-1)
        return -F.logsigmoid(pos_s - neg_s).mean()

def demo():
    print("=" * 60); print("Recommendation (LightGCN)"); print("=" * 60)
    torch.manual_seed(42); np.random.seed(42)
    n_users, n_items = 50, 30
    # 주고받음 만들기
    interactions = []
    for u in range(n_users):
        n_int = np.random.randint(3, 8)
        items = np.random.choice(n_items, n_int, replace=False)
        for i in items: interactions.append((u, i))
    # 두 쪽 edge_index 짓기(물건 번호를 n_users만큼 밀기)
    src = [u for u,i in interactions]+[i+n_users for u,i in interactions]
    dst = [i+n_users for u,i in interactions]+[u for u,i in interactions]
    ei = torch.tensor([src, dst], dtype=torch.long)

    model = LightGCN(n_users, n_items, embed_dim=16, num_layers=2)
    opt = torch.optim.Adam(model.parameters(), lr=0.01)
    user_items = {}
    for u,i in interactions: user_items.setdefault(u, []).append(i)

    model.train()
    for epoch in range(50):
        opt.zero_grad()
        u_emb, i_emb = model(ei, n_users, n_items)
        users, pos, neg = [], [], []
        for u, items in user_items.items():
            for i in items:
                users.append(u); pos.append(i)
                ni = np.random.randint(n_items)
                while ni in items: ni = np.random.randint(n_items)
                neg.append(ni)
        loss = model.bpr_loss(u_emb, i_emb, torch.tensor(users), torch.tensor(pos), torch.tensor(neg))
        loss.backward(); opt.step()
        if (epoch+1) % 10 == 0: print(f"  Epoch {epoch+1}: BPR Loss = {loss.item():.4f}")

if __name__ == "__main__":
    demo()
```

```
============================================================
Recommendation (LightGCN)
============================================================
  Epoch 10: BPR Loss = 0.6685
  Epoch 20: BPR Loss = 0.6154
  Epoch 30: BPR Loss = 0.5304
  Epoch 40: BPR Loss = 0.4442
  Epoch 50: BPR Loss = 0.3596
```

기존 출력은 합성 상호작용의 BPR 학습 로스다. 이번 편집에서는 실행하지 않았다. 별도 테스트 추천 목록이나 Recall·NDCG는 계산하지 않으므로 실제 추천 성능의 검증 결과가 아니다.

## 2. 이분 그래프와 인덱스

사용자 50개, 상품 30개를 만든다. 각 사용자는 중복 없이 3~7개 상품을 무작위로 고른다. 전역 그래프에서 사용자는 0~49, 상품은 원래 상품 ID에 50을 더한 50~79를 사용한다. 사용자→상품과 상품→사용자를 모두 저장한다.

모델은 사용자·상품의 16차원 임베딩 테이블만 학습한다. 파라미터는 (50+30)×16=1,280개이며 레이어마다 추가되는 선형 웨이트가 없다. forward에 전달하는 사용자·상품 수와 테이블 크기·인덱스 offset이 일치해야 한다. 신규 사용자·상품은 테이블 밖에 있으므로 피처만으로 즉시 추론하는 모델은 아니다.

## 3. 전파와 레이어 결합

A를 무방향 이분 인접 행렬, D를 차수 행렬로 두면 다음과 같다.

$$E^{(k+1)}=D^{-1/2}AD^{-1/2}E^{(k)},\qquad
\overline E=\frac1{K+1}\sum_{k=0}^K E^{(k)}$$

이 코드에는 자기 루프·ReLU·노드별 Linear가 없다. 초기 테이블 E⁽⁰⁾도 결합에 포함한다. 데모는 K=2이므로 초기·1홉·2홉 표현 세 개의 평균이다. 이분 그래프에서 홀수 단계는 반대 유형, 짝수 단계는 같은 유형의 경로 정보를 받을 수 있다.

각 엣지의 양 끝 차수로 정규화하며 고립 노드는 전파 뒤 0이 된다. 그래도 초기 임베딩을 평균에 포함하므로 최종 표현은 E⁽⁰⁾/(K+1)로 남는다. 이것이 신규 정보가 없는 사용자의 유용한 선호를 학습해 준다는 뜻은 아니다.

## 4. BPR 목적과 음성 상품

사용자 u, 관측 상품 i, 미관측 상품 j에 대해 내적 점수 차이를 높인다.

$$s(u,i)=\overline e_u^\top\overline e_i,\qquad
\mathcal L_{\mathrm{rank}}=-\frac1B\sum_{(u,i,j)}\log\sigma(s(u,i)-s(u,j))$$

[BPR](https://arxiv.org/abs/1205.2618)은 암묵적 피드백의 개인화 순위를 학습하는 목적이다. 클릭·구매를 관측한 상품이 샘플링한 미관측 상품보다 높게 오도록 하지만, 미관측을 확정적인 싫어요로 해석해서는 안 된다. 노출 여부와 수집 바이어스가 다를 수 있다.

각 사용자–관측 상품마다 그 사용자의 관측 목록 밖에서 균등한 상품 하나를 뽑는다. 사용자마다 동일 수의 표본이 아니라 관측 상품이 많은 사용자에게 더 많은 로스 항이 생긴다. 이 구현은 순위 항만 사용하고 원래 테이블에 대한 L2 정규화 항은 없다. 임베딩 스케일과 과적합을 함께 확인해야 한다.

50번 반복마다 전체 그래프를 다시 전파하고 모든 관측 쌍에 새 음성을 붙여 한 번 업데이트한다. 따라서 총 50회 전체 배치 업데이트다. 한 사용자가 모든 상품과 연결되면 음성 선택 while이 끝나지 않지만, 현재 데모의 사용자당 최대 7개·전체 30개에서는 그 경우가 생기지 않는다.

## 5. 누출 없는 추천 평가

데모는 관측 상호작용을 모두 그래프와 학습 양성에 사용한다. 숨겨 둔 검증·테스트는 없고 무작위 상호작용에 안정적인 미래 선호 규칙을 넣지도 않았다. 학습 로스 감소를 새로운 상호작용 예측의 개선으로 읽지 않는다.

홀드아웃 평가를 하려면 먼저 상호작용을 나누고 **학습 상호작용만으로 전파 그래프**를 만들어야 한다. 무방향 역방향 레코드도 함께 제외한다. 각 질의에서 이미 관측한 상품을 추천 후보에서 빼는 규약, 후보 수·음성 샘플링·동점 처리·정답 수를 맞춰 Recall@K·NDCG@K 등을 계산한다.

전체 상품을 점수화하면 사용자 질의당 O(Id)가 들고, 전체 전파는 K레이어에서 O(Kmd)다. 레이어 표현을 모두 쌓는 이 코드는 O(KNd)의 결합용 저장량도 사용한다. 간단한 전파 구조가 모든 데이터·규모에서 일정한 속도나 품질 우위를 보장하지는 않는다.

## 연습문제

<div class="drillbox" markdown>

**연습문제 1.** <span class="diff med" title="중간"></span>
K=2에서 최종 표현에 포함되는 레이어와 파라미터 수를 구하라. 고립 노드의 최종 표현은 무엇인가?

</div>

??? success "연습문제 1 풀이"

    초기·1홉·2홉의 세 표현을 평균한다. 파라미터는 사용자·상품 테이블의 (50+30)×16=1,280개다. 고립 노드는 이후 전파가 0이므로 초기 임베딩의 1/3이 남는다. 자기 루프를 추가한 GCN과 다른 계산이다.



---

<div class="drillbox" markdown>

**연습문제 2.** <span class="diff med" title="중간"></span>
BPR 로스의 점수 차이가 0이면 항별 로스는 얼마인가? 미관측 상품을 실제 음성으로 단정할 수 있는가?

</div>

??? success "연습문제 2 풀이"

    −log σ(0)=log 2다. 관측 상품의 점수를 상대적으로 올리면 로스가 작아진다. 미관측은 싫어서 선택하지 않은 경우뿐 아니라 노출되지 않았거나 아직 선택하지 않은 경우도 있어 실제 음성으로 확정할 수 없다.



---

<div class="drillbox" markdown>

**연습문제 3.** <span class="diff hard" title="어려움"></span>
추천 홀드아웃을 구성할 때 그래프와 평가 후보를 어떻게 다뤄야 하는가?

</div>

??? success "연습문제 3 풀이"

    검증·테스트 상호작용과 역방향 레코드를 전파 그래프에서 제거하고 학습 연결만 사용한다. 후보에서 이미 관측한 상품을 제외하는 규약과 후보 수·음성 분포를 고정한다. 시간 예측은 미래 정보를 제외하고, 모델 선택용 검증과 최종 테스트를 분리한다. 현재 로스 출력은 이런 평가를 수행하지 않는다.



## 정리하며

LightGCN은 사용자·상품 ID 임베딩을 선형 이웃 전파하고 초기레이어를 포함해 결합한다. BPR 순위 항·음성 상품의 의미와 학습 그래프를 구분하며, 합성 학습 로스만으로 추천 일반화 성능을 판단하지 않는다.
