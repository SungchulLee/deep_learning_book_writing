# 계층적 풀링의 두 구현

그래프를 줄이는 방식에는 노드 일부를 남기는 선택과 여러 노드를 새 클러스터에 배정하는 축약이 있다. 이 페이지는 TopKPool과 단순 soft assignment를 나란히 보여 준다. 원 논문의 전체 모델이나 학습 결과가 아니라 변환의 모양을 확인하는 교육용 구현이다.

## 1. 코드와 보존된 출력

```python
"""29.5.4: 켜진 모으기 - TopK과 DiffPool"""
import torch, torch.nn as nn, torch.nn.functional as F

# ========================================================================
# 메인
# ========================================================================

class TopKPool(nn.Module):
    """배운 점수로 위 k개 마디를 고른다."""
    def __init__(self, in_ch, ratio=0.5):
        super().__init__()
        self.score = nn.Linear(in_ch, 1)
        self.ratio = ratio
    def forward(self, x, edge_index):
        scores = self.score(x).squeeze(-1)
        k = max(1, int(self.ratio * x.shape[0]))
        _, idx = scores.topk(k)
        idx = idx.sort()[0]
        x_pool = x[idx] * torch.sigmoid(scores[idx]).unsqueeze(-1)
        # 변 거르기
        mask = torch.zeros(x.shape[0], dtype=torch.bool)
        mask[idx] = True
        node_map = torch.full((x.shape[0],), -1, dtype=torch.long)
        node_map[idx] = torch.arange(k)
        src, dst = edge_index[0], edge_index[1]
        edge_mask = mask[src] & mask[dst]
        new_src = node_map[src[edge_mask]]; new_dst = node_map[dst[edge_mask]]
        return x_pool, torch.stack([new_src, new_dst])

class SimpleDiffPool(nn.Module):
    """간단히 한 DiffPool: 부드러운 뭉치기."""
    def __init__(self, in_ch, n_clusters):
        super().__init__()
        self.assign = nn.Sequential(nn.Linear(in_ch, n_clusters), nn.Softmax(dim=-1))
    def forward(self, x, adj):
        S = self.assign(x)  # [n, k]
        x_pool = S.T @ x     # [k, d]
        adj_pool = S.T @ adj @ S  # [k, k]
        return x_pool, adj_pool, S

def demo():
    print("=" * 60); print("Hierarchical Pooling"); print("=" * 60)
    torch.manual_seed(42)
    x = torch.randn(10, 8)
    ei = torch.tensor([[0,1,1,2,2,3,3,4,5,6,6,7,7,8,8,9],[1,0,2,1,3,2,4,3,6,5,7,6,8,7,9,8]], dtype=torch.long)
    # 위 K개
    topk = TopKPool(8, ratio=0.5)
    x_new, ei_new = topk(x, ei)
    print(f"TopK: {x.shape[0]} -> {x_new.shape[0]} nodes, {ei.shape[1]} -> {ei_new.shape[1]} edges")
    # DiffPool
    adj = torch.zeros(10, 10); adj[ei[0], ei[1]] = 1
    dp = SimpleDiffPool(8, 3)
    x_p, adj_p, S = dp(x, adj)
    print(f"DiffPool: {x.shape[0]} -> {x_p.shape[0]} clusters")
    print(f"  Assignment (first 3 nodes): {S[:3].detach().round(decimals=2)}")

if __name__ == "__main__":
    demo()
```

```
============================================================
Hierarchical Pooling
============================================================
TopK: 10 -> 5 nodes, 16 -> 2 edges
DiffPool: 10 -> 3 clusters
  Assignment (first 3 nodes): tensor([[0.3800, 0.4400, 0.1800],
        [0.2600, 0.4000, 0.3400],
        [0.3300, 0.1900, 0.4800]])
```

데모는 임의의10×8 피처와 두 개의5노드 경로 성분을 사용한다. TopK는5개 노드를 선택하고 SimpleDiffPool은3개 클러스터를 만든다. 두 모듈 모두 무작위 초기화 상태이며 optimizer·학습 목표·예측 평가가 없다.

## 2. TopKPool의 선택 규칙

score는 각 노드 피처에 Linear(in_ch,1)를 적용한다. 인접 구조는 점수 계산에 들어가지 않는다. 원 논문·라이브러리의 정규화 투영이나 GNN 점수와 이름이 비슷해도 같은 식은 아니다.

선택 개수는 k=max(1,floor(ratio·n))이다. 양의 비율에서 ceil을 쓰는 구현과 다르고, ratio=0이나 음수여도 최소1개를 고른다. 빈 그래프에서는 k=1을 뽑을 수 없고 ratio>1이면 k>n으로 topk가 실패할 수 있다. 의도한 입력 범위를 명시해야 한다.

점수 상위 인덱스를 고른 뒤 원래 노드 인덱스 순서로 다시 정렬한다. 출력 피처는 선택한 x에 **sigmoid(score)**를 곱한 값이다. 노드의 선택 여부만 사용하는 것과 다르며, sigmoid를 써서0~1 게이트를 준다.

## 3. 엣지 필터와 재번호화

선택되지 않은 노드에 닿는 엣지를 모두 제거하고, 남은 노드의 기존 번호를0,…,k−1로 바꾼다. 이 결과는 선택 노드의 유도 부분 그래프다. 제거된 노드를 통과하던 경로를 새 엣지로 이어 주는 contraction은 아니다.

따라서 bridge가 제거되면 연결성이 깨질 수 있고, 기존 여러 성분도 그대로 일부만 남을 수 있다. 동점 점수에서 인덱스에 의존하는 선택은 순열 처리의 경계 사례다. hard 선택은 선택 경계에서 매끄럽지 않지만 선택 피처와 게이트 값에는 gradient가 흐를 수 있다.

mask와 node_map은 장치를 지정하지 않아 CPU에 만들어진다. x·edge_index만 CUDA로 옮기면 모든 인덱싱이 자동으로 맞지 않는다. 반환값에 원래 선택 idx나 그래프 소속을 포함하지 않으므로 추가 복원·배치 처리는 별도로 설계해야 한다.

## 4. SimpleDiffPool의 수식

각 노드 피처로 Linear→행별 softmax 배정 S∈R^(n×k)를 만든다. 반환값은 다음과 같다.

$$X'=S^TX,\qquad A'=S^TAS$$

S의 행합은1이며 각 노드는 여러 클러스터에 분수 질량으로 기여한다. X′는 배정 가중 합으로 클러스터 크기로 정규화한 평균이 아니다. A′는 밀집·가중 행렬이 되고 원래 자기 루프가 없어도 클러스터 대각에 값이 생길 수 있다.

예를 들어 연결된 두 노드를 하나의 클러스터에 완전히 배정하면 S=(1,1)ᵀ다. 양방향 A의 원소 합이2이므로 A′=[2]가 된다. 이는 원래 그래프의 방향 항들이 클러스터 내부 대각으로 모인 웨이트다.

## 5. 원 DiffPool과 다른 범위

[DiffPool 연구](https://arxiv.org/abs/1806.08804)는 임베딩 GNN과 배정 GNN, 작업 로스 및 보조 목적을 사용하는 계층 학습을 다룬다. 이 SimpleDiffPool은 배정에 **노드별 Linear만** 사용하고 X 자체를 합한다. 구조는 A′ 계산에는 쓰지만 S의 점수 계산에는 직접 쓰지 않는다.

연결 재구성·배정 엔트로피 로스, 배치별 마스크·계층 반복·예측 head는 없다. 따라서 출력의 배정값이 의미 있는 커뮤니티를 학습했다는 뜻은 아니다. 입력 X가 이미 구조를 반영한 임베딩인지에 따라서도 표현 범위가 달라진다.

## 6. 배치와 비용

두 함수 모두 여러 독립 그래프의 batch 벡터를 받지 않는다. 그래프를 단순히 이어 붙이면 TopK가 전체에서 k개를 고르고, SimpleDiffPool은 공통 클러스터로 서로 다른 그래프를 섞을 수 있다. 데모의 두 연결 성분은 하나의 입력 그래프로 처리한다는 점을 분명히 한다.

TopK 비용은 점수 계산·상위 선택·전체 엣지 필터링으로 나뉜다. SimpleDiffPool은 밀집 A를 받아 S 저장 O(nk), A 저장 O(n²)와 A′ 저장 O(k²)가 필요하다. 축약 후 노드 수가 줄어도 첫 밀집 입력의 비용이 사라지는 것은 아니다.

## 연습문제

<div class="drillbox" markdown>

**연습문제 1.** <span class="diff med" title="중간"></span>
이 구현에서 n=7, ratio=0.5와 ratio=0일 때 k는 얼마인가? ceil 규칙과 비교하라.

</div>

??? success "풀이"

    k=max(1,int(ratio·n))이므로 각각3과1이다. 양의0.5에서 ceil을 쓰면4가 되어 다르다. ratio=0도0개를 반환하지 않는다. 빈 입력이나 범위 밖 비율의 처리는 현재 검사되지 않는다.



---

<div class="drillbox" markdown>

**연습문제 2.** <span class="diff med" title="중간"></span>
경로0–1–2–3에서 노드0,2,3을 남기면 새로운 연결은 무엇인가?

</div>

??? success "풀이"

    기존0,2,3을 새 번호0,1,2로 바꾸고 양 끝이 남은2–3 엣지만 새1–2로 남는다. 새0은 고립된다. 제거된1을 통과하던0–2 경로가 새 직접 엣지로 바뀌지 않는다. 이는 유도 부분 그래프 선택이지 contraction이 아니다.



---

<div class="drillbox" markdown>

**연습문제 3.** <span class="diff hard" title="어려움"></span>
행합1인 S에서 축약 피처와 인접의 전체 합을 유도하고, SimpleDiffPool이 학습된 커뮤니티를 제공하는지 설명하라.

</div>

??? success "풀이"

    클러스터 전체 피처 합은1ᵀSᵀX=(S1)ᵀX=1ᵀX다. A′ 전체 합도(S1)ᵀA(S1)=1ᵀA1로 같다. 이 보존은 원소별 정보나 연결성 보존을 뜻하지 않는다. 모듈은 무작위 노드별 Linear 배정만 만들고 훈련·구조 보조 목적이 없어 의미 있는 커뮤니티 학습을 입증하지 않는다.



## 정리하며

TopK는 노드 선택·게이트·유도 엣지 재번호화를, SimpleDiffPool은 분수 배정의 가중 합을 구현한다. 두 변환의 구조 로스, 단일 그래프·장치 계약과 원 DiffPool의 학습 범위를 구분한다.
