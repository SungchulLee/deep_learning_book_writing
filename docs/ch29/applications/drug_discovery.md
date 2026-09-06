# 약 찾기

약 찾기는 그래프 신경망의 주요 쓰임새 마당이다. 분자 그래프가 원자를 마디로, 화학 결합을 변으로 자연스럽게 나타낸다. 그래프 신경망 바탕 모델은 분자 얼개와 단백질 과녁의 결합 나타냄을 배워 약과 과녁의 주고받음을 헤아릴 수 있다. 이 방식은 방대한 화학 서고를 셈으로 걸러 약 개발의 이른 단계를 크게 빠르게 할 수 있다.

## 코드

```python
"""29.7.2: 약 찾기 - 약과 과녁의 서로 작용 헤아리기."""
import torch, torch.nn as nn, torch.nn.functional as F
import numpy as np

# ========================================================================
# 메인
# ========================================================================

class DrugTargetPredictor(nn.Module):
    def __init__(self, drug_feat=10, target_feat=8, hidden=32):
        super().__init__()
        self.drug_enc = nn.Sequential(nn.Linear(drug_feat, hidden), nn.ReLU(), nn.Linear(hidden, hidden))
        self.target_enc = nn.Sequential(nn.Linear(target_feat, hidden), nn.ReLU(), nn.Linear(hidden, hidden))
        self.predictor = nn.Sequential(nn.Linear(hidden*2, hidden), nn.ReLU(), nn.Linear(hidden, 1))

    def forward(self, drug_graph, target_feat):
        x, ei = drug_graph
        n = x.shape[0]; src, dst = ei[0], ei[1]
        loop = torch.arange(n); sa = torch.cat([src, loop]); da = torch.cat([dst, loop])
        h = self.drug_enc[0](x)
        out = torch.zeros(n, h.shape[1]); out.scatter_add_(0, da.unsqueeze(1).expand_as(h[sa]), h[sa])
        drug_emb = F.relu(self.drug_enc[2](F.relu(out))).mean(dim=0, keepdim=True)
        target_emb = self.target_enc(target_feat.unsqueeze(0))
        return torch.sigmoid(self.predictor(torch.cat([drug_emb, target_emb], dim=-1)))

def demo():
    print("=" * 60); print("Drug-Target Interaction Prediction"); print("=" * 60)
    torch.manual_seed(42); np.random.seed(42)
    n_drugs, n_targets = 50, 10
    drugs = [(torch.randn(np.random.randint(5,12), 10),
              torch.tensor([[i,i+1,i+1,i] for i in range(np.random.randint(4,11))], dtype=torch.long).T.contiguous()
              if np.random.randint(4,11) > 0 else torch.tensor([[0,1],[1,0]], dtype=torch.long))
             for _ in range(n_drugs)]
    # 변 번호 바로잡기
    drugs_clean = []
    for x, _ in drugs:
        n = x.shape[0]; edges = [(i,(i+1)%n) for i in range(n-1)]
        src = [e[0] for e in edges]+[e[1] for e in edges]
        dst = [e[1] for e in edges]+[e[0] for e in edges]
        drugs_clean.append((x, torch.tensor([src, dst], dtype=torch.long)))
    targets = [torch.randn(8) for _ in range(n_targets)]
    interactions = [(np.random.randint(n_drugs), np.random.randint(n_targets), float(np.random.random()>0.5)) for _ in range(200)]

    model = DrugTargetPredictor(); opt = torch.optim.Adam(model.parameters(), lr=0.001)
    model.train()
    for epoch in range(30):
        total_loss = 0
        for d_idx, t_idx, label in interactions[:160]:
            opt.zero_grad()
            pred = model(drugs_clean[d_idx], targets[t_idx])
            loss = F.binary_cross_entropy(pred.squeeze(), torch.tensor(label))
            loss.backward(); opt.step(); total_loss += loss.item()
        if (epoch+1) % 10 == 0:
            print(f"  Epoch {epoch+1}: Loss = {total_loss/160:.4f}")

if __name__ == "__main__":
    demo()```

## 논의

`DrugTargetPredictor` 모델은 약 분자와 단백질 과녁을 따로 부호화한 뒤 합쳐 주고받음을 헤아리는 두 갈래 얼개를 보인다. 약 부호기는 그래프 신경망 같은 모으기로 분자 그래프 특징을 다루고, 과녁 부호기는 앞먹임 신경망으로 단백질 특징 벡터를 다룬다. 이어 붙인 박아 넣기는 주고받음 확률을 내놓는 헤아리개 신경망을 지난다.

익히기 절차는 알려진 약-과녁 주고받음에 이진 교차 엔트로피 손실을 써서 이 문제를 이진 가름으로 다룬다. 모델은 주고받는 짝이 가까이 놓이는 함께 쓰는 숨은 공간에 약과 과녁을 박아 넣는 법을 배운다. 새 약이나 과녁을 따로 부호화해 다시 익히지 않고도 기존 서고와 견줄 수 있어 이 방식은 키울 수 있다.

실제로 최고 수준의 약-과녁 주고받음 모델은 3차원 분자 기하, 결합 갈래, 원자 성질을 담는 더 정교한 분자 부호기(쪽지 건네기 신경망이나 SchNet 같은 것)를 쓴다. 또한 단백질 부호기에 눈길 얼개를 쓰고, 얼개가 비슷한 분자가 생물 활성에서 크게 다른 활성 절벽을 조심스레 다루며 더 큰 자료 묶음을 쓴다.

## 연습문제

**연습문제 1.**
만든 자료 묶음으로 `DrugTargetPredictor`을 익히고 30바퀴 뒤의 마지막 익히기 손실을 적어라. 그다음 남겨 둔 주고받음(번호 160~200)에서 문턱 0.5으로 이진 정확도를 셈해 따져라.

??? success "연습문제 1 풀이"
    보여 주기를 돌리면 10, 20, 30바퀴의 익히기 손실이 찍힌다. 따지려면 시험 묶음에서 헤아림을 셈한다:
    ```python
    model.eval()
    correct = 0
    for d_idx, t_idx, label in interactions[160:]:
        pred = model(drugs_clean[d_idx], targets[t_idx])
        correct += (pred.item() > 0.5) == label
    print(f"Test accuracy: {correct / 40:.4f}")
    ```
    아무렇게나 만든 자료에서는 이름표도 아무거나이므로 정확도가 보통 50~60% 언저리이다. 얼개가 있는 자료(실제 약-과녁 주고받음)에서는 정확도가 훨씬 높을 것이다.

---

**연습문제 2.**
원자 켜 특징에서 약 박아 넣기를 얻을 때 합 모으기가 아니라 평균 모으기를 쓰는 까닭을 밝혀라. 어떤 상황에서 합 모으기가 나은가?

??? success "연습문제 2 풀이"
    평균 모으기는 크기에 불변인 나타냄을 만든다. 원자 5개짜리 분자와 15개짜리 분자가 비슷한 크기의 박아 넣기를 낸다. 뒤따르는 헤아리개(여러 층 신경망)가 들임의 잣수에 민감할 때 이는 중요하다. 반면 합 모으기는 크기 앎을 담는다. 큰 분자가 큰 박아 넣기를 낸다. 분자 크기가 과녁 성질을 헤아리는 데 쓸모 있을 때(보기로 분자량이 어떤 약물 동태 성질과 얽힐 때)나 잣수 차이를 다루는 고르게 맞추기 층과 함께 쓸 때 합 모으기가 낫다.

---

**연습문제 3.**
쪽지 건네기 걸음에 변 특징(결합 갈래)을 담도록 모델을 넓혀라. 변마다 결합 갈래 특징 벡터를 더하고 모으는 동안 그 특징을 담도록 약 부호기를 고쳐라.

??? success "연습문제 3 풀이"
    ```python
    class DrugTargetPredictorWithBonds(nn.Module):
        def __init__(self, drug_feat=10, bond_feat=4, target_feat=8, hidden=32):
            super().__init__()
            self.edge_enc = nn.Linear(bond_feat, hidden)
            self.drug_enc = nn.Sequential(
                nn.Linear(drug_feat, hidden), nn.ReLU(), nn.Linear(hidden, hidden))
            self.target_enc = nn.Sequential(
                nn.Linear(target_feat, hidden), nn.ReLU(), nn.Linear(hidden, hidden))
            self.predictor = nn.Sequential(
                nn.Linear(hidden*2, hidden), nn.ReLU(), nn.Linear(hidden, 1))

        def forward(self, drug_graph, target_feat):
            x, ei, bond_feats = drug_graph
            h = self.drug_enc[0](x)
            edge_weights = torch.sigmoid(self.edge_enc(bond_feats))
            # Weighted message passing using edge features
            msg = h[ei[0]] * edge_weights
            out = torch.zeros_like(h)
            out.scatter_add_(0, ei[1].unsqueeze(1).expand_as(msg), msg)
            drug_emb = F.relu(self.drug_enc[2](F.relu(out))).mean(0, keepdim=True)
            target_emb = self.target_enc(target_feat.unsqueeze(0))
            return torch.sigmoid(self.predictor(torch.cat([drug_emb, target_emb], -1)))
    ```
    이 고침은 모델이 홑 결합, 겹 결합, 방향족 결합을 가르게 하며 이는 분자 성질을 정확히 헤아리는 데 결정적이다.
