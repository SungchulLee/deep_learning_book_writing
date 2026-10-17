# 신약 개발과 약물–표적 상호작용 예측

약물–표적 상호작용 예측은 분자와 단백질 등의 표적 정보를 결합해 특정 실험의 상호작용 여부나 친화도를 예측하는 문제다. 이 페이지는 분자 그래프 인코더와 표적 피처 인코더를 합치는 기본 구조를 설명한다. 데모의 피처·레이블은 무작위 합성값이므로 실제 결합·약효·신약 후보의 성능을 검증한 실험은 아니다.

## 1. 코드와 출력

```python
"""29.7.2: 약 찾기 - 약과 타깃의 서로 작용 헤아리기."""
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
    demo()
```

```
============================================================
Drug-Target Interaction Prediction
============================================================
  Epoch 10: Loss = 0.6190
  Epoch 20: Loss = 0.4803
  Epoch 30: Loss = 0.3269
```

기존 출력은 30에포크 학습 중의 BCE 기록이다. 본문 편집에서는 코드를 실행하지 않았다. 테스트 정확도나 실제 약물 데이터를 평가한 지표는 기본 데모에 없다.

## 2. 두 인코더와 결합 방식

약물 입력은 노드별 10차원 피처와 엣지 인덱스다. 먼저 원자 피처를 32차원으로 투영하고, 자기 루프를 더한 이웃의 상태를 **정규화 없이 합산**한다. 두 번의 ReLU와 후속 Linear를 거친 노드 상태를 평균해 하나의 약물 벡터를 만든다. 이웃 합산은 한 번이며 세 레이어 GCN이나 결합 속성 모델은 아니다.

표적 입력은 길이 8의 벡터다. 별도의 8→32→32 MLP로 표적 벡터를 만든다. 두 벡터를 이어 붙여 64→32→1 predictor와 시그모이드에 넣는다. 기본 파라미터 수는 4,865개다.

이는 같은 공간의 거리나 내적으로 상호작용을 정의하는 모델이 아니다. 이어 붙인 표현의 비선형 결합을 학습하므로 상호작용이 있는 약물·표적 벡터가 반드시 가까워져야 하는 것은 아니다. 후보별 인코더 출력을 재사용할 수는 있지만 모든 약물–표적 쌍의 점수 계산 비용까지 사라지는 것은 아니다.

## 3. 합성 데이터의 한계

약물 50개는 5~11개 노드의 경로 그래프로 정리하며, 표적 10개는 무작위 8차원 벡터다. 최초 drugs 목록에서 만든 엣지 텐서는 나중에 버리고 drugs_clean의 경로 연결만 사용한다. 처음 텐서를 학습용 분자 그래프라고 해석하면 안 된다.

200개 상호작용은 약물·표적 ID를 무작위로 고르고, 피처와 독립적인 확률 0.5의 이진 레이블을 붙인다. 같은 쌍이 반복되거나 서로 다른 레이블을 받을 수 있다. 앞 160개를 매번 같은 순서로 30회 순회해 총 4,800번 업데이트한다.

독립적인 새 레이블을 예측하는 기대 정확도는 신호가 없으면 0.5다. 테스트 40개에서는 표본 변동이 있고, 반복 ID·쌍과 레이블 잡음도 해석해야 한다. 낮은 학습 로스가 실제 상호작용 규칙을 발견했다는 증거는 아니며, 실제 데이터라면 반드시 더 높은 정확도가 나온다고도 단정할 수 없다.

## 4. 평균 readout과 장치 조건

평균 readout은 동일한 노드 상태를 반복할 때 평균값을 유지하지만, 전체 모델의 분자 크기 불변성을 보장하지 않는다. 앞의 이웃 합산이 차수와 연결 구조의 영향을 받기 때문이다. 분자 크기와 함께 늘어나는 목표인지, 평균적·국소적 성질인지에 따라 sum·mean·크기 피처를 비교한다. sum을 사용해도 임베딩 노름이 반드시 분자 크기와 단조 관계인 것은 아니다.

현재 forward의 자기 루프와 출력 버퍼는 기본 CPU에서 생성한다. 모델·입력만 GPU로 옮겨 바로 동작하는 완성된 장치 독립 구현은 아니다. 여러 분자를 한 번에 처리하는 batch 인덱스도 없다. 입력 피처·장치·인덱스와 배치 계약을 확장 전에 확인해야 한다.

## 5. 평가와 결합 피처 확장

연습문제의 평가 조각은 학습에 사용하지 않은 뒤 40개 레코드에서 임계값 0.5 정확도를 계산한다. 하지만 이미 학습에서 본 약물·표적이 재등장할 수 있으므로 새 약물이나 새 표적에 대한 평가와 다르다. random pair, cold-drug, cold-target, 둘 다 새로운 경우를 구분하고 중복·유사성·관측 시점을 관리해야 한다.

결합 피처 변형은 각 엣지의 피처를 32차원 시그모이드 게이트로 바꿔 메시지에 좌표별로 곱한다. 기본 모델과 달리 이 변형은 자기 루프를 내부에서 더하지 않는다. 따라서 결합 정보만 추가한 공정한 단일 요인 비교가 아니며, 입력의 양방향 엣지와 결합 피처도 열별로 정렬해야 한다.

실제 분자·표적의 입력과 실험 레이블을 준비한 뒤 목표·분할·지표를 정해야 한다. [MoleculeNet](https://arxiv.org/abs/1703.00564)처럼 데이터·피처·평가 규약을 함께 명시하는 관점이 중요하다. 이 예제는 후보를 계산으로 점수화하는 구조이며 실험적 결합이나 약물 효과를 확인하는 절차를 대신하지 않는다.

## 연습문제

<div class="drillbox" markdown>

**연습문제 1.** <span class="diff med" title="중간"></span>
아래 평가 조각의 범위와 독립 무작위 레이블에서 기대할 수 있는 성능을 설명하라.

</div>

??? success "풀이"

    interactions[160:]의 40개 레코드를 임계값 0.5로 평가한다. model·interactions·drugs_clean·targets는 demo의 지역 변수이므로 조각을 그대로 함수 밖에 놓으면 사용할 수 없다. 같은 범위에 배치하거나 반환해야 하며 eval은 no_grad를 뜻하지 않는다. 피처와 독립적인 새 레이블의 기대 정확도는 0.5이고 유한 표본은 변동한다. 실제 데이터의 정확도는 이 데모로 예측할 수 없다.

    ```python
    model.eval()
    correct = 0
    for d_idx, t_idx, label in interactions[160:]:
        pred = model(drugs_clean[d_idx], targets[t_idx])
        correct += (pred.item() > 0.5) == label
    print(f"Test accuracy: {correct / 40:.4f}")
    ```


---

<div class="drillbox" markdown>

**연습문제 2.** <span class="diff med" title="중간"></span>
평균 readout이 전체 약물 인코더를 크기 불변으로 만드는지 설명하라.

</div>

??? success "풀이"

    같은 상태 벡터들의 단순 반복에는 평균이 같지만, 노드 상태는 정규화하지 않은 이웃 합산과 비선형을 거쳐 생성된다. 그래프 크기·차수가 바뀌면 평균할 벡터 자체가 달라질 수 있어 전체 모델의 불변성을 보장하지 않는다. 목표의 총량·평균 성격에 맞춰 readout과 크기 피처를 비교한다.



---

<div class="drillbox" markdown>

**연습문제 3.** <span class="diff hard" title="어려움"></span>
아래 결합 피처 모델에서 메시지의 의미와 기본 모델 대비 함께 바뀌는 조건을 설명하라.

</div>

??? success "풀이"

    결합 피처를 선형 투영·시그모이드로 변환한 좌표별 게이트를 출발 노드 메시지에 곱한다. 결합 피처의 의미·열 정렬을 제공해야 유형을 구분할 수 있다. 이 변형은 자기 루프도 추가하지 않아 기본 모델과 그 조건이 다르며 단순히 결합 정보만의 효과로 성능 차이를 해석할 수 없다. 새 모델의 학습·평가 코드와 데이터는 별도로 필요하다.

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


## 정리하며

약물·표적을 따로 인코딩해 결합하는 구조를 살펴보았다. 데모는 무작위 레이블의 합성 문제이며, 정규화 없는 이웃 합산·평균 readout·CPU 버퍼·결합 게이트의 실제 동작과 신규 약물·표적 평가의 차이를 구분해야 한다.
