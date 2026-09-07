# 개념 목 모형
## 들머리

**개념 목 모형(CBM)**은 설계로 풀이할 수 있게 만든다. 그물이 먼저 사람이 풀이할 수 있는 개념 모둠을 미루어 보게 하고, 그 개념만으로 마지막 미루어 봄을 하게 한다. 이렇게 하면 풀이되는 가운데 나타냄의 "목"이 생긴다.

일이 끝난 뒤 풀이하는 방법(SHAP, Grad-CAM)과 달리 CBM은 **본디부터 풀이된다**. 미루어 봄마다 또렷한 개념 살아남을 따라 되짚을 수 있다.

## 얼개

### 여느 CBM

```
들임 x → 개념 미루개 → [c₁, c₂, ..., cₖ] → 일감 미루개 → y
```

들임 $\mathbf{x}$이 주어지면 모형이 먼저 개념 값을 미루어 본다.

$$
\hat{c}_i = g_i(\mathbf{x}), \quad i = 1, \ldots, k
$$

그러고 나서 개념으로 마지막 미루어 봄을 한다.

$$
\hat{y} = h(\hat{c}_1, \hat{c}_2, \ldots, \hat{c}_k)
$$

### 익힘 목표

$$
\mathcal{L} = \underbrace{\mathcal{L}_{\text{task}}(\hat{y}, y)}_{\text{일감 잃음}} + \lambda \underbrace{\sum_{i=1}^{k} \mathcal{L}_{\text{concept}}(\hat{c}_i, c_i)}_{\text{개념 잃음}}
$$

여기서 $c_i$은 참값 개념 이름표다.

## PyTorch 짜보기

```python
import torch
import torch.nn as nn

class ConceptBottleneckModel(nn.Module):
    """
    개념 머리와 일감 머리를 따로 둔 개념 목 모형.
    """

    def __init__(
        self,
        backbone: nn.Module,
        backbone_dim: int,
        n_concepts: int,
        n_classes: int,
        concept_names: list = None
    ):
        super().__init__()
        self.backbone = backbone
        self.concept_names = concept_names or [f'c_{i}' for i in range(n_concepts)]

        # 개념 미루개
        self.concept_head = nn.Sequential(
            nn.Linear(backbone_dim, 256),
            nn.ReLU(),
            nn.Linear(256, n_concepts),
            nn.Sigmoid()
        )

        # 일감 미루개(개념만 쓴다)
        self.task_head = nn.Sequential(
            nn.Linear(n_concepts, 64),
            nn.ReLU(),
            nn.Linear(64, n_classes)
        )

    def forward(self, x, return_concepts=False):
        features = self.backbone(x)
        concepts = self.concept_head(features)
        output = self.task_head(concepts)

        if return_concepts:
            return output, concepts
        return output

    def explain(self, x):
        """사람이 읽을 수 있는 풀이를 만든다."""
        output, concepts = self.forward(x, return_concepts=True)

        concept_values = concepts[0].detach().cpu().numpy()
        prediction = output.argmax(dim=1).item()

        # 일감 머리의 짐이 개념 → 미루어 봄의 얽힘을 보인다
        task_weights = self.task_head[0].weight.data[prediction].cpu().numpy()
        contributions = concept_values * task_weights

        explanation = []
        sorted_idx = np.argsort(np.abs(contributions))[::-1]
        for idx in sorted_idx:
            explanation.append({
                'concept': self.concept_names[idx],
                'value': concept_values[idx],
                'contribution': contributions[idx]
            })

        return prediction, explanation

    def intervene(self, x, concept_idx, new_value):
        """
        되돌려 세워 따진다: 개념 값이 달랐다면 어땠을까?
        이는 CBM만이 지닌 나은 점이다.
        """
        _, concepts = self.forward(x, return_concepts=True)
        concepts_modified = concepts.clone()
        concepts_modified[0, concept_idx] = new_value
        return self.task_head(concepts_modified)
```

## 계량 금융에 쓰기

### 신용 점수 매기기 CBM

```python
# 개념: 빚 비율 높음, 벌이 든든함, 신용 자취 긺,
#       씀씀이 낮음, 요즘 밀린 적 없음
concept_names = [
    '높은 빚 비율', '든든한 벌이', '긴 신용 자취',
    '낮은 씀씀이', '요즘 밀린 적 없음', '고루 섞인 신용'
]

model = ConceptBottleneckModel(
    backbone=feature_extractor,
    backbone_dim=512,
    n_concepts=6,
    n_classes=2,
    concept_names=concept_names
)

# 판단을 풀이한다
pred, explanation = model.explain(applicant_features)
print(f"판단: {'받아들임' if pred == 0 else '물리침'}")
for item in explanation[:4]:
    print(f"  {item['concept']}: {item['value']:.2f} "
          f"(이바지: {item['contribution']:+.3f})")
```

## 간추림

개념 목 모형은 세움새 자체로 풀이할 수 있게 하며, 풀이와 되돌려 세운 손댐을 함께 준다. 맞바꿈은 익힐 때 개념 이름표가 있어야 하고, 개념 모둠이 모자라면 맞음이 떨어질 수 있다는 것이다.

## 살펴볼 거리

1. Koh, P. W., et al. (2020). "Concept Bottleneck Models." *ICML*.

2. Yuksekgonul, M., et al. (2022). "Post-hoc Concept Bottleneck Models." *ICLR*.

## 익힘 문제

**익힘 1.**
이 마디에서 밝힌 풀이 방법을, XOR 들임을 가르는 ReLU 살림의 두 켜 신경 그물에 걸어라. 들임 $x = [1, 1]$에 대한 풀이를 셈하여라.

??? success "익힘 1 풀이"
    짐이 $W_1, b_1, W_2, b_2$인 익힌 XOR 그물에서 내놓기는 $f(x) = W_2 \cdot \text{ReLU}(W_1 x + b_1) + b_2$이다. 풀이 방법은 들임 결마다 몫을 내놓는다. $x = [1, 1]$(갈래 0)이면 두 결 모두 음수 가름에 이바지한다. 몫 값은 방법마다 다르다. 기울기 바탕 방법은 $\partial f / \partial x_i$을 셈하고, 흔들어 보는 방법은 결을 가렸을 때 내놓기가 얼마나 바뀌는지 잰다. XOR 문제는 판단 금이 선형이 아니므로 선형 풀이 방법이 그르칠 수 있음을 보여 준다. $\square$

---

**익힘 2.**
이 마디의 풀이 방법이 온전함 공리를 채우는지, 곧 어떤 밑금 $x_0$에 대해 모든 결 몫의 합이 $f(x) - f(x_0)$과 같은지 증명하거나 뒤집어라.

??? success "익힘 2 풀이"
    온전함 공리(섀플리 값 이론에서는 효율이라고도 한다)는 몫의 합이 들임에서의 모형 내놓기와 밑금에서의 내놓기의 차이와 같다는 것이다. 이 방법이 온전함을 채우는지는 그 세움새에 달렸다. 기울기 방법은 온전함을 채우지 못한다(기울기는 그 자리의 것이고 길을 따라 쌓은 것이 아니다). 쌓은 기울기는 세움새 자체로 온전함을 채운다(길을 따라 미적분의 밑정리를 쓴다). SHAP 값은 섀플리 공리로 효율을 채운다. 온전함을 어기는 방법은 몫을 너무 많거나 적게 매길 수 있어, 온 몫을 온 세상 풀이로 믿기 어렵게 만든다. $\square$

---

**익힘 3.**
이 방법이 내놓는 풀이가 얼마나 미더운지 따지는 시험을 꾸며라. 짚어 준 결이 참으로 모형에 중요한지를 넣기와 빼기 곡선으로 재어라.

??? success "익힘 3 풀이"
    절차는 이렇다. (1) 시험 그림마다 결 몫을 셈한다. (2) 빼기: 몫이 큰 차례로 결을 하나씩 가리며 모형의 자신함이 떨어지는 모습을 적는다. 미더운 풀이면 자신함이 빠르게 떨어진다. (3) 넣기: 빈 밑금에서 시작해 몫이 큰 차례로 결을 하나씩 드러내며 자신함이 오르는 모습을 적는다. 미더운 풀이면 자신함이 빠르게 오른다. (4) 두 곡선의 아래 넓이를 셈한다. (5) 아무렇게나 매긴 차례(밑금)와 다른 방법에 견준다. 미더운 방법이면 빼기 넓이가 작고 넣기 넓이가 커야 한다. 통계로 미더우려면 시험 표본 1000개 넘게 되풀이한다. $\square$

---

**익힘 4.**
이 풀이 방법을 신용 부도를 미루어 보는 금융 모형에 어떻게 걸 수 있는지 다루어라. 풀이가 채워야 할 규정 요건은 무엇인가?

??? success "익힘 4 풀이"
    신용 모형에는 규정(ECOA, GDPR 22조)이 불리한 판단마다 그 사람에게 맞춘 풀이를 바란다. 방법은 다음을 내놓아야 한다. (1) 물리침에 가장 크게 이바지한 인자(불리한 처분 까닭). (2) 한결같은 풀이(비슷한 신청자는 비슷한 풀이를 받는다). (3) 손에 잡히는 풀이(신청자가 무엇을 바꾸어야 하는지 안다). 이 마디의 풀이 방법으로 결의 중요함을 짚을 수 있으나, 든든함(들임이 조금 바뀌었다고 풀이가 확 달라지면 안 된다)과 옳음(중요한 결을 없애면 미루어 봄이 바뀌어야 한다)을 따져 보아야 한다. 지켜야 할 됨됨이는 대리 차별이 드러나지 않도록 조심히 다루어야 한다. $\square$
