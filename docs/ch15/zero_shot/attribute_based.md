# 속성 기반 제로샷 학습

속성 기반 제로샷 학습은 클래스의 색·형태·기능 등 속성 정보를 사용해 학습 클래스와 미학습 클래스를 연결한다. DAP는 이미지에서 속성을 직접 예측하고, IAP는 학습 클래스의 예측을 통해 속성 확률을 간접적으로 구한다. 아래 구현에는 원래 방법을 단순화하거나 다른 점수로 대체한 부분이 있으므로 수식과 구분해 설명한다.

## 1. 속성 표현

### 속성의 정의

속성은 여러 클래스에 공통으로 적용할 수 있는 성질이다. 클래스별 대표 속성을 정의할 수도 있고 이미지별 실제 속성을 표시할 수도 있다. 클래스 속성이 모든 이미지에서 관측되는 것은 아니므로 두 수준의 라벨을 구분한다.

$$\mathbf{a}_c = [a_c^{(1)}, a_c^{(2)}, \ldots, a_c^{(M)}]$$

여기서 $M$은 속성의 개수이고 $a_c^{(m)}$은 클래스 $c$에서 속성 $m$의 있음과 세기를 나타낸다.

### 속성의 종류

**이진 속성**: $a_c^{(m)} \in \{0, 1\}$

- 성질이 있는지 없는지
- 보기: "has_stripes", "can_fly", "has_4_legs"

**연속 속성**: $a_c^{(m)} \in [0, 1]$ 또는 $\mathbb{R}$

- 성질의 정도나 세기
- 보기: "size"가 코끼리는 0.9, 생쥐는 0.1

**상대 속성**: 비교 값

- 보기: "larger_than_cat", "faster_than_tortoise"

### 속성 행렬

속성 정보 전체는 속성 행렬에 담긴다.

$$A \in \mathbb{R}^{C \times M}$$

여기서 $C = |\mathcal{Y}^s \cup \mathcal{Y}^u|$은 클래스의 총 개수이다.

**속성 행렬 예시:**

| 클래스 | has_fur | has_feathers | has_scales | 4_legs | can_fly | is_large | carnivore |
|-------|---------|--------------|------------|--------|---------|----------|-----------|
| 개 | 1.0 | 0.0 | 0.0 | 1.0 | 0.0 | 0.3 | 0.8 |
| 고양이 | 1.0 | 0.0 | 0.0 | 1.0 | 0.0 | 0.1 | 1.0 |
| 새 | 0.0 | 1.0 | 0.0 | 0.0 | 1.0 | 0.1 | 0.3 |
| 말 | 1.0 | 0.0 | 0.0 | 1.0 | 0.0 | 0.9 | 0.0 |
| 독수리 | 0.0 | 1.0 | 0.0 | 0.0 | 1.0 | 0.6 | 1.0 |

---

## 2. 직접 속성 예측(DAP)

### 개념

DAP는 이미지 특징에서 각 속성이 존재할 확률을 예측한다. 여러 속성의 확률을 곱해 클래스 점수를 계산하는 과정에서 조건부 독립 근사를 사용한다. 인코더를 공유하는 여러 출력 헤드를 공동 학습할 수도 있으므로 속성마다 별도 모델을 학습해야 하는 것은 아니다.

### 속성 예측과 클래스 점수

속성 m이 존재할 예측 확률을 q_m(x)=P(a_m=1|x)라 하자. 이진 클래스 속성 a_c에서 아래 단순화 점수를 만들 수 있다.

$$S(c,x)=\sum_m[a_{cm}\log q_m(x)+(1-a_{cm})\log(1-q_m(x))]$$

이는 코드가 사용하는 로그 점수다. 연속값 속성에 같은 식을 적용하면 소프트 타깃 교차 엔트로피에 대응하지만, 이진 속성 사건의 확률이라는 해석과는 다르다.

원 DAP의 베이즈 점수에는 속성 사전확률로 나누는 항이 포함된다. 클래스 속성을 고정된 이진 서명으로 보고 속성 분포를 곱으로 근사하면 다음 형태가 된다.

$$S_{\mathrm{DAP}}(c,x)=\log P(c)+\sum_m\log\frac{P(a_m=a_{cm}\mid x)}{P(a_m=a_{cm})}$$

후보 클래스 사이에 같은 항은 argmax에서 생략할 수 있지만 속성 사전확률 항이 항상 같지는 않다. 아래 코드는 이 보정을 생략하고 균등 클래스 사전확률을 가정한 간단한 비교를 수행한다. 예측 확률이 0 또는 1에 가까우면 로그의 수치 안정성도 고려해야 한다.

### PyTorch 구현

forward는 속성 확률을 반환하고 predict_class는 class_names의 행 인덱스를 반환한다. 속성 행렬과 학습 타깃을 CPU에 생성하므로 GPU 사용 시 장치를 맞춰야 한다. class_attributes의 키는 batch_y의 원본 라벨과 일치해야 하며, 텐서 스칼라를 사전 키로 그대로 사용하는 방식도 확인한다. 예측 메서드가 eval이나 no_grad를 자동 설정하지 않으므로 평가 모드는 호출자가 지정한다.

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class DirectAttributePrediction(nn.Module):
    """
    영 예시 학습을 위한 곧은 속성 예측(DAP).
    
    서로 독립인 속성 가려내개를 배우고, 속성 맞추기로
    못 본 부류를 맞히는 데 쓴다.
    """
    
    def __init__(self, visual_dim: int, n_attributes: int, hidden_dim: int = 512):
        super().__init__()
        
        self.n_attributes = n_attributes
        
        # 함께 쓰는 특징 뽑개
        self.feature_extractor = nn.Sequential(
            nn.Linear(visual_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.BatchNorm1d(hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.3)
        )
        
        # 속성 예측기(속성마다 하나)
        self.attribute_heads = nn.ModuleList([
            nn.Linear(hidden_dim // 2, 1) for _ in range(n_attributes)
        ])
    
    def forward(self, x):
        """
        속성 확률을 맞힌다.
        
        인수:
            x: 꼴이 (batch_size, visual_dim)인 시각 특징
        
        반환값:
            꼴이 (batch_size, n_attributes)인 속성 확률
        """
        features = self.feature_extractor(x)
        
        # 속성마다 맞힌다
        attr_logits = [head(features) for head in self.attribute_heads]
        attr_logits = torch.cat(attr_logits, dim=1)  # (batch, n_attributes)
        
        return torch.sigmoid(attr_logits)
    
    def predict_class(self, x, class_attributes, class_names):
        """
        DAP으로 부류를 맞힌다.
        
        인수:
            x: 시각 특징 (batch_size, visual_dim)
            class_attributes: 부류 이름을 속성 벡터로 옮기는 사전
            class_names: 후보 부류 이름 목록
        
        반환값:
            맞힌 부류 첨자
        """
        # 속성 확률을 얻는다
        attr_probs = self.forward(x)  # (batch, n_attributes)
        
        # 후보 부류의 속성 행렬을 만든다
        attr_matrix = torch.stack([
            torch.tensor(class_attributes[c], dtype=torch.float32)
            for c in class_names
        ])  # (n_classes, n_attributes)
        
        # 부류마다 로그 확률을 셈한다
        # P(c|x) ∝ ∏_m P(a_m|x)^a_c^m * (1-P(a_m|x))^(1-a_c^m)
        log_probs = attr_probs.unsqueeze(1)  # (batch, 1, n_attrs)
        attr_matrix = attr_matrix.unsqueeze(0)  # (1, n_classes, n_attrs)
        
        eps = 1e-7  # 수치 안정성
        log_scores = (
            attr_matrix * torch.log(log_probs + eps) +
            (1 - attr_matrix) * torch.log(1 - log_probs + eps)
        )
        class_scores = log_scores.sum(dim=2)  # (batch, n_classes)
        
        return torch.argmax(class_scores, dim=1)

def train_dap_model(model, train_loader, class_attributes, epochs=50, lr=0.001):
    """
    이진 교차 엔트로피 손실로 DAP 모델을 익힌다.
    """
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.BCELoss()
    
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        
        for batch_x, batch_y in train_loader:
            # 표본마다 목표 속성을 얻는다
            batch_attrs = torch.stack([
                torch.tensor(class_attributes[y], dtype=torch.float32)
                for y in batch_y
            ])
            
            # 순전파
            pred_attrs = model(batch_x)
            loss = criterion(pred_attrs, batch_attrs)
            
            # 역전파
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1}/{epochs}, Loss: {total_loss/len(train_loader):.4f}")
```

---

## 3. 간접 속성 예측 IAP

### 원래 IAP의 경로

IAP는 먼저 학습 클래스 확률 P(y|x)를 예측한다. 그다음 클래스별 속성 정보를 이용해 속성 확률을 추정한다.

$$P(a_m=1\mid x)=\sum_{y\in\mathcal Y^s}P(a_m=1\mid y)P(y\mid x)$$

얻은 속성 확률은 DAP와 같은 클래스 속성 점수 계산에 사용할 수 있다. 따라서 원래 IAP를 속성 독립 가정이 전혀 없는 방법으로 설명하는 것은 부정확하다.

### 아래 코드의 단순화

아래 IndirectAttributePrediction은 속성별 확률을 계산하지 않고 학습·미학습 클래스의 속성 유사도로 클래스 점수를 직접 전달한다.

$$S(u,x)=\sum_{s\in\mathcal Y^s}P(s\mid x)\operatorname{sim}(a_s,a_u)$$

코사인 또는 음의 제곱 거리의 지수를 유사도로 사용한다. 결과는 후보 클래스에 대해 정규화된 확률이 아니며, 원 IAP와 다른 전이 점수 방식이다. 이름과 달리 구현 범위는 이 식으로 이해한다.

### PyTorch 구현

seen_classes 순서는 학습 분류기의 출력 열과 일치해야 하고 반환 인덱스는 unseen_classes의 위치를 뜻한다. temperature는 양수여야 한다. 속성 행렬·라벨의 키와 장치, 평가 모드에 관한 조건은 DAP 예제와 같다.

```python
class IndirectAttributePrediction(nn.Module):
    """
    영 예시 학습을 위한 에두른 속성 예측(IAP).
    
    본 부류 가려내개를 배우고 속성 닮음으로
    못 본 부류로 옮긴다.
    """
    
    def __init__(self, visual_dim: int, n_seen_classes: int, hidden_dim: int = 512):
        super().__init__()
        
        self.classifier = nn.Sequential(
            nn.Linear(visual_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.BatchNorm1d(hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim // 2, n_seen_classes)
        )
    
    def forward(self, x):
        """
        본 부류의 로짓을 셈한다.
        """
        return self.classifier(x)
    
    def predict_class(self, x, seen_classes, unseen_classes, class_attributes,
                     similarity='cosine', temperature=1.0):
        """
        IAP으로 못 본 부류를 맞힌다.
        
        인수:
            x: 시각 특징 (batch_size, visual_dim)
            seen_classes: 본 부류 이름 목록
            unseen_classes: 못 본 부류 이름 목록
            class_attributes: 부류 이름을 속성 벡터로 옮기는 사전
            similarity: 'cosine' 또는 'euclidean'
            temperature: 소프트맥스 온도
        """
        # 본 부류의 확률을 얻는다
        logits = self.forward(x)
        seen_probs = F.softmax(logits / temperature, dim=1)  # (batch, n_seen)
        
        # 속성 행렬을 만든다
        seen_attrs = torch.stack([
            torch.tensor(class_attributes[c], dtype=torch.float32)
            for c in seen_classes
        ])  # (n_seen, n_attrs)
        
        unseen_attrs = torch.stack([
            torch.tensor(class_attributes[c], dtype=torch.float32)
            for c in unseen_classes
        ])  # (n_unseen, n_attrs)
        
        # 비슷함 행렬을 셈한다
        if similarity == 'cosine':
            seen_norm = F.normalize(seen_attrs, dim=1)
            unseen_norm = F.normalize(unseen_attrs, dim=1)
            sim_matrix = seen_norm @ unseen_norm.T  # (n_seen, n_unseen)
        else:  # 유클리드
            # 거리 제곱의 음수
            diff = seen_attrs.unsqueeze(1) - unseen_attrs.unsqueeze(0)
            sim_matrix = -torch.sum(diff ** 2, dim=2)
            sim_matrix = torch.exp(sim_matrix)  # 닮음으로 바꾼다
        
        # 확률을 옮긴다: P(unseen|x) = sum_s P(seen_s|x) * sim(seen_s, unseen)
        unseen_scores = seen_probs @ sim_matrix  # (batch, n_unseen)
        
        return torch.argmax(unseen_scores, dim=1)

def train_iap_model(model, train_loader, seen_classes, epochs=50, lr=0.001):
    """
    본 부류에서 교차 엔트로피 손실로 IAP 모델을 익힌다.
    """
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    
    # 이름표 대응을 만든다
    class_to_idx = {c: i for i, c in enumerate(seen_classes)}
    
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        correct = 0
        total = 0
        
        for batch_x, batch_y in train_loader:
            # 이름표를 첨자로 바꾼다
            batch_labels = torch.tensor([class_to_idx[y] for y in batch_y])
            
            # 순전파
            logits = model(batch_x)
            loss = criterion(logits, batch_labels)
            
            # 역전파
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            _, predicted = logits.max(1)
            correct += (predicted == batch_labels).sum().item()
            total += batch_labels.size(0)
        
        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1}/{epochs}, "
                  f"Loss: {total_loss/len(train_loader):.4f}, "
                  f"Acc: {100*correct/total:.2f}%")
```

---

## 4. DAP와 IAP 비교

| 항목 | 원 DAP | 원 IAP |
|---|---|---|
| 첫 예측 목표 | 이미지의 속성 | 학습 클래스 |
| 속성 확률 | 직접 예측 | 클래스 확률과 클래스 속성을 합산 |
| 미학습 클래스 비교 | 속성 서명과 비교 | 추정한 속성 확률을 서명과 비교 |
| 주요 오차 경로 | 속성 추정·독립 근사 | 클래스 예측·속성 전이·독립 근사 |

두 방법은 속성이 시각적으로 예측 가능하고 미학습 클래스 구분에 도움이 되는지에 영향을 받는다. 아래 점수 전이 구현의 결과를 원 IAP의 성능이라고 보고하지 않는다.

### 실험 성능

compare_dap_iap는 y_train 대신 0부터 표본 수−1까지를 라벨로 넣는다. 따라서 클래스 라벨과 속성 사전이 맞지 않아 실제 비교에 사용할 수 없다. 학습 후 eval도 호출하지 않아 드롭아웃·BatchNorm이 학습 상태로 남는다. 평가 정답은 미학습 클래스의 로컬 인덱스와 일치시켜야 한다. 이 조각은 검증된 벤치마크 결과를 제공하지 않는다.

```python
def compare_dap_iap(X_train, y_train, X_test, y_test, 
                    seen_classes, unseen_classes, class_attributes):
    """
    같은 데이터셋에서 DAP과 IAP을 견준다.
    """
    from torch.utils.data import TensorDataset, DataLoader
    
    # 데이터 로더 생성
    train_dataset = TensorDataset(
        torch.tensor(X_train, dtype=torch.float32),
        torch.tensor(range(len(y_train)))
    )
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    
    # 모델을 초기화한다
    n_attributes = len(class_attributes[seen_classes[0]])
    dap_model = DirectAttributePrediction(X_train.shape[1], n_attributes)
    iap_model = IndirectAttributePrediction(X_train.shape[1], len(seen_classes))
    
    # 두 모델을 모두 익힌다
    print("Training DAP...")
    train_dap_model(dap_model, train_loader, class_attributes, epochs=50)
    
    print("\nTraining IAP...")
    train_iap_model(iap_model, train_loader, seen_classes, epochs=50)
    
    # 평가한다
    X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
    
    dap_preds = dap_model.predict_class(X_test_tensor, class_attributes, unseen_classes)
    iap_preds = iap_model.predict_class(X_test_tensor, seen_classes, unseen_classes, 
                                        class_attributes)
    
    dap_acc = (dap_preds.numpy() == y_test).mean()
    iap_acc = (iap_preds.numpy() == y_test).mean()
    
    print(f"\nResults:")
    print(f"DAP Accuracy: {dap_acc*100:.2f}%")
    print(f"IAP Accuracy: {iap_acc*100:.2f}%")
    print(f"Random Baseline: {100/len(unseen_classes):.2f}%")
```

---

## 5. 속성의 질과 고르기

### 속성 설계 원칙

클래스를 구분하는 데 유용하고 입력에서 관측 가능한 속성을 선택한다. 줄무늬나 색은 직접적인 시각 단서가 될 수 있지만 서식지나 식성은 이미지 한 장에서 바로 확인하기 어려울 수 있다. 클래스 수준의 배경 지식과 이미지에서 보이는 특징을 구분한다.

속성의 상관관계, 누락과 라벨 일관성을 확인한다. 같은 클래스 속성이라도 자세나 가림에 따라 관측 가능성이 달라질 수 있다. 정량적 품질 평가는 학습·검증 분할 안에서 수행한다.

### 속성 통계

아래 코드는 클래스 속성의 분산과 평균 차이를 출력한다. 마지막 coverage는 정규화하지 않은 내적의 최댓값이므로 코사인 유사도나 실제 전이 성능을 직접 나타내지 않는다. 모든 클래스 벡터의 차원을 맞추고 빈 클래스 집합을 제외해야 한다.

속성의 질을 뜯어본다.

```python
def analyze_attributes(class_attributes, seen_classes, unseen_classes):
    """
    속성의 가름 힘과 아우름을 뜯어본다.
    """
    import numpy as np
    
    seen_attrs = np.array([class_attributes[c] for c in seen_classes])
    unseen_attrs = np.array([class_attributes[c] for c in unseen_classes])
    
    # 속성별 분석
    n_attrs = seen_attrs.shape[1]
    
    print("Attribute Analysis:")
    print("-" * 50)
    
    for i in range(n_attrs):
        seen_var = np.var(seen_attrs[:, i])
        unseen_var = np.var(unseen_attrs[:, i])
        mean_diff = abs(seen_attrs[:, i].mean() - unseen_attrs[:, i].mean())
        
        print(f"Attribute {i}: "
              f"Seen var={seen_var:.3f}, "
              f"Unseen var={unseen_var:.3f}, "
              f"Mean diff={mean_diff:.3f}")
    
    # 전체 아우름: 못 본 부류의 속성 가운데 얼마나가 본 부류로 덮이는가
    similarity_matrix = unseen_attrs @ seen_attrs.T
    max_similarities = similarity_matrix.max(axis=1)
    
    print(f"\nUnseen class coverage (max similarity to seen):")
    for i, c in enumerate(unseen_classes):
        print(f"  {c}: {max_similarities[i]:.3f}")
```

---

## 연습문제

<div class="drillbox" markdown>

**연습문제 1.** <span class="diff med" title="중간"></span>
DAP의 확률 곱과 로그 점수의 관계를 설명하라.

</div>

??? success "연습문제 1 풀이"
    속성별 사건 확률을 곱하는 대신 로그를 더하면 곱셈 언더플로를 줄일 수 있다. 단순 코드의 점수에는 속성 사전확률 보정이 없으며 균등 클래스 사전확률을 가정한다. 확률의 로그를 안정적으로 계산하는 방법도 필요하다.

---

<div class="drillbox" markdown>

**연습문제 2.** <span class="diff med" title="중간"></span>
원 IAP와 아래 IndirectAttributePrediction의 차이를 설명하라.

</div>

??? success "연습문제 2 풀이"
    원 IAP는 학습 클래스의 사후확률로 속성별 확률을 추정한 뒤 클래스 속성과 비교한다. 이 코드는 클래스 속성 유사도로 미학습 클래스 점수를 직접 합산한다. 중간 표현과 확률 가정이 다르므로 같은 알고리즘이라고 볼 수 없다.

---

<div class="drillbox" markdown>

**연습문제 3.** <span class="diff med" title="중간"></span>
속성별 헤드가 여러 개이면 각각 독립적으로 학습되는가?

</div>

??? success "연습문제 3 풀이"
    코드는 특징 추출기를 공유하고 여러 속성의 BCE를 함께 최적화한다. 헤드가 분리되어도 공유 특징을 통해 학습이 연결된다. 속성 확률을 곱하는 단계의 조건부 독립 근사와 매개변수 공유 여부는 다른 개념이다.

---

<div class="drillbox" markdown>

**연습문제 4.** <span class="diff easy" title="쉬움"></span>
클래스 속성과 이미지별 속성이 다를 수 있는 예를 들라.

</div>

??? success "연습문제 4 풀이"
    새 클래스에 날개가 있다는 속성을 부여해도 특정 이미지에서는 날개가 가려질 수 있다. 클래스 수준 속성을 모든 이미지의 정답으로 사용하면 관측 가능한 속성 라벨과 차이가 생긴다. 평가 시 이런 라벨의 의미를 명시한다.

## 정리하며

속성 기반 제로샷 학습은 클래스 간에 공유되는 속성으로 지식을 전달한다. 원 DAP·IAP의 확률 경로와 단순화된 코드의 점수를 구분하고, 속성 사전확률·라벨 매핑·평가 모드를 확인한다.

**참고**: [Lampert 등의 속성 기반 분류 연구와 자료](https://pub.ista.ac.at/~chl/ABC/).
