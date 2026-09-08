# 속성 기반 영 예시 학습

속성 기반 방법은 영 예시 학습의 바탕이 되는 접근법이다. 부류마다 뜻 속성의 모임으로 그려지고, 모델은 시각 특징에서 이 속성을 맞히는 법을 배운다. 이 절은 속성 기반의 두 갈래인 곧은 속성 예측(DAP)과 에두른 속성 예측(IAP)을 다룬다.

---

## 1. 속성 표현

### 속성의 정의

속성이란 여러 부류를 그려 낼 수 있는 뜻의 성질이다. 부류 $c$마다 속성 벡터를 갖는다.

$$\mathbf{a}_c = [a_c^{(1)}, a_c^{(2)}, \ldots, a_c^{(M)}]$$

여기서 $M$은 속성의 개수이고 $a_c^{(m)}$은 부류 $c$에서 속성 $m$의 있음과 세기를 나타낸다.

### 속성의 종류

**이진 속성**: $a_c^{(m)} \in \{0, 1\}$

- 성질이 있는지 없는지
- 보기: "has_stripes", "can_fly", "has_4_legs"

**이어진 속성**: $a_c^{(m)} \in [0, 1]$ 또는 $\mathbb{R}$

- 성질의 정도나 세기
- 보기: "size"가 코끼리는 0.9, 생쥐는 0.1

**상대 속성**: 견줌 값

- 보기: "larger_than_cat", "faster_than_tortoise"

### 속성 행렬

속성 정보 전체는 속성 행렬에 담긴다.

$$A \in \mathbb{R}^{C \times M}$$

여기서 $C = |\mathcal{Y}^s \cup \mathcal{Y}^u|$은 부류의 총 개수이다.

**속성 행렬 보기:**

| 부류 | has_fur | has_feathers | has_scales | 4_legs | can_fly | is_large | carnivore |
|-------|---------|--------------|------------|--------|---------|----------|-----------|
| 개 | 1.0 | 0.0 | 0.0 | 1.0 | 0.0 | 0.3 | 0.8 |
| 고양이 | 1.0 | 0.0 | 0.0 | 1.0 | 0.0 | 0.1 | 1.0 |
| 새 | 0.0 | 1.0 | 0.0 | 0.0 | 1.0 | 0.1 | 0.3 |
| 말 | 1.0 | 0.0 | 0.0 | 1.0 | 0.0 | 0.9 | 0.0 |
| 독수리 | 0.0 | 1.0 | 0.0 | 0.0 | 1.0 | 0.6 | 1.0 |

---

## 2. 곧은 속성 예측(DAP)

### 개념

DAP은 속성마다를 서로 독립인 이진 분류 문제로 다룬다. 그림이 주어지면 속성마다 있을 확률을 맞힌 다음, 이 예측을 알려진 부류의 속성 서명과 견준다.

### 수식으로 나타내기

**1단계: 속성 가려내기 배우기**

속성 $m = 1, \ldots, M$마다 가려내개를 배운다.

$$P(a_m = 1 | \mathbf{x}) = \sigma(w_m^\top \phi(\mathbf{x}) + b_m)$$

여기서 $\sigma$은 시그모이드 함수이고 $\phi(\mathbf{x})$은 시각 특징을 뽑아낸다.

**2단계: 속성 맞히기**

시험 그림 $\mathbf{x}$에 대해 모든 속성을 맞힌다.

$$\hat{\mathbf{a}}(\mathbf{x}) = [P(a_1 | \mathbf{x}), P(a_2 | \mathbf{x}), \ldots, P(a_M | \mathbf{x})]$$

**3단계: 부류 확률 셈하기**

부류가 주어졌을 때 속성이 조건부로 독립이라고 놓으면 다음과 같다.

$$P(c | \mathbf{x}) \propto P(c) \prod_{m=1}^{M} P(a_m | \mathbf{x})^{a_c^{(m)}} \cdot (1 - P(a_m | \mathbf{x}))^{1 - a_c^{(m)}}$$

**4단계: 부류 맞히기**

$$\hat{y} = \arg\max_{c \in \mathcal{Y}^u} P(c | \mathbf{x})$$

### 로그 공간 정식화

수치 안정성을 위해 로그 공간에서 다룬다.

$$\log P(c | \mathbf{x}) = \text{const} + \sum_{m=1}^{M} \left[ a_c^{(m)} \log P(a_m | \mathbf{x}) + (1 - a_c^{(m)}) \log(1 - P(a_m | \mathbf{x})) \right]$$

### PyTorch 구현

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

## 3. 에두른 속성 예측(IAP)

### 개념

IAP은 다르게 다가간다. 속성을 곧바로 맞히는 대신 먼저 본 부류의 가려내개를 배우고, 속성의 닮음을 써서 그 예측을 못 본 부류로 옮긴다.

### 수식으로 나타내기

**1단계: 본 부류 가려내개 배우기**

본 부류에 대한 보통의 가려내개를 익힌다.

$$P(y^s | \mathbf{x}) = \text{softmax}(W^\top \phi(\mathbf{x})) \quad \text{for } y^s \in \mathcal{Y}^s$$

**2단계: 속성 닮음으로 옮기기**

못 본 부류 $c^u$에 대해 다음을 셈한다.

$$P(c^u | \mathbf{x}) \propto \sum_{c^s \in \mathcal{Y}^s} P(c^s | \mathbf{x}) \cdot \text{sim}(\mathbf{a}_{c^s}, \mathbf{a}_{c^u})$$

**닮음 함수:**

코사인 닮음:

$$\text{sim}(\mathbf{a}_1, \mathbf{a}_2) = \frac{\mathbf{a}_1 \cdot \mathbf{a}_2}{\|\mathbf{a}_1\| \|\mathbf{a}_2\|}$$

거리의 음수에 대한 지수:

$$\text{sim}(\mathbf{a}_1, \mathbf{a}_2) = \exp(-\gamma \|\mathbf{a}_1 - \mathbf{a}_2\|^2)$$

**3단계: 부류 맞히기**

$$\hat{y} = \arg\max_{c \in \mathcal{Y}^u} P(c | \mathbf{x})$$

### 직관

IAP은 본 부류 예측과 못 본 부류 속성 사이의 상관을 끌어 쓴다.

- 그림이 "개"(본 부류)처럼 보이고
- 속성 공간에서 "말"(못 본 부류)이 "개"와 닮았다면
- 그 그림에는 "말"이 있을 만하다

### PyTorch 구현

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

## 4. 견줌: DAP과 IAP

### 이론적 차이

| 갈래 | DAP | IAP |
|--------|-----|-----|
| **학습 목표** | 속성 맞히기 | 본 부류 가려내기 |
| **독립 가정** | 속성이 서로 독립이다 | 독립 가정 없음 |
| **옮김 장치** | 속성 맞춤 | 속성 닮음 |
| **본 부류의 앎** | 속성을 거쳐서만 | 온전한 가려내기 |

### 좋은 점과 나쁜 점

**DAP의 이점:**

- 중간 예측을 풀이할 수 있다
- 속성이 참으로 독립일 때 잘 굴러간다
- 속성 분석으로 오차를 진단할 수 있다

**DAP의 흠:**

- 독립 가정이 깨질 때가 많다
- 속성 가려내개에서 오차가 번져 나간다
- 속성마다 따로 익혀야 한다

**IAP의 이점:**

- 독립 가정이 없다
- 본 부류 분포의 상관을 끌어 쓴다
- 가려내개를 하나만 익히면 된다

**IAP의 흠:**

- 본 부류와 못 본 부류의 속성 겹침에 크게 기댄다
- 닮음이 모자라면 잘 옮겨 가지 못할 수 있다
- 풀이하기가 더 어렵다

### 실험 성능

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

**가름**: 속성은 부류를 갈라내야 한다

- 나쁨: "has_atoms"(모든 물체에 참이다)
- 좋음: "has_stripes"(얼룩말과 말을 갈라낸다)

**알아볼 수 있음**: 속성은 눈으로 알아볼 수 있어야 한다

- 나쁨: "born_in_Africa"(눈에 보이지 않는다)
- 좋음: "black_and_white"(눈에 보이는 본새)

**뜻있음**: 속성은 뜻있는 성질을 잡아내야 한다

- 나쁨: "feature_42_positive"(제멋대로이다)
- 좋음: "is_carnivore"(뜻있는 생물학적 성질)

### 속성 통계

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

**연습문제 1.**
영 예시 학습을 정의하고 소수 예시 학습과 어떻게 다른지 설명하라.

??? success "연습문제 1 풀이"
    영 예시 학습은 학습 중에 한 번도 보지 못한 부류의 사례를, 본 부류와 못 본 부류를 잇는 딸린 정보(속성, 글 설명, 낱말 묻힘)를 써서 가려낸다. 소수 예시 학습은 이름표 붙은 보기를 몇 개 쓴다. 영 예시 학습은 대상 부류의 보기가 아예 없어도 되며 오로지 뜻 표현을 거친 앎의 옮김에 기댄다.

---

**연습문제 2.**
영 예시 학습의 중심 쏠림 문제와 그 다루는 법을 설명하라.

??? success "연습문제 2 풀이"
    차원이 높은 공간에서는 어떤 점('중심점')이 참 부류와 상관없이 다른 많은 점의 최근접 이웃이 된다. 그래서 최근접 이웃 기반 영 예시 분류에서 어떤 부류가 너무 자주 예측된다. 해법: 묻힘을 고르거나, 눈금 맞춘 점수 매기기를 쓰거나, 전용 거리 재기를 배운다.

---

**연습문제 3.**
영 예시 학습의 속성 기반 접근법과 묻힘 기반 접근법을 견주어라.

??? success "연습문제 3 풀이"
    속성 기반: 부류마다 이진이나 이어진 속성 벡터로 그려진다(이를테면 '줄무늬가 있다', '털이 있다'). 모델은 속성을 맞히는 법을 배운 다음 부류 원형과 맞춘다. 묻힘 기반: 시각 특징과 부류 설명(이를테면 부류 이름의 word2vec)을 함께 쓰는 공간으로 옮긴다. 묻힘 방법이 규모를 키우기 쉽고 손 표시가 덜 든다.

---

**연습문제 4.**
일반화된 영 예시 학습(GZSL)의 치우침 문제란 무엇인가?

??? success "연습문제 4 풀이"
    GZSL에서는 시험 때 본 부류와 못 본 부류가 함께 나온다. 모델은 본 부류로 익혔으므로 그쪽으로 치우친다. 해법: 눈금 맞춘 쌓기(본 부류 점수에서 치우침을 빼기), 본 부류와 못 본 부류를 가르는 분포 밖 알아채기, 또는 못 본 부류의 특징을 지어내는 생성 접근법.

## 정리하며

속성 기반 방법은 영 예시 학습의 바탕이 되는 접근법을 준다.

1. **DAP**은 서로 독립인 속성 예측기를 배우고 그 예측을 부류 서명과 맞춘다
2. **IAP**은 본 부류 가려내개를 배우고 속성 닮음으로 옮긴다
3. 두 방법 모두 꼼꼼히 설계된, 가름 힘 있는 속성이 있어야 한다
4. DAP과 IAP 사이의 고름은 속성 독립 가정과 본 부류·못 본 부류 속성의 겹침에 달렸다

속성 기반 방법은 풀이할 수 있고 쓸모 있지만 값비싼 손 표시가 있어야 한다. 다음 절은 미리 학습된 낱말 묻힘을 써서 이 요구를 없애는 뜻 묻힘 접근법을 다룬다.
