# 일반화된 영 예시 학습
## 개요

일반화된 영 예시 학습(GZSL)은 여느 ZSL을, 시험 사례가 본 부류에서도 못 본 부류에서도 올 수 있는 더 현실에 가까운 상황으로 넓힌다. 이 절은 GZSL의 문제 정식화, 평가 지표, 그리고 본 부류 쪽 치우침을 다루는 기법을 살핀다.

## 문제 정식화

### 여느 ZSL과 GZSL

**여느 ZSL:**

- 학습: 본 부류 $\mathcal{Y}^s$으로 배운다
- 시험: 못 본 부류 $\mathcal{Y}^u$에서만 맞힌다
- 가정: 시험 사례가 못 본 부류에서 왔음을 안다

**일반화된 ZSL:**

- 학습: 본 부류 $\mathcal{Y}^s$으로 배운다
- 시험: 모든 부류 $\mathcal{Y}^s \cup \mathcal{Y}^u$에서 맞힌다
- 시험 사례가 어느 영역에 드는지에 대한 가정이 없다

### 형식적 정의

다음이 주어졌다고 하자.

- 학습 집합: $\mathcal{D}^{tr} = \{(\mathbf{x}_i, y_i) : y_i \in \mathcal{Y}^s\}$
- 시험 집합: $\mathcal{D}^{te} = \{(\mathbf{x}_j, y_j) : y_j \in \mathcal{Y}^s \cup \mathcal{Y}^u\}$
- 모든 부류에 대한 뜻 묻힘

목표: 가려내개 $f: \mathcal{X} \rightarrow \mathcal{Y}^s \cup \mathcal{Y}^u$을 배운다

### 치우침 문제

본 부류로 익힌 모델은 **본 부류 쪽으로 크게 치우친다**.

$$P(\hat{y} \in \mathcal{Y}^s | y \in \mathcal{Y}^u) >> P(\hat{y} \in \mathcal{Y}^u | y \in \mathcal{Y}^u)$$

**까닭:**

1. 본 부류의 특징은 익힌 모델에게 분포 안에 있다
2. 어울림 점수가 본 부류에서 자연스레 더 높다
3. 시각 특징이 본 부류에서 더 가름 힘이 있을 수 있다

**예:**

- 본 부류 정확도: 85%
- 못 본 부류 정확도: 5%
- 모델이 거의 늘 본 부류를 내놓는다!

## 평가 지표

### 영역마다의 정확도

**본 부류에서의 정확도:**

$$\text{acc}^s = \frac{1}{|\mathcal{D}^{te}_s|} \sum_{(\mathbf{x}, y) \in \mathcal{D}^{te}_s} \mathbb{1}[f(\mathbf{x}) = y]$$

**못 본 부류에서의 정확도:**

$$\text{acc}^u = \frac{1}{|\mathcal{D}^{te}_u|} \sum_{(\mathbf{x}, y) \in \mathcal{D}^{te}_u} \mathbb{1}[f(\mathbf{x}) = y]$$

### 조화 평균(GZSL의 으뜸 지표)

$$H = \frac{2 \times \text{acc}^s \times \text{acc}^u}{\text{acc}^s + \text{acc}^u}$$

**성질:**

- 본 부류와 못 본 부류의 성능이 치우치면 벌을 준다
- $\text{acc}^s = 0$이거나 $\text{acc}^u = 0$이면 $H = 0$이다
- 두 정확도가 모두 완벽할 때만 $H = 1$이다
- 대칭적이다. 본 부류와 못 본 부류를 똑같이 다룬다

**왜 조화 평균인가?**

- 산술 평균이라면 모두 본 부류로 내놓는 모델도 좋은 점수를 받을 수 있다
- 조화 평균은 두 영역 사이의 균형을 강요한다
- 두 영역이 모두 중요한 현실의 쓸모를 비춘다

### 본 부류-못 본 부류 곡선 아래 넓이(AUSUC)

눈금 맞춤 매개변수를 바꾸어 가며 $\text{acc}^s$과 $\text{acc}^u$을 그린다.

$$\text{AUSUC} = \int_0^1 \text{acc}^s(\text{acc}^u) \, d(\text{acc}^u)$$

이는 한 작동점만이 아니라 맞바꿈 공간 전체를 담아낸다.

### 평가 코드

```python
import numpy as np
from collections import defaultdict

def evaluate_gzsl(predictions, labels, seen_classes, unseen_classes):
    """
    GZSL을 두루 평가하기.
    
    인수:
        predictions: 맞힌 부류 이름
        labels: 참 부류 이름
        seen_classes: 본 부류 이름 목록
        unseen_classes: 못 본 부류 이름 목록
    
    반환값:
        모든 지표를 담은 사전
    """
    predictions = np.array(predictions)
    labels = np.array(labels)
    
    # 본 부류와 못 본 부류 시험 표본의 가리개
    seen_mask = np.isin(labels, seen_classes)
    unseen_mask = np.isin(labels, unseen_classes)
    
    # 영역별 정확도
    acc_seen = np.mean(predictions[seen_mask] == labels[seen_mask]) if seen_mask.sum() > 0 else 0
    acc_unseen = np.mean(predictions[unseen_mask] == labels[unseen_mask]) if unseen_mask.sum() > 0 else 0
    
    # 조화 평균
    if acc_seen + acc_unseen > 0:
        harmonic = 2 * acc_seen * acc_unseen / (acc_seen + acc_unseen)
    else:
        harmonic = 0
    
    # 전체 정확도
    overall = np.mean(predictions == labels)
    
    # 클래스별 정확도
    per_class_acc = {}
    for cls in np.unique(labels):
        cls_mask = labels == cls
        per_class_acc[cls] = np.mean(predictions[cls_mask] == labels[cls_mask])
    
    return {
        'acc_seen': acc_seen,
        'acc_unseen': acc_unseen,
        'harmonic_mean': harmonic,
        'overall': overall,
        'per_class': per_class_acc
    }
```

## 눈금 맞춤 기법

### 문턱값 눈금 맞춤

못 본 부류의 점수에 상수를 더한다.

$$\text{score}^{calib}_c = \begin{cases}
\text{score}_c & (c \in \mathcal{Y}^s) \\
\text{score}_c + \gamma & (c \in \mathcal{Y}^u)
\end{cases}$$

여기서 $\gamma$은 검증 데이터로 최적화한 눈금 맞춤 매개변수이다.

```python
def calibrated_prediction(scores_seen, scores_unseen, calibration):
    """
    눈금을 맞추어 맞힌다.
    
    인수:
        scores_seen: (batch, n_seen) 본 부류의 점수
        scores_unseen: (batch, n_unseen) 못 본 부류의 점수
        calibration: 못 본 부류 점수에 더하는 값
    
    반환값:
        predictions, domain(본 부류/못 본 부류)
    """
    # 눈금 맞춤을 씌운다
    scores_unseen_calib = scores_unseen + calibration
    
    # 점수를 합친다
    max_seen = scores_seen.max(dim=1)
    max_unseen = scores_unseen_calib.max(dim=1)
    
    # 영역을 정한다
    predict_unseen = max_unseen.values > max_seen.values
    
    predictions = torch.where(
        predict_unseen,
        max_unseen.indices,
        max_seen.indices
    )
    
    return predictions, predict_unseen
```

### 온도 눈금 조절

본 부류의 점수를 온도 $T > 1$으로 나눈다.

$$\text{score}^{scaled}_c = \begin{cases}
\text{score}_c / T & (c \in \mathcal{Y}^s) \\
\text{score}_c & (c \in \mathcal{Y}^u)
\end{cases}$$

이는 본 부류의 확률을 더 고르게 만들어 자신감을 낮춘다.

### 가장 좋은 눈금 맞춤 찾기

```python
def find_optimal_calibration(model, val_loader, class_embeddings,
                             seen_classes, unseen_classes,
                             calibration_range=np.arange(-2, 2, 0.1)):
    """
    검증 집합에서 조화 평균을 가장 크게 하는 눈금 맞춤을 찾는다.
    """
    model.eval()
    
    # 모든 예측과 점수를 얻는다
    all_scores_seen = []
    all_scores_unseen = []
    all_labels = []
    
    with torch.no_grad():
        for batch_x, batch_y in val_loader:
            # 본 부류의 점수를 셈한다
            seen_embs = torch.stack([
                torch.tensor(class_embeddings[c], dtype=torch.float32)
                for c in seen_classes
            ])
            scores_seen = model(batch_x, seen_embs)
            
            # 못 본 부류의 점수를 셈한다
            unseen_embs = torch.stack([
                torch.tensor(class_embeddings[c], dtype=torch.float32)
                for c in unseen_classes
            ])
            scores_unseen = model(batch_x, unseen_embs)
            
            all_scores_seen.append(scores_seen)
            all_scores_unseen.append(scores_unseen)
            all_labels.extend(batch_y)
    
    scores_seen = torch.cat(all_scores_seen)
    scores_unseen = torch.cat(all_scores_unseen)
    labels = np.array(all_labels)
    
    # 가장 좋은 눈금 맞춤을 찾는다
    best_harmonic = 0
    best_calibration = 0
    
    results = []
    
    for calib in calibration_range:
        # 눈금 맞춤을 씌운다
        scores_unseen_calib = scores_unseen + calib
        
        # 예측한다
        all_scores = torch.cat([scores_seen, scores_unseen_calib], dim=1)
        all_classes = seen_classes + unseen_classes
        
        pred_indices = all_scores.argmax(dim=1).numpy()
        predictions = [all_classes[i] for i in pred_indices]
        
        # 평가한다
        metrics = evaluate_gzsl(predictions, labels, seen_classes, unseen_classes)
        
        results.append({
            'calibration': calib,
            **metrics
        })
        
        if metrics['harmonic_mean'] > best_harmonic:
            best_harmonic = metrics['harmonic_mean']
            best_calibration = calib
    
    return best_calibration, results
```

## 문 장치

### 개념

사례가 본 부류 영역에 드는지 못 본 부류 영역에 드는지를 맞히는 문 망을 익힌다.

$$g(\mathbf{x}) = P(\text{영역} = \text{못 본 부류} | \mathbf{x})$$

그런 다음 예측의 길을 나눈다.

- $g(\mathbf{x}) > 0.5$이면 못 본 부류에서 맞힌다
- 그렇지 않으면 본 부류에서 맞힌다

### 구현

```python
class GatedGZSL(nn.Module):
    """
    배운 문 장치를 갖춘 GZSL 모델.
    """
    
    def __init__(self, visual_dim: int, semantic_dim: int,
                 embedding_dim: int = 256):
        super().__init__()
        
        # 시각-뜻 어울림 모델
        self.visual_encoder = nn.Sequential(
            nn.Linear(visual_dim, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(1024, embedding_dim)
        )
        
        self.semantic_encoder = nn.Linear(semantic_dim, embedding_dim)
        
        # 문 망: 본 부류인지 못 본 부류인지 맞힌다
        self.gate = nn.Sequential(
            nn.Linear(visual_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )
    
    def encode_visual(self, x):
        return F.normalize(self.visual_encoder(x), dim=1)
    
    def encode_semantic(self, s):
        return F.normalize(self.semantic_encoder(s), dim=1)
    
    def compatibility(self, visual, semantic):
        v_enc = self.encode_visual(visual)
        s_enc = self.encode_semantic(semantic)
        
        if len(s_enc.shape) == 2 and s_enc.shape[0] != visual.shape[0]:
            return v_enc @ s_enc.T
        return torch.sum(v_enc * s_enc, dim=1)
    
    def predict_gate(self, visual):
        """못 본 부류 영역일 확률을 맞힌다."""
        return self.gate(visual)
    
    def gated_predict(self, visual, seen_embeddings, unseen_embeddings,
                      seen_classes, unseen_classes):
        """
        문 장치로 맞힌다.
        """
        # 문의 예측을 얻는다
        gate_prob = self.predict_gate(visual).squeeze()
        
        # 어울림 점수를 얻는다
        scores_seen = self.compatibility(visual, seen_embeddings)
        scores_unseen = self.compatibility(visual, unseen_embeddings)
        
        # 문을 거친 예측
        predictions = []
        for i in range(len(visual)):
            if gate_prob[i] > 0.5:
                # 못 본 부류에서 맞힌다
                pred_idx = scores_unseen[i].argmax().item()
                predictions.append(unseen_classes[pred_idx])
            else:
                # 본 부류에서 맞힌다
                pred_idx = scores_seen[i].argmax().item()
                predictions.append(seen_classes[pred_idx])
        
        return predictions

def train_gated_gzsl(model, train_loader, class_embeddings, 
                     seen_classes, unseen_classes, epochs=50):
    """
    문을 갖춘 GZSL 모델을 익힌다.
    
    참고: 문을 익히려면 못 본 부류 표본이 조금 필요하며,
    대개 검증 쪼갬이나 지어낸 특징에서 얻는다.
    """
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    # 묻힘을 마련한다
    seen_embs = torch.stack([
        torch.tensor(class_embeddings[c], dtype=torch.float32)
        for c in seen_classes
    ])
    
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        
        for batch_x, batch_y in train_loader:
            # 어울림 손실(순위)
            pos_emb = torch.stack([
                torch.tensor(class_embeddings[y], dtype=torch.float32)
                for y in batch_y
            ])
            
            # 아무렇게나 고른 음의 보기
            neg_idx = torch.randint(0, len(seen_classes), (len(batch_y),))
            neg_emb = seen_embs[neg_idx]
            
            pos_score = model.compatibility(batch_x, pos_emb)
            neg_score = model.compatibility(batch_x, neg_emb)
            
            ranking_loss = torch.clamp(0.2 - pos_score + neg_score, min=0).mean()
            
            # 문 손실(학습 표본은 모두 본 부류 영역에서 온다)
            gate_pred = model.predict_gate(batch_x)
            gate_loss = F.binary_cross_entropy(gate_pred, torch.zeros_like(gate_pred))
            
            # 결합된 손실
            loss = ranking_loss + 0.5 * gate_loss
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1}/{epochs}, Loss: {total_loss/len(train_loader):.4f}")
```

## 생성 접근법

### 개념

못 본 부류의 시각 특징을 지어낸 다음 보통의 가려내개를 익힌다.

$$G: \mathcal{S} \times \mathcal{Z} \rightarrow \mathcal{V}$$

여기서 $\mathcal{Z}$은 잡음 분포이다.

### VAE로 특징 지어내기

```python
class FeatureVAE(nn.Module):
    """
    뜻 묻힘에서 시각 특징을 지어내는 VAE.
    """
    
    def __init__(self, visual_dim: int, semantic_dim: int, latent_dim: int = 64):
        super().__init__()
        
        # 부호기: 시각 + 뜻 -> 숨은 값
        self.encoder = nn.Sequential(
            nn.Linear(visual_dim + semantic_dim, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU()
        )
        self.fc_mu = nn.Linear(256, latent_dim)
        self.fc_var = nn.Linear(256, latent_dim)
        
        # 복호기: 숨은 값 + 뜻 -> 시각
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim + semantic_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Linear(512, visual_dim)
        )
    
    def encode(self, visual, semantic):
        x = torch.cat([visual, semantic], dim=1)
        h = self.encoder(x)
        return self.fc_mu(h), self.fc_var(h)
    
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def decode(self, z, semantic):
        x = torch.cat([z, semantic], dim=1)
        return self.decoder(x)
    
    def forward(self, visual, semantic):
        mu, logvar = self.encode(visual, semantic)
        z = self.reparameterize(mu, logvar)
        recon = self.decode(z, semantic)
        return recon, mu, logvar
    
    def generate(self, semantic, n_samples=1):
        """주어진 뜻 묻힘의 시각 특징을 지어낸다."""
        z = torch.randn(n_samples, self.fc_mu.out_features)
        semantic_expanded = semantic.unsqueeze(0).expand(n_samples, -1)
        return self.decode(z, semantic_expanded)

def train_feature_vae(vae, train_loader, class_embeddings, epochs=100):
    """특징 VAE를 익힌다."""
    optimizer = torch.optim.Adam(vae.parameters(), lr=0.001)
    
    for epoch in range(epochs):
        vae.train()
        total_loss = 0
        
        for batch_v, batch_y in train_loader:
            batch_s = torch.stack([
                torch.tensor(class_embeddings[y], dtype=torch.float32)
                for y in batch_y
            ])
            
            recon, mu, logvar = vae(batch_v, batch_s)
            
            # 되살림 손실
            recon_loss = F.mse_loss(recon, batch_v)
            
            # KL 벌어짐
            kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
            kl_loss = kl_loss / batch_v.shape[0]
            
            loss = recon_loss + 0.001 * kl_loss
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        if (epoch + 1) % 20 == 0:
            print(f"Epoch {epoch+1}/{epochs}, Loss: {total_loss/len(train_loader):.4f}")

def generate_unseen_features(vae, class_embeddings, unseen_classes, 
                              n_per_class=100):
    """못 본 부류의 특징을 지어낸다."""
    vae.eval()
    
    X_gen = []
    y_gen = []
    
    with torch.no_grad():
        for cls in unseen_classes:
            semantic = torch.tensor(class_embeddings[cls], dtype=torch.float32)
            features = vae.generate(semantic, n_samples=n_per_class)
            
            X_gen.append(features)
            y_gen.extend([cls] * n_per_class)
    
    return torch.cat(X_gen), y_gen
```

## 전달적 GZSL

### 개념

학습 중에 이름표 없는 시험 데이터를 끌어 써서 못 본 부류의 분포에 맞추어 간다.

### 스스로 익히기 접근법

```python
def transductive_gzsl(model, train_loader, test_loader, 
                      class_embeddings, seen_classes, unseen_classes,
                      iterations=5, confidence_threshold=0.9):
    """
    스스로 익히기를 쓰는 전달적 GZSL.
    """
    # 본 부류로 처음 익히기
    train_model(model, train_loader, class_embeddings, seen_classes)
    
    for iteration in range(iterations):
        print(f"\nTransductive iteration {iteration + 1}/{iterations}")
        
        # 시험 데이터에서 맞힌다
        model.eval()
        pseudo_X = []
        pseudo_y = []
        
        with torch.no_grad():
            for batch_x, _ in test_loader:
                # 자신감과 함께 예측을 얻는다
                all_classes = seen_classes + unseen_classes
                all_embs = torch.stack([
                    torch.tensor(class_embeddings[c], dtype=torch.float32)
                    for c in all_classes
                ])
                
                scores = model(batch_x, all_embs)
                probs = F.softmax(scores, dim=1)
                
                max_probs, pred_indices = probs.max(dim=1)
                
                # 못 본 부류에 대한 자신감 높은 예측만 남긴다
                for i in range(len(batch_x)):
                    if max_probs[i] > confidence_threshold:
                        pred_class = all_classes[pred_indices[i]]
                        if pred_class in unseen_classes:
                            pseudo_X.append(batch_x[i])
                            pseudo_y.append(pred_class)
        
        if len(pseudo_X) > 0:
            print(f"  Added {len(pseudo_X)} pseudo-labeled unseen samples")
            
            # 학습 데이터와 합친다
            # ... (합친 데이터셋을 만들고 다시 익힌다)
        else:
            print("  No confident predictions on unseen classes")
            break
```

## 요약

일반화된 영 예시 학습은 여느 ZSL을 넘어서는 큰 어려움을 안긴다.

1. **본 부류 쪽 치우침**이 GZSL의 으뜸가는 걸림돌이다
2. **조화 평균**이 표준 평가 지표이며 치우친 성능에 벌을 준다
3. **눈금 맞춤 기법**(문턱값, 온도)으로 본 부류와 못 본 부류의 균형을 손볼 수 있다
4. **문 장치**는 예측을 알맞은 영역으로 보내는 법을 배운다
5. **생성 접근법**은 못 본 부류의 특징을 지어내어 GZSL을 지도 학습으로 바꾼다
6. **전달적 방법**은 영역 적응을 위해 이름표 없는 시험 데이터를 끌어 쓴다

좋은 버릇은 다음과 같다.

- 전체 정확도만이 아니라 늘 조화 평균으로 평가하라
- 따로 떼어 둔 검증 집합에서 눈금 맞춤 매개변수를 교차 검증하라
- 치우침이 심하면 생성 접근법을 생각해 보라
- 본 부류와 못 본 부류의 맞바꿈 곡선 전체(AUSUC)를 알려라

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
