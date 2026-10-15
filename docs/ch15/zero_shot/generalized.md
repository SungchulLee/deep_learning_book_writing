# 일반화된 제로샷 학습

일반화된 제로샷 학습(GZSL)은 학습 클래스와 미학습 클래스를 모두 후보로 두고 예측한다. 학습 클래스에 대한 높은 점수 때문에 미학습 클래스를 놓칠 수 있으므로, 두 그룹의 정확도와 균형을 함께 평가한다. 점수 보정, 게이팅, 특징 생성과 전이적 학습의 역할을 구분한다.

## 1. 문제 정식화

### 예측 후보의 차이

기존 ZSL은 미학습 클래스 $\mathcal Y^u$만 후보로 두고, GZSL은 $\mathcal Y^s\cup\mathcal Y^u$에서 예측한다. 두 집합은 겹치지 않으며 모든 후보 클래스의 의미 표현을 제공한다. GZSL도 후보 밖의 임의 클래스를 자동으로 처리하는 오픈셋 인식과는 구별한다.

### 학습 클래스 편향

학습 클래스의 손실로만 학습하면 그 클래스의 점수가 상대적으로 높아질 수 있다. 미학습 클래스 정답이 학습 클래스로 잘못 예측되는 비율과 각 그룹의 정확도를 확인한다. 학습 클래스 정확도가 높고 미학습 클래스 정확도가 낮다는 사실만으로 모든 오차가 이 편향 때문이라고 단정할 수는 없다.

## 2. 평가 지표

### 그룹별 정확도

모든 후보 클래스에서 예측한 결과를 정답이 학습 클래스인 그룹과 미학습 클래스인 그룹으로 나눠 평가한다. GZSL 벤치마크에서는 그룹 안의 클래스별 정확도를 평균 내는 매크로 정확도를 흔히 사용한다.

$$A_s=\frac1{|\mathcal Y^s|}\sum_{c\in\mathcal Y^s}\frac{\sum_{i:y_i=c}\mathbf1[\hat y_i=c]}{n_c}$$

A_u도 미학습 클래스에서 같은 방식으로 계산한다. 아래 코드는 그룹 안의 모든 표본을 합친 마이크로 정확도를 반환하므로 클래스별 표본 수가 다르면 매크로 값과 다르다. 표본이 없는 클래스·그룹은 미관측으로 처리하는 규칙이 필요하다.

### 조화 평균

$$H=\frac{2A_sA_u}{A_s+A_u}$$

어느 한쪽이 낮으면 H도 낮아지므로 두 그룹의 균형을 드러낸다. 둘 다 0일 때는 H=0으로 정의한다. H만 제시하지 말고 A_s와 A_u도 함께 보고한다. 실제 적용의 클래스 빈도나 오차 비용은 이 균등 가중치와 다를 수 있다.

### 학습·미학습 정확도 곡선의 면적

점수 보정 계수를 변화시키면 A_s와 A_u의 트레이드오프 곡선을 얻는다. AUSUC는 이 곡선이 실제로 도달한 범위의 면적이다. 임의로 0~1 전체 구간에 함수를 정의한 것처럼 해석하지 않는다. 아래 코드에는 AUSUC 적분 구현이 없다.

### 평가 코드

predictions와 labels는 같은 길이의 원본 클래스 ID이며 그룹 목록은 겹치지 않아야 한다. 코드가 없는 그룹의 정확도를 0으로 반환하는 것은 편의 규칙으로, 관측한 실패율 100%를 뜻하지 않는다. overall은 전체 표본 정확도이고 per_class는 클래스별 정확도다.

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

---

## 3. 보정 기법

### 상수 점수 보정

미학습 클래스 전체에 γ를 더하는 것은 학습 클래스 전체에서 γ를 빼는 것과 같은 argmax를 만든다. 아래 calibrated_prediction은 두 그룹의 로컬 인덱스를 반환하므로 predict_unseen과 함께 원본 클래스 목록으로 복원해야 한다. 이 인덱스를 그대로 합치면 서로 다른 클래스가 같은 번호로 충돌한다.

미학습 클래스의 점수에 상수를 더한다.

$$\text{score}^{calib}_c = \begin{cases}
\text{score}_c & (c \in \mathcal{Y}^s) \\
\text{score}_c + \gamma & (c \in \mathcal{Y}^u)
\end{cases}$$

여기서 $\gamma$은 검증 데이터로 최적화한 보정 매개변수이다.

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

### 온도 스케일 조절

학습 클래스의 점수를 온도 $T > 1$으로 나눈다.

$$\text{score}^{scaled}_c = \begin{cases}
\text{score}_c / T & (c \in \mathcal{Y}^s) \\
\text{score}_c & (c \in \mathcal{Y}^u)
\end{cases}$$

학습 클래스 내부의 소프트맥스는 T>1에서 더 완만해진다. 하지만 두 그룹을 함께 비교할 때 학습 클래스의 점수를 나누는 것이 항상 그쪽 예측을 줄이지는 않는다. 음의 점수는 0에 가까워지며 오히려 커진다. 상수 보정과 같은 작동이라고 볼 수 없다.

### 검증 데이터에서 보정 선택

최종 테스트 클래스와 분리한 검증용 가상 미학습 클래스로 γ를 선택한다. 모델은 (배치 크기, 후보 클래스 수)의 전체 점수 행렬을 반환해야 한다. 입력 행 수로 대응 쌍 여부를 추정하는 모델에는 이 조건을 명시해야 한다. CPU에서 생성하는 임베딩과 numpy 변환도 GPU 평가에 맞춰야 한다.

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

---

## 4. 게이팅

### 개념

게이트는 입력이 학습 클래스 그룹에 속하는지 미학습 클래스 그룹에 속하는지 추정하고 해당 그룹 안에서 분류한다. 게이트의 학습에는 두 그룹을 구별할 감독 신호나 적절한 대체 과제가 필요하다.

$$g(\mathbf{x}) = P(\text{영역} = \text{미학습 클래스} | \mathbf{x})$$

그런 다음 예측의 길을 나눈다.

- $g(\mathbf{x}) > 0.5$이면 미학습 클래스에서 맞힌다
- 그렇지 않으면 학습 클래스에서 맞힌다

### 구현

아래 train_gated_gzsl은 게이트 정답을 모두 0으로 주므로 학습 클래스만 선택하도록 학습한다. unseen_classes 인수도 학습에 사용되지 않아 미학습 그룹을 알아내는 게이트가 완성되지 않는다. 음성 클래스가 양성과 같을 수도 있다. gated_predict의 squeeze는 배치 1에서 스칼라가 되며, compatibility는 배치·클래스 수가 같을 때 전체 점수 행렬을 만들지 않는 한계도 있다. F 임포트는 별도로 필요하다.

```python
import torch
import torch.nn as nn


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

---

## 5. 생성 접근법

### 개념

의미 벡터와 잠재 잡음을 조건으로 미학습 클래스의 시각 특징을 생성하고, 실제 학습 클래스 특징과 합쳐 분류기를 학습할 수 있다. 생성기는 학습 클래스 자료에서 배운 관계를 새 의미 벡터로 일반화해야 하므로 생성 표본이 실제 분포를 보장하지는 않는다.

$$G: \mathcal{S} \times \mathcal{Z} \rightarrow \mathcal{V}$$

여기서 $\mathcal{Z}$은 잡음 분포이다.

### VAE로 특징 지어내기

FeatureVAE는 시각 특징과 의미 벡터에서 잠재 분포를 추정하고 조건부 복원을 학습한다. 코드의 MSE는 전체 원소 평균, KL은 배치당 잠재 차원 합이므로 특징·잠재 차원에 따라 상대 스케일이 달라진다. 생성용 잡음과 의미 벡터를 CPU에 만들기 때문에 모델과 장치를 맞춰야 한다. 생성 후 최종 GZSL 분류기를 학습하는 단계는 포함되지 않는다.

```python
class FeatureVAE(nn.Module):
    """
    뜻 묻힘에서 시각 특징을 지어내는 VAE.
    """
    
    def __init__(self, visual_dim: int, semantic_dim: int, latent_dim: int = 64):
        super().__init__()
        
        # 인코더: 시각 + 뜻 -> 숨은 값
        self.encoder = nn.Sequential(
            nn.Linear(visual_dim + semantic_dim, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU()
        )
        self.fc_mu = nn.Linear(256, latent_dim)
        self.fc_var = nn.Linear(256, latent_dim)
        
        # 디코더: 숨은 값 + 뜻 -> 시각
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

---

## 6. 전이적 GZSL

### 개념

전이적(transductive) 설정은 평가할 타깃 집합의 입력을 라벨 없이 학습에 활용한다. 개별 쿼리를 처음 보는 귀납적 평가와 조건이 다르므로 따로 보고한다. 타깃 정답으로 임계값이나 반복 수를 고르면 평가 정보가 누수된다.

### 의사 라벨 자기 학습

아래 함수의 train_model은 정의되어 있지 않다. 또한 고신뢰 의사 라벨을 모아 개수만 출력하고 모델을 재학습하지 않는다. 따라서 반복해도 의사 라벨 적응을 구현한 것이 아니다. 실제 자기 학습에는 의사 라벨을 이용한 갱신과 오류 증폭을 점검할 독립적인 검증 절차가 필요하다.

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

---

## 연습문제

<div class="drillbox" markdown>

**연습문제 1.** <span class="diff med" title="중간"></span>
A_s=0.8, A_u=0.2일 때 산술 평균과 조화 평균을 계산하라.

</div>

??? success "연습문제 1 풀이"
    산술 평균은 0.5이고 조화 평균은 2×0.8×0.2/(0.8+0.2)=0.32다. 조화 평균은 낮은 그룹의 성능에 더 민감하다. 각 그룹의 정확도와 매크로·마이크로 기준도 함께 제시한다.

---

<div class="drillbox" markdown>

**연습문제 2.** <span class="diff med" title="중간"></span>
그룹별 마이크로 정확도와 매크로 정확도가 다른 예를 들라.

</div>

??? success "연습문제 2 풀이"
    한 클래스는 100개 중 100개 정답이고 다른 클래스는 1개 중 0개 정답이면 마이크로 정확도는 100/101, 클래스 매크로 정확도는 0.5다. 클래스별 표본 수가 불균형하면 큰 차이가 생긴다.

---

<div class="drillbox" markdown>

**연습문제 3.** <span class="diff med" title="중간"></span>
미학습 클래스에 γ를 더하는 보정이 학습 클래스에서 γ를 빼는 것과 같은 이유를 설명하라.

</div>

??? success "연습문제 3 풀이"
    전체 후보 점수에 공통 상수 γ를 더하거나 빼도 argmax는 변하지 않는다. 두 방식은 모든 후보의 점수가 같은 상수만큼 다른 관계이므로 같은 순위를 낸다. 그룹별 온도 조절은 일반적으로 이 관계를 만족하지 않는다.

---

<div class="drillbox" markdown>

**연습문제 4.** <span class="diff easy" title="쉬움"></span>
모든 게이트 학습 정답을 0으로 주면 미학습 클래스를 구분할 수 있는가?

</div>

??? success "연습문제 4 풀이"
    그 손실은 모든 입력에 학습 클래스 그룹을 선택하도록 유도한다. 새 그룹을 구별하는 감독 신호가 없으므로 게이팅 설명의 목표와 맞지 않는다. 검증용 가상 미학습 클래스나 적절한 생성·이상 탐지 목표 등을 설계해 따로 평가해야 한다.

## 정리하며

GZSL은 학습·미학습 클래스를 함께 예측하므로 그룹별 성능과 점수 편향을 평가한다. 매크로·마이크로 정확도를 구분하고 보정 계수는 독립적인 검증 데이터에서 선택한다. 게이팅·생성·전이적 방법은 필요한 학습 신호와 데이터 사용 조건이 서로 다르다.
