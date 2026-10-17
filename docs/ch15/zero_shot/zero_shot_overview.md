# 제로샷 학습의 기초

제로샷 학습은 대상 클래스의 라벨 있는 학습 표본 대신 속성이나 텍스트 설명 등 보조 정보를 이용해 분류한다. 여기서는 학습 클래스와 미학습 클래스가 분리된 고전적인 시각 제로샷 학습 설정을 설명한다.

---

## 1. 문제 정식화

학습 클래스에는 라벨 있는 시각 표본이 있고, 미학습 클래스에는 같은 방식의 라벨 있는 학습 표본이 없다. 모든 클래스의 의미 설명을 통해 두 집합을 연결한다. 사전 학습 모델이 이미 어떤 데이터를 보았는지는 별도로 확인해야 한다.

### 형식적 정의

$\mathcal{X}$을 입력 공간(이를테면 이미지), $\mathcal{Y}$을 레이블 공간이라 하자. ZSL에서 레이블 공간은 다음으로 나뉜다.

- **학습 클래스**: $\mathcal{Y}^s = \{y_1^s, y_2^s, \ldots, y_K^s\}$ — 학습 보기가 있는 클래스
- **미학습 클래스**: $\mathcal{Y}^u = \{y_1^u, y_2^u, \ldots, y_L^u\}$ — 학습 보기가 없는 클래스

근본 제약은 다음과 같다.

$$\mathcal{Y}^s \cap \mathcal{Y}^u = \emptyset$$

### 학습과 시험의 규약

**학습 단계:**

- 레이블 붙은 데이터를 쓸 수 있다: $\mathcal{D}^{tr} = \{(\mathbf{x}_i, y_i) : y_i \in \mathcal{Y}^s\}_{i=1}^{N}$
- 모든 클래스의 의미 설명을 쓸 수 있다: $\{\mathbf{s}_c\}_{c \in \mathcal{Y}^s \cup \mathcal{Y}^u}$

**시험 단계(기존 ZSL):**

- 시험 사례는 미학습 클래스에만 든다
- 예측 공간: $\mathcal{Y}^u$

**시험 단계(일반화된 ZSL):**

- 시험 사례는 어느 클래스에나 들 수 있다
- 예측 공간: $\mathcal{Y}^s \cup \mathcal{Y}^u$

---

## 2. 의미 표현

ZSL의 키는 학습 클래스와 미학습 클래스에 두루 쓰이는 표현을 주는 의미 공간 $\mathcal{S}$이다.

### 속성 기반 표현

클래스 $c$마다 속성 벡터 $\mathbf{a}_c \in \mathbb{R}^M$으로 그려지며 $M$은 속성의 개수이다.

$$\mathbf{a}_c = [a_c^{(1)}, a_c^{(2)}, \ldots, a_c^{(M)}]$$

속성은 다음일 수 있다.

- **이진**: 있고 없음을 나타내는 $a_c^{(m)} \in \{0, 1\}$
- **연속값**: 세기와 관련도를 나타내는 $a_c^{(m)} \in [0, 1]$

**보기: 동물의 속성**

| 동물 | has_fur | has_stripes | is_large | has_4_legs | can_fly |
|--------|---------|-------------|----------|------------|---------|
| 개 | 1 | 0 | 0.3 | 1 | 0 |
| 고양이 | 1 | 0 | 0.1 | 1 | 0 |
| 얼룩말 | 1 | 1 | 0.9 | 1 | 0 |
| 독수리 | 0 | 0 | 0.6 | 0 | 1 |

### 단어 임베딩 표현

클래스 이름이나 설명을 텍스트 인코더로 벡터화할 수 있다. 언어적 유사도가 시각적 유사도와 항상 일치하는 것은 아니므로 이미지 피처와 연결하는 학습 또는 정렬된 사전 학습 모델이 필요하다.

$$\mathbf{s}_c = \text{Embed}(\text{classname}_c) \in \mathbb{R}^D$$

흔한 임베딩 방법은 다음과 같다.

- Word2Vec(스킵그램, CBOW)
- GloVe(전역 벡터)
- FastText(서브워드 정보)
- BERT, GPT 임베딩

**장점과 한계:**

클래스별 속성을 사람이 일일이 표시하는 비용을 줄이고 텍스트 코퍼스의 관계를 활용한다. 다만 클래스 이름의 중의성, 어휘 누락, 언어·도메인 차이를 처리해야 한다. 단어 임베딩을 얻는 것만으로 시각 분류가 완성되지는 않는다.

1. 손 표시가 필요 없다
2. 풍부한 의미 관계를 알아서 담는다
3. 엄청난 텍스트 코퍼스로 미리 학습되어 있다
4. 말에 담긴 앎을 시각 영역으로 옮긴다

### 의미 유사도

의미 공간은 유사도 관계를 자아낸다.

$$\text{sim}(c_1, c_2) = f(\mathbf{s}_{c_1}, \mathbf{s}_{c_2})$$

흔한 유사도 함수는 다음과 같다.

**코사인 유사도:**

$$\text{sim}_{\cos}(\mathbf{s}_1, \mathbf{s}_2) = \frac{\mathbf{s}_1 \cdot \mathbf{s}_2}{\|\mathbf{s}_1\| \|\mathbf{s}_2\|}$$

**유클리드 거리(유사도으로 바꾼 것):**

$$\text{sim}_{\text{euc}}(\mathbf{s}_1, \mathbf{s}_2) = \exp(-\|\mathbf{s}_1 - \mathbf{s}_2\|^2)$$

**점곱:**

$$\text{sim}_{\text{dot}}(\mathbf{s}_1, \mathbf{s}_2) = \mathbf{s}_1^\top \mathbf{s}_2$$

---

## 3. 호환성 함수

호환성 함수 $F: \mathcal{V} \times \mathcal{S} \rightarrow \mathbb{R}$은 시각 피처가 의미 표현과 얼마나 잘 맞는지를 잰다.

### 선형 투영을 통한 비교

$$F(\mathbf{v}, \mathbf{s}) = \mathbf{v}^\top W \mathbf{s}$$

여기서 $W \in \mathbb{R}^{d_v \times d_s}$은 배운 투영 행렬이다.

### 쌍선형 호환성

위 선형 투영의 내적도 두 입력에 대해 쌍선형인 같은 식이다. 각 입력을 고정하면 다른 입력에 선형이며, 두 표현을 단순히 연결한 선형 분류기와는 다르다.

$$F(\mathbf{v}, \mathbf{s}) = \mathbf{v}^\top W \mathbf{s}$$

W는 시각 차원과 의미 차원의 상호작용을 학습한다. 앞의 식과 동일하므로 별도의 더 복잡한 모형으로 세지 않는다.

### 뉴럴 호환성

비선형 호환성에 뉴럴 넷트웍을 쓴다.

$$F(\mathbf{v}, \mathbf{s}) = g_\theta(\mathbf{v}, \mathbf{s})$$

흔한 구조는 다음과 같다.

- 이어 붙인 뒤 MLP
- 교차 주의 장치
- 잔차 호환성 망

### 임베딩 기반 호환성

시각·의미 표현을 공통 공간으로 투영한다.

$$F(\mathbf{v}, \mathbf{s}) = f_\phi(\mathbf{v})^\top g_\psi(\mathbf{s})$$

또는 코사인 유사도를 쓴다.

$$F(\mathbf{v}, \mathbf{s}) = \frac{f_\phi(\mathbf{v}) \cdot g_\psi(\mathbf{s})}{\|f_\phi(\mathbf{v})\| \|g_\psi(\mathbf{s})\|}$$

---

## 4. ZSL의 로스 함수

### 교차 엔트로피 로스

ZSL을 학습 클래스 위의 분류로 다룬다.

$$\mathcal{L}_{CE} = -\sum_{(\mathbf{x}, y) \in \mathcal{D}^{tr}} \log \frac{\exp(F(\phi(\mathbf{x}), \mathbf{s}_y))}{\sum_{c \in \mathcal{Y}^s} \exp(F(\phi(\mathbf{x}), \mathbf{s}_c))}$$

### 순위 로스(힌지 로스)

맞는 클래스가 틀린 클래스보다 더 잘 어울리도록 북돋운다.

$$\mathcal{L}_{rank} = \sum_{(\mathbf{x}, y)} \sum_{c \neq y} \max(0, \Delta + F(\phi(\mathbf{x}), \mathbf{s}_c) - F(\phi(\mathbf{x}), \mathbf{s}_y))$$

여기서 $\Delta$은 마진 파라메터이다.

### 트리플렛 로스

임베딩의 거리를 곧바로 최적화한다.

$$\mathcal{L}_{triplet} = \sum_{(\mathbf{x}, y)} \max(0, \|f(\mathbf{x}) - \mathbf{s}_y\|^2 - \|f(\mathbf{x}) - \mathbf{s}_{y^-}\|^2 + \alpha)$$

여기서 $y^-$은 음의 클래스이고 $\alpha$은 마진이다.

### 대조 로스

쌍으로 유사도를 학습하기 위한 것이다.

$$\mathcal{L}_{cont} = \sum_{i,j} y_{ij} D_{ij}^2 + (1 - y_{ij}) \max(0, m - D_{ij})^2$$

여기서 쌍이 같은 클래스에서 오면 $y_{ij} = 1$이고 그렇지 않으면 0이다.

---

## 5. 시각 피처 추출

시각 피처는 사전 학습 CNN이나 시각·언어 인코더에서 추출할 수 있다. 제로샷 평가 클래스가 사전 학습의 라벨 또는 이미지에 포함되었는지에 따라 평가의 의미가 달라진다.

### CNN 백본

| 모델 예시 | 피처 추출 위치 | 차원 조건 |
|---|---|---|
| VGG 계열 | 마지막 분류기 전 은닉 레이어 | 구성에 따라 4096 등 |
| ResNet-101 | 전역 평균 풀링 뒤 | 일반적인 구성에서 2048 |
| Inception-v3 | 최종 풀링 뒤 | 일반적인 구성에서 2048 |
| CLIP 계열 | 이미지·텍스트 투영 출력 | 체크포인트에 따라 다름 |

### 피처 추출 파이프라인

아래 예제는 한 이미지를 2048차원 ResNet 피처로 바꾼다. 모델·입력의 장치와 해당 웨이트에 맞는 전처리를 일치시킨다. 이 벡터는 단어 임베딩과 차원·좌표계가 다르므로 그대로 비교할 수 없다.

```python
import torch
import torchvision.models as models
import torchvision.transforms as transforms

# 미리 학습된 ResNet을 불러온다
resnet = models.resnet101(pretrained=True)
# 가려내기 머리를 없앤다
feature_extractor = torch.nn.Sequential(*list(resnet.children())[:-1])
feature_extractor.eval()

# 표준 앞손질
preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225])
])

# 피처를 뽑는다
def extract_features(image):
    with torch.no_grad():
        x = preprocess(image).unsqueeze(0)
        features = feature_extractor(x)
        return features.squeeze()  # 꼴: (2048,)
```

---

## 6. 시험 때의 예측

### 의미 공간에서의 최근접 이웃

시각 피처가 $\mathbf{v} = \phi(\mathbf{x})$인 시험 사례 $\mathbf{x}$에 대해 다음과 같다.

1. 투영한 피처가나 호환성 점수를 계산한다
2. 가장 잘 어울리는 클래스를 찾는다

**기존 ZSL:**

$$\hat{y} = \arg\max_{c \in \mathcal{Y}^u} F(\mathbf{v}, \mathbf{s}_c)$$

**일반화된 ZSL:**

$$\hat{y} = \arg\max_{c \in \mathcal{Y}^s \cup \mathcal{Y}^u} F(\mathbf{v}, \mathbf{s}_c)$$

### 구현

predict_zsl은 이미 같은 차원과 좌표계에 정렬된 NumPy 배열을 가정한다. 코사인 비교의 영벡터, 빈 클래스 목록과 미지원 compatibility_fn은 호출 전에 처리해야 한다. 클래스 이름 순서는 임베딩 행 순서와 같아야 한다.

```python
import numpy as np

def predict_zsl(visual_features, class_embeddings, class_names, 
                compatibility_fn='cosine'):
    """
    영 예시 학습으로 부류를 맞힌다.
    
    인수:
        visual_features: (n_samples, d_v) 시각 피처 행렬
        class_embeddings: (n_classes, d_s) 뜻 묻힘 행렬
        class_names: 부류 이름 목록
        compatibility_fn: 'cosine', 'dot' 또는 'euclidean'
    
    반환값:
        predictions: 맞힌 부류 이름 목록
    """
    if compatibility_fn == 'cosine':
        # 코사인 닮음을 위해 고른다
        v_norm = visual_features / np.linalg.norm(visual_features, axis=1, keepdims=True)
        s_norm = class_embeddings / np.linalg.norm(class_embeddings, axis=1, keepdims=True)
        scores = v_norm @ s_norm.T
    elif compatibility_fn == 'dot':
        scores = visual_features @ class_embeddings.T
    elif compatibility_fn == 'euclidean':
        # 거리의 음수를 점수로 쓴다
        scores = -np.sum((visual_features[:, None, :] - class_embeddings[None, :, :]) ** 2, axis=2)
    
    pred_indices = np.argmax(scores, axis=1)
    predictions = [class_names[idx] for idx in pred_indices]
    
    return predictions, scores
```

---

## 7. 핵심 어려움

### 영역 이동

학습 클래스의 시각 피처는 미학습 클래스와 분포가 다를 수 있다.

- **클래스 안 분산**: 같은 클래스의 사례끼리도 달라 보인다
- **클래스 사이 유사도**: 미학습 클래스끼리가 학습 클래스와보다 더 닮았을 수 있다
- **투영 영역 이동**: 배운 옮김이 잘 일반화되지 않을 수 있다

### 허브니스 문제

허브니스(hubness)는 일부 대표 벡터가 많은 입력의 최근접 이웃으로 반복 선택되는 현상이다. 반드시 기하학적 중심에 있는 점을 뜻하지 않으며, 고차원 공간과 투영 과정에서 나타날 수 있다.

$$\text{중심쏠림}(c) = \sum_{\mathbf{x} \in \text{test}} \mathbb{1}[\text{NN}(\mathbf{x}) = c]$$

그래서 어떤 클래스가 너무 자주 예측된다.

**누그러뜨리는 전략:**

- 국소 스케일 조절(CSLS)을 쓴다
- 중심 쏠림 줄이기 기법을 쓴다
- 임베딩 학습에 정규화를 쓴다

### 의미와 시각의 차이

의미 공간이 시각의 갈림을 온전히 담지 못할 수 있다.

- 눈으로 닮은 클래스가 서로 다른 의미 표현을 가질 수 있다
- 눈으로 다른 클래스가 닮은 의미 표현을 가질 수 있다
- 촘촘한 갈림은 속성이나 단어 임베딩으로 담기지 않을 수 있다

### GZSL의 바이어스

학습 클래스만으로 로스를 최적화하면 학습 클래스의 점수가 상대적으로 높아질 수 있다. GZSL은 학습·미학습 클래스 전체에서 예측하므로 이런 바이어스를 평가하고 검증 데이터에서 점수 보정을 선택한다. 모든 모델이 같은 정도로 바이어스된다는 법칙은 아니다.

학습 클래스로만 학습된 모델은 학습 클래스를 내놓는 쪽으로 크게 치우친다.

- 학습 클래스의 피처는 분포 안에 있다
- 호환성 점수가 학습 클래스에서 자연스레 더 높다
- 미학습 클래스는 거의 예측되지 않는다

이는 뒤 절에서 다루는 보정과 균형 잡기 기법으로 다룬다.

---

## 연습문제

<div class="drillbox" markdown>

**연습문제 1.** <span class="diff med" title="중간"></span>
고전적인 ZSL에서 미학습 클래스에 대해 제공되는 정보와 제공되지 않는 정보를 설명하라.

</div>

??? success "풀이"
    클래스 속성이나 이름·설명 등의 의미 정보는 제공할 수 있다. 귀납적 설정에서는 해당 클래스의 라벨 있는 시각 학습 표본을 사용하지 않는다. 전이적 설정은 라벨 없는 타깃 표본을 활용할 수 있으므로 프로토콜을 별도로 명시한다.

---

<div class="drillbox" markdown>

**연습문제 2.** <span class="diff med" title="중간"></span>
허브니스가 분류에 미치는 영향과 확인 방법을 설명하라.

</div>

??? success "풀이"
    각 클래스가 최근접 이웃으로 선택되는 횟수를 집계한다. 실제 클래스 빈도나 난이도와 무관하게 일부 벡터에 예측이 몰리는지 확인한다. 정규화나 국소 스케일 보정 등을 비교할 수 있지만 빈도 바이어스만으로 원인을 단정하지 않는다.

---

<div class="drillbox" markdown>

**연습문제 3.** <span class="diff med" title="중간"></span>
시각 피처와 의미 임베딩의 차원이 같으면 직접 비교해도 되는가?

</div>

??? success "풀이"
    같은 차원은 내적 계산의 필요조건일 뿐 좌표계가 정렬되었다는 뜻은 아니다. 학습 클래스의 이미지·설명으로 투영을 학습하거나 이미지·텍스트가 공동 학습된 인코더를 사용한다. 정규화만으로 의미 정렬을 대신할 수 없다.

---

<div class="drillbox" markdown>

**연습문제 4.** <span class="diff easy" title="쉬움"></span>
기존 ZSL과 GZSL의 예측 후보 집합을 비교하라.

</div>

??? success "풀이"
    기존 ZSL은 미학습 클래스만 후보로 둔다. GZSL은 학습 클래스와 미학습 클래스의 합집합에서 예측한다. 둘은 난이도와 평가 조건이 다르므로 같은 정확도로 직접 비교하지 않는다.

## 정리하며

제로샷 분류는 클래스 의미 정보와 입력 피처를 연결해 라벨 있는 학습 표본이 없는 클래스를 예측한다. 후보 클래스 집합, 사전 학습 데이터와 타깃 표본 사용 여부를 명시하고, 피처 차원뿐 아니라 두 공간의 정렬을 확인한다.
